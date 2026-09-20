//! Piecewise scale math for an axis break: the segments a break leaves
//! behind, the pixel positions it reserves for its indicators, and the
//! data-space queries callers ask about a gap.
//!
//! Split out of `break_axis/mod.rs` (#229) when that module reached 2634
//! source lines carrying five jobs. This is the one job that never touches a
//! `SceneNode`: everything here is arithmetic over a domain and a pixel range,
//! which is why it could leave without dragging any of the scene rewrite with
//! it. `BreakRemap` and its single `impl` stay in `mod.rs` — see that module's
//! header for why. Precedent for the split: `render/scale_resolve/`,
//! `projection/`, `transform/stats/`, `transform/smooth/`.

/// Segments of a broken scale. Each segment covers a contiguous range of the
/// original data domain and maps to a contiguous range of pixels.
///
/// Constructed by `apply_break_to_scale` and consumed by `broken_scale_map`,
/// which [`BreakRemap`] calls to re-position scene coordinates onto the broken
/// axis.
///
/// [`BreakRemap`]: super::BreakRemap
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
    ///
    /// [`BreakRemap`]: super::BreakRemap
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
    /// `scene_build::build_panel_breaks` reads this to emit
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
///
/// [`BreakRemap`]: super::BreakRemap
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
///
/// [`BreakRemap::coord`]: super::BreakRemap::coord
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
///
/// [`BreakRemap::data_extent_visible`]: super::BreakRemap::data_extent_visible
pub fn gap_spanned_by(a: f64, b: f64, result: &BreakResult) -> Option<(f64, f64)> {
    let (lo, hi) = (a.min(b), a.max(b));
    result.segments.windows(2).find_map(|w| {
        let (_, hi0) = w[0].ordered_data();
        let (lo1, _) = w[1].ordered_data();
        (lo <= hi0 && hi >= lo1).then_some((hi0, lo1))
    })
}
