//! Break-axis test coverage.
//!
//! Promoted out of an inline `#[cfg(test)] mod tests { ... }` block in
//! `break_axis.rs` (this crate's Rust test-module convention: promote when
//! the test body exceeds the source module rather than on a line-count
//! threshold — mirroring `render/annotation/tests.rs`'s promotion earlier in
//! this same wave). At the time of promotion `break_axis.rs` was 5240 lines
//! with the inline test module starting at line 2619 — the 2622-line test
//! body outsized the 2618-line source above it. Precedent:
//! `render/scale_resolve/tests.rs`, `projection/tests.rs`,
//! `transform/stats/tests.rs`, `render/annotation/tests.rs`. Pure code
//! motion: every test below is unchanged from its inline form, still
//! exercising the real functions in `super::*`.

use ferrum_scene::MarkBatchKind;

use super::*;
use crate::layout::{Rect, ThemeInputs};

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
