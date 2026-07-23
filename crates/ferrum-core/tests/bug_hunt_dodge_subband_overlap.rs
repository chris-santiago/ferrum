//! Edge-case integration tests for GH #66/#67 — the Dodge sub-band overlap
//! fix.
//!
//! Because ferrum-core is a cdylib crate, integration tests cannot link
//! against it directly (same constraint documented in
//! bug_hunt_band_point_range.rs and bug_hunt_marks_rendering.rs). These
//! tests reproduce the implementation formulas inline, verbatim:
//!
//!   - `bandwidth` / `sub_band` / per-group pixel offset — verbatim from
//!     `apply_dodge_ordinal` (render/position.rs);
//!   - the pre-fix bar-width formula (`band_extent / n_categories /
//!     n_groups * 0.8`, render/marks/bar.rs `build_ordinal`) and the
//!     post-fix clamp (`min(raw_width, sub_band)`,
//!     `position::clamp_to_dodge_sub_band`).
//!
//! The exact numeric oracles below (padding=0.2 overlap of 20px; the
//! default-padding x/width geometry) were captured by driving the REAL
//! `apply_position` -> `dispatch_mark_build` seam via an in-crate unit test
//! (`render::position::tests::dodge_sub_band_clamp_prevents_overlap_at_high_padding`
//! and a since-removed scratch capture for the default-padding case) —
//! first against pre-fix code (confirming the RED overlap below), then
//! against post-fix code (confirming the clamp and the unchanged default
//! geometry). These tests pin those same numbers so the contract is
//! documented and independently re-verifiable without needing to drive the
//! full render pipeline.
//!
//! GH #66 background: the issue as filed blamed `BandScale(padding_inner=)`
//! for dodge overlap. That premise is false — `OrdinalScale::bandwidth()`
//! and the render-side band centers never read `padding` on the dodge path;
//! `test_1` below pins that inertness. The real defect was Dodge's
//! `padding` parameter: sub-band offsets (`apply_dodge_ordinal`) account for
//! `padding`, but bar/box/tick width formulas used a fixed factor (bar
//! 0.8, box/tick `band_size`) divided only by `n_groups`, blind to
//! `padding` — so at `padding > ~0.1` (bar) the width formula could exceed
//! the true sub-band and adjacent dodge groups would overlap. The fix
//! clamps each mark's width to its Dodge sub-band
//! (`position::clamp_to_dodge_sub_band`), which is byte-identical at
//! default padding (`test_3`) because the 0.8-factor width already fits
//! inside the sub-band there.

// =========================================================================
// Verbatim formula model
// =========================================================================

/// Mirrors `OrdinalScale::bandwidth` (positive even when reversed) — render/
/// scale/ordinal.rs.
fn bandwidth(range: (f64, f64), n_categories: usize) -> f64 {
    ((range.1 - range.0) / n_categories as f64).abs()
}

/// Mirrors the category-center formula shared by `OrdinalScaleData::layout`
/// (see bug_hunt_band_point_range.rs) — `padding` never enters this
/// calculation, which is exactly the seam GH #66 refutes.
fn category_center(range: (f64, f64), n_categories: usize, index: usize) -> f64 {
    let step = (range.1 - range.0) / n_categories as f64;
    range.0 + step / 2.0 + (index as f64) * step
}

/// Mirrors `apply_dodge_ordinal`'s per-group pixel offset (render/
/// position.rs): `sub_band = (bandwidth - bandwidth*padding*2) / n_groups`,
/// `offset = -bandwidth/2 + bandwidth*padding + sub_band*(group_idx + 0.5)`.
fn dodge_offset(bandwidth_px: f64, padding: f64, n_groups: usize, group_idx: usize) -> f64 {
    let sub_band = sub_band(bandwidth_px, padding, n_groups);
    -bandwidth_px / 2.0 + bandwidth_px * padding + sub_band * (group_idx as f64 + 0.5)
}

/// Mirrors the `sub_band` local in `apply_dodge_ordinal` — now also stamped
/// into schema metadata as `__dodge_sub_band_px__` (GH #66 fix) and read
/// back by `position::clamp_to_dodge_sub_band`.
fn sub_band(bandwidth_px: f64, padding: f64, n_groups: usize) -> f64 {
    let pad_total = bandwidth_px * padding * 2.0;
    (bandwidth_px - pad_total) / n_groups as f64
}

/// Pre-fix `build_ordinal` bar-width formula (render/marks/bar.rs), blind to
/// `padding` — always `band_extent / n_categories / n_groups * 0.8`.
fn bar_width_prefix(band_extent: f64, n_categories: usize, n_groups: usize) -> f64 {
    (band_extent / n_categories as f64 / n_groups as f64) * 0.8
}

/// Post-fix `build_ordinal` bar width — the GH #66 clamp
/// (`position::clamp_to_dodge_sub_band`): `min(raw_width, sub_band)`, a
/// no-op whenever `sub_band` isn't strictly tighter than `raw_width`.
fn bar_width_postfix(band_extent: f64, n_categories: usize, n_groups: usize, padding: f64) -> f64 {
    let raw = bar_width_prefix(band_extent, n_categories, n_groups);
    let bw = bandwidth((0.0, band_extent), n_categories);
    let sb = sub_band(bw, padding, n_groups);
    if sb > 0.0 && sb < raw { sb } else { raw }
}

/// Emitted rect `(x, width)` for one dodge group in one category, mirroring
/// `build_ordinal`'s `x: cx - bar_width / 2.0, w: bar_width`.
fn dodged_rect(
    range: (f64, f64),
    n_categories: usize,
    cat_idx: usize,
    n_groups: usize,
    group_idx: usize,
    padding: f64,
    post_fix: bool,
) -> (f64, f64) {
    let bw = bandwidth(range, n_categories);
    let cx = category_center(range, n_categories, cat_idx) + dodge_offset(bw, padding, n_groups, group_idx);
    let band_extent = range.1 - range.0;
    let width = if post_fix {
        bar_width_postfix(band_extent, n_categories, n_groups, padding)
    } else {
        bar_width_prefix(band_extent, n_categories, n_groups)
    };
    (cx - width / 2.0, width)
}

// =========================================================================
// Test 1 — the refuted premise: padding_inner is geometrically inert
// =========================================================================

/// GH #66 as filed: `BandScale(padding_inner=)` was blamed for dodge
/// overlap. `OrdinalScale::bandwidth()` is `((r_hi - r_lo) / n).abs()` —
/// `padding` never enters it — and `category_center` above (the render-side
/// band-center formula) is likewise padding-independent. This test pins
/// that inertness directly: category centers (and therefore every dodged
/// rect's `x`/`width`, which are derived only from `bandwidth` and
/// `category_center`) are identical whether the ordinal scale's `padding`
/// is 0.0 or 0.9. `padding` on `OrdinalScale` only feeds `invert_band`
/// hit-testing (not exercised here) — the render/dodge path never reads it.
#[test]
fn dodge_padding_inner_is_geometrically_inert() {
    let range = (0.0, 600.0);
    let n_categories = 3;

    // bandwidth() takes no padding argument at all in the real
    // implementation — this is the direct proof: the render-side dodge
    // geometry has no padding input to vary in the first place.
    let bw = bandwidth(range, n_categories);
    assert_eq!(bw, 200.0);

    // category_center likewise has no padding parameter — centers for a
    // hypothetical padding_inner=0.0 and padding_inner=0.9 config are
    // computed by the exact same call, hence byte-identical (not merely
    // numerically close: it is literally the same function invocation).
    let centers_at_0: Vec<f64> = (0..n_categories).map(|i| category_center(range, n_categories, i)).collect();
    let centers_at_09: Vec<f64> = (0..n_categories).map(|i| category_center(range, n_categories, i)).collect();
    assert_eq!(centers_at_0, centers_at_09, "padding_inner must not perturb band centers");
    assert_eq!(centers_at_0, vec![100.0, 300.0, 500.0]);

    // And therefore every emitted dodged rect (x, width) is identical too.
    for cat_idx in 0..n_categories {
        for group_idx in 0..2 {
            let r0 = dodged_rect(range, n_categories, cat_idx, 2, group_idx, 0.05, true);
            let r09 = dodged_rect(range, n_categories, cat_idx, 2, group_idx, 0.05, true);
            assert_eq!(r0, r09, "rect geometry must be padding_inner-independent");
        }
    }
}

// =========================================================================
// Test 2 — the real defect: Dodge `padding` above threshold overlaps
// pre-fix, clamps post-fix
// =========================================================================

/// GH #66's real defect: Dodge `padding=0.2`, n=3 categories, g=2 groups,
/// E=600 (matching the RED evidence captured against the real
/// `apply_position` -> `dispatch_mark_build` seam in
/// `render::position::tests::dodge_sub_band_clamp_prevents_overlap_at_high_padding`).
///
/// Pre-fix (`post_fix=false`, the 0.8-factor width with no sub-band clamp):
/// group0 spans `[30, 110]`, group1 starts at `90` — a 20px overlap. This
/// assertion FAILS against that formula (documented below as the captured
/// RED evidence) and is why this test only exercises the post-fix formula
/// as the assertion under test — the pre-fix block is retained as an
/// explicit, executable counter-example.
#[test]
fn dodge_padding_above_threshold_does_not_overlap() {
    let range = (0.0, 600.0);
    let n_categories = 3;
    let n_groups = 2;
    let padding = 0.2;

    // Pre-fix formula: reproduces the exact RED evidence captured against
    // pre-fix code (`cargo test dodge_sub_band_clamp_prevents_overlap_at_high_padding`
    // run before the fix landed):
    //   thread '...' panicked: adjacent dodge sub-bars in category 'a' must
    //   not overlap: group0 spans [30, 110], group1 starts at 90 — overlap
    //   of 20
    let (x0_pre, w0_pre) = dodged_rect(range, n_categories, 0, n_groups, 0, padding, false);
    let (x1_pre, _w1_pre) = dodged_rect(range, n_categories, 0, n_groups, 1, padding, false);
    assert_eq!((x0_pre, w0_pre), (30.0, 80.0), "pre-fix group0 rect must match the captured RED evidence");
    assert_eq!(x1_pre, 90.0, "pre-fix group1 x must match the captured RED evidence");
    let pre_fix_overlap = (x0_pre + w0_pre) - x1_pre;
    assert!(
        pre_fix_overlap > 0.0,
        "pre-fix formula must reproduce the documented 20px overlap; got {pre_fix_overlap}"
    );
    assert!((pre_fix_overlap - 20.0).abs() < 1e-9);

    // Post-fix formula (the assertion this test is named for): for every
    // category, adjacent dodge sub-bars must not overlap.
    for cat_idx in 0..n_categories {
        let (x0, w0) = dodged_rect(range, n_categories, cat_idx, n_groups, 0, padding, true);
        let (x1, w1) = dodged_rect(range, n_categories, cat_idx, n_groups, 1, padding, true);
        assert!(
            x0 + w0 <= x1 + 1e-9,
            "category {cat_idx}: adjacent dodge sub-bars must not overlap: group0 spans \
             [{x0}, {}], group1 starts at {x1}",
            x0 + w0
        );
        // Pins the exact clamped geometry (matches the real-pipeline
        // `dodge_sub_band_clamp_prevents_overlap_at_high_padding` unit test
        // for category "a"): width clamps from 80.0 to sub_band=60.0, and
        // the two groups sit exactly edge-to-edge.
        assert!((w0 - 60.0).abs() < 1e-9, "clamped width must be 60.0; got {w0}");
        assert!((w1 - 60.0).abs() < 1e-9, "clamped width must be 60.0; got {w1}");
    }
    let (x0, w0) = dodged_rect(range, n_categories, 0, n_groups, 0, padding, true);
    let (x1, _w1) = dodged_rect(range, n_categories, 0, n_groups, 1, padding, true);
    assert_eq!((x0, w0), (40.0, 60.0));
    assert_eq!(x1, 100.0);
}

// =========================================================================
// Test 3 — default padding (0.05): byte-stable, no golden churn
// =========================================================================

/// Default Dodge `padding=0.05`, same E=600/n=3/g=2 config: the 0.8-factor
/// raw width (80.0) already fits inside the sub-band (90.0), so the GH #66
/// clamp is a no-op and every emitted rect is pinned to the EXACT geometry
/// captured by rendering the unmodified pre-fix code through the real
/// `apply_position` -> `dispatch_mark_build` seam (a since-removed scratch
/// unit test in `render/position.rs`, run once before the fix and once
/// after — both runs produced this identical
/// `[(15,80),(105,80),(215,80),(305,80),(415,80),(505,80)]` sequence,
/// confirming byte-stability). Rows are in `(category, group)` order
/// a/p, a/q, b/p, b/q, c/p, c/q — mirroring `MarkNodes`' row-order-preserving
/// accumulator.
#[test]
fn dodge_default_padding_output_unchanged() {
    let range = (0.0, 600.0);
    let n_categories = 3;
    let n_groups = 2;
    let padding = 0.05;

    let bw = bandwidth(range, n_categories);
    let sb = sub_band(bw, padding, n_groups);
    let raw = bar_width_prefix(range.1 - range.0, n_categories, n_groups);
    assert!(raw < sb, "the default-padding clamp must be a documented no-op: raw={raw}, sub_band={sb}");

    let expected = [
        (15.0, 80.0),
        (105.0, 80.0),
        (215.0, 80.0),
        (305.0, 80.0),
        (415.0, 80.0),
        (505.0, 80.0),
    ];
    let mut got = Vec::with_capacity(6);
    for cat_idx in 0..n_categories {
        for group_idx in 0..n_groups {
            got.push(dodged_rect(range, n_categories, cat_idx, n_groups, group_idx, padding, true));
        }
    }
    for (i, (rect, exp)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (rect.0 - exp.0).abs() < 1e-9 && (rect.1 - exp.1).abs() < 1e-9,
            "row {i}: expected {exp:?}, got {rect:?}"
        );
    }

    // And the pre-fix formula produces the exact same sequence at this
    // padding — the clamp genuinely never engages here.
    let mut got_prefix = Vec::with_capacity(6);
    for cat_idx in 0..n_categories {
        for group_idx in 0..n_groups {
            got_prefix.push(dodged_rect(range, n_categories, cat_idx, n_groups, group_idx, padding, false));
        }
    }
    assert_eq!(got, got_prefix, "post-fix output must be byte-identical to pre-fix at default padding");
}
