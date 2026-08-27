//! Projection test coverage.
//!
//! This module carries two kinds of tests:
//! - the original small in-src round-trip suite (one point per projection), and
//! - the bulk of edge-case coverage relocated from `tests/bug_hunt_projection.rs`
//!   (R1, 2026-08-27), which previously re-derived every formula in a standalone
//!   integration-test crate and asserted against the *copy* rather than the real
//!   `forward`/`inverse` dispatch. Every test below calls the real functions in
//!   `super::*` (including the `#[cfg(test)]`-gated `inverse`/`*_inv` helpers,
//!   which is why this coverage can only live here rather than in `tests/`).
//!
//! Two genuine tautologies from the mirror file were dropped outright rather than
//! ported: `bug_hunt_r2_stereographic_not_implemented` (no assertion — it only
//! documented that `GeoProjection` has no `Stereographic` variant) and
//! `bug_hunt_r2_equal_earth_inv_theta_converges_one_step` (pure algebra that
//! substitutes `theta_init = y / EE_M` back into `(EE_M * theta_init - y) / EE_M`,
//! which is `0` by construction regardless of what any implementation does — it
//! called no function under test). A handful of near-duplicate mirror tests
//! (identical point + tolerance, or a pure-algebra restatement of a property a
//! sibling test already checks against the real functions) were merged into the
//! single surviving test rather than kept as byte-identical copies; those cases
//! are noted inline.

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Shared round-trip helper (pre-existing).
// ═══════════════════════════════════════════════════════════════════════════

fn round_trip(proj: GeoProjection, lon: f64, lat: f64, tol: f64) {
    let (x, y) = forward(proj, lon, lat);
    if !x.is_finite() || !y.is_finite() { return; }
    let (lon2, lat2) = inverse(proj, x, y);
    assert!(
        (lon2 - lon).abs() < tol,
        "{:?} lon: {} → {} (diff {})", proj, lon, lon2, (lon2-lon).abs()
    );
    assert!(
        (lat2 - lat).abs() < tol,
        "{:?} lat: {} → {} (diff {})", proj, lat, lat2, (lat2-lat).abs()
    );
}

#[test] fn mercator_round_trip() { round_trip(GeoProjection::Mercator, -73.98, 40.74, 1e-10); }
#[test] fn equirectangular_round_trip() { round_trip(GeoProjection::Equirectangular, 20.0, 45.0, 1e-10); }
#[test] fn equal_earth_round_trip() { round_trip(GeoProjection::EqualEarth, 30.0, 45.0, 1e-10); }
#[test] fn natural_earth_round_trip() { round_trip(GeoProjection::NaturalEarth, -30.0, 20.0, 1e-10); }
#[test] fn orthographic_round_trip() { round_trip(GeoProjection::Orthographic, 0.0, 45.0, 1e-10); }
#[test] fn albers_usa_round_trip() { round_trip(GeoProjection::AlbersUsa, -87.6, 41.8, 1e-4); }

// ═══════════════════════════════════════════════════════════════════════════
// NaN input propagation — forward
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn mercator_nan_lon_produces_nan() {
    let (x, _y) = mercator_fwd(f64::NAN, 40.0);
    assert!(x.is_nan(), "mercator: NaN lon should yield NaN x, got {x}");
}

#[test]
fn mercator_nan_lat_produces_nan() {
    // f64::NAN.clamp(a, b) returns NaN in Rust.
    let (_x, y) = mercator_fwd(0.0, f64::NAN);
    assert!(y.is_nan(), "mercator: NaN lat should yield NaN y, got {y}");
}

#[test]
fn equirect_nan_lon() {
    let (x, _) = equirect_fwd(f64::NAN, 0.0);
    assert!(x.is_nan(), "equirect: NaN lon should yield NaN x, got {x}");
}

#[test]
fn equal_earth_nan_lat() {
    let (_, y) = equal_earth_fwd(0.0, f64::NAN);
    assert!(y.is_nan(), "equal-earth: NaN lat should yield NaN y, got {y}");
}

#[test]
fn orthographic_nan_input_no_panic() {
    // cos(NaN) = NaN, NaN * anything = NaN, NaN < 0.0 = false, so the hemisphere
    // guard falls through and we reach the trig expression with lam = NaN.
    let (x, _y) = orthographic_fwd(f64::NAN, 0.0);
    assert!(x.is_nan(), "orthographic: NaN lon should yield NaN x, got {x}");
}

#[test]
fn natural_earth_nan_lat_no_panic() {
    let (_x, y) = natural_earth_fwd(0.0, f64::NAN);
    assert!(y.is_nan(), "natural-earth: NaN lat should yield NaN y, got {y}");
}

#[test]
fn albers_usa_nan_lon_no_panic() {
    // NaN < -127.5 and NaN < -154.0 are both false, so routing falls through to
    // continental, where to_rad(NaN) propagates NaN through sin/cos.
    let (x, _y) = albers_usa_fwd(f64::NAN, 41.8);
    assert!(x.is_nan() || x.is_finite(), "albers: NaN lon should not panic");
}

// ═══════════════════════════════════════════════════════════════════════════
// Infinity input handling — forward
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn mercator_inf_lat_clamped() {
    let (_x, y) = mercator_fwd(0.0, f64::INFINITY);
    assert!(y.is_finite(), "mercator: +inf lat should clamp to finite, got {y}");
}

#[test]
fn mercator_neg_inf_lat_clamped() {
    let (_x, y) = mercator_fwd(0.0, f64::NEG_INFINITY);
    assert!(y.is_finite(), "mercator: -inf lat should clamp to finite, got {y}");
}

#[test]
fn mercator_inf_lon() {
    let (x, _) = mercator_fwd(f64::INFINITY, 0.0);
    assert!(x.is_infinite(), "mercator: inf lon should yield inf x, got {x}");
}

#[test]
fn equirect_inf_lat() {
    let (_, y) = equirect_fwd(0.0, f64::INFINITY);
    assert!(y.is_infinite(), "equirect: inf lat should yield inf y, got {y}");
}

#[test]
fn equal_earth_inf_lat_produces_nan() {
    // sin(inf) = NaN → asin(NaN) = NaN → theta = NaN → y = NaN.
    let (_, y) = equal_earth_fwd(0.0, f64::INFINITY);
    assert!(y.is_nan(), "equal-earth: inf lat should yield NaN y, got {y}");
}

#[test]
fn natural_earth_inf_lon() {
    let (x, _) = natural_earth_fwd(f64::INFINITY, 0.0);
    assert!(!x.is_finite() || x.is_nan(), "natural-earth: inf lon should not yield finite x, got {x}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Mercator: pole clamping and boundary
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn mercator_exactly_at_clamp_boundary() {
    let lat = 85.051_129_34_f64;
    let (x, y) = mercator_fwd(0.0, lat);
    assert!(x.is_finite() && y.is_finite(), "mercator at clamp boundary must be finite: ({x}, {y})");
    let (_, y90) = mercator_fwd(0.0, 90.0);
    assert!((y - y90).abs() < 1e-10, "mercator at clamp boundary should equal clamped pole: y={y}, y90={y90}");
}

#[test]
fn mercator_very_large_lat() {
    let (_, y) = mercator_fwd(0.0, 1e300);
    assert!(y.is_finite(), "mercator: 1e300 lat should clamp to finite, got {y}");
}

#[test]
fn mercator_neg_antimeridian_round_trip() {
    round_trip(GeoProjection::Mercator, -180.0, 30.0, 1e-10);
}

#[test]
fn mercator_clamp_loses_information() {
    // lat=86 gets clamped to 85.051129 in forward, so the inverse recovers the
    // clamped value, not 86 — information loss is the documented behavior.
    let lat = 86.0;
    let (x, y) = mercator_fwd(0.0, lat);
    let (_, lat2) = mercator_inv(x, y);
    assert!(
        (lat2 - lat).abs() > 0.5,
        "mercator at lat=86 should lose info due to clamping: expected recovery near 85.05, got {lat2}"
    );
    assert!(
        (lat2 - 85.051_129_34).abs() < 1e-6,
        "mercator at lat=86: inverse should recover the clamped value ~85.051, got {lat2}"
    );
}

#[test]
fn mercator_inside_clamp_round_trip() {
    round_trip(GeoProjection::Mercator, -45.0, -85.0, 1e-10);
}

// ═══════════════════════════════════════════════════════════════════════════
// Equirectangular edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn equirect_full_sweep_round_trip() {
    for lon in [-180.0, -90.0, 0.0, 90.0, 180.0] {
        for lat in [-90.0, -45.0, 0.0, 45.0, 90.0] {
            round_trip(GeoProjection::Equirectangular, lon, lat, 1e-12);
        }
    }
}

#[test]
fn equirect_inv_large_x_no_panic() {
    let (lon, lat) = equirect_inv(1e6, 0.0);
    assert!(lon.is_finite() && lat.is_finite(), "equirect inv with large x should be finite, got ({lon}, {lat})");
}

#[test]
fn equirect_negative_zero_same_as_zero() {
    let (x_pos, y_pos) = equirect_fwd(0.0, 0.0);
    let (x_neg, y_neg) = equirect_fwd(-0.0, -0.0);
    assert!(
        (x_pos - x_neg).abs() < 1e-15 && (y_pos - y_neg).abs() < 1e-15,
        "equirect: -0.0 should equal 0.0"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Natural Earth: ne_poly / ne_poly_deriv must include the phi^8 term
// ═══════════════════════════════════════════════════════════════════════════

/// NE_B has 5 coefficients; the Patterson y-factor polynomial has a phi^8 term
/// (`NE_B[4] = -0.005916`) that is significant at high latitudes (commit
/// `bd182a8` fixed a prior implementation that silently dropped it, producing
/// ~0.34 error at the poles). Property test: comparing the real `ne_poly`
/// output against a 4-term-only baseline (computed inline, not by re-deriving
/// `ne_poly`'s own algorithm) proves the 5th coefficient is actually applied.
/// Merges three mirror tests that pinned this same fact from different
/// angles (`bug_hunt_natural_earth_ne_poly_drops_fifth_coefficient`,
/// `bug_hunt_natural_earth_y_at_pole_matches_reference`,
/// `bug_hunt_r2_regression_ne_poly_includes_phi8`).
#[test]
fn natural_earth_ne_poly_includes_phi8_term() {
    let phi = to_rad(90.0);
    let p2 = phi * phi;
    let p4 = p2 * p2;
    let p6 = p4 * p2;
    let four_term_only = NE_B[0] + p2 * NE_B[1] + p4 * NE_B[2] + p6 * NE_B[3];
    let real = ne_poly(&NE_B, phi);
    let dropped_term_magnitude = (real - four_term_only).abs();
    assert!(
        dropped_term_magnitude > 0.1,
        "ne_poly should apply the phi^8 term for NE_B at the pole; \
         real={real}, 4-term baseline={four_term_only}, diff={dropped_term_magnitude}"
    );
}

/// Same property for the derivative: `ne_poly_deriv` must include
/// `8 * phi^7 * NE_B[4]`, or Newton-Raphson convergence in `natural_earth_inv`
/// degrades. Merges `bug_hunt_natural_earth_deriv_drops_fifth_term` and
/// `bug_hunt_r2_regression_ne_poly_deriv_includes_phi7`.
#[test]
fn natural_earth_ne_poly_deriv_includes_phi7_term() {
    let phi = to_rad(70.0);
    let p3 = phi * phi * phi;
    let p5 = p3 * phi * phi;
    let three_term_only = 2.0 * phi * NE_B[1] + 4.0 * p3 * NE_B[2] + 6.0 * p5 * NE_B[3];
    let real = ne_poly_deriv(&NE_B, phi);
    let dropped_term_magnitude = (real - three_term_only).abs();
    assert!(
        dropped_term_magnitude > 0.01,
        "ne_poly_deriv should apply the 8*phi^7*NE_B[4] term; \
         real={real}, 3-term baseline={three_term_only}, diff={dropped_term_magnitude}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Natural Earth: round-trip across the latitude range
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn natural_earth_equator_round_trip() {
    round_trip(GeoProjection::NaturalEarth, 45.0, 0.0, 1e-10);
}

#[test]
fn natural_earth_inv_at_origin() {
    let (lon, lat) = natural_earth_inv(0.0, 0.0);
    assert!((lon).abs() < 1e-10 && (lat).abs() < 1e-10, "natural-earth inverse at (0,0) should give (0,0), got ({lon}, {lat})");
}

#[test]
fn natural_earth_moderate_lat_round_trip() {
    round_trip(GeoProjection::NaturalEarth, -120.0, 45.0, 1e-6);
}

#[test]
fn natural_earth_full_latitude_sweep_round_trip() {
    for lat_i in -8..=8 {
        round_trip(GeoProjection::NaturalEarth, 45.0, lat_i as f64 * 10.0, 1e-6);
    }
}

#[test]
fn natural_earth_antimeridian_round_trip() {
    round_trip(GeoProjection::NaturalEarth, 180.0, 0.0, 1e-10);
}

#[test]
fn natural_earth_x_factor_at_pole_nonzero() {
    // At lat=90, x = lam * ne_poly(NE_A, phi); if that factor were ~0, longitude
    // would be unrecoverable at the pole for every lon.
    let phi = to_rad(90.0);
    let xf = ne_poly(&NE_A, phi);
    assert!(xf > 0.5 && xf < 2.0, "natural-earth x-factor at pole should be ~1.0, got {xf}");
}

#[test]
fn natural_earth_inv_very_negative_y_no_panic() {
    let (lon, lat) = natural_earth_inv(0.0, -100.0);
    let _ = (lon, lat);
}

#[test]
fn natural_earth_inv_large_y_no_panic() {
    let (lon, lat) = natural_earth_inv(0.0, 100.0);
    let _ = (lon, lat);
}

#[test]
fn natural_earth_inv_y_zero_converges_immediately() {
    let (lon, lat) = natural_earth_inv(0.5, 0.0);
    assert!(lat.abs() < 1e-12, "natural-earth inv at y=0: lat should be 0, got {lat}");
    let expected_lon = to_deg(0.5 / NE_A[0]);
    assert!((lon - expected_lon).abs() < 1e-10, "natural-earth inv at y=0, x=0.5: lon should be {expected_lon}, got {lon}");
}

#[test]
fn natural_earth_inv_near_zero_derivative_no_panic() {
    // The Newton-Raphson step clamps the derivative via `dfy.max(1e-12)`; near a
    // genuine y (not the synthetic phi=PI case the mirror explored, which is
    // outside any real projected range) the inverse must still converge.
    let (_, y_extreme) = natural_earth_fwd(0.0, 89.9);
    let (_, lat_recovered) = natural_earth_inv(0.0, y_extreme);
    assert!((lat_recovered - 89.9).abs() < 0.1, "natural-earth inverse at y for lat=89.9: recovered {lat_recovered}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Natural Earth: antisymmetry
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn natural_earth_y_antisymmetry() {
    for lat in [10.0, 30.0, 60.0, 85.0] {
        let (_, y_pos) = natural_earth_fwd(0.0, lat);
        let (_, y_neg) = natural_earth_fwd(0.0, -lat);
        assert!((y_pos + y_neg).abs() < 1e-12, "natural-earth y should be antisymmetric: y({lat})={y_pos}, y(-{lat})={y_neg}");
    }
}

#[test]
fn natural_earth_x_antisymmetry() {
    for lon in [10.0, 90.0, 180.0] {
        let (x_pos, _) = natural_earth_fwd(lon, 45.0);
        let (x_neg, _) = natural_earth_fwd(-lon, 45.0);
        assert!((x_pos + x_neg).abs() < 1e-12, "natural-earth x should be antisymmetric: x({lon})={x_pos}, x(-{lon})={x_neg}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Natural Earth: hand-computed ne_poly / ne_poly_deriv values
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ne_poly_ne_a_at_phi_one() {
    // p2=p4=p6=1, so the sum degenerates to a plain sum of coefficients.
    let val = ne_poly(&NE_A, 1.0);
    let expected = NE_A[0] + NE_A[1] + NE_A[2] + NE_A[3];
    assert!((val - expected).abs() < 1e-12, "ne_poly(NE_A, 1.0) = {val}, expected {expected}");
}

#[test]
fn ne_poly_ne_b_at_zero() {
    let val = ne_poly(&NE_B, 0.0);
    assert!((val - NE_B[0]).abs() < 1e-15, "ne_poly(NE_B, 0) should be NE_B[0]={}, got {val}", NE_B[0]);
}

#[test]
fn ne_poly_deriv_at_zero() {
    let val = ne_poly_deriv(&NE_B, 0.0);
    assert!(val.abs() < 1e-15, "ne_poly_deriv(NE_B, 0) should be 0, got {val}");
}

#[test]
fn ne_poly_deriv_ne_b_at_one() {
    let val = ne_poly_deriv(&NE_B, 1.0);
    let expected = 2.0 * NE_B[1] + 4.0 * NE_B[2] + 6.0 * NE_B[3] + 8.0 * NE_B[4];
    assert!((val - expected).abs() < 1e-12, "ne_poly_deriv(NE_B, 1.0) = {val}, expected {expected}");
}

#[test]
fn ne_poly_deriv_ne_a_at_pi_over_4() {
    // Regression snapshot (not re-derived from NE_A's coefficients at test
    // time): NE_A has only 4 elements (no phi^7 term), so this exercises the
    // `coeffs.len() > 4` false branch of `ne_poly_deriv`.
    let val = ne_poly_deriv(&NE_A, PI / 4.0);
    let expected = -0.05616903382768482;
    assert!((val - expected).abs() < 1e-12, "ne_poly_deriv(NE_A, pi/4) = {val}, expected {expected}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Orthographic: hemisphere boundary edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn orthographic_at_horizon_lon90() {
    let (x, y) = orthographic_fwd(90.0, 0.0);
    assert!(x.is_finite() && y.is_finite(), "orthographic at lon=90 (horizon) should be finite, got ({x}, {y})");
    assert!((x - 1.0).abs() < 1e-10, "orthographic at lon=90: x should be ~1.0, got {x}");
}

#[test]
fn orthographic_just_past_horizon() {
    let (x, y) = orthographic_fwd(91.0, 0.0);
    assert!(x.is_nan() && y.is_nan(), "orthographic at lon=91 should be NaN, got ({x}, {y})");
}

#[test]
fn orthographic_neg_horizon() {
    let (x, y) = orthographic_fwd(-90.0, 0.0);
    assert!(x.is_finite() && y.is_finite(), "orthographic at lon=-90 (horizon) should be finite, got ({x}, {y})");
    assert!((x - (-1.0)).abs() < 1e-10, "orthographic at lon=-90: x should be ~-1.0, got {x}");
}

#[test]
fn orthographic_north_pole() {
    let (x, y) = orthographic_fwd(0.0, 90.0);
    assert!(x.is_finite() && y.is_finite(), "orthographic at north pole should be finite, got ({x}, {y})");
    assert!(x.abs() < 1e-10, "orthographic north pole: x should be ~0, got {x}");
    assert!((y - 1.0).abs() < 1e-10, "orthographic north pole: y should be ~1, got {y}");
}

#[test]
fn orthographic_inv_at_origin() {
    let (lon, lat) = orthographic_inv(0.0, 0.0);
    assert!(lon.is_finite() && lat.is_finite(), "orthographic inverse at (0,0) should be finite, got ({lon}, {lat})");
    assert!(lon.abs() < 1e-6 && lat.abs() < 1e-6, "orthographic inverse at (0,0) should give ~(0,0), got ({lon}, {lat})");
}

/// Near-pole round trip (lon=10, lat=89). Also stands in for the mirror's
/// separate `bug_hunt_r2_regression_orthographic_near_pole_round_trip`, which
/// pinned the identical point/tolerance as a "did the Snyder-formula fix
/// hold" regression — same contract, one test.
#[test]
fn orthographic_near_pole_round_trip() {
    let (x, y) = orthographic_fwd(10.0, 89.0);
    if x.is_nan() { return; } // back hemisphere
    let (lon2, lat2) = orthographic_inv(x, y);
    assert!((lon2 - 10.0).abs() < 1e-4, "orthographic near-pole lon round-trip: 10 -> {lon2}");
    assert!((lat2 - 89.0).abs() < 1e-4, "orthographic near-pole lat round-trip: 89 -> {lat2}");
}

#[test]
fn orthographic_pole_with_various_lon() {
    // Due to floating point, cos(pi/2) is ~6.12e-17, not exactly 0, so lon=180
    // (cos(lam) = -1) legitimately trips the < 0.0 guard and yields NaN.
    for lon in [-180.0, -90.0, 0.0, 90.0, 180.0] {
        let (x, y) = orthographic_fwd(lon, 90.0);
        if x.is_nan() {
            let cos_lam = to_rad(lon).cos();
            assert!(cos_lam < 0.0 || cos_lam.abs() < 1e-10, "orthographic NaN at lat=90, lon={lon} with cos(lam)={cos_lam}");
        } else {
            assert!(x.is_finite() && y.is_finite(), "orthographic at lat=90, lon={lon} should be finite, got ({x}, {y})");
        }
    }
}

#[test]
fn orthographic_back_hemisphere() {
    for lon in [91.0, 120.0, 150.0, 179.0, -91.0, -120.0, -179.0] {
        let (x, y) = orthographic_fwd(lon, 0.0);
        assert!(x.is_nan() && y.is_nan(), "orthographic back hemisphere at lon={lon}: expected NaN, got ({x}, {y})");
    }
}

#[test]
fn orthographic_back_hemisphere_high_lat() {
    // cos(60°)=0.5, cos(120°)=-0.5 → product -0.25 < 0 → NaN.
    let (x, y) = orthographic_fwd(120.0, 60.0);
    assert!(x.is_nan() && y.is_nan(), "orthographic back hemisphere at lon=120, lat=60: expected NaN, got ({x}, {y})");
}

// ═══════════════════════════════════════════════════════════════════════════
// Orthographic: known-point numeric correctness
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn orthographic_known_point_45_45() {
    let (x, y) = orthographic_fwd(45.0, 45.0);
    assert!((x - 0.5).abs() < 1e-10, "orthographic at (45,45): x should be 0.5, got {x}");
    assert!((y - 2.0_f64.sqrt() / 2.0).abs() < 1e-10, "orthographic at (45,45): y should be sqrt(2)/2, got {y}");
}

#[test]
fn orthographic_lat0_lon30() {
    let (x, y) = orthographic_fwd(30.0, 0.0);
    assert!((x - 0.5).abs() < 1e-10, "orthographic at (30,0): x should be 0.5, got {x}");
    assert!(y.abs() < 1e-10, "orthographic at (30,0): y should be 0, got {y}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Orthographic: comprehensive round-trip grid + rim/disk boundary
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn orthographic_grid_round_trip() {
    for lon_i in -8..=8 {
        for lat_i in -8..=8 {
            let lon = lon_i as f64 * 10.0;
            let lat = lat_i as f64 * 10.0;
            let (x, y) = orthographic_fwd(lon, lat);
            if x.is_nan() { continue; }
            let (lon2, lat2) = orthographic_inv(x, y);
            assert!((lon2 - lon).abs() < 1e-6, "orthographic grid round-trip lon at ({lon},{lat}): {lon} -> {lon2}");
            assert!((lat2 - lat).abs() < 1e-6, "orthographic grid round-trip lat at ({lon},{lat}): {lat} -> {lat2}");
        }
    }
}

#[test]
fn orthographic_very_near_pole() {
    let lon = 45.0;
    let lat = 89.99;
    let (x, y) = orthographic_fwd(lon, lat);
    if x.is_nan() { return; }
    let (lon2, lat2) = orthographic_inv(x, y);
    assert!((lat2 - lat).abs() < 0.01, "orthographic at lat=89.99: lat round-trip {lat} -> {lat2}");
    // Longitude precision legitimately degrades this close to the pole.
    assert!((lon2 - lon).abs() < 5.0, "orthographic at lat=89.99: lon round-trip {lon} -> {lon2}");
}

#[test]
fn orthographic_inv_at_rim() {
    let (lon, lat) = orthographic_inv(1.0, 0.0);
    assert!(lon.is_finite() && lat.is_finite(), "orthographic inv at rim (1,0) should be finite, got ({lon}, {lat})");
    assert!((lon - 90.0).abs() < 1e-6, "orthographic inv at rim (1,0): lon should be 90, got {lon}");
    assert!(lat.abs() < 1e-6, "orthographic inv at rim (1,0): lat should be 0, got {lat}");
}

#[test]
fn orthographic_inv_outside_disk() {
    let (lon, lat) = orthographic_inv(0.8, 0.7); // rho = sqrt(1.13) > 1
    assert!(lon.is_nan() && lat.is_nan(), "orthographic inv outside disk should be NaN, got ({lon}, {lat})");
}

#[test]
fn orthographic_inv_unit_circle() {
    let angles = [0.0, PI / 6.0, PI / 4.0, PI / 3.0, PI / 2.0, 2.0 * PI / 3.0, PI, 5.0 * PI / 4.0, 3.0 * PI / 2.0];
    for &angle in &angles {
        let x = angle.cos();
        let y = angle.sin();
        let (lon, lat) = orthographic_inv(x, y);
        assert!(lon.is_finite() && lat.is_finite(), "orthographic inv on unit circle at angle={angle}: ({lon}, {lat}) should be finite");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Equal Earth: round-trip fidelity and symmetry
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn equal_earth_equator_round_trip() {
    round_trip(GeoProjection::EqualEarth, -75.0, 0.0, 1e-10);
}

#[test]
fn equal_earth_high_lat_round_trip() {
    round_trip(GeoProjection::EqualEarth, 120.0, 85.0, 1e-6);
}

#[test]
fn equal_earth_north_south_symmetry() {
    for lat in [10.0, 30.0, 60.0, 85.0] {
        let (x_n, y_n) = equal_earth_fwd(50.0, lat);
        let (x_s, y_s) = equal_earth_fwd(50.0, -lat);
        assert!((x_n - x_s).abs() < 1e-12, "equal-earth x should be symmetric: x_n={x_n}, x_s={x_s} at lat=+/-{lat}");
        assert!((y_n + y_s).abs() < 1e-12, "equal-earth y should be antisymmetric: y_n={y_n}, y_s={y_s} at lat=+/-{lat}");
    }
}

#[test]
fn equal_earth_lon_360_is_double_180() {
    // x is linear in lambda, so x(360) should be exactly 2*x(180).
    let (x180, _) = equal_earth_fwd(180.0, 0.0);
    let (x360, _) = equal_earth_fwd(360.0, 0.0);
    assert!((x360 - 2.0 * x180).abs() < 1e-10, "equal-earth: x(360) should be 2*x(180), got x360={x360}, 2*x180={}", 2.0 * x180);
}

#[test]
fn ee_x_factor_at_zero() {
    let val = ee_x_factor(0.0);
    assert!((val - EE_A[0]).abs() < 1e-15, "ee_x_factor(0) should be EE_A[0]={}, got {val}", EE_A[0]);
}

#[test]
fn ee_x_factor_at_one() {
    let expected = EE_A[0] + EE_A[1] + EE_A[2] + EE_A[3] + EE_A[4];
    let actual = ee_x_factor(1.0);
    assert!((actual - expected).abs() < 1e-15, "ee_x_factor(1) should be sum(EE_A)={expected}, got {actual}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Equal Earth: pole round-trip, dateline, and asin-domain safety
// ═══════════════════════════════════════════════════════════════════════════

/// Also covers `bug_hunt_r2_equal_earth_theta_lat_recovery_at_pole`, which
/// re-derived the same theta-at-pole/lat-recovery identity by hand instead of
/// calling `equal_earth_fwd`/`equal_earth_inv` — the real round-trip at
/// lat=90 below exercises the identical contract through the real functions.
#[test]
fn equal_earth_pole_round_trip() {
    for lat in [90.0, -90.0] {
        // x=0 at the poles regardless of lon, so only lat recovery is meaningful.
        let (x, y) = equal_earth_fwd(0.0, lat);
        let (_lon2, lat2) = equal_earth_inv(x, y);
        assert!((lat2 - lat).abs() < 1e-6, "equal-earth pole lat round-trip: {lat} -> {lat2}");
    }
}

#[test]
fn equal_earth_dateline_equator() {
    let (x, y) = equal_earth_fwd(180.0, 0.0);
    assert!(x.is_finite() && y.is_finite(), "equal-earth at dateline/equator should be finite: ({x}, {y})");
    assert!(y.abs() < 1e-12, "equal-earth at equator: y should be 0, got {y}");
    let expected_x = PI * EE_A[0];
    assert!((x - expected_x).abs() < 1e-10, "equal-earth at dateline/equator: x should be pi*A[0]={expected_x}, got {x}");
}

#[test]
fn equal_earth_pole_with_nonzero_lon_round_trip() {
    let lon = 180.0;
    let lat = 90.0;
    let (x, y) = equal_earth_fwd(lon, lat);
    let (lon2, lat2) = equal_earth_inv(x, y);
    assert!((lat2 - lat).abs() < 1e-4, "equal-earth pole+lon=180: lat {lat} -> {lat2}");
    if x.abs() > 1e-10 {
        assert!((lon2 - lon).abs() < 1e-4, "equal-earth pole+lon=180: lon {lon} -> {lon2} (x={x})");
    }
}

#[test]
fn equal_earth_lat89_round_trip() {
    round_trip(GeoProjection::EqualEarth, 30.0, 89.0, 1e-4);
}

#[test]
fn equal_earth_inv_theta_at_max_valid_y_is_pole() {
    // theta=pi/3 is the max valid theta (asin argument saturates at 1 there).
    let y_max = EE_M * (PI / 3.0);
    let (_, lat) = equal_earth_inv(0.0, y_max);
    assert!((lat - 90.0).abs() < 1e-6, "equal-earth inv at max valid y: lat should be 90, got {lat}");
}

/// The forward asin argument is `(sqrt(3)/2) * sin(phi)`, always within
/// `[-0.866, 0.866]` since `sqrt(3)/2 < 1`. Rather than re-deriving that bound
/// algebraically (which the mirror test did without calling any real
/// function), this exercises the real `equal_earth_fwd` across the full
/// latitude range and confirms no domain violation ever surfaces as NaN.
#[test]
fn equal_earth_fwd_never_hits_asin_domain_violation() {
    for lat in [-90.0, -89.0, -45.0, 0.0, 45.0, 89.0, 90.0] {
        let (_, y) = equal_earth_fwd(0.0, lat);
        assert!(y.is_finite(), "equal-earth fwd at lat={lat} should never violate asin domain, got y={y}");
    }
}

#[test]
fn equal_earth_inv_asin_domain_at_large_y_no_panic() {
    // At y = EE_M * pi/2 the *inverse's* asin argument is 2/sqrt(3) > 1 (out of
    // the forward's normal range) — Rust's asin returns NaN for |x|>1, not a panic.
    let y_critical = EE_M * (PI / 2.0);
    let (_, lat) = equal_earth_inv(0.0, y_critical);
    assert!(lat.is_nan() || lat.is_finite(), "equal-earth inv at critical y: should not panic, got {lat}");
}

#[test]
fn equal_earth_inv_large_y_no_panic() {
    let (lon, lat) = equal_earth_inv(0.0, 100.0);
    let _ = (lon, lat);
}

#[test]
fn equal_earth_inv_large_x_no_panic() {
    let (lon, lat) = equal_earth_inv(100.0, 0.0);
    assert!(lon.is_finite(), "equal-earth inv with large x should give finite lon, got {lon}");
    assert!(lat.abs() < 1e-10, "equal-earth inv at y=0: lat should be 0, got {lat}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Albers USA: boundary-exact routing thresholds
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn albers_usa_exactly_at_alaska_boundary() {
    // lat=49.5 is NOT > 49.5, so this must stay on the continental path.
    let (x_boundary, _) = albers_usa_fwd(-130.0, 49.5);
    let (x_continental, _) = albers_conic_fwd(-130.0, 49.5, 29.5, 45.5, 38.0, -96.0);
    assert!((x_boundary - x_continental).abs() < 1e-12, "lat=49.5 should NOT enter Alaska branch (needs > 49.5)");
}

#[test]
fn albers_usa_just_above_alaska_threshold() {
    let (x_above, _) = albers_usa_fwd(-130.0, 49.501);
    let (x_continental, _) = albers_conic_fwd(-130.0, 49.501, 29.5, 45.5, 38.0, -96.0);
    assert!((x_above - x_continental).abs() > 0.01, "lat=49.501 with lon<-127.5 should enter Alaska branch");
}

#[test]
fn albers_usa_alaska_lon_boundary() {
    // lon=-127.5 is NOT < -127.5, so this must stay on the continental path.
    let (x_boundary, _) = albers_usa_fwd(-127.5, 60.0);
    let (x_continental, _) = albers_conic_fwd(-127.5, 60.0, 29.5, 45.5, 38.0, -96.0);
    assert!((x_boundary - x_continental).abs() < 1e-12, "lon=-127.5 should NOT enter Alaska branch (needs < -127.5)");
}

#[test]
fn albers_usa_hawaii_lat_lower_boundary() {
    let (x_boundary, _) = albers_usa_fwd(-160.0, 18.0);
    let (x_continental, _) = albers_conic_fwd(-160.0, 18.0, 29.5, 45.5, 38.0, -96.0);
    assert!((x_boundary - x_continental).abs() < 1e-12, "lat=18.0 should NOT enter Hawaii branch (needs > 18.0)");
}

#[test]
fn albers_usa_hawaii_lat_upper_boundary() {
    let (x_boundary, _) = albers_usa_fwd(-160.0, 24.0);
    let (x_continental, _) = albers_conic_fwd(-160.0, 24.0, 29.5, 45.5, 38.0, -96.0);
    assert!((x_boundary - x_continental).abs() < 1e-12, "lat=24.0 should NOT enter Hawaii branch (needs < 24.0)");
}

#[test]
fn albers_usa_texas_goes_continental() {
    let (x, y) = albers_usa_fwd(-100.0, 30.0);
    let (cx, cy) = albers_conic_fwd(-100.0, 30.0, 29.5, 45.5, 38.0, -96.0);
    assert!((x - cx).abs() < 1e-12 && (y - cy).abs() < 1e-12, "Texas should use continental conic");
}

#[test]
fn albers_usa_ocean_point_south_of_hawaii_goes_continental() {
    let (x, y) = albers_usa_fwd(-160.0, 17.0);
    assert!(x.is_finite() && y.is_finite(), "albers USA at ocean point should be finite: ({x}, {y})");
    let (cx, cy) = albers_conic_fwd(-160.0, 17.0, 29.5, 45.5, 38.0, -96.0);
    assert!((x - cx).abs() < 1e-12 && (y - cy).abs() < 1e-12, "ocean point should use continental conic");
}

#[test]
fn albers_usa_above_hawaii_boundary_goes_continental() {
    let (x, y) = albers_usa_fwd(-160.0, 25.0);
    let (cx, cy) = albers_conic_fwd(-160.0, 25.0, 29.5, 45.5, 38.0, -96.0);
    assert!((x - cx).abs() < 1e-12 && (y - cy).abs() < 1e-12, "lat=25 with lon=-160 should use continental conic, not Hawaii");
}

#[test]
fn albers_usa_alaska_routing_discontinuity() {
    let lon = -130.0;
    let (x_in, y_in) = albers_usa_fwd(lon, 49.501); // Alaska path
    let (x_out, y_out) = albers_usa_fwd(lon, 49.499); // continental path
    let jump = ((x_in - x_out).powi(2) + (y_in - y_out).powi(2)).sqrt();
    assert!(jump > 0.1, "expected large discontinuity at Alaska boundary; jump = {jump}");
    assert!(x_in.is_finite() && y_in.is_finite(), "Alaska side should be finite: ({x_in}, {y_in})");
    assert!(x_out.is_finite() && y_out.is_finite(), "Continental side should be finite: ({x_out}, {y_out})");
}

#[test]
fn albers_usa_hawaii_routing_discontinuity() {
    let lon = -160.0;
    let (x_in, y_in) = albers_usa_fwd(lon, 18.001); // Hawaii path
    let (x_out, y_out) = albers_usa_fwd(lon, 17.999); // continental path
    let jump = ((x_in - x_out).powi(2) + (y_in - y_out).powi(2)).sqrt();
    assert!(jump > 0.1, "expected large discontinuity at Hawaii lower boundary; jump = {jump}");
    assert!(x_in.is_finite() && y_in.is_finite(), "Hawaii side should be finite: ({x_in}, {y_in})");
    assert!(x_out.is_finite() && y_out.is_finite(), "Continental side should be finite: ({x_out}, {y_out})");
}

// ═══════════════════════════════════════════════════════════════════════════
// Albers USA inverse: Alaska/Hawaii-inset detection and round trip
// ═══════════════════════════════════════════════════════════════════════════

/// `albers_usa_inv` detects the Alaska inset's output region (`x < -1.3 &&
/// y < -0.4`) and inverts through the Alaska conic (55, 65, 50, -154), so a
/// forward point that entered the Alaska inset round-trips accurately — it
/// is not limited to the continental conic. `1e-6` pins the real precision
/// (measured error ~1.4e-14) tightly enough that a regression in the inset
/// detection thresholds (which would silently fall through to the
/// continental conic instead) is caught.
#[test]
fn albers_usa_inv_alaska_inset_round_trips() {
    let lon = -160.0;
    let lat = 64.0;
    assert!(lat > 49.5 && lon < -127.5, "test expects Alaska routing");
    let (x, y) = albers_usa_fwd(lon, lat);
    let (lon2, lat2) = albers_usa_inv(x, y);
    assert!((lon2 - lon).abs() < 1e-6, "albers_usa_inv should route the Alaska inset: lon {lon} -> {lon2}");
    assert!((lat2 - lat).abs() < 1e-6, "albers_usa_inv should route the Alaska inset: lat {lat} -> {lat2}");
}

/// Same contract as above for the Hawaii inset's output region
/// (`0.0 < x < 0.9 && y < -0.7`), inverted through the Hawaii conic
/// (8, 18, 13, -157).
#[test]
fn albers_usa_inv_hawaii_inset_round_trips() {
    let lon = -157.0;
    let lat = 21.0;
    assert!(lat < 24.0 && lat > 18.0 && lon < -154.0, "test expects Hawaii routing");
    let (x, y) = albers_usa_fwd(lon, lat);
    let (lon2, lat2) = albers_usa_inv(x, y);
    assert!((lon2 - lon).abs() < 1e-6, "albers_usa_inv should route the Hawaii inset: lon {lon} -> {lon2}");
    assert!((lat2 - lat).abs() < 1e-6, "albers_usa_inv should route the Hawaii inset: lat {lat} -> {lat2}");
}

#[test]
fn albers_usa_inv_extreme_coords_asin_domain_no_panic() {
    let (lon, lat) = albers_usa_inv(100.0, 100.0);
    let _ = (lon, lat);
}

#[test]
fn albers_usa_inv_at_center_accurate() {
    let lon0 = -96.0;
    let lat0 = 38.0;
    let (x, y) = albers_usa_fwd(lon0, lat0);
    let (lon2, lat2) = albers_usa_inv(x, y);
    assert!((lon2 - lon0).abs() < 1e-6, "albers USA inverse at center lon: {} -> {} (diff {})", lon0, lon2, (lon2 - lon0).abs());
    assert!((lat2 - lat0).abs() < 1e-6, "albers USA inverse at center lat: {} -> {} (diff {})", lat0, lat2, (lat2 - lat0).abs());
}

#[test]
fn albers_usa_inv_x_zero_on_central_meridian() {
    let (x, y) = albers_usa_fwd(-96.0, 40.0);
    assert!(x.abs() < 1e-10, "x at central meridian should be ~0, got {x}");
    let (lon2, lat2) = albers_usa_inv(x, y);
    assert!((lon2 - (-96.0)).abs() < 1e-4, "albers inv at x=0: lon should be ~-96, got {lon2}");
    assert!((lat2 - 40.0).abs() < 1e-4, "albers inv at x=0: lat should be ~40, got {lat2}");
}

#[test]
fn albers_usa_continental_round_trip_grid() {
    let test_points = [
        (-74.0, 40.7),   // New York
        (-87.6, 41.9),   // Chicago
        (-118.2, 34.1),  // Los Angeles
        (-122.4, 37.8),  // San Francisco
        (-77.0, 38.9),   // Washington DC
        (-95.4, 29.8),   // Houston
        (-104.9, 39.7),  // Denver
    ];
    for (lon, lat) in test_points {
        assert!(!(lat > 49.5 && lon < -127.5), "({lon},{lat}) should not be Alaska");
        assert!(!(lat < 24.0 && lat > 18.0 && lon < -154.0), "({lon},{lat}) should not be Hawaii");
        round_trip(GeoProjection::AlbersUsa, lon, lat, 1e-4);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Albers conic: numeric correctness and degenerate configurations
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn albers_conic_equal_standard_parallels_finite() {
    let (x, y) = albers_conic_fwd(0.0, 0.0, 45.0, 45.0, 45.0, 0.0);
    assert!(x.is_finite() && y.is_finite(), "albers conic with sp1==sp2 should be finite: ({x}, {y})");
}

#[test]
fn albers_conic_zero_standard_parallels_no_panic() {
    // n = (sin(0)+sin(0))/2 = 0 → division by zero; infinity/NaN are acceptable.
    let (x, y) = albers_conic_fwd(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let _ = (x, y);
}

#[test]
fn albers_conic_on_central_meridian_x_zero() {
    let lon0 = -96.0;
    let (x, _) = albers_conic_fwd(lon0, 40.0, 29.5, 45.5, 38.0, lon0);
    assert!(x.abs() < 1e-12, "albers conic on central meridian should give x=0, got {x}");
}

#[test]
fn albers_conic_southern_hemisphere_finite_on_meridian() {
    let (x, y) = albers_conic_fwd(0.0, -35.0, -30.0, -45.0, -37.5, 0.0);
    assert!(x.is_finite() && y.is_finite(), "albers conic with southern hemisphere parallels should produce finite coords, got ({x}, {y})");
    assert!(x.abs() < 1e-10, "albers conic southern hemisphere on central meridian: x should be ~0, got {x}");
}

#[test]
fn albers_conic_equatorial_symmetric_no_panic() {
    // sp1=-20, sp2=20 → n = 0 → division by zero; must not panic.
    let (x, y) = albers_conic_fwd(0.0, 0.0, -20.0, 20.0, 0.0, 0.0);
    let _ = (x, y);
}

#[test]
fn albers_conic_near_south_pole_parallels_finite() {
    let (x, y) = albers_conic_fwd(10.0, -85.0, -89.0, -89.0, -89.0, 0.0);
    assert!(x.is_finite() && y.is_finite(), "albers conic near south pole should produce finite coords, got ({x}, {y})");
}

#[test]
fn albers_conic_center_point_maps_to_origin() {
    // Also stands in for `bug_hunt_r2_albers_rho0_formula` (a pure-algebra
    // restatement of `rho0 = sqrt(C)/n - sin(phi0)/n == (sqrt(C)-sin(phi0))/n`
    // that called no real function) and `bug_hunt_r2_albers_rho0_vs_snyder`
    // (identical center-point assertion) — both are exactly this property of
    // the real `albers_conic_fwd`'s rho0 computation.
    let sp1 = 29.5;
    let sp2 = 45.5;
    let lat0 = 38.0;
    let lon0 = -96.0;
    let (x, y) = albers_conic_fwd(lon0, lat0, sp1, sp2, lat0, lon0);
    assert!(x.abs() < 1e-10, "albers conic at center: x should be 0, got {x}");
    assert!(y.abs() < 1e-10, "albers conic at center: y should be ~0, got {y}");
}

#[test]
fn albers_conic_east_west_symmetry() {
    let sp1 = 29.5;
    let sp2 = 45.5;
    let lat0 = 38.0;
    let lon0 = -96.0;
    let delta_lon = 20.0;
    let (x_east, y_east) = albers_conic_fwd(lon0 + delta_lon, 40.0, sp1, sp2, lat0, lon0);
    let (x_west, y_west) = albers_conic_fwd(lon0 - delta_lon, 40.0, sp1, sp2, lat0, lon0);
    assert!((x_east + x_west).abs() < 1e-10, "albers conic east-west: x_east={x_east}, x_west={x_west} should be opposite");
    assert!((y_east - y_west).abs() < 1e-10, "albers conic east-west: y_east={y_east}, y_west={y_west} should be equal");
}

#[test]
fn albers_conic_lat_monotonicity() {
    // rho shrinks as lat increases (northern hemisphere), so y = rho0 - rho
    // grows monotonically along the central meridian.
    let sp1 = 29.5;
    let sp2 = 45.5;
    let lat0 = 38.0;
    let lon0 = -96.0;
    let mut prev_y = f64::NEG_INFINITY;
    for lat_i in (25..=55).step_by(5) {
        let lat = lat_i as f64;
        let (_, y) = albers_conic_fwd(lon0, lat, sp1, sp2, lat0, lon0);
        assert!(y > prev_y, "albers conic y should increase as lat increases: y({})={} <= y(prev)={}", lat, y, prev_y);
        prev_y = y;
    }
}

#[test]
fn albers_conic_rho_clamped_to_zero_stays_finite() {
    // At very high latitude the `.max(0.0)` clamp on `c - 2n*sin(phi)` can kick
    // in; when it does, rho=0 must still give a finite (x=0, y=rho0) point.
    let sp1 = 29.5;
    let sp2 = 45.5;
    let lat0 = 38.0;
    let lon0 = -96.0;
    let phi = to_rad(89.0);
    let phi1 = to_rad(sp1);
    let phi2 = to_rad(sp2);
    let n = (phi1.sin() + phi2.sin()) / 2.0;
    let c = phi2.cos().powi(2) + 2.0 * n * phi2.sin();
    let rho_arg = c - 2.0 * n * phi.sin();
    if rho_arg < 0.0 {
        let (x, y) = albers_conic_fwd(-96.0, 89.0, sp1, sp2, lat0, lon0);
        assert!(x.abs() < 1e-10, "albers conic with clamped rho: x should be 0, got {x}");
        assert!(y.is_finite(), "albers conic with clamped rho: y should be finite, got {y}");
    }
}

#[test]
fn albers_conic_inv_extreme_coords_no_panic() {
    // rho becomes huge → (c - rho^2*n^2)/(2n) can exceed asin's [-1,1] domain.
    let (lon, lat) = albers_conic_inv(100.0, 100.0, 29.5, 45.5, 38.0, -96.0);
    let _ = (lon, lat);
}

// ═══════════════════════════════════════════════════════════════════════════
// Mercator: numeric correctness at known points
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn mercator_lon180_gives_pi() {
    let (x, _) = mercator_fwd(180.0, 0.0);
    assert!((x - PI).abs() < 1e-12, "mercator at lon=180 should give x=pi, got {x}");
}

#[test]
fn mercator_lon90_gives_half_pi() {
    let (x, _) = mercator_fwd(90.0, 0.0);
    assert!((x - PI / 2.0).abs() < 1e-12, "mercator at lon=90 should give x=pi/2, got {x}");
}

#[test]
fn mercator_equator_y_zero() {
    let (_, y) = mercator_fwd(0.0, 0.0);
    assert!(y.abs() < 1e-15, "mercator at equator: y should be 0, got {y}");
}

#[test]
fn mercator_lat45_numeric() {
    // Regression snapshot: ln(tan(pi/4 + to_rad(45)/2)) = ln(tan(3*pi/8)).
    let (_, y) = mercator_fwd(0.0, 45.0);
    let expected = 0.8813735870195428;
    assert!((y - expected).abs() < 1e-12, "mercator at lat=45: y={y}, expected={expected}");
}

#[test]
fn mercator_wide_round_trip() {
    for lat in (-80..=80).step_by(10) {
        round_trip(GeoProjection::Mercator, 45.0, lat as f64, 1e-10);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Mercator inverse: infinite / zero y
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn mercator_inv_inf_y_gives_pole() {
    let (lon, lat) = mercator_inv(0.0, f64::INFINITY);
    assert!(lon.is_finite(), "mercator inv: inf y should give finite lon, got {lon}");
    assert!((lat - 90.0).abs() < 1e-6, "mercator inv: inf y should give lat=90, got {lat}");
}

#[test]
fn mercator_inv_neg_inf_y_gives_south_pole() {
    let (lon, lat) = mercator_inv(0.0, f64::NEG_INFINITY);
    assert!(lon.is_finite(), "mercator inv: -inf y should give finite lon, got {lon}");
    assert!((lat - (-90.0)).abs() < 1e-6, "mercator inv: -inf y should give lat=-90, got {lat}");
}

#[test]
fn mercator_inv_inf_x() {
    let (lon, _) = mercator_inv(f64::INFINITY, 0.0);
    assert!(lon.is_infinite(), "mercator inv: inf x should give inf lon, got {lon}");
}

#[test]
fn mercator_inv_y_zero_gives_lat_zero() {
    let (_, lat) = mercator_inv(0.0, 0.0);
    assert!(lat.abs() < 1e-12, "mercator inv at y=0 should give lat=0, got {lat}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Cross-projection: NaN / infinity / large-value stress, no panic
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn all_projections_double_nan_no_panic() {
    let _ = mercator_fwd(f64::NAN, f64::NAN);
    let _ = equirect_fwd(f64::NAN, f64::NAN);
    let _ = equal_earth_fwd(f64::NAN, f64::NAN);
    let _ = natural_earth_fwd(f64::NAN, f64::NAN);
    let _ = orthographic_fwd(f64::NAN, f64::NAN);
    let _ = albers_usa_fwd(f64::NAN, f64::NAN);
}

#[test]
fn all_inverse_projections_double_nan_no_panic() {
    let _ = mercator_inv(f64::NAN, f64::NAN);
    let _ = equirect_inv(f64::NAN, f64::NAN);
    let _ = equal_earth_inv(f64::NAN, f64::NAN);
    let _ = natural_earth_inv(f64::NAN, f64::NAN);
    let _ = orthographic_inv(f64::NAN, f64::NAN);
}

#[test]
fn all_inv_inf_no_panic() {
    let _ = mercator_inv(f64::INFINITY, f64::INFINITY);
    let _ = equirect_inv(f64::INFINITY, f64::INFINITY);
    let _ = equal_earth_inv(f64::INFINITY, f64::INFINITY);
    let _ = natural_earth_inv(f64::INFINITY, f64::INFINITY);
    let _ = orthographic_inv(f64::INFINITY, f64::INFINITY);
}

#[test]
fn all_inv_neg_inf_no_panic() {
    let _ = mercator_inv(f64::NEG_INFINITY, f64::NEG_INFINITY);
    let _ = equirect_inv(f64::NEG_INFINITY, f64::NEG_INFINITY);
    let _ = equal_earth_inv(f64::NEG_INFINITY, f64::NEG_INFINITY);
    let _ = natural_earth_inv(f64::NEG_INFINITY, f64::NEG_INFINITY);
    let _ = orthographic_inv(f64::NEG_INFINITY, f64::NEG_INFINITY);
}

#[test]
fn very_large_lon_no_panic() {
    let _ = mercator_fwd(1e6, 0.0);
    let _ = equirect_fwd(1e6, 0.0);
    let _ = equal_earth_fwd(1e6, 0.0);
    let _ = natural_earth_fwd(1e6, 0.0);
    let _ = orthographic_fwd(1e6, 0.0);
    let _ = albers_usa_fwd(1e6, 0.0);
}

#[test]
fn very_large_lat_no_panic() {
    let _ = mercator_fwd(0.0, 1e6);
    let _ = equirect_fwd(0.0, 1e6);
    let _ = equal_earth_fwd(0.0, 1e6);
    let _ = natural_earth_fwd(0.0, 1e6);
    let _ = orthographic_fwd(0.0, 1e6);
    let _ = albers_usa_fwd(0.0, 1e6);
}

#[test]
fn subnormal_lat_no_panic() {
    let tiny = 5e-324_f64; // smallest positive subnormal
    let _ = mercator_fwd(0.0, tiny);
    let _ = equirect_fwd(0.0, tiny);
    let _ = equal_earth_fwd(0.0, tiny);
    let _ = natural_earth_fwd(0.0, tiny);
    let _ = orthographic_fwd(0.0, tiny);
}

#[test]
fn min_positive_no_panic() {
    let tiny = f64::MIN_POSITIVE;
    for fwd in [mercator_fwd, equirect_fwd, equal_earth_fwd, natural_earth_fwd, orthographic_fwd] {
        let (x, y) = fwd(tiny, tiny);
        assert!(x.is_finite() && y.is_finite(), "projection with MIN_POSITIVE should be finite, got ({x}, {y})");
    }
}

#[test]
fn negative_zero_same_as_zero() {
    let (x_pos, y_pos) = mercator_fwd(0.0, 0.0);
    let (x_neg, y_neg) = mercator_fwd(-0.0, -0.0);
    assert!((x_pos - x_neg).abs() < 1e-15 && (y_pos - y_neg).abs() < 1e-15, "mercator: -0.0 should equal 0.0");
}

#[test]
fn all_projections_origin_gives_zero() {
    // Subsumes `bug_hunt_equirect_origin` (same contract, narrower scope).
    type Fwd = fn(f64, f64) -> (f64, f64);
    let projs: [(&str, Fwd); 5] = [
        ("mercator", mercator_fwd),
        ("equirect", equirect_fwd),
        ("equal_earth", equal_earth_fwd),
        ("natural_earth", natural_earth_fwd),
        ("orthographic", orthographic_fwd),
    ];
    for (name, fwd) in projs {
        let (x, y) = fwd(0.0, 0.0);
        assert!(x.abs() < 1e-12, "{name} at origin: x should be 0, got {x}");
        assert!(y.abs() < 1e-12, "{name} at origin: y should be 0, got {y}");
    }
}
