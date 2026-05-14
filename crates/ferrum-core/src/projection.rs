//! Pure-Rust map projection functions.
//!
//! Forward: (lon°, lat°) → normalized (x, y).
//! Inverse: normalized (x, y) → (lon°, lat°).
//! Newton-Raphson tolerance: 1e-12, max 20 iterations.

use std::f64::consts::{FRAC_PI_4, PI};

use ferrum_scene::GeoProjection;

#[inline(always)]
fn to_rad(deg: f64) -> f64 { deg * PI / 180.0 }

// Only needed by #[cfg(test)] inverse functions; suppress dead-code lint.
#[cfg(test)]
#[inline(always)]
fn to_deg(r: f64) -> f64 { r * 180.0 / PI }

// ── Public dispatch ──────────────────────────────────────────────────────

pub fn forward(proj: GeoProjection, lon: f64, lat: f64) -> (f64, f64) {
    match proj {
        GeoProjection::Mercator        => mercator_fwd(lon, lat),
        GeoProjection::Equirectangular => equirect_fwd(lon, lat),
        GeoProjection::EqualEarth      => equal_earth_fwd(lon, lat),
        GeoProjection::NaturalEarth    => natural_earth_fwd(lon, lat),
        GeoProjection::Orthographic    => orthographic_fwd(lon, lat),
        GeoProjection::AlbersUsa       => albers_usa_fwd(lon, lat),
    }
}

/// Inverse projection: normalized (x, y) → (lon°, lat°).
///
/// Kept under `#[cfg(test)]` until a non-test caller lands (geo hit-testing in Phase 11e
/// will move this into a shared crate accessible from ferrum-wasm). Round-trip tests
/// live in the same file.
#[cfg(test)]
pub fn inverse(proj: GeoProjection, x: f64, y: f64) -> (f64, f64) {
    match proj {
        GeoProjection::Mercator        => mercator_inv(x, y),
        GeoProjection::Equirectangular => equirect_inv(x, y),
        GeoProjection::EqualEarth      => equal_earth_inv(x, y),
        GeoProjection::NaturalEarth    => natural_earth_inv(x, y),
        GeoProjection::Orthographic    => orthographic_inv(x, y),
        GeoProjection::AlbersUsa       => albers_usa_inv(x, y),
    }
}

// ── Mercator ─────────────────────────────────────────────────────────────

fn mercator_fwd(lon: f64, lat: f64) -> (f64, f64) {
    let lat_c = lat.clamp(-85.051_129_34, 85.051_129_34);
    (to_rad(lon), (FRAC_PI_4 + to_rad(lat_c) / 2.0).tan().ln())
}


#[cfg(test)]
fn mercator_inv(x: f64, y: f64) -> (f64, f64) {
    (to_deg(x), to_deg(2.0 * y.exp().atan() - (PI / 2.0)))
}

// ── Equirectangular ───────────────────────────────────────────────────────

fn equirect_fwd(lon: f64, lat: f64) -> (f64, f64) { (to_rad(lon), to_rad(lat)) }

#[cfg(test)]
fn equirect_inv(x: f64, y: f64) -> (f64, f64) { (to_deg(x), to_deg(y)) }

// ── Equal Earth ───────────────────────────────────────────────────────────
// Savcic et al. 2018.

const EE_A: [f64; 5] = [1.340264, -0.081106, 0.000893, 0.003796, -0.001722];
const EE_M: f64 = PI / 2.0;

fn ee_x_factor(t2: f64) -> f64 {
    let t4 = t2 * t2;
    let t6 = t4 * t2;
    let t8 = t4 * t4;
    EE_A[0] + EE_A[1]*t2 + EE_A[2]*t4 + EE_A[3]*t6 + EE_A[4]*t8
}

fn equal_earth_fwd(lon: f64, lat: f64) -> (f64, f64) {
    let lam = to_rad(lon);
    let phi = to_rad(lat);
    let theta = ((3.0_f64.sqrt() / 2.0) * phi.sin()).asin();
    let t2 = theta * theta;
    (lam * ee_x_factor(t2), EE_M * theta)
}


#[cfg(test)]
fn equal_earth_inv(x: f64, y: f64) -> (f64, f64) {
    let mut theta = y / EE_M;
    for _ in 0..20 {
        let delta = (EE_M * theta - y) / EE_M;
        theta -= delta;
        if delta.abs() < 1e-12 { break; }
    }
    let t2 = theta * theta;
    let lf = ee_x_factor(t2);
    let lon = if lf.abs() < 1e-12 { 0.0 } else { to_deg(x / lf) };
    let lat = to_deg(((2.0 * theta.sin()) / 3.0_f64.sqrt()).asin());
    (lon, lat)
}

// ── Natural Earth ─────────────────────────────────────────────────────────
// Patterson et al. polynomial approximation.

const NE_A: [f64; 4] = [0.8707, -0.1310, 0.0772, 0.0];
const NE_B: [f64; 5] = [1.007226, 0.015085, -0.044475, 0.028874, -0.005916];

fn ne_poly(coeffs: &[f64], phi: f64) -> f64 {
    let p2 = phi * phi;
    let p4 = p2 * p2;
    let p6 = p4 * p2;
    coeffs[0] + p2*coeffs[1] + p4*coeffs[2] + p6*coeffs[3]
}

fn natural_earth_fwd(lon: f64, lat: f64) -> (f64, f64) {
    let lam = to_rad(lon);
    let phi = to_rad(lat);
    (lam * ne_poly(&NE_A, phi), phi * ne_poly(&NE_B, phi))
}


#[cfg(test)]
fn natural_earth_inv(x: f64, y: f64) -> (f64, f64) {
    let mut phi = y;
    for _ in 0..20 {
        let fy = phi * ne_poly(&NE_B, phi);
        let dfy = ne_poly(&NE_B, phi);
        let delta = (fy - y) / dfy.max(1e-12);
        phi -= delta;
        if delta.abs() < 1e-12 { break; }
    }
    let lf = ne_poly(&NE_A, phi);
    let lon = if lf.abs() < 1e-12 { 0.0 } else { to_deg(x / lf) };
    (lon, to_deg(phi))
}

// ── Orthographic ──────────────────────────────────────────────────────────

fn orthographic_fwd(lon: f64, lat: f64) -> (f64, f64) {
    let lam = to_rad(lon);
    let phi = to_rad(lat);
    let cos_phi = phi.cos();
    if cos_phi * lam.cos() < 0.0 { return (f64::NAN, f64::NAN); }
    (cos_phi * lam.sin(), phi.sin())
}


#[cfg(test)]
fn orthographic_inv(x: f64, y: f64) -> (f64, f64) {
    let rho = (x*x + y*y).sqrt();
    if rho > 1.0 { return (f64::NAN, f64::NAN); }
    let c = rho.asin();
    let lat = (c.cos() * y / rho.max(1e-12)).asin();
    let lon = x.atan2((c.cos() * (1.0 - rho*rho).sqrt()));
    (to_deg(lon), to_deg(lat))
}

// ── Albers USA ────────────────────────────────────────────────────────────

fn albers_conic_fwd(lon: f64, lat: f64, sp1: f64, sp2: f64, lat0: f64, lon0: f64) -> (f64, f64) {
    let (phi1, phi2, phi0, lam0) = (to_rad(sp1), to_rad(sp2), to_rad(lat0), to_rad(lon0));
    let n = (phi1.sin() + phi2.sin()) / 2.0;
    let c = phi2.cos().powi(2) + 2.0*n*phi2.sin();
    let rho0 = c.sqrt() / n - phi0.sin() / n;
    let phi = to_rad(lat);
    let lam = to_rad(lon);
    let rho = (c - 2.0*n*phi.sin()).max(0.0).sqrt() / n;
    let theta = n * (lam - lam0);
    (rho * theta.sin(), rho0 - rho * theta.cos())
}

fn albers_usa_fwd(lon: f64, lat: f64) -> (f64, f64) {
    if lat > 49.5 && lon < -127.5 {
        let (ax, ay) = albers_conic_fwd(lon, lat, 55.0, 65.0, 50.0, -154.0);
        return (ax*0.35 - 2.0, ay*0.35 - 0.9);
    }
    if lat < 24.0 && lat > 18.0 && lon < -154.0 {
        let (hx, hy) = albers_conic_fwd(lon, lat, 8.0, 18.0, 13.0, -157.0);
        return (hx + 0.4, hy - 1.3);
    }
    albers_conic_fwd(lon, lat, 29.5, 45.5, 38.0, -96.0)
}


#[cfg(test)]
fn albers_usa_inv(x: f64, y: f64) -> (f64, f64) {
    // Approximate inverse using main continental conic.
    let (phi1, phi2, phi0, lam0) = (to_rad(29.5), to_rad(45.5), to_rad(38.0), to_rad(-96.0));
    let n = (phi1.sin() + phi2.sin()) / 2.0;
    let c = phi2.cos().powi(2) + 2.0*n*phi2.sin();
    let rho0 = c.sqrt() / n - phi0.sin() / n;
    let dy = rho0 - y;
    let rho = (x*x + dy*dy).sqrt();
    let phi = ((c - rho*rho*n*n) / (2.0*n)).asin();
    let lam = x.atan2(dy) / n + lam0;
    (to_deg(lam), to_deg(phi))
}


#[cfg(test)]
mod tests {
    use super::*;

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
}
