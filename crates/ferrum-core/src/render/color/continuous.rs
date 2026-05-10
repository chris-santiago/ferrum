//! Continuous colormaps for raster/hex/bivariate-density marks.
//! Backed by `colorous` for the 5 named maps; supports user Gradient and Reverse.

use crate::render::color::categorical::{from_rgba, Color};

#[derive(Debug, Clone, PartialEq)]
pub enum NamedContinuous {
    Viridis,
    Plasma,
    Magma,
    Inferno,
    Cividis,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ContinuousScheme {
    Named(NamedContinuous),
    Gradient(Vec<(f64, Color)>),
    Reverse(Box<ContinuousScheme>),
}

impl NamedContinuous {
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "viridis" => Some(Self::Viridis),
            "plasma"  => Some(Self::Plasma),
            "magma"   => Some(Self::Magma),
            "inferno" => Some(Self::Inferno),
            "cividis" => Some(Self::Cividis),
            _ => None,
        }
    }

    pub fn list() -> &'static [&'static str] {
        &["viridis", "plasma", "magma", "inferno", "cividis"]
    }

    fn colorous_gradient(&self) -> colorous::Gradient {
        match self {
            Self::Viridis => colorous::VIRIDIS,
            Self::Plasma  => colorous::PLASMA,
            Self::Magma   => colorous::MAGMA,
            Self::Inferno => colorous::INFERNO,
            Self::Cividis => colorous::CIVIDIS,
        }
    }
}

impl ContinuousScheme {
    /// Sample at t ∈ [0, 1]. t outside [0, 1] is clamped.
    pub fn sample(&self, t: f64) -> Color {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Named(n) => {
                let c = n.colorous_gradient().eval_continuous(t);
                from_rgba(c.r, c.g, c.b, 255)
            }
            Self::Gradient(stops) => sample_gradient(stops, t),
            Self::Reverse(inner) => inner.sample(1.0 - t),
        }
    }
}

fn sample_gradient(stops: &[(f64, Color)], t: f64) -> Color {
    if stops.is_empty() {
        return from_rgba(0, 0, 0, 255);
    }
    if t <= stops[0].0 { return stops[0].1; }
    if t >= stops[stops.len() - 1].0 { return stops[stops.len() - 1].1; }
    // binary search for bracketing pair
    let i = stops.partition_point(|(p, _)| *p <= t);
    let (t0, c0) = stops[i - 1];
    let (t1, c1) = stops[i];
    let u = (t - t0) / (t1 - t0);
    from_rgba(
        lerp_u8(c0.red, c1.red, u),
        lerp_u8(c0.green, c1.green, u),
        lerp_u8(c0.blue, c1.blue, u),
        lerp_u8(c0.alpha, c1.alpha, u),
    )
}

fn lerp_u8(a: u8, b: u8, t: f64) -> u8 {
    (a as f64 + (b as f64 - a as f64) * t).round().clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viridis_endpoints_match_colorous_reference() {
        let s = ContinuousScheme::Named(NamedContinuous::Viridis);
        let c0 = s.sample(0.0);
        let c1 = s.sample(1.0);
        // Viridis 0.0 → ~RGB(68, 1, 84); 1.0 → ~RGB(253, 231, 37)
        assert!(c0.red < 80 && c0.green < 20 && c0.blue < 100, "viridis(0): {:?}", c0);
        assert!(c1.red > 240 && c1.green > 220 && c1.blue < 50, "viridis(1): {:?}", c1);
    }

    #[test]
    fn gradient_two_stop_interpolates_in_linear_space() {
        let red  = from_rgba(255, 0, 0, 255);
        let blue = from_rgba(0, 0, 255, 255);
        let s = ContinuousScheme::Gradient(vec![(0.0, red), (1.0, blue)]);
        let mid = s.sample(0.5);
        assert_eq!(mid.red, 128);
        assert_eq!(mid.blue, 128);
        assert_eq!(mid.green, 0);
    }

    #[test]
    fn reverse_inverts_t() {
        let s = ContinuousScheme::Reverse(Box::new(
            ContinuousScheme::Named(NamedContinuous::Viridis)));
        let c0 = s.sample(0.0);
        let c1 = s.sample(1.0);
        // Reversed viridis: 0.0 should look like normal viridis(1.0)
        assert!(c0.red > 240 && c0.green > 220);
        assert!(c1.red < 80);
    }
}
