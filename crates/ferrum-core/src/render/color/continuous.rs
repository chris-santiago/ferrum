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

// --- Phase 8b Task 37: PyO3 bindings for ContinuousScheme + Gradient ---

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

/// Continuous color scheme for quantitative color encodings.
///
/// Wraps a named colormap or a user-defined gradient and exposes it as a
/// Python value that can be passed to ``Color(scale=...)`` to control how
/// numeric values are mapped to colors.
///
/// Named built-in colormaps: ``"viridis"``, ``"plasma"``, ``"magma"``,
/// ``"inferno"``, ``"cividis"``.
///
/// Do not call the constructor directly. Obtain an instance via
/// ``ferrum.continuous_palette(name)`` (named map) or
/// ``ferrum.Gradient(stops)`` (custom gradient).
///
/// Methods
/// -------
/// from_name(name) : ContinuousScheme
///     Look up a built-in colormap by name.
/// reversed() : ContinuousScheme
///     Return a new scheme that samples the inverse of this one (1 - t).
///
/// See Also
/// --------
/// ferrum.continuous_palette : Named colormap lookup.
/// ferrum.Gradient : Custom gradient construction.
#[pyclass(name = "ContinuousScheme", module = "ferrum._core")]
#[derive(Debug, Clone)]
pub struct PyContinuousScheme(pub ContinuousScheme);

#[pymethods]
impl PyContinuousScheme {
    /// Look up a built-in continuous colormap by name. Accepted names:
    /// "viridis", "plasma", "magma", "inferno", "cividis".
    #[staticmethod]
    fn from_name(name: &str) -> PyResult<Self> {
        NamedContinuous::from_name(name)
            .map(|n| Self(ContinuousScheme::Named(n)))
            .ok_or_else(|| PyValueError::new_err(format!(
                "Unknown colormap: '{name}'; available: viridis, plasma, magma, inferno, cividis"
            )))
    }

    /// Return a new scheme that samples the inverse of this scheme (1 - t).
    fn reversed(&self) -> Self {
        Self(ContinuousScheme::Reverse(Box::new(self.0.clone())))
    }

    fn __repr__(&self) -> String {
        format!("ContinuousScheme({:?})", self.0)
    }
}

/// Build a continuous color scheme from explicit color stops.
///
/// Returns a ``ContinuousScheme`` that interpolates linearly in RGB
/// between adjacent ``(t, color)`` pairs. Pass the result to
/// ``Color(scale=...)`` to use a custom gradient for a color encoding.
///
/// Parameters
/// ----------
/// stops : list[tuple[float, str]]
///     Pairs of ``t`` in ``[0, 1]`` and CSS color strings.  Each color may
///     be an ``#rrggbb`` or ``#rrggbbaa`` hex literal, or one of the common
///     named colors: ``"red"``, ``"green"``, ``"blue"``, ``"white"``,
///     ``"black"``, ``"yellow"``, ``"magenta"``, ``"cyan"``,
///     ``"gray"`` / ``"grey"``.
///     Endpoints ``(0.0, ...)`` and ``(1.0, ...)`` should be present.
///
/// Returns
/// -------
/// ContinuousScheme
///     Scheme that interpolates linearly between adjacent stops.
///
/// Examples
/// --------
/// ::
///
///     import ferrum as fr
///     scheme = fr.Gradient([(0.0, "#ffffff"), (0.5, "#888888"), (1.0, "#000000")])
///     chart = fr.Chart(df).encode(color=fr.Color("density", scale=scheme))
#[pyfunction]
#[allow(non_snake_case)]
pub fn Gradient(stops: Vec<(f64, String)>) -> PyResult<PyContinuousScheme> {
    let mut color_stops = Vec::with_capacity(stops.len());
    for (t, name) in stops {
        let color = parse_color_string(&name)
            .map_err(|e| PyValueError::new_err(format!("Gradient: {e}")))?;
        color_stops.push((t, color));
    }
    Ok(PyContinuousScheme(ContinuousScheme::Gradient(color_stops)))
}

/// Parse a color string. Accepts `#rrggbb` / `#rrggbbaa` (delegated to
/// `categorical::from_hex_str`) and a small set of common named colors.
fn parse_color_string(s: &str) -> Result<Color, String> {
    let trimmed = s.trim();
    if trimmed.starts_with('#') {
        return crate::render::color::categorical::from_hex_str(trimmed)
            .map_err(|e| format!("{e}"));
    }
    let named: Option<(u8, u8, u8)> = match trimmed.to_ascii_lowercase().as_str() {
        "red"     => Some((255,   0,   0)),
        "green"   => Some((  0, 128,   0)),
        "blue"    => Some((  0,   0, 255)),
        "white"   => Some((255, 255, 255)),
        "black"   => Some((  0,   0,   0)),
        "yellow"  => Some((255, 255,   0)),
        "magenta" => Some((255,   0, 255)),
        "cyan"    => Some((  0, 255, 255)),
        "gray" | "grey" => Some((128, 128, 128)),
        _ => None,
    };
    if let Some((r, g, b)) = named {
        return Ok(from_rgba(r, g, b, 255));
    }
    Err(format!("unrecognized color: '{s}' (use a named color or #rrggbb / #rrggbbaa)"))
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
