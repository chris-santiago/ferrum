//! Continuous colormaps for raster/hex/bivariate-density marks.
//! Backed by `colorous` for the 5 named maps; supports user Gradient and Reverse.

use crate::render::color::primitive::{from_rgba, Color};

#[derive(Debug, Clone, PartialEq)]
pub enum NamedContinuous {
    Viridis,
    Plasma,
    Magma,
    Inferno,
    Cividis,
    Blues,
    Reds,
    Greens,
    Oranges,
    Purples,
    RdBu,
    // Paper Ink family
    CoolBlue,
    WarmOchre,
    BlueToRed,
    // Slate Citrus family
    NightBlue,
    ElectricLime,
    CyanToAmber,
    // Arctic Signal family
    SignalBlue,
    EmberOrange,
    BlueToViolet,
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
            "viridis"        => Some(Self::Viridis),
            "plasma"         => Some(Self::Plasma),
            "magma"          => Some(Self::Magma),
            "inferno"        => Some(Self::Inferno),
            "cividis"        => Some(Self::Cividis),
            "blues"          => Some(Self::Blues),
            "reds"           => Some(Self::Reds),
            "greens"         => Some(Self::Greens),
            "oranges"        => Some(Self::Oranges),
            "purples"        => Some(Self::Purples),
            "rdbu"           => Some(Self::RdBu),
            "cool_blue"      => Some(Self::CoolBlue),
            "warm_ochre"     => Some(Self::WarmOchre),
            "blue_to_red"    => Some(Self::BlueToRed),
            "night_blue"     => Some(Self::NightBlue),
            "electric_lime"  => Some(Self::ElectricLime),
            "cyan_to_amber"  => Some(Self::CyanToAmber),
            "signal_blue"    => Some(Self::SignalBlue),
            "ember_orange"   => Some(Self::EmberOrange),
            "blue_to_violet" => Some(Self::BlueToViolet),
            _ => None,
        }
    }

    pub fn list() -> &'static [&'static str] {
        &[
            "viridis", "plasma", "magma", "inferno", "cividis",
            "blues", "reds", "greens", "oranges", "purples",
            "rdbu",
            "cool_blue", "warm_ochre", "blue_to_red",
            "night_blue", "electric_lime", "cyan_to_amber",
            "signal_blue", "ember_orange", "blue_to_violet",
        ]
    }

    fn colorous_gradient(&self) -> Option<colorous::Gradient> {
        match self {
            Self::Viridis => Some(colorous::VIRIDIS),
            Self::Plasma  => Some(colorous::PLASMA),
            Self::Magma   => Some(colorous::MAGMA),
            Self::Inferno => Some(colorous::INFERNO),
            Self::Cividis => Some(colorous::CIVIDIS),
            Self::Blues   => Some(colorous::BLUES),
            Self::Reds    => Some(colorous::REDS),
            Self::Greens  => Some(colorous::GREENS),
            Self::Oranges => Some(colorous::ORANGES),
            Self::Purples => Some(colorous::PURPLES),
            Self::RdBu    => Some(colorous::RED_BLUE),
            _ => None,
        }
    }

    pub fn sample(&self, t: f64) -> Color {
        if let Some(g) = self.colorous_gradient() {
            let c = g.eval_continuous(t);
            from_rgba(c.r, c.g, c.b, 255)
        } else {
            sample_gradient(&self.custom_stops(), t)
        }
    }

    fn custom_stops(&self) -> Vec<(f64, Color)> {
        let hexes: &[u32] = match self {
            Self::CoolBlue     => &[0xEFF6FF, 0xDBEAFE, 0x93C5FD, 0x60A5FA, 0x2563EB, 0x1D4ED8, 0x1E3A8A],
            Self::WarmOchre    => &[0xFFF7E6, 0xFDECC8, 0xF8D88A, 0xD4A017, 0xB45309, 0x92400E, 0x78350F],
            Self::BlueToRed    => &[0x1E3A8A, 0x60A5FA, 0xDBEAFE, 0xFAF7F2, 0xFDE68A, 0xDC2626, 0x7F1D1D],
            Self::NightBlue    => &[0x1E293B, 0x1D4ED8, 0x2563EB, 0x60A5FA, 0x93C5FD, 0xBFDBFE, 0xE0F2FE],
            Self::ElectricLime => &[0x365314, 0x4D7C0F, 0x65A30D, 0x84CC16, 0xA3E635, 0xBEF264, 0xD9F99D],
            // Midpoint 0x111827 matches Slate Citrus bg — zero values recede on the dark canvas.
            Self::CyanToAmber  => &[0x155E75, 0x0891B2, 0x67E8F9, 0x111827, 0xFDE68A, 0xF59E0B, 0xB45309],
            Self::SignalBlue   => &[0xF0F9FF, 0xE0F2FE, 0x7DD3FC, 0x38BDF8, 0x0284C7, 0x0369A1, 0x0C4A6E],
            Self::EmberOrange  => &[0xFFF7ED, 0xFED7AA, 0xFDBA74, 0xEA580C, 0xC2410C, 0x9A3412, 0x7C2D12],
            Self::BlueToViolet => &[0x0C4A6E, 0x38BDF8, 0xBAE6FD, 0xF8FAFC, 0xE9D5FF, 0xA78BFA, 0x6D28D9],
            _ => unreachable!(),
        };
        let step = 1.0 / (hexes.len() - 1) as f64;
        hexes.iter().enumerate().map(|(i, &h)| {
            let r = ((h >> 16) & 0xFF) as u8;
            let g = ((h >> 8) & 0xFF) as u8;
            let b = (h & 0xFF) as u8;
            (i as f64 * step, from_rgba(r, g, b, 255))
        }).collect()
    }
}

impl ContinuousScheme {
    /// Sample at t ∈ [0, 1]. t outside [0, 1] is clamped.
    pub fn sample(&self, t: f64) -> Color {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Named(n) => n.sample(t),
            Self::Gradient(stops) => sample_gradient(stops, t),
            Self::Reverse(inner) => inner.sample(1.0 - t),
        }
    }
}

fn sample_gradient(stops: &[(f64, Color)], t: f64) -> Color {
    if stops.is_empty() || !t.is_finite() {
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
/// ``"inferno"``, ``"cividis"``, ``"blues"``, ``"rdbu"``,
/// ``"cool_blue"``, ``"warm_ochre"``, ``"blue_to_red"``,
/// ``"night_blue"``, ``"electric_lime"``, ``"cyan_to_amber"``,
/// ``"signal_blue"``, ``"ember_orange"``, ``"blue_to_violet"``.
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
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> scheme = fm.ContinuousScheme("viridis")
/// >>> fm.Chart(df).mark_point().encode(
/// ...     x="x", y="y", color=fm.Color("value", scheme=scheme)
/// ... )
#[pyclass(name = "ContinuousScheme", module = "ferrum._core")]
#[derive(Debug, Clone)]
pub struct PyContinuousScheme(pub ContinuousScheme);

#[pymethods]
impl PyContinuousScheme {
    #[staticmethod]
    fn from_name(name: &str) -> PyResult<Self> {
        NamedContinuous::from_name(name)
            .map(|n| Self(ContinuousScheme::Named(n)))
            .ok_or_else(|| PyValueError::new_err(format!(
                "Unknown colormap: '{name}'; available: {}",
                NamedContinuous::list().join(", ")
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
///     be an ``#rrggbb`` or ``#rrggbbaa`` hex literal, or any of the 148
///     standard CSS named colors (e.g. ``"steelblue"``, ``"tomato"``,
///     ``"cornflowerblue"``).
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
        let color = crate::render::color::parse_color(&name)
            .map_err(|e| PyValueError::new_err(format!("Gradient: {e}")))?;
        color_stops.push((t, color));
    }
    Ok(PyContinuousScheme(ContinuousScheme::Gradient(color_stops)))
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

    #[test]
    fn cool_blue_endpoints() {
        let s = ContinuousScheme::Named(NamedContinuous::CoolBlue);
        let c0 = s.sample(0.0); // #EFF6FF
        let c1 = s.sample(1.0); // #1E3A8A
        assert!(c0.red > 0xE0 && c0.blue == 0xFF, "cool_blue(0): {:?}", c0);
        assert!(c1.red == 0x1E && c1.green == 0x3A && c1.blue == 0x8A, "cool_blue(1): {:?}", c1);
    }

    #[test]
    fn blue_to_red_midpoint_is_paper_ink_bg() {
        let s = ContinuousScheme::Named(NamedContinuous::BlueToRed);
        let mid = s.sample(0.5); // #FAF7F2
        assert_eq!(mid.red, 0xFA);
        assert_eq!(mid.green, 0xF7);
        assert_eq!(mid.blue, 0xF2);
    }

    #[test]
    fn all_custom_schemes_resolve_via_from_name() {
        for name in &[
            "cool_blue", "warm_ochre", "blue_to_red",
            "night_blue", "electric_lime", "cyan_to_amber",
            "signal_blue", "ember_orange", "blue_to_violet",
        ] {
            assert!(NamedContinuous::from_name(name).is_some(), "{name} not found");
        }
    }

    #[test]
    fn reds_endpoints_match_colorbrewer() {
        let s = ContinuousScheme::Named(NamedContinuous::Reds);
        let c0 = s.sample(0.0); // near-white ~#fff5f0: R≈255, G≈245, B≈240
        let c1 = s.sample(1.0); // dark crimson ~#67000d: R≈103, G≈0, B≈13
        assert!(c0.red > 220 && c0.green > 200 && c0.blue > 200, "reds(0): {:?}", c0);
        assert!(c1.red > 80 && c1.green < 30 && c1.blue < 40, "reds(1): {:?}", c1);
    }

    #[test]
    fn greens_endpoints_match_colorbrewer() {
        let s = ContinuousScheme::Named(NamedContinuous::Greens);
        let c0 = s.sample(0.0); // near-white ~#f7fcf5: R≈247, G≈252, B≈245
        let c1 = s.sample(1.0); // dark forest ~#00441b: R≈0, G≈68, B≈27
        assert!(c0.red > 220 && c0.green > 220 && c0.blue > 220, "greens(0): {:?}", c0);
        assert!(c1.red < 30 && c1.green > 50 && c1.blue < 40, "greens(1): {:?}", c1);
    }

    #[test]
    fn oranges_endpoints_match_colorbrewer() {
        let s = ContinuousScheme::Named(NamedContinuous::Oranges);
        let c0 = s.sample(0.0); // near-white ~#fff5eb: R≈255, G≈245, B≈235
        let c1 = s.sample(1.0); // dark orange-brown ~#7f2704: R≈127, G≈39, B≈4
        assert!(c0.red > 220 && c0.green > 200 && c0.blue > 180, "oranges(0): {:?}", c0);
        assert!(c1.red > 100 && c1.green < 60 && c1.blue < 20, "oranges(1): {:?}", c1);
    }

    #[test]
    fn purples_endpoints_match_colorbrewer() {
        let s = ContinuousScheme::Named(NamedContinuous::Purples);
        let c0 = s.sample(0.0); // near-white ~#fcfbfd: R≈252, G≈251, B≈253
        let c1 = s.sample(1.0); // dark purple ~#3f007d: R≈63, G≈0, B≈125
        assert!(c0.red > 220 && c0.green > 220 && c0.blue > 220, "purples(0): {:?}", c0);
        assert!(c1.red < 80 && c1.green < 20 && c1.blue > 80, "purples(1): {:?}", c1);
    }

    #[test]
    fn new_schemes_resolve_via_from_name() {
        for name in &["reds", "greens", "oranges", "purples"] {
            assert!(NamedContinuous::from_name(name).is_some(), "{name} not found");
        }
    }
}
