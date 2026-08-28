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

    /// The canonical name this variant round-trips through `from_name`.
    /// Inverse of `from_name` — kept as an explicit match (mirroring
    /// `colorous_gradient`'s style) rather than a `list()` index lookup, so
    /// adding a variant can't silently desync name ordering from data.
    /// `pub(crate)`: only consumed within this module (by
    /// `ContinuousScheme::to_sequential_wire_form`) today.
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::Viridis => "viridis",
            Self::Plasma => "plasma",
            Self::Magma => "magma",
            Self::Inferno => "inferno",
            Self::Cividis => "cividis",
            Self::Blues => "blues",
            Self::Reds => "reds",
            Self::Greens => "greens",
            Self::Oranges => "oranges",
            Self::Purples => "purples",
            Self::RdBu => "rdbu",
            Self::CoolBlue => "cool_blue",
            Self::WarmOchre => "warm_ochre",
            Self::BlueToRed => "blue_to_red",
            Self::NightBlue => "night_blue",
            Self::ElectricLime => "electric_lime",
            Self::CyanToAmber => "cyan_to_amber",
            Self::SignalBlue => "signal_blue",
            Self::EmberOrange => "ember_orange",
            Self::BlueToViolet => "blue_to_violet",
        }
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

    /// Resolve to this scheme's `ScaleSpec::Sequential` wire form
    /// (F-L04-02 second revision, spec §4.2 amended 2026-08-28; re-shaped in
    /// the spec reviewer's cycle-3 pass to carry real `t` positions — see
    /// `ScaleSpec::Sequential::stops`'s doc comment for why), unwrapping any
    /// `Reverse` wrapper.
    ///
    /// A `Named` scheme resolves to `Named { name, reverse }`, net-XORing the
    /// reverse flag through nested `Reverse`s so `Reverse(Reverse(Named(_)))`
    /// correctly resolves back to `reverse: false`.
    ///
    /// A `Gradient` scheme resolves to `Gradient { stops }` — its `(t,
    /// color)` pairs, colors normalized to hex, `t` positions carried as-is
    /// (not re-spaced). Any `Reverse` wrapper is composed directly into the
    /// stop positions (`t -> 1 - t`) rather than carried as a separate flag,
    /// so the render side never has to re-derive it: `ContinuousScheme::
    /// Gradient`'s stops are ascending by construction (`Gradient(...)`'s
    /// pyfunction now validates strictly-ascending `t`), and reversing an
    /// ascending sequence's *order* while mapping each `t -> 1 - t` yields an
    /// ascending sequence again — `.rev()` then map, no re-sort needed (the
    /// two order-reversals cancel). This keeps `ScaleSpec::Sequential.reverse`
    /// meaningful only for the `scheme`-name case; a `stops`-carrying spec is
    /// always emitted with `reverse: false`.
    pub(crate) fn to_sequential_wire_form(&self) -> SequentialWireForm {
        fn walk(scheme: &ContinuousScheme, reverse: bool) -> SequentialWireForm {
            match scheme {
                ContinuousScheme::Named(n) => {
                    SequentialWireForm::Named { name: n.name(), reverse }
                }
                ContinuousScheme::Reverse(inner) => walk(inner, !reverse),
                ContinuousScheme::Gradient(stops) => {
                    let hex_stops = |t: f64, c: Color| (t, crate::render::color::primitive::to_hex_string(c));
                    let stops: Vec<(f64, String)> = if reverse {
                        stops.iter().rev().map(|&(t, c)| hex_stops(1.0 - t, c)).collect()
                    } else {
                        stops.iter().map(|&(t, c)| hex_stops(t, c)).collect()
                    };
                    SequentialWireForm::Gradient { stops }
                }
            }
        }
        walk(self, false)
    }
}

/// The two ways a [`ContinuousScheme`] serializes onto `ScaleSpec::Sequential`
/// (see [`ContinuousScheme::to_sequential_wire_form`]).
pub(crate) enum SequentialWireForm {
    Named { name: &'static str, reverse: bool },
    Gradient { stops: Vec<(f64, String)> },
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
/// Construct directly with a colormap name, or obtain an instance via
/// ``ferrum.continuous_palette(name)`` (equivalent named-map lookup) or
/// ``ferrum.Gradient(stops)`` (custom gradient).
///
/// Methods
/// -------
/// from_name(name) : ContinuousScheme
///     Look up a built-in colormap by name (equivalent to the constructor).
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
/// ...     x="x", y="y", color=fm.Color("value", scale=scheme)
/// ... )
#[pyclass(name = "ContinuousScheme", module = "ferrum._core")]
#[derive(Debug, Clone)]
pub struct PyContinuousScheme(pub ContinuousScheme);

#[pymethods]
impl PyContinuousScheme {
    /// Construct from a built-in colormap name. Same validation as
    /// `from_name` — kept as the single lookup so the two never drift.
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        Self::from_name(name)
    }

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

    /// Emit this scheme's canonical `ScaleSpec` as a wire dict (SPEC-04
    /// bridge — mirrors `SequentialScale::_to_scale_spec_dict`).
    ///
    /// A named-colormap scheme (`fm.continuous_palette("viridis")`,
    /// `fm.ContinuousScheme("viridis")`) serializes to
    /// `{"type": "sequential", "scheme": "viridis", ...}`, identical to
    /// `fm.SequentialScale(scheme="viridis")` — the two render identically.
    ///
    /// A `Gradient`-backed scheme (F-L04-02 second revision, spec §4.2
    /// amended 2026-08-28 — supersedes the earlier refusal, whose "works via
    /// `Color(scheme=...)`" premise was verified false) serializes to
    /// `{"type": "sequential", "stops": [[t, hex], ...], ...}` instead:
    /// explicit `(t, color)` stop pairs — the real `t` positions, not
    /// re-spaced (spec reviewer cycle-3: a colors-only wire form silently
    /// discarded a documented, validated parameter) — with `Reverse`
    /// composed into the stop positions (see
    /// `ContinuousScheme::to_sequential_wire_form`). `Color(scale=...)` is
    /// the only route a `Gradient` renders through; `scheme=` stays
    /// name-string-only.
    fn _to_scale_spec_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let spec = match self.0.to_sequential_wire_form() {
            SequentialWireForm::Named { name, reverse } => {
                crate::spec::encoding::ScaleSpec::Sequential {
                    scheme: Some(name.to_string()),
                    domain: None,
                    reverse,
                    stops: None,
                }
            }
            SequentialWireForm::Gradient { stops } => {
                crate::spec::encoding::ScaleSpec::Sequential {
                    scheme: None,
                    domain: None,
                    reverse: false,
                    stops: Some(stops),
                }
            }
        };
        crate::scale::core::scale_spec_to_py_dict(py, spec)
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
///     At least 2 pairs of ``t`` in ``[0, 1]`` (strictly increasing) and CSS
///     color strings.  Each color may be an ``#rrggbb`` or ``#rrggbbaa`` hex
///     literal, or any of the 148 standard CSS named colors (e.g.
///     ``"steelblue"``, ``"tomato"``, ``"cornflowerblue"``).
///     Endpoints ``(0.0, ...)`` and ``(1.0, ...)`` should be present.
///
/// Returns
/// -------
/// ContinuousScheme
///     Scheme that interpolates linearly between adjacent stops.
///
/// Raises
/// ------
/// ValueError
///     If fewer than 2 stops are given, a ``t`` value is outside ``[0, 1]``
///     or non-finite, the ``t`` values are not strictly increasing, or a
///     color string fails to parse.
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
    // Spec reviewer cycle 3 (finding 2): fewer than 2 stops can't describe a
    // gradient, and the pre-fix version constructed one anyway (only the
    // color half of each pair was validated), silently rendering as if no
    // scale had been given at all — a Python-reachable silent-fallback path
    // this batch exists to close. Reject at the constructor, in the
    // function's existing "Gradient: {e}" error convention, so the render
    // side's own `len(stops) < 2` branch is genuinely unreachable except
    // from a hand-written wire spec that bypasses this constructor.
    if stops.len() < 2 {
        return Err(PyValueError::new_err(format!(
            "Gradient: need at least 2 stops, got {}",
            stops.len()
        )));
    }
    let ts: Vec<f64> = stops.iter().map(|(t, _)| *t).collect();
    // Reuses scale::core's shared finite/ascending vocabulary (ThresholdScale
    // and BinOrdinalScale's constructors, and the discretizing color
    // resolver, all raise the identical "must be strictly sorted ascending"
    // sentence for their own boundary lists) — a stop's `t` position is the
    // same "ordered breakpoints" shape, so this fits without adaptation.
    crate::scale::core::validate_finite("Gradient: t", &ts)?;
    if let Some(bad) = ts.iter().find(|t| !(0.0..=1.0).contains(*t)) {
        return Err(PyValueError::new_err(format!(
            "Gradient: t must be within [0, 1]; got {bad}"
        )));
    }
    if !crate::scale::core::is_strictly_ascending(&ts) {
        return Err(PyValueError::new_err(format!(
            "Gradient: {}",
            crate::scale::core::not_strictly_ascending_message("stops (by t)")
        )));
    }
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

    #[test]
    fn name_round_trips_through_from_name_for_every_listed_scheme() {
        for &name in NamedContinuous::list() {
            let variant = NamedContinuous::from_name(name).unwrap();
            assert_eq!(variant.name(), name, "name() did not round-trip for {name}");
        }
    }

    // --- F-L04-02: PyContinuousScheme constructible + serializable ---

    #[test]
    fn named_scheme_resolves_name_with_reverse_false() {
        let s = ContinuousScheme::Named(NamedContinuous::Viridis);
        match s.to_sequential_wire_form() {
            SequentialWireForm::Named { name, reverse } => {
                assert_eq!(name, "viridis");
                assert!(!reverse);
            }
            SequentialWireForm::Gradient { .. } => panic!("expected Named"),
        }
    }

    #[test]
    fn reversed_named_scheme_carries_reverse_true() {
        let s = ContinuousScheme::Reverse(Box::new(
            ContinuousScheme::Named(NamedContinuous::Viridis)));
        match s.to_sequential_wire_form() {
            SequentialWireForm::Named { name, reverse } => {
                assert_eq!(name, "viridis");
                assert!(reverse);
            }
            SequentialWireForm::Gradient { .. } => panic!("expected Named"),
        }
    }

    #[test]
    fn double_reversed_named_scheme_cancels_back_to_false() {
        let s = ContinuousScheme::Reverse(Box::new(ContinuousScheme::Reverse(Box::new(
            ContinuousScheme::Named(NamedContinuous::Plasma)))));
        match s.to_sequential_wire_form() {
            SequentialWireForm::Named { name, reverse } => {
                assert_eq!(name, "plasma");
                assert!(!reverse);
            }
            SequentialWireForm::Gradient { .. } => panic!("expected Named"),
        }
    }

    // --- F-L04-02 second revision: Gradient stops wire form ---

    #[test]
    fn gradient_scheme_resolves_stops_with_positions_in_order() {
        let red = from_rgba(255, 0, 0, 255);
        let blue = from_rgba(0, 0, 255, 255);
        let s = ContinuousScheme::Gradient(vec![(0.0, red), (1.0, blue)]);
        match s.to_sequential_wire_form() {
            SequentialWireForm::Gradient { stops } => {
                assert_eq!(
                    stops,
                    vec![(0.0, "#ff0000".to_string()), (1.0, "#0000ff".to_string())]
                );
            }
            SequentialWireForm::Named { .. } => panic!("expected Gradient"),
        }
    }

    /// Spec reviewer cycle-3 finding 1: a non-uniform `t` position must
    /// survive to the wire form, not get re-spaced to `i / (n - 1)`.
    #[test]
    fn gradient_scheme_preserves_non_uniform_t_positions() {
        let red = from_rgba(255, 0, 0, 255);
        let green = from_rgba(0, 255, 0, 255);
        let blue = from_rgba(0, 0, 255, 255);
        let s = ContinuousScheme::Gradient(vec![(0.0, red), (0.9, green), (1.0, blue)]);
        match s.to_sequential_wire_form() {
            SequentialWireForm::Gradient { stops } => {
                assert_eq!(
                    stops,
                    vec![
                        (0.0, "#ff0000".to_string()),
                        (0.9, "#00ff00".to_string()),
                        (1.0, "#0000ff".to_string()),
                    ]
                );
            }
            SequentialWireForm::Named { .. } => panic!("expected Gradient"),
        }
    }

    #[test]
    fn reversed_gradient_scheme_composes_t_to_one_minus_t() {
        let red = from_rgba(255, 0, 0, 255);
        let green = from_rgba(0, 255, 0, 255);
        let blue = from_rgba(0, 0, 255, 255);
        let gradient = ContinuousScheme::Gradient(vec![(0.0, red), (0.9, green), (1.0, blue)]);
        let s = ContinuousScheme::Reverse(Box::new(gradient));
        match s.to_sequential_wire_form() {
            SequentialWireForm::Gradient { stops } => {
                // t -> 1 - t for every stop, re-ordered ascending: (1.0, red)
                // becomes (0.0, red), (0.9, green) -> (0.1, green), (0.0, blue)
                // -> (1.0, blue). Computed via subtraction (not a literal) to
                // avoid a hand-typed float mismatch on the exact bit pattern
                // `1.0 - 0.9` rounds to.
                assert_eq!(
                    stops,
                    vec![
                        (1.0 - 1.0, "#0000ff".to_string()),
                        (1.0 - 0.9, "#00ff00".to_string()),
                        (1.0 - 0.0, "#ff0000".to_string()),
                    ]
                );
            }
            SequentialWireForm::Named { .. } => panic!("expected Gradient"),
        }
    }

    #[test]
    fn double_reversed_gradient_scheme_cancels_back_to_forward_order() {
        let red = from_rgba(255, 0, 0, 255);
        let blue = from_rgba(0, 0, 255, 255);
        let gradient = ContinuousScheme::Gradient(vec![(0.0, red), (1.0, blue)]);
        let s = ContinuousScheme::Reverse(Box::new(ContinuousScheme::Reverse(Box::new(gradient))));
        match s.to_sequential_wire_form() {
            SequentialWireForm::Gradient { stops } => {
                assert_eq!(
                    stops,
                    vec![(0.0, "#ff0000".to_string()), (1.0, "#0000ff".to_string())]
                );
            }
            SequentialWireForm::Named { .. } => panic!("expected Gradient"),
        }
    }

    #[test]
    fn py_continuous_scheme_new_matches_from_name() {
        // #[new] must apply the same validation/lookup as `from_name` — both
        // are the constructor path `Color(scale=fm.ContinuousScheme("viridis"))`
        // and `Color(scale=fm.continuous_palette("viridis"))` rely on.
        let via_new = PyContinuousScheme::new("viridis").unwrap();
        let via_from_name = PyContinuousScheme::from_name("viridis").unwrap();
        assert_eq!(via_new.0, via_from_name.0);
    }

    #[test]
    fn py_continuous_scheme_new_rejects_unknown_name() {
        // `PyErr::to_string()` needs an initialized interpreter (fetches the
        // traceback), which this crate's plain `cargo test` run doesn't set
        // up (no other test in the crate calls `Python::with_gil`) — assert
        // on the `Err` shape instead of the formatted message.
        assert!(PyContinuousScheme::new("not-a-real-colormap").is_err());
    }

    // --- Spec reviewer cycle 3, finding 2: Gradient(...) degenerate-input rejection ---

    #[test]
    fn gradient_rejects_zero_stops() {
        assert!(Gradient(vec![]).is_err());
    }

    #[test]
    fn gradient_rejects_one_stop() {
        assert!(Gradient(vec![(0.0, "red".to_string())]).is_err());
    }

    #[test]
    fn gradient_rejects_t_below_zero() {
        assert!(Gradient(vec![(-0.1, "red".to_string()), (1.0, "blue".to_string())]).is_err());
    }

    #[test]
    fn gradient_rejects_t_above_one() {
        assert!(Gradient(vec![(0.0, "red".to_string()), (1.1, "blue".to_string())]).is_err());
    }

    #[test]
    fn gradient_rejects_non_finite_t() {
        assert!(Gradient(vec![(f64::NAN, "red".to_string()), (1.0, "blue".to_string())]).is_err());
    }

    #[test]
    fn gradient_rejects_descending_t() {
        assert!(Gradient(vec![(0.5, "red".to_string()), (0.2, "blue".to_string())]).is_err());
    }

    #[test]
    fn gradient_rejects_duplicate_t() {
        // Strictly ascending, not merely non-decreasing — equal t values are
        // rejected too, matching ThresholdScale/BinOrdinalScale's own
        // strictly-ascending contract.
        assert!(Gradient(vec![(0.5, "red".to_string()), (0.5, "blue".to_string())]).is_err());
    }

    #[test]
    fn gradient_accepts_valid_ascending_stops() {
        assert!(Gradient(vec![
            (0.0, "red".to_string()),
            (0.9, "green".to_string()),
            (1.0, "blue".to_string()),
        ])
        .is_ok());
    }
}
