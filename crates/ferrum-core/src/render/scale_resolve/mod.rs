//! Build ResolvedScales from a ChartSpec + a post-transform RecordBatch.
//! Phase 7 supports: LinearScale, OrdinalScale, TimeScale on x/y;
//! CategoricalColorScale on color.
//! Phase 8a adds: LogScale, SymlogScale (via explicit ScaleSpec override);
//! SizeScale, ShapeScale, OpacityScale for new encoding channels.

mod auxiliary;
mod color;
mod domain;
mod positional;
mod seam;
#[cfg(test)]
mod tests;

use std::borrow::Cow;
use std::collections::HashMap;

use arrow::datatypes::DataType as ArrowDataType;
use arrow::record_batch::RecordBatch;

use crate::layout::ThemeInputs;
use crate::scale::linear::LinearScale;
use crate::scale::log::LogScale;
use crate::scale::ordinal::OrdinalScale;
use crate::scale::pow::PowScale;
use crate::scale::symlog::SymlogScale;
use crate::scale::time::TimeScale;
use crate::spec::chart::ChartSpec;
use crate::spec::encoding::DataType as SpecDataType;

use super::color::Color;
use super::RenderError;

// ── Re-exports: sub-module functions ────────────────────────────────────────

pub use self::auxiliary::{build_opacity_scale, build_shape_scale, build_size_scale};
pub use self::color::build_color_scale;

// Domain-union helpers re-exported for the composite resolve pass
// (`render::composite`), which unions per-channel domains across a composite
// tree's leaves through the same facet mechanism (`domain` is a private
// submodule, so the pub(in crate::render) fns need a reachable path here).
pub(in crate::render) use self::domain::{
    distinct_positional_categories_shared, locate_field, numeric_domain_union,
};

// Seam vocabulary shared with `render::composite`'s resolve pass — see
// `seam.rs` for why these live here rather than in `composite` (dependency
// must point one way: composite → scale_resolve).
pub(in crate::render) use self::seam::{Channel, LeafScaleContext, SharedDomain};

// Internal re-exports used by the orchestrator in this module.
use self::positional::{
    apply_coord_domain_overrides, build_axis_scale, PositionalFields,
};

// ── Types ───────────────────────────────────────────────────────────────────

/// Sealed-enum wrapper over Phase 4 scales, used during render.
/// Phase 7: Linear/Ordinal/Time. Phase 8a adds: Log, Symlog.
/// Phase 12: Pow (power/sqrt transform).
#[derive(Debug, Clone)]
pub enum ScaleKind {
    Linear(LinearScale),
    Ordinal(OrdinalScale),
    Time(TimeScale),
    Log(LogScale),
    Symlog(SymlogScale),
    Pow(PowScale),
}

/// Policy for non-finite fraction projections in
/// [`ScaleKind::project_values_to_fractions`]. Makes the labeled-vs-unlabeled
/// distinction explicit at each call site rather than implicit in two parallel
/// projection loops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NonFinite {
    /// Drop each non-finite projection individually. For unlabeled minor ticks,
    /// where dropping one entry cannot misalign anything.
    DropOne,
    /// If any projection is non-finite, return an empty vec. For labeled major /
    /// explicit ticks that must stay index-aligned with `tick_labels`.
    RejectAll,
}

/// Dispatch a method call to all `ScaleKind` variants.
macro_rules! dispatch_all {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            ScaleKind::Linear(s) => s.$method($($arg),*),
            ScaleKind::Ordinal(s) => s.$method($($arg),*),
            ScaleKind::Time(s) => s.$method($($arg),*),
            ScaleKind::Log(s) => s.$method($($arg),*),
            ScaleKind::Symlog(s) => s.$method($($arg),*),
            ScaleKind::Pow(s) => s.$method($($arg),*),
        }
    };
}

/// Dispatch a method call to the continuous `ScaleKind` variants
/// (Linear, Time, Log, Symlog, Pow). Ordinal is excluded — callers must
/// handle it separately.
macro_rules! dispatch_continuous {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            ScaleKind::Linear(s) => s.$method($($arg),*),
            ScaleKind::Time(s) => s.$method($($arg),*),
            ScaleKind::Log(s) => s.$method($($arg),*),
            ScaleKind::Symlog(s) => s.$method($($arg),*),
            ScaleKind::Pow(s) => s.$method($($arg),*),
            ScaleKind::Ordinal(_) => unreachable!(),
        }
    };
}

impl ScaleKind {
    /// Map a quantitative or temporal value to a pixel coordinate.
    /// Returns `None` for ordinal scales (use `to_pixel_str` instead) and for
    /// inputs that fall outside the scale's domain (Phase 9c — position
    /// adjustments such as Jitter can push values past the original domain;
    /// the underlying scale returns `NaN` rather than `None` in that case).
    pub fn to_pixel_f64(&self, x: f64) -> Option<f64> {
        if matches!(self, Self::Ordinal(_)) { return None; }
        let p = dispatch_continuous!(self, scale_internal, x);
        if p.is_finite() { Some(p) } else { None }
    }

    /// Map an ordinal/string value to a pixel band center.
    /// Returns `None` for non-ordinal scales or unknown categories.
    pub fn to_pixel_str(&self, value: &str) -> Option<f64> {
        match self {
            Self::Ordinal(s) => s.scale_internal(value),
            _ => None,
        }
    }

    /// Generate tick values as displayable strings.
    pub fn tick_labels(&self, count_hint: usize) -> Vec<String> {
        match self {
            Self::Linear(s) => s
                .ticks_internal(count_hint)
                .into_iter()
                .map(super::format::format_numeric)
                .collect(),
            Self::Ordinal(s) => s
                .ticks_internal()
                .into_iter()
                .map(|v| super::format::format_ordinal(&v))
                .collect(),
            Self::Time(s) => {
                let ticks = s.ticks_internal(count_hint);
                let spacing = if ticks.len() >= 2 {
                    (ticks[1] - ticks[0]) as i64
                } else {
                    86_400_000
                };
                ticks
                    .into_iter()
                    .map(|t| super::format::format_time(t as i64, spacing))
                    .collect()
            }
            Self::Log(s) => s
                .ticks_internal(count_hint)
                .into_iter()
                .map(super::format::format_numeric)
                .collect(),
            Self::Symlog(s) => s
                .ticks_internal(count_hint)
                .into_iter()
                .map(super::format::format_numeric)
                .collect(),
            Self::Pow(s) => s
                .ticks_internal(count_hint)
                .into_iter()
                .map(super::format::format_numeric)
                .collect(),
        }
    }

    /// Raw tick *values* (epoch-milliseconds) for a temporal scale, in the same
    /// order and count as [`tick_labels`](Self::tick_labels). Returns `None` for
    /// non-temporal scales. Used by `prepare.rs` to apply an explicit
    /// `chrono`/d3 time format to the underlying timestamps (the formatted
    /// strings from `tick_labels` have already lost the epoch values).
    pub(crate) fn temporal_tick_values(&self, count_hint: usize) -> Option<Vec<i64>> {
        match self {
            Self::Time(s) => Some(
                s.ticks_internal(count_hint)
                    .into_iter()
                    .map(|v| v as i64)
                    .collect(),
            ),
            _ => None,
        }
    }

    /// Grid item 18: project this scale's minor ticks to **domain fractions**
    /// `t ∈ [0, 1]` over the scale's resolved range — the *same* projection
    /// majors use (`project_values_to_fractions`), so a minor at domain value
    /// `v` lands at the identical pixel the major projection of `v` would give.
    ///
    /// Layout then maps each fraction onto the panel's mark range via the same
    /// `inset_pixel_range` padding inset that places majors and data marks (see
    /// [`crate::layout::axis::build_minor_ticks`]) — **not** the naive
    /// `origin + frac * extent`. This keeps minor gridlines aligned with both
    /// the projected major gridlines and the data marks.
    ///
    /// Unlike the all-or-nothing major carrier, a single out-of-domain minor is
    /// dropped individually: minors carry no label, so dropping one cannot
    /// misalign anything. Ordinal (and any non-continuous) scales return an
    /// empty vec, matching the engine's empty `minor_ticks_internal()`.
    pub(crate) fn minor_tick_fractions(&self) -> Vec<f64> {
        if matches!(self, Self::Ordinal(_)) {
            return Vec::new();
        }
        let minors = dispatch_continuous!(self, minor_ticks_internal);
        let positions: Vec<f64> = minors.into_iter().map(|t| t.position).collect();
        // Minors carry no label, so a single out-of-domain minor can be dropped
        // individually without misaligning anything (`DropOne`). Majors must stay
        // index-aligned with `tick_labels`, so they use `RejectAll`.
        self.project_values_to_fractions(&positions, NonFinite::DropOne)
    }

    /// Return the data-space domain `(lo, hi)` for continuous scales.
    /// Returns `None` for ordinal scales (no numeric domain).
    pub(crate) fn data_domain(&self) -> Option<(f64, f64)> {
        match self {
            ScaleKind::Ordinal(_) => None,
            _ => {
                let [lo, hi] = dispatch_continuous!(self, domain_pair);
                Some((lo, hi))
            }
        }
    }

    /// Pixel-range used when constructing this scale (lo, hi).
    pub fn pixel_range(&self) -> (f64, f64) {
        let r = dispatch_all!(self, range_pair);
        (r[0], r[1])
    }

    /// Signed band-pixel extent (`r1 − r0`, in range order) of an ordinal
    /// positional scale's range, but only when the resolver recorded that
    /// range as **explicitly supplied** by the user (`BandScale`/`PointScale`/
    /// positional `OrdinalScale` `range=`) rather than the panel-extent
    /// fallback. `None` for every other scale kind, and `None` for an ordinal
    /// scale using the fallback range — even though that range is numerically
    /// a valid extent, explicitness is a fact recorded at construction, never
    /// inferred by comparing floats (band-geometry unification design §6).
    ///
    /// Consumer contract: `scale.explicit_band_extent().map(f64::abs).unwrap_or(panel_extent)`.
    pub(in crate::render) fn explicit_band_extent(&self) -> Option<f64> {
        match self {
            ScaleKind::Ordinal(s) => s.explicit_band_extent(),
            _ => None,
        }
    }

    /// Absolute band-center pixels, one per category in `tick_labels` (domain)
    /// order, for an ordinal positional scale whose pixel range was **explicitly
    /// supplied** by the user (`BandScale`/`PointScale`/positional `OrdinalScale`
    /// `range=`); `None` for every other scale kind and for the panel-extent
    /// fallback (band-geometry unification design §6). These are the same pixels
    /// [`to_pixel_str`](Self::to_pixel_str) yields for marks, so a categorical
    /// axis that places its tick labels/grid lines here agrees with the marks
    /// (spec §7). Consumed by [`crate::layout::axis`] as `categorical_positions`
    /// on the [`AxisInput`](crate::layout::AxisInput), in place of the
    /// `uniform_center` fallback; the `None` case leaves that path byte-identical.
    pub(in crate::render) fn explicit_band_centers(&self) -> Option<Vec<f64>> {
        match self {
            ScaleKind::Ordinal(s) => s.explicit_band_centers(),
            _ => None,
        }
    }

    /// Re-anchor an EXPLICIT ordinal positional range onto a different facet
    /// panel (GH #70 — explicit Band/Point/positional-Ordinal ranges are
    /// chart-absolute by design, so every panel must translate by its own
    /// displacement from the reference panel). Delegates to
    /// [`OrdinalScale::translate_explicit_range`], which itself no-ops unless
    /// this scale's range was recorded as user-supplied
    /// (`explicit_pixel_range`). A no-op for every non-`Ordinal` scale kind —
    /// continuous scales' explicit ranges are out of this fix's scope.
    pub(in crate::render) fn translate_explicit_ordinal_range(&mut self, offset: f64) {
        if let ScaleKind::Ordinal(s) = self {
            s.translate_explicit_range(offset);
        }
    }

    /// Continuous-axis scale-projection support (continuous-axis tick design,
    /// 2026-05-30). Returns, for each major tick, its **domain fraction**
    /// `t ∈ [0, 1]` — the scale's normalized projection of the tick value,
    /// independent of the pixel range. Computed as `(pixel - r0) / (r1 - r0)`
    /// over this scale's own resolved range, so the scale's nonlinearity
    /// (log/pow/symlog/time) is captured. Layout maps each fraction onto the
    /// panel's mark range via the *same* padding inset that places data marks,
    /// so a tick at value `v` and a data mark at value `v` share a pixel.
    ///
    /// `Ordinal` (and any categorical) scales return an empty vec — categorical
    /// axes keep uniform-slot placement and supply no projected fractions.
    ///
    /// The companion padding fraction (recovered by the caller from the
    /// provisional `[0,1]`-range scale's `pixel_range`) tells layout how much to
    /// inset the panel range before interpolating these fractions.
    pub(crate) fn tick_fractions(&self, count_hint: usize) -> Vec<f64> {
        if matches!(self, Self::Ordinal(_)) {
            return Vec::new();
        }
        // Project the SAME tick values that `tick_labels`/`ticks_internal`
        // produce — one fraction per value, in the same order — so the carrier
        // stays index-aligned with `tick_labels`.
        let values = dispatch_continuous!(self, ticks_internal, count_hint);
        self.project_values_to_fractions(&values, NonFinite::RejectAll)
    }

    /// Raw numeric tick *values* for a continuous scale, in the same order and
    /// count as [`tick_labels`](Self::tick_labels) (both delegate to
    /// `ticks_internal`). `None` for ordinal scales (no numeric tick domain).
    /// Used by `prepare.rs` to apply the per-axis `tick_min_step` / `tick_extra`
    /// adjustments in data space (B5 unit 2).
    pub(crate) fn tick_values_raw(&self, count_hint: usize) -> Option<Vec<f64>> {
        if matches!(self, Self::Ordinal(_)) {
            return None;
        }
        Some(dispatch_continuous!(self, ticks_internal, count_hint))
    }

    /// Continuous-axis scale-projection support for explicit tick values
    /// (`configure_axis(tick_values=[...])`). Projects each supplied data value
    /// to a domain fraction `t ∈ [0, 1]` over this scale's resolved range, in
    /// input order. Returns an empty vec for ordinal scales (explicit numeric
    /// tick values are meaningless on a categorical axis).
    pub(crate) fn value_fractions(&self, values: &[f64]) -> Vec<f64> {
        if matches!(self, Self::Ordinal(_)) {
            return Vec::new();
        }
        self.project_values_to_fractions(values, NonFinite::RejectAll)
    }

    /// Project data values to domain fractions `(scale(v) - r0) / (r1 - r0)`,
    /// shared by [`tick_fractions`](Self::tick_fractions),
    /// [`value_fractions`](Self::value_fractions), and
    /// [`minor_tick_fractions`](Self::minor_tick_fractions). Callers must rule
    /// out ordinal scales first.
    ///
    /// A zero-span (degenerate domain: all-equal / single distinct / all-null
    /// column) always yields an **empty** vec regardless of policy.
    ///
    /// Non-finite projections (a value the scale maps to `NaN`/`±inf`, e.g.
    /// out-of-domain on an unclamped scale) are handled per [`NonFinite`]:
    ///
    /// * [`NonFinite::RejectAll`] — labeled major / explicit ticks. The result
    ///   must stay **index-aligned** with `tick_labels`, so a single non-finite
    ///   projection collapses the whole vec to empty. The caller then drops the
    ///   carrier (`None`) and layout falls back to uniform-slot placement —
    ///   exactly the pre-projection (baseline) behavior for degenerate axes.
    /// * [`NonFinite::DropOne`] — unlabeled minor ticks. Each non-finite
    ///   projection is dropped individually; dropping one minor cannot misalign
    ///   anything.
    fn project_values_to_fractions(&self, values: &[f64], non_finite: NonFinite) -> Vec<f64> {
        let (r0, r1) = self.pixel_range();
        let span = r1 - r0;
        if span == 0.0 {
            return Vec::new();
        }
        let fractions = values
            .iter()
            .map(|&v| {
                let px = dispatch_continuous!(self, scale_internal, v);
                (px - r0) / span
            });
        match non_finite {
            NonFinite::DropOne => fractions.filter(|f| f.is_finite()).collect(),
            NonFinite::RejectAll => {
                let collected: Vec<f64> = fractions.collect();
                if collected.iter().all(|f| f.is_finite()) {
                    collected
                } else {
                    Vec::new()
                }
            }
        }
    }

    /// Continuous-axis scale-projection support: the padding fraction implied by
    /// this scale's resolved pixel range, recovered as the inset relative to its
    /// own range span. For the provisional `[0,1]`-range scales built in
    /// `prepare.rs` this equals the `padding_frac` that
    /// [`crate::layout::geometry::inset_pixel_range`] used (the cap never binds
    /// at a span of 1). Layout uses it to inset the panel mark range identically.
    /// Returns `0.0` for ordinal scales.
    pub(crate) fn padding_fraction(&self) -> f64 {
        if matches!(self, Self::Ordinal(_)) {
            return 0.0;
        }
        let (r0, r1) = self.pixel_range();
        // The provisional scale spans the normalized base range (0,1) or (1,0);
        // after insetting, the band is `(pad, 1-pad)` (x) or `(1-pad, pad)` (y).
        // The inset distance from the nearer base edge [0, 1] is `pad`, which —
        // at unit span — is the padding fraction.
        let lo = r0.min(r1);
        let hi = r0.max(r1);
        lo.min(1.0 - hi).max(0.0)
    }

    pub fn tick_data(&self, count_hint: usize) -> Vec<ferrum_scene::Tick> {
        match self {
            Self::Ordinal(_) => Vec::new(),
            Self::Linear(s) => s
                .ticks_internal(count_hint)
                .into_iter()
                .filter_map(|v| {
                    let px = s.scale_internal(v);
                    if px.is_finite() {
                        Some(ferrum_scene::Tick {
                            value: v,
                            label: super::format::format_numeric(v),
                            pixel: px,
                        })
                    } else {
                        None
                    }
                })
                .collect(),
            Self::Time(s) => {
                let ticks = s.ticks_internal(count_hint);
                let spacing = if ticks.len() >= 2 {
                    (ticks[1] - ticks[0]) as i64
                } else {
                    86_400_000
                };
                ticks
                    .into_iter()
                    .filter_map(|v| {
                        let px = s.scale_internal(v);
                        if px.is_finite() {
                            Some(ferrum_scene::Tick {
                                value: v,
                                label: super::format::format_time(v as i64, spacing),
                                pixel: px,
                            })
                        } else {
                            None
                        }
                    })
                    .collect()
            }
            Self::Log(s) => s
                .ticks_internal(count_hint)
                .into_iter()
                .filter_map(|v| {
                    let px = s.scale_internal(v);
                    if px.is_finite() {
                        Some(ferrum_scene::Tick {
                            value: v,
                            label: super::format::format_numeric(v),
                            pixel: px,
                        })
                    } else {
                        None
                    }
                })
                .collect(),
            Self::Symlog(s) => s
                .ticks_internal(count_hint)
                .into_iter()
                .filter_map(|v| {
                    let px = s.scale_internal(v);
                    if px.is_finite() {
                        Some(ferrum_scene::Tick {
                            value: v,
                            label: super::format::format_numeric(v),
                            pixel: px,
                        })
                    } else {
                        None
                    }
                })
                .collect(),
            Self::Pow(s) => s
                .ticks_internal(count_hint)
                .into_iter()
                .filter_map(|v| {
                    let px = s.scale_internal(v);
                    if px.is_finite() {
                        Some(ferrum_scene::Tick {
                            value: v,
                            label: super::format::format_numeric(v),
                            pixel: px,
                        })
                    } else {
                        None
                    }
                })
                .collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ColorScale {
    Categorical {
        domain: Vec<String>,
        palette: Cow<'static, [Color]>,
    },
    /// Continuous color scale: maps a numeric value to a color via a
    /// ContinuousScheme. Used by heatmap, raster, and any chart with an
    /// explicit linear color scale spec.
    ///
    /// `midpoint` is `Some(mid)` only for diverging scales with a non-geometric
    /// center.  When present, the normalization is piecewise-linear:
    ///   t = 0.5 * (v - lo) / (mid - lo)   for v <= mid
    ///   t = 0.5 + 0.5 * (v - mid) / (hi - mid)  for v > mid
    /// Sequential scales always have `midpoint = None` and use the existing
    /// pure-linear `(v - lo) / (hi - lo)` normalization.
    Continuous {
        domain: (f64, f64),
        scheme: crate::render::color::ContinuousScheme,
        /// Explicit diverging midpoint. `None` → pure-linear (sequential).
        midpoint: Option<f64>,
    },
}

impl ColorScale {
    pub fn lookup(&self, value: &str) -> Option<Color> {
        match self {
            Self::Categorical { domain, palette } => domain
                .iter()
                .position(|v| v == value)
                .map(|i| palette[i % palette.len()]),
            Self::Continuous { domain, scheme, midpoint } => {
                let v: f64 = value.parse().ok()?;
                let t = normalize_continuous(*domain, *midpoint, v);
                Some(scheme.sample(t))
            }
        }
    }

    /// Categorical domain order (== legend category order), if this is a
    /// categorical scale. `None` for continuous scales. Used by the dodge
    /// position adjustment to order sub-band slots so they match the legend.
    pub(crate) fn categorical_domain(&self) -> Option<&[String]> {
        match self {
            Self::Categorical { domain, .. } => Some(domain),
            Self::Continuous { .. } => None,
        }
    }

    /// Sample at numeric value (Continuous variant only). Returns None for
    /// Categorical scales.
    pub fn lookup_f64(&self, value: f64) -> Option<Color> {
        match self {
            Self::Continuous { domain, scheme, midpoint } => {
                let t = normalize_continuous(*domain, *midpoint, value);
                Some(scheme.sample(t))
            }
            _ => None,
        }
    }
}

/// Normalize a data value to `t ∈ [0, 1]` for color lookup.
///
/// Sequential scales (midpoint = None): pure-linear `(v - lo) / (hi - lo)`.
/// Diverging scales with an explicit midpoint: piecewise-linear so that
/// `lo → 0`, `mid → 0.5`, `hi → 1`, placing the scheme's neutral center at
/// the user-supplied `mid` rather than the geometric center.
///
/// The result is clamped to `[0, 1]` in both branches.
pub(super) fn normalize_continuous(domain: (f64, f64), midpoint: Option<f64>, v: f64) -> f64 {
    let (lo, hi) = domain;
    if hi <= lo {
        return 0.5;
    }
    match midpoint {
        None => {
            // Sequential: pure-linear normalization.
            ((v - lo) / (hi - lo)).clamp(0.0, 1.0)
        }
        Some(mid) => {
            // Diverging: piecewise-linear around the midpoint.
            if v <= mid {
                let denom = mid - lo;
                if denom == 0.0 { return 0.5; }
                (0.5 * (v - lo) / denom).clamp(0.0, 0.5)
            } else {
                let denom = hi - mid;
                if denom == 0.0 { return 0.5; }
                (0.5 + 0.5 * (v - mid) / denom).clamp(0.5, 1.0)
            }
        }
    }
}

/// A linear size scale: maps a quantitative field to a radius/diameter in pixels.
///
/// The [`min_px`](Self::min_px) / [`max_px`](Self::max_px) endpoints are
/// stored as `inner`'s pixel range — there's no separate storage. Use the
/// accessor methods rather than re-reading `inner.pixel_range()` at call
/// sites so the intent stays readable.
#[derive(Debug, Clone)]
pub struct SizeScale {
    /// The underlying linear scale (typically `ScaleKind::Linear`). Its
    /// pixel range encodes the `[min_px, max_px]` band.
    pub inner: ScaleKind,
}

impl SizeScale {
    /// Pixel diameter for the smallest data value (range lower bound).
    /// Default behavior: 3.0 px (set by `build_size_scale` from theme).
    pub fn min_px(&self) -> f64 { self.inner.pixel_range().0 }
    /// Pixel diameter for the largest data value (range upper bound).
    /// Default behavior: 30.0 px (set by `build_size_scale` from theme).
    pub fn max_px(&self) -> f64 { self.inner.pixel_range().1 }
}

/// An ordinal shape scale: maps a categorical field to one of 8 shapes.
#[derive(Debug, Clone)]
pub struct ShapeScale {
    pub domain: Vec<String>,   // distinct values in encounter order
    pub shapes: Vec<ShapeKind>, // mapped from SHAPE_PALETTE
}

impl ShapeScale {
    pub fn lookup(&self, value: &str) -> Option<ShapeKind> {
        self.domain
            .iter()
            .position(|v| v == value)
            .map(|i| self.shapes[i])
    }
}

/// The 8 point shapes available to the shape scale and `mark_point(shape=…)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeKind {
    Circle,
    Square,
    Cross,
    Diamond,
    TriangleUp,
    TriangleDown,
    /// Short vertical line marker (`"|"` or `"vline"`).
    VLine,
    /// Short horizontal line marker (`"-"` or `"hline"`).
    HLine,
}

impl ShapeKind {
    /// Canonical shape name, the inverse of `point::shape_from_str`. Used by the
    /// shape legend so each entry's glyph matches what the mark draws.
    pub fn name(self) -> &'static str {
        match self {
            ShapeKind::Circle => "circle",
            ShapeKind::Square => "square",
            ShapeKind::Cross => "cross",
            ShapeKind::Diamond => "diamond",
            ShapeKind::TriangleUp => "triangle-up",
            ShapeKind::TriangleDown => "triangle-down",
            ShapeKind::VLine => "vline",
            ShapeKind::HLine => "hline",
        }
    }
}

/// Fixed 8-shape palette used by `build_shape_scale`. Wraps on overflow.
pub const SHAPE_PALETTE: [ShapeKind; 8] = [
    ShapeKind::Circle,
    ShapeKind::Square,
    ShapeKind::Cross,
    ShapeKind::Diamond,
    ShapeKind::TriangleUp,
    ShapeKind::TriangleDown,
    ShapeKind::VLine,
    ShapeKind::HLine,
];

/// A linear opacity scale: maps a quantitative field to `[min_opacity, max_opacity]`.
///
/// The endpoints are stored as `inner`'s pixel range — no separate
/// storage. Use the accessor methods at call sites for readability.
#[derive(Debug, Clone)]
pub struct OpacityScale {
    /// The underlying linear scale (typically `ScaleKind::Linear`). Its
    /// pixel range encodes the `[min_opacity, max_opacity]` band.
    pub inner: ScaleKind,
}

impl OpacityScale {
    /// Opacity for the smallest data value (range lower bound).
    /// Default behavior: 0.1 (set by `build_opacity_scale` from theme).
    pub fn min_opacity(&self) -> f64 { self.inner.pixel_range().0 }
    /// Opacity for the largest data value (range upper bound).
    /// Default behavior: 1.0 (set by `build_opacity_scale` from theme).
    pub fn max_opacity(&self) -> f64 { self.inner.pixel_range().1 }
}

/// Per-layer independent y-scale slots (secondary-y-axis, GH #52).
///
/// `slots[0]` is always the primary y-scale — identical to
/// [`ResolvedScales::y`] — driving the left axis and gridlines. Each later
/// entry is an independent layer's own resolved y-scale, in the order the
/// independent layers appear (slot `k` = k-th `independent_y` layer, drawn on
/// the right, stacked outward). `layer_slot[i]` is the slot layer `i`'s marks
/// map through; shared layers map to slot 0.
///
/// The default (empty `slots`, empty `layer_slot`) means the chart has no
/// independent-y layer: every layer uses [`ResolvedScales::y`] and consumers
/// resolve byte-identically to the pre-#52 shared path. Axis emission (Task 3)
/// and the interactive coordinate state (Task 8) read these same slots so mark
/// geometry, ticks, and domain state agree by construction.
#[derive(Debug, Clone, Default)]
pub struct YScaleSlots {
    slots: Vec<ScaleKind>,
    layer_slot: Vec<usize>,
}

impl YScaleSlots {
    /// Build slots from an ordered `slots` list (index 0 = primary) and a
    /// `layer_slot` map (layer index → slot index). Callers guarantee slot 0 is
    /// the primary y and that every `layer_slot` entry indexes into `slots`.
    pub fn new(slots: Vec<ScaleKind>, layer_slot: Vec<usize>) -> Self {
        Self { slots, layer_slot }
    }

    /// A single-slot value wrapping `scale` as the only (primary) slot, with
    /// an empty `layer_slot` map — every layer index falls back to slot 0.
    ///
    /// For a `ResolvedScales` whose `.y` has been reassigned to one layer's
    /// own scale (e.g. the per-layer clone in `build_panel_mark_batches` that
    /// binds an independent-y layer), this keeps `y_slots` self-describing:
    /// `slots()` reports exactly the one scale `.y` now points to, instead of
    /// the stale multi-slot list from the panel-level `ResolvedScales` it was
    /// cloned from. `has_independent()` is `false` and `slot_for_layer` is `0`
    /// for any layer index, matching a chart with no independent-y layer.
    pub fn single(scale: ScaleKind) -> Self {
        Self { slots: vec![scale], layer_slot: Vec::new() }
    }

    /// Ordered y-scales, `slots[0]` the primary/left axis. Empty when the chart
    /// has no independent-y layer.
    ///
    /// Read by axis emission (Task 3, iterates slots → left axis for slot 0,
    /// stacked right axes for slots 1..n) and the interactive coordinate state
    /// (Task 8, `scene_build.rs` emits one y-domain per slot into
    /// `CoordKind::Cartesian::y_domains`). Not consumed by the Task 2
    /// resolution/binding path, which routes through [`ResolvedScales::y_for_layer`].
    pub fn slots(&self) -> &[ScaleKind] {
        &self.slots
    }

    /// Whether any independent y-slot exists beyond the primary.
    ///
    /// Read by axis-band layout (Task 3) and the per-slot scene-coord pass
    /// (Task 8) to decide whether there is dual-axis state to emit. Not
    /// consumed by the Task 2 resolution/binding path.
    pub fn has_independent(&self) -> bool {
        self.slots.len() > 1
    }

    /// Slot index a layer's marks map through. Shared layers — and every layer
    /// when the chart has no independent slot — map to slot 0.
    pub fn slot_for_layer(&self, layer_idx: usize) -> usize {
        self.layer_slot.get(layer_idx).copied().unwrap_or(0)
    }
}

/// The structural layer→y-slot plan (secondary-y-axis, GH #52 / #72), computed
/// **once** at prepare time from the layers' `independent_y` flags and stored on
/// [`crate::render::prepare::PreparedInputs`].
///
/// Before #72 the same layer→slot mapping was re-derived independently at three
/// sites — the prepare axis-input push loop, the per-panel [`YScaleSlots`] slot
/// loop, and the axis router's position-inference (`i + 1`) — plus two
/// order-coupled consumers. A change to one loop silently desynced axis labels,
/// mark placement, and interactive `y_domains` with no compile-time signal
/// (design-review S3-2). This plan is the single derivation: every site reads
/// [`slot_for_layer`](Self::slot_for_layer) / [`secondary_layers`](Self::secondary_layers)
/// instead of walking the layers itself.
///
/// Mirrors the #16 [`LegendBandPlan`](crate::render::composite_render) compute-
/// once/consume-later pattern and its **index-keying** rationale: the plan keys
/// slots by layer *index* (structural, re-derivable from the layer list's shape)
/// rather than by any per-panel resolved artifact, so it stays valid across the
/// prepare→scene_build stage boundary and across panels without re-computation.
///
/// `slots` follow GH #63's split indexing convention untouched: `slot_for_layer`
/// returns 0 for the primary/left axis and layer, and `1..=n` for the n
/// independent-y layers in layer order — the SAME 1-based secondary numbering
/// `y_slot_levels` (`crates/ferrum-scene`) and the WASM crate's
/// `secondary_affines` (`render.rs`) index by. The split (all-slot 0-based
/// collections like `y_domains`/`slot_rescales`/`panel_slot_counts` vs.
/// secondary-only 1-based collections like `y_slot_levels`/`secondary_affines`)
/// is intentional, not accidental drift: a secondary-only list has no slot-0
/// entry to pad, so reindexing it to all-slot would waste an unused element on
/// every single-y-plus-one chart. GH #63 killed the alternative — hand-computed
/// `slot - 1` at each consumer — in favor of one named accessor per collection,
/// so the 1-based-vs-0-based split lives in exactly one place per collection,
/// never at a call site:
/// - `y_domains` / `slot_rescales` / `panel_slot_counts`: already all-slot
///   0-based, read directly (`y_domains.get(y_slot)`) or via
///   [`crate::render::scale_resolve`]'s sibling
///   `transform_slot_index`/`panel_slot_range` helpers
///   (`crates/ferrum-wasm/src/scene_load.rs`) for the flat-packed
///   `slot_rescales` array.
/// - `secondary_layers()`/`slot_for_layer()` above: this plan's own accessors.
/// - `secondary_affines` (WASM `render.rs`, GH #60/#73): read only through
///   `secondary_affine_for_slot` (`crates/ferrum-wasm/src/text_json.rs`), which
///   owns the `slot - 1` translation and the documented `.last()`/panel-affine
///   fallback chain.
/// - `y_slot_levels`: builder-only from Rust's side (`scene_build.rs`'s
///   `.skip(1)`, the one place per collection the split's *range* belongs);
///   no Rust code reads it back — the JS frontend is its only consumer, so no
///   Rust accessor exists for it (nothing to encapsulate on this side yet).
///
/// The default (empty) plan means no independent-y layer — the byte-stable
/// pre-#52 shared path.
#[derive(Debug, Clone, Default)]
pub struct YSlotPlan {
    /// layer index → slot index. Empty on the shared path (no independent-y
    /// layer); otherwise one entry per layer, `0` for the primary/shared layers
    /// and `1..=n` for the independent-y layers in layer order.
    layer_slot: Vec<usize>,
    /// The layer indices that own a secondary (right-axis) slot, in slot order:
    /// `secondary_layers[k]` is the layer drawn on slot `k + 1`. Length = number
    /// of independent-y layers. Empty on the shared path.
    secondary_layers: Vec<usize>,
}

impl YSlotPlan {
    /// Derive the plan from each layer's `independent_y` flag, in layer order.
    /// Layer 0 is always the primary/left axis regardless of its flag; each
    /// later layer whose flag is set takes the next secondary slot. Returns the
    /// empty (default) plan when no such layer exists, so the shared path is
    /// byte-identical to pre-#52.
    ///
    /// This is the ONE place the `skip(1).filter(independent_y)` derivation
    /// lives; every consumer reads the resulting map.
    pub fn from_layer_flags<I: IntoIterator<Item = bool>>(flags: I) -> Self {
        let mut layer_slot: Vec<usize> = Vec::new();
        let mut secondary_layers: Vec<usize> = Vec::new();
        for (layer_idx, flag) in flags.into_iter().enumerate() {
            // Layer 0 is always the primary/left axis; only later flagged layers
            // take a secondary slot. `secondary_layers.len()` after the push is
            // the 1-based slot index (slot 1 = first independent layer).
            let slot = if layer_idx != 0 && flag {
                secondary_layers.push(layer_idx);
                secondary_layers.len()
            } else {
                0
            };
            layer_slot.push(slot);
        }
        if secondary_layers.is_empty() {
            // Byte-stable shared path: an empty plan mirrors `YScaleSlots::default`.
            return Self::default();
        }
        Self { layer_slot, secondary_layers }
    }

    /// Whether any independent y-slot exists beyond the primary.
    pub fn has_independent(&self) -> bool {
        !self.secondary_layers.is_empty()
    }

    /// Slot index a layer's marks map through. Shared layers — and every layer
    /// when the chart has no independent slot — map to slot 0.
    pub fn slot_for_layer(&self, layer_idx: usize) -> usize {
        self.layer_slot.get(layer_idx).copied().unwrap_or(0)
    }

    /// The layer indices owning a secondary slot, in slot order: index `k` is the
    /// layer drawn on slot `k + 1`. Consumed by the prepare axis-input builder,
    /// the per-panel [`YScaleSlots`] resolution, and the axis router — all in
    /// this one order.
    pub fn secondary_layers(&self) -> &[usize] {
        &self.secondary_layers
    }

    /// The full layer→slot map, for handing to [`YScaleSlots::new`] so the
    /// per-panel resolved slots carry this plan's map rather than re-deriving it.
    pub fn layer_slot(&self) -> &[usize] {
        &self.layer_slot
    }
}

#[derive(Debug, Clone)]
pub struct ResolvedScales {
    pub x: ScaleKind,
    pub y: ScaleKind,
    pub color: Option<ColorScale>,
    // Phase 8a:
    pub size: Option<SizeScale>,
    pub shape: Option<ShapeScale>,
    pub opacity: Option<OpacityScale>,
    // Phase 8b: paired-channel field names. The x2/y2 axis is shared with x/y
    // (their domain is unioned in `build_axis_scale`); this field surfaces the
    // bound field name so downstream code (mark drawers, legends) can read it
    // off `ResolvedScales` without re-walking the spec encoding.
    pub x2: Option<String>,
    pub y2: Option<String>,
    /// Per-layer independent y-scale slots (secondary-y-axis, GH #52). Default
    /// (empty) means every layer shares the primary `y` — the byte-stable
    /// pre-#52 path. See [`YScaleSlots`].
    pub y_slots: YScaleSlots,
}

impl ResolvedScales {
    /// The y-scale a given layer's marks map through. Layers that share the
    /// primary y (the default, and every layer on a chart with no independent
    /// slot) get `y`; an independent layer gets its own slot scale.
    pub fn y_for_layer(&self, layer_idx: usize) -> &ScaleKind {
        let slot = self.y_slots.slot_for_layer(layer_idx);
        // `slots` is empty on the shared path; slot 0 mirrors `y`, so fall back
        // to `y` whenever the slot list is not populated.
        self.y_slots.slots.get(slot).unwrap_or(&self.y)
    }
}

// ── Shared helpers used by sub-modules ──────────────────────────────────────

pub(in crate::render) fn infer_spec_type(
    enc: &crate::spec::encoding::EncodingSpec,
    dtype: &ArrowDataType,
) -> SpecDataType {
    if let Some(t) = enc.type_ {
        return t;
    }
    match dtype {
        ArrowDataType::Float32
        | ArrowDataType::Float64
        | ArrowDataType::Int8
        | ArrowDataType::Int16
        | ArrowDataType::Int32
        | ArrowDataType::Int64
        | ArrowDataType::UInt8
        | ArrowDataType::UInt16
        | ArrowDataType::UInt32
        | ArrowDataType::UInt64 => SpecDataType::Quantitative,
        ArrowDataType::Date32 | ArrowDataType::Date64 | ArrowDataType::Timestamp(_, _) => {
            SpecDataType::Temporal
        }
        ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Boolean => {
            SpecDataType::Nominal
        }
        _ => SpecDataType::Nominal,
    }
}

/// Canonical (min, max) for a numeric Arrow column (SPINE-09: one wrapper, was
/// three identical per-submodule copies). Re-exported so `domain`, `positional`,
/// and `auxiliary` all reach the same `arrow_cast::min_max_f64` via `super::`.
pub(super) use super::arrow_cast::min_max_f64 as column_min_max_f64;

/// Compute (min, max) for a numeric Arrow column, skipping NaN/null values.
/// Returns (0.0, 1.0) when no finite values are present.
///
/// `pub(in crate::render)` so the composite resolve pass (`render::composite`)
/// can compute a leaf's shared color/size extent through the same primitive the
/// facet-shared continuous-color path uses (10-pre-b).
pub(in crate::render) fn numeric_extent(col: &dyn arrow::array::Array) -> (f64, f64) {
    super::arrow_cast::finite_min_max_f64(col).unwrap_or((0.0, 1.0))
}

/// Select the batch for categorical domain resolution in faceted charts.
///
/// When `facet_shared` is true, returns the global `FINAL_OUTPUT_KEY` batch
/// (so every panel uses global first-appearance order, matching the legend).
/// Falls back to `primary_batch` when:
/// - `facet_shared` is false (non-faceted chart)
/// - `FINAL_OUTPUT_KEY` is absent from `transform_outputs`
/// - The field is absent from the global batch
///
/// Used by both `color.rs` (categorical color) and `auxiliary.rs` (shape) so
/// that the same global-domain guarantee applies to all categorical channels.
pub(super) fn shared_categorical_batch<'a>(
    primary_batch: &'a RecordBatch,
    field: &str,
    transform_outputs: &'a HashMap<String, RecordBatch>,
    facet_shared: bool,
) -> &'a RecordBatch {
    if !facet_shared {
        return primary_batch;
    }
    use crate::transform::core::FINAL_OUTPUT_KEY;
    let Some(global_batch) = transform_outputs.get(FINAL_OUTPUT_KEY) else {
        return primary_batch;
    };
    if global_batch.column_by_name(field).is_none() {
        return primary_batch;
    }
    global_batch
}

/// Union a per-panel (lo, hi) numeric extent with the global `FINAL_OUTPUT_KEY`
/// batch's extent for `field`.
///
/// Returns the unioned range, which equals the global range since the per-panel
/// batch is a partition of the global batch. Falls back to `panel_extent` when:
/// - `FINAL_OUTPUT_KEY` is absent from `transform_outputs`
/// - The field is absent from the global batch
/// - The global column has no finite values
///
/// Used by both continuous-color (`color.rs`) and auxiliary (`auxiliary.rs`)
/// scale builders for the T3 faceted-shared-extent fix.
pub(super) fn union_panel_with_global_extent(
    panel_extent: (f64, f64),
    field: &str,
    transform_outputs: &HashMap<String, RecordBatch>,
) -> (f64, f64) {
    use crate::transform::core::FINAL_OUTPUT_KEY;
    let Some(global_batch) = transform_outputs.get(FINAL_OUTPUT_KEY) else {
        return panel_extent;
    };
    let Some(col) = global_batch.column_by_name(field) else {
        return panel_extent;
    };
    let (g_lo, g_hi) = numeric_extent(col.as_ref());
    let (p_lo, p_hi) = panel_extent;
    (p_lo.min(g_lo), p_hi.max(g_hi))
}

/// First-appearance-order distinct string values of `field` (nulls dropped),
/// the categorical-domain primitive shared by the color/shape scale builders.
///
/// `pub(in crate::render)` so the composite resolve pass (`render::composite`)
/// can union a leaf's shared categorical color domain through the same primitive
/// the categorical color path uses (10-pre-b).
pub(in crate::render) fn distinct_values_in_order(
    batch: &RecordBatch,
    field: &str,
) -> Result<Vec<String>, RenderError> {
    super::arrow_cast::distinct_values_in_order(batch, field)
}

/// Positional (x:N / y:N) ordinal domain: like [`distinct_values_in_order`] but
/// surfaces a null row as its own category (FA-9). Used only by the positional
/// scale path so color/shape/legend domains keep dropping nulls.
fn distinct_positional_categories(
    batch: &RecordBatch,
    field: &str,
) -> Result<Vec<String>, RenderError> {
    super::arrow_cast::distinct_positional_categories(batch, field)
}

/// T4: per-channel "shared faceted positional scale" flags `(x_shared, y_shared)`.
///
/// `true` only when the chart is faceted AND that channel's
/// [`ResolveMode`](crate::layout::facet::ResolveMode) is `Shared` (the default
/// when `resolve` is omitted). Non-faceted charts and `Independent` channels
/// yield `false`, so [`build_axis_scale`] keeps per-panel-only domain resolution
/// byte-identically. Passed to `build_axis_scale` → `numeric_domain_union` /
/// `distinct_positional_categories_shared` as `include_final`.
fn facet_shared_flags(spec: &ChartSpec) -> (bool, bool) {
    use crate::layout::facet::ResolveMode;
    match &spec.facet {
        Some(facet) => (
            facet.resolve.x == ResolveMode::Shared,
            facet.resolve.y == ResolveMode::Shared,
        ),
        None => (false, false),
    }
}

/// T3: "shared faceted auxiliary scale" flag for non-positional channels
/// (continuous color, size, opacity).
///
/// Returns `true` when the chart is faceted (`spec.facet.is_some()`). There is
/// no independent option for these channels — `FacetResolve` only has `x`/`y`
/// fields — so the gate is simply whether the chart is faceted at all. Passing
/// `false` for non-faceted charts keeps the per-panel-only domain resolution
/// byte-identical to pre-T3 behavior.
fn facet_aux_shared(spec: &ChartSpec) -> bool {
    spec.facet.is_some()
}

// ── Private helpers ─────────────────────────────────────────────────────────

/// Build the four auxiliary (non-positional) scales for a chart spec.
///
/// Computes `force_cat` and `aux_shared` internally from `spec` so the three
/// dispatch sites in `resolve_scales_with_outputs` share one definition. Warning
/// push order is: `color_warns` (via `extend`), then `shape_warn` (via `push`).
///
/// `leaf_scales` is the 10-pre-b composite seam: `Some` only for a composite leaf
/// whose parent shares `color`/`size`. It seeds the color/size auto path with the
/// domain unioned across the composite's leaves, exactly as `leaf_scales`
/// (x/y) seeds the positional axes. `None` for standalone (flat/facet) renders
/// reproduces the pre-10-pre-b behavior byte-for-byte.
#[allow(clippy::type_complexity)]
fn build_auxiliary_scales(
    spec: &ChartSpec,
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    theme: &ThemeInputs,
    leaf_scales: Option<&LeafScaleContext>,
    warnings: &mut Vec<crate::render::RenderWarning>,
) -> Result<(Option<ColorScale>, Option<SizeScale>, Option<ShapeScale>, Option<OpacityScale>), RenderError> {
    // FA-5: area marks always group color discretely; force categorical.
    let force_cat = matches!(spec.mark, crate::spec::mark::Mark::Area);
    // T3: when the chart is faceted, auxiliary non-positional channels (continuous
    // color, size, opacity) union the global FINAL_OUTPUT_KEY batch so per-panel
    // marks normalize through the same domain as the global legend/colorbar.
    let aux_shared = facet_aux_shared(spec);
    // 10-pre-b: composite shared color/size domains (None → standalone path).
    let color_domain = leaf_scales.and_then(|c| c.color.as_ref());
    let size_domain = leaf_scales.and_then(|c| c.size.as_ref());
    let (color, color_warns) = build_color_scale(&spec.encoding, primary_batch, transform_outputs, theme, force_cat, aux_shared, color_domain)?;
    warnings.extend(color_warns);
    let (size, size_warns) = build_size_scale(&spec.encoding, primary_batch, transform_outputs, aux_shared, theme, size_domain)?;
    warnings.extend(size_warns);
    let (shape, shape_warns) = build_shape_scale(&spec.encoding, primary_batch, transform_outputs, aux_shared)?;
    warnings.extend(shape_warns);
    let (opacity, opacity_warns) = build_opacity_scale(&spec.encoding, primary_batch, transform_outputs, aux_shared, theme)?;
    warnings.extend(opacity_warns);
    Ok((color, size, shape, opacity))
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Build a dummy unit `LinearScale` that maps `[0, 1]` domain onto the given
/// pixel range. Used when a particular axis is not encoded (Geoshape ignores
/// both axes; single-axis Tick/Rule ignores the absent axis). The mark builder
/// for those marks never reads the dummy scale; it exists so `ResolvedScales`
/// is always fully populated.
///
/// `ascending` controls the pixel-range ordering: `true` → `[lo, hi]` (x
/// convention), `false` → `[hi, lo]` (y convention, where the top pixel is
/// smaller and the bottom is larger).
#[inline]
fn dummy_unit_scale(pixel_range: (f64, f64), ascending: bool) -> ScaleKind {
    let (lo, hi) = pixel_range;
    let range = if ascending { vec![lo, hi] } else { vec![hi, lo] };
    ScaleKind::Linear(LinearScale::new_internal(vec![0.0, 1.0], range, false, false))
}

// ── Main entry points ───────────────────────────────────────────────────────

/// Build scales from spec + post-transform batch + pixel ranges.
/// Pixel ranges are panel-relative; caller passes panel.plot_area bounds.
///
/// This is the back-compat single-batch entry point. For Phase 8b layered charts
/// where encoding fields may live in named transform outputs other than
/// `__final__`, prefer `resolve_scales_with_outputs`.
pub fn resolve_scales(
    spec: &ChartSpec,
    batch: &RecordBatch,
    x_pixel_range: (f64, f64),
    y_pixel_range: (f64, f64),
    theme: &ThemeInputs,
) -> Result<(ResolvedScales, Vec<crate::render::RenderWarning>), RenderError> {
    // Empty map → behavior is identical to the pre-8b single-batch path:
    // build_axis_scale falls through to `primary_batch` only.
    let outputs: HashMap<String, RecordBatch> = HashMap::new();
    resolve_scales_with_outputs(spec, batch, &outputs, x_pixel_range, y_pixel_range, theme)
}

/// Phase 8b variant: numeric axis domains union the encoding field's range
/// across `primary_batch` and every batch in `transform_outputs` that contains
/// the field. Categorical scales (color/shape/size/opacity) and ordinal axis
/// scales remain primary-batch-driven; for composite marks the categorical
/// axis field (e.g. boxplot's `x="group"`) is preserved on every named output
/// produced by the composite mark's transform pipeline, so the primary batch
/// is sufficient there.
///
/// Standalone (flat/facet) entry point: no composite leaf-scale context. Delegates
/// to [`resolve_scales_with_leaf_context`] with `None`, mirroring the way
/// [`resolve_scales`] delegates here with an empty outputs map. Byte-identical to
/// the pre-D4b behavior for every existing caller.
pub fn resolve_scales_with_outputs(
    spec: &ChartSpec,
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    x_pixel_range: (f64, f64),
    y_pixel_range: (f64, f64),
    theme: &ThemeInputs,
) -> Result<(ResolvedScales, Vec<crate::render::RenderWarning>), RenderError> {
    resolve_scales_with_leaf_context(
        spec,
        primary_batch,
        transform_outputs,
        x_pixel_range,
        y_pixel_range,
        theme,
        None,
    )
}

/// Reactive-rescale substitution (D6): turn `domainParam` references into
/// concrete domains before scale resolution. No-op when `params` is empty (the
/// byte-stability gate) or when a referenced param yields no static numeric
/// domain (selection / unmatched name → left `None` for auto-infer).
///
/// Lives in `scale_resolve` (not `scene_build`) so it is reachable from BOTH the
/// prepare stage (secondary-y axis inputs) and scene_build (marks, `y_domains`),
/// per the seam doc's one-way dependency: prepare → scale_resolve ← scene_build.
/// Relocated here (#72) so the two per-layer y resolutions share one param-aware
/// path — see [`resolve_layer_y_slot_scale`].
pub(in crate::render) fn resolve_param_domains(spec: &mut ChartSpec) {
    if spec.params.is_empty() {
        return;
    }
    let store = crate::spec::parameter::ParamStore::new(&spec.params);
    if store.is_empty() {
        return;
    }
    let enc = &mut spec.encoding;
    for channel in [
        enc.x.as_mut(),
        enc.y.as_mut(),
        enc.color.as_mut(),
        enc.size.as_mut(),
        enc.opacity.as_mut(),
    ]
    .into_iter()
    .flatten()
    {
        let Some(scale) = channel.scale.as_mut() else { continue };
        let Some(name) = scale.domain_param().map(str::to_owned) else { continue };
        if let Some(domain) = store.numeric_domain(&name) {
            scale.set_domain(domain);
        }
        // else: leave domain = None → auto-infer (empty-selection semantics).
    }
}

/// The chart-level resolution context shared by every per-layer y-slot
/// resolution: the chart spec, transform outputs, theme, and composite
/// leaf-scale context. These four travel together into
/// [`resolve_scales_with_leaf_context`] at every call site in this module
/// (both here and in the `spec`/`primary_batch`/pixel-range form it wraps),
/// so bundling them here — rather than `resolve_layer_y_slot_scale` threading
/// four loose parameters alongside the layer-specific ones — keeps the
/// per-layer, per-call inputs (`layer_mark`, `layer_encoding`, `layer_batch`,
/// the pixel ranges) visually distinct from the resolution-wide context.
///
/// This mirrors, at the lower `scale_resolve` layer, the same shared-context
/// bundling [`PanelResolveCtx`](crate::render::scene_build::PanelResolveCtx)
/// does one layer up in `scene_build` — kept as a separate, smaller type
/// (four fields, not five) because `scene_build`'s `PreparedInputs` and
/// `ChartConfig` are consumer-feature types this lower engine module cannot
/// depend on without inverting the dependency (see the seam doc at the top of
/// `scale_resolve/seam.rs`): `scene_build` calls down into `scale_resolve`,
/// never the reverse.
#[derive(Clone, Copy)]
pub(in crate::render) struct LayerScaleCtx<'a> {
    pub(in crate::render) spec: &'a ChartSpec,
    pub(in crate::render) transform_outputs: &'a HashMap<String, RecordBatch>,
    pub(in crate::render) theme: &'a ThemeInputs,
    pub(in crate::render) leaf_scales: Option<&'a LeafScaleContext>,
}

/// Resolve one independent layer's y-scale slot (secondary-y-axis, GH #52 / #72).
///
/// The single param-aware per-layer y-domain resolution shared by the two stages
/// that must agree: the prepare stage's `build_secondary_y_axis_inputs` (axis
/// ticks/title/band width, with a placeholder pixel range) and scene_build's
/// `resolve_layer_y_scale` (per-panel mark placement + scene `y_domains`, with
/// the panel-real pixel range). Both derive the same logical scale; only the
/// caller-supplied pixel range differs, and domain-derived tick fractions do not
/// depend on it, so the two resolutions agree bit-for-bit on the domain.
///
/// The layer's own encoding overlays the chart encoding, `layers: None` stops
/// [`numeric_domain_union`] from re-unioning sibling layers' y fields (so this
/// slot spans exactly its own data), and `domainParam` references are substituted
/// into concrete domains via [`resolve_param_domains`] BEFORE resolution — the
/// piece the prepare path previously lacked (#72), which made static right-axis
/// ticks diverge from mark placement once layer params reached the wire.
///
/// Returns the resolved `y` [`ScaleKind`] plus any warnings the resolution
/// produced (the caller propagates them). No-op param substitution keeps
/// param-free dual-axis charts byte-identical.
pub(in crate::render) fn resolve_layer_y_slot_scale(
    ctx: &LayerScaleCtx,
    layer_mark: crate::spec::mark::Mark,
    layer_encoding: &crate::spec::encoding::Encoding,
    layer_batch: &RecordBatch,
    x_pixel_range: (f64, f64),
    y_pixel_range: (f64, f64),
) -> Result<(ScaleKind, Vec<crate::render::RenderWarning>), RenderError> {
    let &LayerScaleCtx { spec, transform_outputs, theme, leaf_scales } = ctx;

    let mut merged_encoding = spec.encoding.clone();
    merged_encoding.overlay_from(layer_encoding);
    let mut layer_spec = ChartSpec {
        mark: layer_mark,
        encoding: merged_encoding,
        layers: None,
        ..spec.clone()
    };
    resolve_param_domains(&mut layer_spec);

    let (layer_scales, warnings) = resolve_scales_with_leaf_context(
        &layer_spec,
        layer_batch,
        transform_outputs,
        x_pixel_range,
        y_pixel_range,
        theme,
        leaf_scales,
    )?;
    Ok((layer_scales.y, warnings))
}

/// D4b composite seam: the full scale-resolution form, threading an optional
/// per-leaf resolved-domain context so a composite-shared leaf resolves its
/// positional axes on the auto path (facet padding/`nice`), seeded by the shared
/// domain. `leaf_scales` is `Some` only for a composite leaf; `None` reproduces
/// the standalone behavior byte-for-byte. A channel carrying a genuine user
/// `enc.scale` short-circuits at the explicit-scale bypass inside
/// [`build_axis_scale`] before the context is consulted, so user scale still wins.
#[allow(clippy::too_many_arguments)]
pub(in crate::render) fn resolve_scales_with_leaf_context(
    spec: &ChartSpec,
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    x_pixel_range: (f64, f64),
    y_pixel_range: (f64, f64),
    theme: &ThemeInputs,
    leaf_scales: Option<&LeafScaleContext>,
) -> Result<(ResolvedScales, Vec<crate::render::RenderWarning>), RenderError> {
    let mut warnings = Vec::new();

    // Per-channel composite shared domains (D4b). `None` on a channel → that axis
    // resolves exactly as it would standalone.
    let x_shared_domain = leaf_scales.and_then(|c| c.x.as_ref());
    let y_shared_domain = leaf_scales.and_then(|c| c.y.as_ref());

    // T4: per-channel "shared faceted positional scale" flag. When this chart is
    // faceted AND the channel resolves `ResolveMode::Shared` (the documented
    // default), the auto-inferred positional domain unions the global all-panels
    // batch (`transform_outputs[FINAL_OUTPUT_KEY]`) so per-panel marks scale
    // through the same global domain the shared axis displays. Strictly gated:
    // `false` when not faceted (single panel already has the global batch as
    // primary → byte-identical) and `false` for `Independent` channels (the
    // per-panel escape hatch stays byte-identical).
    let (x_shared, y_shared) = facet_shared_flags(spec);

    // Geoshape marks read geometry from __geometry__ column and don't use x/y scales.
    // Return dummy unit scales so the renderer can proceed; the mark builder ignores them.
    if matches!(spec.mark, crate::spec::mark::Mark::Geoshape) {
        return Ok((
            ResolvedScales {
                x: dummy_unit_scale(x_pixel_range, true),
                y: dummy_unit_scale(y_pixel_range, false),
                color: None, size: None, shape: None, opacity: None,
                x2: None, y2: None, y_slots: YScaleSlots::default(),
            },
            warnings,
        ));
    }

    // Tick, Rule, and Arc marks support single-axis mode: only x or only y is
    // encoded, and the mark builder can proceed using just the present
    // channel.
    // Tick: x-only = x-rug, y-only = y-rug.
    // Rule: y-only = horizontal span, x-only = vertical span.
    // Arc (pie/donut/coxcomb/sunburst under CoordPolar): the coord's *theta*
    // channel is mandatory but the *radius* channel is optional — an absent
    // radius means "full radius", not a missing encoding. Only the coord's
    // theta axis may be the sole positional channel here; a missing theta
    // channel, or a non-polar coord, is not single-axis-eligible and falls
    // through to the unconditional x/y check below, which errors as before.
    // Synthesize a dummy unit scale for the absent axis so the mark builder
    // can access the present scale without scale_resolve erroring.
    let single_axis_eligible = match spec.mark {
        crate::spec::mark::Mark::Tick | crate::spec::mark::Mark::Rule => true,
        crate::spec::mark::Mark::Arc => match &spec.coord {
            Some(crate::spec::coord::CoordKind::Polar { theta, .. }) => match theta {
                ferrum_scene::PolarThetaChannel::X => spec.encoding.x.is_some(),
                ferrum_scene::PolarThetaChannel::Y => spec.encoding.y.is_some(),
            },
            _ => false,
        },
        _ => false,
    };
    if single_axis_eligible {
        let has_x = spec.encoding.x.is_some();
        let has_y = spec.encoding.y.is_some();
        if has_x && !has_y {
            let x_enc = spec.encoding.x.as_ref().unwrap();
            let x2_enc = spec.encoding.x2.as_ref();
            // Stack-aware x-axis (GH #77 follow-up): see
            // `position::axis_batch_for_x` for the rationale.
            let x_batch = crate::render::position::axis_batch_for_x(spec, &x_enc.field, primary_batch);
            let pos_fields = PositionalFields { x: Some(x_enc.field.as_str()), y: None };
            let x = build_axis_scale("x", x_enc, x2_enc, pos_fields, &x_batch, transform_outputs, x_pixel_range, spec, x_shared, x_shared_domain, &mut warnings)?;
            let (color, size, shape, opacity) = build_auxiliary_scales(spec, primary_batch, transform_outputs, theme, leaf_scales, &mut warnings)?;
            return Ok((ResolvedScales {
                x, y: dummy_unit_scale(y_pixel_range, false),
                color, size, shape, opacity, x2: None, y2: None,
                y_slots: YScaleSlots::default(),
            }, warnings));
        }
        if !has_x && has_y {
            let y_enc = spec.encoding.y.as_ref().unwrap();
            let y2_enc = spec.encoding.y2.as_ref();
            let y_batch = crate::render::position::axis_batch_for_y(spec, &y_enc.field, primary_batch);
            let pos_fields = PositionalFields { x: None, y: Some(y_enc.field.as_str()) };
            let y = build_axis_scale("y", y_enc, y2_enc, pos_fields, &y_batch, transform_outputs, y_pixel_range, spec, y_shared, y_shared_domain, &mut warnings)?;
            let (color, size, shape, opacity) = build_auxiliary_scales(spec, primary_batch, transform_outputs, theme, leaf_scales, &mut warnings)?;
            return Ok((ResolvedScales {
                x: dummy_unit_scale(x_pixel_range, true), y,
                color, size, shape, opacity, x2: None, y2: None,
                y_slots: YScaleSlots::default(),
            }, warnings));
        }
    }

    let x_enc = spec
        .encoding
        .x
        .as_ref()
        .ok_or(RenderError::EncodingTypeMismatch {
            channel: "x",
            expected: "EncodingSpec",
            got: "None".into(),
        })?;
    let y_enc = spec
        .encoding
        .y
        .as_ref()
        .ok_or(RenderError::EncodingTypeMismatch {
            channel: "y",
            expected: "EncodingSpec",
            got: "None".into(),
        })?;

    // Phase 8b: paired-channel endpoints (x2/y2) extend the primary axis domain
    // when set, so e.g. ribbons whose y2 lies above y don't render past the
    // resolved range and produce non-finite pixels downstream.
    let x2_enc = spec.encoding.x2.as_ref();
    let y2_enc = spec.encoding.y2.as_ref();
    // Data-aware sort (channel shorthand `"-y"`, sort-field objects) needs both
    // positional field names so an ordinal axis can aggregate the opposite
    // channel's quantitative field.
    let pos_fields = PositionalFields {
        x: Some(x_enc.field.as_str()),
        y: Some(y_enc.field.as_str()),
    };
    // Stack-aware x-axis (GH #77 follow-up): resolve against the post-Stack
    // batch when the spec carries a matching Stack adjustment whose
    // resolved value axis is X. See `position::axis_batch_for_x` for the
    // rationale.
    let x_batch = crate::render::position::axis_batch_for_x(spec, &x_enc.field, primary_batch);
    let mut x = build_axis_scale("x", x_enc, x2_enc, pos_fields, &x_batch, transform_outputs, x_pixel_range, spec, x_shared, x_shared_domain, &mut warnings)?;
    // Stack-aware y-axis: resolve against the post-Stack batch when the
    // spec carries a matching Stack adjustment. See
    // `position::axis_batch_for_y` for the rationale.
    let y_batch = crate::render::position::axis_batch_for_y(spec, &y_enc.field, primary_batch);
    let mut y = build_axis_scale("y", y_enc, y2_enc, pos_fields, &y_batch, transform_outputs, y_pixel_range, spec, y_shared, y_shared_domain, &mut warnings)?;

    // CoordCartesian / CoordFixed domain overrides: explicit xlim/ylim pins the
    // data domain; expand=false removes the default 5% inward padding.
    apply_coord_domain_overrides(spec, &mut x, &mut y, x_pixel_range, y_pixel_range);

    // Color/size/shape/opacity scales are primary-batch only. These channels
    // do not currently participate in cross-layer scale unification: each is
    // resolved against the chart-level transformed batch (i.e. __final__),
    // matching Phase 8a behavior. (build_color_scale is the one exception —
    // it accepts transform_outputs because composite-mark color fields may
    // live in a named output rather than primary.) FA-5 (force_cat) and T3
    // (aux_shared) logic lives in `build_auxiliary_scales`.
    let (color, size, shape, opacity) = build_auxiliary_scales(spec, primary_batch, transform_outputs, theme, leaf_scales, &mut warnings)?;

    let x2_field_name = x2_enc.map(|e| e.field.clone());
    let y2_field_name = y2_enc.map(|e| e.field.clone());

    Ok((
        ResolvedScales {
            x,
            y,
            color,
            size,
            shape,
            opacity,
            x2: x2_field_name,
            y2: y2_field_name,
            // Per-slot y resolution is layered/layer-aware and lives in
            // `scene_build::resolve_panel_scales`, which has the layer list.
            // The single-batch resolver always yields the shared (empty) default.
            y_slots: YScaleSlots::default(),
        },
        warnings,
    ))
}
