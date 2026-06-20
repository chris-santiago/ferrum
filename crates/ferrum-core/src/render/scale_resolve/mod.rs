//! Build ResolvedScales from a ChartSpec + a post-transform RecordBatch.
//! Phase 7 supports: LinearScale, OrdinalScale, TimeScale on x/y;
//! CategoricalColorScale on color.
//! Phase 8a adds: LogScale, SymlogScale (via explicit ScaleSpec override);
//! SizeScale, ShapeScale, OpacityScale for new encoding channels.

mod auxiliary;
mod color;
mod domain;
mod positional;
#[cfg(test)]
mod tests;

use std::borrow::Cow;
use std::collections::HashMap;

use arrow::array::Array;
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

#[derive(Debug)]
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
}

// ── Shared helpers used by sub-modules ──────────────────────────────────────

fn infer_spec_type(
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

fn column_min_max_f64(col: &dyn Array) -> Result<(f64, f64), String> {
    super::arrow_cast::min_max_f64(col)
}

/// Compute (min, max) for a numeric Arrow column, skipping NaN/null values.
/// Returns (0.0, 1.0) when no finite values are present.
fn numeric_extent(col: &dyn arrow::array::Array) -> (f64, f64) {
    super::arrow_cast::finite_min_max_f64(col).unwrap_or((0.0, 1.0))
}

fn distinct_values_in_order(
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
pub fn resolve_scales_with_outputs(
    spec: &ChartSpec,
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    x_pixel_range: (f64, f64),
    y_pixel_range: (f64, f64),
    theme: &ThemeInputs,
) -> Result<(ResolvedScales, Vec<crate::render::RenderWarning>), RenderError> {
    let mut warnings = Vec::new();

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
        let dummy_x = ScaleKind::Linear(LinearScale::new_internal(
            vec![0.0, 1.0], vec![x_pixel_range.0, x_pixel_range.1], false, false,
        ));
        let dummy_y = ScaleKind::Linear(LinearScale::new_internal(
            vec![0.0, 1.0], vec![y_pixel_range.1, y_pixel_range.0], false, false,
        ));
        return Ok((
            ResolvedScales { x: dummy_x, y: dummy_y, color: None, size: None, shape: None, opacity: None,
                x2: None, y2: None },
            warnings,
        ));
    }

    // Tick and Rule marks support single-axis mode: only x or only y is encoded.
    // Tick: x-only = x-rug, y-only = y-rug.
    // Rule: y-only = horizontal span, x-only = vertical span.
    // Synthesize a dummy unit scale for the absent axis so the mark builder
    // can access its present scale without scale_resolve erroring.
    if matches!(spec.mark, crate::spec::mark::Mark::Tick | crate::spec::mark::Mark::Rule) {
        let has_x = spec.encoding.x.is_some();
        let has_y = spec.encoding.y.is_some();
        if has_x && !has_y {
            let x_enc = spec.encoding.x.as_ref().unwrap();
            let x2_enc = spec.encoding.x2.as_ref();
            let pos_fields = PositionalFields { x: Some(x_enc.field.as_str()), y: None };
            let x = build_axis_scale("x", x_enc, x2_enc, pos_fields, primary_batch, transform_outputs, x_pixel_range, spec, x_shared, &mut warnings)?;
            let dummy_y = ScaleKind::Linear(LinearScale::new_internal(
                vec![0.0, 1.0], vec![y_pixel_range.1, y_pixel_range.0], false, false,
            ));
            // FA-5: area marks always group color discretely; force categorical.
            let force_cat = matches!(spec.mark, crate::spec::mark::Mark::Area);
            let (color, color_warns) = build_color_scale(&spec.encoding, primary_batch, transform_outputs, theme, force_cat)?;
            warnings.extend(color_warns);
            let size = build_size_scale(&spec.encoding, primary_batch, theme)?;
            let (shape, shape_warn) = build_shape_scale(&spec.encoding, primary_batch)?;
            if let Some(w) = shape_warn { warnings.push(w); }
            let opacity = build_opacity_scale(&spec.encoding, primary_batch, theme)?;
            return Ok((ResolvedScales { x, y: dummy_y, color, size, shape, opacity, x2: None, y2: None }, warnings));
        }
        if !has_x && has_y {
            let y_enc = spec.encoding.y.as_ref().unwrap();
            let y2_enc = spec.encoding.y2.as_ref();
            let y_batch = crate::render::position::axis_batch_for_y(spec, &y_enc.field, primary_batch);
            let pos_fields = PositionalFields { x: None, y: Some(y_enc.field.as_str()) };
            let y = build_axis_scale("y", y_enc, y2_enc, pos_fields, &y_batch, transform_outputs, y_pixel_range, spec, y_shared, &mut warnings)?;
            let dummy_x = ScaleKind::Linear(LinearScale::new_internal(
                vec![0.0, 1.0], vec![x_pixel_range.0, x_pixel_range.1], false, false,
            ));
            // FA-5: area marks always group color discretely; force categorical.
            let force_cat = matches!(spec.mark, crate::spec::mark::Mark::Area);
            let (color, color_warns) = build_color_scale(&spec.encoding, primary_batch, transform_outputs, theme, force_cat)?;
            warnings.extend(color_warns);
            let size = build_size_scale(&spec.encoding, primary_batch, theme)?;
            let (shape, shape_warn) = build_shape_scale(&spec.encoding, primary_batch)?;
            if let Some(w) = shape_warn { warnings.push(w); }
            let opacity = build_opacity_scale(&spec.encoding, primary_batch, theme)?;
            return Ok((ResolvedScales { x: dummy_x, y, color, size, shape, opacity, x2: None, y2: None }, warnings));
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
    let mut x = build_axis_scale("x", x_enc, x2_enc, pos_fields, primary_batch, transform_outputs, x_pixel_range, spec, x_shared, &mut warnings)?;
    // Stack-aware y-axis: resolve against the post-Stack batch when the
    // spec carries a matching Stack adjustment. See
    // `position::axis_batch_for_y` for the rationale.
    let y_batch = crate::render::position::axis_batch_for_y(spec, &y_enc.field, primary_batch);
    let mut y = build_axis_scale("y", y_enc, y2_enc, pos_fields, &y_batch, transform_outputs, y_pixel_range, spec, y_shared, &mut warnings)?;

    // CoordCartesian / CoordFixed domain overrides: explicit xlim/ylim pins the
    // data domain; expand=false removes the default 5% inward padding.
    apply_coord_domain_overrides(spec, &mut x, &mut y, x_pixel_range, y_pixel_range);

    // Color/size/shape/opacity scales are primary-batch only. These channels
    // do not currently participate in cross-layer scale unification: each is
    // resolved against the chart-level transformed batch (i.e. __final__),
    // matching Phase 8a behavior. (build_color_scale is the one exception —
    // it accepts transform_outputs because composite-mark color fields may
    // live in a named output rather than primary.)
    //
    // FA-5: area marks always group color as discrete categories
    // (col_as_ordinal_category_str), so their color scale must be Categorical
    // regardless of the column's Arrow dtype.  This ensures legend swatches
    // and area fill colors both use the same palette.
    let force_cat = matches!(spec.mark, crate::spec::mark::Mark::Area);
    let (color, color_warns) = build_color_scale(
        &spec.encoding, primary_batch, transform_outputs, theme, force_cat,
    )?;
    warnings.extend(color_warns);

    let size = build_size_scale(&spec.encoding, primary_batch, theme)?;
    let (shape, shape_warn) = build_shape_scale(&spec.encoding, primary_batch)?;
    if let Some(w) = shape_warn {
        warnings.push(w);
    }
    let opacity = build_opacity_scale(&spec.encoding, primary_batch, theme)?;

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
        },
        warnings,
    ))
}
