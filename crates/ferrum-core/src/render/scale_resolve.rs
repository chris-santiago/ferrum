//! Build ResolvedScales from a ChartSpec + a post-transform RecordBatch.
//! Phase 7 supports: LinearScale, OrdinalScale, TimeScale on x/y;
//! CategoricalColorScale on color.
//! Phase 8a adds: LogScale, SymlogScale (via explicit ScaleSpec override);
//! SizeScale, ShapeScale, OpacityScale for new encoding channels.

use std::collections::HashMap;

use arrow::array::Array;
use arrow::datatypes::DataType as ArrowDataType;
use arrow::record_batch::RecordBatch;

use crate::layout::ThemeInputs;
use crate::scale::linear::LinearScale;
use crate::scale::log::LogScale;
use crate::scale::ordinal::OrdinalScale;
use crate::scale::symlog::SymlogScale;
use crate::scale::time::TimeScale;
use crate::spec::chart::ChartSpec;
use crate::spec::encoding::DataType as SpecDataType;

use super::color::Color;
use super::palette;
use super::RenderError;

/// Sealed-enum wrapper over Phase 4 scales, used during render.
/// Phase 7: Linear/Ordinal/Time. Phase 8a adds: Log, Symlog.
#[derive(Debug, Clone)]
pub enum ScaleKind {
    Linear(LinearScale),
    Ordinal(OrdinalScale),
    Time(TimeScale),
    Log(LogScale),
    Symlog(SymlogScale),
}

/// Dispatch a method call to all five `ScaleKind` variants.
macro_rules! dispatch_all {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            ScaleKind::Linear(s) => s.$method($($arg),*),
            ScaleKind::Ordinal(s) => s.$method($($arg),*),
            ScaleKind::Time(s) => s.$method($($arg),*),
            ScaleKind::Log(s) => s.$method($($arg),*),
            ScaleKind::Symlog(s) => s.$method($($arg),*),
        }
    };
}

/// Dispatch a method call to the four continuous `ScaleKind` variants
/// (Linear, Time, Log, Symlog). Ordinal is excluded — callers must
/// handle it separately.
macro_rules! dispatch_continuous {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            ScaleKind::Linear(s) => s.$method($($arg),*),
            ScaleKind::Time(s) => s.$method($($arg),*),
            ScaleKind::Log(s) => s.$method($($arg),*),
            ScaleKind::Symlog(s) => s.$method($($arg),*),
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
        }
    }

    /// Pixel-range used when constructing this scale (lo, hi).
    pub fn pixel_range(&self) -> (f64, f64) {
        let r = dispatch_all!(self, range_pair);
        (r[0], r[1])
    }
}

#[derive(Debug, Clone)]
pub enum ColorScale {
    Categorical {
        domain: Vec<String>,
        palette: &'static [Color],
    },
    /// Continuous color scale: maps a numeric value to a color via a
    /// ContinuousScheme. Used by heatmap, raster, and any chart with an
    /// explicit linear color scale spec.
    Continuous {
        domain: (f64, f64),
        scheme: crate::render::color::ContinuousScheme,
    },
}

impl ColorScale {
    pub fn lookup(&self, value: &str) -> Option<Color> {
        match self {
            Self::Categorical { domain, palette } => domain
                .iter()
                .position(|v| v == value)
                .map(|i| palette[i % palette.len()]),
            Self::Continuous { domain, scheme } => {
                let v: f64 = value.parse().ok()?;
                let (lo, hi) = *domain;
                let t = if hi > lo { (v - lo) / (hi - lo) } else { 0.5 };
                Some(scheme.sample(t.clamp(0.0, 1.0)))
            }
        }
    }

    /// Sample at numeric value (Continuous variant only). Returns None for
    /// Categorical scales.
    pub fn lookup_f64(&self, value: f64) -> Option<Color> {
        match self {
            Self::Continuous { domain, scheme } => {
                let (lo, hi) = *domain;
                let t = if hi > lo { (value - lo) / (hi - lo) } else { 0.5 };
                Some(scheme.sample(t.clamp(0.0, 1.0)))
            }
            _ => None,
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

/// An ordinal shape scale: maps a categorical field to one of 6 shapes.
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

/// The 6 point shapes available to the shape scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeKind {
    Circle,
    Square,
    Cross,
    Diamond,
    TriangleUp,
    TriangleDown,
}

/// Fixed 6-shape palette used by `build_shape_scale`.
pub const SHAPE_PALETTE: [ShapeKind; 6] = [
    ShapeKind::Circle,
    ShapeKind::Square,
    ShapeKind::Cross,
    ShapeKind::Diamond,
    ShapeKind::TriangleUp,
    ShapeKind::TriangleDown,
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
    let x = build_axis_scale("x", x_enc, x2_enc, primary_batch, transform_outputs, x_pixel_range, spec)?;

    // Stack-aware y-axis: resolve against the post-Stack batch when the
    // spec carries a matching Stack adjustment. See
    // `position::axis_batch_for_y` for the rationale.
    let y_batch = crate::render::position::axis_batch_for_y(spec, &y_enc.field, primary_batch);
    let y = build_axis_scale("y", y_enc, y2_enc, &y_batch, transform_outputs, y_pixel_range, spec)?;

    // Color/size/shape/opacity scales are primary-batch only. These channels
    // do not currently participate in cross-layer scale unification: each is
    // resolved against the chart-level transformed batch (i.e. __final__),
    // matching Phase 8a behavior. (build_color_scale is the one exception —
    // it accepts transform_outputs because composite-mark color fields may
    // live in a named output rather than primary.)
    let (color, color_warn) = build_color_scale(
        &spec.encoding, primary_batch, transform_outputs, theme,
    )?;
    if let Some(w) = color_warn {
        warnings.push(w);
    }

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

/// Themes-T4 default inward padding fraction for quantitative / temporal
/// scales when neither `scale.padding` nor `scale.domain` is set.
const DEFAULT_SCALE_PADDING_FRAC: f64 = 0.05;

/// Themes-T4 maximum inward padding in pixels. The padding band is the
/// smaller of `fraction * span` and this cap so small panels don't lose
/// too much plot area.
const SCALE_PADDING_MAX_PX: f64 = 8.0;

/// Resolve the effective padding fraction.
///
/// Precedence:
///  - `Some(p)` → use `p` (including 0.0 to disable).
///  - `None && has_explicit_domain` → 0.0 (user-specified domain wins).
///  - `None && !has_explicit_domain` → `DEFAULT_SCALE_PADDING_FRAC`.
fn resolve_padding_fraction(scale_padding: Option<f64>, has_explicit_domain: bool) -> f64 {
    if let Some(p) = scale_padding {
        return p;
    }
    if has_explicit_domain {
        return 0.0;
    }
    DEFAULT_SCALE_PADDING_FRAC
}

/// Inset a pixel range by the resolved padding band (capped at
/// `SCALE_PADDING_MAX_PX`). Handles inverted ranges (y-axis is
/// `(high_px, low_px)` for quantitative scales).
fn inset_pixel_range(pr: (f64, f64), padding_frac: f64) -> (f64, f64) {
    if padding_frac == 0.0 {
        return pr;
    }
    let span = (pr.1 - pr.0).abs();
    let pad = (span * padding_frac).min(SCALE_PADDING_MAX_PX);
    let sign = if pr.1 >= pr.0 { 1.0 } else { -1.0 };
    (pr.0 + sign * pad, pr.1 - sign * pad)
}

fn build_axis_scale(
    channel: &'static str,
    enc: &crate::spec::encoding::EncodingSpec,
    paired_enc: Option<&crate::spec::encoding::EncodingSpec>,
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    pixel_range: (f64, f64),
    spec: &ChartSpec,
) -> Result<ScaleKind, RenderError> {
    // Phase 8b: composite-mark layers (boxplot/errorbar/etc.) read from named
    // transform outputs whose schemas differ from `__final__`. The primary
    // batch may not even contain `enc.field` — pick any batch that does.
    let located = locate_field(&enc.field, primary_batch, transform_outputs)
        .ok_or_else(|| RenderError::UnknownColumn { name: enc.field.clone() })?;
    let dtype = infer_spec_type(enc, located.col.data_type());
    let pr = axis_pixel_range(channel, &dtype, pixel_range);

    // If an explicit ScaleSpec is present, honor it (Phase 8a).
    if let Some(scale_spec) = &enc.scale {
        return build_from_scale_spec(scale_spec, enc, located.batch, pr);
    }

    // No explicit scale spec → quantitative/temporal get the T4 default
    // 5% inward padding (capped at 8px); ordinal/nominal use band-side
    // half-step padding internally and are unaffected here.
    let inset = inset_pixel_range(pr, resolve_padding_fraction(None, false));

    match dtype {
        SpecDataType::Quantitative => {
            let (min, max) = numeric_domain_union(
                channel, &enc.field, paired_enc.map(|p| p.field.as_str()),
                primary_batch, transform_outputs, spec,
            )?;
            Ok(ScaleKind::Linear(LinearScale::new_internal(
                vec![min, max], vec![inset.0, inset.1], false, false,
            )))
        }
        SpecDataType::Ordinal | SpecDataType::Nominal => {
            let mut domain = distinct_values_in_order(located.batch, &enc.field)?;
            apply_sort_to_domain(&mut domain, enc.sort.as_ref());
            Ok(ScaleKind::Ordinal(OrdinalScale::new_internal(
                domain, vec![pr.0, pr.1], 0.0,
            )))
        }
        SpecDataType::Temporal => {
            let (min, max) = numeric_domain_union(
                channel, &enc.field, paired_enc.map(|p| p.field.as_str()),
                primary_batch, transform_outputs, spec,
            )?;
            Ok(ScaleKind::Time(TimeScale::new_internal(
                vec![min, max], vec![inset.0, inset.1], false, false,
            )))
        }
    }
}

/// Apply `encoding.sort` to an ordinal domain in place.
///
/// Accepted forms (mirrors the Vega-Lite `sort` field):
/// - `"ascending"` — sort alphabetically ascending (locale-independent byte order).
/// - `"descending"` — sort alphabetically descending.
/// - JSON array of strings — replace domain with that explicit order. Values not
///   present in the original domain are silently ignored; values in the original
///   domain that are absent from the array are appended at the end in their
///   original relative order so no data disappears from the scale.
/// - Absent or any other JSON value — no-op (preserves insertion order).
fn apply_sort_to_domain(domain: &mut Vec<String>, sort: Option<&serde_json::Value>) {
    match sort {
        None => {}
        Some(serde_json::Value::String(s)) if s == "ascending" => {
            domain.sort_unstable();
        }
        Some(serde_json::Value::String(s)) if s == "descending" => {
            domain.sort_unstable_by(|a, b| b.cmp(a));
        }
        Some(serde_json::Value::Array(arr)) => {
            let explicit: Vec<String> = arr
                .iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect();
            // Keep only values that appear in the domain, in explicit order.
            // Then append any domain values not covered by the explicit list.
            let mut reordered: Vec<String> = explicit
                .iter()
                .filter(|v| domain.contains(v))
                .cloned()
                .collect();
            for v in domain.iter() {
                if !explicit.contains(v) {
                    reordered.push(v.clone());
                }
            }
            *domain = reordered;
        }
        _ => {} // Unknown sort spec — no-op.
    }
}

/// Y-axis pixel range is inverted ONLY for quantitative/temporal scales —
/// Cartesian convention: low data value → bottom of plot. Ordinal/nominal
/// y-axes keep the natural top-down order so heatmaps, confusion matrices,
/// and other categorical-row charts render with first row at the top.
fn axis_pixel_range(channel: &str, dtype: &SpecDataType, pixel_range: (f64, f64)) -> (f64, f64) {
    if channel == "y" && !matches!(dtype, SpecDataType::Ordinal | SpecDataType::Nominal) {
        (pixel_range.1, pixel_range.0)
    } else {
        pixel_range
    }
}

/// Compute the unioned numeric/temporal extent for an axis field across
/// the relevant batches.
///
/// Extent rule:
///   - If `primary_batch` contains the field, use it as the starting
///     extent (preserves single-batch / faceted-panel semantics —
///     `FINAL_OUTPUT_KEY` is NOT unioned, so per-panel scales remain
///     independent when nothing else references a named output).
///   - Additionally union the field's extent across every named output
///     that some layer references via `data_source`. Required when a
///     layer's `data_source` points at a named transform whose output
///     extends past the primary batch — e.g. `ReferenceLine` for the
///     y=x diagonal in `calibration_chart`, whose endpoints [0, 1] must
///     be reachable even when the primary calibration curve sits inside
///     (0.05, 0.95).
///   - When the field is absent from `primary_batch`, fall back to
///     unioning across all named outputs that contain it (Phase 8b
///     composite-mark rule — e.g. boxplot whisker fields living in the
///     `box` named output).
///   - The paired field (x2/y2) follows the same lookup discipline.
fn numeric_domain_union(
    channel: &str,
    field: &str,
    paired_field: Option<&str>,
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    spec: &ChartSpec,
) -> Result<(f64, f64), RenderError> {
    let layer_data_sources: std::collections::HashSet<&str> = match &spec.layers {
        Some(layers) => layers.iter().filter_map(|l| l.data_source.as_deref()).collect(),
        None => std::collections::HashSet::new(),
    };
    let (mut mn, mut mx) = (f64::INFINITY, f64::NEG_INFINITY);
    let mut accumulate = |c: &dyn Array, source_field: &str| -> Result<(), RenderError> {
        let (a, b) = column_min_max_f64(c).map_err(|_| RenderError::UnsupportedDtype {
            field: source_field.to_string(),
            dtype: format!("{:?}", c.data_type()),
            context: None,
        })?;
        if a < mn { mn = a; }
        if b > mx { mx = b; }
        Ok(())
    };

    let mut union_field = |f: &str| -> Result<(), RenderError> {
        let primary_has = primary_batch.column_by_name(f).is_some();
        if let Some(c) = primary_batch.column_by_name(f) {
            accumulate(c.as_ref(), f)?;
        }
        for (key, batch) in transform_outputs.iter() {
            let key_is_referenced = layer_data_sources.contains(key.as_str());
            if !primary_has || key_is_referenced {
                if let Some(c) = batch.column_by_name(f) {
                    accumulate(c.as_ref(), f)?;
                }
            }
        }
        Ok(())
    };

    union_field(field)?;
    if let Some(p) = paired_field {
        union_field(p)?;
    }

    if !mn.is_finite() || !mx.is_finite() {
        return Err(RenderError::EmptyDomain {
            channel: channel.to_string(),
            field: field.to_string(),
        });
    }
    Ok((mn, mx))
}

/// Result of looking up a field across the primary batch and named
/// transform outputs. Carries both the source batch and the resolved
/// column so callers don't have to re-`column_by_name(...).expect(...)`
/// after the lookup — the "field is present in this batch" invariant
/// lives in the type, not in a comment.
pub(crate) struct LocatedColumn<'a> {
    pub(crate) batch: &'a RecordBatch,
    pub(crate) col: &'a dyn Array,
}

/// Pick the batch whose schema contains `field` and return both the batch
/// and the resolved column. Prefer `primary_batch` for back-compat
/// single-batch behavior; fall back to any named output (iteration order
/// is HashMap-undefined but only matters when the field appears in
/// multiple named outputs and not in primary, which is unusual).
fn locate_field<'a>(
    field: &str,
    primary_batch: &'a RecordBatch,
    transform_outputs: &'a HashMap<String, RecordBatch>,
) -> Option<LocatedColumn<'a>> {
    if let Some(c) = primary_batch.column_by_name(field) {
        return Some(LocatedColumn { batch: primary_batch, col: c.as_ref() });
    }
    for batch in transform_outputs.values() {
        if let Some(c) = batch.column_by_name(field) {
            return Some(LocatedColumn { batch, col: c.as_ref() });
        }
    }
    None
}

/// Build a ScaleKind from an explicit ScaleSpec, using the given pixel range.
fn build_from_scale_spec(
    scale_spec: &crate::spec::encoding::ScaleSpec,
    enc: &crate::spec::encoding::EncodingSpec,
    batch: &RecordBatch,
    pr: (f64, f64),
) -> Result<ScaleKind, RenderError> {
    use crate::spec::encoding::ScaleSpec;

    let col = batch
        .column_by_name(&enc.field)
        .ok_or_else(|| RenderError::UnknownColumn { name: enc.field.clone() })?;

    Ok(match scale_spec {
        ScaleSpec::Linear { domain, range, nice, clamp, padding, zero } => {
            let (mut d, r) = resolve_continuous_domain_and_range(domain, range, *padding, col.as_ref(), &enc.field, pr)?;
            if *zero && d.len() == 2 {
                if d[0] > 0.0 { d[0] = 0.0; }
                if d[1] < 0.0 { d[1] = 0.0; }
            }
            ScaleKind::Linear(LinearScale::new_internal(d, r, *clamp, *nice))
        }
        ScaleSpec::Log { base, domain, range, nice, clamp, padding } => {
            let (d, r) = resolve_continuous_domain_and_range(domain, range, *padding, col.as_ref(), &enc.field, pr)?;
            ScaleKind::Log(LogScale::new_internal(d, r, *base, *clamp, *nice))
        }
        ScaleSpec::Time { domain, range, nice, clamp, padding } => {
            let (d, r) = resolve_continuous_domain_and_range(domain, range, *padding, col.as_ref(), &enc.field, pr)?;
            ScaleKind::Time(TimeScale::new_internal(d, r, *clamp, *nice))
        }
        ScaleSpec::Symlog { constant, domain, range, nice, clamp, padding } => {
            let (d, r) = resolve_continuous_domain_and_range(domain, range, *padding, col.as_ref(), &enc.field, pr)?;
            ScaleKind::Symlog(SymlogScale::new_internal(d, r, *constant, *clamp, *nice))
        }
        ScaleSpec::Ordinal { domain, range, padding } => {
            let mut d = match domain {
                Some(d) => d.clone(),
                None => distinct_values_in_order(batch, &enc.field)?,
            };
            apply_sort_to_domain(&mut d, enc.sort.as_ref());
            // Ordinal scales use their own internal half-step padding;
            // the T4 quantitative inset does NOT apply here.
            ScaleKind::Ordinal(OrdinalScale::new_internal(
                d,
                range.clone().unwrap_or_else(|| vec![pr.0, pr.1]),
                *padding,
            ))
        }
    })
}

/// Resolve `(domain, range)` for a continuous ScaleSpec variant (Linear /
/// Log / Time / Symlog — every continuous scale shares this prologue).
///
/// Themes-T4 padding precedence: if the user supplied `padding`, honor it
/// (including 0.0 to disable). Else if `domain` is explicit, suppress
/// padding to 0.0 (user-specified domain wins). Else fall back to the
/// 5% default. An explicit `range` overrides padding entirely — `range`
/// is treated as the final pixel band.
fn resolve_continuous_domain_and_range(
    domain: &Option<Vec<f64>>,
    range: &Option<Vec<f64>>,
    padding: Option<f64>,
    col: &dyn Array,
    field: &str,
    pr: (f64, f64),
) -> Result<(Vec<f64>, Vec<f64>), RenderError> {
    let d = match domain {
        Some(d) => d.clone(),
        None => {
            let (mn, mx) = column_min_max_f64(col).map_err(|_| RenderError::UnsupportedDtype {
                field: field.to_string(),
                dtype: format!("{:?}", col.data_type()),
                context: Some("scale"),
            })?;
            vec![mn, mx]
        }
    };
    let r = if let Some(r) = range {
        r.clone()
    } else {
        let frac = resolve_padding_fraction(padding, domain.is_some());
        let inset = inset_pixel_range(pr, frac);
        vec![inset.0, inset.1]
    };
    Ok((d, r))
}

/// Build a SizeScale if `encoding.size` is present.
///
/// Honors a user-supplied `scale.range` (Phase 10f); when absent, falls back
/// to `[theme.point_size_min, theme.point_size_max]`. This lets diagnostic
/// marks (intercluster_distance, future bubble charts) request larger point
/// sizes per the spec without modifying the global theme.
/// Resolve the color encoding into a `ColorScale`.
///
/// Routes to `Continuous` when the field's Arrow dtype is `Float64` or
/// `UInt64`, else `Categorical`. (F16 will widen the numeric detection
/// to consult `EncodingSpec.type_` first and treat all numeric dtypes
/// as continuous; today's narrow check is preserved.)
///
/// Color is the one secondary channel that may live outside the primary
/// batch — composite-mark color fields can resolve via `transform_outputs`
/// — so this function accepts the named-outputs map. Size/shape/opacity
/// builders below are primary-batch-only by design.
///
/// Returns `(scale, optional palette-overflow warning)` so the caller
/// can fold the warning into its accumulator alongside the
/// build_shape_scale warning.
pub fn build_color_scale(
    encoding: &crate::spec::encoding::Encoding,
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    theme: &ThemeInputs,
) -> Result<(Option<ColorScale>, Option<crate::render::RenderWarning>), RenderError> {
    let Some(c_enc) = &encoding.color else {
        return Ok((None, None));
    };
    let located = locate_field(&c_enc.field, primary_batch, transform_outputs)
        .ok_or_else(|| RenderError::UnknownColumn { name: c_enc.field.clone() })?;

    // F16 — Color type inference policy.
    //
    // Pre-F16 the continuous-vs-categorical decision was a narrow dtype
    // check: `matches!(dtype, Float64 | UInt64)`. Every other numeric
    // dtype (Float32, Int8/16/32/64, UInt8/16/32) silently fell into the
    // categorical branch — a latent bug that violated ferrum-spec.md §3
    // line 52 ("no magic inference that silently fails").
    //
    // The new policy mirrors `infer_spec_type` (used by axis scales):
    //   - `EncodingSpec.type_` wins when explicitly set.
    //   - Otherwise infer from Arrow dtype: numeric/temporal → continuous,
    //     string/boolean → categorical.
    //
    // Conflict path: `type_ = Quantitative` on a non-numeric column
    // becomes EncodingTypeMismatch rather than silently routing through
    // numeric_extent's (0.0, 1.0) fallback.
    let inferred = infer_spec_type(c_enc, located.col.data_type());
    let is_continuous_color = matches!(
        inferred,
        SpecDataType::Quantitative | SpecDataType::Temporal,
    );
    if is_continuous_color {
        if !crate::render::arrow_cast::is_numeric(located.col.data_type()) {
            return Err(RenderError::EncodingTypeMismatch {
                channel: "color",
                expected: "numeric column for quantitative/temporal type",
                got: format!("{:?}", located.col.data_type()),
            });
        }
        // Numeric domain: min/max from the column, ignoring NaNs.
        let (lo, hi) = numeric_extent(located.col);
        // Scheme: prefer encoding.scheme (set by heatmap's `cmap=` arg),
        // else auto-detect diverging (domain spans negative to positive) →
        // theme.diverging_scheme, else fall back to theme.sequential_scheme.
        use crate::render::color::{ContinuousScheme, NamedContinuous};
        let scheme = c_enc
            .scheme
            .as_deref()
            .and_then(NamedContinuous::from_name)
            .map(ContinuousScheme::Named)
            .unwrap_or_else(|| {
                // D15: auto-diverging detection — when the domain spans
                // both negative and positive values, use the theme's
                // diverging scheme rather than the sequential one.
                let theme_scheme = if lo < 0.0 && hi > 0.0 {
                    &theme.diverging_scheme
                } else {
                    &theme.sequential_scheme
                };
                NamedContinuous::from_name(theme_scheme)
                    .map(ContinuousScheme::Named)
                    .unwrap_or(ContinuousScheme::Named(NamedContinuous::Viridis))
            });
        Ok((Some(ColorScale::Continuous { domain: (lo, hi), scheme }), None))
    } else {
        let domain = distinct_values_in_order(primary_batch, &c_enc.field)?;
        // Precedence: encoding.scheme (per-encoding override, e.g. heatmap
        // cmap=) → theme.color_scheme (Theme default) → OKABE_ITO fallback.
        // A sequential scheme name (viridis/plasma/…) on a nominal encoding
        // can't be interpolated for n categories yet, so we substitute
        // tableau10 — the canonical Vega-Lite categorical default — rather
        // than collapsing to OKABE_ITO silently.
        let resolved_name: &str = c_enc.scheme.as_deref().unwrap_or(&theme.color_scheme);
        let palette: &'static [Color] = if palette::is_sequential_scheme(resolved_name) {
            palette::categorical_palette("tableau10")
        } else {
            palette::categorical_palette(resolved_name)
        };
        let warn = (domain.len() > palette.len()).then(|| {
            crate::render::RenderWarning::ColorPaletteOverflowed { categories: domain.len() as u32 }
        });
        Ok((Some(ColorScale::Categorical { domain, palette }), warn))
    }
}

pub fn build_size_scale(
    encoding: &crate::spec::encoding::Encoding,
    batch: &RecordBatch,
    theme: &ThemeInputs,
) -> Result<Option<SizeScale>, RenderError> {
    let Some(size_enc) = &encoding.size else {
        return Ok(None);
    };
    let col = batch
        .column_by_name(&size_enc.field)
        .ok_or_else(|| RenderError::UnknownColumn { name: size_enc.field.clone() })?;
    let (min, max) = column_min_max_f64(col).map_err(|_| RenderError::UnsupportedDtype {
        field: size_enc.field.clone(),
        dtype: format!("{:?}", col.data_type()),
        context: Some("size"),
    })?;
    let (lo, hi) = if let Some(crate::spec::encoding::ScaleSpec::Linear { range, .. })
        = &size_enc.scale
    {
        if let Some(r) = range {
            if r.len() == 2 {
                (r[0], r[1])
            } else {
                (theme.point_size_min, theme.point_size_max)
            }
        } else {
            (theme.point_size_min, theme.point_size_max)
        }
    } else {
        (theme.point_size_min, theme.point_size_max)
    };
    let inner = ScaleKind::Linear(LinearScale::new_internal(
        vec![min, max],
        vec![lo, hi],
        false,
        true,
    ));
    let _ = (lo, hi); // bounds now read from inner.pixel_range() via accessors
    Ok(Some(SizeScale { inner }))
}

/// Build a ShapeScale if `encoding.shape` is present.
/// Returns the scale (if built) and an optional overflow warning.
pub fn build_shape_scale(
    encoding: &crate::spec::encoding::Encoding,
    batch: &RecordBatch,
) -> Result<(Option<ShapeScale>, Option<crate::render::RenderWarning>), RenderError> {
    let Some(shape_enc) = &encoding.shape else {
        return Ok((None, None));
    };
    let distinct = distinct_values_in_order(batch, &shape_enc.field)?;
    let warn = if distinct.len() > SHAPE_PALETTE.len() {
        Some(crate::render::RenderWarning::ShapePaletteOverflowed {
            categories: distinct.len() as u32,
        })
    } else {
        None
    };
    let shapes: Vec<ShapeKind> = (0..distinct.len())
        .map(|i| SHAPE_PALETTE[i % SHAPE_PALETTE.len()])
        .collect();
    Ok((Some(ShapeScale { domain: distinct, shapes }), warn))
}

/// Build an OpacityScale if `encoding.opacity` is present.
pub fn build_opacity_scale(
    encoding: &crate::spec::encoding::Encoding,
    batch: &RecordBatch,
    theme: &ThemeInputs,
) -> Result<Option<OpacityScale>, RenderError> {
    let Some(op_enc) = &encoding.opacity else {
        return Ok(None);
    };
    let col = batch
        .column_by_name(&op_enc.field)
        .ok_or_else(|| RenderError::UnknownColumn { name: op_enc.field.clone() })?;
    let (min, max) = column_min_max_f64(col).map_err(|_| RenderError::UnsupportedDtype {
        field: op_enc.field.clone(),
        dtype: format!("{:?}", col.data_type()),
        context: Some("opacity"),
    })?;
    let inner = ScaleKind::Linear(LinearScale::new_internal(
        vec![min, max],
        vec![theme.opacity_min, theme.opacity_max],
        true,
        false,
    ));
    Ok(Some(OpacityScale { inner }))
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{Field, Schema};
    use std::sync::Arc;

    fn make_batch_q_q_n() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", ArrowDataType::Float64, false),
            Field::new("y", ArrowDataType::Float64, false),
            Field::new("species", ArrowDataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0])),
                Arc::new(StringArray::from(vec!["a", "b", "a", "c", "b", "c"])),
            ],
        )
        .unwrap()
    }

    fn make_spec_with_color() -> ChartSpec {
        use crate::spec::data_ref::DataRef;
        use crate::spec::encoding::{Encoding, EncodingSpec};
        use crate::spec::mark::Mark;
        ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: Some(EncodingSpec { field: "species".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        }
    }

    #[test]
    fn quantitative_x_resolves_to_linear() {
        let s = make_spec_with_color();
        let b = make_batch_q_q_n();
        let theme = ThemeInputs::default();
        let (scales, warnings) = resolve_scales(&s, &b, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
        assert!(matches!(scales.x, ScaleKind::Linear(_)));
        assert!(matches!(scales.y, ScaleKind::Linear(_)));
        assert!(warnings.is_empty());
    }

    #[test]
    fn color_encoding_builds_categorical_in_encounter_order() {
        let s = make_spec_with_color();
        let b = make_batch_q_q_n();
        let theme = ThemeInputs::default();
        let (scales, _) = resolve_scales(&s, &b, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
        let cs = scales.color.unwrap();
        match cs {
            ColorScale::Categorical { domain, .. } => {
                assert_eq!(domain, vec!["a".to_string(), "b".to_string(), "c".to_string()]);
            }
            ColorScale::Continuous { .. } => panic!("expected Categorical, got Continuous"),
        }
    }

    #[test]
    fn unknown_x_column_errors() {
        let mut s = make_spec_with_color();
        s.encoding.x = Some(crate::spec::encoding::EncodingSpec {
            field: "missing".into(),
            type_: None,
            ..Default::default()
        });
        let b = make_batch_q_q_n();
        let theme = ThemeInputs::default();
        let err = resolve_scales(&s, &b, (0.0, 100.0), (0.0, 80.0), &theme).unwrap_err();
        assert!(matches!(err, RenderError::UnknownColumn { .. }));
    }

    #[test]
    fn color_overflow_emits_warning() {
        use crate::spec::data_ref::DataRef;
        use crate::spec::encoding::{Encoding, EncodingSpec};
        use crate::spec::mark::Mark;
        // Themes-T4: default palette is tableau10 (10 colors); 11+ categories
        // are needed to trigger ColorPaletteOverflowed (was 9+ under OKABE_ITO).
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", ArrowDataType::Float64, false),
            Field::new("y", ArrowDataType::Float64, false),
            Field::new("g", ArrowDataType::Utf8, false),
        ]));
        let groups: Vec<String> = (0..11).map(|i| format!("g{i}")).collect();
        let groups_str: Vec<&str> = groups.iter().map(String::as_str).collect();
        let xs: Vec<f64> = (0..11).map(|i| i as f64).collect();
        let ys = xs.clone();
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(xs)),
                Arc::new(Float64Array::from(ys)),
                Arc::new(StringArray::from(groups_str)),
            ],
        )
        .unwrap();
        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: Some(EncodingSpec { field: "g".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        };
        let theme = ThemeInputs::default();
        let (_, warnings) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
        assert!(matches!(
            warnings[0],
            crate::render::RenderWarning::ColorPaletteOverflowed { categories: 11 }
        ));
    }

    // --- Phase 8a new tests ---

    fn make_batch_q_q_n_n_q() -> RecordBatch {
        // x, y (quantitative), species (nominal), size_val (quantitative), opacity_val (quantitative)
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", ArrowDataType::Float64, false),
            Field::new("y", ArrowDataType::Float64, false),
            Field::new("species", ArrowDataType::Utf8, false),
            Field::new("size_val", ArrowDataType::Float64, false),
            Field::new("opacity_val", ArrowDataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
                Arc::new(StringArray::from(vec!["cat", "dog", "bird"])),
                Arc::new(Float64Array::from(vec![1.0, 5.0, 10.0])),
                Arc::new(Float64Array::from(vec![0.2, 0.5, 0.9])),
            ],
        )
        .unwrap()
    }

    fn make_spec_with_size() -> ChartSpec {
        use crate::spec::data_ref::DataRef;
        use crate::spec::encoding::{Encoding, EncodingSpec};
        use crate::spec::mark::Mark;
        ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                size: Some(EncodingSpec { field: "size_val".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        }
    }

    fn make_spec_with_shape() -> ChartSpec {
        use crate::spec::data_ref::DataRef;
        use crate::spec::encoding::{Encoding, EncodingSpec};
        use crate::spec::mark::Mark;
        ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                shape: Some(EncodingSpec { field: "species".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        }
    }

    fn make_spec_with_opacity() -> ChartSpec {
        use crate::spec::data_ref::DataRef;
        use crate::spec::encoding::{Encoding, EncodingSpec};
        use crate::spec::mark::Mark;
        ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                opacity: Some(EncodingSpec { field: "opacity_val".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        }
    }

    #[test]
    fn explicit_log_scale_overrides_auto_detection() {
        use crate::spec::encoding::ScaleSpec;
        let mut s = make_spec_with_color();
        s.encoding.x.as_mut().unwrap().scale = Some(ScaleSpec::Log {
            base: 10.0,
            domain: Some(vec![1.0, 1000.0]),
            range: None,
            nice: false,
            clamp: false,
            padding: None,
        });
        let b = make_batch_q_q_n();
        let theme = ThemeInputs::default();
        let (scales, _) = resolve_scales(&s, &b, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
        assert!(matches!(scales.x, ScaleKind::Log(_)));
    }

    #[test]
    fn size_scale_defaults_to_theme_point_size_range() {
        // Themes-T4: point_size_min/max defaults flipped 3.0/30.0 → 4.0/36.0.
        let batch = make_batch_q_q_n_n_q();
        let theme = ThemeInputs::default();
        let scale = build_size_scale(&make_spec_with_size().encoding, &batch, &theme)
            .unwrap()
            .unwrap();
        assert_eq!(scale.min_px(), 4.0);
        assert_eq!(scale.max_px(), 36.0);
    }

    #[test]
    fn shape_scale_picks_from_6_shape_palette_in_order() {
        let batch = make_batch_q_q_n_n_q();
        let (scale, warn) = build_shape_scale(&make_spec_with_shape().encoding, &batch).unwrap();
        let scale = scale.unwrap();
        assert!(warn.is_none()); // 3 categories, no overflow
        assert_eq!(scale.shapes.len(), 3);
        assert_eq!(scale.shapes[0], ShapeKind::Circle);
        assert_eq!(scale.shapes[1], ShapeKind::Square);
        assert_eq!(scale.shapes[2], ShapeKind::Cross);
    }

    // --- Phase 8b: y2/x2 extends the primary axis domain ---

    #[test]
    fn y2_field_extends_y_domain_when_set() {
        use crate::spec::data_ref::DataRef;
        use crate::spec::encoding::{Encoding, EncodingSpec};
        use crate::spec::mark::Mark;
        // y in [0, 2], y2 in [1, 3]; without the fix the y domain would top out
        // at 2.0 and y2=3.0 would scale to a non-finite/out-of-range pixel.
        let schema = Arc::new(Schema::new(vec![
            Field::new("t", ArrowDataType::Float64, false),
            Field::new("lo", ArrowDataType::Float64, false),
            Field::new("hi", ArrowDataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
                Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            ],
        )
        .unwrap();
        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Ribbon,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "t".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "lo".into(), type_: None, ..Default::default() }),
                y2: Some(EncodingSpec { field: "hi".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        };
        let theme = ThemeInputs::default();
        let (scales, _) =
            resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
        // The maximum y2 value (3.0) must scale to a finite pixel, proving y2
        // was included when computing the y-axis data extent.
        let pixel_at_y2_max = scales.y.to_pixel_f64(3.0).expect("linear y returns Some");
        assert!(
            pixel_at_y2_max.is_finite(),
            "y-axis pixel for max y2 must be finite, got: {pixel_at_y2_max}"
        );
        // Sanity: pixel for y2=3.0 must lie inside the requested y pixel range
        // [0.0, 80.0] (Y inverts so the bound is loosely [0, 80]).
        assert!(
            (0.0..=80.0).contains(&pixel_at_y2_max),
            "y2 max should map within the y pixel range, got: {pixel_at_y2_max}"
        );
    }

    #[test]
    fn x2_field_extends_x_domain_when_set() {
        use crate::spec::data_ref::DataRef;
        use crate::spec::encoding::{Encoding, EncodingSpec};
        use crate::spec::mark::Mark;
        let schema = Arc::new(Schema::new(vec![
            Field::new("xa", ArrowDataType::Float64, false),
            Field::new("xb", ArrowDataType::Float64, false),
            Field::new("y", ArrowDataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 5.0])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
            ],
        )
        .unwrap();
        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Ribbon,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "xa".into(), type_: None, ..Default::default() }),
                x2: Some(EncodingSpec { field: "xb".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        };
        let theme = ThemeInputs::default();
        let (scales, _) =
            resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
        // x2 max is 5.0 (outside [0, 2] range of x). Must map to a finite pixel
        // and lie within the requested x pixel range [0.0, 100.0].
        let px = scales.x.to_pixel_f64(5.0).expect("linear x returns Some");
        assert!(px.is_finite(), "x-axis pixel for max x2 must be finite, got: {px}");
        assert!(
            (0.0..=100.0).contains(&px),
            "x2 max should map within the x pixel range, got: {px}"
        );
    }

    // --- Phase 8b Task 36: x2/y2 field names surfaced on ResolvedScales ---

    #[test]
    fn resolved_scales_include_x2_y2_field_names_when_set() {
        use crate::spec::data_ref::DataRef;
        use crate::spec::encoding::{Encoding, EncodingSpec};
        use crate::spec::mark::Mark;
        let schema = Arc::new(Schema::new(vec![
            Field::new("a",  ArrowDataType::Float64, false),
            Field::new("a2", ArrowDataType::Float64, false),
            Field::new("b",  ArrowDataType::Float64, false),
            Field::new("b2", ArrowDataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![0.0, 1.0])),
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
                Arc::new(Float64Array::from(vec![0.0, 1.0])),
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
            ],
        )
        .unwrap();
        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Ribbon,
            encoding: Encoding {
                x:  Some(EncodingSpec { field: "a".into(),  type_: None, ..Default::default() }),
                x2: Some(EncodingSpec { field: "a2".into(), type_: None, ..Default::default() }),
                y:  Some(EncodingSpec { field: "b".into(),  type_: None, ..Default::default() }),
                y2: Some(EncodingSpec { field: "b2".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        };
        let theme = ThemeInputs::default();
        let (scales, _) =
            resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
        assert_eq!(scales.x2.as_deref(), Some("a2"));
        assert_eq!(scales.y2.as_deref(), Some("b2"));
    }

    #[test]
    fn resolved_scales_x2_y2_default_to_none_for_8a_charts() {
        use crate::spec::data_ref::DataRef;
        use crate::spec::encoding::{Encoding, EncodingSpec};
        use crate::spec::mark::Mark;
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", ArrowDataType::Float64, false),
            Field::new("b", ArrowDataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![0.0, 1.0])),
                Arc::new(Float64Array::from(vec![0.0, 1.0])),
            ],
        )
        .unwrap();
        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "a".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "b".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        };
        let theme = ThemeInputs::default();
        let (scales, _) =
            resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
        assert!(scales.x2.is_none(), "x2 should be None for 8a charts: {:?}", scales.x2);
        assert!(scales.y2.is_none(), "y2 should be None for 8a charts: {:?}", scales.y2);
    }

    #[test]
    fn opacity_scale_defaults_to_0_1_to_1_0() {
        let batch = make_batch_q_q_n_n_q();
        let theme = ThemeInputs::default();
        let scale = build_opacity_scale(&make_spec_with_opacity().encoding, &batch, &theme)
            .unwrap()
            .unwrap();
        assert_eq!(scale.min_opacity(), 0.1);
        assert_eq!(scale.max_opacity(), 1.0);
    }
}
