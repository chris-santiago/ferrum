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

impl ScaleKind {
    /// Map a quantitative or temporal value to a pixel coordinate.
    /// Returns `None` for ordinal scales (use `to_pixel_str` instead).
    pub fn to_pixel_f64(&self, x: f64) -> Option<f64> {
        match self {
            Self::Linear(s) => Some(s.scale_internal(x)),
            Self::Time(s) => Some(s.scale_internal(x)),
            Self::Log(s) => Some(s.scale_internal(x)),
            Self::Symlog(s) => Some(s.scale_internal(x)),
            Self::Ordinal(_) => None,
        }
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
        match self {
            Self::Linear(s) => {
                let r = s.range_pair();
                (r[0], r[1])
            }
            Self::Ordinal(s) => {
                let r = s.range_pair();
                (r[0], r[1])
            }
            Self::Time(s) => {
                let r = s.range_pair();
                (r[0], r[1])
            }
            Self::Log(s) => {
                let r = s.range_pair();
                (r[0], r[1])
            }
            Self::Symlog(s) => {
                let r = s.range_pair();
                (r[0], r[1])
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum ColorScale {
    Categorical {
        domain: Vec<String>,
        palette: &'static [Color],
    },
}

impl ColorScale {
    pub fn lookup(&self, value: &str) -> Option<Color> {
        match self {
            Self::Categorical { domain, palette } => domain
                .iter()
                .position(|v| v == value)
                .map(|i| palette[i % palette.len()]),
        }
    }
}

/// A linear size scale: maps a quantitative field to a radius/diameter in pixels.
#[derive(Debug, Clone)]
pub struct SizeScale {
    pub inner: ScaleKind, // typically Linear
    pub min_px: f64,      // default 3.0
    pub max_px: f64,      // default 30.0
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

/// A linear opacity scale: maps a quantitative field to [min_opacity, max_opacity].
#[derive(Debug, Clone)]
pub struct OpacityScale {
    pub inner: ScaleKind, // typically Linear
    pub min_opacity: f64, // default 0.1
    pub max_opacity: f64, // default 1.0
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
    let x = build_axis_scale("x", x_enc, x2_enc, primary_batch, transform_outputs, x_pixel_range)?;
    let y = build_axis_scale("y", y_enc, y2_enc, primary_batch, transform_outputs, y_pixel_range)?;

    // Color/size/shape/opacity scales are primary-batch only. These channels
    // do not currently participate in cross-layer scale unification: each is
    // resolved against the chart-level transformed batch (i.e. __final__),
    // matching Phase 8a behavior.
    let color = if let Some(c_enc) = &spec.encoding.color {
        let domain = distinct_values_in_order(primary_batch, &c_enc.field)?;
        let palette: &'static [Color] = match &c_enc.scheme {
            Some(name) => palette::categorical_palette(name),
            None => &*palette::OKABE_ITO,
        };
        if domain.len() > palette.len() {
            warnings.push(crate::render::RenderWarning::ColorPaletteOverflowed {
                categories: domain.len() as u32,
            });
        }
        Some(ColorScale::Categorical { domain, palette })
    } else {
        None
    };

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

fn build_axis_scale(
    channel: &'static str,
    enc: &crate::spec::encoding::EncodingSpec,
    paired_enc: Option<&crate::spec::encoding::EncodingSpec>,
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    pixel_range: (f64, f64),
) -> Result<ScaleKind, RenderError> {
    // Phase 8b: composite-mark layers (boxplot/errorbar/errorband/etc.) read
    // from named transform outputs whose schemas differ from `__final__`. The
    // primary batch may not even contain `enc.field`. Prefer the primary batch
    // when present (preserves single-batch behavior + back-compat goldens), but
    // otherwise pick any named output that does carry the field. The dtype is
    // inferred from whichever batch we ended up using to look up `enc.field`.
    //
    // For the numeric-extent computation below we additionally union across
    // every named output that carries `enc.field`; for Utf8/categorical axes,
    // the lookup batch is sufficient (composite marks preserve the categorical
    // axis field in every named output, so the primary batch suffices).
    let lookup_batch = locate_field_batch(&enc.field, primary_batch, transform_outputs)
        .ok_or_else(|| RenderError::UnknownColumn { name: enc.field.clone() })?;
    let col = lookup_batch
        .column_by_name(&enc.field)
        .expect("locate_field_batch guarantees field presence");
    let dtype = infer_spec_type(enc, col.data_type());
    // Y-axis pixel range is inverted (top of plot is min y, bottom is max y).
    let pr = if channel == "y" {
        (pixel_range.1, pixel_range.0)
    } else {
        pixel_range
    };

    // If an explicit ScaleSpec is present, honor it (Phase 8a).
    if let Some(scale_spec) = &enc.scale {
        return build_from_scale_spec(scale_spec, enc, lookup_batch, pr);
    }

    // Numeric / temporal extent.
    //
    // Back-compat rule: if `primary_batch` contains `enc.field`, the extent is
    // computed from `primary_batch` alone — single-batch and faceted-panel
    // semantics are preserved (panels keep their per-panel-filtered domain).
    //
    // Phase 8b composite-mark rule: when `primary_batch` does NOT contain the
    // field (e.g. boxplot's `lower_whisker` lives in the `box` named output,
    // not in `__final__`), union the field's extent across every batch in
    // `transform_outputs` that does contain it. The same rule applies
    // independently to the paired (x2/y2) field.
    let combined_min_max = || -> Result<(f64, f64), String> {
        let (mut mn, mut mx) = (f64::INFINITY, f64::NEG_INFINITY);
        let mut accumulate = |c: &dyn Array| -> Result<(), String> {
            let (a, b) = column_min_max_f64(c)?;
            if a < mn {
                mn = a;
            }
            if b > mx {
                mx = b;
            }
            Ok(())
        };

        // Primary field.
        if let Some(c) = primary_batch.column_by_name(&enc.field) {
            accumulate(c.as_ref())?;
        } else {
            for batch in transform_outputs.values() {
                if let Some(c) = batch.column_by_name(&enc.field) {
                    accumulate(c.as_ref())?;
                }
            }
        }
        // Paired (x2/y2) field — same lookup discipline.
        if let Some(p) = paired_enc {
            if let Some(c) = primary_batch.column_by_name(&p.field) {
                accumulate(c.as_ref())?;
            } else {
                for batch in transform_outputs.values() {
                    if let Some(c) = batch.column_by_name(&p.field) {
                        accumulate(c.as_ref())?;
                    }
                }
            }
        }

        if !mn.is_finite() || !mx.is_finite() {
            return Err(format!("no usable values found for field '{}'", enc.field));
        }
        Ok((mn, mx))
    };

    match dtype {
        SpecDataType::Quantitative => {
            let (min, max) = combined_min_max()
                .map_err(|e| RenderError::ScaleResolutionFailed(format!("{channel}: {e}")))?;
            Ok(ScaleKind::Linear(LinearScale::new_internal(
                vec![min, max],
                vec![pr.0, pr.1],
                false,
                false,
            )))
        }
        SpecDataType::Ordinal | SpecDataType::Nominal => {
            let domain = distinct_values_in_order(lookup_batch, &enc.field)?;
            Ok(ScaleKind::Ordinal(OrdinalScale::new_internal(
                domain,
                vec![pr.0, pr.1],
                0.0,
            )))
        }
        SpecDataType::Temporal => {
            let (min, max) = combined_min_max()
                .map_err(|e| RenderError::ScaleResolutionFailed(format!("{channel}: {e}")))?;
            Ok(ScaleKind::Time(TimeScale::new_internal(
                vec![min, max],
                vec![pr.0, pr.1],
                false,
                false,
            )))
        }
    }
}

/// Pick the batch whose schema contains `field`. Prefer `primary_batch` for
/// back-compat single-batch behavior; fall back to any named output (iteration
/// order is HashMap-undefined but only matters when the field appears in
/// multiple named outputs and not in primary, which is unusual). Returns None
/// if no batch contains the field.
fn locate_field_batch<'a>(
    field: &str,
    primary_batch: &'a RecordBatch,
    transform_outputs: &'a HashMap<String, RecordBatch>,
) -> Option<&'a RecordBatch> {
    if primary_batch.column_by_name(field).is_some() {
        return Some(primary_batch);
    }
    for batch in transform_outputs.values() {
        if batch.column_by_name(field).is_some() {
            return Some(batch);
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

    let default_pixel_range = vec![pr.0, pr.1];

    Ok(match scale_spec {
        ScaleSpec::Linear { domain, range, nice, clamp, .. } => {
            let d = match domain {
                Some(d) => d.clone(),
                None => {
                    let (mn, mx) = column_min_max_f64(col)
                        .map_err(|e| RenderError::ScaleResolutionFailed(e))?;
                    vec![mn, mx]
                }
            };
            ScaleKind::Linear(LinearScale::new_internal(
                d,
                range.clone().unwrap_or(default_pixel_range),
                *clamp,
                *nice,
            ))
        }
        ScaleSpec::Log { base, domain, range, nice, clamp } => {
            let d = match domain {
                Some(d) => d.clone(),
                None => {
                    let (mn, mx) = column_min_max_f64(col)
                        .map_err(|e| RenderError::ScaleResolutionFailed(e))?;
                    vec![mn, mx]
                }
            };
            ScaleKind::Log(LogScale::new_internal(
                d,
                range.clone().unwrap_or(default_pixel_range),
                *base,
                *clamp,
                *nice,
            ))
        }
        ScaleSpec::Time { domain, range, nice, clamp } => {
            let d = match domain {
                Some(d) => d.clone(),
                None => {
                    let (mn, mx) = column_min_max_f64(col)
                        .map_err(|e| RenderError::ScaleResolutionFailed(e))?;
                    vec![mn, mx]
                }
            };
            ScaleKind::Time(TimeScale::new_internal(
                d,
                range.clone().unwrap_or(default_pixel_range),
                *clamp,
                *nice,
            ))
        }
        ScaleSpec::Symlog { constant, domain, range, nice, clamp } => {
            let d = match domain {
                Some(d) => d.clone(),
                None => {
                    let (mn, mx) = column_min_max_f64(col)
                        .map_err(|e| RenderError::ScaleResolutionFailed(e))?;
                    vec![mn, mx]
                }
            };
            ScaleKind::Symlog(SymlogScale::new_internal(
                d,
                range.clone().unwrap_or(default_pixel_range),
                *constant,
                *clamp,
                *nice,
            ))
        }
        ScaleSpec::Ordinal { domain, range, padding } => {
            let d = match domain {
                Some(d) => d.clone(),
                None => distinct_values_in_order(batch, &enc.field)?,
            };
            ScaleKind::Ordinal(OrdinalScale::new_internal(
                d,
                range.clone().unwrap_or(default_pixel_range),
                *padding,
            ))
        }
    })
}

/// Build a SizeScale if `encoding.size` is present.
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
    let (min, max) = column_min_max_f64(col)
        .map_err(|e| RenderError::ScaleResolutionFailed(format!("size: {e}")))?;
    let inner = ScaleKind::Linear(LinearScale::new_internal(
        vec![min, max],
        vec![theme.point_size_min, theme.point_size_max],
        false,
        true,
    ));
    Ok(Some(SizeScale {
        inner,
        min_px: theme.point_size_min,
        max_px: theme.point_size_max,
    }))
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
    let (min, max) = column_min_max_f64(col)
        .map_err(|e| RenderError::ScaleResolutionFailed(format!("opacity: {e}")))?;
    let inner = ScaleKind::Linear(LinearScale::new_internal(
        vec![min, max],
        vec![theme.opacity_min, theme.opacity_max],
        true,
        false,
    ));
    Ok(Some(OpacityScale {
        inner,
        min_opacity: theme.opacity_min,
        max_opacity: theme.opacity_max,
    }))
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
    use arrow::array::{
        Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array,
        UInt8Array, UInt16Array, UInt32Array, UInt64Array,
        TimestampMillisecondArray,
    };
    if let Some(a) = col.as_any().downcast_ref::<Float64Array>() {
        let min = a.iter().flatten().fold(f64::INFINITY, f64::min);
        let max = a.iter().flatten().fold(f64::NEG_INFINITY, f64::max);
        Ok((min, max))
    } else if let Some(a) = col.as_any().downcast_ref::<Float32Array>() {
        let min = a.iter().flatten().fold(f32::INFINITY, f32::min) as f64;
        let max = a.iter().flatten().fold(f32::NEG_INFINITY, f32::max) as f64;
        Ok((min, max))
    } else if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
        let min = a.iter().flatten().fold(i64::MAX, i64::min) as f64;
        let max = a.iter().flatten().fold(i64::MIN, i64::max) as f64;
        Ok((min, max))
    } else if let Some(a) = col.as_any().downcast_ref::<Int32Array>() {
        let min = a.iter().flatten().fold(i32::MAX, i32::min) as f64;
        let max = a.iter().flatten().fold(i32::MIN, i32::max) as f64;
        Ok((min, max))
    } else if let Some(a) = col.as_any().downcast_ref::<Int16Array>() {
        let min = a.iter().flatten().fold(i16::MAX, i16::min) as f64;
        let max = a.iter().flatten().fold(i16::MIN, i16::max) as f64;
        Ok((min, max))
    } else if let Some(a) = col.as_any().downcast_ref::<Int8Array>() {
        let min = a.iter().flatten().fold(i8::MAX, i8::min) as f64;
        let max = a.iter().flatten().fold(i8::MIN, i8::max) as f64;
        Ok((min, max))
    } else if let Some(a) = col.as_any().downcast_ref::<UInt64Array>() {
        // Bin transform produces count as UInt64.
        let min = a.iter().flatten().fold(u64::MAX, u64::min) as f64;
        let max = a.iter().flatten().fold(u64::MIN, u64::max) as f64;
        Ok((min, max))
    } else if let Some(a) = col.as_any().downcast_ref::<UInt32Array>() {
        let min = a.iter().flatten().fold(u32::MAX, u32::min) as f64;
        let max = a.iter().flatten().fold(u32::MIN, u32::max) as f64;
        Ok((min, max))
    } else if let Some(a) = col.as_any().downcast_ref::<UInt16Array>() {
        let min = a.iter().flatten().fold(u16::MAX, u16::min) as f64;
        let max = a.iter().flatten().fold(u16::MIN, u16::max) as f64;
        Ok((min, max))
    } else if let Some(a) = col.as_any().downcast_ref::<UInt8Array>() {
        let min = a.iter().flatten().fold(u8::MAX, u8::min) as f64;
        let max = a.iter().flatten().fold(u8::MIN, u8::max) as f64;
        Ok((min, max))
    } else if let Some(a) = col.as_any().downcast_ref::<TimestampMillisecondArray>() {
        let min = a.iter().flatten().fold(i64::MAX, i64::min) as f64;
        let max = a.iter().flatten().fold(i64::MIN, i64::max) as f64;
        Ok((min, max))
    } else {
        Err(format!("unsupported column dtype: {:?}", col.data_type()))
    }
}

fn distinct_values_in_order(
    batch: &RecordBatch,
    field: &str,
) -> Result<Vec<String>, RenderError> {
    use arrow::array::{BooleanArray, Int64Array, LargeStringArray, StringArray};
    let col = batch
        .column_by_name(field)
        .ok_or_else(|| RenderError::UnknownColumn { name: field.to_string() })?;
    let mut seen = std::collections::HashSet::<String>::new();
    let mut out = Vec::<String>::new();
    let mut push = |s: String| {
        if seen.insert(s.clone()) {
            out.push(s);
        }
    };
    if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
        for v in a.iter().flatten() {
            push(v.to_string());
        }
    } else if let Some(a) = col.as_any().downcast_ref::<LargeStringArray>() {
        // Polars produces LargeUtf8 (LargeStringArray) for string columns.
        for v in a.iter().flatten() {
            push(v.to_string());
        }
    } else if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
        for v in a.iter().flatten() {
            push(v.to_string());
        }
    } else if let Some(a) = col.as_any().downcast_ref::<BooleanArray>() {
        for v in a.iter().flatten() {
            push(v.to_string());
        }
    } else {
        return Err(RenderError::ScaleResolutionFailed(format!(
            "can't enumerate distinct values from column dtype {:?}",
            col.data_type()
        )));
    }
    Ok(out)
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
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", ArrowDataType::Float64, false),
            Field::new("y", ArrowDataType::Float64, false),
            Field::new("g", ArrowDataType::Utf8, false),
        ]));
        let groups: Vec<String> = (0..10).map(|i| format!("g{i}")).collect();
        let groups_str: Vec<&str> = groups.iter().map(String::as_str).collect();
        let xs: Vec<f64> = (0..10).map(|i| i as f64).collect();
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
        };
        let theme = ThemeInputs::default();
        let (_, warnings) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
        assert!(matches!(
            warnings[0],
            crate::render::RenderWarning::ColorPaletteOverflowed { categories: 10 }
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
        });
        let b = make_batch_q_q_n();
        let theme = ThemeInputs::default();
        let (scales, _) = resolve_scales(&s, &b, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
        assert!(matches!(scales.x, ScaleKind::Log(_)));
    }

    #[test]
    fn size_scale_defaults_to_3_to_30_px() {
        let batch = make_batch_q_q_n_n_q();
        let theme = ThemeInputs::default();
        let scale = build_size_scale(&make_spec_with_size().encoding, &batch, &theme)
            .unwrap()
            .unwrap();
        assert_eq!(scale.min_px, 3.0);
        assert_eq!(scale.max_px, 30.0);
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
        assert_eq!(scale.min_opacity, 0.1);
        assert_eq!(scale.max_opacity, 1.0);
    }
}
