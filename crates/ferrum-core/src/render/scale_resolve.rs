//! Build ResolvedScales from a ChartSpec + a post-transform RecordBatch.
//! Phase 7 supports: LinearScale, OrdinalScale, TimeScale on x/y;
//! CategoricalColorScale on color.

use arrow::array::Array;
use arrow::datatypes::DataType as ArrowDataType;
use arrow::record_batch::RecordBatch;

use crate::scale::linear::LinearScale;
use crate::scale::ordinal::OrdinalScale;
use crate::scale::time::TimeScale;
use crate::spec::chart::ChartSpec;
use crate::spec::encoding::DataType as SpecDataType;

use super::color::Color;
use super::palette::OKABE_ITO;
use super::RenderError;

/// Sealed-enum wrapper over Phase 4 scales, used during render.
/// Phase 7 only constructs Linear/Ordinal/Time variants.
#[derive(Debug)]
pub enum ScaleKind {
    Linear(LinearScale),
    Ordinal(OrdinalScale),
    Time(TimeScale),
}

impl ScaleKind {
    /// Map a quantitative or temporal value to a pixel coordinate.
    /// Returns `None` for ordinal scales (use `to_pixel_str` instead).
    pub fn to_pixel_f64(&self, x: f64) -> Option<f64> {
        match self {
            Self::Linear(s) => Some(s.scale_internal(x)),
            Self::Time(s) => Some(s.scale_internal(x)),
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
        }
    }
}

#[derive(Debug)]
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

#[derive(Debug)]
pub struct ResolvedScales {
    pub x: ScaleKind,
    pub y: ScaleKind,
    pub color: Option<ColorScale>,
}

/// Build scales from spec + post-transform batch + pixel ranges.
/// Pixel ranges are panel-relative; caller passes panel.plot_area bounds.
pub fn resolve_scales(
    spec: &ChartSpec,
    batch: &RecordBatch,
    x_pixel_range: (f64, f64),
    y_pixel_range: (f64, f64),
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

    let x = build_axis_scale("x", x_enc, batch, x_pixel_range)?;
    let y = build_axis_scale("y", y_enc, batch, y_pixel_range)?;

    let color = if let Some(c_enc) = &spec.encoding.color {
        let domain = distinct_values_in_order(batch, &c_enc.field)?;
        if domain.len() > OKABE_ITO.len() {
            warnings.push(crate::render::RenderWarning::ColorPaletteOverflowed {
                categories: domain.len() as u32,
            });
        }
        // OKABE_ITO is `static LazyLock<[Color; 8]>`; deref then coerce array → slice
        // with `'static` lifetime, since the underlying storage outlives the program.
        let palette: &'static [Color] = &*OKABE_ITO;
        Some(ColorScale::Categorical { domain, palette })
    } else {
        None
    };

    Ok((ResolvedScales { x, y, color }, warnings))
}

fn build_axis_scale(
    channel: &'static str,
    enc: &crate::spec::encoding::EncodingSpec,
    batch: &RecordBatch,
    pixel_range: (f64, f64),
) -> Result<ScaleKind, RenderError> {
    let col = batch
        .column_by_name(&enc.field)
        .ok_or_else(|| RenderError::UnknownColumn { name: enc.field.clone() })?;
    let dtype = infer_spec_type(enc, col.data_type());
    // Y-axis pixel range is inverted (top of plot is min y, bottom is max y).
    let pr = if channel == "y" {
        (pixel_range.1, pixel_range.0)
    } else {
        pixel_range
    };
    match dtype {
        SpecDataType::Quantitative => {
            let (min, max) = column_min_max_f64(col)
                .map_err(|e| RenderError::ScaleResolutionFailed(format!("{channel}: {e}")))?;
            Ok(ScaleKind::Linear(LinearScale::new_internal(
                vec![min, max],
                vec![pr.0, pr.1],
                false,
                false,
            )))
        }
        SpecDataType::Ordinal | SpecDataType::Nominal => {
            let domain = distinct_values_in_order(batch, &enc.field)?;
            Ok(ScaleKind::Ordinal(OrdinalScale::new_internal(
                domain,
                vec![pr.0, pr.1],
                0.0,
            )))
        }
        SpecDataType::Temporal => {
            let (min, max) = column_min_max_f64(col)
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
    use arrow::array::{Float64Array, Int64Array, TimestampMillisecondArray};
    if let Some(a) = col.as_any().downcast_ref::<Float64Array>() {
        let min = a.iter().flatten().fold(f64::INFINITY, f64::min);
        let max = a.iter().flatten().fold(f64::NEG_INFINITY, f64::max);
        Ok((min, max))
    } else if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
        let min = a.iter().flatten().fold(i64::MAX, i64::min) as f64;
        let max = a.iter().flatten().fold(i64::MIN, i64::max) as f64;
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
    use arrow::array::{BooleanArray, Int64Array, StringArray};
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
        }
    }

    #[test]
    fn quantitative_x_resolves_to_linear() {
        let s = make_spec_with_color();
        let b = make_batch_q_q_n();
        let (scales, warnings) = resolve_scales(&s, &b, (0.0, 100.0), (0.0, 80.0)).unwrap();
        assert!(matches!(scales.x, ScaleKind::Linear(_)));
        assert!(matches!(scales.y, ScaleKind::Linear(_)));
        assert!(warnings.is_empty());
    }

    #[test]
    fn color_encoding_builds_categorical_in_encounter_order() {
        let s = make_spec_with_color();
        let b = make_batch_q_q_n();
        let (scales, _) = resolve_scales(&s, &b, (0.0, 100.0), (0.0, 80.0)).unwrap();
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
        let err = resolve_scales(&s, &b, (0.0, 100.0), (0.0, 80.0)).unwrap_err();
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
        };
        let (_, warnings) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0)).unwrap();
        assert!(matches!(
            warnings[0],
            crate::render::RenderWarning::ColorPaletteOverflowed { categories: 10 }
        ));
    }
}
