use super::*;
use arrow::array::{Float64Array, StringArray};
use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
use std::sync::Arc;

use crate::layout::ThemeInputs;
use arrow::record_batch::RecordBatch;

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
    selections: Vec::new(), conditionals: Vec::new(),
    chart_description: None,
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
    selections: Vec::new(), conditionals: Vec::new(),
    chart_description: None,
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
    selections: Vec::new(), conditionals: Vec::new(),
    chart_description: None,
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
    selections: Vec::new(), conditionals: Vec::new(),
    chart_description: None,
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
    selections: Vec::new(), conditionals: Vec::new(),
    chart_description: None,
    }
}

#[test]
fn explicit_log_scale_overrides_auto_detection() {
    use crate::spec::encoding::ScaleSpec;
    let mut s = make_spec_with_color();
    s.encoding.x.as_mut().unwrap().scale = Some(ScaleSpec::Log {
        base: 10.0,
        common: crate::spec::encoding::ContinuousScaleCommon {
            domain: Some(vec![1.0, 1000.0]),
            range: None,
            clamp: false,
            padding: None,
        },
        nice: false,
    });
    let b = make_batch_q_q_n();
    let theme = ThemeInputs::default();
    let (scales, _) = resolve_scales(&s, &b, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
    assert!(matches!(scales.x, ScaleKind::Log(_)));
}

#[test]
fn size_scale_defaults_to_theme_point_size_range() {
    let batch = make_batch_q_q_n_n_q();
    let theme = ThemeInputs::default();
    let scale = build_size_scale(&make_spec_with_size().encoding, &batch, &theme)
        .unwrap()
        .unwrap();
    assert_eq!(scale.min_px(), 4.0);
    assert_eq!(scale.max_px(), 36.0);
}

#[test]
fn shape_scale_picks_from_8_shape_palette_in_order() {
    let batch = make_batch_q_q_n_n_q();
    let (scale, warn) = build_shape_scale(&make_spec_with_shape().encoding, &batch).unwrap();
    let scale = scale.unwrap();
    assert!(warn.is_none());
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
    selections: Vec::new(), conditionals: Vec::new(),
    chart_description: None,
    };
    let theme = ThemeInputs::default();
    let (scales, _) =
        resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
    let pixel_at_y2_max = scales.y.to_pixel_f64(3.0).expect("linear y returns Some");
    assert!(
        pixel_at_y2_max.is_finite(),
        "y-axis pixel for max y2 must be finite, got: {pixel_at_y2_max}"
    );
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
    selections: Vec::new(), conditionals: Vec::new(),
    chart_description: None,
    };
    let theme = ThemeInputs::default();
    let (scales, _) =
        resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
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
    selections: Vec::new(), conditionals: Vec::new(),
    chart_description: None,
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
    selections: Vec::new(), conditionals: Vec::new(),
    chart_description: None,
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

// --- Encoding resolution / inheritance edge-case tests ---

#[test]
fn single_value_domain_expands_to_symmetric_band() {
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ArrowDataType::Float64, false),
        Field::new("y", ArrowDataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![5.0, 5.0, 5.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
        ],
    )
    .unwrap();
    let spec = ChartSpec {
        data: DataRef::default(),
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
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
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
    };
    let theme = ThemeInputs::default();
    let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
    let px = scales.x.to_pixel_f64(5.0).expect("should return Some for constant domain");
    assert!(px.is_finite(), "pixel for constant-domain value must be finite, got: {px}");
    assert!((0.0..=100.0).contains(&px), "pixel should be within x range, got: {px}");
}

#[test]
fn single_value_zero_domain_expands_to_minus_one_plus_one() {
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ArrowDataType::Float64, false),
        Field::new("y", ArrowDataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![0.0, 0.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
        ],
    )
    .unwrap();
    let spec = ChartSpec {
        data: DataRef::default(),
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
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
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
    };
    let theme = ThemeInputs::default();
    let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
    let px = scales.x.to_pixel_f64(0.0).expect("should return Some");
    assert!(px.is_finite(), "pixel must be finite, got: {px}");
    let px_lo = scales.x.to_pixel_f64(-1.0).expect("should return Some for -1.0");
    let px_hi = scales.x.to_pixel_f64(1.0).expect("should return Some for 1.0");
    assert!(px_lo.is_finite() && px_hi.is_finite());
    assert!(px > px_lo.min(px_hi) - 1.0 && px < px_lo.max(px_hi) + 1.0);
}

#[test]
fn child_scale_type_override_wins_over_parent_auto_detection() {
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec, ScaleSpec};
    use crate::spec::mark::Mark;
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ArrowDataType::Float64, false),
        Field::new("y", ArrowDataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.0, 10.0, 100.0, 1000.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
        ],
    )
    .unwrap();
    let spec = ChartSpec {
        data: DataRef::default(),
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec {
                field: "x".into(),
                type_: Some(crate::spec::encoding::DataType::Quantitative),
                scale: Some(ScaleSpec::Symlog {
                    constant: 2.0,
                    common: crate::spec::encoding::ContinuousScaleCommon {
                        domain: None, range: None, clamp: false, padding: None,
                    },
                    nice: false,
                }),
                ..Default::default()
            }),
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
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
    };
    let theme = ThemeInputs::default();
    let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
    assert!(matches!(scales.x, ScaleKind::Symlog(_)), "expected Symlog, got: {:?}", scales.x);
}

#[test]
fn all_null_column_resolves_to_default_domain() {
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    use arrow::array::Float64Array;
    use arrow::buffer::NullBuffer;
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ArrowDataType::Float64, true),
        Field::new("y", ArrowDataType::Float64, true),
    ]));
    let nulls = NullBuffer::new_null(3);
    let x_arr = Float64Array::new(
        vec![0.0, 0.0, 0.0].into(),
        Some(nulls.clone()),
    );
    let y_arr = Float64Array::new(
        vec![0.0, 0.0, 0.0].into(),
        Some(nulls),
    );
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(x_arr), Arc::new(y_arr)],
    )
    .unwrap();
    let spec = ChartSpec {
        data: DataRef::default(),
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
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
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
    };
    let theme = ThemeInputs::default();
    let result = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme);
    assert!(result.is_ok(), "all-null column should not error: {:?}", result.err());
    let (scales, _) = result.unwrap();
    let px0 = scales.x.to_pixel_f64(0.0).expect("returns Some");
    let px1 = scales.x.to_pixel_f64(1.0).expect("returns Some");
    assert!(px0.is_finite() && px1.is_finite());
}

#[test]
fn empty_color_domain_produces_no_color_scale() {
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ArrowDataType::Float64, false),
        Field::new("y", ArrowDataType::Float64, false),
        Field::new("c", ArrowDataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Float64Array::from(Vec::<f64>::new())),
            Arc::new(Float64Array::from(Vec::<f64>::new())),
            Arc::new(StringArray::from(Vec::<&str>::new())),
        ],
    )
    .unwrap();
    let spec = ChartSpec {
        data: DataRef::default(),
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            color: Some(EncodingSpec { field: "c".into(), type_: None, ..Default::default() }),
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
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
    };
    let theme = ThemeInputs::default();
    let result = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme);
    assert!(result.is_ok(), "empty batch should not error: {:?}", result.err());
    let (scales, _) = result.unwrap();
    match &scales.color {
        None => {}
        Some(ColorScale::Categorical { domain, .. }) => {
            assert!(domain.is_empty(), "empty batch should produce empty color domain");
        }
        Some(other) => panic!("unexpected color scale variant: {:?}", other),
    }
}

// --- Phase 12: Power/Sqrt scale position resolution ---

#[test]
fn pow_scale_exponent_2_compresses_low_values() {
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec, ScaleSpec};
    use crate::spec::mark::Mark;
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ArrowDataType::Float64, false),
        Field::new("y", ArrowDataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![0.0, 5.0, 10.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
        ],
    )
    .unwrap();
    let spec = ChartSpec {
        data: DataRef::default(),
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec {
                field: "x".into(),
                type_: None,
                scale: Some(ScaleSpec::Pow {
                    exponent: 2.0,
                    common: crate::spec::encoding::ContinuousScaleCommon {
                        domain: Some(vec![0.0, 10.0]),
                        range: Some(vec![0.0, 100.0]),
                        clamp: false,
                        padding: None,
                    },
                }),
                ..Default::default()
            }),
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
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
    };
    let theme = ThemeInputs::default();
    let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();

    assert!(matches!(scales.x, ScaleKind::Pow(_)));

    let px_at_5 = scales.x.to_pixel_f64(5.0).expect("pow scale returns Some for in-domain");
    assert!(
        (px_at_5 - 25.0).abs() < 1e-9,
        "x=5 with exponent=2 should map to pixel 25.0, got {px_at_5}"
    );

    let px_at_0 = scales.x.to_pixel_f64(0.0).expect("pow scale returns Some for domain start");
    assert!((px_at_0 - 0.0).abs() < 1e-9, "x=0 should map to pixel 0, got {px_at_0}");

    let px_at_10 = scales.x.to_pixel_f64(10.0).expect("pow scale returns Some for domain end");
    assert!((px_at_10 - 100.0).abs() < 1e-9, "x=10 should map to pixel 100, got {px_at_10}");
}

#[test]
fn sqrt_scale_expands_low_values() {
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec, ScaleSpec};
    use crate::spec::mark::Mark;
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ArrowDataType::Float64, false),
        Field::new("y", ArrowDataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![0.0, 5.0, 10.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
        ],
    )
    .unwrap();
    let spec = ChartSpec {
        data: DataRef::default(),
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec {
                field: "x".into(),
                type_: None,
                scale: Some(ScaleSpec::Sqrt {
                    common: crate::spec::encoding::ContinuousScaleCommon {
                        domain: Some(vec![0.0, 10.0]),
                        range: Some(vec![0.0, 100.0]),
                        clamp: false,
                        padding: None,
                    },
                }),
                ..Default::default()
            }),
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
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
    };
    let theme = ThemeInputs::default();
    let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();

    assert!(matches!(scales.x, ScaleKind::Pow(_)));

    let expected = (5.0_f64.sqrt() / 10.0_f64.sqrt()) * 100.0;
    let px_at_5 = scales.x.to_pixel_f64(5.0).expect("sqrt scale returns Some for in-domain");
    assert!(
        (px_at_5 - expected).abs() < 1e-6,
        "x=5 with sqrt scale should map to ~{expected:.4}, got {px_at_5}"
    );
}

/// Continuous-axis scale projection: a zero-span (degenerate) domain must NOT
/// emit non-finite fractions — `(scale(v) - r0)/span` is `0/0 = NaN` there.
/// Both `tick_fractions` (auto) and `value_fractions` (explicit `tick_values`)
/// must return an empty vec so the caller drops the carrier and layout falls
/// back to uniform-slot placement (baseline behavior), never feeding a NaN to
/// the SVG renderer's non-finite guard.
#[test]
fn fractions_on_zero_span_domain_are_empty_not_nan() {
    use crate::scale::linear::LinearScale;
    // Degenerate domain: lo == hi (all-equal column / single distinct value).
    let scale = ScaleKind::Linear(LinearScale::new_internal(
        vec![3.0, 3.0],
        vec![0.0, 1.0],
        false,
        false,
    ));

    let auto = scale.tick_fractions(10);
    assert!(
        auto.is_empty(),
        "tick_fractions on a zero-span domain must be empty (no NaN), got {auto:?}"
    );
    assert!(auto.iter().all(|f| f.is_finite()));

    let explicit = scale.value_fractions(&[1.0, 2.0, 3.0]);
    assert!(
        explicit.is_empty(),
        "value_fractions on a zero-span domain must be empty (no NaN), got {explicit:?}"
    );
    assert!(explicit.iter().all(|f| f.is_finite()));
}

/// Companion to the zero-span case: on a normal linear domain the projection is
/// finite and index-aligned with the input values (the path that must keep
/// working for `configure_axis(tick_values=[...])`).
#[test]
fn value_fractions_on_normal_domain_are_finite_and_aligned() {
    use crate::scale::linear::LinearScale;
    // Domain [0, 10] over the normalized [0, 1] provisional range.
    let scale = ScaleKind::Linear(LinearScale::new_internal(
        vec![0.0, 10.0],
        vec![0.0, 1.0],
        false,
        false,
    ));
    let fr = scale.value_fractions(&[0.0, 5.0, 10.0]);
    assert_eq!(fr.len(), 3, "one fraction per input value, index-aligned");
    assert!(fr.iter().all(|f| f.is_finite()));
    assert!((fr[0] - 0.0).abs() < 1e-9);
    assert!((fr[1] - 0.5).abs() < 1e-9);
    assert!((fr[2] - 1.0).abs() < 1e-9);
}
