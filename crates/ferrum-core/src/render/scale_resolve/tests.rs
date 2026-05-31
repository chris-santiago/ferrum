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
            scheme: None,
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
                        domain: None, range: None, clamp: false, padding: None, scheme: None,
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
                        scheme: None,
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
                        scheme: None,
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

// ── D1: categorical/ordinal color scale accepts explicit string range ─────────

/// D1 regression: an ordinal color encoding with an explicit string `range`
/// (paired with `domain`) must resolve each domain value to its specified color.
/// Prior to the fix, `ScaleSpec::Ordinal.range` was `Option<Vec<f64>>` which
/// silently dropped string values, causing the resolver to always fall through
/// to the categorical palette.
#[test]
fn d1_ordinal_color_explicit_string_range_resolves_per_domain() {
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec, ScaleSpec};
    use crate::spec::mark::Mark;

    // Three categories: A=gray, B=gray, C=accent red. This "gray everything,
    // accent one" case validates that repeated colors and distinct colors both
    // resolve correctly.
    let gray = "#cccccc";
    let accent = "#e4572e";

    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ArrowDataType::Float64, false),
        Field::new("y", ArrowDataType::Float64, false),
        Field::new("cat", ArrowDataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(StringArray::from(vec!["A", "B", "C"])),
        ],
    )
    .unwrap();

    // Build scale spec JSON: {"type": "ordinal", "domain": ["A","B","C"], "range": ["#cccccc","#cccccc","#e4572e"]}
    let scale_json = serde_json::json!({
        "type": "ordinal",
        "domain": ["A", "B", "C"],
        "range": [gray, gray, accent]
    });
    let scale_spec: ScaleSpec = serde_json::from_value(scale_json).unwrap();

    let spec = ChartSpec {
        data: DataRef::default(),
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            color: Some(EncodingSpec {
                field: "cat".into(),
                type_: None,
                scale: Some(scale_spec),
                ..Default::default()
            }),
            ..Default::default()
        },
        transforms: Vec::new(),
        facet: None,
        layers: None,
        coord: None,
        mark_style: None,
        position: None,
        title: None,
        axis_x: None,
        axis_y: None,
        selections: Vec::new(),
        conditionals: Vec::new(),
        chart_description: None,
    };

    let theme = ThemeInputs::default();
    let (scales, warnings) =
        resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
    // No overflow warning: the explicit range exactly matches the domain.
    assert!(
        warnings.is_empty(),
        "explicit string range should not produce overflow warning: {warnings:?}"
    );

    let color_scale = scales.color.expect("color scale must be resolved");
    match &color_scale {
        ColorScale::Categorical { domain, palette } => {
            assert_eq!(domain, &["A", "B", "C"], "domain order preserved");
            // A and B must map to gray (#cccccc = rgb(204,204,204)).
            let c_a = palette[0];
            let c_b = palette[1];
            let c_c = palette[2];
            assert_eq!(
                (c_a.red, c_a.green, c_a.blue), (0xcc, 0xcc, 0xcc),
                "A should map to gray (#cccccc), got rgb({},{},{})", c_a.red, c_a.green, c_a.blue
            );
            assert_eq!(
                (c_b.red, c_b.green, c_b.blue), (0xcc, 0xcc, 0xcc),
                "B should map to gray (#cccccc), got rgb({},{},{})", c_b.red, c_b.green, c_b.blue
            );
            // C must map to accent (#e4572e = rgb(228,87,46)).
            assert_eq!(
                (c_c.red, c_c.green, c_c.blue), (0xe4, 0x57, 0x2e),
                "C should map to accent (#e4572e), got rgb({},{},{})", c_c.red, c_c.green, c_c.blue
            );
            // Lookup by string value must also agree.
            let looked_up_c = color_scale.lookup("C").expect("C must be in domain");
            assert_eq!(
                (looked_up_c.red, looked_up_c.green, looked_up_c.blue),
                (0xe4, 0x57, 0x2e),
                "lookup('C') should return accent color"
            );
        }
        other => panic!("expected Categorical color scale, got {other:?}"),
    }
}

/// D1 regression: when no explicit string range is given, the categorical
/// resolver must continue to use the default palette (okabe_ito / tableau10),
/// unchanged from pre-fix behavior.
#[test]
fn d1_ordinal_color_without_explicit_range_uses_default_palette() {
    let spec = make_spec_with_color();
    let batch = make_batch_q_q_n();
    let theme = ThemeInputs::default();
    let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
    let color_scale = scales.color.expect("color scale must be resolved");
    match &color_scale {
        ColorScale::Categorical { domain, palette } => {
            assert_eq!(domain, &["a", "b", "c"]);
            // Default palette (paper_ink or okabe_ito) first color must not be gray.
            let c0 = palette[0];
            // The default palette colors are never #cccccc (204,204,204); assert this
            // to confirm the explicit-range path did not corrupt the default.
            assert!(
                !(c0.red == 0xcc && c0.green == 0xcc && c0.blue == 0xcc),
                "default palette must not be the explicit gray range color"
            );
        }
        other => panic!("expected Categorical color scale, got {other:?}"),
    }
}

/// D1 regression: string range with CSS named colors (not only hex) must also
/// work, since `parse_color` supports named colors.
#[test]
fn d1_ordinal_color_named_css_colors_in_range() {
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec, ScaleSpec};
    use crate::spec::mark::Mark;

    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ArrowDataType::Float64, false),
        Field::new("y", ArrowDataType::Float64, false),
        Field::new("cat", ArrowDataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(StringArray::from(vec!["X", "Y"])),
        ],
    )
    .unwrap();

    let scale_json = serde_json::json!({
        "type": "ordinal",
        "domain": ["X", "Y"],
        "range": ["steelblue", "tomato"]
    });
    let scale_spec: ScaleSpec = serde_json::from_value(scale_json).unwrap();

    let spec = ChartSpec {
        data: DataRef::default(),
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            color: Some(EncodingSpec {
                field: "cat".into(),
                type_: None,
                scale: Some(scale_spec),
                ..Default::default()
            }),
            ..Default::default()
        },
        transforms: Vec::new(),
        facet: None,
        layers: None,
        coord: None,
        mark_style: None,
        position: None,
        title: None,
        axis_x: None,
        axis_y: None,
        selections: Vec::new(),
        conditionals: Vec::new(),
        chart_description: None,
    };

    let theme = ThemeInputs::default();
    let (scales, _) =
        resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
    let color_scale = scales.color.expect("color scale must be resolved");
    match &color_scale {
        ColorScale::Categorical { domain, palette } => {
            assert_eq!(domain, &["X", "Y"]);
            // steelblue = rgb(70, 130, 180)
            assert_eq!(
                (palette[0].red, palette[0].green, palette[0].blue),
                (70, 130, 180),
                "X should map to steelblue"
            );
            // tomato = rgb(255, 99, 71)
            assert_eq!(
                (palette[1].red, palette[1].green, palette[1].blue),
                (255, 99, 71),
                "Y should map to tomato"
            );
        }
        other => panic!("expected Categorical color scale, got {other:?}"),
    }
}

/// Finding 1 regression: when data rows appear in an order that differs from the
/// declared `scale.domain`, colors must follow the DECLARED domain order, not
/// data first-appearance order.
///
/// Data: rows ["C","A","B"] (C appears first).
/// Domain: ["A","B","C"] (declared, A is position 0).
/// Range: [gray, gray, accent] (accent is at position 2, i.e. for "C").
///
/// Expected: "C" → accent (#e4572e), "A" → gray, "B" → gray.
/// Before the fix, "C" would have been at position 0 (data order) and gotten
/// gray, while "A" would have gotten accent — the wrong assignment.
#[test]
fn d1_declared_domain_overrides_data_appearance_order() {
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec, ScaleSpec};
    use crate::spec::mark::Mark;

    let gray = "#cccccc";
    let accent = "#e4572e";

    // Data rows arrive in C, A, B order — C appears first.
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ArrowDataType::Float64, false),
        Field::new("y", ArrowDataType::Float64, false),
        Field::new("cat", ArrowDataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            // C first, then A, then B — data-appearance order is C=0, A=1, B=2.
            Arc::new(StringArray::from(vec!["C", "A", "B"])),
        ],
    )
    .unwrap();

    // Declared domain is A, B, C — accent is at position 2 (for "C").
    let scale_json = serde_json::json!({
        "type": "ordinal",
        "domain": ["A", "B", "C"],
        "range": [gray, gray, accent]
    });
    let scale_spec: ScaleSpec = serde_json::from_value(scale_json).unwrap();

    let spec = ChartSpec {
        data: DataRef::default(),
        mark: Mark::Bar,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "cat".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            color: Some(EncodingSpec {
                field: "cat".into(),
                type_: None,
                scale: Some(scale_spec),
                ..Default::default()
            }),
            ..Default::default()
        },
        transforms: Vec::new(),
        facet: None,
        layers: None,
        coord: None,
        mark_style: None,
        position: None,
        title: None,
        axis_x: None,
        axis_y: None,
        selections: Vec::new(),
        conditionals: Vec::new(),
        chart_description: None,
    };

    let theme = ThemeInputs::default();
    let (scales, warnings) =
        resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
    assert!(
        warnings.is_empty(),
        "explicit string range should not produce warnings: {warnings:?}"
    );

    let color_scale = scales.color.expect("color scale must be resolved");
    match &color_scale {
        ColorScale::Categorical { domain, palette } => {
            // Domain must follow the DECLARED order ["A","B","C"], not data order ["C","A","B"].
            assert_eq!(domain, &["A", "B", "C"], "domain must follow declared order");

            // "A" is at declared position 0 → gray.
            assert_eq!(
                (palette[0].red, palette[0].green, palette[0].blue),
                (0xcc, 0xcc, 0xcc),
                "A (position 0) must map to gray; got rgb({},{},{})",
                palette[0].red, palette[0].green, palette[0].blue
            );
            // "B" is at declared position 1 → gray.
            assert_eq!(
                (palette[1].red, palette[1].green, palette[1].blue),
                (0xcc, 0xcc, 0xcc),
                "B (position 1) must map to gray; got rgb({},{},{})",
                palette[1].red, palette[1].green, palette[1].blue
            );
            // "C" is at declared position 2 → accent.
            assert_eq!(
                (palette[2].red, palette[2].green, palette[2].blue),
                (0xe4, 0x57, 0x2e),
                "C (position 2) must map to accent (#e4572e); got rgb({},{},{})",
                palette[2].red, palette[2].green, palette[2].blue
            );

            // Lookup by string must also agree with declared order, not appearance order.
            let looked_up_c = color_scale.lookup("C").expect("C must be in domain");
            assert_eq!(
                (looked_up_c.red, looked_up_c.green, looked_up_c.blue),
                (0xe4, 0x57, 0x2e),
                "lookup('C') must return accent color regardless of data order"
            );
            let looked_up_a = color_scale.lookup("A").expect("A must be in domain");
            assert_eq!(
                (looked_up_a.red, looked_up_a.green, looked_up_a.blue),
                (0xcc, 0xcc, 0xcc),
                "lookup('A') must return gray regardless of data order"
            );
        }
        other => panic!("expected Categorical color scale, got {other:?}"),
    }
}

/// Finding 2 regression: when any color string in an explicit range fails to
/// parse, the resolver must emit a `ColorRangeParseFailure` warning naming the
/// offending entry and fall through to the default theme palette.  It must NOT
/// silently substitute the palette without any warning.
#[test]
fn d1_invalid_color_in_range_emits_warning_and_falls_through_to_default_palette() {
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec, ScaleSpec};
    use crate::spec::mark::Mark;

    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ArrowDataType::Float64, false),
        Field::new("y", ArrowDataType::Float64, false),
        Field::new("cat", ArrowDataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(StringArray::from(vec!["A", "B"])),
        ],
    )
    .unwrap();

    // Second entry is a typo that cannot be parsed.
    let scale_json = serde_json::json!({
        "type": "ordinal",
        "domain": ["A", "B"],
        "range": ["#cccccc", "#not-a-color"]
    });
    let scale_spec: ScaleSpec = serde_json::from_value(scale_json).unwrap();

    let spec = ChartSpec {
        data: DataRef::default(),
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            color: Some(EncodingSpec {
                field: "cat".into(),
                type_: None,
                scale: Some(scale_spec),
                ..Default::default()
            }),
            ..Default::default()
        },
        transforms: Vec::new(),
        facet: None,
        layers: None,
        coord: None,
        mark_style: None,
        position: None,
        title: None,
        axis_x: None,
        axis_y: None,
        selections: Vec::new(),
        conditionals: Vec::new(),
        chart_description: None,
    };

    let theme = ThemeInputs::default();
    let (scales, warnings) =
        resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();

    // Must have exactly one warning: ColorRangeParseFailure naming the bad entry.
    assert_eq!(warnings.len(), 1, "expected exactly one warning, got: {warnings:?}");
    match &warnings[0] {
        crate::render::RenderWarning::ColorRangeParseFailure { entry } => {
            assert_eq!(
                entry, "#not-a-color",
                "warning must name the offending entry; got: {entry}"
            );
        }
        other => panic!("expected ColorRangeParseFailure warning, got: {other:?}"),
    }

    // The color scale must fall through to the default palette (not the explicit range).
    let color_scale = scales.color.expect("color scale must be resolved");
    match &color_scale {
        ColorScale::Categorical { palette, .. } => {
            // The default palette's first color must not be the explicitly declared gray.
            let c0 = palette[0];
            assert!(
                !(c0.red == 0xcc && c0.green == 0xcc && c0.blue == 0xcc),
                "default palette must not start with the declared gray; parse-failure fallback must use theme palette"
            );
        }
        other => panic!("expected Categorical color scale, got {other:?}"),
    }
}

// ── D4: rect/heatmap color encoding honors scheme/cmap ────────────────────────

/// D4 regression: a color encoding with `scale={"type": "linear", "scheme": "blues"}`
/// must use the Blues continuous colormap. Prior to the fix, `ContinuousScaleCommon`
/// had no `scheme` field, so serde silently dropped the scheme and the resolver
/// fell back to the theme default, causing `cmap="blues"` and `cmap="rdbu"` to
/// produce byte-identical SVGs.
#[test]
fn d4_linear_scale_scheme_honored_in_continuous_color() {
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{
        ContinuousScaleCommon, Encoding, EncodingSpec, ScaleSpec,
    };
    use crate::spec::mark::Mark;

    // Identical data but different `scheme` → must produce different colors at
    // the same data value.
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ArrowDataType::Utf8, false),
        Field::new("y", ArrowDataType::Utf8, false),
        Field::new("v", ArrowDataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["a", "b"])),
            Arc::new(StringArray::from(vec!["x", "y"])),
            Arc::new(Float64Array::from(vec![0.0, 10.0])),
        ],
    )
    .unwrap();

    fn make_spec_with_scheme(scheme: &str) -> ChartSpec {
        use crate::spec::data_ref::DataRef;
        use crate::spec::encoding::{ContinuousScaleCommon, Encoding, EncodingSpec, ScaleSpec};
        use crate::spec::mark::Mark;
        ChartSpec {
            data: DataRef::default(),
            mark: Mark::Rect,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: Some(EncodingSpec {
                    field: "v".into(),
                    type_: Some(crate::spec::encoding::DataType::Quantitative),
                    scale: Some(ScaleSpec::Linear {
                        common: ContinuousScaleCommon {
                            domain: None,
                            range: None,
                            clamp: false,
                            padding: None,
                            scheme: Some(scheme.to_string()),
                        },
                        nice: false,
                        zero: false,
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
            position: None,
            title: None,
            axis_x: None,
            axis_y: None,
            selections: Vec::new(),
            conditionals: Vec::new(),
            chart_description: None,
        }
    }

    let theme = ThemeInputs::default();
    let (scales_blues, _) =
        resolve_scales(&make_spec_with_scheme("blues"), &batch, (0.0, 100.0), (0.0, 80.0), &theme)
            .unwrap();
    let (scales_rdbu, _) =
        resolve_scales(&make_spec_with_scheme("rdbu"), &batch, (0.0, 100.0), (0.0, 80.0), &theme)
            .unwrap();

    // Both must resolve to Continuous (not Categorical).
    let blues_scale = scales_blues.color.expect("blues color scale must be resolved");
    let rdbu_scale = scales_rdbu.color.expect("rdbu color scale must be resolved");

    match (&blues_scale, &rdbu_scale) {
        (
            ColorScale::Continuous { scheme: blues_scheme, .. },
            ColorScale::Continuous { scheme: rdbu_scheme, .. },
        ) => {
            // The two schemes must be different objects.
            assert_ne!(
                blues_scheme, rdbu_scheme,
                "blues and rdbu schemes must differ; both resolved to {blues_scheme:?}"
            );
            // Sample at t=0.5 — each scheme must produce a different color.
            let blues_mid = blues_scheme.sample(0.5);
            let rdbu_mid = rdbu_scheme.sample(0.5);
            assert_ne!(
                (blues_mid.red, blues_mid.green, blues_mid.blue),
                (rdbu_mid.red, rdbu_mid.green, rdbu_mid.blue),
                "blues and rdbu must produce different colors at t=0.5"
            );
        }
        _ => panic!(
            "expected both to be Continuous; blues={blues_scale:?}, rdbu={rdbu_scale:?}"
        ),
    }
}

/// D4 regression: the encoding-level `scheme` field (top-level on EncodingSpec,
/// not inside `scale`) must still work for continuous color after the fix.
/// This tests the original path (pre-heatmap) remains intact.
#[test]
fn d4_encoding_level_scheme_still_works_for_continuous_color() {
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{DataType as SDT, Encoding, EncodingSpec};
    use crate::spec::mark::Mark;

    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ArrowDataType::Float64, false),
        Field::new("y", ArrowDataType::Float64, false),
        Field::new("v", ArrowDataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0])),
        ],
    )
    .unwrap();
    let spec = ChartSpec {
        data: DataRef::default(),
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            color: Some(EncodingSpec {
                field: "v".into(),
                type_: Some(SDT::Quantitative),
                // scheme at the encoding level (top-level field, not inside scale)
                scheme: Some("viridis".into()),
                ..Default::default()
            }),
            ..Default::default()
        },
        transforms: Vec::new(),
        facet: None,
        layers: None,
        coord: None,
        mark_style: None,
        position: None,
        title: None,
        axis_x: None,
        axis_y: None,
        selections: Vec::new(),
        conditionals: Vec::new(),
        chart_description: None,
    };
    let theme = ThemeInputs::default();
    let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
    let color_scale = scales.color.expect("color scale must be resolved");
    match &color_scale {
        ColorScale::Continuous { scheme, .. } => {
            // Viridis mid-point is a greenish color; just verify it's Continuous and
            // the scheme is Named(Viridis) (not the theme default).
            use crate::render::color::{ContinuousScheme, NamedContinuous};
            assert_eq!(
                scheme,
                &ContinuousScheme::Named(NamedContinuous::Viridis),
                "scheme should be Viridis"
            );
        }
        other => panic!("expected Continuous color scale, got {other:?}"),
    }
}

/// D4 regression: `scheme` inside `scale` dict parsed from JSON round-trips.
/// Ensures that `{"type": "linear", "scheme": "rdbu"}` survives serde
/// round-trip without the scheme being dropped.
#[test]
fn d4_linear_scale_with_scheme_survives_serde_round_trip() {
    use crate::spec::encoding::ScaleSpec;
    let json = r#"{"type":"linear","scheme":"rdbu"}"#;
    let parsed: ScaleSpec = serde_json::from_str(json).unwrap();
    match &parsed {
        ScaleSpec::Linear { common, .. } => {
            assert_eq!(
                common.scheme.as_deref(),
                Some("rdbu"),
                "scheme must survive serde round-trip through ScaleSpec::Linear; got {:?}",
                common.scheme
            );
        }
        other => panic!("expected Linear variant, got {other:?}"),
    }
    // Re-serialize and check scheme is preserved.
    let re = serde_json::to_string(&parsed).unwrap();
    assert!(
        re.contains(r#""scheme":"rdbu""#),
        "re-serialized JSON must contain scheme: {re}"
    );
}
