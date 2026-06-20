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
    params: Vec::new(),
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
    params: Vec::new(),
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
    params: Vec::new(),
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
    params: Vec::new(),
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
    params: Vec::new(),
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
            domain_param: None,
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
    params: Vec::new(),
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
    params: Vec::new(),
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
    params: Vec::new(),
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
    params: Vec::new(),
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
        params: Vec::new(),
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
        params: Vec::new(),
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
                        domain_param: None,
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
        params: Vec::new(),
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
        params: Vec::new(),
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
        params: Vec::new(),
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
                        domain_param: None,
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
        params: Vec::new(),
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
                        domain_param: None,
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
        params: Vec::new(),
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
        params: Vec::new(),
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
        params: Vec::new(),
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
        params: Vec::new(),
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
        params: Vec::new(),
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
                            domain_param: None,
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
            params: Vec::new(),
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
        params: Vec::new(),
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

// ── D5: value-sort + sort spec on a categorical positional axis ───────────────

/// Bar-chart batch: x is categorical ("a","b","c") with multiple rows per
/// category; y is quantitative. Per-category SUM(y): a=1+1=2, b=5+5=10, c=3+3=6.
/// MEAN(y) is identical here (two equal rows each), so a dedicated mean batch is
/// built below where mean and sum disagree.
fn make_bar_batch_for_sort() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("cat", ArrowDataType::Utf8, false),
        Field::new("y", ArrowDataType::Float64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["a", "b", "c", "a", "b", "c"])),
            Arc::new(Float64Array::from(vec![1.0, 5.0, 3.0, 1.0, 5.0, 3.0])),
        ],
    )
    .unwrap()
}

fn make_bar_spec_with_sort(sort: serde_json::Value) -> ChartSpec {
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{DataType, Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    ChartSpec {
        data: DataRef::default(),
        mark: Mark::Bar,
        encoding: Encoding {
            x: Some(EncodingSpec {
                field: "cat".into(),
                type_: Some(DataType::Nominal),
                sort: Some(sort),
                ..Default::default()
            }),
            y: Some(EncodingSpec { field: "y".into(), type_: Some(DataType::Quantitative), ..Default::default() }),
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
        params: Vec::new(),
    }
}

/// Read the resolved x-axis ordinal domain order, asserting the scale is ordinal.
fn x_ordinal_domain(scales: &ResolvedScales) -> Vec<String> {
    match &scales.x {
        ScaleKind::Ordinal(s) => s.ticks_internal(),
        other => panic!("expected ordinal x scale, got {other:?}"),
    }
}

#[test]
fn d5_sort_neg_y_orders_categories_by_descending_sum() {
    // SUM(y): a=2, b=10, c=6. Descending → [b, c, a].
    let spec = make_bar_spec_with_sort(serde_json::json!("-y"));
    let batch = make_bar_batch_for_sort();
    let theme = ThemeInputs::default();
    let (scales, warnings) =
        resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
    assert!(warnings.is_empty(), "valid sort must not warn: {warnings:?}");
    assert_eq!(x_ordinal_domain(&scales), vec!["b", "c", "a"]);
}

#[test]
fn d5_sort_y_orders_categories_by_ascending_sum() {
    // SUM(y): a=2, b=10, c=6. Ascending → [a, c, b].
    let spec = make_bar_spec_with_sort(serde_json::json!("y"));
    let batch = make_bar_batch_for_sort();
    let theme = ThemeInputs::default();
    let (scales, warnings) =
        resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
    assert!(warnings.is_empty(), "valid sort must not warn: {warnings:?}");
    assert_eq!(x_ordinal_domain(&scales), vec!["a", "c", "b"]);
}

#[test]
fn d5_sort_field_object_mean_descending() {
    // Build a batch where MEAN and SUM disagree:
    //   a: [10] → sum 10, mean 10
    //   b: [1, 1, 1] → sum 3, mean 1
    //   c: [4, 6] → sum 10, mean 5
    // Descending MEAN → [a(10), c(5), b(1)]. (Descending SUM would tie a and c.)
    let schema = Arc::new(Schema::new(vec![
        Field::new("cat", ArrowDataType::Utf8, false),
        Field::new("y", ArrowDataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["a", "b", "b", "b", "c", "c"])),
            Arc::new(Float64Array::from(vec![10.0, 1.0, 1.0, 1.0, 4.0, 6.0])),
        ],
    )
    .unwrap();
    let spec = make_bar_spec_with_sort(serde_json::json!({
        "field": "y", "op": "mean", "order": "descending"
    }));
    let theme = ThemeInputs::default();
    let (scales, warnings) =
        resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
    assert!(warnings.is_empty(), "valid sort must not warn: {warnings:?}");
    assert_eq!(x_ordinal_domain(&scales), vec!["a", "c", "b"]);
}

#[test]
fn d5_explicit_array_yields_that_domain_order() {
    let spec = make_bar_spec_with_sort(serde_json::json!(["c", "a", "b"]));
    let batch = make_bar_batch_for_sort();
    let theme = ThemeInputs::default();
    let (scales, warnings) =
        resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
    assert!(warnings.is_empty(), "explicit array must not warn: {warnings:?}");
    assert_eq!(x_ordinal_domain(&scales), vec!["c", "a", "b"]);
}

#[test]
fn d5_ascending_descending_keywords_sort_alphabetically() {
    let batch = make_bar_batch_for_sort();
    let theme = ThemeInputs::default();

    let asc = make_bar_spec_with_sort(serde_json::json!("ascending"));
    let (scales_asc, _) =
        resolve_scales(&asc, &batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
    assert_eq!(x_ordinal_domain(&scales_asc), vec!["a", "b", "c"]);

    let desc = make_bar_spec_with_sort(serde_json::json!("descending"));
    let (scales_desc, _) =
        resolve_scales(&desc, &batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
    assert_eq!(x_ordinal_domain(&scales_desc), vec!["c", "b", "a"]);
}

#[test]
fn d5_invalid_sort_field_falls_back_without_panic_and_warns() {
    // Sort-field object naming a column that does not exist: domain stays in
    // insertion order (a, b, c) and a SortSpecIgnored warning is emitted.
    let spec = make_bar_spec_with_sort(serde_json::json!({
        "field": "does_not_exist", "op": "sum", "order": "descending"
    }));
    let batch = make_bar_batch_for_sort();
    let theme = ThemeInputs::default();
    let (scales, warnings) =
        resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
    // Insertion order preserved.
    assert_eq!(x_ordinal_domain(&scales), vec!["a", "b", "c"]);
    assert!(
        warnings
            .iter()
            .any(|w| matches!(w, crate::render::RenderWarning::SortSpecIgnored { .. })),
        "missing sort field must emit SortSpecIgnored, got: {warnings:?}"
    );
}

#[test]
fn d5_sort_field_object_count_orders_by_row_count() {
    // a: 1 row, b: 3 rows, c: 2 rows. Descending count → [b, c, a].
    let schema = Arc::new(Schema::new(vec![
        Field::new("cat", ArrowDataType::Utf8, false),
        Field::new("y", ArrowDataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["a", "b", "b", "b", "c", "c"])),
            Arc::new(Float64Array::from(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        ],
    )
    .unwrap();
    let spec = make_bar_spec_with_sort(serde_json::json!({
        "field": "y", "op": "count", "order": "descending"
    }));
    let theme = ThemeInputs::default();
    let (scales, warnings) =
        resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
    assert!(warnings.is_empty(), "valid count sort must not warn: {warnings:?}");
    assert_eq!(x_ordinal_domain(&scales), vec!["b", "c", "a"]);
}

// ── F1: Float64 ordinal axis scale resolution ────────────────────────────────

/// F1 regression: a Float64 column explicitly typed `:O` must resolve to an
/// ordinal x-scale without returning `UnsupportedDtype`. Previously, only
/// `distinct_values_in_order` (domain side) lacked Float support; the drawer
/// side (`col_as_ordinal_category_str`) already handled Float. This test
/// validates that both sides now agree and the scale resolves successfully.
#[test]
fn f1_float64_explicit_ordinal_type_resolves_to_ordinal_scale() {
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{DataType as SpecDataType, Encoding, EncodingSpec};
    use crate::spec::mark::Mark;

    let schema = Arc::new(Schema::new(vec![
        Field::new("yr", ArrowDataType::Float64, false),
        Field::new("v",  ArrowDataType::Float64, false),
    ]));
    // Three rows: yr=[2000.0, 2001.0, 2002.0], v=[1.0, 2.0, 3.0]
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![2000.0, 2001.0, 2002.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
        ],
    )
    .unwrap();

    let spec = ChartSpec {
        data: DataRef::default(),
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec {
                field: "yr".into(),
                type_: Some(SpecDataType::Ordinal),
                ..Default::default()
            }),
            y: Some(EncodingSpec { field: "v".into(), type_: None, ..Default::default() }),
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
        params: Vec::new(),
    };

    let theme = ThemeInputs::default();
    let (scales, warnings) =
        resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
    assert!(warnings.is_empty(), "float ordinal must not produce warnings: {warnings:?}");

    // x must resolve to an ordinal scale with integer-formatted domain strings.
    let domain = x_ordinal_domain(&scales);
    assert_eq!(domain, vec!["2000", "2001", "2002"],
        "Float64 ordinal domain must be integer-formatted (not '2000.0')"
    );

    // Each domain entry must map to a finite pixel (domain↔drawer agreement).
    for label in &domain {
        let px = scales.x.to_pixel_str(label);
        assert!(
            px.is_some(),
            "domain value {label:?} must map to a pixel via to_pixel_str"
        );
        assert!(
            px.unwrap().is_finite(),
            "pixel for {label:?} must be finite, got {:?}", px
        );
    }
}

/// F1 companion: Float64 ordinal with fractional values must resolve and
/// produce decimal-formatted domain strings that match what the drawer emits.
#[test]
fn f1_float64_ordinal_fractional_values_resolve() {
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{DataType as SpecDataType, Encoding, EncodingSpec};
    use crate::spec::mark::Mark;

    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ArrowDataType::Float64, false),
        Field::new("y", ArrowDataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.5, 2.5, 1.5])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
        ],
    )
    .unwrap();

    let spec = ChartSpec {
        data: DataRef::default(),
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec {
                field: "x".into(),
                type_: Some(SpecDataType::Ordinal),
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
        axis_x: None,
        axis_y: None,
        selections: Vec::new(),
        conditionals: Vec::new(),
        chart_description: None,
        params: Vec::new(),
    };

    let theme = ThemeInputs::default();
    let (scales, warnings) =
        resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
    assert!(warnings.is_empty(), "float ordinal must not produce warnings: {warnings:?}");

    // Fractional values must keep decimal format; deduped in encounter order.
    let domain = x_ordinal_domain(&scales);
    assert_eq!(domain, vec!["1.5", "2.5"]);

    // Domain entries must map to finite pixels.
    for label in &domain {
        let px = scales.x.to_pixel_str(label);
        assert!(px.is_some() && px.unwrap().is_finite(),
            "domain value {label:?} must map to a finite pixel"
        );
    }
}

// ── F5: characterization — default categorical palette is byte-stable ─────────

/// Characterization test pinning the default categorical palette produced by
/// `build_color_scale` for a string-typed color field with no explicit range.
///
/// The default theme uses `paper_ink`; its first color is rgb(0x25, 0x63, 0xEB).
/// This test must pass byte-identically before and after the F5 helper-extraction
/// refactor, proving that the helper produces the same output as the two inline
/// constructions it replaces.
#[test]
fn f5_default_categorical_palette_is_byte_stable() {
    let spec = make_spec_with_color();
    let batch = make_batch_q_q_n();
    let theme = ThemeInputs::default();
    let (scales, warnings) =
        resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
    assert!(warnings.is_empty(), "default path must not warn: {warnings:?}");
    let color_scale = scales.color.expect("color scale must be resolved");
    match &color_scale {
        ColorScale::Categorical { domain, palette } => {
            assert_eq!(domain, &["a", "b", "c"]);
            // Default theme = paper_ink; first color is rgb(0x25, 0x63, 0xEB).
            assert_eq!(
                (palette[0].red, palette[0].green, palette[0].blue),
                (0x25, 0x63, 0xEB),
                "default palette first color must be paper_ink[0] = rgb(0x25,0x63,0xEB); \
                 got rgb({},{},{})",
                palette[0].red, palette[0].green, palette[0].blue
            );
        }
        other => panic!("expected Categorical color scale, got {other:?}"),
    }
}

/// Characterization test pinning the fallback palette produced when an explicit
/// color-string range fails to parse.  The parse-failure arm now delegates to the
/// same `build_default_categorical_scale` helper as the default arm; this test
/// proves the two arms produce identical palette colors for an equivalent input.
#[test]
fn f5_parse_failure_fallback_palette_matches_default_palette() {
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec, ScaleSpec};
    use crate::spec::mark::Mark;

    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ArrowDataType::Float64, false),
        Field::new("y", ArrowDataType::Float64, false),
        Field::new("species", ArrowDataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
        ],
    )
    .unwrap();

    // Spec with a bad range that will trigger the parse-failure arm.
    let bad_range_spec = ChartSpec {
        data: DataRef::default(),
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            color: Some(EncodingSpec {
                field: "species".into(),
                type_: None,
                scale: Some(serde_json::from_value(serde_json::json!({
                    "type": "ordinal",
                    "domain": ["a", "b", "c"],
                    "range": ["#ff0000", "#not-a-color", "#0000ff"]
                })).unwrap()),
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
        params: Vec::new(),
    };

    let theme = ThemeInputs::default();

    // Fallback result (parse failure arm).
    let (scales_fallback, warnings_fallback) =
        resolve_scales(&bad_range_spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
    assert_eq!(warnings_fallback.len(), 1, "expected exactly one parse-failure warning");
    assert!(matches!(
        warnings_fallback[0],
        crate::render::RenderWarning::ColorRangeParseFailure { .. }
    ));

    // Default result (no range spec).
    let (scales_default, warnings_default) =
        resolve_scales(&make_spec_with_color(), &batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
    assert!(warnings_default.is_empty());

    // Both must produce the same palette colors (same domain, same palette bytes).
    match (
        scales_fallback.color.as_ref().unwrap(),
        scales_default.color.as_ref().unwrap(),
    ) {
        (
            ColorScale::Categorical { domain: d1, palette: p1 },
            ColorScale::Categorical { domain: d2, palette: p2 },
        ) => {
            assert_eq!(d1, d2, "fallback and default must have the same domain");
            assert_eq!(p1.len(), p2.len(), "palette lengths must match");
            for (i, (c1, c2)) in p1.iter().zip(p2.iter()).enumerate() {
                assert_eq!(
                    (c1.red, c1.green, c1.blue),
                    (c2.red, c2.green, c2.blue),
                    "palette color at index {i} must be identical in fallback and default paths"
                );
            }
        }
        _ => panic!("expected both to be Categorical"),
    }
}

// ── Gap 2: piecewise diverging midpoint normalization ─────────────────────────

#[test]
fn normalize_continuous_sequential_is_pure_linear() {
    // None midpoint → pure-linear (v - lo) / (hi - lo)
    assert!((super::normalize_continuous((0.0, 1.0), None, 0.0) - 0.0).abs() < 1e-9);
    assert!((super::normalize_continuous((0.0, 1.0), None, 0.5) - 0.5).abs() < 1e-9);
    assert!((super::normalize_continuous((0.0, 1.0), None, 1.0) - 1.0).abs() < 1e-9);
    // Clamping: out-of-range values clamp to [0, 1]
    assert!((super::normalize_continuous((0.0, 1.0), None, -1.0) - 0.0).abs() < 1e-9);
    assert!((super::normalize_continuous((0.0, 1.0), None, 2.0) - 1.0).abs() < 1e-9);
}

#[test]
fn normalize_continuous_diverging_symmetric_midpoint_equals_linear() {
    // mid = (lo + hi) / 2 → piecewise result must match pure-linear
    let lo = -1.0_f64;
    let hi = 1.0_f64;
    let mid = 0.0_f64; // geometric center
    for &v in &[-1.0, -0.5, 0.0, 0.5, 1.0_f64] {
        let piecewise = super::normalize_continuous((lo, hi), Some(mid), v);
        let linear = super::normalize_continuous((lo, hi), None, v);
        assert!(
            (piecewise - linear).abs() < 1e-9,
            "symmetric mid={mid}: v={v}, piecewise={piecewise}, linear={linear}"
        );
    }
}

#[test]
fn normalize_continuous_diverging_noncentral_midpoint() {
    // domain=[-1, 1], mid=0.5 (non-central)
    // lo → t=0, mid → t=0.5, hi → t=1
    let domain = (-1.0_f64, 1.0_f64);
    let mid = 0.5_f64;
    assert!((super::normalize_continuous(domain, Some(mid), -1.0) - 0.0).abs() < 1e-9,
        "lo must map to 0");
    assert!((super::normalize_continuous(domain, Some(mid), 0.5) - 0.5).abs() < 1e-9,
        "mid must map to 0.5");
    assert!((super::normalize_continuous(domain, Some(mid), 1.0) - 1.0).abs() < 1e-9,
        "hi must map to 1.0");
    // val=0.0 < mid=0.5 → left half: t = 0.5 * (0.0 - (-1.0)) / (0.5 - (-1.0)) = 0.5 * 1/1.5 ≈ 0.333
    let t_at_geom_center = super::normalize_continuous(domain, Some(mid), 0.0);
    assert!(t_at_geom_center > 0.0 && t_at_geom_center < 0.5,
        "val=0.0 < mid=0.5 must map to left half (0, 0.5), got {t_at_geom_center}");
    // val=0.0 must be less than 0.5 (reddish, left of neutral)
    assert!(t_at_geom_center < 0.5,
        "geometric center 0.0 must be left of neutral when mid=0.5, got {t_at_geom_center}");
}

#[test]
fn normalize_continuous_diverging_clamping() {
    let domain = (-1.0_f64, 1.0_f64);
    let mid = 0.0_f64;
    // Values outside domain clamp to [0, 1]
    assert!((super::normalize_continuous(domain, Some(mid), -2.0) - 0.0).abs() < 1e-9);
    assert!((super::normalize_continuous(domain, Some(mid), 2.0) - 1.0).abs() < 1e-9);
}

#[test]
fn normalize_continuous_degenerate_domain_returns_half() {
    // Zero-span domain returns 0.5
    assert!((super::normalize_continuous((1.0, 1.0), None, 1.0) - 0.5).abs() < 1e-9);
    assert!((super::normalize_continuous((1.0, 1.0), Some(1.0), 1.0) - 0.5).abs() < 1e-9);
}

// ── D2d: per-channel independent vs shared facet scale resolution ─────────────
//
// Validates that resolving scales from a per-panel partition (independent mode)
// yields a panel-specific domain, while resolving from the merged batch (shared
// mode) yields a unified domain across all panels. This is the behavioral
// invariant that FacetSpec.resolve.y = Independent vs Shared must satisfy.
//
// The test uses `resolve_scales_with_outputs` directly — the function that both
// `prepare.rs` (shared path, called on the full batch) and `scene_build.rs`
// (per-panel path, called on the filtered panel batch) call. The resolution
// itself is the same function; independent vs shared is determined by which
// batch is passed.

/// D2d: resolving from a per-panel partition with disjoint y ranges must yield
/// per-panel domains that differ from each other and from the global domain.
///
/// Partition A: y ∈ [0, 5]     (low range)
/// Partition B: y ∈ [100, 200] (high range)
/// Global merged: y ∈ [0, 200]
///
/// Independent path: each partition → its own y-domain.
/// Shared path: merged batch → single y-domain [0, 200] for all panels.
#[test]
fn d2d_independent_resolve_yields_per_panel_domains() {
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    use std::collections::HashMap;

    // Build the two panel batches with disjoint y ranges.
    let make_panel_batch = |x_vals: Vec<f64>, y_vals: Vec<f64>| -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", ArrowDataType::Float64, false),
            Field::new("y", ArrowDataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(x_vals)),
                Arc::new(Float64Array::from(y_vals)),
            ],
        )
        .unwrap()
    };

    let panel_a = make_panel_batch(vec![1.0, 2.0, 3.0], vec![0.0, 2.5, 5.0]);
    let panel_b = make_panel_batch(vec![1.0, 2.0, 3.0], vec![100.0, 150.0, 200.0]);

    // Merged batch simulating the shared (global) path.
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ArrowDataType::Float64, false),
        Field::new("y", ArrowDataType::Float64, false),
    ]));
    let merged = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![0.0, 2.5, 5.0, 100.0, 150.0, 200.0])),
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
        params: Vec::new(),
    };
    let theme = ThemeInputs::default();
    let empty_outputs: HashMap<String, RecordBatch> = HashMap::new();
    let pixel_range = (0.0, 200.0);

    // Independent path: each panel resolves from its own partition.
    let (scales_a, _) = resolve_scales_with_outputs(
        &spec, &panel_a, &empty_outputs, pixel_range, pixel_range, &theme,
    ).unwrap();
    let (scales_b, _) = resolve_scales_with_outputs(
        &spec, &panel_b, &empty_outputs, pixel_range, pixel_range, &theme,
    ).unwrap();

    // Shared path: both panels resolve from the global merged batch.
    let (scales_global, _) = resolve_scales_with_outputs(
        &spec, &merged, &empty_outputs, pixel_range, pixel_range, &theme,
    ).unwrap();

    // Independent: domains must differ between panels A and B.
    let (a_lo, a_hi) = scales_a.y.data_domain().expect("panel A y must have a domain");
    let (b_lo, b_hi) = scales_b.y.data_domain().expect("panel B y must have a domain");
    assert!(
        (a_hi - b_hi).abs() > 50.0,
        "independent panels must have different y domains: A=[{a_lo},{a_hi}], B=[{b_lo},{b_hi}]"
    );
    assert!(
        a_hi < 20.0,
        "panel A y-max must be near 5.0 (from its partition), got {a_hi}"
    );
    assert!(
        b_lo > 50.0,
        "panel B y-min must be near 100.0 (from its partition), got {b_lo}"
    );

    // Shared: global domain must cover both ranges.
    let (g_lo, g_hi) = scales_global.y.data_domain().expect("global y must have a domain");
    assert!(
        g_lo <= 1.0 && g_hi >= 199.0,
        "shared global y must span both partitions: [{g_lo},{g_hi}]"
    );

    // The per-panel (independent) domains must be subsets of the global domain.
    assert!(a_lo >= g_lo - 0.1 && a_hi <= g_hi + 0.1,
        "panel A domain [{a_lo},{a_hi}] must be contained in global [{g_lo},{g_hi}]");
    assert!(b_lo >= g_lo - 0.1 && b_hi <= g_hi + 0.1,
        "panel B domain [{b_lo},{b_hi}] must be contained in global [{g_lo},{g_hi}]");
}

// ── T4: shared faceted positional scale for RAW fields ───────────────────────
//
// The per-panel callsite (`scene_build.rs`) passes the panel's own partition as
// `primary_batch` and the global all-panels batch as
// `transform_outputs[FINAL_OUTPUT_KEY]`. Before T4 a faceted RAW field's
// positional domain was the per-panel min/max only — marks were per-panel-scaled
// while the shared axis showed the global domain (marks and axis disagree).
//
// These tests drive `resolve_scales_with_outputs` exactly as `scene_build.rs`
// does: panel batch as primary, global batch under `__final__`. A faceted spec
// with the default (Shared) resolve must union the global batch so the per-panel
// domain spans every panel; an Independent spec must keep per-panel domains.

/// Helper: a `__final__`-keyed transform_outputs map carrying the global batch,
/// mirroring what `prep.transform_outputs` holds at the per-panel callsite.
fn final_outputs(global: RecordBatch) -> HashMapForTests {
    let mut m: HashMapForTests = std::collections::HashMap::new();
    m.insert(crate::transform::core::FINAL_OUTPUT_KEY.to_string(), global);
    m
}

type HashMapForTests = std::collections::HashMap<String, RecordBatch>;

/// Build a faceted point spec on x/y with the given per-channel resolve modes.
fn faceted_xy_spec(
    x_resolve: crate::layout::facet::ResolveMode,
    y_resolve: crate::layout::facet::ResolveMode,
) -> ChartSpec {
    use crate::layout::facet::{FacetMode, FacetResolve, FacetSpec};
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    ChartSpec {
        data: DataRef::default(),
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            ..Default::default()
        },
        transforms: Vec::new(),
        facet: Some(FacetSpec {
            field: "panel".into(),
            row: None,
            mode: FacetMode::Wrap { ncols: 2 },
            spacing: None,
            resolve: FacetResolve { x: x_resolve, y: y_resolve },
        }),
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
        params: Vec::new(),
    }
}

fn xy_batch(x_vals: Vec<f64>, y_vals: Vec<f64>) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ArrowDataType::Float64, false),
        Field::new("y", ArrowDataType::Float64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(x_vals)),
            Arc::new(Float64Array::from(y_vals)),
        ],
    )
    .unwrap()
}

/// T4 numeric, x-channel: a faceted RAW x with disjoint per-panel ranges
/// (panel A x∈[0,10], panel B x∈[0,100]) must resolve panel A's x domain to the
/// GLOBAL [0,100] under Shared. Panel A's x=10 mark then lands at ~10% of the
/// plot width, NOT the right edge.
#[test]
fn t4_shared_facet_numeric_x_unions_global_domain() {
    let panel_a = xy_batch(vec![0.0, 5.0, 10.0], vec![0.0, 1.0, 2.0]);
    let panel_b = xy_batch(vec![0.0, 50.0, 100.0], vec![0.0, 1.0, 2.0]);
    let global = xy_batch(
        vec![0.0, 5.0, 10.0, 0.0, 50.0, 100.0],
        vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
    );
    let outputs = final_outputs(global);
    let spec = faceted_xy_spec(
        crate::layout::facet::ResolveMode::Shared,
        crate::layout::facet::ResolveMode::Shared,
    );
    let theme = ThemeInputs::default();
    let pr = (0.0, 100.0);

    // Panel A resolved exactly as scene_build does: panel batch primary, global
    // batch under __final__.
    let (scales_a, _) =
        resolve_scales_with_outputs(&spec, &panel_a, &outputs, pr, pr, &theme).unwrap();
    let (a_lo, a_hi) = scales_a.x.data_domain().expect("panel A x domain");
    assert!(
        a_lo <= 0.0 + 1e-6 && a_hi >= 100.0 - 1e-6,
        "Shared facet: panel A x domain must span the GLOBAL [0,100], got [{a_lo},{a_hi}]"
    );

    // Mark-position discrimination: panel A's x=10 mark must NOT be at the right
    // edge. On the global [0,100] domain with a 5% inward pad over (0,100),
    // x=10 maps to ~5 + 0.10 * 90 = ~14px, far left of the 100px right edge.
    let px_10 = scales_a.x.to_pixel_f64(10.0).expect("x=10 must project");
    let px_100 = scales_a.x.to_pixel_f64(100.0).expect("x=100 must project");
    assert!(
        px_10 < 0.30 * 100.0,
        "Shared facet: panel A x=10 must land in the left third (global domain), got {px_10}px"
    );
    assert!(
        (px_100 - 95.0).abs() < 2.0,
        "Shared facet: x=100 (global max) must land near the right edge, got {px_100}px"
    );
    // The per-panel max (x=10) is far from the panel-relative right edge.
    assert!(
        px_100 - px_10 > 50.0,
        "x=10 and x=100 must be far apart on the shared domain: 10@{px_10}px, 100@{px_100}px"
    );
}

/// T4 numeric, y-channel: a faceted RAW y with disjoint per-panel ranges must
/// resolve each panel's y domain to the GLOBAL union under Shared.
#[test]
fn t4_shared_facet_numeric_y_unions_global_domain() {
    let panel_a = xy_batch(vec![0.0, 1.0, 2.0], vec![0.0, 2.5, 5.0]);
    let panel_b = xy_batch(vec![0.0, 1.0, 2.0], vec![100.0, 150.0, 200.0]);
    let global = xy_batch(
        vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        vec![0.0, 2.5, 5.0, 100.0, 150.0, 200.0],
    );
    let outputs = final_outputs(global);
    let spec = faceted_xy_spec(
        crate::layout::facet::ResolveMode::Shared,
        crate::layout::facet::ResolveMode::Shared,
    );
    let theme = ThemeInputs::default();
    let pr = (0.0, 200.0);

    let (scales_a, _) =
        resolve_scales_with_outputs(&spec, &panel_a, &outputs, pr, pr, &theme).unwrap();
    let (a_lo, a_hi) = scales_a.y.data_domain().expect("panel A y domain");
    assert!(
        a_lo <= 0.0 + 1e-6 && a_hi >= 200.0 - 1e-6,
        "Shared facet: panel A y domain must span GLOBAL [0,200], got [{a_lo},{a_hi}]"
    );

    // Resolve panel B to confirm it also gets the global domain.
    let (scales_b, _) =
        resolve_scales_with_outputs(&spec, &panel_b, &outputs, pr, pr, &theme).unwrap();
    let (b_lo, b_hi) = scales_b.y.data_domain().expect("panel B y domain");
    assert!(
        b_lo <= 0.0 + 1e-6 && b_hi >= 200.0 - 1e-6,
        "Shared facet: panel B y domain must span GLOBAL [0,200], got [{b_lo},{b_hi}]"
    );

    // Mark-position discrimination (y-axis is INVERTED: small data y → large pixel y
    // / bottom of plot). Panel A's max data-y is 5, which occupies only the bottom
    // 2.5% of the global [0,200] domain. On a 200px plot range it must land near the
    // bottom (large pixel ≈ 190 px), not near the top as per-panel scaling would
    // produce (where y=5 would be the panel-relative max → pixel ≈ 5 px).
    //
    // With a 5% inward pad over [0,200] the pixel range is ~[190, 10] (bottom→top),
    // so y=5 → ~190 - 0.025 * 180 ≈ 185.5 px.  We just assert it lands in the top
    // 30% of PIXEL space (px < 0.70 * 200 = 140) which is the BOTTOM 70% of DATA
    // space — far from the panel-relative top that per-panel scaling would yield.
    let py_5 = scales_a.y.to_pixel_f64(5.0).expect("y=5 must project");
    let py_0 = scales_a.y.to_pixel_f64(0.0).expect("y=0 must project");
    let py_200 = scales_a.y.to_pixel_f64(200.0).expect("y=200 must project");
    // y-axis inversion: the bottom of the data range (y=0) maps to the largest
    // pixel value; the top of the data range (y=200) maps to the smallest.
    assert!(
        py_0 > py_200,
        "y-axis must be inverted: y=0 pixel ({py_0}) must be > y=200 pixel ({py_200})"
    );
    // y=5 (panel A's data max, but only 2.5% of the global range) must be near the
    // bottom of the plot, i.e. near y=0 in pixel terms — NOT near y=200 pixel pos.
    assert!(
        (py_5 - py_0).abs() < 0.10 * (py_0 - py_200),
        "Shared facet: panel A y=5 must land near the BOTTOM of the global domain \
         (within 10% of y=0 pixel pos); per-panel scaling would wrongly put it at \
         the TOP. Got py_5={py_5}, py_0={py_0}, py_200={py_200}"
    );
    // y=200 (global max) must land near the top edge.
    assert!(
        (py_200 - pr.0).abs() < 2.0 * (pr.1 - pr.0) / 200.0 * 10.0,
        "Shared facet: y=200 (global max) must land near the top edge, got {py_200}px"
    );
}

/// T4 Independent-mode preservation: with `share_scale(x="independent")` (here,
/// `ResolveMode::Independent` on x), panel A's x domain must stay per-panel
/// ([0,10]) — the escape hatch is byte-identical to the pre-T4 behavior. The
/// global batch under __final__ must be ignored for that channel.
#[test]
fn t4_independent_facet_numeric_x_keeps_per_panel_domain() {
    let panel_a = xy_batch(vec![0.0, 5.0, 10.0], vec![0.0, 1.0, 2.0]);
    let global = xy_batch(
        vec![0.0, 5.0, 10.0, 0.0, 50.0, 100.0],
        vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
    );
    let outputs = final_outputs(global);
    let spec = faceted_xy_spec(
        crate::layout::facet::ResolveMode::Independent,
        crate::layout::facet::ResolveMode::Shared,
    );
    let theme = ThemeInputs::default();
    let pr = (0.0, 100.0);

    let (scales_a, _) =
        resolve_scales_with_outputs(&spec, &panel_a, &outputs, pr, pr, &theme).unwrap();
    let (a_lo, a_hi) = scales_a.x.data_domain().expect("panel A x domain");
    assert!(
        a_lo <= 0.0 + 1e-6 && a_hi <= 10.0 + 1e-6 && a_hi >= 10.0 - 1e-6,
        "Independent facet: panel A x domain must stay per-panel [0,10], got [{a_lo},{a_hi}]"
    );
    // Mark-position discrimination: with the per-panel domain, x=10 (panel max)
    // lands at the panel-relative right edge (~95px after the 5% inward pad).
    let px_10 = scales_a.x.to_pixel_f64(10.0).expect("x=10 must project");
    assert!(
        px_10 > 0.70 * 100.0,
        "Independent facet: panel A x=10 (its own max) must be near the right edge, got {px_10}px"
    );
}

/// T4 categorical/band arm: a faceted categorical x where panel A is missing a
/// category present only in panel B must, under Shared, get the FULL category
/// set (shared band layout) so band widths/positions match the shared axis.
#[test]
fn t4_shared_facet_categorical_x_unions_global_categories() {
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    use crate::layout::facet::{FacetMode, FacetResolve, FacetSpec, ResolveMode};

    let cat_batch = |cats: Vec<&str>| -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", ArrowDataType::Utf8, false),
            Field::new("y", ArrowDataType::Float64, false),
        ]));
        let n = cats.len();
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(cats)),
                Arc::new(Float64Array::from(vec![1.0; n])),
            ],
        )
        .unwrap()
    };

    // Panel A has only {a, c}; panel B has {b, c}. Global has {a, b, c}.
    let panel_a = cat_batch(vec!["a", "c"]);
    let global = cat_batch(vec!["a", "c", "b", "c"]);
    let outputs = final_outputs(global);

    let spec = ChartSpec {
        data: DataRef::default(),
        mark: Mark::Bar,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "cat".into(), type_: Some(SpecDataType::Nominal), ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            ..Default::default()
        },
        transforms: Vec::new(),
        facet: Some(FacetSpec {
            field: "panel".into(),
            row: None,
            mode: FacetMode::Wrap { ncols: 2 },
            spacing: None,
            resolve: FacetResolve { x: ResolveMode::Shared, y: ResolveMode::Shared },
        }),
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
        params: Vec::new(),
    };
    let theme = ThemeInputs::default();
    let pr = (0.0, 100.0);

    let (scales_a, _) =
        resolve_scales_with_outputs(&spec, &panel_a, &outputs, pr, pr, &theme).unwrap();
    // Panel A (only {a,c}) must still place a band for category 'b' (present in
    // the global set) so its band layout matches the shared axis.
    let px_b = scales_a.x.to_pixel_str("b");
    assert!(
        px_b.is_some(),
        "Shared facet: panel A x scale must include category 'b' from the global set"
    );
    // All three categories must resolve, and to distinct band centers.
    let pa = scales_a.x.to_pixel_str("a").expect("a");
    let pb = scales_a.x.to_pixel_str("b").expect("b");
    let pc = scales_a.x.to_pixel_str("c").expect("c");
    assert!(
        (pa - pb).abs() > 1.0 && (pb - pc).abs() > 1.0 && (pa - pc).abs() > 1.0,
        "Shared facet: a/b/c bands must occupy distinct slots: a@{pa}, b@{pb}, c@{pc}"
    );
}

/// T4 categorical Independent-mode preservation: with x Independent, panel A
/// (only {a,c}) must NOT see category 'b' — per-panel category set preserved.
#[test]
fn t4_independent_facet_categorical_x_keeps_per_panel_categories() {
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    use crate::layout::facet::{FacetMode, FacetResolve, FacetSpec, ResolveMode};

    let cat_batch = |cats: Vec<&str>| -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", ArrowDataType::Utf8, false),
            Field::new("y", ArrowDataType::Float64, false),
        ]));
        let n = cats.len();
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(cats)),
                Arc::new(Float64Array::from(vec![1.0; n])),
            ],
        )
        .unwrap()
    };
    let panel_a = cat_batch(vec!["a", "c"]);
    let global = cat_batch(vec!["a", "c", "b", "c"]);
    let outputs = final_outputs(global);

    let spec = ChartSpec {
        data: DataRef::default(),
        mark: Mark::Bar,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "cat".into(), type_: Some(SpecDataType::Nominal), ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            ..Default::default()
        },
        transforms: Vec::new(),
        facet: Some(FacetSpec {
            field: "panel".into(),
            row: None,
            mode: FacetMode::Wrap { ncols: 2 },
            spacing: None,
            resolve: FacetResolve { x: ResolveMode::Independent, y: ResolveMode::Shared },
        }),
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
        params: Vec::new(),
    };
    let theme = ThemeInputs::default();
    let pr = (0.0, 100.0);

    let (scales_a, _) =
        resolve_scales_with_outputs(&spec, &panel_a, &outputs, pr, pr, &theme).unwrap();
    assert!(
        scales_a.x.to_pixel_str("b").is_none(),
        "Independent facet: panel A must NOT include category 'b' from other panels"
    );
    assert!(scales_a.x.to_pixel_str("a").is_some(), "panel A keeps 'a'");
    assert!(scales_a.x.to_pixel_str("c").is_some(), "panel A keeps 'c'");
}

/// Non-faceted byte-identity guard: with `facet = None`, the flag is false, and a
/// single batch resolves the same domain whether or not `__final__` carries a
/// wider batch (the flag gates that union off). Mirrors the non-faceted invariant.
#[test]
fn t4_non_faceted_ignores_final_output() {
    let primary = xy_batch(vec![0.0, 5.0, 10.0], vec![0.0, 1.0, 2.0]);
    // A wider __final__ batch must NOT widen a non-faceted chart's domain.
    let wider = xy_batch(vec![0.0, 50.0, 100.0], vec![0.0, 1.0, 2.0]);
    let outputs = final_outputs(wider);
    let spec = make_spec_with_color_xy_only();
    let theme = ThemeInputs::default();
    let pr = (0.0, 100.0);

    let (scales, _) =
        resolve_scales_with_outputs(&spec, &primary, &outputs, pr, pr, &theme).unwrap();
    let (lo, hi) = scales.x.data_domain().expect("x domain");
    assert!(
        lo <= 0.0 + 1e-6 && hi <= 10.0 + 1e-6 && hi >= 10.0 - 1e-6,
        "Non-faceted: domain must stay [0,10] (flag false → __final__ not unioned), got [{lo},{hi}]"
    );
}

// ── T4 data-aware sort: shared faceted categorical uses global batch ──────────
//
// Regression for F1 (round-8 sweep): `apply_sort_to_domain` was called with
// `SortContext.batch = located.batch` (per-panel) even when `include_final` is
// true (faceted Shared). Data-aware sort forms (`"-y"` etc.) aggregate per
// category from that batch, so each panel re-sorted the shared category vector
// by its OWN aggregate — marks land under the wrong tick.
//
// Fix: when `include_final`, build `SortContext.batch` from the global
// `FINAL_OUTPUT_KEY` batch (mirroring `distinct_positional_categories_shared`).

/// Build a batch with columns `cat` (Utf8) and `y` (Float64).
fn cat_y_batch(cats: Vec<&str>, ys: Vec<f64>) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("cat", ArrowDataType::Utf8, false),
        Field::new("y", ArrowDataType::Float64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(cats)),
            Arc::new(Float64Array::from(ys)),
        ],
    )
    .unwrap()
}

/// Build a faceted bar spec with nominal x (`field="cat"`, `sort="-y"`) and
/// quantitative y, using the given per-channel resolve modes.
fn faceted_nominal_sort_spec(
    x_resolve: crate::layout::facet::ResolveMode,
) -> ChartSpec {
    use crate::layout::facet::{FacetMode, FacetResolve, FacetSpec};
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    ChartSpec {
        data: DataRef::default(),
        mark: Mark::Bar,
        encoding: Encoding {
            x: Some(EncodingSpec {
                field: "cat".into(),
                type_: Some(SpecDataType::Nominal),
                sort: Some(serde_json::json!("-y")),
                ..Default::default()
            }),
            y: Some(EncodingSpec {
                field: "y".into(),
                type_: Some(SpecDataType::Quantitative),
                ..Default::default()
            }),
            ..Default::default()
        },
        transforms: Vec::new(),
        facet: Some(FacetSpec {
            field: "panel".into(),
            row: None,
            mode: FacetMode::Wrap { ncols: 2 },
            spacing: None,
            resolve: FacetResolve { x: x_resolve, y: crate::layout::facet::ResolveMode::Shared },
        }),
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
        params: Vec::new(),
    }
}

/// T4 data-aware sort (Shared): two facet panels whose per-panel y-aggregates
/// INVERT the global ranking.
///
/// Panel A:  a=100, b=50, c=10  → per-panel sort("-y") order: ["a","b","c"]
/// Panel B:  a=1,   b=5,  c=10  → per-panel sort("-y") order: ["c","b","a"]
/// Global:   a=101, b=55, c=20  → global sort("-y")   order: ["a","b","c"]
///
/// Under the buggy code, each panel would use its OWN aggregate to re-sort the
/// shared category vector:
///   - Panel A ends up with ["a","b","c"] (accidentally matches global).
///   - Panel B ends up with ["c","b","a"] (WRONG — mark "a" lands under "c").
///
/// After the fix both panels must agree with the global ranking ["a","b","c"].
#[test]
fn t4_shared_facet_categorical_data_aware_sort_uses_global_batch() {
    use crate::layout::facet::ResolveMode;

    // Per-panel data: panel A has high a, panel B has high c.
    let panel_a = cat_y_batch(vec!["a", "a", "b", "b", "c", "c"],
                               vec![50.0, 50.0, 25.0, 25.0, 5.0, 5.0]);
    let panel_b = cat_y_batch(vec!["a", "a", "b", "b", "c", "c"],
                               vec![0.5,  0.5,  2.5,  2.5,  5.0, 5.0]);

    // Global batch = concat of both panels.
    // Sums: a=101, b=55, c=20.  Descending order: ["a","b","c"].
    let global = cat_y_batch(
        vec!["a", "a", "b", "b", "c", "c", "a", "a", "b", "b", "c", "c"],
        vec![50.0, 50.0, 25.0, 25.0, 5.0, 5.0, 0.5, 0.5, 2.5, 2.5, 5.0, 5.0],
    );
    let outputs = final_outputs(global);

    let spec = faceted_nominal_sort_spec(ResolveMode::Shared);
    let theme = ThemeInputs::default();
    let pr = (0.0, 300.0);

    // Panel A: resolve the x scale (shared faceted, sort="-y").
    let (scales_a, warnings_a) =
        resolve_scales_with_outputs(&spec, &panel_a, &outputs, pr, pr, &theme).unwrap();
    assert!(
        warnings_a.is_empty(),
        "panel A: unexpected sort warnings: {warnings_a:?}"
    );

    // Panel B: resolve the x scale (shared faceted, sort="-y").
    let (scales_b, warnings_b) =
        resolve_scales_with_outputs(&spec, &panel_b, &outputs, pr, pr, &theme).unwrap();
    assert!(
        warnings_b.is_empty(),
        "panel B: unexpected sort warnings: {warnings_b:?}"
    );

    // Global order is a > b > c (descending sums: 101 > 55 > 20).
    // Both panels must map "a" to a LOWER pixel (leftmost) than "b", and
    // "b" lower than "c", since OrdinalScale assigns pixels left-to-right
    // in domain order.  The shared axis sort is [-y] descending → a,b,c.
    let px_a_a = scales_a.x.to_pixel_str("a").expect("panel A: 'a' must be in domain");
    let px_a_b = scales_a.x.to_pixel_str("b").expect("panel A: 'b' must be in domain");
    let px_a_c = scales_a.x.to_pixel_str("c").expect("panel A: 'c' must be in domain");
    assert!(
        px_a_a < px_a_b && px_a_b < px_a_c,
        "panel A: global sort('-y') must give domain order a<b<c (pixels left→right), \
         got a@{px_a_a}, b@{px_a_b}, c@{px_a_c}"
    );

    let px_b_a = scales_b.x.to_pixel_str("a").expect("panel B: 'a' must be in domain");
    let px_b_b = scales_b.x.to_pixel_str("b").expect("panel B: 'b' must be in domain");
    let px_b_c = scales_b.x.to_pixel_str("c").expect("panel B: 'c' must be in domain");
    assert!(
        px_b_a < px_b_b && px_b_b < px_b_c,
        "panel B: global sort('-y') must give domain order a<b<c (pixels left→right), \
         got a@{px_b_a}, b@{px_b_b}, c@{px_b_c} \
         (buggy: panel B's per-panel order would be c<b<a)"
    );

    // Both panels must agree on the same pixel positions (same shared domain/scale).
    assert!(
        (px_a_a - px_b_a).abs() < 1e-9 &&
        (px_a_b - px_b_b).abs() < 1e-9 &&
        (px_a_c - px_b_c).abs() < 1e-9,
        "shared panels must have identical pixel positions: \
         A:[{px_a_a},{px_a_b},{px_a_c}] vs B:[{px_b_a},{px_b_b},{px_b_c}]"
    );
}

/// T4 data-aware sort (Independent): per-panel sort order is preserved when
/// x resolves `Independent`. Panel B's per-panel aggregate has c > b > a, so
/// under Independent, panel B's domain must be ["c","b","a"].
#[test]
fn t4_independent_facet_categorical_data_aware_sort_uses_per_panel_batch() {
    use crate::layout::facet::ResolveMode;

    let panel_b = cat_y_batch(vec!["a", "a", "b", "b", "c", "c"],
                               vec![0.5,  0.5,  2.5,  2.5,  5.0, 5.0]);
    let global = cat_y_batch(
        vec!["a", "a", "b", "b", "c", "c", "a", "a", "b", "b", "c", "c"],
        vec![50.0, 50.0, 25.0, 25.0, 5.0, 5.0, 0.5, 0.5, 2.5, 2.5, 5.0, 5.0],
    );
    let outputs = final_outputs(global);

    let spec = faceted_nominal_sort_spec(ResolveMode::Independent);
    let theme = ThemeInputs::default();
    let pr = (0.0, 300.0);

    let (scales_b, warnings_b) =
        resolve_scales_with_outputs(&spec, &panel_b, &outputs, pr, pr, &theme).unwrap();
    assert!(
        warnings_b.is_empty(),
        "panel B Independent: unexpected sort warnings: {warnings_b:?}"
    );

    // Panel B independent: per-panel sums are a=1, b=5, c=10.
    // Descending → domain order ["c","b","a"]: c is leftmost, a rightmost.
    let px_b_a = scales_b.x.to_pixel_str("a").expect("panel B: 'a' must be in domain");
    let px_b_b = scales_b.x.to_pixel_str("b").expect("panel B: 'b' must be in domain");
    let px_b_c = scales_b.x.to_pixel_str("c").expect("panel B: 'c' must be in domain");
    assert!(
        px_b_c < px_b_b && px_b_b < px_b_a,
        "panel B Independent: per-panel sort('-y') must give domain order c<b<a \
         (pixels left→right: c has highest per-panel y sum), \
         got a@{px_b_a}, b@{px_b_b}, c@{px_b_c}"
    );
}

/// Minimal non-faceted x/y point spec for the byte-identity guard.
fn make_spec_with_color_xy_only() -> ChartSpec {
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    ChartSpec {
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
        axis_x: None,
        axis_y: None,
        selections: Vec::new(),
        conditionals: Vec::new(),
        chart_description: None,
        params: Vec::new(),
    }
}
