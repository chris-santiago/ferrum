use super::*;
use arrow::array::{Float64Array, StringArray};
use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
use std::sync::Arc;

use crate::layout::ThemeInputs;
use crate::scale::discrete::{DiscreteLayout, DEFAULT_ALIGN};
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
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    ChartSpec {
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            color: Some(EncodingSpec { field: "species".into(), type_: None, ..Default::default() }),
            ..Default::default()
        },
        ..Default::default()
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
        other => panic!("expected Categorical, got {other:?}"),
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
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            color: Some(EncodingSpec { field: "g".into(), type_: None, ..Default::default() }),
            ..Default::default()
        },
        ..Default::default()
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
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    ChartSpec {
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            size: Some(EncodingSpec { field: "size_val".into(), type_: None, ..Default::default() }),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn make_spec_with_shape() -> ChartSpec {
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    ChartSpec {
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            shape: Some(EncodingSpec { field: "species".into(), type_: None, ..Default::default() }),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn make_spec_with_opacity() -> ChartSpec {
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    ChartSpec {
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            opacity: Some(EncodingSpec { field: "opacity_val".into(), type_: None, ..Default::default() }),
            ..Default::default()
        },
        ..Default::default()
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
            reverse: false,
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
    let outputs: std::collections::HashMap<String, RecordBatch> = std::collections::HashMap::new();
    let (scale, warns) = build_size_scale(&make_spec_with_size().encoding, &batch, &outputs, false, &theme, None, &[])
        .unwrap();
    assert!(warns.is_empty());
    let scale = scale.unwrap();
    assert_eq!(scale.min_px(), 4.0);
    assert_eq!(scale.max_px(), 36.0);
}

/// `Size(scale=LogScale(domain=[1, 10], range=[10, 40]))` resolves to
/// `ScaleKind::Log`, through the exact same `LogScale::new_internal` the
/// positional `x`/`y` resolution uses, so `size_val` 1 (domain low) and 10
/// (domain high) map to the range ends and a mid-domain value maps
/// logarithmically, not linearly, between them.
#[test]
fn a_log_size_scale_resolves_to_scale_kind_log_and_maps_logarithmically() {
    use crate::spec::encoding::{ContinuousScaleCommon, ScaleSpec};
    let batch = make_batch_q_q_n_n_q();
    let theme = ThemeInputs::default();
    let outputs: std::collections::HashMap<String, RecordBatch> = std::collections::HashMap::new();
    let mut spec = make_spec_with_size();
    spec.encoding.size.as_mut().unwrap().scale = Some(ScaleSpec::Log {
        base: 10.0,
        common: ContinuousScaleCommon {
            domain: Some(vec![1.0, 10.0]),
            range: Some(vec![10.0, 40.0]),
            clamp: false,
            padding: None,
            scheme: None,
            domain_param: None,
            reverse: false,
        },
        nice: false,
    });
    let (scale, warns) = build_size_scale(&spec.encoding, &batch, &outputs, false, &theme, None, &[]).unwrap();
    assert!(warns.is_empty());
    let scale = scale.unwrap();
    assert!(matches!(scale.inner, ScaleKind::Log(_)));
    assert_eq!(scale.inner.to_pixel_f64(1.0), Some(10.0));
    assert_eq!(scale.inner.to_pixel_f64(10.0), Some(40.0));
    let mid = scale.inner.to_pixel_f64(3.1622776601683795).expect("sqrt(10) is in-domain");
    assert!((mid - 25.0).abs() < 1e-9, "log10(3.162)=0.5, so the half-decade point maps to the range midpoint, got {mid}");
}

/// `Size(scale=PowScale(exponent=2, domain=[0, 10], range=[0, 100]))` and
/// `Size(scale=SqrtScale(domain=[0, 10], range=[0, 100]))` resolve to
/// `ScaleKind::Pow`, mirroring the positional `pow_scale_exponent_2_
/// compresses_low_values`/`sqrt_scale_expands_low_values` tests: `Sqrt`
/// carries no `nice` field and resolves as `exponent: 0.5`, exactly as
/// `build_from_scale_spec`'s `ScaleSpec::Sqrt` arm does for x/y.
#[test]
fn pow_and_sqrt_size_scales_resolve_to_scale_kind_pow() {
    use crate::spec::encoding::{ContinuousScaleCommon, ScaleSpec};
    let batch = make_batch_q_q_n_n_q();
    let theme = ThemeInputs::default();
    let outputs: std::collections::HashMap<String, RecordBatch> = std::collections::HashMap::new();
    let common = |domain: Vec<f64>, range: Vec<f64>| ContinuousScaleCommon {
        domain: Some(domain),
        range: Some(range),
        clamp: false,
        padding: None,
        scheme: None,
        domain_param: None,
        reverse: false,
    };

    let mut pow_spec = make_spec_with_size();
    pow_spec.encoding.size.as_mut().unwrap().scale =
        Some(ScaleSpec::Pow { exponent: 2.0, common: common(vec![0.0, 10.0], vec![0.0, 100.0]) });
    let pow_scale = build_size_scale(&pow_spec.encoding, &batch, &outputs, false, &theme, None, &[])
        .unwrap()
        .0
        .unwrap();
    assert!(matches!(pow_scale.inner, ScaleKind::Pow(_)));
    assert_eq!(pow_scale.inner.to_pixel_f64(5.0), Some(25.0), "5^2 / 10^2 * 100 = 25");

    let mut sqrt_spec = make_spec_with_size();
    sqrt_spec.encoding.size.as_mut().unwrap().scale =
        Some(ScaleSpec::Sqrt { common: common(vec![0.0, 100.0], vec![0.0, 10.0]) });
    let sqrt_scale = build_size_scale(&sqrt_spec.encoding, &batch, &outputs, false, &theme, None, &[])
        .unwrap()
        .0
        .unwrap();
    assert!(matches!(sqrt_scale.inner, ScaleKind::Pow(_)));
    assert_eq!(sqrt_scale.inner.to_pixel_f64(25.0), Some(5.0), "sqrt(25)/sqrt(100) * 10 = 5");
}

/// A value outside an explicit domain clamps to the nearer range end, on a
/// `Size(scale=LogScale(...))` exactly as it does on `Size(scale=
/// LinearScale(...))` (`a_size_domain_is_honored_and_clamps`,
/// `render::prepare::tests`) -- the numeric auxiliary channels' own
/// always-clamp contract, carried through unchanged for the new scale
/// kinds.
#[test]
fn a_log_size_domain_clamps_out_of_domain_values() {
    use crate::spec::encoding::{ContinuousScaleCommon, ScaleSpec};
    let batch = make_batch_q_q_n_n_q();
    let theme = ThemeInputs::default();
    let outputs: std::collections::HashMap<String, RecordBatch> = std::collections::HashMap::new();
    let mut spec = make_spec_with_size();
    spec.encoding.size.as_mut().unwrap().scale = Some(ScaleSpec::Log {
        base: 10.0,
        common: ContinuousScaleCommon {
            domain: Some(vec![2.0, 3.0]),
            range: Some(vec![10.0, 40.0]),
            clamp: false,
            padding: None,
            scheme: None,
            domain_param: None,
            reverse: false,
        },
        nice: false,
    });
    let scale = build_size_scale(&spec.encoding, &batch, &outputs, false, &theme, None, &[]).unwrap().0.unwrap();
    assert_eq!(scale.inner.to_pixel_f64(0.5), Some(10.0), "below the domain clamps to the low range end");
    assert_eq!(scale.inner.to_pixel_f64(100.0), Some(40.0), "above the domain clamps to the high range end");
}

/// A size `domainParam` reaches the size scale the marks draw with and the
/// provisional scale `build_scene`'s once-per-chart swarm packing bound
/// (`position::chart_swarms`) and the size legend read: both map the param's
/// `[0, 2]`, not the data extent.
#[test]
fn a_size_domain_param_reaches_the_drawn_and_the_provisional_size_scale() {
    use crate::spec::encoding::{ContinuousScaleCommon, ScaleSpec};
    use ferrum_scene::{ParamKind, ParameterSpec};
    let mut spec = make_spec_with_size();
    spec.encoding.size.as_mut().unwrap().scale = Some(ScaleSpec::Linear {
        common: ContinuousScaleCommon {
            domain: None,
            range: None,
            clamp: false,
            padding: None,
            scheme: None,
            domain_param: Some("sd".into()),
            reverse: false,
        },
        nice: false,
        zero: false,
        system_injected: false,
    });
    spec.params = vec![ParameterSpec {
        name: "sd".into(),
        kind: ParamKind::Variable,
        value: Some(serde_json::json!([0, 2])),
        bind: None,
        select: None,
    }];
    let mut substituted = spec.clone();
    resolve_param_domains(&mut substituted);
    let batch = make_batch_q_q_n_n_q();
    let theme = ThemeInputs::default();
    let outputs: std::collections::HashMap<String, RecordBatch> = std::collections::HashMap::new();
    let drawn = build_size_scale(&substituted.encoding, &batch, &outputs, false, &theme, None, &[]).unwrap().0.unwrap();
    assert_eq!(drawn.inner.data_domain(), Some((0.0, 2.0)), "the marks draw over the param's domain");

    pyo3::Python::initialize();
    let prep = crate::render::prepare::prepare_render_inputs(
        &spec,
        &batch,
        &theme,
        crate::layout::Viewport { width: 600.0, height: 400.0 },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .expect("the spec prepares");
    let provisional = prep.provisional_scales.size.expect("a provisional size scale");
    assert_eq!(provisional.inner.data_domain(), Some((0.0, 2.0)), "the swarm bound and the legend read the param's domain");
}

#[test]
fn shape_scale_picks_from_8_shape_palette_in_order() {
    let batch = make_batch_q_q_n_n_q();
    let outputs: std::collections::HashMap<String, RecordBatch> = std::collections::HashMap::new();
    let (scale, warns) = build_shape_scale(&make_spec_with_shape().encoding, &batch, &outputs, /* facet_shared */ false, /* coord_flipped */ false, &[]).unwrap();
    let scale = scale.unwrap();
    assert!(warns.is_empty());
    assert_eq!(scale.shapes.len(), 3);
    assert_eq!(scale.shapes[0], ShapeKind::Circle);
    assert_eq!(scale.shapes[1], ShapeKind::Square);
    assert_eq!(scale.shapes[2], ShapeKind::Cross);
}

// --- KG-8: shape encoding honors sort ---

/// Helper: build a spec with shape encoding and an optional sort value.
fn make_spec_with_shape_sort(sort_json: Option<serde_json::Value>) -> ChartSpec {
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    ChartSpec {
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            shape: Some(EncodingSpec {
                field: "species".into(),
                type_: None,
                sort: sort_json,
                ..Default::default()
            }),
            ..Default::default()
        },
        ..Default::default()
    }
}

/// KG-8 baseline: absent sort keeps first-appearance order (byte-identical).
/// Data rows: "cat", "bird", "dog" → domain must be ["cat", "bird", "dog"].
#[test]
fn shape_absent_sort_preserves_first_appearance_order() {
    use arrow::array::StringArray;
    use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
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
            Arc::new(StringArray::from(vec!["cat", "bird", "dog"])),
        ],
    )
    .unwrap();
    let spec = make_spec_with_shape_sort(None);
    let outputs = std::collections::HashMap::new();
    let (scale, warns) = build_shape_scale(&spec.encoding, &batch, &outputs, /* facet_shared */ false, /* coord_flipped */ false, &[]).unwrap();
    let scale = scale.unwrap();
    assert!(warns.is_empty());
    // First-appearance order: cat=0, bird=1, dog=2
    assert_eq!(scale.domain, vec!["cat", "bird", "dog"]);
    // Shape assignments must match that order
    assert_eq!(scale.shapes[0], ShapeKind::Circle);   // cat → circle
    assert_eq!(scale.shapes[1], ShapeKind::Square);   // bird → square
    assert_eq!(scale.shapes[2], ShapeKind::Cross);    // dog  → cross
}

/// KG-8: sort="ascending" reorders domain alphabetically.
/// Data: "cat", "bird", "dog" → domain must be ["bird", "cat", "dog"].
#[test]
fn shape_sort_ascending_reorders_domain_alphabetically() {
    use arrow::array::StringArray;
    use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
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
            Arc::new(StringArray::from(vec!["cat", "bird", "dog"])),
        ],
    )
    .unwrap();
    let spec = make_spec_with_shape_sort(Some(serde_json::json!("ascending")));
    let outputs = std::collections::HashMap::new();
    let (scale, warns) = build_shape_scale(&spec.encoding, &batch, &outputs, /* facet_shared */ false, /* coord_flipped */ false, &[]).unwrap();
    let scale = scale.unwrap();
    assert!(warns.is_empty());
    assert_eq!(scale.domain, vec!["bird", "cat", "dog"]);
    // After sort, bird is at index 0 → circle, cat → square, dog → cross
    assert_eq!(scale.shapes[0], ShapeKind::Circle);
    assert_eq!(scale.shapes[1], ShapeKind::Square);
    assert_eq!(scale.shapes[2], ShapeKind::Cross);
}

/// KG-8: sort="descending" reorders domain in reverse alphabetical order.
/// Data: "cat", "bird", "dog" → domain must be ["dog", "cat", "bird"].
#[test]
fn shape_sort_descending_reorders_domain_reverse_alphabetically() {
    use arrow::array::StringArray;
    use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
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
            Arc::new(StringArray::from(vec!["cat", "bird", "dog"])),
        ],
    )
    .unwrap();
    let spec = make_spec_with_shape_sort(Some(serde_json::json!("descending")));
    let outputs = std::collections::HashMap::new();
    let (scale, warns) = build_shape_scale(&spec.encoding, &batch, &outputs, /* facet_shared */ false, /* coord_flipped */ false, &[]).unwrap();
    let scale = scale.unwrap();
    assert!(warns.is_empty());
    assert_eq!(scale.domain, vec!["dog", "cat", "bird"]);
}

/// KG-8: sort="-y" (data-aware channel shorthand) orders the shape domain
/// by the per-category sum of y values, descending.
///
/// Data: cat rows y=[10,30], bird rows y=[20], dog rows y=[40].
/// y-sums: cat=40, bird=20, dog=40. Tie between cat and dog resolved by
/// `sort_by` (stable on equal keys → original relative order preserved for
/// ties: cat appears before dog in first-appearance).
/// Expected descending by sum: dog=40, cat=40, bird=20 — but since
/// `sort_by` is stable and cat precedes dog in domain, the tie-break keeps
/// cat before dog. Expected: ["cat", "dog", "bird"] or ["dog", "cat", "bird"]
/// depending on sort stability. We assert bird is last (lowest sum) and does
/// NOT have the circle glyph — the discriminating part.
#[test]
fn shape_sort_data_aware_minus_y_orders_by_y_aggregate_descending() {
    use arrow::array::StringArray;
    use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ArrowDataType::Float64, false),
        Field::new("y", ArrowDataType::Float64, false),
        Field::new("species", ArrowDataType::Utf8, false),
    ]));
    // cat: y=10,30 (sum=40), bird: y=20 (sum=20), dog: y=40 (sum=40)
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0])),
            Arc::new(StringArray::from(vec!["cat", "bird", "cat", "dog"])),
        ],
    )
    .unwrap();
    let spec = make_spec_with_shape_sort(Some(serde_json::json!("-y")));
    let outputs = std::collections::HashMap::new();
    let (scale, warns) = build_shape_scale(&spec.encoding, &batch, &outputs, /* facet_shared */ false, /* coord_flipped */ false, &[]).unwrap();
    let scale = scale.unwrap();
    assert!(warns.is_empty(), "no warnings expected for valid -y sort: {warns:?}");
    // bird has the lowest y-sum (20) → must be last in the descending domain
    assert_eq!(scale.domain.last(), Some(&"bird".to_string()), "bird must be last (lowest y-sum)");
    // bird must NOT get the circle glyph (index 0)
    let bird_idx = scale.domain.iter().position(|v| v == "bird").unwrap();
    assert_ne!(scale.shapes[bird_idx], ShapeKind::Circle, "bird must not get circle after -y sort");
}

/// KG-8: explicit-array sort form reorders the domain to match the array order.
/// sort=["dog","bird","cat"] → domain must be ["dog", "bird", "cat"].
#[test]
fn shape_sort_explicit_array_overrides_first_appearance_order() {
    use arrow::array::StringArray;
    use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
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
            Arc::new(StringArray::from(vec!["cat", "bird", "dog"])),
        ],
    )
    .unwrap();
    let spec = make_spec_with_shape_sort(Some(serde_json::json!(["dog", "bird", "cat"])));
    let outputs = std::collections::HashMap::new();
    let (scale, warns) = build_shape_scale(&spec.encoding, &batch, &outputs, /* facet_shared */ false, /* coord_flipped */ false, &[]).unwrap();
    let scale = scale.unwrap();
    assert!(warns.is_empty());
    assert_eq!(scale.domain, vec!["dog", "bird", "cat"]);
    // dog is now at index 0 → circle
    assert_eq!(scale.shapes[0], ShapeKind::Circle);
    // bird at index 1 → square
    assert_eq!(scale.shapes[1], ShapeKind::Square);
    // cat at index 2 → cross
    assert_eq!(scale.shapes[2], ShapeKind::Cross);
}

/// KG-8: a data-aware sort referencing a missing field emits SortSpecIgnored
/// and leaves the domain in first-appearance order (never panics).
#[test]
fn shape_sort_missing_field_emits_warning_not_panic() {
    use arrow::array::StringArray;
    use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
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
            Arc::new(StringArray::from(vec!["cat", "bird", "dog"])),
        ],
    )
    .unwrap();
    // Sort by a field that doesn't exist in the batch
    let sort_obj = serde_json::json!({"field": "nonexistent_column", "op": "mean", "order": "descending"});
    let spec = make_spec_with_shape_sort(Some(sort_obj));
    let outputs = std::collections::HashMap::new();
    let (scale, warns) = build_shape_scale(&spec.encoding, &batch, &outputs, /* facet_shared */ false, /* coord_flipped */ false, &[]).unwrap();
    let scale = scale.unwrap();
    // Must emit a SortSpecIgnored warning
    assert!(!warns.is_empty(), "expected SortSpecIgnored warning for missing field");
    assert!(
        warns.iter().any(|w| matches!(w, crate::render::RenderWarning::SortSpecIgnored { .. })),
        "expected SortSpecIgnored warning, got: {warns:?}"
    );
    // Domain must remain in first-appearance order
    assert_eq!(scale.domain, vec!["cat", "bird", "dog"]);
}

// --- #123 D1: Shape/StrokeDash `scale=` domain+range honored (Q1) ---

/// Helper: build a spec with a shape encoding carrying an explicit `scale=`.
fn make_spec_with_shape_scale(scale: Option<crate::spec::encoding::ScaleSpec>) -> ChartSpec {
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    ChartSpec {
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            shape: Some(EncodingSpec { field: "species".into(), type_: None, scale, ..Default::default() }),
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Helper: build a spec with a stroke_dash encoding carrying an explicit `scale=`.
fn make_spec_with_stroke_dash_scale(scale: Option<crate::spec::encoding::ScaleSpec>) -> ChartSpec {
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    ChartSpec {
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            stroke_dash: Some(EncodingSpec {
                field: "level".into(),
                type_: None,
                scale,
                ..Default::default()
            }),
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Three species rows in first-appearance order `["cat", "bird", "dog"]`.
fn species_batch() -> RecordBatch {
    use arrow::array::StringArray;
    use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ArrowDataType::Float64, false),
        Field::new("y", ArrowDataType::Float64, false),
        Field::new("species", ArrowDataType::Utf8, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
            Arc::new(StringArray::from(vec!["cat", "bird", "dog"])),
        ],
    )
    .unwrap()
}

fn ordinal_scale(domain: Option<Vec<&str>>, range: Option<Vec<&str>>) -> crate::spec::encoding::ScaleSpec {
    let mut json = serde_json::json!({"type": "ordinal"});
    if let Some(d) = domain {
        json["domain"] = serde_json::json!(d);
    }
    if let Some(r) = range {
        json["range"] = serde_json::json!(r);
    }
    serde_json::from_value(json).expect("ordinal scale spec must deserialize")
}

/// #123 D1: an explicit `scale.domain` reorders the domain and, with it,
/// which glyph each category gets — unconditionally, unlike color's
/// range-gated D1 (no `range` is set here at all).
#[test]
fn shape_scale_explicit_domain_reassigns_glyph_assignment() {
    let batch = species_batch();
    let outputs = std::collections::HashMap::new();
    let baseline_spec = make_spec_with_shape_scale(None);
    let (baseline, _) = build_shape_scale(&baseline_spec.encoding, &batch, &outputs, false, false, &[]).unwrap();
    let baseline = baseline.unwrap();
    assert_eq!(baseline.domain, vec!["cat", "bird", "dog"], "sanity: first-appearance order");

    let spec = make_spec_with_shape_scale(Some(ordinal_scale(Some(vec!["dog", "bird", "cat"]), None)));
    let (scale, warns) = build_shape_scale(&spec.encoding, &batch, &outputs, false, false, &[]).unwrap();
    let scale = scale.unwrap();
    assert!(warns.is_empty(), "no warnings expected: {warns:?}");
    assert_eq!(scale.domain, vec!["dog", "bird", "cat"], "declared domain wins over first-appearance order");
    // dog is now first -> circle; cat (was first, circle) is now last -> cross.
    assert_eq!(scale.lookup("dog"), Some(ShapeKind::Circle));
    assert_eq!(scale.lookup("cat"), Some(ShapeKind::Cross));
    assert_ne!(scale.lookup("cat"), baseline.lookup("cat"), "glyph assignment must change");
}

/// #123 D1: an explicit `scale.range` of shape names assigns glyphs by name
/// (cycling positionally against the domain) instead of the fixed
/// `SHAPE_PALETTE` order.
#[test]
fn shape_scale_explicit_range_selects_the_named_glyphs() {
    let batch = species_batch();
    let outputs = std::collections::HashMap::new();
    let spec = make_spec_with_shape_scale(Some(ordinal_scale(
        Some(vec!["cat", "bird", "dog"]),
        Some(vec!["square", "cross", "diamond"]),
    )));
    let (scale, warns) = build_shape_scale(&spec.encoding, &batch, &outputs, false, false, &[]).unwrap();
    let scale = scale.unwrap();
    assert!(warns.is_empty(), "no palette-overflow warning for an explicit range: {warns:?}");
    assert_eq!(scale.domain, vec!["cat", "bird", "dog"]);
    assert_eq!(scale.lookup("cat"), Some(ShapeKind::Square));
    assert_eq!(scale.lookup("bird"), Some(ShapeKind::Cross));
    assert_eq!(scale.lookup("dog"), Some(ShapeKind::Diamond));
}

/// #123: an unrecognized range entry refuses with
/// `RenderError::ShapeRangeEntryUnrecognized`, naming the entry, rather than
/// silently drawing it as a circle. This is the hand-built-spec defense
/// behind Python's own eager `_VALID_POINT_SHAPES` gate at
/// `Shape(scale=...)` construction.
#[test]
fn shape_scale_unknown_range_entry_refuses() {
    let batch = species_batch();
    let outputs = std::collections::HashMap::new();
    let spec = make_spec_with_shape_scale(Some(ordinal_scale(
        Some(vec!["cat", "bird", "dog"]),
        Some(vec!["square", "hexagon", "diamond"]),
    )));
    let err = build_shape_scale(&spec.encoding, &batch, &outputs, false, false, &[]).unwrap_err();
    assert_eq!(
        err,
        crate::render::RenderError::ShapeRangeEntryUnrecognized { entry: "hexagon".to_string() },
        "must refuse naming the offending entry"
    );
}

/// #123: a hand-built range that mixes in a NON-STRING entry (e.g. a number
/// where a shape name belongs) must refuse, not fall through to the default
/// palette: a declared-but-malformed range is not an absent one.
#[test]
fn shape_scale_non_string_range_entry_refuses_rather_than_silently_dropping() {
    let batch = species_batch();
    let outputs = std::collections::HashMap::new();
    let scale: crate::spec::encoding::ScaleSpec = serde_json::from_value(serde_json::json!({
        "type": "ordinal",
        "domain": ["cat", "bird", "dog"],
        "range": ["square", 4, "diamond"],
    }))
    .expect("ordinal scale spec must deserialize");
    let spec = make_spec_with_shape_scale(Some(scale));
    let err = build_shape_scale(&spec.encoding, &batch, &outputs, false, false, &[]).unwrap_err();
    assert_eq!(
        err,
        crate::render::RenderError::ShapeRangeEntryUnrecognized { entry: "4".to_string() },
        "a numeric range entry must refuse, not silently fall through to the default palette"
    );
}

/// #123: a range shorter than the resolved domain cycles positionally
/// (`range[i % range.len()]`), mirroring `SHAPE_PALETTE`'s own wraparound —
/// the explicit-range path is a drop-in replacement for the default
/// palette's indexing rule, not a different one.
#[test]
fn shape_scale_range_shorter_than_domain_cycles() {
    use arrow::array::StringArray;
    use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ArrowDataType::Float64, false),
        Field::new("y", ArrowDataType::Float64, false),
        Field::new("species", ArrowDataType::Utf8, false),
    ]));
    // 4 distinct categories, a 2-entry range.
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0])),
            Arc::new(StringArray::from(vec!["cat", "bird", "dog", "fox"])),
        ],
    )
    .unwrap();
    let outputs = std::collections::HashMap::new();
    let spec = make_spec_with_shape_scale(Some(ordinal_scale(
        Some(vec!["cat", "bird", "dog", "fox"]),
        Some(vec!["square", "cross"]),
    )));
    let (scale, warns) = build_shape_scale(&spec.encoding, &batch, &outputs, false, false, &[]).unwrap();
    let scale = scale.unwrap();
    assert!(warns.is_empty(), "no palette-overflow warning for an explicit range: {warns:?}");
    assert_eq!(scale.lookup("cat"), Some(ShapeKind::Square));
    assert_eq!(scale.lookup("bird"), Some(ShapeKind::Cross));
    assert_eq!(scale.lookup("dog"), Some(ShapeKind::Square), "cycles back to range[0]");
    assert_eq!(scale.lookup("fox"), Some(ShapeKind::Cross), "cycles back to range[1]");
}

/// #123: a range with NO declared domain still resolves — the domain comes
/// from data first-appearance order (exactly as the unscaled default path
/// does), and the range assigns positionally against that data-derived
/// order, not against any name-to-shape mapping.
#[test]
fn shape_scale_range_with_no_declared_domain_uses_data_order() {
    let batch = species_batch(); // first-appearance order: cat, bird, dog
    let outputs = std::collections::HashMap::new();
    let spec = make_spec_with_shape_scale(Some(ordinal_scale(
        None,
        Some(vec!["diamond", "cross", "square"]),
    )));
    let (scale, warns) = build_shape_scale(&spec.encoding, &batch, &outputs, false, false, &[]).unwrap();
    let scale = scale.unwrap();
    assert!(warns.is_empty(), "no palette-overflow warning for an explicit range: {warns:?}");
    assert_eq!(scale.domain, vec!["cat", "bird", "dog"], "domain falls back to data first-appearance order");
    assert_eq!(scale.lookup("cat"), Some(ShapeKind::Diamond));
    assert_eq!(scale.lookup("bird"), Some(ShapeKind::Cross));
    assert_eq!(scale.lookup("dog"), Some(ShapeKind::Square));
}

/// #123: an explicit `scale.domain` composes with `sort=` rather than one
/// overriding the other — `sorted_categorical_domain` applies `sort` on top
/// of WHATEVER domain it resolved, declared or data-derived.
///
/// Uses the explicit-ARRAY sort form, not `"ascending"`/`"descending"`: an
/// alphabetical sort re-derives the final order from the domain's SET of
/// values alone, so it produces the identical result whether the pre-sort
/// domain was the declared order or the data-derived order — it cannot
/// discriminate the two. The explicit-array form naming only a STRICT
/// SUBSET of the domain does discriminate: values not named keep their
/// *relative order from whatever domain was fed in*, so a declared domain
/// with a different bird/cat relative order than the data (`cat, bird, dog`)
/// produces a different leftover order than falling back to data order
/// would. Declared `["dog", "bird", "cat"]`, `sort=["dog"]`: honoring the
/// declared domain must leave `bird` before `cat` (their declared relative
/// order); silently falling back to data order instead would leave `cat`
/// before `bird` (the data's relative order) — `["dog", "cat", "bird"]`.
#[test]
fn shape_scale_explicit_domain_composes_with_sort() {
    let batch = species_batch(); // data order: cat, bird, dog
    let outputs = std::collections::HashMap::new();
    // `ordinal_scale` has no `sort` slot — `sort` lives on the encoding, not
    // the scale spec — so it is set directly on the built spec below.
    let scale = ordinal_scale(Some(vec!["dog", "bird", "cat"]), None);
    let mut spec = make_spec_with_shape_scale(Some(scale));
    spec.encoding.shape.as_mut().unwrap().sort = Some(serde_json::json!(["dog"]));
    let (scale, warns) = build_shape_scale(&spec.encoding, &batch, &outputs, false, false, &[]).unwrap();
    let scale = scale.unwrap();
    assert!(warns.is_empty(), "no warnings expected: {warns:?}");
    assert_eq!(
        scale.domain,
        vec!["dog", "bird", "cat"],
        "sort= must apply on top of the DECLARED domain's relative order for \
         leftover categories, not the data-derived order"
    );
}

/// #123: no `scale=` at all must resolve byte-identically to the pre-#123
/// behavior — the fixed `SHAPE_PALETTE` in first-appearance order, no
/// warnings. Guards against the D1 domain/range plumbing changing the
/// unscaled default path.
#[test]
fn shape_scale_absent_is_byte_identical_to_pre_123() {
    let batch = species_batch();
    let outputs = std::collections::HashMap::new();
    let spec = make_spec_with_shape_scale(None);
    let (scale, warns) = build_shape_scale(&spec.encoding, &batch, &outputs, false, false, &[]).unwrap();
    let scale = scale.unwrap();
    assert!(warns.is_empty());
    assert_eq!(scale.domain, vec!["cat", "bird", "dog"]);
    assert_eq!(scale.shapes, vec![ShapeKind::Circle, ShapeKind::Square, ShapeKind::Cross]);
}

/// #123 D1: an explicit `scale.domain` reorders the dash-pattern assignment,
/// mirroring shape's own domain-reorder cell.
#[test]
fn stroke_dash_scale_explicit_domain_reassigns_pattern_assignment() {
    let batch = appearance_batch();
    let outputs = std::collections::HashMap::new();
    let baseline_spec = make_spec_with_stroke_dash_scale(None);
    let (baseline, _) =
        build_stroke_dash_scale(&baseline_spec.encoding, &batch, &outputs, false, false, &[]).unwrap();
    let baseline = baseline.unwrap();
    assert_eq!(baseline.domain, vec!["low", "mid", "high"], "sanity: first-appearance order");

    let spec =
        make_spec_with_stroke_dash_scale(Some(ordinal_scale(Some(vec!["mid", "low", "high"]), None)));
    let (scale, warns) = build_stroke_dash_scale(&spec.encoding, &batch, &outputs, false, false, &[]).unwrap();
    let scale = scale.unwrap();
    assert!(warns.is_empty(), "no warnings expected: {warns:?}");
    assert_eq!(scale.domain, vec!["mid", "low", "high"], "declared domain wins over first-appearance order");
    assert_ne!(
        scale.dash_for("low"),
        baseline.dash_for("low"),
        "dash-pattern assignment must change"
    );
}

/// #123: an explicit `scale.range` on `stroke_dash` refuses outright — the
/// categorical scale's output is a fixed set of `STROKE_DASH_LEGACY_SLOTS`
/// palette patterns (not `DASH_PALETTE`'s full, longer table), not a value a
/// scale can redeclare. Python's own `StrokeDash._validate` refuses this at
/// construction, so this exercises the hand-built-spec defense
/// (`ChartSpec::from_json` and similar raw-spec routes).
#[test]
fn stroke_dash_scale_explicit_range_refuses() {
    let batch = appearance_batch();
    let outputs = std::collections::HashMap::new();
    let spec = make_spec_with_stroke_dash_scale(Some(ordinal_scale(None, Some(vec!["a", "b"]))));
    let err = build_stroke_dash_scale(&spec.encoding, &batch, &outputs, false, false, &[]).unwrap_err();
    assert_eq!(
        err,
        crate::render::RenderError::StrokeDashRangeUnsupported {
            entries: vec!["a".to_string(), "b".to_string()],
        },
        "must refuse, naming the received range verbatim"
    );
}

/// #123: no `scale=` at all must resolve byte-identically to the pre-#123
/// behavior for stroke_dash too.
#[test]
fn stroke_dash_scale_absent_is_byte_identical_to_pre_123() {
    let batch = appearance_batch();
    let outputs = std::collections::HashMap::new();
    let spec = make_spec_with_stroke_dash_scale(None);
    let (scale, warns) = build_stroke_dash_scale(&spec.encoding, &batch, &outputs, false, false, &[]).unwrap();
    let scale = scale.unwrap();
    assert!(warns.is_empty());
    assert_eq!(scale.domain, vec!["low", "mid", "high"]);
}

/// #123: an explicit `scale.domain` composes with `sort=` for stroke-dash
/// too, mirroring [`shape_scale_explicit_domain_composes_with_sort`] — the
/// same `sorted_categorical_domain` helper both channels call.
#[test]
fn stroke_dash_scale_explicit_domain_composes_with_sort() {
    let batch = appearance_batch(); // data order: low, mid, high
    let outputs = std::collections::HashMap::new();
    // Same explicit-array discriminator as `shape_scale_explicit_domain_composes_with_sort`
    // (see its doc comment): `sort=["high"]` forces "high" first; whether the
    // leftover ("mid", "low") keeps ITS DECLARED relative order or falls back
    // to the data's relative order ("low", "mid") is what distinguishes a
    // real domain hand-off from a silently ignored one.
    let scale = ordinal_scale(Some(vec!["mid", "low", "high"]), None);
    let mut spec = make_spec_with_stroke_dash_scale(Some(scale));
    spec.encoding.stroke_dash.as_mut().unwrap().sort = Some(serde_json::json!(["high"]));
    let (scale, warns) = build_stroke_dash_scale(&spec.encoding, &batch, &outputs, false, false, &[]).unwrap();
    let scale = scale.unwrap();
    assert!(warns.is_empty(), "no warnings expected: {warns:?}");
    assert_eq!(
        scale.domain,
        vec!["high", "mid", "low"],
        "sort= must apply on top of the DECLARED domain's relative order for \
         leftover categories, not the data-derived order"
    );
}

/// #123: pins the exact `Display` text of each aux-range refusal on its own.
#[test]
fn shape_range_entry_unrecognized_display_text_is_pinned() {
    let err = crate::render::RenderError::ShapeRangeEntryUnrecognized { entry: "hexagon".to_string() };
    assert_eq!(err.to_string(), "Shape(scale=...) range entry 'hexagon' is not a recognized shape name");
}

/// Pins the exact text for a multi-entry received range: the message names
/// the value received, as `ShapeRangeEntryUnrecognized`'s does.
#[test]
fn stroke_dash_range_unsupported_display_text_is_pinned() {
    let err = crate::render::RenderError::StrokeDashRangeUnsupported {
        entries: vec!["a".to_string(), "b".to_string()],
    };
    assert_eq!(
        err.to_string(),
        "StrokeDash(scale=...) does not honor an explicit range (received ['a', 'b']): \
         the categorical scale's output is a fixed set of 4 palette patterns, not a value a scale can redeclare"
    );
}

/// The empty-entries edge case still names an (empty) received list rather
/// than panicking or omitting the parenthetical — `entries` is built from
/// whatever `explicit_ordinal_range` returns, which is never empty in
/// practice (`ScaleSpec::Ordinal { range: Some(r), .. } if !r.is_empty()`),
/// but `Display` itself makes no such assumption.
#[test]
fn stroke_dash_range_unsupported_display_text_handles_empty_entries() {
    let err = crate::render::RenderError::StrokeDashRangeUnsupported { entries: vec![] };
    assert_eq!(
        err.to_string(),
        "StrokeDash(scale=...) does not honor an explicit range (received []): \
         the categorical scale's output is a fixed set of 4 palette patterns, not a value a scale can redeclare"
    );
}

// --- Phase 8b: y2/x2 extends the primary axis domain ---

#[test]
fn y2_field_extends_y_domain_when_set() {
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
        mark: Mark::Ribbon,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "t".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "lo".into(), type_: None, ..Default::default() }),
            y2: Some(EncodingSpec { field: "hi".into(), type_: None, ..Default::default() }),
            ..Default::default()
        },
        ..Default::default()
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
        mark: Mark::Ribbon,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "xa".into(), type_: None, ..Default::default() }),
            x2: Some(EncodingSpec { field: "xb".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            ..Default::default()
        },
        ..Default::default()
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
        mark: Mark::Ribbon,
        encoding: Encoding {
            x:  Some(EncodingSpec { field: "a".into(),  type_: None, ..Default::default() }),
            x2: Some(EncodingSpec { field: "a2".into(), type_: None, ..Default::default() }),
            y:  Some(EncodingSpec { field: "b".into(),  type_: None, ..Default::default() }),
            y2: Some(EncodingSpec { field: "b2".into(), type_: None, ..Default::default() }),
            ..Default::default()
        },
        ..Default::default()
    };
    let theme = ThemeInputs::default();
    let (scales, _) =
        resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
    assert_eq!(scales.x2.as_deref(), Some("a2"));
    assert_eq!(scales.y2.as_deref(), Some("b2"));
}

#[test]
fn resolved_scales_x2_y2_default_to_none_for_8a_charts() {
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
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "a".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "b".into(), type_: None, ..Default::default() }),
            ..Default::default()
        },
        ..Default::default()
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
    let outputs: std::collections::HashMap<String, RecordBatch> = std::collections::HashMap::new();
    let (scale, warns) = build_opacity_channel_scale(
        OpacityChannel::Opacity,
        &make_spec_with_opacity().encoding,
        &batch,
        &outputs,
        false,
        &theme,
        &[],
    )
    .unwrap();
    assert!(warns.is_empty());
    let scale = scale.unwrap();
    assert_eq!(scale.min_opacity(), 0.1);
    assert_eq!(scale.max_opacity(), 1.0);
}

/// `Opacity(scale=LogScale(...))` resolves through the same gate and the
/// same `ScaleKind::Log` path `build_size_scale` does, with its own
/// `DrawableInterval::Alpha` clamp (the opacity band, not the size band):
/// an explicit `[0.2, 0.6]` range is honored and a value outside the
/// explicit domain clamps to the nearer end, never dropped.
#[test]
fn a_log_opacity_scale_resolves_to_scale_kind_log_and_clamps() {
    use crate::spec::encoding::{ContinuousScaleCommon, ScaleSpec};
    let batch = make_batch_q_q_n_n_q();
    let theme = ThemeInputs::default();
    let outputs: std::collections::HashMap<String, RecordBatch> = std::collections::HashMap::new();
    let mut spec = make_spec_with_opacity();
    spec.encoding.opacity.as_mut().unwrap().scale = Some(ScaleSpec::Log {
        base: 10.0,
        common: ContinuousScaleCommon {
            domain: Some(vec![0.2, 2.0]),
            range: Some(vec![0.2, 0.6]),
            clamp: false,
            padding: None,
            scheme: None,
            domain_param: None,
            reverse: false,
        },
        nice: false,
    });
    let (scale, warns) =
        build_opacity_channel_scale(OpacityChannel::Opacity, &spec.encoding, &batch, &outputs, false, &theme, &[])
            .unwrap();
    assert!(warns.is_empty());
    let scale = scale.unwrap();
    assert!(matches!(scale.inner, ScaleKind::Log(_)));
    assert_eq!(scale.inner.to_pixel_f64(0.2), Some(0.2));
    assert_eq!(scale.inner.to_pixel_f64(2.0), Some(0.6));
    assert_eq!(scale.inner.to_pixel_f64(0.01), Some(0.2), "below the domain clamps to the low alpha end");
    assert_eq!(scale.inner.to_pixel_f64(50.0), Some(0.6), "above the domain clamps to the high alpha end");
}

// --- Encoding resolution / inheritance edge-case tests ---

#[test]
fn single_value_domain_expands_to_symmetric_band() {
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
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            ..Default::default()
        },
        ..Default::default()
    };
    let theme = ThemeInputs::default();
    let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
    let px = scales.x.to_pixel_f64(5.0).expect("should return Some for constant domain");
    assert!(px.is_finite(), "pixel for constant-domain value must be finite, got: {px}");
    assert!((0.0..=100.0).contains(&px), "pixel should be within x range, got: {px}");
}

#[test]
fn single_value_zero_domain_expands_to_minus_one_plus_one() {
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
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            ..Default::default()
        },
        ..Default::default()
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
                        reverse: false,
                    },
                    nice: false,
                }),
                ..Default::default()
            }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            ..Default::default()
        },
        ..Default::default()
    };
    let theme = ThemeInputs::default();
    let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
    assert!(matches!(scales.x, ScaleKind::Symlog(_)), "expected Symlog, got: {:?}", scales.x);
}

#[test]
fn all_null_column_resolves_to_default_domain() {
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
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            ..Default::default()
        },
        ..Default::default()
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
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            color: Some(EncodingSpec { field: "c".into(), type_: None, ..Default::default() }),
            ..Default::default()
        },
        ..Default::default()
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
                        reverse: false,
                    },
                }),
                ..Default::default()
            }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            ..Default::default()
        },
        ..Default::default()
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
                        reverse: false,
                    },
                }),
                ..Default::default()
            }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            ..Default::default()
        },
        ..Default::default()
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

/// GH #104 quality-review remediation (S3-1): `tick_data` and
/// `project_values_to_fractions` (the test above) are sibling accessors on
/// the same `ScaleKind` and must agree on what a degenerate (zero-span)
/// domain renders. Before the `degenerate_ratio` fix, `scale_internal`
/// returned NaN for a degenerate domain, and every `tick_data` arm's
/// `is_finite()` filter dropped it — zero ticks. This pins that outcome
/// deliberately (rather than the alternative of emitting one tick at the
/// range midpoint) so a degenerate-domain axis's SVG output is unchanged by
/// #104's `scale()` fix — no golden re-bless required for this consequence.
#[test]
fn tick_data_on_zero_span_domain_returns_no_ticks() {
    use crate::scale::linear::LinearScale;
    // Degenerate domain: lo == hi (all-equal column / single distinct value).
    let scale = ScaleKind::Linear(LinearScale::new_internal(
        vec![3.0, 3.0],
        vec![0.0, 100.0],
        false,
        false,
    ));
    let ticks = scale.tick_data(10);
    assert!(
        ticks.is_empty(),
        "tick_data on a zero-span domain must stay empty (matching the pre-#104 \
         outcome and the project_values_to_fractions sibling policy), got {ticks:?}"
    );
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
        ..Default::default()
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
        ..Default::default()
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
        ..Default::default()
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

/// #126 (re-aimed from the former warn-and-fall-through contract): when any
/// color string in an explicit range fails to parse, the resolver must refuse
/// with a `RenderError::ColorRangeParseFailure` naming the offending entry.
/// It must NOT silently substitute the default theme palette for what the
/// user asked for.
#[test]
fn d1_invalid_color_in_range_refuses() {
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
        ..Default::default()
    };

    let theme = ThemeInputs::default();
    let err = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap_err();

    // Must refuse with a ColorRangeParseFailure naming the bad entry.
    assert_eq!(
        err,
        crate::render::RenderError::ColorRangeParseFailure { entry: "#not-a-color".into() },
        "must refuse naming the offending entry, not fall through to the theme palette"
    );
}

// ── D4: rect/heatmap color encoding honors scheme/cmap ────────────────────────

/// D4 regression: a color encoding with `scale={"type": "linear", "scheme": "blues"}`
/// must use the Blues continuous colormap. Prior to the fix, `ContinuousScaleCommon`
/// had no `scheme` field, so serde silently dropped the scheme and the resolver
/// fell back to the theme default, causing `cmap="blues"` and `cmap="rdbu"` to
/// produce byte-identical SVGs.
#[test]
fn d4_linear_scale_scheme_honored_in_continuous_color() {
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
        use crate::spec::encoding::{ContinuousScaleCommon, Encoding, EncodingSpec, ScaleSpec};
        use crate::spec::mark::Mark;
        ChartSpec {
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
                            reverse: false,
                        },
                        nice: false,
                        zero: false,
                        system_injected: false,
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            },
            ..Default::default()
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
        ..Default::default()
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
    use crate::spec::encoding::{DataType, Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    ChartSpec {
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
        ..Default::default()
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
        ..Default::default()
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
        ..Default::default()
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
/// #126 (re-aimed): a bad range entry used to fall through to a palette that
/// happened to match the no-range default — this test formerly pinned that
/// coincidence. There is no more fallback palette to compare against: the
/// resolver now refuses outright, naming the offending entry, and a
/// no-range spec is unaffected (still resolves the default palette).
#[test]
fn f5_parse_failure_refuses_and_default_is_unaffected() {
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
        ..Default::default()
    };

    let theme = ThemeInputs::default();

    let err = resolve_scales(&bad_range_spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme)
        .unwrap_err();
    assert_eq!(
        err,
        crate::render::RenderError::ColorRangeParseFailure { entry: "#not-a-color".into() },
    );

    // A no-range spec is unaffected: still resolves the default palette,
    // byte-identically to before this fix.
    let (scales_default, warnings_default) =
        resolve_scales(&make_spec_with_color(), &batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
    assert!(warnings_default.is_empty());
    assert!(matches!(scales_default.color, Some(ColorScale::Categorical { .. })));
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
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            ..Default::default()
        },
        ..Default::default()
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
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    ChartSpec {
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            ..Default::default()
        },
        facet: Some(FacetSpec {
            field: "panel".into(),
            row: None,
            title: None,
            row_title: None,
            sort: None,
            row_sort: None,
            repeat: Vec::new(),
            mode: FacetMode::Wrap { ncols: 2 },
            spacing: None,
            resolve: FacetResolve { x: x_resolve, y: y_resolve },
        }),
        ..Default::default()
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
        mark: Mark::Bar,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "cat".into(), type_: Some(SpecDataType::Nominal), ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            ..Default::default()
        },
        facet: Some(FacetSpec {
            field: "panel".into(),
            row: None,
            title: None,
            row_title: None,
            sort: None,
            row_sort: None,
            repeat: Vec::new(),
            mode: FacetMode::Wrap { ncols: 2 },
            spacing: None,
            resolve: FacetResolve { x: ResolveMode::Shared, y: ResolveMode::Shared },
        }),
        ..Default::default()
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
        mark: Mark::Bar,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "cat".into(), type_: Some(SpecDataType::Nominal), ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            ..Default::default()
        },
        facet: Some(FacetSpec {
            field: "panel".into(),
            row: None,
            title: None,
            row_title: None,
            sort: None,
            row_sort: None,
            repeat: Vec::new(),
            mode: FacetMode::Wrap { ncols: 2 },
            spacing: None,
            resolve: FacetResolve { x: ResolveMode::Independent, y: ResolveMode::Shared },
        }),
        ..Default::default()
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
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    ChartSpec {
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
        facet: Some(FacetSpec {
            field: "panel".into(),
            row: None,
            title: None,
            row_title: None,
            sort: None,
            row_sort: None,
            repeat: Vec::new(),
            mode: FacetMode::Wrap { ncols: 2 },
            spacing: None,
            resolve: FacetResolve { x: x_resolve, y: crate::layout::facet::ResolveMode::Shared },
        }),
        ..Default::default()
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

// ── T3: shared faceted auxiliary scale (size / continuous-color / opacity) ──────
//
// T3 mirrors T4 for non-positional channels. A faceted chart's size, continuous
// color, and opacity scales are built ONCE from the global data for the legend/
// colorbar (`prepare.rs`), but the per-panel marks were resolved from the
// per-panel batch. This meant a `sz=10` mark in a panel whose data range is
// [1,10] rendered at the SAME size as a `sz=100` mark in a panel with data range
// [1,100] — both normalized to their panel's max — even though the global legend
// clearly shows different sizes for 10 and 100.
//
// Fix: when `facet_shared` is true, union the per-panel extent with the global
// FINAL_OUTPUT_KEY batch, so per-panel marks normalize through the global domain.
//
// These tests drive `resolve_scales_with_outputs` with a per-panel batch as
// primary and the global all-panels batch under `__final__`, mirroring exactly
// how `scene_build.rs` calls it.

/// Build a faceted point spec on x/y with a size encoding.
fn faceted_size_spec() -> ChartSpec {
    use crate::layout::facet::{FacetMode, FacetResolve, FacetSpec};
    use crate::spec::encoding::{DataType, Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    ChartSpec {
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            size: Some(EncodingSpec {
                field: "sz".into(),
                type_: Some(DataType::Quantitative),
                ..Default::default()
            }),
            ..Default::default()
        },
        facet: Some(FacetSpec {
            field: "panel".into(),
            row: None,
            title: None,
            row_title: None,
            sort: None,
            row_sort: None,
            repeat: Vec::new(),
            mode: FacetMode::Wrap { ncols: 2 },
            spacing: None,
            resolve: FacetResolve::default(),
        }),
        ..Default::default()
    }
}

/// Build a faceted point spec on x/y with a continuous (Float64) color encoding.
fn faceted_continuous_color_spec() -> ChartSpec {
    use crate::layout::facet::{FacetMode, FacetResolve, FacetSpec};
    use crate::spec::encoding::{DataType, Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    ChartSpec {
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            color: Some(EncodingSpec {
                field: "c".into(),
                type_: Some(DataType::Quantitative),
                ..Default::default()
            }),
            ..Default::default()
        },
        facet: Some(FacetSpec {
            field: "panel".into(),
            row: None,
            title: None,
            row_title: None,
            sort: None,
            row_sort: None,
            repeat: Vec::new(),
            mode: FacetMode::Wrap { ncols: 2 },
            spacing: None,
            resolve: FacetResolve::default(),
        }),
        ..Default::default()
    }
}

/// Build a faceted point spec on x/y with an opacity encoding.
fn faceted_opacity_spec() -> ChartSpec {
    use crate::layout::facet::{FacetMode, FacetResolve, FacetSpec};
    use crate::spec::encoding::{DataType, Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    ChartSpec {
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            opacity: Some(EncodingSpec {
                field: "op".into(),
                type_: Some(DataType::Quantitative),
                ..Default::default()
            }),
            ..Default::default()
        },
        facet: Some(FacetSpec {
            field: "panel".into(),
            row: None,
            title: None,
            row_title: None,
            sort: None,
            row_sort: None,
            repeat: Vec::new(),
            mode: FacetMode::Wrap { ncols: 2 },
            spacing: None,
            resolve: FacetResolve::default(),
        }),
        ..Default::default()
    }
}

/// Build a 3-column batch (x, y, sz/c/op) for T3 tests.
fn t3_sz_batch(x_vals: Vec<f64>, y_vals: Vec<f64>, sz_vals: Vec<f64>) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x",  ArrowDataType::Float64, false),
        Field::new("y",  ArrowDataType::Float64, false),
        Field::new("sz", ArrowDataType::Float64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(x_vals)),
            Arc::new(Float64Array::from(y_vals)),
            Arc::new(Float64Array::from(sz_vals)),
        ],
    )
    .unwrap()
}

fn t3_c_batch(x_vals: Vec<f64>, y_vals: Vec<f64>, c_vals: Vec<f64>) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ArrowDataType::Float64, false),
        Field::new("y", ArrowDataType::Float64, false),
        Field::new("c", ArrowDataType::Float64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(x_vals)),
            Arc::new(Float64Array::from(y_vals)),
            Arc::new(Float64Array::from(c_vals)),
        ],
    )
    .unwrap()
}

fn t3_op_batch(x_vals: Vec<f64>, y_vals: Vec<f64>, op_vals: Vec<f64>) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x",  ArrowDataType::Float64, false),
        Field::new("y",  ArrowDataType::Float64, false),
        Field::new("op", ArrowDataType::Float64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(x_vals)),
            Arc::new(Float64Array::from(y_vals)),
            Arc::new(Float64Array::from(op_vals)),
        ],
    )
    .unwrap()
}

/// T3 size: panel A has sz∈[1,10]; panel B has sz∈[1,100]; global∈[1,100].
///
/// Before T3: panel A's sz=10 normalized to 1.0 (max_px) since 10 was the panel
/// max. After T3: panel A's sz=10 normalizes to the SAME pixel as panel B's
/// sz=10 (both use the global [1,100] domain), landing at ~10% of the size range.
#[test]
fn t3_shared_facet_size_unions_global_domain() {
    let panel_a = t3_sz_batch(vec![0.0, 1.0], vec![0.0, 1.0], vec![1.0, 10.0]);
    let panel_b = t3_sz_batch(vec![0.0, 1.0], vec![0.0, 1.0], vec![1.0, 100.0]);
    let global  = t3_sz_batch(
        vec![0.0, 1.0, 0.0, 1.0], vec![0.0, 1.0, 0.0, 1.0], vec![1.0, 10.0, 1.0, 100.0],
    );
    let outputs = final_outputs(global);
    let spec = faceted_size_spec();
    let theme = ThemeInputs::default();
    let pr = (0.0, 100.0);

    let (scales_a, _) =
        resolve_scales_with_outputs(&spec, &panel_a, &outputs, pr, pr, &theme).unwrap();
    let (scales_b, _) =
        resolve_scales_with_outputs(&spec, &panel_b, &outputs, pr, pr, &theme).unwrap();

    let size_a = scales_a.size.expect("panel A must have size scale");
    let size_b = scales_b.size.expect("panel B must have size scale");

    // Both panels must use the global domain [1, 100].
    let (a_lo, a_hi) = size_a.inner.data_domain().expect("size A domain");
    let (b_lo, b_hi) = size_b.inner.data_domain().expect("size B domain");
    assert!(
        a_lo <= 1.0 + 1e-6 && a_hi >= 100.0 - 1e-6,
        "T3: panel A size domain must span GLOBAL [1,100], got [{a_lo},{a_hi}]"
    );
    assert!(
        b_lo <= 1.0 + 1e-6 && b_hi >= 100.0 - 1e-6,
        "T3: panel B size domain must span GLOBAL [1,100], got [{b_lo},{b_hi}]"
    );

    // Mark-size discrimination: panel A's sz=10 must NOT be the max-pixel mark.
    // On the global [1,100] domain, sz=10 is ~10% of the way from min to max.
    // So its pixel radius must be far from max_px.
    let px_sz10_a = size_a.inner.to_pixel_f64(10.0).expect("sz=10 must project on panel A");
    let px_sz100_a = size_a.inner.to_pixel_f64(100.0).expect("sz=100 must project on panel A");
    assert!(
        px_sz10_a < 0.5 * (size_a.min_px() + size_a.max_px()),
        "T3: panel A sz=10 must be below the midpoint of the size range (global domain); \
         got px={px_sz10_a}, mid={}",
        0.5 * (size_a.min_px() + size_a.max_px())
    );
    assert!(
        (px_sz100_a - size_a.max_px()).abs() < 1e-6,
        "T3: sz=100 (global max) must map to max_px on panel A; got {px_sz100_a}"
    );

    // Panel A and B must agree: sz=10 maps to the same pixel on both (global domain).
    let px_sz10_b = size_b.inner.to_pixel_f64(10.0).expect("sz=10 must project on panel B");
    assert!(
        (px_sz10_a - px_sz10_b).abs() < 1e-6,
        "T3: sz=10 must map to the same pixel on both panels (shared global domain); \
         panel_A={px_sz10_a}, panel_B={px_sz10_b}"
    );
}

/// T3 continuous color: panel A has c∈[1,10]; panel B has c∈[1,100]; global∈[1,100].
///
/// Before T3: panel A's c=10 normalized to t=1.0 (scheme max color) since 10 was
/// the panel-A max. After T3: c=10 normalizes to t≈0.09 on the global [1,100]
/// domain, producing a much lighter color. Panel A and panel B must produce the
/// same color for c=10 (they use the same global domain).
#[test]
fn t3_shared_facet_continuous_color_unions_global_domain() {
    let panel_a = t3_c_batch(vec![0.0, 1.0], vec![0.0, 1.0], vec![1.0, 10.0]);
    let panel_b = t3_c_batch(vec![0.0, 1.0], vec![0.0, 1.0], vec![1.0, 100.0]);
    let global  = t3_c_batch(
        vec![0.0, 1.0, 0.0, 1.0], vec![0.0, 1.0, 0.0, 1.0], vec![1.0, 10.0, 1.0, 100.0],
    );
    let outputs = final_outputs(global);
    let spec = faceted_continuous_color_spec();
    let theme = ThemeInputs::default();
    let pr = (0.0, 100.0);

    let (scales_a, _) =
        resolve_scales_with_outputs(&spec, &panel_a, &outputs, pr, pr, &theme).unwrap();
    let (scales_b, _) =
        resolve_scales_with_outputs(&spec, &panel_b, &outputs, pr, pr, &theme).unwrap();

    let color_a = scales_a.color.expect("panel A must have color scale");
    let color_b = scales_b.color.expect("panel B must have color scale");

    // Both must be Continuous (Float64 field, Quantitative type).
    let (domain_a, _mid_a) = match &color_a {
        ColorScale::Continuous { domain, midpoint, .. } => (*domain, *midpoint),
        other => panic!("panel A color scale must be Continuous, got {other:?}"),
    };
    let (domain_b, _mid_b) = match &color_b {
        ColorScale::Continuous { domain, midpoint, .. } => (*domain, *midpoint),
        other => panic!("panel B color scale must be Continuous, got {other:?}"),
    };

    // Both panels must use the global domain [1, 100].
    assert!(
        domain_a.0 <= 1.0 + 1e-6 && domain_a.1 >= 100.0 - 1e-6,
        "T3: panel A color domain must span GLOBAL [1,100], got {:?}", domain_a
    );
    assert!(
        domain_b.0 <= 1.0 + 1e-6 && domain_b.1 >= 100.0 - 1e-6,
        "T3: panel B color domain must span GLOBAL [1,100], got {:?}", domain_b
    );

    // Mark-color discrimination: panel A's c=10 must NOT be the scheme maximum.
    // On global [1,100], c=10 → t ≈ (10-1)/(100-1) ≈ 0.091 (near the dark end of
    // viridis). The scheme max color (t=1.0) is the bright yellow. They must differ.
    let color_at_10_a = color_a.lookup_f64(10.0).expect("c=10 must project on panel A");
    let color_at_100_a = color_a.lookup_f64(100.0).expect("c=100 must project on panel A");
    // Viridis: t≈0.09 is dark purple; t=1.0 is bright yellow. They clearly differ.
    assert_ne!(
        (color_at_10_a.red, color_at_10_a.green, color_at_10_a.blue),
        (color_at_100_a.red, color_at_100_a.green, color_at_100_a.blue),
        "T3: c=10 and c=100 must produce different colors on global [1,100] domain"
    );

    // Panel A and B must produce the same color for c=10.
    let color_at_10_b = color_b.lookup_f64(10.0).expect("c=10 must project on panel B");
    assert_eq!(
        (color_at_10_a.red, color_at_10_a.green, color_at_10_a.blue),
        (color_at_10_b.red, color_at_10_b.green, color_at_10_b.blue),
        "T3: c=10 must produce the same color on both panels (shared global domain); \
         panel_A=({},{},{}) panel_B=({},{},{})",
        color_at_10_a.red, color_at_10_a.green, color_at_10_a.blue,
        color_at_10_b.red, color_at_10_b.green, color_at_10_b.blue
    );
}

/// T3 opacity: panel A has op∈[1,10]; panel B has op∈[1,100]; global∈[1,100].
///
/// Before T3: panel A's op=10 normalized to max_opacity since 10 was the panel max.
/// After T3: op=10 normalizes to ~10% of the opacity range on the global [1,100]
/// domain. Both panels must produce the same opacity for op=10.
#[test]
fn t3_shared_facet_opacity_unions_global_domain() {
    let panel_a = t3_op_batch(vec![0.0, 1.0], vec![0.0, 1.0], vec![1.0, 10.0]);
    let panel_b = t3_op_batch(vec![0.0, 1.0], vec![0.0, 1.0], vec![1.0, 100.0]);
    let global  = t3_op_batch(
        vec![0.0, 1.0, 0.0, 1.0], vec![0.0, 1.0, 0.0, 1.0], vec![1.0, 10.0, 1.0, 100.0],
    );
    let outputs = final_outputs(global);
    let spec = faceted_opacity_spec();
    let theme = ThemeInputs::default();
    let pr = (0.0, 100.0);

    let (scales_a, _) =
        resolve_scales_with_outputs(&spec, &panel_a, &outputs, pr, pr, &theme).unwrap();
    let (scales_b, _) =
        resolve_scales_with_outputs(&spec, &panel_b, &outputs, pr, pr, &theme).unwrap();

    let opacity_a = scales_a.opacity.expect("panel A must have opacity scale");
    let opacity_b = scales_b.opacity.expect("panel B must have opacity scale");

    // Both panels must use the global domain [1, 100].
    let (a_lo, a_hi) = opacity_a.inner.data_domain().expect("opacity A domain");
    let (b_lo, b_hi) = opacity_b.inner.data_domain().expect("opacity B domain");
    assert!(
        a_lo <= 1.0 + 1e-6 && a_hi >= 100.0 - 1e-6,
        "T3: panel A opacity domain must span GLOBAL [1,100], got [{a_lo},{a_hi}]"
    );
    assert!(
        b_lo <= 1.0 + 1e-6 && b_hi >= 100.0 - 1e-6,
        "T3: panel B opacity domain must span GLOBAL [1,100], got [{b_lo},{b_hi}]"
    );

    // Panel A's op=10 must NOT produce max_opacity under the global domain.
    let op_at_10_a = opacity_a.inner.to_pixel_f64(10.0)
        .expect("op=10 must project on panel A");
    let op_at_100_a = opacity_a.inner.to_pixel_f64(100.0)
        .expect("op=100 must project on panel A");
    assert!(
        op_at_10_a < 0.5 * (opacity_a.min_opacity() + opacity_a.max_opacity()),
        "T3: panel A op=10 must be below the midpoint of the opacity range (global domain); \
         got op={op_at_10_a}, mid={}",
        0.5 * (opacity_a.min_opacity() + opacity_a.max_opacity())
    );
    assert!(
        (op_at_100_a - opacity_a.max_opacity()).abs() < 1e-6,
        "T3: op=100 (global max) must map to max_opacity on panel A; got {op_at_100_a}"
    );

    // Panel A and B must produce the same opacity for op=10.
    let op_at_10_b = opacity_b.inner.to_pixel_f64(10.0)
        .expect("op=10 must project on panel B");
    assert!(
        (op_at_10_a - op_at_10_b).abs() < 1e-6,
        "T3: op=10 must map to the same opacity on both panels (shared global domain); \
         panel_A={op_at_10_a}, panel_B={op_at_10_b}"
    );
}

/// T3 non-faceted byte-identity: a non-faceted size/color/opacity chart must
/// resolve identically to the pre-T3 behavior (no global union). The per-panel
/// extent dominates, no FINAL_OUTPUT_KEY lookup occurs.
#[test]
fn t3_non_faceted_size_color_opacity_byte_identical() {
    // Non-faceted chart: primary batch has sz/c/op∈[1,10]; global batch has [1,100].
    // Before and after T3, the non-faceted path must NOT union the global batch,
    // so the domain stays [1,10] (per-panel is all data).
    let panel_only = {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x",  ArrowDataType::Float64, false),
            Field::new("y",  ArrowDataType::Float64, false),
            Field::new("sz", ArrowDataType::Float64, false),
            Field::new("c",  ArrowDataType::Float64, false),
            Field::new("op", ArrowDataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![0.0, 1.0])),
                Arc::new(Float64Array::from(vec![0.0, 1.0])),
                Arc::new(Float64Array::from(vec![1.0, 10.0])),
                Arc::new(Float64Array::from(vec![1.0, 10.0])),
                Arc::new(Float64Array::from(vec![1.0, 10.0])),
            ],
        )
        .unwrap()
    };
    // A global batch with a wider range (to confirm it's NOT unioned).
    let global = {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x",  ArrowDataType::Float64, false),
            Field::new("y",  ArrowDataType::Float64, false),
            Field::new("sz", ArrowDataType::Float64, false),
            Field::new("c",  ArrowDataType::Float64, false),
            Field::new("op", ArrowDataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![0.0, 1.0])),
                Arc::new(Float64Array::from(vec![0.0, 1.0])),
                Arc::new(Float64Array::from(vec![1.0, 100.0])),
                Arc::new(Float64Array::from(vec![1.0, 100.0])),
                Arc::new(Float64Array::from(vec![1.0, 100.0])),
            ],
        )
        .unwrap()
    };
    let outputs = final_outputs(global);

    // Use non-faceted specs (no facet field).
    use crate::spec::encoding::{DataType, Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    let spec = ChartSpec {
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            size: Some(EncodingSpec {
                field: "sz".into(),
                type_: Some(DataType::Quantitative),
                ..Default::default()
            }),
            color: Some(EncodingSpec {
                field: "c".into(),
                type_: Some(DataType::Quantitative),
                ..Default::default()
            }),
            opacity: Some(EncodingSpec {
                field: "op".into(),
                type_: Some(DataType::Quantitative),
                ..Default::default()
            }),
            ..Default::default()
        },  // non-faceted
        layers: None,
        ..Default::default()
    };
    let theme = ThemeInputs::default();
    let pr = (0.0, 100.0);

    let (scales, _) =
        resolve_scales_with_outputs(&spec, &panel_only, &outputs, pr, pr, &theme).unwrap();

    // Size domain must be [1, 10] (per-panel only, no global union).
    let size = scales.size.expect("must have size scale");
    let (sz_lo, sz_hi) = size.inner.data_domain().expect("size domain");
    assert!(
        (sz_lo - 1.0).abs() < 1e-6 && (sz_hi - 10.0).abs() < 1e-6,
        "T3 non-faceted: size domain must be PER-PANEL [1,10], not global [1,100]; \
         got [{sz_lo},{sz_hi}]"
    );

    // Continuous color domain must be [1, 10] (per-panel only).
    let color = scales.color.expect("must have color scale");
    let domain_c = match &color {
        ColorScale::Continuous { domain, .. } => *domain,
        other => panic!("expected Continuous color, got {other:?}"),
    };
    assert!(
        (domain_c.0 - 1.0).abs() < 1e-6 && (domain_c.1 - 10.0).abs() < 1e-6,
        "T3 non-faceted: color domain must be PER-PANEL [1,10], not global [1,100]; \
         got {:?}", domain_c
    );

    // Opacity domain must be [1, 10] (per-panel only).
    let opacity = scales.opacity.expect("must have opacity scale");
    let (op_lo, op_hi) = opacity.inner.data_domain().expect("opacity domain");
    assert!(
        (op_lo - 1.0).abs() < 1e-6 && (op_hi - 10.0).abs() < 1e-6,
        "T3 non-faceted: opacity domain must be PER-PANEL [1,10], not global [1,100]; \
         got [{op_lo},{op_hi}]"
    );
}

// ── T3-cat: categorical faceted color uses global first-appearance order ──────

/// Build a batch with (x, y, k) columns for categorical-color T3 tests.
fn t3_cat_batch(x_vals: Vec<f64>, y_vals: Vec<f64>, k_vals: Vec<&str>) -> RecordBatch {
    use arrow::array::StringArray;
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ArrowDataType::Float64, false),
        Field::new("y", ArrowDataType::Float64, false),
        Field::new("k", ArrowDataType::Utf8, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(x_vals)),
            Arc::new(Float64Array::from(y_vals)),
            Arc::new(StringArray::from(k_vals)),
        ],
    )
    .unwrap()
}

/// Build a faceted point spec on x/y with a nominal (Utf8) color encoding.
fn faceted_categorical_color_spec() -> ChartSpec {
    use crate::layout::facet::{FacetMode, FacetResolve, FacetSpec};
    use crate::spec::encoding::{DataType, Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    ChartSpec {
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            color: Some(EncodingSpec {
                field: "k".into(),
                type_: Some(DataType::Nominal),
                ..Default::default()
            }),
            ..Default::default()
        },
        facet: Some(FacetSpec {
            field: "panel".into(),
            row: None,
            title: None,
            row_title: None,
            sort: None,
            row_sort: None,
            repeat: Vec::new(),
            mode: FacetMode::Wrap { ncols: 2 },
            spacing: None,
            resolve: FacetResolve::default(),
        }),
        ..Default::default()
    }
}

/// T3-cat: categorical color is assigned by GLOBAL first-appearance order so
/// the same category string maps to the same palette color in every panel.
///
/// Reproduction of the reviewer's green/#2563eb-vs-#dc2626 scenario:
/// - Panel A rows arrive in order ["green", "blue"] → per-panel: green=0, blue=1.
/// - Panel B rows arrive in order ["blue", "green"] → per-panel: blue=0, green=1.
/// Without the fix: panel A gives green→palette[0] and panel B gives green→palette[1]
/// (different colors). With the fix: both panels use the global batch whose first-
/// appearance order is ["green", "blue"] → green=0 in both panels (same color).
///
/// fail-before/pass-after: confirmed by temporarily swapping domain_batch back
/// to primary_batch and observing palette[0] ≠ palette[1].
#[test]
fn t3_categorical_color_faceted_uses_global_first_appearance_order() {
    // Panel A: rows arrive green first, then blue.
    let panel_a = t3_cat_batch(
        vec![0.0, 1.0], vec![0.0, 1.0], vec!["green", "blue"],
    );
    // Panel B: rows arrive blue first, then green — reversed order.
    let panel_b = t3_cat_batch(
        vec![0.0, 1.0], vec![0.0, 1.0], vec!["blue", "green"],
    );
    // Global batch: all rows concatenated, first appearance is green (from panel A rows first).
    let global = t3_cat_batch(
        vec![0.0, 1.0, 0.0, 1.0],
        vec![0.0, 1.0, 0.0, 1.0],
        vec!["green", "blue", "blue", "green"],
    );
    let outputs = final_outputs(global);
    let spec = faceted_categorical_color_spec();
    let theme = ThemeInputs::default();
    let pr = (0.0, 100.0);

    let (scales_a, _) =
        resolve_scales_with_outputs(&spec, &panel_a, &outputs, pr, pr, &theme).unwrap();
    let (scales_b, _) =
        resolve_scales_with_outputs(&spec, &panel_b, &outputs, pr, pr, &theme).unwrap();

    let color_a = scales_a.color.expect("panel A must have color scale");
    let color_b = scales_b.color.expect("panel B must have color scale");

    // Both panels must use the GLOBAL first-appearance domain ["green", "blue"].
    let (domain_a, palette_a) = match &color_a {
        ColorScale::Categorical { domain, palette } => (domain.clone(), palette.clone()),
        other => panic!("panel A color must be Categorical, got {other:?}"),
    };
    let (domain_b, palette_b) = match &color_b {
        ColorScale::Categorical { domain, palette } => (domain.clone(), palette.clone()),
        other => panic!("panel B color must be Categorical, got {other:?}"),
    };

    // Both domains must equal the global first-appearance order: ["green", "blue"].
    assert_eq!(
        domain_a, vec!["green".to_string(), "blue".to_string()],
        "T3-cat: panel A domain must follow GLOBAL first-appearance order [green, blue]; got {domain_a:?}"
    );
    assert_eq!(
        domain_b, vec!["green".to_string(), "blue".to_string()],
        "T3-cat: panel B domain must follow GLOBAL first-appearance order [green, blue]; got {domain_b:?}"
    );

    // "green" must map to the SAME palette color in both panels.
    // Both panels have domain = ["green", "blue"], so green → index 0.
    let green_a = palette_a[0];
    let green_b = palette_b[0];
    assert_eq!(
        (green_a.red, green_a.green, green_a.blue),
        (green_b.red, green_b.green, green_b.blue),
        "T3-cat: 'green' must map to the same palette color in both panels (shared global domain); \
         panel_A=rgb({},{},{}) panel_B=rgb({},{},{})",
        green_a.red, green_a.green, green_a.blue,
        green_b.red, green_b.green, green_b.blue,
    );

    // Sanity: "blue" must map to palette[1] (second position in global order).
    let blue_a = palette_a[1];
    let blue_b = palette_b[1];
    assert_eq!(
        (blue_a.red, blue_a.green, blue_a.blue),
        (blue_b.red, blue_b.green, blue_b.blue),
        "T3-cat: 'blue' must map to the same palette color in both panels; \
         panel_A=rgb({},{},{}) panel_B=rgb({},{},{})",
        blue_a.red, blue_a.green, blue_a.blue,
        blue_b.red, blue_b.green, blue_b.blue,
    );

    // The two categories must have DIFFERENT colors (palette[0] ≠ palette[1]).
    assert_ne!(
        (green_a.red, green_a.green, green_a.blue),
        (blue_a.red, blue_a.green, blue_a.blue),
        "T3-cat: 'green' and 'blue' must be assigned different palette colors; \
         both got rgb({},{},{})", green_a.red, green_a.green, green_a.blue,
    );
}

/// Minimal non-faceted x/y point spec for the byte-identity guard.
fn make_spec_with_color_xy_only() -> ChartSpec {
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    ChartSpec {
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            ..Default::default()
        },
        ..Default::default()
    }
}

// ── T3-shape: faceted SHAPE encoding uses global first-appearance order ────────
//
// The last data-driven channel to close the faceted shared-scale class.
// build_shape_scale was building its domain from the per-panel batch, causing
// the same category to map to a DIFFERENT glyph in panels whose local
// first-appearance order differed from the global order.  The shape legend
// is built globally (from FINAL_OUTPUT_KEY), so panels and legend disagree.
//
// Fix: shared_categorical_batch selects the global batch when facet_shared=true,
// exactly as categorical color does via the same helper.

/// Build a (x, y, grp) batch for T3-shape tests.
fn t3_shape_batch(x_vals: Vec<f64>, y_vals: Vec<f64>, grp_vals: Vec<&str>) -> RecordBatch {
    use arrow::array::StringArray;
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ArrowDataType::Float64, false),
        Field::new("y", ArrowDataType::Float64, false),
        Field::new("grp", ArrowDataType::Utf8, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(x_vals)),
            Arc::new(Float64Array::from(y_vals)),
            Arc::new(StringArray::from(grp_vals)),
        ],
    )
    .unwrap()
}

/// Build a faceted point spec on x/y with a nominal shape encoding.
fn faceted_shape_spec() -> ChartSpec {
    use crate::layout::facet::{FacetMode, FacetResolve, FacetSpec};
    use crate::spec::encoding::{DataType, Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    ChartSpec {
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            shape: Some(EncodingSpec {
                field: "grp".into(),
                type_: Some(DataType::Nominal),
                ..Default::default()
            }),
            ..Default::default()
        },
        facet: Some(FacetSpec {
            field: "panel".into(),
            row: None,
            title: None,
            row_title: None,
            sort: None,
            row_sort: None,
            repeat: Vec::new(),
            mode: FacetMode::Wrap { ncols: 2 },
            spacing: None,
            resolve: FacetResolve::default(),
        }),
        ..Default::default()
    }
}

/// T3-shape: shape domain is assigned by GLOBAL first-appearance order so
/// the same category string maps to the same glyph in every panel.
///
/// Reproduction scenario:
/// - Global order: x, y, z  →  {x:Circle, y:Square, z:Cross}
/// - Panel A rows: grp=[x, y]    → per-panel: x=0 (Circle), y=1 (Square)
/// - Panel B rows: grp=[y, z]    → per-panel: y=0 (Circle), z=1 (Square)
///
/// Without the fix: panel A gives y→Square, panel B gives y→Circle.
/// With the fix: both panels use the global domain [x, y, z] so y→Square
/// in every panel.
///
/// fail-before/pass-after: confirmed by the structural analysis — before this
/// fix, `build_shape_scale` called `distinct_values_in_order(batch, ...)` on
/// the per-panel batch (no facet_shared param), which was identical to the
/// categorical-color bug that f13b7da fixed. The test asserts the shape domain
/// equals global first-appearance order in BOTH panels, which is impossible
/// under the per-panel path for panel B (its local order is [y, z], not [x, y, z]).
#[test]
fn t3_shape_faceted_uses_global_first_appearance_order() {
    // Panel A: rows arrive in order x, y (local domain = [x, y]).
    let panel_a = t3_shape_batch(vec![0.0, 1.0], vec![0.0, 1.0], vec!["x", "y"]);
    // Panel B: rows arrive in order y, z — reversed/disjoint (local domain = [y, z]).
    let panel_b = t3_shape_batch(vec![0.0, 1.0], vec![0.0, 1.0], vec!["y", "z"]);
    // Global batch: all rows concatenated; first-appearance order = [x, y, z].
    let global = t3_shape_batch(
        vec![0.0, 1.0, 0.0, 1.0],
        vec![0.0, 1.0, 0.0, 1.0],
        vec!["x", "y", "y", "z"],
    );
    let outputs = final_outputs(global);
    let spec = faceted_shape_spec();
    let theme = ThemeInputs::default();
    let pr = (0.0, 100.0);

    let (scales_a, _) =
        resolve_scales_with_outputs(&spec, &panel_a, &outputs, pr, pr, &theme).unwrap();
    let (scales_b, _) =
        resolve_scales_with_outputs(&spec, &panel_b, &outputs, pr, pr, &theme).unwrap();

    let shape_a = scales_a.shape.expect("panel A must have shape scale");
    let shape_b = scales_b.shape.expect("panel B must have shape scale");

    // Both panels must use the GLOBAL first-appearance domain [x, y, z].
    assert_eq!(
        shape_a.domain,
        vec!["x".to_string(), "y".to_string(), "z".to_string()],
        "T3-shape: panel A domain must follow GLOBAL first-appearance order [x, y, z]; got {:?}",
        shape_a.domain,
    );
    assert_eq!(
        shape_b.domain,
        vec!["x".to_string(), "y".to_string(), "z".to_string()],
        "T3-shape: panel B domain must follow GLOBAL first-appearance order [x, y, z]; got {:?}",
        shape_b.domain,
    );

    // "y" must map to the SAME glyph in both panels.
    // Global domain: x=0 (Circle), y=1 (Square), z=2 (Cross).
    let glyph_y_a = shape_a.lookup("y").expect("y must be in panel A shape domain");
    let glyph_y_b = shape_b.lookup("y").expect("y must be in panel B shape domain");
    assert_eq!(
        glyph_y_a, glyph_y_b,
        "T3-shape: 'y' must map to the same glyph in both panels (shared global domain); \
         panel_A={:?} panel_B={:?}",
        glyph_y_a, glyph_y_b,
    );
    // Confirm it's the expected glyph: global index 1 → Square.
    assert_eq!(
        glyph_y_a,
        ShapeKind::Square,
        "T3-shape: 'y' must be at global index 1 (Square); got {:?}",
        glyph_y_a,
    );

    // Sanity: all three categories must have distinct glyphs.
    let glyph_x = shape_a.lookup("x").expect("x in panel A");
    let glyph_z = shape_b.lookup("z").expect("z in panel B");
    assert_eq!(glyph_x, ShapeKind::Circle, "T3-shape: 'x' must be global index 0 (Circle)");
    assert_eq!(glyph_z, ShapeKind::Cross, "T3-shape: 'z' must be global index 2 (Cross)");
}

/// T3-shape non-faceted byte-identity guard: without a facet spec, the shape
/// domain must be the per-panel first-appearance order, identical to pre-fix
/// behavior. The presence of a wider global batch in transform_outputs must NOT
/// affect a non-faceted chart's shape domain.
#[test]
fn t3_shape_non_faceted_byte_identical() {
    use crate::spec::encoding::{DataType, Encoding, EncodingSpec};
    use crate::spec::mark::Mark;

    // Per-panel batch: local order is [y, x] (y appears first).
    let panel = t3_shape_batch(vec![0.0, 1.0], vec![0.0, 1.0], vec!["y", "x"]);
    // Global batch has a wider domain [x, y, z] with x first — but non-faceted
    // charts must NOT use it.
    let global = t3_shape_batch(
        vec![0.0, 1.0, 2.0],
        vec![0.0, 1.0, 2.0],
        vec!["x", "y", "z"],
    );
    let outputs = final_outputs(global);

    // Non-faceted spec: no facet field.
    let spec = ChartSpec {
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            shape: Some(EncodingSpec {
                field: "grp".into(),
                type_: Some(DataType::Nominal),
                ..Default::default()
            }),
            ..Default::default()
        }, // non-faceted
        layers: None,
        ..Default::default()
    };
    let theme = ThemeInputs::default();
    let pr = (0.0, 100.0);

    let (scales, _) =
        resolve_scales_with_outputs(&spec, &panel, &outputs, pr, pr, &theme).unwrap();

    let shape = scales.shape.expect("shape scale must be resolved");

    // Non-faceted: domain must be PER-PANEL first-appearance order [y, x],
    // NOT the global [x, y, z].
    assert_eq!(
        shape.domain,
        vec!["y".to_string(), "x".to_string()],
        "T3-shape non-faceted: domain must be PER-PANEL order [y, x], not global [x, y, z]; \
         got {:?}",
        shape.domain,
    );
    // "y" is at local index 0 → Circle.
    let glyph_y = shape.lookup("y").expect("y must be in domain");
    assert_eq!(
        glyph_y,
        ShapeKind::Circle,
        "T3-shape non-faceted: 'y' must be at per-panel index 0 (Circle); got {:?}",
        glyph_y,
    );
}

// ── T1.9 (SPINE-11): dummy_unit_scale range-ordering contract ───────────────

/// `dummy_unit_scale` with `ascending=true` must produce a LinearScale whose
/// pixel range is `[lo, hi]` (the x convention, where increasing data maps to
/// increasing pixel). The three Geoshape/Tick/Rule call sites that pass
/// `ascending=true` depend on this ordering.
#[test]
fn dummy_unit_scale_ascending_produces_lo_hi_range() {
    let scale = dummy_unit_scale((10.0, 90.0), true);
    assert!(
        matches!(scale, ScaleKind::Linear(_)),
        "dummy_unit_scale must return a LinearScale variant"
    );
    let (r0, r1) = scale.pixel_range();
    assert_eq!(
        (r0, r1),
        (10.0, 90.0),
        "ascending=true must produce pixel_range (lo, hi) = (10.0, 90.0), got ({r0}, {r1})"
    );
}

/// `dummy_unit_scale` with `ascending=false` must produce a LinearScale whose
/// pixel range is `[hi, lo]` (the y convention, where the top pixel is smaller
/// and the bottom pixel is larger). The three Geoshape/Tick/Rule call sites
/// that pass `ascending=false` depend on this reversed ordering so that high
/// data values map toward the top of the viewport.
#[test]
fn dummy_unit_scale_descending_produces_hi_lo_range() {
    let scale = dummy_unit_scale((10.0, 90.0), false);
    assert!(
        matches!(scale, ScaleKind::Linear(_)),
        "dummy_unit_scale must return a LinearScale variant"
    );
    let (r0, r1) = scale.pixel_range();
    assert_eq!(
        (r0, r1),
        (90.0, 10.0),
        "ascending=false must produce pixel_range (hi, lo) = (90.0, 10.0), got ({r0}, {r1})"
    );
}

// ── SPEC-09 regression: categorical early-return in minor_tick_fractions ──────

/// Regression: guards SPEC-09's deletion of the categorical per-struct
/// `minor_ticks_internal` methods (OrdinalScale, CategoricalColorScale, etc.).
///
/// The safety of those deletions rests entirely on the early-return at the top
/// of `ScaleKind::minor_tick_fractions` (scale_resolve/mod.rs ~L199):
///
///   if matches!(self, Self::Ordinal(_)) { return Vec::new(); }
///
/// If that guard is ever removed while the per-struct methods remain absent,
/// the `dispatch_continuous!` macro would hit `ScaleKind::Ordinal(_) =>
/// unreachable!()` and panic at runtime. This test catches that regression by
/// exercising the real `minor_tick_fractions` entry point directly.
///
/// Two assertions:
/// 1. Ordinal → empty vec (the early-return fires before `dispatch_continuous!`).
/// 2. Linear → non-empty vec (the live `dispatch_continuous!` path still works).
#[test]
fn ordinal_scalekind_yields_empty_minor_fractions_via_early_return() {
    use crate::scale::linear::LinearScale;
    use crate::scale::ordinal::OrdinalScale;

    // --- 1. Ordinal: early-return must produce an empty vec, not a panic. ---
    let ordinal = ScaleKind::Ordinal(OrdinalScale::new_internal(
        vec!["a".into(), "b".into(), "c".into()],
        vec![0.0, 100.0],
        DiscreteLayout::band(0.1, 0.0, DEFAULT_ALIGN),
    ));
    let ordinal_minors = ordinal.minor_tick_fractions();
    assert!(
        ordinal_minors.is_empty(),
        "ScaleKind::Ordinal must yield empty minor_tick_fractions via the early-return \
         guard (SPEC-09 deletion invariant), got: {ordinal_minors:?}"
    );

    // --- 2. Linear: dispatch_continuous! path must still produce minors. ---
    // Domain [0, 10] over the provisional [0, 1] range — produces minor ticks
    // between the major tick positions.
    let linear = ScaleKind::Linear(LinearScale::new_internal(
        vec![0.0, 10.0],
        vec![0.0, 1.0],
        false,
        false,
    ));
    let linear_minors = linear.minor_tick_fractions();
    assert!(
        !linear_minors.is_empty(),
        "ScaleKind::Linear must yield non-empty minor_tick_fractions via \
         dispatch_continuous! (SPEC-09 live path), got empty"
    );
    assert!(
        linear_minors.iter().all(|f| f.is_finite()),
        "all minor tick fractions must be finite, got: {linear_minors:?}"
    );
}

// ── D4b: composite resolved-domain context threading ─────────────────────────
//
// These drive `resolve_scales_with_leaf_context` — the seam Task 5b's composite
// renderer calls per leaf — exactly as `scene_build.rs`/`prepare.rs` do, but with
// a non-`None` `LeafScaleContext`. They assert a composite-shared leaf resolves on
// the AUTO scale path (facet padding/`nice`), never the explicit-scale bypass,
// and that a `None` context is a byte-identical no-op.

/// x/y quantitative spec over `make_batch_q_q_n` (x ∈ [1,6], y ∈ [10,60]).
fn qq_spec() -> ChartSpec {
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    ChartSpec {
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn d4b_shared_numeric_seeds_auto_path_with_padding() {
    use crate::render::scale_resolve::{LeafScaleContext, SharedDomain};
    use std::collections::HashMap;

    let batch = make_batch_q_q_n(); // x ∈ [1, 6]
    let spec = qq_spec();
    let theme = ThemeInputs::default();
    let empty: HashMap<String, RecordBatch> = HashMap::new();
    let pr = (0.0, 100.0);

    // Baseline: no context → x domain is the leaf's own extent [1, 6].
    let (base, _) =
        resolve_scales_with_outputs(&spec, &batch, &empty, pr, pr, &theme).unwrap();
    let (b_lo, b_hi) = base.x.data_domain().unwrap();
    assert!((b_lo - 1.0).abs() < 1e-9 && (b_hi - 6.0).abs() < 1e-9, "baseline [{b_lo},{b_hi}]");

    // Shared context: x seeded to a wider extent than this leaf's data.
    let ctx = LeafScaleContext {
        x: Some(SharedDomain::Numeric { lo: 0.0, hi: 1000.0 }),
        y: None,
        ..Default::default()
    };
    let (shared, _) =
        resolve_scales_with_leaf_context(&spec, &spec.encoding, &batch, &empty, &crate::transform::run::OutputLineages::new(), pr, pr, &theme, Some(&ctx)).unwrap();

    // Domain is REPLACED by the shared extent, not the leaf's own [1, 6].
    let (lo, hi) = shared.x.data_domain().unwrap();
    assert!((lo - 0.0).abs() < 1e-9 && (hi - 1000.0).abs() < 1e-9,
        "shared numeric extent must seed the axis domain, got [{lo},{hi}]");

    // AUTO path discriminator: the pixel range is inset by DEFAULT_SCALE_PADDING_FRAC
    // (0.05 → (5, 95) for pr (0, 100)). The explicit-scale bypass forces padding 0
    // (full (0, 100)), so an inset range proves this resolved through the auto path
    // — exactly like a facet-shared panel — and NOT the set_domain bypass D4b rejects.
    let (rlo, rhi) = shared.x.pixel_range();
    assert!((rlo - 5.0).abs() < 1e-9 && (rhi - 95.0).abs() < 1e-9,
        "shared numeric must resolve on the auto (padded) path: pixel_range [{rlo},{rhi}] \
         (expected (5, 95); (0, 100) would mean the padding-0 bypass)");

    // y carried no shared domain → resolves standalone [10, 60] (untouched).
    let (ylo, yhi) = shared.y.data_domain().unwrap();
    assert!((ylo - 10.0).abs() < 1e-9 && (yhi - 60.0).abs() < 1e-9, "y untouched [{ylo},{yhi}]");
}

#[test]
fn d4b_shared_ordinal_seeds_category_vector() {
    use crate::render::scale_resolve::{LeafScaleContext, SharedDomain};
    use crate::spec::encoding::EncodingSpec;
    use std::collections::HashMap;

    // Ordinal x on `species` (distinct in-appearance: a, b, c).
    let batch = make_batch_q_q_n();
    let mut spec = qq_spec();
    spec.encoding.x = Some(EncodingSpec { field: "species".into(), type_: None, ..Default::default() });
    let theme = ThemeInputs::default();
    let empty: HashMap<String, RecordBatch> = HashMap::new();
    let pr = (0.0, 100.0);

    // Baseline: leaf's own category vector.
    let (base, _) =
        resolve_scales_with_outputs(&spec, &batch, &empty, pr, pr, &theme).unwrap();
    assert_eq!(x_ordinal_domain(&base), vec!["a", "b", "c"]);

    // Shared ordinal union carries categories this leaf's data does not have.
    let ctx = LeafScaleContext {
        x: Some(SharedDomain::Ordinal(vec![
            "z".into(), "a".into(), "b".into(), "c".into(), "d".into(),
        ])),
        y: None,
        ..Default::default()
    };
    let (shared, _) =
        resolve_scales_with_leaf_context(&spec, &spec.encoding, &batch, &empty, &crate::transform::run::OutputLineages::new(), pr, pr, &theme, Some(&ctx)).unwrap();
    // The union category vector seeds the axis — every leaf renders the full domain.
    assert_eq!(x_ordinal_domain(&shared), vec!["z", "a", "b", "c", "d"]);
}

#[test]
fn d4b_shared_ordinal_ignores_local_data_aware_sort() {
    use crate::render::scale_resolve::{LeafScaleContext, SharedDomain};
    use crate::spec::encoding::EncodingSpec;
    use std::collections::HashMap;

    // A data-aware sort ("-y") would reorder categories by each leaf's OWN
    // aggregate — different leaves would disagree about the "shared" order.
    // The seeded union vector (already order-preserving per D2) is
    // authoritative: local sort must be skipped.
    let batch = make_batch_q_q_n();
    let mut spec = qq_spec();
    spec.encoding.x = Some(EncodingSpec {
        field: "species".into(),
        type_: None,
        sort: Some(serde_json::json!("-y")),
        ..Default::default()
    });
    let theme = ThemeInputs::default();
    let empty: HashMap<String, RecordBatch> = HashMap::new();
    let pr = (0.0, 100.0);

    let ctx = LeafScaleContext {
        x: Some(SharedDomain::Ordinal(vec![
            "z".into(), "a".into(), "b".into(), "c".into(), "d".into(),
        ])),
        y: None,
        ..Default::default()
    };
    let (shared, _) =
        resolve_scales_with_leaf_context(&spec, &spec.encoding, &batch, &empty, &crate::transform::run::OutputLineages::new(), pr, pr, &theme, Some(&ctx)).unwrap();
    // Seeded order survives verbatim despite the "-y" sort on the encoding.
    assert_eq!(x_ordinal_domain(&shared), vec!["z", "a", "b", "c", "d"]);
}

#[test]
fn d4b_none_context_is_byte_identical_noop() {
    use std::collections::HashMap;

    let batch = make_batch_q_q_n();
    let spec = qq_spec();
    let theme = ThemeInputs::default();
    let empty: HashMap<String, RecordBatch> = HashMap::new();
    let pr = (0.0, 100.0);

    let (plain, _) =
        resolve_scales_with_outputs(&spec, &batch, &empty, pr, pr, &theme).unwrap();
    let (threaded_none, _) =
        resolve_scales_with_leaf_context(&spec, &spec.encoding, &batch, &empty, &crate::transform::run::OutputLineages::new(), pr, pr, &theme, None).unwrap();

    assert_eq!(plain.x.data_domain(), threaded_none.x.data_domain());
    assert_eq!(plain.y.data_domain(), threaded_none.y.data_domain());
    assert_eq!(plain.x.pixel_range(), threaded_none.x.pixel_range());
    assert_eq!(plain.y.pixel_range(), threaded_none.y.pixel_range());
}

#[test]
fn d4b_user_scale_wins_over_shared() {
    use crate::render::scale_resolve::{LeafScaleContext, SharedDomain};
    use crate::spec::encoding::{ContinuousScaleCommon, EncodingSpec, ScaleSpec};
    use std::collections::HashMap;

    // x carries a genuine user scale with an explicit domain [0, 500].
    let batch = make_batch_q_q_n();
    let mut spec = qq_spec();
    spec.encoding.x = Some(EncodingSpec {
        field: "x".into(),
        type_: None,
        scale: Some(ScaleSpec::Linear {
            common: ContinuousScaleCommon {
                domain: Some(vec![0.0, 500.0]),
                range: None,
                clamp: false,
                padding: None,
                scheme: None,
                domain_param: None,
                reverse: false,
            },
            nice: false,
            zero: false,
            system_injected: false,
        }),
        ..Default::default()
    });
    let theme = ThemeInputs::default();
    let empty: HashMap<String, RecordBatch> = HashMap::new();
    let pr = (0.0, 100.0);

    // Inject a DIFFERENT shared domain on the same channel.
    let ctx = LeafScaleContext {
        x: Some(SharedDomain::Numeric { lo: -100.0, hi: 100.0 }),
        y: None,
        ..Default::default()
    };
    let (scales, _) =
        resolve_scales_with_leaf_context(&spec, &spec.encoding, &batch, &empty, &crate::transform::run::OutputLineages::new(), pr, pr, &theme, Some(&ctx)).unwrap();

    // User scale wins: the explicit [0, 500] domain is used, NOT the shared [-100, 100].
    let (lo, hi) = scales.x.data_domain().unwrap();
    assert!((lo - 0.0).abs() < 1e-9 && (hi - 500.0).abs() < 1e-9,
        "user enc.scale must win over the composite shared domain, got [{lo},{hi}]");
    // And it short-circuited at the explicit-scale bypass (padding 0 → full range),
    // never reaching the context consultation on the auto path.
    let (rlo, rhi) = scales.x.pixel_range();
    assert!((rlo - 0.0).abs() < 1e-9 && (rhi - 100.0).abs() < 1e-9,
        "user explicit domain bypasses padding: pixel_range [{rlo},{rhi}] (expected (0, 100))");
}

/// 10-pre-b seam test: a `LeafScaleContext.color` categorical union threaded
/// through `resolve_scales_with_leaf_context` must SEED the built
/// `ColorScale`'s domain — proving the composite_domain parameter actually
/// reaches `build_color_scale` (a signature-order slip in the seam would pass
/// every unit test on the union logic alone).
#[test]
fn d10preb_shared_categorical_color_seeds_domain_through_seam() {
    use crate::render::scale_resolve::{LeafScaleContext, SharedDomain};
    use std::collections::HashMap;

    let spec = make_spec_with_color();
    let batch = make_batch_q_q_n(); // species ∈ {a, b, c}
    let theme = ThemeInputs::default();
    let empty: HashMap<String, RecordBatch> = HashMap::new();
    let pr = (0.0, 100.0);

    // Baseline: the leaf's own categories.
    let (base, _) =
        resolve_scales_with_outputs(&spec, &batch, &empty, pr, pr, &theme).unwrap();
    match base.color.expect("baseline color scale") {
        ColorScale::Categorical { domain, .. } => assert_eq!(domain, vec!["a", "b", "c"]),
        other => panic!("expected Categorical, got {other:?}"),
    }

    // Shared union carries categories this leaf's data lacks, in union order.
    let ctx = LeafScaleContext {
        color: Some(SharedDomain::Ordinal(vec![
            "z".into(), "a".into(), "b".into(), "c".into(), "d".into(),
        ])),
        ..Default::default()
    };
    let (shared, _) =
        resolve_scales_with_leaf_context(&spec, &spec.encoding, &batch, &empty, &crate::transform::run::OutputLineages::new(), pr, pr, &theme, Some(&ctx)).unwrap();
    match shared.color.expect("shared color scale") {
        ColorScale::Categorical { domain, .. } => assert_eq!(
            domain,
            vec!["z", "a", "b", "c", "d"],
            "shared categorical color union must seed the palette domain verbatim",
        ),
        other => panic!("expected Categorical, got {other:?}"),
    }
}

/// 10-pre-b seam test: a `LeafScaleContext.size` numeric extent threaded
/// through the full resolve must override the built `SizeScale`'s data
/// domain (pixel range untouched — size sharing is domain-only).
#[test]
fn d10preb_shared_size_extent_seeds_scale_through_seam() {
    use crate::render::scale_resolve::{LeafScaleContext, SharedDomain};
    use std::collections::HashMap;

    let spec = make_spec_with_size();
    let batch = make_batch_q_q_n_n_q();
    let theme = ThemeInputs::default();
    let empty: HashMap<String, RecordBatch> = HashMap::new();
    let pr = (0.0, 100.0);

    let (base, _) =
        resolve_scales_with_outputs(&spec, &batch, &empty, pr, pr, &theme).unwrap();
    let base_size = base.size.expect("baseline size scale");
    let (blo, bhi) = base_size.inner.data_domain().unwrap();

    let ctx = LeafScaleContext {
        size: Some(SharedDomain::Numeric { lo: -50.0, hi: 500.0 }),
        ..Default::default()
    };
    let (shared, _) =
        resolve_scales_with_leaf_context(&spec, &spec.encoding, &batch, &empty, &crate::transform::run::OutputLineages::new(), pr, pr, &theme, Some(&ctx)).unwrap();
    let shared_size = shared.size.expect("shared size scale");
    let (lo, hi) = shared_size.inner.data_domain().unwrap();
    assert!(
        (lo - -50.0).abs() < 1e-9 && (hi - 500.0).abs() < 1e-9,
        "shared size extent must override the leaf's own [{blo},{bhi}], got [{lo},{hi}]",
    );
    // Pixel range (the [min_px, max_px] band) stays theme-driven, not shared.
    assert_eq!(shared_size.min_px(), base_size.min_px());
    assert_eq!(shared_size.max_px(), base_size.max_px());
}

// --- Issue #40: positional Diverging scale must not truncate to [lo, mid] ---

/// A `Diverging` `ScaleSpec` on a positional (x) channel must resolve to a
/// Linear scale spanning the full `[lo, hi]` extent — not `[lo, mid]`, which
/// is what `DivergingScale::new`'s 3-element `[lo, mid, hi]` domain
/// collapses to if it is passed straight through as an explicit 2-endpoint
/// linear domain (issue #40).
#[test]
fn positional_diverging_scale_uses_full_lo_hi_extent_not_lo_mid() {
    use crate::spec::encoding::ScaleSpec;
    let mut s = make_spec_with_color();
    s.encoding.x.as_mut().unwrap().scale = Some(ScaleSpec::Diverging {
        scheme: None,
        domain: Some(vec![1.0, 2.5, 4.0]),
        domain_mid: None,
    });
    let b = make_batch_q_q_n();
    let theme = ThemeInputs::default();
    let (scales, _) = resolve_scales(&s, &b, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
    match scales.x {
        ScaleKind::Linear(l) => {
            let [lo, hi] = l.domain_pair();
            assert_eq!(lo, 1.0);
            assert_eq!(hi, 4.0, "positional Diverging axis must not truncate to [lo, mid]");
        }
        other => panic!("expected Linear, got {other:?}"),
    }
}

// --- P8: Arc single-axis (theta-only) exemption ---

/// An arc spec under `CoordPolar` with only the theta-mapped positional
/// encoding present (no radius channel) must resolve — not error with
/// `EncodingTypeMismatch` — mirroring the Tick|Rule single-axis exemption.
/// Pre-fix (RED): `Mark::Arc` was absent from the exemption block, so the
/// unconditional `spec.encoding.y.ok_or(EncodingTypeMismatch)` fired.
#[test]
fn arc_theta_only_resolves_via_dummy_radius_scale() {
    use crate::spec::coord::CoordKind;
    use crate::spec::encoding::{DataType, Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    use ferrum_scene::{PolarDirection, PolarThetaChannel};

    let spec = ChartSpec {
        mark: Mark::Arc,
        encoding: Encoding {
            x: Some(EncodingSpec {
                field: "x".into(),
                type_: Some(DataType::Quantitative),
                ..Default::default()
            }),
            y: None,
            ..Default::default()
        },
        coord: Some(CoordKind::Polar {
            theta: PolarThetaChannel::X,
            start_angle: 0.0,
            direction: PolarDirection::Clockwise,
            inner_radius: 0.0,
            outer_radius: None,
            pad_angle: 0.0,
        }),
        ..Default::default()
    };
    let batch = make_batch_q_q_n();
    let theme = ThemeInputs::default();

    let (scales, warnings) =
        resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme)
            .expect("theta-only arc spec must resolve, not error");
    // `make_batch_q_q_n`'s "x" column is `[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]`; the
    // default (no explicit `enc.scale`) auto-inferred quantitative domain is
    // the raw data extent with no padding/nice (`positional::build_axis_scale`).
    // Asserting the exact domain — not just `ScaleKind::Linear(_)` — is what
    // discriminates a real data-driven scale from `dummy_unit_scale`, which is
    // also `ScaleKind::Linear` with domain `[0.0, 1.0]`.
    match scales.x {
        ScaleKind::Linear(l) => {
            assert_eq!(l.domain_pair(), [1.0, 6.0], "theta axis (x) must be the real data-driven scale, not the dummy unit scale");
        }
        other => panic!("expected data-driven Linear x scale, got {other:?}"),
    }
    match scales.y {
        ScaleKind::Linear(l) => {
            assert_eq!(l.domain_pair(), [0.0, 1.0], "absent radius axis (y) must be the dummy unit scale");
        }
        other => panic!("expected dummy Linear y scale, got {other:?}"),
    }
    assert!(warnings.is_empty());
}

/// The theta channel is mandatory, unlike radius: an arc spec that binds only
/// the *non*-theta axis (radius, without the theta-mapped channel) must keep
/// erroring with `EncodingTypeMismatch`, not silently resolve via the
/// single-axis exemption. Guards against a gate that keys on "exactly one of
/// x/y is present" without checking *which* one — that would let
/// `mark_arc().encode(radius=...)` (theta unset) through to
/// `render/marks/arc.rs`, which returns an empty mark for a `None` theta
/// field, turning today's loud `EncodingTypeMismatch` into a silently blank
/// chart.
#[test]
fn arc_missing_theta_still_errors() {
    use crate::spec::coord::CoordKind;
    use crate::spec::encoding::{DataType, Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    use ferrum_scene::{PolarDirection, PolarThetaChannel};

    // theta="x", but only "y" (radius) is bound — the theta channel itself is missing.
    let spec = ChartSpec {
        mark: Mark::Arc,
        encoding: Encoding {
            x: None,
            y: Some(EncodingSpec {
                field: "y".into(),
                type_: Some(DataType::Quantitative),
                ..Default::default()
            }),
            ..Default::default()
        },
        coord: Some(CoordKind::Polar {
            theta: PolarThetaChannel::X,
            start_angle: 0.0,
            direction: PolarDirection::Clockwise,
            inner_radius: 0.0,
            outer_radius: None,
            pad_angle: 0.0,
        }),
        ..Default::default()
    };
    let batch = make_batch_q_q_n();
    let theme = ThemeInputs::default();

    let err = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme)
        .expect_err("arc with a missing theta channel must still error, not resolve via the single-axis exemption");
    match err {
        RenderError::EncodingTypeMismatch { channel, .. } => assert_eq!(channel, "x"),
        other => panic!("expected EncodingTypeMismatch{{channel: \"x\", ..}}, got {other:?}"),
    }
}

// --- Batch E (Task 6): Arc requires CoordPolar (MarkRequiresCoord) --------

use crate::spec::coord::CoordKind;

/// Shared builder for the four non-Polar-coord refusal tests below — the
/// only thing that varies between them is `coord`, so a full `ChartSpec`
/// literal per test would be near-verbatim duplication of the fixture used
/// throughout this file (operating principle 11).
fn arc_spec_with_coord(coord: Option<CoordKind>) -> ChartSpec {
    use crate::spec::encoding::{DataType, Encoding, EncodingSpec};
    use crate::spec::mark::Mark;

    ChartSpec {
        mark: Mark::Arc,
        encoding: Encoding {
            x: Some(EncodingSpec {
                field: "x".into(),
                type_: Some(DataType::Quantitative),
                ..Default::default()
            }),
            y: None,
            ..Default::default()
        },
        coord,
        ..Default::default()
    }
}

/// Asserts `mark_arc` under `coord` refuses with the typed
/// `MarkRequiresCoord` error, BEFORE the generic `EncodingTypeMismatch`
/// fallthrough can fire (pre-fix RED: `mark_arc` under any of these coords
/// surfaced "encoding 'y' expected EncodingSpec, got None" instead — proven
/// at `missing_y_encoding_names_y_when_not_flipped` above, which pins that
/// exact string for the ordinary (non-Arc) case this guard must not touch).
fn assert_arc_requires_polar(coord: Option<CoordKind>) {
    let spec = arc_spec_with_coord(coord);
    let batch = make_batch_q_q_n();
    let theme = ThemeInputs::default();
    let err = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme)
        .expect_err("mark_arc under a non-Polar coord must refuse");
    match &err {
        RenderError::MarkRequiresCoord { mark, required, .. } => {
            assert_eq!(*mark, "mark_arc");
            assert_eq!(*required, "Polar");
        }
        other => panic!("expected MarkRequiresCoord, got {other:?}"),
    }
    // The message names the STATE, and (since `#172`(1)) no longer prescribes
    // the fix. The prescription was `"; add .coord(fm.CoordPolar(...))"`, and it
    // was false on the path that mattered most: a merged-flat `LayerChart(bar,
    // arc.coord(fm.CoordPolar()))` reached this same refusal after DROPPING the
    // arc layer's coord, so the caller was told to add what they had written.
    // That drop is fixed — the coord now rides `Layer::coord` — and the
    // prescription is retired rather than left standing beside the fix, because
    // a refusal naming a state stays true on every path that raises it.
    assert_eq!(format!("{err}"), "mark_arc requires a Polar coord");
}

/// No coord at all (defaults to Cartesian downstream) must refuse.
#[test]
fn arc_no_coord_refuses_mark_requires_coord() {
    assert_arc_requires_polar(None);
}

/// Explicit Cartesian coord must refuse.
#[test]
fn arc_cartesian_coord_refuses_mark_requires_coord() {
    assert_arc_requires_polar(Some(CoordKind::Cartesian {
        x_domain: None,
        y_domain: None,
        expand: true,
        clip: true,
    }));
}

/// CoordFlip must refuse (flip is not polar).
#[test]
fn arc_flip_coord_refuses_mark_requires_coord() {
    assert_arc_requires_polar(Some(CoordKind::Flip));
}

/// The crosswise case — CoordGeo (the OTHER coord-requiring mark's own
/// requirement) must refuse for Arc exactly like any other non-Polar coord.
#[test]
fn arc_geo_coord_refuses_mark_requires_coord() {
    use crate::spec::coord::GeoProjection;
    assert_arc_requires_polar(Some(CoordKind::Geo { projection: GeoProjection::Mercator }));
}

/// Arc with BOTH `x` and `y` bound under a non-Polar coord. Every test above
/// uses an x-only encoding — exactly the shape that pre-fix tripped the
/// misleading `EncodingTypeMismatch` on the missing `y`, so those tests pin
/// only the error-ORDERING half of the finding. With `y` also bound, scale
/// resolution pre-fix had nothing missing to complain about and resolved
/// cleanly, so this branch used to reach `marks::arc::build`'s own non-Polar
/// arm and silently render nothing — the OTHER half of the finding (quality
/// review round 1, Required Fix 2). Must still refuse with
/// `MarkRequiresCoord`, proving the refusal is the coord check itself, not a
/// missing-channel fallthrough that happens to look the same.
#[test]
fn arc_both_x_and_y_bound_under_non_polar_coord_refuses() {
    use crate::spec::encoding::{DataType, EncodingSpec};

    let mut spec = arc_spec_with_coord(Some(CoordKind::Cartesian {
        x_domain: None,
        y_domain: None,
        expand: true,
        clip: true,
    }));
    spec.encoding.y = Some(EncodingSpec {
        field: "y".into(),
        type_: Some(DataType::Quantitative),
        ..Default::default()
    });
    let batch = make_batch_q_q_n();
    let theme = ThemeInputs::default();
    let err = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme).expect_err(
        "mark_arc with both x and y bound under a non-Polar coord must still refuse",
    );
    match &err {
        RenderError::MarkRequiresCoord { mark, required, .. } => {
            assert_eq!(*mark, "mark_arc");
            assert_eq!(*required, "Polar");
        }
        other => panic!("expected MarkRequiresCoord, got {other:?}"),
    }
}

// --- R3: user-facing channel names under CoordFlip ------------------------

/// A bare (non-Tick/Rule/Arc) mark missing its y encoding, unflipped, must
/// name the resolved channel `'y'` (byte-identical to today).
#[test]
fn missing_y_encoding_names_y_when_not_flipped() {
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;

    let spec = ChartSpec {
        mark: Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "x".into(), ..Default::default() }),
            y: None,
            ..Default::default()
        },
        ..Default::default()
    };
    let batch = make_batch_q_q_n();
    let theme = ThemeInputs::default();
    let err = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme)
        .expect_err("missing y encoding on a non-single-axis mark must error");
    assert_eq!(
        format!("{err}"),
        "encoding 'y' expected EncodingSpec, got None"
    );
}

/// R3: under `CoordFlip`, `spec.encoding` here is already the RESOLVED
/// (post-flip) layer encoding — mirroring what `prepare::build_layers`
/// produces (SPINE-12): a user who wrote ONLY `y=...` (never `x=...`) ends up,
/// post-swap, with `x` present and `y` absent — which is what actually trips
/// this branch. The rendered message must still say `'x'`, the channel the
/// user themselves left unset, not the internal post-flip `'y'` token.
#[test]
fn missing_encoding_names_users_channel_under_flip() {
    use crate::spec::coord::CoordKind;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;

    let spec = ChartSpec {
        mark: Mark::Point,
        encoding: Encoding {
            // Post-flip: this is what was the user's OWN y encoding.
            x: Some(EncodingSpec { field: "x".into(), ..Default::default() }),
            // Post-flip: the user never wrote an x encoding, so this (their
            // resolved y slot) stays unset.
            y: None,
            ..Default::default()
        },
        coord: Some(CoordKind::Flip),
        ..Default::default()
    };
    let batch = make_batch_q_q_n();
    let theme = ThemeInputs::default();
    let err = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme)
        .expect_err("missing (post-flip) y encoding must still error");
    match &err {
        RenderError::EncodingTypeMismatch { channel, coord_flipped, .. } => {
            // The resolved/internal token stays "y" — never un-flipped in the
            // struct field, only in the rendered Display text.
            assert_eq!(*channel, "y");
            assert!(*coord_flipped);
        }
        other => panic!("expected EncodingTypeMismatch, got {other:?}"),
    }
    assert_eq!(
        format!("{err}"),
        "encoding 'x' expected EncodingSpec, got None",
        "message must name the user's own unset channel ('x'), not the resolved post-flip token ('y')"
    );
}

/// Mirror of `arc_theta_only_resolves_via_dummy_radius_scale` for the other
/// theta direction: `theta="y"`, only the `y` positional encoding present,
/// `x` absent. Must resolve with a real y scale and a dummy unit x scale.
/// Pre-fix (RED) for the same reason as the x-direction test: the
/// unconditional `spec.encoding.x.ok_or(EncodingTypeMismatch)` fired.
#[test]
fn arc_theta_y_only_resolves_via_dummy_radius_scale() {
    use crate::spec::coord::CoordKind;
    use crate::spec::encoding::{DataType, Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    use ferrum_scene::{PolarDirection, PolarThetaChannel};

    let spec = ChartSpec {
        mark: Mark::Arc,
        encoding: Encoding {
            x: None,
            y: Some(EncodingSpec {
                field: "y".into(),
                type_: Some(DataType::Quantitative),
                ..Default::default()
            }),
            ..Default::default()
        },
        coord: Some(CoordKind::Polar {
            theta: PolarThetaChannel::Y,
            start_angle: 0.0,
            direction: PolarDirection::Clockwise,
            inner_radius: 0.0,
            outer_radius: None,
            pad_angle: 0.0,
        }),
        ..Default::default()
    };
    let batch = make_batch_q_q_n();
    let theme = ThemeInputs::default();

    let (scales, warnings) =
        resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme)
            .expect("theta-only arc spec (theta=y) must resolve, not error");
    // `make_batch_q_q_n`'s "y" column is `[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]`;
    // same rationale as the theta="x" test above — an exact domain assertion
    // discriminates the real data-driven scale from `dummy_unit_scale`.
    match scales.y {
        ScaleKind::Linear(l) => {
            assert_eq!(l.domain_pair(), [10.0, 60.0], "theta axis (y) must be the real data-driven scale, not the dummy unit scale");
        }
        other => panic!("expected data-driven Linear y scale, got {other:?}"),
    }
    match scales.x {
        ScaleKind::Linear(l) => {
            assert_eq!(l.domain_pair(), [0.0, 1.0], "absent radius axis (x) must be the dummy unit scale");
        }
        other => panic!("expected dummy Linear x scale, got {other:?}"),
    }
    assert!(warnings.is_empty());
}

mod positional_extent_classification {
    use crate::spec::encoding::{ContinuousScaleCommon, ScaleSpec};

    fn common_with_domain(domain: Option<Vec<f64>>) -> ContinuousScaleCommon {
        ContinuousScaleCommon {
            domain,
            range: None,
            clamp: false,
            padding: None,
            scheme: None,
            domain_param: None,
            reverse: false,
        }
    }

    #[test]
    fn diverging_three_elem_domain_extracts_lo_hi() {
        let spec = ScaleSpec::Diverging {
            scheme: None,
            domain: Some(vec![1.0, 2.5, 4.0]),
            domain_mid: None,
        };
        assert_eq!(spec.positional_extent(), Some(vec![1.0, 4.0]));
    }

    #[test]
    fn sequential_two_elem_domain_is_identity() {
        let spec = ScaleSpec::Sequential { scheme: None, domain: Some(vec![0.0, 10.0]), reverse: false, stops: None };
        assert_eq!(spec.positional_extent(), Some(vec![0.0, 10.0]));
    }

    #[test]
    fn quantize_domain_extracts_outer_bounds() {
        let spec = ScaleSpec::Quantize { domain: Some(vec![0.0, 5.0, 10.0]), range: None };
        assert_eq!(spec.positional_extent(), Some(vec![0.0, 10.0]));
    }

    #[test]
    fn quantile_domain_is_none_binning_artifact() {
        let spec = ScaleSpec::Quantile { domain: Some(vec![1.0, 2.0, 3.0, 4.0]), range: None };
        assert_eq!(spec.positional_extent(), None);
    }

    #[test]
    fn threshold_domain_is_none_binning_artifact() {
        let spec = ScaleSpec::Threshold { domain: Some(vec![10.0, 20.0]), range: None };
        assert_eq!(spec.positional_extent(), None);
    }

    #[test]
    fn bin_ordinal_is_always_none() {
        let spec = ScaleSpec::BinOrdinal { bins: Some(vec![0.0, 1.0, 2.0]), scheme: None, bin: None };
        assert_eq!(spec.positional_extent(), None);
    }

    #[test]
    fn diverging_domain_none_is_none() {
        let spec = ScaleSpec::Diverging { scheme: None, domain: None, domain_mid: None };
        assert_eq!(spec.positional_extent(), None);
    }

    #[test]
    fn diverging_empty_domain_is_none_no_panic() {
        let spec = ScaleSpec::Diverging { scheme: None, domain: Some(vec![]), domain_mid: None };
        assert_eq!(spec.positional_extent(), None);
    }

    #[test]
    fn sequential_single_elem_domain_is_none_no_panic() {
        let spec = ScaleSpec::Sequential { scheme: None, domain: Some(vec![5.0]), reverse: false, stops: None };
        assert_eq!(spec.positional_extent(), None);
    }

    #[test]
    fn linear_common_domain_passes_through() {
        let spec = ScaleSpec::Linear {
            common: common_with_domain(Some(vec![0.0, 100.0])),
            nice: false,
            zero: false,
            system_injected: false,
        };
        assert_eq!(spec.positional_extent(), Some(vec![0.0, 100.0]));
    }

    #[test]
    fn ordinal_has_no_continuous_extent() {
        let spec = ScaleSpec::Ordinal { domain: None, range: None, padding: 0.0 };
        assert_eq!(spec.positional_extent(), None);
    }
}

// ── YSlotPlan derivation (GH #72) ────────────────────────────────────────────

#[test]
fn yslot_plan_empty_when_no_layer_independent() {
    // The shared path: no layer flagged → the default (empty) plan, matching the
    // byte-stable pre-#52 behavior.
    let plan = YSlotPlan::from_layer_flags([false, false, false]);
    assert!(!plan.has_independent());
    assert!(plan.secondary_layers().is_empty());
    assert!(plan.layer_slot().is_empty());
    // Every layer maps through slot 0.
    assert_eq!(plan.slot_for_layer(0), 0);
    assert_eq!(plan.slot_for_layer(2), 0);
}

#[test]
fn yslot_plan_layer0_flag_is_ignored() {
    // Layer 0 is always the primary/left axis regardless of its flag, so a set
    // flag on layer 0 alone leaves the plan empty.
    let plan = YSlotPlan::from_layer_flags([true, false]);
    assert!(!plan.has_independent());
    assert_eq!(plan.slot_for_layer(0), 0);
}

#[test]
fn yslot_plan_assigns_slots_in_layer_order() {
    // Layers 1 and 3 independent → slots 1 and 2, in layer order; layer 2
    // (shared) stays on slot 0.
    let plan = YSlotPlan::from_layer_flags([false, true, false, true]);
    assert!(plan.has_independent());
    assert_eq!(plan.secondary_layers(), &[1, 3]);
    assert_eq!(plan.layer_slot(), &[0, 1, 0, 2]);
    assert_eq!(plan.slot_for_layer(0), 0);
    assert_eq!(plan.slot_for_layer(1), 1);
    assert_eq!(plan.slot_for_layer(2), 0);
    assert_eq!(plan.slot_for_layer(3), 2);
    // Out-of-range layer indices fall back to the primary slot.
    assert_eq!(plan.slot_for_layer(9), 0);
}

// ── Batch A §4.2: discretizing color scales (Quantize/Quantile/Threshold/BinOrdinal) ──

/// A rect chart whose `color` channel is the numeric column `v`, carrying the
/// given scale spec JSON. Used by the discretizing-color tests below.
fn spec_with_color_scale_json(scale_json: serde_json::Value) -> ChartSpec {
    use crate::spec::encoding::{DataType as SDT, Encoding, EncodingSpec, ScaleSpec};
    use crate::spec::mark::Mark;
    let scale: ScaleSpec = serde_json::from_value(scale_json).unwrap();
    ChartSpec {
        mark: Mark::Rect,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
            color: Some(EncodingSpec {
                field: "v".into(),
                type_: Some(SDT::Quantitative),
                scale: Some(scale),
                ..Default::default()
            }),
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Categorical x/y with a numeric `v` color column spanning `values`.
fn batch_with_numeric_color(values: Vec<f64>) -> RecordBatch {
    let n = values.len();
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ArrowDataType::Utf8, false),
        Field::new("y", ArrowDataType::Utf8, false),
        Field::new("v", ArrowDataType::Float64, true),
    ]));
    let labels: Vec<String> = (0..n).map(|i| format!("c{i}")).collect();
    let refs: Vec<&str> = labels.iter().map(String::as_str).collect();
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(refs.clone())),
            Arc::new(StringArray::from(refs)),
            Arc::new(Float64Array::from(values)),
        ],
    )
    .unwrap()
}

fn resolve_color(spec: &ChartSpec, batch: &RecordBatch) -> (ColorScale, Vec<crate::render::RenderWarning>) {
    let theme = ThemeInputs::default();
    let (scales, warnings) =
        resolve_scales(spec, batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap();
    (scales.color.expect("color scale must be resolved"), warnings)
}

/// #126: the refusal-path twin of [`resolve_color`], for specs expected to
/// fail color-scale resolution (e.g. an unparseable explicit range).
fn resolve_color_err(spec: &ChartSpec, batch: &RecordBatch) -> crate::render::RenderError {
    let theme = ThemeInputs::default();
    resolve_scales(spec, batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap_err()
}

fn buckets_of(scale: &ColorScale) -> &DiscretizedColors {
    match scale {
        ColorScale::Discretizing(b) => b,
        other => panic!("expected a Discretizing color scale, got {other:?}"),
    }
}

fn rgb(c: crate::render::color::Color) -> (u8, u8, u8) {
    (c.red, c.green, c.blue)
}

/// Quantize with an explicit `domain` divides *that* extent — not the data
/// extent — into `len(range)` uniform buckets, each painted its range color.
#[test]
fn quantize_explicit_domain_buckets_values_into_range_colors() {
    let spec = spec_with_color_scale_json(serde_json::json!({
        "type": "quantize",
        "domain": [0.0, 100.0],
        "range": ["#ff0000", "#00ff00", "#0000ff", "#ffffff"],
    }));
    // Data covers only 0..10 — the declared domain must win.
    let batch = batch_with_numeric_color(vec![0.0, 5.0, 10.0]);
    let (scale, warnings) = resolve_color(&spec, &batch);
    assert!(warnings.is_empty(), "clean spec must not warn: {warnings:?}");

    let buckets = buckets_of(&scale);
    assert_eq!(buckets.bounds(), &[0.0, 25.0, 50.0, 75.0, 100.0]);
    assert_eq!(buckets.colors().len(), 4);
    for (value, expected) in [
        (10.0, (0xff, 0x00, 0x00)),
        (30.0, (0x00, 0xff, 0x00)),
        (60.0, (0x00, 0x00, 0xff)),
        (90.0, (0xff, 0xff, 0xff)),
        // Open ends: outside the declared domain still lands in the end bucket.
        (-5.0, (0xff, 0x00, 0x00)),
        (500.0, (0xff, 0xff, 0xff)),
        // A boundary belongs to the bucket above it.
        (25.0, (0x00, 0xff, 0x00)),
    ] {
        assert_eq!(
            rgb(scale.lookup_f64(value).expect("finite value has a bucket")),
            expected,
            "value {value} landed in the wrong bucket"
        );
    }
}

/// Quantize with no declared `domain` buckets the *data* extent instead.
#[test]
fn quantize_without_explicit_domain_buckets_data_extent() {
    let spec = spec_with_color_scale_json(serde_json::json!({
        "type": "quantize",
        "range": ["#ff0000", "#0000ff"],
    }));
    let batch = batch_with_numeric_color(vec![0.0, 4.0, 10.0]);
    let (scale, _) = resolve_color(&spec, &batch);
    let buckets = buckets_of(&scale);
    assert_eq!(buckets.bounds(), &[0.0, 5.0, 10.0]);
    assert_eq!(rgb(scale.lookup_f64(4.0).unwrap()), (0xff, 0x00, 0x00));
    assert_eq!(rgb(scale.lookup_f64(6.0).unwrap()), (0x00, 0x00, 0xff));
}

/// Spec §4.2 (amended 2026-08-28): a descending Quantize `domain=[hi, lo]`
/// normalizes to ascending bounds with the swatch order reversed, so the first
/// `range` entry stays on the declared `lo` end — the reversed-colormap reading.
/// Deterministic and exactly the mirror of the ascending spelling: this pins
/// both directions of the same 3-color partition against each other.
#[test]
fn quantize_descending_domain_reverses_swatch_order() {
    let palette = ["#ff0000", "#00ff00", "#0000ff"];
    let batch = batch_with_numeric_color(vec![0.0, 50.0, 100.0]);
    let resolve = |domain: [f64; 2]| {
        let spec = spec_with_color_scale_json(serde_json::json!({
            "type": "quantize",
            "domain": domain,
            "range": palette,
        }));
        resolve_color(&spec, &batch).0
    };

    let ascending = resolve([0.0, 100.0]);
    let descending = resolve([100.0, 0.0]);

    // Identical bucket geometry either way — only the swatch order flips.
    let bounds: Vec<f64> = vec![0.0, 100.0 / 3.0, 200.0 / 3.0, 100.0];
    assert_eq!(buckets_of(&ascending).bounds(), bounds.as_slice());
    assert_eq!(buckets_of(&descending).bounds(), bounds.as_slice());
    assert_eq!(
        buckets_of(&descending).colors().iter().rev().map(|c| rgb(*c)).collect::<Vec<_>>(),
        buckets_of(&ascending).colors().iter().map(|c| rgb(*c)).collect::<Vec<_>>(),
        "descending swatches are the ascending swatches reversed"
    );

    // Per-value: the first range entry (#ff0000) sits on the declared lo end.
    for (value, asc, desc) in [
        (10.0, (0xff, 0x00, 0x00), (0x00, 0x00, 0xff)),
        (50.0, (0x00, 0xff, 0x00), (0x00, 0xff, 0x00)),
        (90.0, (0x00, 0x00, 0xff), (0xff, 0x00, 0x00)),
    ] {
        assert_eq!(rgb(ascending.lookup_f64(value).unwrap()), asc, "ascending at {value}");
        assert_eq!(rgb(descending.lookup_f64(value).unwrap()), desc, "descending at {value}");
    }
}

/// Threshold: `k` declared boundaries partition the line into `k + 1` buckets,
/// both ends open. The labeling bounds widen to bracket the data.
#[test]
fn threshold_k_boundaries_produce_k_plus_one_buckets() {
    let spec = spec_with_color_scale_json(serde_json::json!({
        "type": "threshold",
        "domain": [10.0, 20.0],
    }));
    let batch = batch_with_numeric_color(vec![0.0, 15.0, 30.0]);
    let (scale, _) = resolve_color(&spec, &batch);
    let buckets = buckets_of(&scale);
    assert_eq!(buckets.colors().len(), 3, "2 thresholds → 3 buckets");
    assert_eq!(buckets.bounds(), &[0.0, 10.0, 20.0, 30.0]);

    let low = scale.lookup_f64(5.0).unwrap();
    let mid = scale.lookup_f64(15.0).unwrap();
    let high = scale.lookup_f64(25.0).unwrap();
    assert_ne!(rgb(low), rgb(mid));
    assert_ne!(rgb(mid), rgb(high));
    // Boundaries belong to the bucket above; the end buckets are open.
    assert_eq!(rgb(scale.lookup_f64(10.0).unwrap()), rgb(mid));
    assert_eq!(rgb(scale.lookup_f64(20.0).unwrap()), rgb(high));
    assert_eq!(rgb(scale.lookup_f64(-999.0).unwrap()), rgb(low));
    assert_eq!(rgb(scale.lookup_f64(999.0).unwrap()), rgb(high));
}

/// Quantile buckets at the sample's quantile cut-points: with two buckets the
/// single cut is the sample median, so the split is by rank, not by extent.
#[test]
fn quantile_buckets_split_at_sample_quantiles() {
    let spec = spec_with_color_scale_json(serde_json::json!({
        "type": "quantile",
        "domain": [1.0, 2.0, 3.0, 100.0],
        "range": [0.0, 1.0],
    }));
    let batch = batch_with_numeric_color(vec![1.0, 2.0, 3.0, 100.0]);
    let (scale, _) = resolve_color(&spec, &batch);
    let buckets = buckets_of(&scale);
    assert_eq!(buckets.colors().len(), 2);
    // Median of [1, 2, 3, 100] is 2.5 — an extent split would have cut at 50.5.
    assert_eq!(buckets.bounds(), &[1.0, 2.5, 100.0]);
    let lower = scale.lookup_f64(2.0).unwrap();
    let upper = scale.lookup_f64(3.0).unwrap();
    assert_ne!(rgb(lower), rgb(upper));
}

/// With no declared sample, Quantile computes its cut-points from the encoded
/// column itself.
#[test]
fn quantile_without_domain_uses_the_data_column_as_its_sample() {
    let spec = spec_with_color_scale_json(serde_json::json!({
        "type": "quantile",
        "range": [0.0, 1.0],
    }));
    let batch = batch_with_numeric_color(vec![1.0, 2.0, 3.0, 100.0]);
    let (scale, _) = resolve_color(&spec, &batch);
    assert_eq!(buckets_of(&scale).bounds(), &[1.0, 2.5, 100.0]);
}

/// BinOrdinal: bin edges become bucket boundaries and its `scheme` names the
/// colormap the swatches are sampled from.
#[test]
fn bin_ordinal_bins_become_buckets_and_scheme_is_honored() {
    let batch = batch_with_numeric_color(vec![5.0, 15.0, 25.0, 35.0]);
    let blues = spec_with_color_scale_json(serde_json::json!({
        "type": "bin-ordinal",
        "bins": [10.0, 20.0, 30.0],
        "scheme": "blues",
    }));
    let reds = spec_with_color_scale_json(serde_json::json!({
        "type": "bin-ordinal",
        "bins": [10.0, 20.0, 30.0],
        "scheme": "reds",
    }));
    let (blues_scale, _) = resolve_color(&blues, &batch);
    let (reds_scale, _) = resolve_color(&reds, &batch);

    let blue_buckets = buckets_of(&blues_scale);
    assert_eq!(blue_buckets.colors().len(), 4, "3 edges → 4 bins");
    assert_eq!(blue_buckets.bounds(), &[5.0, 10.0, 20.0, 30.0, 35.0]);
    assert_ne!(
        rgb(blues_scale.lookup_f64(15.0).unwrap()),
        rgb(reds_scale.lookup_f64(15.0).unwrap()),
        "scheme= must change the swatch colors"
    );
}

/// A *categorical* scheme name on BinOrdinal contributes its palette entries
/// directly rather than being dropped in favor of the theme colormap.
#[test]
fn bin_ordinal_categorical_scheme_uses_palette_entries() {
    let spec = spec_with_color_scale_json(serde_json::json!({
        "type": "bin-ordinal",
        "bins": [10.0, 20.0],
        "scheme": "tableau10",
    }));
    let batch = batch_with_numeric_color(vec![5.0, 15.0, 25.0]);
    let (scale, warnings) = resolve_color(&spec, &batch);
    let expected = crate::render::color::palette::categorical_palette("tableau10");
    let colors = buckets_of(&scale).colors();
    assert_eq!(colors.len(), 3);
    for (i, c) in colors.iter().enumerate() {
        assert_eq!(rgb(*c), rgb(expected[i]), "bucket {i} must take palette entry {i}");
    }
    assert!(warnings.is_empty(), "3 buckets fit tableau10: {warnings:?}");
}

/// Spec §4.2 (amended 2026-08-28): when the bucket count exceeds a categorical
/// scheme's palette, the entries cycle **and** the recycling is reported —
/// mirroring `build_default_categorical_scale`'s `ColorPaletteOverflowed`
/// contract rather than degrading silently. The colors still render.
#[test]
fn categorical_scheme_overflow_warns_and_cycles_the_palette() {
    // `dark2` has 8 entries; 9 bin edges → 10 buckets forces two wraps.
    let bins: Vec<f64> = (1..=9).map(|i| i as f64).collect();
    let spec = spec_with_color_scale_json(serde_json::json!({
        "type": "bin-ordinal",
        "bins": bins,
        "scheme": "dark2",
    }));
    let batch = batch_with_numeric_color(vec![0.0, 5.0, 10.0]);
    let (scale, warnings) = resolve_color(&spec, &batch);

    let entries = crate::render::color::palette::categorical_palette("dark2");
    let colors = buckets_of(&scale).colors();
    assert_eq!(colors.len(), 10);
    assert!(colors.len() > entries.len(), "test must actually overflow the palette");
    assert_eq!(
        warnings,
        vec![crate::render::RenderWarning::ColorPaletteOverflowed { categories: 10 }],
        "palette recycling must be reported, not silent"
    );
    // Cycling, not truncation: every bucket still has a color, and bucket
    // `entries.len()` wraps back to entry 0.
    for (i, c) in colors.iter().enumerate() {
        assert_eq!(rgb(*c), rgb(entries[i % entries.len()]), "bucket {i}");
    }
    assert!(scale.lookup_f64(9.5).is_some(), "the top bucket still paints");
}

/// The categorical-scheme branch is keyed on the resolved scheme name, so it is
/// reachable from any discretizing variant through the top-level
/// `encoding.scheme` — not only from `BinOrdinal`'s own `scheme` field.
/// (`Threshold`'s `range` is numeric outputs, so its swatches always come from
/// a scheme.)
#[test]
fn categorical_scheme_branch_applies_to_threshold_via_encoding_scheme() {
    use crate::spec::encoding::EncodingSpec;
    let mut spec = spec_with_color_scale_json(serde_json::json!({
        "type": "threshold",
        "domain": [3.0, 7.0],
    }));
    let color = spec.encoding.color.take().expect("color encoding");
    spec.encoding.color = Some(EncodingSpec { scheme: Some("set1".into()), ..color });

    let batch = batch_with_numeric_color(vec![0.0, 5.0, 10.0]);
    let (scale, _) = resolve_color(&spec, &batch);
    let entries = crate::render::color::palette::categorical_palette("set1");
    let colors = buckets_of(&scale).colors();
    assert_eq!(colors.len(), 3);
    for (i, c) in colors.iter().enumerate() {
        assert_eq!(rgb(*c), rgb(entries[i]), "bucket {i} takes set1 entry {i}");
    }
}

/// #126 (re-aimed): an unparseable entry in a Quantize `range` refuses,
/// naming the offending entry, rather than discarding the range and falling
/// back to scheme-sampled swatches.
#[test]
fn quantize_range_parse_failure_refuses() {
    let spec = spec_with_color_scale_json(serde_json::json!({
        "type": "quantize",
        "domain": [0.0, 10.0],
        "range": ["#ff0000", "not-a-color"],
    }));
    let batch = batch_with_numeric_color(vec![0.0, 10.0]);
    let err = resolve_color_err(&spec, &batch);
    assert_eq!(
        err,
        crate::render::RenderError::ColorRangeParseFailure { entry: "not-a-color".into() },
    );
}

/// Null and NaN rows resolve to no bucket color, so the mark falls back exactly
/// as it does on the continuous path.
#[test]
fn discretizing_color_has_no_bucket_for_non_finite_values() {
    let spec = spec_with_color_scale_json(serde_json::json!({
        "type": "threshold",
        "domain": [10.0],
    }));
    let batch = batch_with_numeric_color(vec![0.0, f64::NAN, 20.0]);
    let (scale, _) = resolve_color(&spec, &batch);
    assert!(scale.lookup_f64(f64::NAN).is_none());
    assert!(scale.lookup_f64(f64::INFINITY).is_none());
    assert!(scale.lookup_f64(5.0).is_some());
    // Non-finite rows must not poison the labeling bounds.
    assert!(buckets_of(&scale).bounds().iter().all(|b| b.is_finite()));
}

/// Discretizing scales read the numeric column, so mark builders load
/// `col_as_f64` for them exactly as they do for continuous color.
#[test]
fn discretizing_color_reads_the_numeric_column() {
    let spec = spec_with_color_scale_json(serde_json::json!({
        "type": "threshold",
        "domain": [10.0],
    }));
    let batch = batch_with_numeric_color(vec![0.0, 20.0]);
    let (scale, _) = resolve_color(&spec, &batch);
    assert_eq!(scale.input(), ColorInput::Numeric);
    assert!(scale.categorical_domain().is_none());
    // String lookup routes through the numeric parse, matching Continuous.
    assert_eq!(
        rgb(scale.lookup("20").unwrap()),
        rgb(scale.lookup_f64(20.0).unwrap())
    );
}

/// Sequential and Diverging color specs are untouched by the discretizing
/// path: both still resolve to `Continuous` with their declared domain and
/// midpoint (the §7 byte-identity invariant for the absent case).
#[test]
fn sequential_and_diverging_specs_still_resolve_to_continuous() {
    let batch = batch_with_numeric_color(vec![-5.0, 0.0, 5.0]);

    let seq = spec_with_color_scale_json(serde_json::json!({
        "type": "sequential",
        "scheme": "blues",
        "domain": [0.0, 50.0],
    }));
    let (seq_scale, _) = resolve_color(&seq, &batch);
    match seq_scale {
        ColorScale::Continuous { domain, midpoint, .. } => {
            assert_eq!(domain, (0.0, 50.0));
            assert_eq!(midpoint, None);
        }
        other => panic!("expected Continuous, got {other:?}"),
    }

    let div = spec_with_color_scale_json(serde_json::json!({
        "type": "diverging",
        "scheme": "rdbu",
        "domain": [-10.0, 2.0, 10.0],
    }));
    let (div_scale, _) = resolve_color(&div, &batch);
    match div_scale {
        ColorScale::Continuous { domain, midpoint, .. } => {
            assert_eq!(domain, (-10.0, 10.0));
            assert_eq!(midpoint, Some(2.0));
        }
        other => panic!("expected Continuous, got {other:?}"),
    }
}

/// The heatmap `vmin=`/`vmax=` path: a Linear color scale's `common.domain`
/// pins the continuous extent instead of being dropped for the data extent.
#[test]
fn linear_common_domain_pins_the_continuous_color_extent() {
    let batch = batch_with_numeric_color(vec![0.0, 5.0, 10.0]);
    let pinned = spec_with_color_scale_json(serde_json::json!({
        "type": "linear",
        "domain": [0.0, 100.0],
    }));
    let (scale, _) = resolve_color(&pinned, &batch);
    match scale {
        ColorScale::Continuous { domain, .. } => assert_eq!(domain, (0.0, 100.0)),
        other => panic!("expected Continuous, got {other:?}"),
    }

    // No declared domain → the data extent, unchanged.
    let auto = spec_with_color_scale_json(serde_json::json!({ "type": "linear" }));
    let (auto_scale, _) = resolve_color(&auto, &batch);
    match auto_scale {
        ColorScale::Continuous { domain, .. } => assert_eq!(domain, (0.0, 10.0)),
        other => panic!("expected Continuous, got {other:?}"),
    }
}

/// A discretizing spec that declares no bucket count at all (hand-written JSON;
/// the Python constructors make `range`/`domain`/`bins` mandatory) resolves as
/// a plain continuous scale rather than an empty partition.
#[test]
fn discretizing_spec_without_a_bucket_count_falls_back_to_continuous() {
    let batch = batch_with_numeric_color(vec![0.0, 10.0]);
    for json in [
        serde_json::json!({ "type": "quantize", "domain": [0.0, 10.0] }),
        serde_json::json!({ "type": "quantile" }),
        serde_json::json!({ "type": "threshold" }),
        serde_json::json!({ "type": "bin-ordinal" }),
    ] {
        let spec = spec_with_color_scale_json(json.clone());
        let (scale, _) = resolve_color(&spec, &batch);
        assert!(
            matches!(scale, ColorScale::Continuous { .. }),
            "{json} should resolve to Continuous, got {scale:?}"
        );
    }
}

// ── Binned color: a `bin-ordinal` scale carrying `bin` and no `bins` ──

/// The edges the `Bin` transform computes over `values` with the wire
/// parameters `params` (`bin_count`, `bin_width`, `extent`, `nice`): each
/// bin's start, then the last bin's end. The oracle every binned color's
/// bounds are checked against.
fn bin_transform_edges(params: &serde_json::Value, values: &[f64]) -> Vec<f64> {
    let mut wire = params.clone();
    wire["field"] = serde_json::json!("v");
    let spec: crate::transform::bin::BinSpec = serde_json::from_value(wire).expect("a Bin wire");
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("v", ArrowDataType::Float64, true)])),
        vec![Arc::new(Float64Array::from(values.to_vec()))],
    )
    .unwrap();
    let out = crate::transform::bin::apply(&spec, &batch).expect("the Bin transform bins");
    let column = |name: &str| -> Vec<f64> {
        out.column_by_name(name).unwrap().as_any().downcast_ref::<Float64Array>().unwrap().values().to_vec()
    };
    let mut edges = column("bin_start");
    edges.push(*column("bin_end").last().expect("at least one bin"));
    edges
}

/// A binned color's scale: `bin-ordinal` carrying `bin` = `params` and no
/// `bins`.
fn binned_color_spec(params: serde_json::Value) -> ChartSpec {
    spec_with_color_scale_json(serde_json::json!({"type": "bin-ordinal", "bin": params}))
}

const BINNED_VALUES: [f64; 8] = [1.3, 2.7, 4.1, 5.5, 8.9, 9.2, 12.4, 3.3];

/// With no explicit `bins`, a binned color's buckets are the bins the `Bin`
/// transform cuts the same values into with the same parameters: Sturges
/// with nicing by default, `bin_count`, `bin_count` without nicing,
/// `bin_width`, and an explicit `extent`. One swatch per bin, and each value
/// paints its own bin's swatch. Rejects a second edge rule (uniform buckets
/// over the raw extent, as `Quantize` makes) and the continuous ramp a
/// `bin-ordinal` scale with no `bins` fell back to.
#[test]
fn a_binned_color_takes_the_bin_transforms_edges_for_the_same_parameters() {
    pyo3::Python::initialize();
    let batch = batch_with_numeric_color(BINNED_VALUES.to_vec());
    for params in [
        serde_json::json!({}),
        serde_json::json!({"bin_count": 3}),
        serde_json::json!({"bin_count": 4, "nice": false}),
        serde_json::json!({"bin_width": 2.5}),
        serde_json::json!({"bin_count": 5, "extent": [0.0, 20.0]}),
    ] {
        let expected = bin_transform_edges(&params, &BINNED_VALUES);
        let (scale, _) = resolve_color(&binned_color_spec(params.clone()), &batch);
        let buckets = buckets_of(&scale);
        assert_eq!(buckets.bounds(), expected.as_slice(), "{params}");
        assert_eq!(buckets.colors().len(), expected.len() - 1, "{params}: one swatch per bin");
        let swatches: Vec<_> = BINNED_VALUES.iter().map(|v| rgb(scale.lookup_f64(*v).unwrap())).collect();
        for (v, swatch) in BINNED_VALUES.iter().zip(&swatches) {
            let bin = expected.windows(2).position(|w| *v >= w[0] && *v < w[1]).unwrap_or(expected.len() - 2);
            assert_eq!(*swatch, rgb(buckets.colors()[bin]), "{params}: {v} paints bin {bin}");
        }
    }
}

/// Explicit `bins` win over `bin`: the buckets are the declared boundaries,
/// exactly as the same scale without `bin` resolves them. The binding's bin
/// rule still marks the scale as a binned color's; the scale without one is
/// not.
#[test]
fn explicit_bins_win_over_a_binned_colors_parameters() {
    pyo3::Python::initialize();
    let batch = batch_with_numeric_color(BINNED_VALUES.to_vec());
    let both = spec_with_color_scale_json(serde_json::json!({
        "type": "bin-ordinal", "bins": [3.0, 6.0], "bin": {"bin_count": 10},
    }));
    let declared = spec_with_color_scale_json(serde_json::json!({"type": "bin-ordinal", "bins": [3.0, 6.0]}));
    let (both, _) = resolve_color(&both, &batch);
    let (declared, _) = resolve_color(&declared, &batch);
    assert_eq!(buckets_of(&both).bounds(), &[1.3, 3.0, 6.0, 12.4]);
    assert_eq!(buckets_of(&both).bounds(), buckets_of(&declared).bounds());
    assert_eq!(buckets_of(&both).colors(), buckets_of(&declared).colors());
    assert!(buckets_of(&both).is_cut_by_bin());
    assert!(!buckets_of(&declared).is_cut_by_bin());
}

/// One distinct value is one bin: the `Bin` transform's unit bin around it.
#[test]
fn a_binned_color_over_one_distinct_value_has_one_bin() {
    pyo3::Python::initialize();
    let (scale, _) = resolve_color(&binned_color_spec(serde_json::json!({})), &batch_with_numeric_color(vec![4.0, 4.0, 4.0]));
    let buckets = buckets_of(&scale);
    assert_eq!(buckets.bounds(), &[3.5, 4.5]);
    assert_eq!(buckets.colors().len(), 1);
    assert_eq!(bin_transform_edges(&serde_json::json!({}), &[4.0, 4.0, 4.0]), vec![3.5, 4.5]);
}

/// A field with no finite value resolves exactly as the same empty field
/// does under a plain continuous color: no bin exists to step through.
#[test]
fn a_binned_color_with_no_finite_value_resolves_as_an_empty_continuous_color() {
    pyo3::Python::initialize();
    let batch = batch_with_numeric_color(vec![f64::NAN, f64::NAN]);
    let (binned, _) = resolve_color(&binned_color_spec(serde_json::json!({"bin_count": 4})), &batch);
    let (continuous, _) = resolve_color(&spec_with_color_scale_json(serde_json::json!({"type": "linear"})), &batch);
    assert!(matches!(binned, ColorScale::Continuous { .. }), "{binned:?}");
    assert_eq!(format!("{binned:?}"), format!("{continuous:?}"));
}

/// `scheme` names the colormap a binned color's swatches are sampled from.
#[test]
fn a_binned_colors_scheme_is_honored() {
    pyo3::Python::initialize();
    let batch = batch_with_numeric_color(BINNED_VALUES.to_vec());
    let scheme = |name: &str| {
        spec_with_color_scale_json(serde_json::json!({"type": "bin-ordinal", "scheme": name, "bin": {"bin_count": 3}}))
    };
    let (blues, _) = resolve_color(&scheme("blues"), &batch);
    let (reds, _) = resolve_color(&scheme("reds"), &batch);
    assert_eq!(buckets_of(&blues).bounds(), buckets_of(&reds).bounds());
    let blue = crate::render::color::ContinuousScheme::Named(crate::render::color::NamedContinuous::from_name("blues").unwrap());
    assert_eq!(rgb(buckets_of(&blues).colors()[0]), rgb(blue.sample(0.0)));
    assert_ne!(rgb(blues.lookup_f64(1.3).unwrap()), rgb(reds.lookup_f64(1.3).unwrap()));
}

/// A layered binned color's edges are cut from every drawing layer's values
/// (layer 0's `mean_z` over 0..2, layer 1's over 10..20), so the layers share
/// one set of bins. Rejects edges cut from the establishing layer alone.
#[test]
fn a_layered_binned_colors_edges_come_from_every_drawing_layers_values() {
    pyo3::Python::initialize();
    let own = serde_json::json!({"field": "mean_z", "scale": {"type": "bin-ordinal", "bin": {"bin_count": 4, "nice": false}}});
    let (spec, chart_encoding) = aux_layered_spec("color", [Some(own.clone()), Some(own)]);
    let color = resolve_aux_layered(&spec, &chart_encoding).color.expect("a color scale");
    assert_eq!(buckets_of(&color).bounds(), bin_transform_edges(&serde_json::json!({"bin_count": 4, "nice": false}), &[0.0, 2.0, 10.0, 20.0]).as_slice());
    assert_eq!(buckets_of(&color).bounds(), &[0.0, 5.0, 10.0, 15.0, 20.0]);
}

/// A binned color bins numbers whatever type a binding declares, so a later
/// layer binding the field `ordinal` (Vega-Lite's binned ordinal legend)
/// adds its values to the edges, with no `LayerScaleExcluded`. Rejects the
/// declared-type exclusion that cut the edges from layer 0 alone.
#[test]
fn a_binned_colors_ordinal_declared_layer_adds_to_the_edges() {
    pyo3::Python::initialize();
    let own = serde_json::json!({"field": "mean_z", "type": "ordinal", "scale": {"type": "bin-ordinal", "bin": {"bin_count": 4, "nice": false}}});
    let (spec, chart_encoding) = aux_layered_spec("color", [Some(own.clone()), Some(own)]);
    let (scales, warnings) = resolve_aux_layered_warned(&spec, &chart_encoding);
    assert_eq!(buckets_of(&scales.color.expect("a color scale")).bounds(), &[0.0, 5.0, 10.0, 15.0, 20.0]);
    assert_eq!(layer_scale_exclusions(warnings), vec![], "no layer excluded");
}

/// A layered quantile color cuts its quantiles from the values every drawing
/// layer draws, the sample a binned color's edges come from. Rejects
/// cut-points taken from the establishing layer's rows alone.
#[test]
fn a_layered_quantile_colors_cut_points_come_from_every_drawing_layers_values() {
    pyo3::Python::initialize();
    let own = serde_json::json!({"field": "mean_z", "scale": {"type": "quantile", "range": [0.0, 1.0]}});
    let (spec, chart_encoding) = aux_layered_spec("color", [Some(own.clone()), Some(own)]);
    let color = resolve_aux_layered(&spec, &chart_encoding).color.expect("a color scale");
    assert_eq!(buckets_of(&color).bounds(), &[0.0, 6.0, 20.0]);
}

/// A text layer's inherited color only groups it, so its rows never widen
/// a binned color's edges; a point layer inheriting the same field does.
/// Layer 1's batch carries the chart's `z` over 10..20.
#[test]
fn a_texts_inherited_color_does_not_widen_a_binned_colors_edges() {
    pyo3::Python::initialize();
    let own = serde_json::json!({"field": "mean_z", "scale": {"type": "bin-ordinal", "bin": {"bin_count": 4, "nice": false}}});
    let (primary, mut outputs) = aux_layered_batches();
    let agg1 = &outputs["agg1"];
    let renamed = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("g", ArrowDataType::Utf8, false),
            Field::new("m", ArrowDataType::Float64, false),
            Field::new("z", ArrowDataType::Float64, false),
            Field::new("c", ArrowDataType::Utf8, false),
        ])),
        agg1.columns().to_vec(),
    )
    .unwrap();
    outputs.insert("agg1".to_string(), renamed);
    for (mark, bounds) in [
        (crate::spec::mark::Mark::Text, vec![0.0, 0.5, 1.0, 1.5, 2.0]),
        (crate::spec::mark::Mark::Point, vec![0.0, 5.0, 10.0, 15.0, 20.0]),
    ] {
        let (mut spec, chart_encoding) = aux_layered_spec("color", [Some(own.clone()), None]);
        spec.layers.as_mut().unwrap()[1].mark = mark;
        let scales = resolve_scales_with_leaf_context(
            &spec,
            &chart_encoding,
            &primary,
            &outputs, &crate::transform::run::OutputLineages::new(),
            (0.0, 100.0),
            (0.0, 80.0),
            &ThemeInputs::default(),
            None,
        )
        .expect("resolves")
        .0;
        let color = scales.color.expect("a color scale");
        assert_eq!(buckets_of(&color).bounds(), bounds.as_slice(), "{mark:?}");
    }
}

// ── F-L04-02 second revision: Sequential `stops` → ContinuousScheme::Gradient ──

/// A `Sequential` spec carrying `stops` resolves to a `Gradient`-backed
/// `Continuous` scale, and `lookup_f64` interpolates at the *declared*
/// (non-uniform) `t` positions rather than re-spacing them to `i / (n - 1)`
/// (spec reviewer cycle 3, finding 1 — rust-quality-reviewer cycle-1 gap:
/// this path had zero Rust unit coverage, only Python SVG-difference tests).
#[test]
fn sequential_stops_resolve_to_gradient_and_interpolate_at_declared_t() {
    let spec = spec_with_color_scale_json(serde_json::json!({
        "type": "sequential",
        "stops": [[0.0, "#ff0000"], [0.9, "#00ff00"], [1.0, "#0000ff"]],
    }));
    // Data extent (0.0, 10.0) so a data value of 9.0 normalizes to t = 0.9 —
    // exactly the declared (non-uniform) middle stop's position.
    let batch = batch_with_numeric_color(vec![0.0, 10.0]);
    let (scale, warnings) = resolve_color(&spec, &batch);
    assert!(warnings.is_empty(), "well-formed stops must not warn: {warnings:?}");
    match &scale {
        ColorScale::Continuous {
            scheme: crate::render::color::ContinuousScheme::Gradient(stops),
            ..
        } => {
            assert_eq!(stops.len(), 3);
            assert_eq!(stops[1].0, 0.9, "declared t must survive, not re-space to i/(n-1)");
        }
        other => panic!("expected Continuous with a Gradient scheme, got {other:?}"),
    }
    // t = 0.9 lands exactly on the middle stop's own position, so
    // `sample_gradient` returns that stop's color exactly (u = 0 at the
    // bracketing pair's lower end) — pins that lookup_f64 actually uses the
    // declared t, not an evenly-spaced re-derivation (which would place the
    // 2nd of 3 stops at t = 0.5, not 0.9, and return a red/green blend here
    // instead of pure green).
    assert_eq!(rgb(scale.lookup_f64(9.0).unwrap()), (0x00, 0xff, 0x00));
}

/// The two raw-dict-reachable edges finding 1 (rust-quality-reviewer cycle 1)
/// found untested: an unparseable stop refuses (#126, re-aimed from the
/// former warn-and-fall-through) with the same `ColorRangeParseFailure`
/// convention as every other explicit-color-list path, all-or-nothing; and a
/// `stops` list with fewer than 2 usable entries falls through to plain
/// scheme-name resolution WITH a named `GradientTooFewStops` warning
/// (wave-1 close remediation: this used to be the last silent arm in
/// `resolve_continuous_scheme`, reachable only from a hand-written dict
/// spec, e.g. `Color(scale={"type": "sequential", "stops": [[0.0,
/// "red"]]})`, since `Gradient(...)`'s Python constructor guards the
/// primary route).
#[test]
fn sequential_stops_raw_dict_edges() {
    let batch = batch_with_numeric_color(vec![0.0, 10.0]);

    // Unparseable stop: all-or-nothing, refuses naming the bad entry — not a
    // partial or silently-wrong gradient, and not a silent swap for the
    // theme's default scheme either.
    let bad_stop = spec_with_color_scale_json(serde_json::json!({
        "type": "sequential",
        "stops": [[0.0, "notacolor"], [1.0, "blue"]],
    }));
    let err = resolve_color_err(&bad_stop, &batch);
    assert_eq!(
        err,
        crate::render::RenderError::ColorRangeParseFailure { entry: "notacolor".into() },
    );

    // Fewer than 2 usable stops: named `GradientTooFewStops` warning, then
    // falls through to scheme resolution (wave-1 close remediation — this
    // is a malformed-partition shape, not a parse failure, so #126's
    // ColorRangeParseFailure convention does not apply, but it must not be
    // silent either).
    let one_stop = spec_with_color_scale_json(serde_json::json!({
        "type": "sequential",
        "stops": [[0.0, "red"]],
    }));
    let (scale, warnings) = resolve_color(&one_stop, &batch);
    assert!(
        warnings.iter().any(|w| matches!(
            w,
            crate::render::RenderWarning::GradientTooFewStops { count: 1 }
        )),
        "a < 2-stop fall-through must warn by name: {warnings:?}"
    );
    assert!(
        !matches!(
            scale,
            ColorScale::Continuous { scheme: crate::render::color::ContinuousScheme::Gradient(_), .. }
        ),
        "a single stop cannot describe a gradient; must fall through to scheme resolution"
    );
}

/// #240-4b (routed rider): an unrecognized `scheme`/`cmap` name used to fall
/// back to the theme's default scheme with NO warning at all — indistinguishable
/// from the user never having asked for a scheme. It must now emit a named
/// `RenderWarning::UnknownColorScheme` while still rendering with the theme
/// default (warn-degraded, not a refusal: the render is meaningful, just not
/// painted with the asked-for palette).
#[test]
fn continuous_unknown_scheme_name_warns_and_falls_back_to_theme_default() {
    use crate::render::color::{ContinuousScheme, NamedContinuous};

    let spec = spec_with_color_scale_json(serde_json::json!({
        "type": "linear",
        "scheme": "not-a-real-scheme",
    }));
    let batch = batch_with_numeric_color(vec![0.0, 10.0]);
    let (scale, warnings) = resolve_color(&spec, &batch);

    assert!(
        warnings.iter().any(|w| matches!(
            w,
            crate::render::RenderWarning::UnknownColorScheme { name } if name == "not-a-real-scheme"
        )),
        "expected an UnknownColorScheme warning naming the bad name, got {warnings:?}"
    );
    match scale {
        ColorScale::Continuous { scheme: ContinuousScheme::Named(named), .. } => {
            assert_eq!(
                named,
                NamedContinuous::from_name("cool_blue").unwrap(),
                "must fall back to the theme's default sequential scheme"
            );
        }
        other => panic!("expected Continuous with a Named scheme, got {other:?}"),
    }

    // Control: no scheme requested at all → no warning, same theme default.
    let no_scheme_spec = spec_with_color_scale_json(serde_json::json!({ "type": "linear" }));
    let (no_scheme_scale, no_scheme_warnings) = resolve_color(&no_scheme_spec, &batch);
    assert!(
        no_scheme_warnings.is_empty(),
        "requesting nothing must not warn: {no_scheme_warnings:?}"
    );
    match no_scheme_scale {
        ColorScale::Continuous { scheme: ContinuousScheme::Named(named), .. } => {
            assert_eq!(
                named,
                NamedContinuous::from_name("cool_blue").unwrap(),
                "the no-scheme control must resolve the identical theme-default scheme"
            );
        }
        other => panic!("expected Continuous with a Named scheme, got {other:?}"),
    }
}

/// An alpha-carrying `#rrggbbaa` stop survives resolution — the Gradient
/// pyfunction's docstring advertises 8-digit hex stops (rust-quality-reviewer
/// cycle-1 finding: nothing pinned this, so a future switch to a 6-digit
/// formatter in `to_sequential_wire_form` could silently drop stop alpha with
/// the suite still green).
#[test]
fn sequential_stops_alpha_channel_round_trips_through_resolution() {
    let spec = spec_with_color_scale_json(serde_json::json!({
        "type": "sequential",
        "stops": [[0.0, "#ff000080"], [1.0, "#0000ffff"]],
    }));
    let batch = batch_with_numeric_color(vec![0.0, 1.0]);
    let (scale, warnings) = resolve_color(&spec, &batch);
    assert!(warnings.is_empty(), "well-formed alpha stops must not warn: {warnings:?}");
    let at_zero = scale.lookup_f64(0.0).unwrap();
    assert_eq!((at_zero.red, at_zero.green, at_zero.blue), (0xff, 0x00, 0x00));
    assert_eq!(at_zero.alpha, 0x80, "alpha must survive stop resolution, not collapse to opaque");
}

/// `DiscretizedColors` refuses any (bounds, colors) pair that is not a
/// partition — the invariant `lookup` relies on lives in the constructor.
#[test]
fn discretized_colors_requires_one_more_bound_than_colors() {
    let c = crate::render::color::from_rgba(0, 0, 0, 255);
    assert!(DiscretizedColors::new(vec![0.0, 1.0], vec![c]).is_some());
    assert!(DiscretizedColors::new(vec![0.0, 1.0], vec![c, c]).is_none());
    assert!(DiscretizedColors::new(vec![0.0, 1.0, 2.0], vec![c]).is_none());
    assert!(DiscretizedColors::new(vec![0.0], Vec::new()).is_none());
}

/// `set_colors` (the `configure_color(range=…)` seam) only applies a
/// replacement that matches the bucket count, and *reports* the refusal with
/// both counts rather than no-op-ing silently (spec §4.2, amended 2026-08-28).
#[test]
fn discretized_colors_set_colors_requires_matching_bucket_count() {
    let black = crate::render::color::from_rgba(0, 0, 0, 255);
    let white = crate::render::color::from_rgba(255, 255, 255, 255);
    let mut buckets = DiscretizedColors::new(vec![0.0, 1.0, 2.0], vec![black, black]).unwrap();

    assert_eq!(
        buckets.set_colors(vec![white]),
        Err(SwatchCountMismatch { expected: 2, received: 1 }),
        "a wrong-length range is refused, naming both counts"
    );
    assert_eq!(rgb(buckets.colors()[0]), rgb(black), "the swatches are untouched");

    assert_eq!(buckets.set_colors(vec![white, white]), Ok(()));
    assert_eq!(rgb(buckets.colors()[0]), rgb(white), "matching-length range is applied");
}

/// A `Quantile` color spec with no declared sample derives its cut-points from
/// the data, so under facet sharing it must read the GLOBAL batch — otherwise
/// each panel cuts at its own quantiles while the colorbar stays global, and the
/// same value paints differently per panel with no diagnostic.
#[test]
fn t3_shared_facet_quantile_sample_uses_the_global_batch() {
    use crate::spec::encoding::ScaleSpec;

    // Panel A's values are all in the global lower half; panel B's in the upper
    // half. Per-panel quantiles would split each panel at its own median (2.5
    // and 6.5); the global median of 1..8 is 4.5.
    let panel_a = t3_c_batch(vec![0.0; 4], vec![0.0; 4], vec![1.0, 2.0, 3.0, 4.0]);
    let panel_b = t3_c_batch(vec![0.0; 4], vec![0.0; 4], vec![5.0, 6.0, 7.0, 8.0]);
    let global = t3_c_batch(
        vec![0.0; 8],
        vec![0.0; 8],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    );
    let outputs = final_outputs(global);

    let mut spec = faceted_continuous_color_spec();
    spec.encoding.color.as_mut().unwrap().scale = Some(
        serde_json::from_value::<ScaleSpec>(serde_json::json!({
            "type": "quantile",
            "range": [0.0, 1.0],
        }))
        .unwrap(),
    );

    let theme = ThemeInputs::default();
    let pr = (0.0, 100.0);
    let (scales_a, _) =
        resolve_scales_with_outputs(&spec, &panel_a, &outputs, pr, pr, &theme).unwrap();
    let (scales_b, _) =
        resolve_scales_with_outputs(&spec, &panel_b, &outputs, pr, pr, &theme).unwrap();
    let (color_a, color_b) = (
        scales_a.color.expect("panel A color"),
        scales_b.color.expect("panel B color"),
    );

    // Both panels cut at the GLOBAL median (4.5), not their own (2.5 / 6.5).
    assert_eq!(buckets_of(&color_a).bounds(), &[1.0, 4.5, 8.0]);
    assert_eq!(buckets_of(&color_b).bounds(), &[1.0, 4.5, 8.0]);

    // The discriminating consequence: every value paints the same in both
    // panels. Under per-panel sampling, 4.0 would be panel A's TOP bucket and
    // panel B's bottom one.
    for v in [1.0, 4.0, 5.0, 8.0] {
        assert_eq!(
            rgb(color_a.lookup_f64(v).unwrap()),
            rgb(color_b.lookup_f64(v).unwrap()),
            "value {v} must paint identically in both panels"
        );
    }
    assert_ne!(
        rgb(color_a.lookup_f64(4.0).unwrap()),
        rgb(color_a.lookup_f64(5.0).unwrap()),
        "the global median must still separate the two buckets"
    );
}

/// The render-side discretizing resolver derives its interior bounds from the
/// shared `uniform_bin_thresholds` helper rather than re-deriving the formula,
/// including on spans whose step is not exactly representable.
///
/// This is one half of the drift guard: the other half — that
/// `QuantizeScale::thresholds()` still routes through the same helper — is
/// `scale::quantize::tests::quantize_thresholds`. Together they pin the single
/// definition; neither asserts it alone.
#[test]
fn quantize_bucket_bounds_come_from_the_shared_bin_threshold_helper() {
    for (lo, hi, n) in [(0.0, 100.0, 4), (0.0, 10.0, 3), (-1.0, 1.0, 7), (2.5, 3.5, 6)] {
        let range: Vec<String> = (0..n).map(|i| format!("#00000{i}")).collect();
        let spec = spec_with_color_scale_json(serde_json::json!({
            "type": "quantize",
            "domain": [lo, hi],
            "range": range,
        }));
        let batch = batch_with_numeric_color(vec![lo, hi]);
        let (scale, _) = resolve_color(&spec, &batch);

        let interior = &buckets_of(&scale).bounds()[1..n];
        let expected = crate::scale::core::uniform_bin_thresholds(lo, hi, n);
        assert_eq!(interior, expected.as_slice(), "domain [{lo},{hi}] into {n} bins");
    }
}

// ── Batch A §4.2 (amended 2026-08-28): declared boundary ordering ─────────────
//
// `Color(scale={...})` accepts a raw dict, bypassing the `ThresholdScale` /
// `BinOrdinalScale` constructors' "strictly sorted ascending" validation, so a
// non-ascending list reaches the resolver. Fully descending normalizes like
// Quantize; non-monotonic is a typed refusal.

/// Try to resolve the color scale, surfacing the resolver's error instead of
/// unwrapping it.
fn try_resolve_color(
    spec: &ChartSpec,
    batch: &RecordBatch,
) -> Result<Option<ColorScale>, crate::render::RenderError> {
    let theme = ThemeInputs::default();
    resolve_scales(spec, batch, (0.0, 100.0), (0.0, 80.0), &theme).map(|(s, _)| s.color)
}

/// A fully descending `Threshold` domain is the mirror of its ascending
/// spelling: identical (monotonic) bounds, identical bucket membership, and the
/// swatches reversed — exactly `Quantize`'s descending contract, so the same
/// user intent reads the same way on every discretizing variant.
#[test]
fn threshold_descending_domain_mirrors_the_ascending_spelling() {
    let batch = batch_with_numeric_color(vec![5.0, 15.0, 25.0]);
    let resolve = |domain: Vec<f64>| {
        let spec = spec_with_color_scale_json(serde_json::json!({
            "type": "threshold",
            "domain": domain,
        }));
        resolve_color(&spec, &batch).0
    };
    let ascending = resolve(vec![10.0, 20.0]);
    let descending = resolve(vec![20.0, 10.0]);

    // Bounds are monotonic either way — the bug this pins produced
    // `[5, 20, 10, 25]`, which `partition_point` cannot partition.
    assert_eq!(buckets_of(&ascending).bounds(), &[5.0, 10.0, 20.0, 25.0]);
    assert_eq!(buckets_of(&descending).bounds(), &[5.0, 10.0, 20.0, 25.0]);
    assert!(
        buckets_of(&descending).bounds().windows(2).all(|w| w[0] < w[1]),
        "bounds must be strictly ascending after normalization"
    );

    // Swatches reversed, mirroring `quantize_descending_domain_reverses_swatch_order`.
    assert_eq!(
        buckets_of(&descending).colors().iter().rev().map(|c| rgb(*c)).collect::<Vec<_>>(),
        buckets_of(&ascending).colors().iter().map(|c| rgb(*c)).collect::<Vec<_>>(),
    );

    // Middle-bucket membership is the symptom the reviewer reproduced: 12 and 18
    // belong to the MIDDLE bucket, and must not paint the top bucket's color.
    let mid = rgb(ascending.lookup_f64(15.0).unwrap());
    let top = rgb(ascending.lookup_f64(25.0).unwrap());
    let bottom = rgb(ascending.lookup_f64(5.0).unwrap());
    assert!(mid != top && mid != bottom, "the three buckets must be distinguishable");
    for v in [12.0, 18.0] {
        assert_eq!(rgb(ascending.lookup_f64(v).unwrap()), mid, "ascending: {v} is middle");
        assert_eq!(
            rgb(descending.lookup_f64(v).unwrap()),
            rgb(buckets_of(&descending).colors()[1]),
            "descending: {v} must land in the middle bucket, not the top"
        );
    }
    // Every swatch is reachable (the bug left the middle one unpaintable).
    let painted: std::collections::HashSet<_> =
        [0.0, 15.0, 30.0].iter().map(|v| rgb(descending.lookup_f64(*v).unwrap())).collect();
    assert_eq!(painted.len(), 3, "all three swatches must be reachable");
}

/// A fully descending `BinOrdinal` bin list normalizes the same way.
#[test]
fn bin_ordinal_descending_bins_normalize_and_reverse_swatches() {
    let batch = batch_with_numeric_color(vec![5.0, 15.0, 25.0]);
    let resolve = |bins: Vec<f64>| {
        let spec = spec_with_color_scale_json(serde_json::json!({
            "type": "bin-ordinal",
            "bins": bins,
            "scheme": "blues",
        }));
        resolve_color(&spec, &batch).0
    };
    let ascending = resolve(vec![10.0, 20.0]);
    let descending = resolve(vec![20.0, 10.0]);

    assert_eq!(buckets_of(&descending).bounds(), buckets_of(&ascending).bounds());
    assert_eq!(
        buckets_of(&descending).colors().iter().rev().map(|c| rgb(*c)).collect::<Vec<_>>(),
        buckets_of(&ascending).colors().iter().map(|c| rgb(*c)).collect::<Vec<_>>(),
    );
    assert_eq!(
        rgb(descending.lookup_f64(15.0).unwrap()),
        rgb(buckets_of(&descending).colors()[1]),
        "a middle value must land in the middle bucket"
    );
}

/// A non-monotonic boundary list has no defensible bucket assignment, so it is a
/// typed refusal quoting the matching constructor's own rejection sentence —
/// never a silent mis-bucketing and never a quiet fallback to continuous color.
#[test]
fn non_monotonic_declared_boundaries_are_a_typed_error() {
    use crate::render::RenderError;
    let batch = batch_with_numeric_color(vec![5.0, 15.0, 25.0]);

    for (json, field) in [
        (serde_json::json!({ "type": "threshold", "domain": [10.0, 30.0, 20.0] }), "domain"),
        (serde_json::json!({ "type": "bin-ordinal", "bins": [10.0, 30.0, 20.0] }), "bins"),
        // A repeated boundary is an empty bucket, rejected by the constructors
        // (`w[0] >= w[1]`) and rejected identically here.
        (serde_json::json!({ "type": "threshold", "domain": [10.0, 10.0] }), "domain"),
    ] {
        let scale_type = json["type"].as_str().unwrap().to_string();
        let spec = spec_with_color_scale_json(json.clone());
        let err = try_resolve_color(&spec, &batch)
            .expect_err(&format!("{json} must be refused, not silently mis-bucketed"));

        // The message quotes the constructor vocabulary verbatim — one
        // vocabulary across the pyclass and render paths.
        let expected_tail = crate::scale::core::not_strictly_ascending_message(field);
        match &err {
            RenderError::ScaleResolutionFailed(msg) => {
                assert!(
                    msg.contains(&expected_tail) && msg.contains(&scale_type),
                    "message must name the scale type and quote {expected_tail:?}; got {msg:?}"
                );
            }
            other => panic!("expected ScaleResolutionFailed, got {other:?}"),
        }
    }
}

/// The refusal message is word-for-word what the `ThresholdScale` /
/// `BinOrdinalScale` constructors raise, so a user hitting either path reads the
/// same sentence.
#[test]
fn boundary_rejection_message_matches_the_constructor_vocabulary() {
    assert_eq!(
        crate::scale::core::not_strictly_ascending_message("domain"),
        "domain must be strictly sorted ascending",
    );
    assert_eq!(
        crate::scale::core::not_strictly_ascending_message("bins"),
        "bins must be strictly sorted ascending",
    );
}

/// Ascending declared lists — every list the constructors would have accepted —
/// resolve exactly as before the ordering check was added.
#[test]
fn ascending_declared_boundaries_are_unchanged() {
    let batch = batch_with_numeric_color(vec![0.0, 15.0, 30.0]);
    let threshold = spec_with_color_scale_json(serde_json::json!({
        "type": "threshold",
        "domain": [10.0, 20.0],
    }));
    let (scale, warnings) = resolve_color(&threshold, &batch);
    assert!(warnings.is_empty(), "an ascending list must not warn: {warnings:?}");
    assert_eq!(buckets_of(&scale).bounds(), &[0.0, 10.0, 20.0, 30.0]);

    // Single-boundary lists are trivially ascending and must NOT read as
    // descending (which would reverse their two swatches).
    let single = spec_with_color_scale_json(serde_json::json!({
        "type": "threshold",
        "domain": [10.0],
    }));
    let (single_scale, _) = resolve_color(&single, &batch);
    let colors = buckets_of(&single_scale).colors();
    assert_eq!(colors.len(), 2);
    assert_eq!(
        rgb(single_scale.lookup_f64(0.0).unwrap()),
        rgb(colors[0]),
        "the low bucket keeps the first swatch"
    );
}

/// `QuantizeScale::new` rejects a zero-width domain, but `Color(scale={...})`
/// takes a raw dict and bypasses it — so the resolver refuses `domain=[5, 5]`
/// with the constructor's own sentence rather than collapsing every boundary
/// onto one value (`[5, 5, 5, 5]` for three buckets: middle swatches
/// unreachable, no diagnostic). An ordinary two-point domain is untouched.
#[test]
fn degenerate_quantize_domain_is_a_typed_error() {
    use crate::render::RenderError;
    let batch = batch_with_numeric_color(vec![0.0, 5.0, 10.0]);
    let quantize = |domain: Vec<f64>| {
        spec_with_color_scale_json(serde_json::json!({
            "type": "quantize",
            "domain": domain,
            "range": ["#ff0000", "#00ff00", "#0000ff"],
        }))
    };

    let err = try_resolve_color(&quantize(vec![5.0, 5.0]), &batch)
        .expect_err("a zero-width declared domain must be refused");
    match &err {
        RenderError::ScaleResolutionFailed(msg) => assert!(
            msg.contains(crate::scale::core::DEGENERATE_DOMAIN_MESSAGE)
                && msg.contains("quantize"),
            "message must name the scale type and quote the constructor sentence; got {msg:?}"
        ),
        other => panic!("expected ScaleResolutionFailed, got {other:?}"),
    }
    // Word-for-word what `QuantizeScale(domain=[5, 5], ...)` raises.
    assert_eq!(
        crate::scale::core::DEGENERATE_DOMAIN_MESSAGE,
        "domain endpoints must differ (lo != hi)",
    );

    // A non-degenerate two-point domain resolves exactly as before.
    let (scale, warnings) = resolve_color(&quantize(vec![0.0, 30.0]), &batch);
    assert!(warnings.is_empty(), "a well-formed domain must not warn: {warnings:?}");
    assert_eq!(buckets_of(&scale).bounds(), &[0.0, 10.0, 20.0, 30.0]);
    assert_eq!(rgb(scale.lookup_f64(15.0).unwrap()), (0x00, 0xff, 0x00));

    // A constant data column still resolves (its degenerate extent is
    // data-derived, not user-declared, so it is not an error).
    let flat = batch_with_numeric_color(vec![7.0, 7.0]);
    let no_domain = spec_with_color_scale_json(serde_json::json!({
        "type": "quantize",
        "range": ["#ff0000", "#0000ff"],
    }));
    assert!(
        try_resolve_color(&no_domain, &flat).is_ok(),
        "a constant column must still render, not error"
    );
}

// ── Batch A §4.3: appearance-channel scales (fill/stroke opacity, stroke dash) ──

/// x/y plus the columns the §4.3 tests bind: `v` (quantitative, extent
/// `[0.2, 0.9]` — every value already a legal raw alpha, so a test that passes
/// under the old raw-clamp path and the new scaled path cannot be told apart by
/// accident), `level` (categorical), and `dash_idx` (numeric palette indices).
fn appearance_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ArrowDataType::Float64, false),
        Field::new("y", ArrowDataType::Float64, false),
        Field::new("v", ArrowDataType::Float64, false),
        Field::new("level", ArrowDataType::Utf8, false),
        Field::new("dash_idx", ArrowDataType::Float64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
            Arc::new(Float64Array::from(vec![0.2, 0.55, 0.9])),
            Arc::new(StringArray::from(vec!["low", "mid", "high"])),
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
        ],
    )
    .unwrap()
}

/// A point spec on x/y carrying the caller's appearance channels, built by
/// deserializing the **wire** encoding shape Python emits (`{"fill_opacity":
/// {"field": …, "type": …, "scale": {"type": "linear", …}}}`) rather than by
/// hand-assembling `EncodingSpec`/`ScaleSpec` values. A hand-built spec would
/// prove the builders honor a shape the wire may never carry; this proves they
/// honor the one it does. Mirrors `spec_with_color_scale_json` above.
fn appearance_spec(channels: serde_json::Value) -> ChartSpec {
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    let mut encoding: Encoding =
        serde_json::from_value(channels).expect("wire encoding must deserialize");
    encoding.x = Some(EncodingSpec { field: "x".into(), ..Default::default() });
    encoding.y = Some(EncodingSpec { field: "y".into(), ..Default::default() });
    ChartSpec {
        mark: Mark::Point,
        encoding,
        ..Default::default()
    }
}

/// Resolve `spec` against `batch` at a fixed pixel range.
fn resolve_appearance(
    spec: &ChartSpec,
    batch: &RecordBatch,
) -> (ResolvedScales, Vec<crate::render::RenderWarning>) {
    let theme = ThemeInputs::default();
    resolve_scales(spec, batch, (0.0, 100.0), (0.0, 80.0), &theme).unwrap()
}

/// §4.3 default: an unscaled `fill_opacity`/`stroke_opacity` on a quantitative
/// field maps the column's extent onto the theme opacity band — the semantics
/// `opacity` already had — instead of passing the raw value through as an alpha.
///
/// Discriminating by construction: the column's values (0.2, 0.55, 0.9) are all
/// legal alphas, so the old raw path would leave them unchanged; the band maps
/// them to 0.1, 0.55, 1.0. Only the 0.55 midpoint coincides.
#[test]
fn unscaled_fill_and_stroke_opacity_map_the_extent_onto_the_theme_band() {
    let batch = appearance_batch();
    let spec = appearance_spec(serde_json::json!({
        "fill_opacity": {"field": "v", "type": "quantitative"},
        "stroke_opacity": {"field": "v", "type": "quantitative"},
    }));
    let (scales, warns) = resolve_appearance(&spec, &batch);
    assert!(warns.is_empty(), "no warnings expected: {warns:?}");

    for (name, resolved) in [
        ("fill_opacity", scales.fill_opacity),
        ("stroke_opacity", scales.stroke_opacity),
    ] {
        let scale = resolved.unwrap_or_else(|| panic!("{name} must resolve a scale"));
        assert_eq!(
            (scale.min_opacity(), scale.max_opacity()),
            (0.1, 1.0),
            "{name} range must be the theme opacity band"
        );
        let at = |v: f64| scale.inner.to_pixel_f64(v).unwrap();
        assert!((at(0.2) - 0.1).abs() < 1e-9, "{name}: extent min → band min, got {}", at(0.2));
        assert!((at(0.9) - 1.0).abs() < 1e-9, "{name}: extent max → band max, got {}", at(0.9));
        assert!((at(0.55) - 0.55).abs() < 1e-9, "{name}: midpoint, got {}", at(0.55));
    }
}

/// §4.3: every opacity-family channel honors an explicit `scale=` domain AND
/// range. Discriminating: under the defaults the same column resolves domain
/// `[0.2, 0.9]` onto band `[0.1, 1.0]`, so `v = 0.2` would be 0.1; honored, it
/// is 0.35.
#[test]
fn opacity_family_honors_an_explicit_wire_scale() {
    let batch = appearance_batch();
    let scale_json = serde_json::json!({
        "type": "linear", "domain": [0.0, 1.0], "range": [0.25, 0.75],
    });
    let spec = appearance_spec(serde_json::json!({
        "opacity": {"field": "v", "type": "quantitative", "scale": scale_json},
        "fill_opacity": {"field": "v", "type": "quantitative", "scale": scale_json},
        "stroke_opacity": {"field": "v", "type": "quantitative", "scale": scale_json},
    }));
    let (scales, warns) = resolve_appearance(&spec, &batch);
    assert!(warns.is_empty(), "no warnings expected: {warns:?}");

    for (name, resolved) in [
        ("opacity", scales.opacity),
        ("fill_opacity", scales.fill_opacity),
        ("stroke_opacity", scales.stroke_opacity),
    ] {
        let scale = resolved.unwrap_or_else(|| panic!("{name} must resolve a scale"));
        assert_eq!(
            (scale.min_opacity(), scale.max_opacity()),
            (0.25, 0.75),
            "{name} must take the scale's range, not the theme band"
        );
        let (lo, hi) = scale.inner.data_domain().expect("linear domain");
        assert!(
            (lo - 0.0).abs() < 1e-9 && (hi - 1.0).abs() < 1e-9,
            "{name} must take the scale's domain, not the data extent; got [{lo},{hi}]"
        );
        let at_min = scale.inner.to_pixel_f64(0.2).unwrap();
        assert!(
            (at_min - 0.35).abs() < 1e-9,
            "{name}: v=0.2 on domain [0,1] → 0.35, not the extent-relative 0.1; got {at_min}"
        );
    }
}

/// A constant column on any opacity-family channel follows the
/// degenerate-domain rule the size and stroke-width scales follow (and
/// Vega-Lite's linear scales): its one value maps to the middle of the band,
/// 0.55 on the default 0.1..1.0 band, on `opacity`, `fill_opacity` and
/// `stroke_opacity` alike. Rejects a channel that keeps the raw value as a
/// literal alpha (0.3) where its siblings map it.
#[test]
fn a_constant_opacity_family_column_maps_to_the_middle_of_the_band() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ArrowDataType::Float64, false),
        Field::new("y", ArrowDataType::Float64, false),
        Field::new("fo", ArrowDataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![0.3, 0.3, 0.3])),
        ],
    )
    .unwrap();
    let spec = appearance_spec(serde_json::json!({
        "fill_opacity": {"field": "fo", "type": "quantitative"},
        "stroke_opacity": {"field": "fo", "type": "quantitative"},
        "opacity": {"field": "fo", "type": "quantitative"},
    }));
    let (scales, warns) = resolve_appearance(&spec, &batch);
    assert!(warns.is_empty(), "no warnings expected: {warns:?}");
    for (channel, scale) in [
        ("opacity", &scales.opacity),
        ("fill_opacity", &scales.fill_opacity),
        ("stroke_opacity", &scales.stroke_opacity),
    ] {
        let scale = scale.as_ref().unwrap_or_else(|| panic!("constant {channel} resolves a scale"));
        assert_eq!(scale.inner.data_domain(), Some((0.3, 0.3)), "{channel}");
        let midpoint = scale.inner.to_pixel_f64(0.3).unwrap();
        assert!((midpoint - 0.55).abs() < 1e-12, "{channel} maps a constant column to the band midpoint: {midpoint}");
    }

    // An explicit `scale=` domain is a real domain even over a constant column,
    // so it still resolves — the guard is about an ABSENT domain, not about
    // refusing to scale constant data the user described.
    let scaled = appearance_spec(serde_json::json!({
        "fill_opacity": {
            "field": "fo", "type": "quantitative",
            "scale": {"type": "linear", "domain": [0.0, 1.0], "range": [0.2, 0.8]},
        },
    }));
    let (scales, _) = resolve_appearance(&scaled, &batch);
    let scale = scales.fill_opacity.expect("an explicit domain resolves over a constant column");
    assert!((scale.inner.to_pixel_f64(0.3).unwrap() - 0.38).abs() < 1e-9);
}

/// §4.1/§6.10 (`#108`): an explicit `scale=` RANGE is a real scale even over a
/// constant column, same as an explicit domain. A range-only `LinearScale`
/// over a degenerate extent resolves the ordinary clamped, un-niced linear
/// scale and collapses to the range's midpoint through the existing
/// degenerate-domain rule (GH #104) — 0.5 for `[0.2, 0.8]`, and still 0.5 for
/// `[-1, 2]` once its endpoints clamp into `[0, 1]` first.
#[test]
fn a_range_only_linear_scale_resolves_over_a_constant_opacity_column() {
    let batch = appearance_batch_constant_v(0.3);

    for channel in ["fill_opacity", "stroke_opacity"] {
        let spec = appearance_spec(serde_json::json!({
            channel: {
                "field": "v", "type": "quantitative",
                "scale": {"type": "linear", "range": [0.2, 0.8]},
            },
        }));
        let (scales, warns) = resolve_appearance(&spec, &batch);
        assert!(warns.is_empty(), "{channel}: no warnings expected: {warns:?}");
        let resolved = match channel {
            "fill_opacity" => scales.fill_opacity,
            "stroke_opacity" => scales.stroke_opacity,
            _ => unreachable!(),
        };
        let scale = resolved
            .unwrap_or_else(|| panic!("{channel}: a range-only scale must resolve over a constant column"));
        let at = scale.inner.to_pixel_f64(0.3).unwrap();
        assert!((at - 0.5).abs() < 1e-9, "{channel}: range [0.2,0.8] over a constant column → 0.5; got {at}");

        let clamped_spec = appearance_spec(serde_json::json!({
            channel: {
                "field": "v", "type": "quantitative",
                "scale": {"type": "linear", "range": [-1.0, 2.0]},
            },
        }));
        let (clamped_scales, clamped_warns) = resolve_appearance(&clamped_spec, &batch);
        assert!(clamped_warns.is_empty(), "{channel}: no warnings expected: {clamped_warns:?}");
        let clamped_scale = match channel {
            "fill_opacity" => clamped_scales.fill_opacity,
            "stroke_opacity" => clamped_scales.stroke_opacity,
            _ => unreachable!(),
        }
        .unwrap_or_else(|| panic!("{channel}: a range-only scale must resolve over a constant column"));
        let clamped_at = clamped_scale.inner.to_pixel_f64(0.3).unwrap();
        assert!(
            (clamped_at - 0.5).abs() < 1e-9,
            "{channel}: range [-1,2] clamps to [0,1] first, so a constant column → 0.5; got {clamped_at}"
        );
    }
}

/// Byte-identity guard for the absent case (§7): a chart binding none of the
/// appearance channels resolves none of the new scales and warns about nothing,
/// so nothing downstream can behave differently than it did pre-batch-A.
#[test]
fn charts_without_appearance_channels_resolve_no_appearance_scales() {
    let batch = appearance_batch();
    let spec = appearance_spec(serde_json::json!({}));
    let (scales, warns) = resolve_appearance(&spec, &batch);
    assert!(scales.fill_opacity.is_none(), "no fill_opacity channel → no scale");
    assert!(scales.stroke_opacity.is_none(), "no stroke_opacity channel → no scale");
    assert!(scales.stroke_dash.is_none(), "no stroke_dash channel → no scale");
    assert!(warns.is_empty(), "no warnings expected: {warns:?}");
}

/// §4.3: a categorical `stroke_dash` field resolves onto the dash index
/// space's legacy slots — index 0 solid, then the three patterns that
/// predate the named-linetype append, in order (never the appended
/// linetypes; see `STROKE_DASH_LEGACY_SLOTS`).
#[test]
fn categorical_stroke_dash_maps_onto_the_dash_palette() {
    use ferrum_scene::stroke_dash_index;
    let batch = appearance_batch();
    let spec = appearance_spec(serde_json::json!({
        "stroke_dash": {"field": "level", "type": "nominal"},
    }));
    let (scales, warns) = resolve_appearance(&spec, &batch);
    assert!(warns.is_empty(), "3 categories fit the palette: {warns:?}");
    let dash = scales.stroke_dash.expect("a categorical dash field must resolve a scale");

    assert_eq!(dash.domain, vec!["low", "mid", "high"], "first-appearance order");
    assert_eq!(dash.dash_for("low"), None, "index 0 is solid → no dasharray");
    assert_eq!(dash.dash_for("mid"), Some(vec![6.0, 3.0]));
    assert_eq!(dash.dash_for("high"), Some(vec![2.0, 3.0]));

    // `lookup` keeps the distinction `dash_for` collapses: solid is an empty
    // pattern IN the domain, an unknown value is absent from it.
    assert_eq!(dash.lookup("low"), Some(&[][..]));
    assert_eq!(dash.lookup("nope"), None);
    assert_eq!(dash.dash_for("nope"), None);

    // Packed-index parity (§7): each resolved pattern round-trips to the palette
    // index the GPU path packs, so SVG and WASM draw the same dash.
    assert_eq!(stroke_dash_index(dash.dash_for("low").as_deref()), 0);
    assert_eq!(stroke_dash_index(dash.dash_for("mid").as_deref()), 1);
    assert_eq!(stroke_dash_index(dash.dash_for("high").as_deref()), 2);
}

/// The numeric contract is untouched (§6): a quantitative `stroke_dash` column
/// resolves NO scale, so mark builders keep reading it as palette indices.
#[test]
fn numeric_stroke_dash_resolves_no_scale() {
    let batch = appearance_batch();
    let spec = appearance_spec(serde_json::json!({"stroke_dash": {"field": "dash_idx"}}));
    let (scales, warns) = resolve_appearance(&spec, &batch);
    assert!(
        scales.stroke_dash.is_none(),
        "a numeric dash column keeps the index contract, it does not resolve a scale"
    );
    assert!(warns.is_empty(), "no warnings expected: {warns:?}");
}

/// …and the user's declared type wins: the documented
/// `StrokeDash("group", type_="N")` form resolves categorically even though the
/// column is numeric.
#[test]
fn a_numeric_dash_column_typed_nominal_resolves_categorically() {
    let batch = appearance_batch();
    let spec = appearance_spec(serde_json::json!({
        "stroke_dash": {"field": "dash_idx", "type": "nominal"},
    }));
    let (scales, _) = resolve_appearance(&spec, &batch);
    let dash = scales.stroke_dash.expect("an explicitly nominal dash field resolves a scale");
    assert_eq!(dash.domain.len(), 3, "one entry per distinct value: {:?}", dash.domain);
    assert_eq!(dash.dash_for(&dash.domain[0].clone()), None, "first category is solid");
    assert_eq!(dash.dash_for(&dash.domain[1].clone()), Some(vec![6.0, 3.0]));
}

/// §4.3: the dash palette is finite, so more categories than slots cycles — and
/// says so, mirroring `ShapePaletteOverflowed`. The warning reaches the caller
/// through `resolve_scales`, which is the registration proof.
#[test]
fn stroke_dash_palette_overflow_warns_and_cycles() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ArrowDataType::Float64, false),
        Field::new("y", ArrowDataType::Float64, false),
        Field::new("level", ArrowDataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
            Arc::new(StringArray::from(vec!["a", "b", "c", "d", "e", "f"])),
        ],
    )
    .unwrap();
    let spec = appearance_spec(serde_json::json!({
        "stroke_dash": {"field": "level", "type": "nominal"},
    }));
    let (scales, warns) = resolve_appearance(&spec, &batch);
    assert!(
        warns.contains(&crate::render::RenderWarning::StrokeDashPaletteOverflowed {
            categories: 6,
        }),
        "6 categories over 4 dash slots must warn; got {warns:?}"
    );
    let dash = scales.stroke_dash.unwrap();
    // 4 slots (solid + 3 patterns), so the 5th and 6th categories repeat the
    // 1st and 2nd — the honest degradation the warning names.
    assert_eq!(dash.dash_for("e"), dash.dash_for("a"));
    assert_eq!(dash.dash_for("f"), dash.dash_for("b"));
}

/// The dash domain honors `sort` through the same primitive the shape scale
/// uses, so a sorted dash legend and sorted dash assignment cannot diverge.
#[test]
fn stroke_dash_domain_honors_sort() {
    let batch = appearance_batch();
    let spec = appearance_spec(serde_json::json!({
        "stroke_dash": {"field": "level", "type": "nominal", "sort": "descending"},
    }));
    let (scales, warns) = resolve_appearance(&spec, &batch);
    assert!(warns.is_empty(), "no warnings expected: {warns:?}");
    let dash = scales.stroke_dash.unwrap();
    assert_eq!(dash.domain, vec!["mid", "low", "high"], "descending byte order");
    assert_eq!(dash.dash_for("mid"), None, "the sorted-first category takes index 0");
}

/// Facet sharing (T3's rule, extended to the new channels): a faceted chart's
/// per-panel `fill_opacity` unions the global `__final__` extent, so the same
/// value paints the same alpha in every panel.
#[test]
fn shared_facet_fill_opacity_unions_the_global_domain() {
    use crate::layout::facet::{FacetMode, FacetResolve, FacetSpec};
    let panel_a = t3_op_batch(vec![0.0, 1.0], vec![0.0, 1.0], vec![1.0, 10.0]);
    let panel_b = t3_op_batch(vec![0.0, 1.0], vec![0.0, 1.0], vec![1.0, 100.0]);
    let global = t3_op_batch(
        vec![0.0, 1.0, 0.0, 1.0],
        vec![0.0, 1.0, 0.0, 1.0],
        vec![1.0, 10.0, 1.0, 100.0],
    );
    let outputs = final_outputs(global);
    let mut spec = appearance_spec(serde_json::json!({
        "fill_opacity": {"field": "op", "type": "quantitative"},
    }));
    spec.facet = Some(FacetSpec {
        field: "panel".into(),
        row: None,
        title: None,
        row_title: None,
        sort: None,
        row_sort: None,
        repeat: Vec::new(),
        mode: FacetMode::Wrap { ncols: 2 },
        spacing: None,
        resolve: FacetResolve::default(),
    });
    let theme = ThemeInputs::default();
    let pr = (0.0, 100.0);

    let (scales_a, _) =
        resolve_scales_with_outputs(&spec, &panel_a, &outputs, pr, pr, &theme).unwrap();
    let (scales_b, _) =
        resolve_scales_with_outputs(&spec, &panel_b, &outputs, pr, pr, &theme).unwrap();
    let a = scales_a.fill_opacity.expect("panel A fill_opacity scale");
    let b = scales_b.fill_opacity.expect("panel B fill_opacity scale");

    let (a_lo, a_hi) = a.inner.data_domain().expect("panel A domain");
    assert!(
        a_lo <= 1.0 + 1e-6 && a_hi >= 100.0 - 1e-6,
        "panel A must span the GLOBAL [1,100] extent, got [{a_lo},{a_hi}]"
    );
    let px_a = a.inner.to_pixel_f64(10.0).unwrap();
    let px_b = b.inner.to_pixel_f64(10.0).unwrap();
    assert!(
        (px_a - px_b).abs() < 1e-9,
        "op=10 must paint the same alpha in both panels; A={px_a}, B={px_b}"
    );
    assert!(
        px_a < 0.5 * (a.min_opacity() + a.max_opacity()),
        "op=10 sits low on the global domain, so it must be below the band midpoint; got {px_a}"
    );
}

// --- Fix round (spec §4.3, amended 2026-09-01, T6 quality review) -------

/// Ruling 1: a non-numeric column bound to `fill_opacity`/`stroke_opacity` is
/// a typed render error naming the channel and the offending dtype — the
/// pre-batch-A silent default-alpha fallback (`col_as_f64(...).ok() -> None`
/// at the mark drawer) is retired as a member of the silent-drop class this
/// batch remediates. Pins the exact message text (not just "is an error")
/// because a mixed-channel chart needs the channel disambiguated.
#[test]
fn non_numeric_opacity_family_column_raises_a_typed_error() {
    let batch = appearance_batch(); // "level" is Utf8
    let theme = ThemeInputs::default();
    for channel in ["fill_opacity", "stroke_opacity"] {
        let spec = appearance_spec(serde_json::json!({
            channel: {"field": "level", "type": "quantitative"},
        }));
        let err = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme)
            .expect_err(&format!("{channel} over a Utf8 column must raise, not default-alpha"));
        assert_eq!(
            err.to_string(),
            format!("{channel}: column 'level' has unsupported dtype: Utf8"),
            "message must name the channel and dtype so a mixed-channel chart is diagnosable"
        );
    }
}

/// Ruling 3: an explicit `scale=` range on an opacity-family channel clamps
/// into `[0.0, 1.0]` before the scale is built, so SVG (which only omits the
/// attribute at >= 1.0) and the GPU-packed instance buffer (which multiplies
/// the resolved value directly, uncomparable) read the SAME already-bounded
/// number — they cannot diverge because there is only one number to diverge
/// from. Covers both an over-range (`[0, 5]`) and a negative endpoint.
#[test]
fn svg_and_packed_alpha_agree_because_the_range_clamps_into_unit_interval() {
    let batch = appearance_batch();
    let over_range = appearance_spec(serde_json::json!({
        "fill_opacity": {
            "field": "v", "type": "quantitative",
            "scale": {"type": "linear", "domain": [0.0, 1.0], "range": [0.0, 5.0]},
        },
    }));
    let (scales, warns) = resolve_appearance(&over_range, &batch);
    assert!(warns.is_empty(), "a Linear scale must not warn: {warns:?}");
    let scale = scales.fill_opacity.expect("fill_opacity must resolve");
    assert_eq!(
        (scale.min_opacity(), scale.max_opacity()),
        (0.0, 1.0),
        "range=[0,5] must clamp to [0,1], not pass 5.0 through to either backend"
    );
    let at_max = scale.inner.to_pixel_f64(1.0).unwrap();
    assert!(
        (at_max - 1.0).abs() < 1e-9,
        "domain max must map to the CLAMPED range max (1.0), not 5.0; got {at_max}"
    );

    let negative_range = appearance_spec(serde_json::json!({
        "stroke_opacity": {
            "field": "v", "type": "quantitative",
            "scale": {"type": "linear", "domain": [0.0, 1.0], "range": [-2.0, 0.5]},
        },
    }));
    let (scales, _) = resolve_appearance(&negative_range, &batch);
    let scale = scales.stroke_opacity.expect("stroke_opacity must resolve");
    assert_eq!(
        (scale.min_opacity(), scale.max_opacity()),
        (0.0, 0.5),
        "a negative endpoint must clamp to 0.0, never a negative alpha"
    );
}

/// `appearance_batch`'s "v" column with every row equal: a degenerate
/// domain.
fn appearance_batch_constant_v(value: f64) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ArrowDataType::Float64, false),
        Field::new("y", ArrowDataType::Float64, false),
        Field::new("v", ArrowDataType::Float64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
            Arc::new(Float64Array::from(vec![value, value, value])),
        ],
    )
    .unwrap()
}

/// Render `spec`/`batch` through the FULL pipeline (`render_svg`, not
/// `resolve_scales`): `auxiliary.rs`'s warnings are pushed from both the
/// provisional pass (`prepare_render_inputs`) and the per-panel pass
/// (`scene_build`), so only the full pipeline exercises the count rule the
/// tests below assert.
fn render_appearance(spec: &ChartSpec, batch: &RecordBatch) -> crate::render::RenderOutput<String> {
    crate::render::render_svg(
        spec,
        batch,
        &ThemeInputs::default(),
        crate::layout::Viewport { width: 600.0, height: 400.0 },
        &crate::render::config::RenderConfig::default(),
        &crate::render::chart_config::ChartConfig::default(),
    )
    .expect("render must succeed")
}

/// Attach a 3-panel Wrap facet on `field` to `spec` — shared by the three
/// faceted whole-chart-fact tests below.
fn n2_with_facet(mut spec: ChartSpec, field: &str) -> ChartSpec {
    use crate::layout::facet::{FacetMode, FacetResolve, FacetSpec};
    spec.facet = Some(FacetSpec {
        field: field.into(),
        row: None,
        title: None,
        row_title: None,
        sort: None,
        row_sort: None,
        repeat: Vec::new(),
        mode: FacetMode::Wrap { ncols: 2 },
        spacing: None,
        resolve: FacetResolve::default(),
    });
    spec
}

/// x/y/`cat` (one row per entry of `categories`, repeated identically in
/// each of 3 panels) plus a 3-value `grp` facet column — shared by the
/// faceted shape and color palette-overflow tests, which differ only in how
/// many categories overflow which palette.
fn n2_faceted_categorical_batch(categories: &[&str]) -> RecordBatch {
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut grp = Vec::new();
    let mut cat = Vec::new();
    let mut n = 0.0;
    for panel in ["p1", "p2", "p3"] {
        for c in categories {
            n += 1.0;
            x.push(n);
            y.push(n * 10.0);
            grp.push(panel);
            cat.push((*c).to_string());
        }
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ArrowDataType::Float64, false),
        Field::new("y", ArrowDataType::Float64, false),
        Field::new("grp", ArrowDataType::Utf8, false),
        Field::new("cat", ArrowDataType::Utf8, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(x)),
            Arc::new(Float64Array::from(y)),
            Arc::new(StringArray::from(grp)),
            Arc::new(StringArray::from(cat)),
        ],
    )
    .unwrap()
}

/// `ShapePaletteOverflowed` is recorded once per render: it is
/// pushed from `auxiliary.rs` and is a whole-chart fact.
#[test]
fn shape_palette_overflow_warning_is_recorded_once_per_render_not_once_per_resolution_pass() {
    let mut spec = make_spec_with_color();
    spec.encoding.shape = spec.encoding.color.clone();
    spec.encoding.color = None;
    // SHAPE_PALETTE has 8 entries; 9 distinct categories overflow it.
    let batch = {
        let schema = make_batch_q_q_n().schema();
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from((1..=9).map(f64::from).collect::<Vec<_>>())),
                Arc::new(Float64Array::from((1..=9).map(|v| v as f64 * 10.0).collect::<Vec<_>>())),
                Arc::new(StringArray::from((1..=9).map(|i| format!("c{i}")).collect::<Vec<_>>())),
            ],
        )
        .unwrap()
    };
    let out = render_appearance(&spec, &batch);
    let count = out
        .warnings
        .iter()
        .filter(|w| matches!(w, crate::render::RenderWarning::ShapePaletteOverflowed { .. }))
        .count();
    assert_eq!(
        count, 1,
        "ShapePaletteOverflowed must be recorded exactly once per render, got {count}: {:?}",
        out.warnings
    );
}

/// `ShapePaletteOverflowed` is recorded once per render across a faceted panel loop.
#[test]
fn shape_palette_overflow_warning_is_recorded_once_per_render_across_a_faceted_panel_loop() {
    let categories: Vec<String> = (1..=9).map(|i| format!("c{i}")).collect();
    let categories: Vec<&str> = categories.iter().map(String::as_str).collect();
    let batch = n2_faceted_categorical_batch(&categories);
    let out = render_appearance(&n2_faceted_shape_spec(), &batch);
    let count = out
        .warnings
        .iter()
        .filter(|w| matches!(w, crate::render::RenderWarning::ShapePaletteOverflowed { .. }))
        .count();
    assert_eq!(
        count, 1,
        "ShapePaletteOverflowed must be recorded exactly once per render across a 3-panel \
         facet, got {count}: {:?}", out.warnings
    );
}

fn n2_faceted_shape_spec() -> ChartSpec {
    let mut spec = make_spec_with_color();
    spec.encoding.shape = spec.encoding.color.take();
    spec.encoding.shape.as_mut().unwrap().field = "cat".into();
    n2_with_facet(spec, "grp")
}

/// `ColorPaletteOverflowed` is the deliberate asymmetry (see
/// `panel_pass_warning_restated_by_every_panel`'s doc): its provisional-pass
/// copy is superseded exactly like the three auxiliary warnings above, but
/// each panel's OWN copy survives, since a panel with an
/// independently-resolved color scale can legitimately overflow differently
/// from its siblings. Exactly 3 (one per panel), not 4 (the provisional
/// pass's extra copy) and not 1 (which would silently drop 2 real per-panel
/// facts).
#[test]
fn color_palette_overflow_drops_its_provisional_copy_but_keeps_every_panel_copy() {
    // paper_ink (the default theme scheme) has 8 entries; 12 categories overflow it.
    let categories: Vec<String> = (1..=12).map(|i| format!("c{i}")).collect();
    let categories: Vec<&str> = categories.iter().map(String::as_str).collect();
    let batch = n2_faceted_categorical_batch(&categories);
    let out = render_appearance(&n2_faceted_color_spec(), &batch);
    let count = out
        .warnings
        .iter()
        .filter(|w| matches!(w, crate::render::RenderWarning::ColorPaletteOverflowed { .. }))
        .count();
    assert_eq!(
        count, 3,
        "ColorPaletteOverflowed must keep one copy per panel (3), not the provisional pass's \
         extra copy (4) or a collapsed single copy (1); got {count}: {:?}", out.warnings
    );
}

fn n2_faceted_color_spec() -> ChartSpec {
    let mut spec = make_spec_with_color();
    spec.encoding.color.as_mut().unwrap().field = "cat".into();
    n2_with_facet(spec, "grp")
}

// ── to_pixel_f64_extrapolated (batch-E T13, mark_function(clip=False)) ──────

/// RED-proven: an out-of-domain input must map PAST the pixel range, not
/// drop to `None`. Domain [0,10] -> range [0,100]; x=15 is 50% past the
/// domain's upper edge, so the extrapolated pixel must be 150 (bit-exact —
/// this is a plain affine extension of the same formula `to_pixel_f64` uses
/// for in-domain values).
#[test]
fn to_pixel_f64_extrapolated_continues_past_domain_linear() {
    let scale = ScaleKind::Linear(LinearScale::new_internal(
        vec![0.0, 10.0], vec![0.0, 100.0], false, false,
    ));
    assert_eq!(scale.to_pixel_f64(15.0), None, "the NaN-dropping sibling must still drop this");
    let extrapolated = scale.to_pixel_f64_extrapolated(15.0)
        .expect("extrapolated mapping must not drop an out-of-domain point");
    assert!((extrapolated - 150.0).abs() < 1e-9, "expected 150.0, got {extrapolated}");

    // Below the domain, too.
    let below = scale.to_pixel_f64_extrapolated(-5.0)
        .expect("extrapolated mapping must not drop a below-domain point either");
    assert!((below - (-50.0)).abs() < 1e-9, "expected -50.0, got {below}");
}

/// In-domain inputs are byte-identical between the two methods — extrapolation
/// only changes behavior for points the domain check would have rejected.
#[test]
fn to_pixel_f64_extrapolated_matches_to_pixel_f64_in_domain() {
    let scale = ScaleKind::Linear(LinearScale::new_internal(
        vec![0.0, 10.0], vec![0.0, 100.0], false, false,
    ));
    for x in [0.0, 2.5, 5.0, 7.5, 10.0] {
        let plain = scale.to_pixel_f64(x).unwrap();
        let extrapolated = scale.to_pixel_f64_extrapolated(x).unwrap();
        assert_eq!(plain.to_bits(), extrapolated.to_bits(), "must be bit-identical in-domain at x={x}");
    }
}

/// `Ordinal` has no continuous mapping to extrapolate — both methods return
/// `None`, matching `to_pixel_f64`'s existing contract.
#[test]
fn to_pixel_f64_extrapolated_ordinal_returns_none() {
    use crate::scale::discrete::DiscreteLayout;
    use crate::scale::ordinal::OrdinalScale;
    let scale = ScaleKind::Ordinal(OrdinalScale::new_internal(
        vec!["a".into(), "b".into()],
        vec![0.0, 100.0],
        DiscreteLayout::UNPADDED,
    ));
    assert_eq!(scale.to_pixel_f64_extrapolated(1.0), None);
}

// --- Wave 4 T6: single-axis eligibility reads the marks a spec DRAWS ------

use crate::spec::layer::Layer;

/// Shared builder for the layered single-axis tests below — only the root mark
/// and the layer list vary between them, so a full `ChartSpec` literal per test
/// would be near-verbatim duplication of this file's fixture (operating
/// principle 11, the `arc_spec_with_coord` precedent above). The chart encoding
/// binds `x` alone: the shape the single-axis exemption exists for, and the one
/// layer promotion leaves behind.
fn layered_x_only_spec(mark: crate::spec::mark::Mark, layers: Vec<Layer>) -> ChartSpec {
    let mut spec = make_spec_with_color();
    spec.mark = mark;
    spec.encoding = single_channel_encoding("x");
    spec.layers = Some(layers);
    spec
}

/// An `Encoding` binding exactly `field` on the channel of the same name —
/// `"x"` or `"y"`, the two `make_batch_q_q_n` positional columns.
fn single_channel_encoding(field: &str) -> crate::spec::encoding::Encoding {
    use crate::spec::encoding::{DataType, Encoding, EncodingSpec};
    let bound = Some(EncodingSpec {
        field: field.into(),
        type_: Some(DataType::Quantitative),
        ..Default::default()
    });
    match field {
        "x" => Encoding { x: bound, ..Default::default() },
        "y" => Encoding { y: bound, ..Default::default() },
        other => panic!("single_channel_encoding: unexpected channel {other}"),
    }
}

fn layer_with(mark: crate::spec::mark::Mark, encoding: crate::spec::encoding::Encoding) -> Layer {
    Layer {
        mark,
        encoding,
        ..Default::default()
    }
}

/// The Rust-side mirror of Python's `_CarrierLayer` (`chart.py`): a layer built
/// `opacity`/`size` 0 so it paints nothing. Kept as its own fixture so a test
/// about the carrier cannot accidentally assert about a VISIBLE layer — the
/// distinction the eligibility predicate turns on.
fn invisible_carrier_layer(
    mark: crate::spec::mark::Mark,
    encoding: crate::spec::encoding::Encoding,
) -> Layer {
    Layer {
        mark_style: Some(crate::spec::mark_style::MarkKwargsSpec {
            size: Some(0.0),
            opacity: Some(0.0),
            ..Default::default()
        }),
        ..layer_with(mark, encoding)
    }
}

fn resolved_domain(scale: &ScaleKind, what: &str) -> [f64; 2] {
    match scale {
        ScaleKind::Linear(l) => l.domain_pair(),
        other => panic!("expected a Linear {what} scale, got {other:?}"),
    }
}

/// A layered spec whose layers draw only rug ticks is single-axis eligible even
/// though its root mark slot holds a placeholder.
///
/// Python's layer promotion empties the mark slot and the wire encoder
/// substitutes `"point"`, so the root mark is not evidence of what is drawn.
/// Pre-fix (RED) the predicate read that placeholder, found `Point`, and the
/// unconditional `spec.encoding.y.ok_or(EncodingTypeMismatch)` fired.
#[test]
fn placeholder_root_mark_with_tick_layers_resolves_via_the_single_axis_exemption() {
    use crate::spec::mark::Mark;

    let spec = layered_x_only_spec(
        Mark::Point,
        vec![
            layer_with(Mark::Tick, single_channel_encoding("x")),
            layer_with(Mark::Tick, single_channel_encoding("x")),
        ],
    );
    let batch = make_batch_q_q_n();
    let theme = ThemeInputs::default();

    let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme)
        .expect("a spec whose layers draw only ticks must resolve, not error on a missing y");

    assert_eq!(
        resolved_domain(&scales.x, "x"),
        [1.0, 6.0],
        "the bound x axis must be the real data-driven scale"
    );
    assert_eq!(
        resolved_domain(&scales.y, "y"),
        [0.0, 1.0],
        "the unbound y axis must be the synthesized dummy unit scale"
    );
}

/// A later layer's own `y` binding reaches scale resolution rather than being
/// papered over by the dummy unit scale.
///
/// `(tick_chart + line_chart)` keeps `tick` in the root slot while drawing a
/// line that needs a real y-domain. Pre-fix (RED) the root mark alone granted
/// the exemption and `y` resolved to `[0.0, 1.0]`, squashing the line into the
/// bottom sliver of the panel under a fabricated `0…1` axis.
#[test]
fn layer_bound_y_resolves_a_real_domain_instead_of_the_dummy_scale() {
    use crate::spec::encoding::{DataType, Encoding, EncodingSpec};
    use crate::spec::mark::Mark;

    let line_encoding = Encoding {
        y: Some(EncodingSpec {
            field: "y".into(),
            type_: Some(DataType::Quantitative),
            ..Default::default()
        }),
        ..single_channel_encoding("x")
    };
    let spec = layered_x_only_spec(
        Mark::Tick,
        vec![
            layer_with(Mark::Tick, single_channel_encoding("x")),
            layer_with(Mark::Line, line_encoding),
        ],
    );
    let batch = make_batch_q_q_n();
    let theme = ThemeInputs::default();

    let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme)
        .expect("a layer-bound y must resolve, not error");

    // `make_batch_q_q_n`'s "y" column is `[10.0 … 60.0]`; `[0.0, 1.0]` here
    // would be `dummy_unit_scale`, which is also `ScaleKind::Linear`.
    assert_eq!(
        resolved_domain(&scales.y, "y"),
        [10.0, 60.0],
        "the line layer's y column must drive the y axis, not a fabricated unit scale"
    );
}

/// When NO layer binds `y` either, the dummy unit scale is still the answer —
/// the eligible mark set is unchanged and a genuinely one-axis spec resolves
/// exactly as before.
#[test]
fn tick_layers_binding_no_y_anywhere_keep_the_dummy_scale() {
    use crate::spec::mark::Mark;

    let spec = layered_x_only_spec(
        Mark::Tick,
        vec![layer_with(Mark::Tick, single_channel_encoding("x"))],
    );
    let batch = make_batch_q_q_n();
    let theme = ThemeInputs::default();

    let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme)
        .expect("a tick-only layered spec must keep resolving");

    assert_eq!(resolved_domain(&scales.x, "x"), [1.0, 6.0]);
    assert_eq!(resolved_domain(&scales.y, "y"), [0.0, 1.0]);
}

/// An invisible CARRIER layer does not vote on eligibility, so it cannot
/// disqualify a spec whose real content draws from one axis.
///
/// `annotate_hline() + annotate_vline()` carries its visual content in
/// annotation primitives and puts a deliberately invisible `point` carrier layer
/// in the layer list (`chart.py`'s `_CarrierLayer`, `opacity`/`size` 0) purely so
/// the spec has a layer at all. A plain all-layers rule would read that `Point`
/// and refuse the chart; [`layer_paints`] excludes it, no drawing mark is left to
/// object, and the y-only exemption stands.
#[test]
fn invisible_carrier_layer_does_not_disqualify_the_single_axis_exemption() {
    use crate::spec::mark::Mark;

    let mut spec = layered_x_only_spec(
        Mark::Rule,
        vec![invisible_carrier_layer(
            Mark::Point,
            single_channel_encoding("y"),
        )],
    );
    spec.encoding = single_channel_encoding("y");
    let batch = make_batch_q_q_n();
    let theme = ThemeInputs::default();

    let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme)
        .expect("an invisible carrier layer must not disqualify the y-only exemption");

    assert_eq!(resolved_domain(&scales.y, "y"), [10.0, 60.0]);
    assert_eq!(
        resolved_domain(&scales.x, "x"),
        [0.0, 1.0],
        "nothing binds x, so it must stay the synthesized dummy unit scale"
    );
}

/// The same carrier layer made VISIBLE disqualifies the spec — the discriminator
/// is that it paints, not that it is a `Point` in a layer list.
///
/// This is the mutation guard for [`layer_paints`]: without the invisibility
/// test, `invisible_carrier_layer_does_not_disqualify_the_single_axis_exemption`
/// above and this test cannot both pass.
#[test]
fn a_visible_point_layer_in_the_same_shape_does_disqualify() {
    use crate::spec::mark::Mark;

    let mut spec = layered_x_only_spec(
        Mark::Rule,
        vec![layer_with(Mark::Point, single_channel_encoding("y"))],
    );
    spec.encoding = single_channel_encoding("y");
    let batch = make_batch_q_q_n();
    let theme = ThemeInputs::default();

    let err = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme)
        .expect_err("a point layer that actually paints needs a real x axis, so this must error");
    match err {
        RenderError::EncodingTypeMismatch { channel, .. } => assert_eq!(channel, "x"),
        other => panic!("expected EncodingTypeMismatch{{channel: \"x\", ..}}, got {other:?}"),
    }
}

/// **The case the predicate turns on.** One eligible mark must not exempt the
/// ineligible mark beside it. A rug tick and a point, both x-only, with nothing
/// binding `y` anywhere: the tick could render from x alone, the point cannot,
/// and there is no `y` for the point to fall back to. This must REFUSE.
///
/// Under an `any`-over-marks rule it rendered instead, placing every point on a
/// fabricated `0…1` axis — measured through the Python API as a chart with the
/// `['1','0.9', … ,'0']` ladder and ZERO `<circle>` elements, i.e. the point
/// layer's whole dataset silently dropped. This is the mixed eligible/ineligible
/// set; an all-`Point` layer list would answer the same under `any`, `all`, or
/// root-only and so proves nothing about the choice.
#[test]
fn a_mixed_eligible_and_ineligible_layer_set_refuses() {
    use crate::spec::mark::Mark;

    let spec = layered_x_only_spec(
        Mark::Tick,
        vec![
            layer_with(Mark::Tick, single_channel_encoding("x")),
            layer_with(Mark::Point, single_channel_encoding("x")),
        ],
    );
    let batch = make_batch_q_q_n();
    let theme = ThemeInputs::default();

    let err = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme).expect_err(
        "a point layer beside a tick, with no y bound anywhere, must refuse rather than \
         drop the points onto a fabricated axis",
    );
    match err {
        RenderError::EncodingTypeMismatch { channel, .. } => assert_eq!(channel, "y"),
        other => panic!("expected EncodingTypeMismatch{{channel: \"y\", ..}}, got {other:?}"),
    }
}

/// The same refusal with no eligible mark anywhere in the set — the baseline the
/// mixed case above must match.
#[test]
fn layered_points_without_any_y_binding_still_error() {
    use crate::spec::mark::Mark;

    let spec = layered_x_only_spec(
        Mark::Point,
        vec![
            layer_with(Mark::Point, single_channel_encoding("x")),
            layer_with(Mark::Point, single_channel_encoding("x")),
        ],
    );
    let batch = make_batch_q_q_n();
    let theme = ThemeInputs::default();

    let err = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme)
        .expect_err("no eligible mark and no y binding anywhere must still error");
    match err {
        RenderError::EncodingTypeMismatch { channel, .. } => assert_eq!(channel, "y"),
        other => panic!("expected EncodingTypeMismatch{{channel: \"y\", ..}}, got {other:?}"),
    }
}

/// The fallback's X arm, symmetric with
/// [`layer_bound_y_resolves_a_real_domain_instead_of_the_dummy_scale`]. A rule
/// spanning `y` layered with a tick binding `x` resolves the tick's real x
/// domain; pre-fix the root `Rule` took the y-only arm and x was the fabricated
/// unit scale. Pinned even though no golden or pre-existing test exercises this
/// direction — absence from the corpus is exactly why the symmetric arm of a
/// symmetric mechanism needs its own test.
#[test]
fn layer_bound_x_resolves_a_real_domain_instead_of_the_dummy_scale() {
    use crate::spec::mark::Mark;

    let mut spec = layered_x_only_spec(
        Mark::Rule,
        vec![
            layer_with(Mark::Rule, single_channel_encoding("y")),
            layer_with(Mark::Tick, single_channel_encoding("x")),
        ],
    );
    spec.encoding = single_channel_encoding("y");
    let batch = make_batch_q_q_n();
    let theme = ThemeInputs::default();

    let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme)
        .expect("a layer-bound x must resolve, not error");

    assert_eq!(
        resolved_domain(&scales.x, "x"),
        [1.0, 6.0],
        "the tick layer's x column must drive the x axis, not a fabricated unit scale"
    );
    assert_eq!(resolved_domain(&scales.y, "y"), [10.0, 60.0]);
}

/// An `independent_y` layer's `y` belongs to its OWN slot and must not become the
/// primary binding.
///
/// `numeric_domain_union` already refuses to let such a layer widen the
/// primary/shared y domain ("that would couple the left axis to a scale it does
/// not own"); supplying the primary binding outright is a stronger coupling than
/// the one that carve-out forbids. Measured without the skip, `LayerChart(rug,
/// curve, resolve={"y": "independent"})` rendered a left y axis duplicating its
/// own secondary axis. With the skip, nothing else binds `y`, so the primary y
/// keeps the synthesized unit scale — the layer's real domain lives in its slot.
#[test]
fn an_independent_y_layer_does_not_supply_the_primary_y_binding() {
    use crate::spec::mark::Mark;

    let secondary = Layer {
        independent_y: true, coord: None,
        ..layer_with(
            Mark::Line,
            crate::spec::encoding::Encoding {
                y: single_channel_encoding("y").y,
                ..single_channel_encoding("x")
            },
        )
    };
    let spec = layered_x_only_spec(
        Mark::Tick,
        vec![layer_with(Mark::Tick, single_channel_encoding("x")), secondary],
    );
    let batch = make_batch_q_q_n();
    let theme = ThemeInputs::default();

    let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme)
        .expect("a rug base with an independent-y layer must keep resolving");

    assert_eq!(resolved_domain(&scales.x, "x"), [1.0, 6.0]);
    assert_eq!(
        resolved_domain(&scales.y, "y"),
        [0.0, 1.0],
        "the independent layer's y must stay in its own slot, not wear the primary axis"
    );
}

/// The same layer with `independent_y` cleared DOES supply the binding — the
/// mutation guard proving the skip above is the flag and not the shape.
#[test]
fn a_shared_y_layer_in_the_same_shape_does_supply_the_primary_y_binding() {
    use crate::spec::mark::Mark;

    let shared = layer_with(
        Mark::Line,
        crate::spec::encoding::Encoding {
            y: single_channel_encoding("y").y,
            ..single_channel_encoding("x")
        },
    );
    let spec = layered_x_only_spec(
        Mark::Tick,
        vec![layer_with(Mark::Tick, single_channel_encoding("x")), shared],
    );
    let batch = make_batch_q_q_n();
    let theme = ThemeInputs::default();

    let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme)
        .expect("a shared-y line layer must supply the primary y binding");

    assert_eq!(resolved_domain(&scales.y, "y"), [10.0, 60.0]);
}

/// Under `CoordFlip` the fallback reads the layer's encoding in the channel's own
/// space, not by channel NAME.
///
/// `spec.layers` is the spec as AUTHORED while the resolved channels are
/// post-flip, so a layer's authored `y` is what occupies the x axis. The layer
/// here authors `y` (the `[10 … 60]` column) and nothing else; under `CoordFlip`
/// that binding must land on **x**. Reading the like-named channel would put it
/// on y and leave x unbound — the wrong-coordinate-space class
/// `numeric_domain_union`'s "ruling 1" comment was written about, one step worse
/// here because a whole axis BINDING is chosen rather than a domain contributor.
#[test]
fn coord_flip_moves_a_layer_binding_into_the_resolved_channels_space() {
    use crate::spec::mark::Mark;

    // Post-flip the chart encoding binds y (to the `[10 … 60]` column). The sole
    // layer binds AUTHORED y, which post-flip occupies x — so x must resolve to
    // that column's real extent. Read by channel NAME instead, the layer's `y`
    // would collide with the already-bound y and leave x unbound, which is the
    // y-only single-axis arm and a fabricated `[0, 1]` x scale. The two outcomes
    // are `[10, 60]` vs `[0, 1]`, so this discriminates the normalization.
    let mut spec = layered_x_only_spec(
        Mark::Tick,
        vec![layer_with(Mark::Tick, single_channel_encoding("y"))],
    );
    spec.encoding = single_channel_encoding("y");
    spec.coord = Some(CoordKind::Flip);
    let batch = make_batch_q_q_n();
    let theme = ThemeInputs::default();

    let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme)
        .expect("the flipped layer binding must resolve on the axis it actually occupies");

    assert_eq!(
        resolved_domain(&scales.x, "x"),
        [10.0, 60.0],
        "the authored-y layer binding occupies the x axis under CoordFlip, so x must \
         resolve that column, not the fabricated unit scale"
    );
    assert_eq!(resolved_domain(&scales.y, "y"), [10.0, 60.0]);
}

// --- Wave 4 T6 cycle 3: a drawing layer must be able to reach both channels ---

/// A layer whose mark DRAWS and needs both axes, with `y` reachable only through
/// [`positional_binding`]'s fallback, is refused rather than silently drawn empty.
///
/// `mark_point(x=) + mark_line(x=,y=)`: the point is layer 0, so the merged chart
/// encoding carries `x` alone; the fallback resolves the y AXIS from the line
/// layer, but no `y` FIELD ever reaches the point, whose builder then reads a
/// column that is not there and returns an empty batch. Measured pre-guard
/// through the public API: renders with `<circle>` count 0 and no warning.
#[test]
fn drawing_layer_that_cannot_reach_a_fallback_sourced_channel_is_refused() {
    use crate::spec::encoding::Encoding;
    use crate::spec::mark::Mark;

    let line_encoding = Encoding {
        y: single_channel_encoding("y").y,
        ..single_channel_encoding("x")
    };
    let spec = layered_x_only_spec(
        Mark::Point,
        vec![
            layer_with(Mark::Point, single_channel_encoding("x")),
            layer_with(Mark::Line, line_encoding),
        ],
    );
    let batch = make_batch_q_q_n();
    let theme = ThemeInputs::default();

    let err = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme).expect_err(
        "a point layer with no reachable y must refuse, not render zero marks on the \
         line layer's axis",
    );
    match err {
        RenderError::LayerMissingPositionalChannel { layer, mark, channel, .. } => {
            assert_eq!((layer, mark, channel), (0, "point", "y"));
        }
        other => panic!("expected LayerMissingPositionalChannel, got {other:?}"),
    }
}

/// The SAME pair in the other order still renders: the working case this rule
/// must not break.
///
/// With the line as layer 0 its `y` is merged INTO the chart encoding, so the
/// point inherits a real column and has something to plot. That inheritance —
/// from the chart encoding, never from the fallback — is the whole discriminator
/// between the two orderings, so this test and the one above cannot both pass
/// under a rule that counts the fallback as "available".
#[test]
fn the_same_pair_in_the_inheriting_order_still_resolves() {
    use crate::spec::encoding::Encoding;
    use crate::spec::mark::Mark;

    let line_encoding = Encoding {
        y: single_channel_encoding("y").y,
        ..single_channel_encoding("x")
    };
    let mut spec = layered_x_only_spec(
        Mark::Line,
        vec![
            layer_with(Mark::Line, line_encoding.clone()),
            layer_with(Mark::Point, single_channel_encoding("x")),
        ],
    );
    // Layer 0's merge: the chart encoding the resolver sees carries the line's y.
    spec.encoding = line_encoding;
    let batch = make_batch_q_q_n();
    let theme = ThemeInputs::default();

    let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme)
        .expect("the point inherits the chart encoding's y, so this must keep resolving");

    assert_eq!(resolved_domain(&scales.x, "x"), [1.0, 6.0]);
    assert_eq!(resolved_domain(&scales.y, "y"), [10.0, 60.0]);
}

/// The x-direction twin of the refusal — the mechanism is symmetric, so its
/// guard must be too.
#[test]
fn drawing_layer_that_cannot_reach_a_fallback_sourced_x_is_refused() {
    use crate::spec::encoding::Encoding;
    use crate::spec::mark::Mark;

    let line_encoding = Encoding {
        y: single_channel_encoding("y").y,
        ..single_channel_encoding("x")
    };
    let mut spec = layered_x_only_spec(
        Mark::Point,
        vec![
            layer_with(Mark::Point, single_channel_encoding("y")),
            layer_with(Mark::Line, line_encoding),
        ],
    );
    spec.encoding = single_channel_encoding("y");
    let batch = make_batch_q_q_n();
    let theme = ThemeInputs::default();

    let err = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme)
        .expect_err("a point layer with no reachable x must refuse");
    match err {
        RenderError::LayerMissingPositionalChannel { layer, mark, channel, .. } => {
            assert_eq!((layer, mark, channel), (0, "point", "x"));
        }
        other => panic!("expected LayerMissingPositionalChannel, got {other:?}"),
    }
}

/// A single-axis-eligible layer in the same position is NOT refused: one channel
/// is enough for a rug tick, so the axis it cannot reach is one its own builder
/// is written to do without. The mutation guard for the eligibility arm of the
/// per-layer rule — swap `Tick` for `Point` here and it becomes the refusal above.
#[test]
fn an_eligible_layer_beside_a_fallback_sourced_channel_still_resolves() {
    use crate::spec::encoding::Encoding;
    use crate::spec::mark::Mark;

    let line_encoding = Encoding {
        y: single_channel_encoding("y").y,
        ..single_channel_encoding("x")
    };
    let spec = layered_x_only_spec(
        Mark::Tick,
        vec![
            layer_with(Mark::Tick, single_channel_encoding("x")),
            layer_with(Mark::Line, line_encoding),
        ],
    );
    let batch = make_batch_q_q_n();
    let theme = ThemeInputs::default();

    let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme)
        .expect("a tick needs only one channel, so the fallback-sourced y must not refuse it");

    assert_eq!(resolved_domain(&scales.y, "y"), [10.0, 60.0]);
}

/// An INVISIBLE layer never triggers the refusal, however incomplete its
/// encoding: it paints nothing, so it has nothing to fail to plot. The same
/// `layer_paints` discriminator the eligibility vote uses, consumed by the
/// per-layer rule.
#[test]
fn an_invisible_layer_is_never_refused_for_a_missing_channel() {
    use crate::spec::encoding::Encoding;
    use crate::spec::mark::Mark;

    let line_encoding = Encoding {
        y: single_channel_encoding("y").y,
        ..single_channel_encoding("x")
    };
    let spec = layered_x_only_spec(
        Mark::Point,
        vec![
            invisible_carrier_layer(Mark::Point, single_channel_encoding("x")),
            layer_with(Mark::Line, line_encoding),
        ],
    );
    let batch = make_batch_q_q_n();
    let theme = ThemeInputs::default();

    let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme)
        .expect("an invisible layer paints nothing, so it cannot be refused for a missing y");

    assert_eq!(resolved_domain(&scales.y, "y"), [10.0, 60.0]);
}

/// "Nothing binds this channel" keeps its own channel-level error. The per-layer
/// rule runs after the generic resolution, so it can only ever fire on the
/// narrower case the fallback created — it never takes over a refusal that
/// already had a correct diagnosis.
#[test]
fn a_channel_bound_nowhere_still_raises_the_channel_level_error() {
    use crate::spec::mark::Mark;

    let spec = layered_x_only_spec(
        Mark::Point,
        vec![
            layer_with(Mark::Point, single_channel_encoding("x")),
            layer_with(Mark::Line, single_channel_encoding("x")),
        ],
    );
    let batch = make_batch_q_q_n();
    let theme = ThemeInputs::default();

    let err = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0), &theme)
        .expect_err("no y anywhere must stay the channel-level refusal");
    match err {
        RenderError::EncodingTypeMismatch { channel, .. } => assert_eq!(channel, "y"),
        other => panic!("expected EncodingTypeMismatch{{channel: \"y\", ..}}, got {other:?}"),
    }
}

// --- A chart-level channel reaches a layer only on the chart's data, from the authored encoding ---

/// An encoding binding both `make_batch_q_q_n` positional columns.
fn both_positional_channels() -> crate::spec::encoding::Encoding {
    crate::spec::encoding::Encoding { y: single_channel_encoding("y").y, ..single_channel_encoding("x") }
}

/// The positional channel other than `channel`.
fn other_channel(channel: &str) -> &'static str {
    if channel == "x" { "y" } else { "x" }
}

/// A line binding both channels as layer 0, then `point` as layer 1, with the
/// line's merge as the rendering encoding.
fn line_then(point: Layer) -> ChartSpec {
    use crate::spec::mark::Mark;
    let mut spec = layered_x_only_spec(Mark::Line, vec![layer_with(Mark::Line, both_positional_channels()), point]);
    spec.encoding = both_positional_channels();
    spec
}

/// Resolves `spec` with `chart_encoding` as its authored chart encoding and the
/// fixture batch registered as the named output `"other"`.
fn resolve_against(
    spec: &ChartSpec,
    chart_encoding: &crate::spec::encoding::Encoding,
) -> Result<(), RenderError> {
    let batch = make_batch_q_q_n();
    let outputs = HashMap::from([("other".to_string(), batch.clone())]);
    resolve_scales_with_leaf_context(
        spec,
        chart_encoding,
        &batch,
        &outputs, &crate::transform::run::OutputLineages::new(),
        (0.0, 100.0),
        (0.0, 80.0),
        &ThemeInputs::default(),
        None,
    )
    .map(|_| ())
}

fn assert_point_layer_refused(result: Result<(), RenderError>, channel: &str) {
    match result {
        Err(RenderError::LayerMissingPositionalChannel { layer, mark, channel: refused, .. }) => {
            assert_eq!((layer, mark, refused), (1, "point", channel));
        }
        other => panic!("expected layer 1 refused for a missing {channel}, got {other:?}"),
    }
}

/// A layer on its own data (`data_source`) does not inherit the chart's
/// positional channels (prepare gives it only the non-positional ones), so a
/// chart-level binding it does not carry itself leaves it nothing to plot on
/// that axis: it is refused rather than drawn empty. A value Python filled onto
/// the layer's own encoding (`inherited_positional`) is the layer's to draw, so
/// the same layer carrying it resolves.
#[test]
fn a_data_source_layer_is_refused_a_chart_level_channel_it_does_not_carry() {
    use crate::spec::layer::PositionalChannel;
    use crate::spec::mark::Mark;

    for (channel, filled) in [("x", PositionalChannel::X), ("y", PositionalChannel::Y)] {
        let routed = |encoding, inherited_positional| Layer {
            data_source: Some("other".into()),
            inherited_positional,
            ..layer_with(Mark::Point, encoding)
        };
        let spec = line_then(routed(single_channel_encoding(other_channel(channel)), Vec::new()));
        assert_point_layer_refused(resolve_against(&spec, &spec.encoding), channel);

        let spec = line_then(routed(both_positional_channels(), vec![filled]));
        resolve_against(&spec, &spec.encoding)
            .unwrap_or_else(|e| panic!("a filled {channel} is the layer's own binding: {e:?}"));
    }
}

/// A layer on the chart's data inherits the AUTHORED chart encoding, not the
/// rendering encoding the resolver's `spec` carries (layer 0's merge). A channel
/// only the rendering encoding binds came from layer 0's own encoding and never
/// reaches the later layer, which is refused; the same layer resolves when the
/// authored chart encoding binds the channel.
#[test]
fn a_chart_data_layer_reaches_only_the_authored_chart_channels() {
    use crate::spec::mark::Mark;

    for channel in ["x", "y"] {
        let spec = line_then(layer_with(Mark::Point, crate::spec::encoding::Encoding::default()));
        let authored = single_channel_encoding(other_channel(channel));
        assert_point_layer_refused(resolve_against(&spec, &authored), channel);

        resolve_against(&spec, &both_positional_channels())
            .unwrap_or_else(|e| panic!("an authored {channel} reaches a chart-data layer: {e:?}"));
    }
}

/// [`AUXILIARY_SCALE_CHANNELS`] names exactly the scales
/// [`build_auxiliary_scales`] builds, in field order: binding each name alone,
/// through the same wire key Python writes, builds that slot and no other.
/// The exhaustive destructure stops compiling when a slot is added without a
/// name.
#[test]
fn auxiliary_scale_channels_name_each_auxiliary_scale_slot_in_order() {
    let batch = make_batch_q_q_n();
    for (i, name) in AUXILIARY_SCALE_CHANNELS.iter().map(|channel| channel.name).enumerate() {
        let field = if matches!(name, "shape" | "stroke_dash") { "species" } else { "x" };
        let encoding: crate::spec::encoding::Encoding =
            serde_json::from_value(serde_json::json!({ name: { "field": field } })).unwrap();
        let spec = ChartSpec { encoding, ..Default::default() };
        let AuxiliaryScales { color, size, shape, opacity, fill_opacity, stroke_opacity, stroke_dash, stroke_width, raster_cells: _ } =
            build_auxiliary_scales(&spec, &spec.encoding, &batch, &HashMap::new(), &crate::transform::run::OutputLineages::new(), &ThemeInputs::default(), None, &mut Vec::new())
                .unwrap();
        let built = [
            color.is_some(),
            size.is_some(),
            shape.is_some(),
            opacity.is_some(),
            fill_opacity.is_some(),
            stroke_opacity.is_some(),
            stroke_dash.is_some(),
            stroke_width.is_some(),
        ];
        let expected: Vec<bool> = (0..built.len()).map(|j| j == i).collect();
        assert_eq!(built.to_vec(), expected, "{name}");
    }
    assert_eq!(auxiliary_scale_channels(), AUXILIARY_SCALE_CHANNELS.map(|channel| channel.name.to_string()));
}

/// With no scheme named, a `Diverging` scale takes the theme's diverging
/// scheme even on a domain that does not straddle zero; a named scheme wins;
/// any other scale keeps choosing by the data's sign.
#[test]
fn diverging_scale_without_a_scheme_takes_the_theme_diverging_scheme() {
    use crate::render::color::{ContinuousScheme, NamedContinuous};
    let positive = batch_with_numeric_color(vec![0.1, 0.5, 0.9]);
    let straddling = batch_with_numeric_color(vec![-0.4, 0.2, 0.9]);
    let theme = ThemeInputs::default();
    let named = |s: &str| ContinuousScheme::Named(NamedContinuous::from_name(s).unwrap());
    let scheme_of = |scale_json: serde_json::Value, batch: &RecordBatch| match resolve_color(
        &spec_with_color_scale_json(scale_json),
        batch,
    )
    .0
    {
        ColorScale::Continuous { scheme, .. } => scheme,
        other => panic!("expected a continuous scale, got {other:?}"),
    };
    let diverging = named(&theme.palette.diverging_scheme);
    let sequential = named(&theme.palette.sequential_scheme);

    assert_eq!(scheme_of(serde_json::json!({"type": "diverging", "domainMid": 0.5}), &positive), diverging);
    assert_eq!(scheme_of(serde_json::json!({"type": "diverging"}), &positive), diverging);
    assert_eq!(
        scheme_of(serde_json::json!({"type": "diverging", "domainMid": 0.5, "scheme": "viridis"}), &positive),
        named("viridis"),
    );
    assert_eq!(scheme_of(serde_json::json!({"type": "linear"}), &positive), sequential);
    assert_eq!(scheme_of(serde_json::json!({"type": "linear"}), &straddling), diverging);
}

// ── Value baseline and the bar zero anchor's value channel ─────────────────

/// `value_baseline_px` answers the 0 pixel whenever 0 is in the domain, and
/// otherwise the plot edge on zero's side, returned exactly as passed.
#[test]
fn value_baseline_px_is_the_zero_pixel_or_the_edge_on_zeros_side() {
    use crate::scale::linear::LinearScale;
    use crate::scale::log::LogScale;
    let linear = |domain: [f64; 2], clamp: bool| {
        ScaleKind::Linear(LinearScale::new_internal(domain.to_vec(), vec![95.0, 5.0], clamp, false))
    };
    // low edge 100 (the panel bottom), high edge 0 (the panel top).
    let baseline = |scale: &ScaleKind| scale.value_baseline_px(100.0, 0.0);

    assert!((baseline(&linear([-2.0, 3.0], false)) - 59.0).abs() < 1e-9, "mixed: the 0 tick");
    assert_eq!(baseline(&linear([0.0, 3.0], false)), 95.0, "all positive: the 0 tick above the inset");
    assert_eq!(baseline(&linear([-3.0, 0.0], false)), 5.0, "all negative: the 0 tick at the top");
    assert_eq!(baseline(&linear([3.0, 0.0], false)), 5.0, "reversed: the 0 tick at the top");
    assert_eq!(baseline(&linear([50.0, 70.0], false)), 100.0, "above 0: the low edge");
    assert_eq!(baseline(&linear([50.0, 70.0], true)), 100.0, "a clamp scale's floor is not the edge");
    assert_eq!(baseline(&linear([-70.0, -50.0], false)), 0.0, "below 0: the high edge");
    assert_eq!(baseline(&linear([70.0, 50.0], false)), 0.0, "reversed above 0: 50 sits at the top");
    assert_eq!(
        baseline(&ScaleKind::Log(LogScale::new_internal(vec![1.0, 100.0], vec![95.0, 5.0], 10.0, false, false))),
        100.0,
        "log: 0 is never in the domain",
    );
    let ordinal = ScaleKind::Ordinal(crate::scale::ordinal::OrdinalScale::new_internal(
        vec!["a".to_string()], vec![0.0, 100.0], DiscreteLayout::UNPADDED,
    ));
    assert_eq!(baseline(&ordinal), 100.0, "no numeric domain: the low edge");
}

/// A layered bar + point spec whose `y` is categorical, read as horizontal
/// bars and resolved over the x axis: `bar` and `point` both draw `x` (1..6)
/// against the `species` categories.
fn layered_bar_point_spec(y_type: crate::spec::encoding::DataType, y_field: &str) -> ChartSpec {
    use crate::spec::encoding::{DataType, Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    let encoding = || Encoding {
        x: Some(EncodingSpec { field: "x".into(), type_: Some(DataType::Quantitative), ..Default::default() }),
        y: Some(EncodingSpec { field: y_field.into(), type_: Some(y_type), ..Default::default() }),
        ..Default::default()
    };
    ChartSpec {
        mark: Mark::Bar,
        encoding: encoding(),
        layers: Some(vec![layer_with(Mark::Bar, encoding()), layer_with(Mark::Point, encoding())]),
        ..Default::default()
    }
}

/// A layered native horizontal bar (categorical `y`) carries its value on
/// `x`, so the layered zero-anchor rule widens x to 0. Unanchored, x spans
/// the data's 1..6 and the shortest bar is a stub.
#[test]
fn a_layered_native_horizontal_bar_anchors_x_at_zero() {
    use crate::spec::encoding::DataType;
    let spec = layered_bar_point_spec(DataType::Nominal, "species");
    let (scales, _) = resolve_scales(&spec, &make_batch_q_q_n(), (0.0, 100.0), (0.0, 80.0), &ThemeInputs::default())
        .expect("a layered horizontal bar resolves");
    assert_eq!(resolved_domain(&scales.x, "x")[0], 0.0, "the value axis x reaches 0");
}

/// The control: a vertical bar layer (quantitative `y`) draws its value on y,
/// so x is its position channel and is never anchored.
#[test]
fn a_layered_vertical_bar_leaves_its_position_axis_unanchored() {
    use crate::spec::encoding::DataType;
    let spec = layered_bar_point_spec(DataType::Quantitative, "y");
    let (scales, _) = resolve_scales(&spec, &make_batch_q_q_n(), (0.0, 100.0), (0.0, 80.0), &ThemeInputs::default())
        .expect("a layered vertical bar resolves");
    assert_eq!(resolved_domain(&scales.x, "x")[0], 1.0, "the position axis x keeps the data's 1");
    assert_eq!(resolved_domain(&scales.y, "y")[0], 0.0, "the value axis y reaches 0");
}

/// The tick labels of the secondary y axis a line + `independent_y` layer
/// resolves, with the secondary layer drawn as `mark` over values
/// 1.85M..2.4M and its `y` carrying `scale`.
fn secondary_axis_labels(
    mark: crate::spec::mark::Mark,
    scale: Option<crate::spec::encoding::ScaleSpec>,
) -> Vec<String> {
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    let field = |name: &str| Some(EncodingSpec { field: name.into(), ..Default::default() });
    let primary = Layer {
        mark: Mark::Line,
        encoding: Encoding { y: field("y0"), ..Default::default() },
        ..Default::default()
    };
    let secondary = Layer {
        mark,
        encoding: Encoding { y: Some(EncodingSpec { field: "y1".into(), scale, ..Default::default() }), ..Default::default() },
        independent_y: true,
        ..Default::default()
    };
    let spec = ChartSpec {
        mark: Mark::Line,
        encoding: Encoding { x: field("x"), ..Default::default() },
        layers: Some(vec![primary, secondary]),
        ..Default::default()
    };
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ArrowDataType::Float64, false),
        Field::new("y0", ArrowDataType::Float64, false),
        Field::new("y1", ArrowDataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![3.0, 2.0, 1.0])),
            Arc::new(Float64Array::from(vec![1.85e6, 2.0e6, 2.4e6])),
        ],
    )
    .unwrap();
    let prep = crate::render::prepare::prepare_render_inputs(
        &spec,
        &batch,
        &ThemeInputs::default(),
        crate::layout::Viewport { width: 600.0, height: 400.0 },
        &crate::render::chart_config::ChartConfig::default(),
        None,
    )
    .expect("a dual-axis chart prepares");
    prep.axes.secondary_y[0].tick_labels.clone()
}

/// A `SecondaryY` bar slot resolves the `y` it is given: the ferrum-authored
/// zero-anchor scale Python serializes onto an independent-y bar reaches 0,
/// exactly as a standalone bar does; with no scale the slot spans its data
/// (the anchor has one producer, the Python injection); a line slot over the
/// same values keeps its data range.
#[test]
fn a_secondary_y_bar_slot_reaches_zero_and_a_line_slot_does_not() {
    use crate::spec::mark::Mark;
    let bar = secondary_axis_labels(Mark::Bar, Some(injected_zero_scale()));
    assert!(bar.iter().any(|l| l == "0"), "the anchored bar slot's axis reaches 0; got {bar:?}");
    let bare = secondary_axis_labels(Mark::Bar, None);
    assert!(!bare.iter().any(|l| l == "0"), "the slot adds no anchor of its own; got {bare:?}");
    let line = secondary_axis_labels(Mark::Line, None);
    assert!(!line.iter().any(|l| l == "0"), "a line slot keeps its data range; got {line:?}");
}

// ── A stacked value axis covers every segment's drawn extent ────────────────

/// A two-bin, two-group stack over `x = [1, 1, 2, 2]`, `g = [p, q, p, q]`.
fn stacked_batch(ys: [f64; 4], with_y2: bool) -> RecordBatch {
    let mut fields = vec![
        Field::new("x", ArrowDataType::Float64, false),
        Field::new("y", ArrowDataType::Float64, false),
        Field::new("g", ArrowDataType::Utf8, false),
    ];
    let mut columns: Vec<arrow::array::ArrayRef> = vec![
        Arc::new(Float64Array::from(vec![1.0, 1.0, 2.0, 2.0])),
        Arc::new(Float64Array::from(ys.to_vec())),
        Arc::new(StringArray::from(vec!["p", "q", "p", "q"])),
    ];
    if with_y2 {
        fields.push(Field::new("lo", ArrowDataType::Float64, false));
        columns.push(Arc::new(Float64Array::from(vec![0.25; 4])));
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
}

/// The ferrum-authored zero-anchor scale a standalone bar's value channel
/// carries.
fn injected_zero_scale() -> crate::spec::encoding::ScaleSpec {
    use crate::spec::encoding::{ContinuousScaleCommon, ScaleSpec};
    ScaleSpec::Linear {
        common: ContinuousScaleCommon {
            domain: None, range: None, clamp: false, padding: None, scheme: None, domain_param: None, reverse: false,
        },
        nice: false,
        zero: true,
        system_injected: true,
    }
}

/// A `mark` spec over [`stacked_batch`]: the value on `value` (`"y"`, or
/// `"x"` for a horizontal stack), the category on the other channel, a bar's
/// zero-anchor scale when `zero_anchor`, a `y2` span when `with_y2`, and a
/// `Stack` when `offset` is `Some`.
fn stack_spec(
    mark: crate::spec::mark::Mark,
    offset: Option<crate::spec::position::StackOffset>,
    value_on_x: bool,
    zero_anchor: bool,
    with_y2: bool,
) -> ChartSpec {
    use crate::spec::encoding::{DataType, Encoding, EncodingSpec};
    use crate::spec::position::{PositionAdjust, StackAnchor, StackValueAxis};
    let quant = |field: &str| EncodingSpec { field: field.into(), type_: Some(DataType::Quantitative), ..Default::default() };
    let value = EncodingSpec { scale: zero_anchor.then(injected_zero_scale), ..quant("y") };
    let (x, y) = if value_on_x { (value, quant("x")) } else { (quant("x"), value) };
    ChartSpec {
        mark,
        encoding: Encoding {
            x: Some(x),
            y: Some(y),
            y2: with_y2.then(|| quant("lo")),
            color: Some(EncodingSpec { field: "g".into(), type_: Some(DataType::Nominal), ..Default::default() }),
            ..Default::default()
        },
        position: offset.map(|offset| PositionAdjust::Stack {
            by: Some("g".into()),
            offset,
            anchor: StackAnchor::Top,
            value_axis: value_on_x.then_some(StackValueAxis::X),
        }),
        ..Default::default()
    }
}

/// The value-axis domain a stacked spec resolves, and the one the same spec
/// resolves from its stacked tops alone (the tops fed in as plain data, with
/// no stack): the domain the value axis had before the base joined it.
fn stacked_and_tops_only_domains(
    mark: crate::spec::mark::Mark,
    offset: crate::spec::position::StackOffset,
    ys: [f64; 4],
    value_on_x: bool,
    zero_anchor: bool,
    with_y2: bool,
) -> ((f64, f64), (f64, f64)) {
    let theme = ThemeInputs::default();
    let batch = stacked_batch(ys, with_y2);
    let value_scale = |scales: &ResolvedScales| {
        if value_on_x { scales.x.data_domain() } else { scales.y.data_domain() }.expect("a continuous value axis")
    };
    let stacked = stack_spec(mark, Some(offset), value_on_x, zero_anchor, with_y2);
    let (scales, _) = resolve_scales(&stacked, &batch, (0.0, 100.0), (0.0, 100.0), &theme).expect("the stack resolves");
    let stacked_domain = value_scale(&scales);

    let stacked_rows = crate::render::position::apply_position(
        &batch, stacked.position.as_ref(), &scales, &stacked.encoding, false, &mut Vec::new(),
    )
    .expect("the stack applies");
    let tops_only = stack_spec(mark, None, value_on_x, zero_anchor, with_y2);
    let (scales, _) = resolve_scales(&tops_only, &stacked_rows, (0.0, 100.0), (0.0, 100.0), &theme).expect("the tops resolve");
    (stacked_domain, value_scale(&scales))
}

/// An all-positive `center` stack's runs start at a nonzero shift, its lowest
/// segment's base (-3.5 here: bin 2's [3, 4] centered on 0) below every top,
/// so the domain covers that base, for bars and areas alike. Read from the
/// tops alone it stops at -0.5 and the segment is truncated.
#[test]
fn a_center_stack_domain_covers_its_lowest_base() {
    use crate::spec::mark::Mark;
    use crate::spec::position::StackOffset;
    for (mark, zero_anchor) in [(Mark::Bar, true), (Mark::Area, false)] {
        let (stacked, tops_only) =
            stacked_and_tops_only_domains(mark, StackOffset::Center, [2.0, 1.0, 3.0, 4.0], false, zero_anchor, false);
        assert!(stacked.0 <= -3.5, "{mark:?}: the lowest base -3.5 is in the domain; got {stacked:?}");
        assert!(tops_only.0 > -3.5, "{mark:?}: premise, the tops alone stop short of it; got {tops_only:?}");
    }
}

/// An all-negative `center` stack's runs start above every top (bin 2's run
/// starts at 3.5), and the domain covers that base.
#[test]
fn an_all_negative_center_stack_domain_covers_its_top_base() {
    use crate::spec::mark::Mark;
    use crate::spec::position::StackOffset;
    let (stacked, _) =
        stacked_and_tops_only_domains(Mark::Bar, StackOffset::Center, [-2.0, -1.0, -3.0, -4.0], false, true, false);
    assert!(stacked.1 >= 3.5, "the top base 3.5 is in the domain; got {stacked:?}");
}

/// A horizontal (value on x) `center` stack takes the same base span on x.
#[test]
fn a_horizontal_center_stack_domain_covers_its_lowest_base_on_x() {
    use crate::spec::mark::Mark;
    use crate::spec::position::StackOffset;
    let (stacked, _) =
        stacked_and_tops_only_domains(Mark::Bar, StackOffset::Center, [2.0, 1.0, 3.0, 4.0], true, true, false);
    assert!(stacked.0 <= -3.5, "the lowest base -3.5 is in the x domain; got {stacked:?}");
}

/// Every stack whose bases are 0 or another segment's top resolves exactly
/// the domain its tops alone give: zero and normalize stacks of either sign,
/// bars (with the zero anchor) and areas (without it, so an all-positive
/// area's domain still excludes its 0 origin), and a mixed-sign center stack
/// whose shared shift lies between its runs' tops.
#[test]
fn a_stack_whose_bases_are_origins_or_tops_keeps_its_tops_only_domain() {
    use crate::spec::mark::Mark;
    use crate::spec::position::StackOffset;
    let cases = [
        (StackOffset::Zero, [2.0, 1.0, 3.0, 4.0]),
        (StackOffset::Normalize, [2.0, 1.0, 3.0, 4.0]),
        (StackOffset::Zero, [-2.0, -1.0, -3.0, -4.0]),
        (StackOffset::Normalize, [-2.0, -1.0, -3.0, -4.0]),
        (StackOffset::Zero, [2.0, -1.0, -2.0, 3.0]),
        (StackOffset::Normalize, [2.0, -1.0, -2.0, 3.0]),
        (StackOffset::Center, [2.0, -1.0, -2.0, 3.0]),
    ];
    for (offset, ys) in cases {
        for (mark, zero_anchor) in [(Mark::Bar, true), (Mark::Area, false)] {
            let (stacked, tops_only) = stacked_and_tops_only_domains(mark, offset, ys, false, zero_anchor, false);
            assert_eq!(stacked, tops_only, "{mark:?} {offset:?} {ys:?}");
        }
    }
    let (area, _) = stacked_and_tops_only_domains(Mark::Area, StackOffset::Zero, [2.0, 1.0, 3.0, 4.0], false, false, false);
    assert!(area.0 > 0.0, "an all-positive stacked area's 0 origin stays out of its domain; got {area:?}");
}

/// A bound `y2` keeps priority over the stack base as the value axis's span:
/// the center stack's -3.5 base does not join the domain, the `y2` column
/// (0.25) does.
#[test]
fn a_bound_y2_keeps_priority_over_the_stack_base() {
    use crate::spec::mark::Mark;
    use crate::spec::position::StackOffset;
    let (stacked, tops_only) =
        stacked_and_tops_only_domains(Mark::Bar, StackOffset::Center, [2.0, 1.0, 3.0, 4.0], false, true, true);
    assert_eq!(stacked, tops_only, "the base is not read when y2 is bound");
    assert!(stacked.0 > -3.5, "got {stacked:?}");
}

/// A layered center stack (a stacked bar plus a point layer over the same
/// rows) carries the stack base through the layered seed read, so the shared
/// y domain covers the lowest base (-3.5) exactly as the unlayered stack's
/// does.
#[test]
fn a_layered_center_stack_domain_covers_its_lowest_base() {
    use crate::spec::mark::Mark;
    use crate::spec::position::StackOffset;
    let stacked = stack_spec(Mark::Bar, Some(StackOffset::Center), false, true, false);
    let bar = Layer {
        mark: Mark::Bar,
        encoding: stacked.encoding.clone(),
        position: stacked.position.clone(),
        ..Default::default()
    };
    let point = Layer { mark: Mark::Point, encoding: stacked.encoding.clone(), ..Default::default() };
    let spec = ChartSpec { layers: Some(vec![bar, point]), ..stacked };
    let (scales, _) = resolve_scales(&spec, &stacked_batch([2.0, 1.0, 3.0, 4.0], false), (0.0, 100.0), (0.0, 100.0), &ThemeInputs::default())
        .expect("the layered stack resolves");
    let (lo, _) = scales.y.data_domain().expect("a continuous y");
    assert!(lo <= -3.5, "the lowest base -3.5 is in the layered domain; got {lo}");
}

// ── chart-level non-positional domains on a layered spec ────────────────────

/// A chart over `g`/`m`/`z` whose two layers each draw their own computed
/// column from their own output: `agg0` (`mean_z` over 0..2, category `c`
/// in `p`, `q`) and `agg1` (`mean_z` over 10..20, `c` in `q`, `r`). The
/// chart-level `channel` binds the raw `z` (the promoted binding); each layer
/// binds `mean_z` itself, and the rendering encoding is layer 0's merge.
/// `layer_channel` is each layer's own binding for `channel`, or `None` for a
/// layer that inherits the chart's.
fn aux_layered_spec(
    channel: &str,
    layer_channel: [Option<serde_json::Value>; 2],
) -> (ChartSpec, crate::spec::encoding::Encoding) {
    let positional = serde_json::json!({"x": {"field": "g"}, "y": {"field": "m"}});
    let with = |base: &serde_json::Value, binding: Option<&serde_json::Value>| {
        let mut encoding = base.clone();
        if let Some(binding) = binding {
            encoding[channel] = binding.clone();
        }
        encoding
    };
    let chart = with(&positional, Some(&serde_json::json!({"field": "z"})));
    let layer = |source: &str, binding: Option<&serde_json::Value>| {
        serde_json::json!({"mark": "point", "data_source": source, "encoding": with(&positional, binding)})
    };
    let [own0, own1] = &layer_channel;
    let spec: ChartSpec = serde_json::from_value(serde_json::json!({
        "data": {"kind": "named", "name": "default"},
        "mark": "point",
        "encoding": with(&positional, own0.as_ref().or(Some(&serde_json::json!({"field": "z"})))),
        "layers": [layer("agg0", own0.as_ref()), layer("agg1", own1.as_ref())],
    }))
    .expect("the layered spec deserializes");
    let chart_encoding = serde_json::from_value(chart).expect("the chart encoding deserializes");
    (spec, chart_encoding)
}

/// The primary batch (`z` over 5..6, no `mean_z`) and the two layer outputs.
fn aux_layered_batches() -> (RecordBatch, HashMap<String, RecordBatch>) {
    let batch = |g: [&str; 2], m: [f64; 2], extra: (&str, [f64; 2]), c: [&str; 2]| {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("g", ArrowDataType::Utf8, false),
                Field::new("m", ArrowDataType::Float64, false),
                Field::new(extra.0, ArrowDataType::Float64, false),
                Field::new("c", ArrowDataType::Utf8, false),
            ])),
            vec![
                Arc::new(StringArray::from(g.to_vec())),
                Arc::new(Float64Array::from(m.to_vec())),
                Arc::new(Float64Array::from(extra.1.to_vec())),
                Arc::new(StringArray::from(c.to_vec())),
            ],
        )
        .unwrap()
    };
    let primary = batch(["a", "b"], [1.0, 2.0], ("z", [5.0, 6.0]), ["s", "s"]);
    let outputs = HashMap::from([
        ("agg0".to_string(), batch(["a", "b"], [1.0, 2.0], ("mean_z", [0.0, 2.0]), ["p", "q"])),
        ("agg1".to_string(), batch(["a", "b"], [1.0, 2.0], ("mean_z", [10.0, 20.0]), ["q", "r"])),
    ]);
    (primary, outputs)
}

fn resolve_aux_layered(spec: &ChartSpec, chart_encoding: &crate::spec::encoding::Encoding) -> ResolvedScales {
    let (primary, outputs) = aux_layered_batches();
    resolve_scales_with_leaf_context(
        spec,
        chart_encoding,
        &primary,
        &outputs, &crate::transform::run::OutputLineages::new(),
        (0.0, 100.0),
        (0.0, 80.0),
        &ThemeInputs::default(),
        None,
    )
    .expect("the layered spec resolves")
    .0
}

/// Size and the opacity family resolve their domain as the union of every
/// layer's own computed column, each read from the batch that layer draws,
/// not from the primary batch (which has no `mean_z`) or from one layer.
#[test]
fn a_layered_numeric_aux_domain_is_the_union_of_each_layers_own_column() {
    pyo3::Python::initialize();
    let own = serde_json::json!({"field": "mean_z"});
    for channel in ["size", "opacity", "fill_opacity", "stroke_opacity"] {
        let (spec, chart_encoding) = aux_layered_spec(channel, [Some(own.clone()), Some(own.clone())]);
        let scales = resolve_aux_layered(&spec, &chart_encoding);
        let inner = match channel {
            "size" => scales.size.map(|s| s.inner),
            "opacity" => scales.opacity.map(|s| s.inner),
            "fill_opacity" => scales.fill_opacity.map(|s| s.inner),
            _ => scales.stroke_opacity.map(|s| s.inner),
        };
        let domain = inner.unwrap_or_else(|| panic!("{channel} resolves a scale")).data_domain();
        assert_eq!(domain, Some((0.0, 20.0)), "{channel}: both layers' mean_z");
    }
}

/// A continuous color resolves the same union.
#[test]
fn a_layered_continuous_color_domain_is_the_union_of_each_layers_own_column() {
    pyo3::Python::initialize();
    let own = serde_json::json!({"field": "mean_z", "type": "quantitative"});
    let (spec, chart_encoding) = aux_layered_spec("color", [Some(own.clone()), Some(own)]);
    match resolve_aux_layered(&spec, &chart_encoding).color {
        Some(ColorScale::Continuous { domain, .. }) => assert_eq!(domain, (0.0, 20.0)),
        other => panic!("expected a continuous color scale, got {other:?}"),
    }
}

/// A categorical color unions the categories each layer draws, in layer order.
#[test]
fn a_layered_categorical_color_domain_unions_each_layers_categories_in_order() {
    pyo3::Python::initialize();
    let own = serde_json::json!({"field": "c", "type": "nominal"});
    let (spec, chart_encoding) = aux_layered_spec("color", [Some(own.clone()), Some(own)]);
    match resolve_aux_layered(&spec, &chart_encoding).color {
        Some(ColorScale::Categorical { domain, .. }) => assert_eq!(domain, vec!["p", "q", "r"]),
        other => panic!("expected a categorical color scale, got {other:?}"),
    }
}

/// A layer that binds nothing on the channel draws the chart-level field from
/// its own batch: layer 1 inherits `z`, which its output lacks, so it adds
/// nothing, while layer 0's own `mean_z` sets the domain. With the inherited
/// field present in that layer's batch it joins the union.
#[test]
fn an_inheriting_layer_reads_the_chart_field_from_its_own_batch() {
    pyo3::Python::initialize();
    let own = serde_json::json!({"field": "mean_z"});
    let (spec, chart_encoding) = aux_layered_spec("size", [Some(own), None]);
    let domain = resolve_aux_layered(&spec, &chart_encoding).size.unwrap().inner.data_domain();
    assert_eq!(domain, Some((0.0, 2.0)), "layer 1's batch has no z; layer 0 alone");

    let (primary, mut outputs) = aux_layered_batches();
    let agg1 = &outputs["agg1"];
    let renamed = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("g", ArrowDataType::Utf8, false),
            Field::new("m", ArrowDataType::Float64, false),
            Field::new("z", ArrowDataType::Float64, false),
            Field::new("c", ArrowDataType::Utf8, false),
        ])),
        agg1.columns().to_vec(),
    )
    .unwrap();
    outputs.insert("agg1".to_string(), renamed);
    let scales = resolve_scales_with_leaf_context(
        &spec,
        &chart_encoding,
        &primary,
        &outputs, &crate::transform::run::OutputLineages::new(),
        (0.0, 100.0),
        (0.0, 80.0),
        &ThemeInputs::default(),
        None,
    )
    .unwrap()
    .0;
    assert_eq!(scales.size.unwrap().inner.data_domain(), Some((0.0, 20.0)), "layer 1 draws the inherited z");
}

/// An explicit scale domain still wins over the union.
#[test]
fn an_explicit_aux_domain_still_wins_over_the_layer_union() {
    pyo3::Python::initialize();
    let own = serde_json::json!({"field": "mean_z", "scale": {"type": "linear", "domain": [0.0, 50.0]}});
    let (spec, chart_encoding) = aux_layered_spec("opacity", [Some(own.clone()), Some(own)]);
    let domain = resolve_aux_layered(&spec, &chart_encoding).opacity.unwrap().inner.data_domain();
    assert_eq!(domain, Some((0.0, 50.0)));
}

/// [`aux_layered_spec`] resolved with its warnings.
fn resolve_aux_layered_warned(
    spec: &ChartSpec,
    chart_encoding: &crate::spec::encoding::Encoding,
) -> (ResolvedScales, Vec<crate::render::RenderWarning>) {
    let (primary, outputs) = aux_layered_batches();
    resolve_scales_with_leaf_context(
        spec,
        chart_encoding,
        &primary,
        &outputs, &crate::transform::run::OutputLineages::new(),
        (0.0, 100.0),
        (0.0, 80.0),
        &ThemeInputs::default(),
        None,
    )
    .expect("the layered spec resolves")
}

/// The one exclusion warning for layer 1 on `channel`, established by layer 0.
fn excluded_layer_one(channel: &str) -> Vec<crate::render::RenderWarning> {
    vec![crate::render::RenderWarning::LayerScaleExcluded {
        layer: 1,
        establishing: 0,
        channel: channel.to_string(),
        placement: crate::render::ExcludedLayerPlacement::SharedAxis,
    }]
}

fn layer_scale_exclusions(warnings: Vec<crate::render::RenderWarning>) -> Vec<crate::render::RenderWarning> {
    warnings
        .into_iter()
        .filter(|w| matches!(w, crate::render::RenderWarning::LayerScaleExcluded { .. }))
        .collect()
}

/// A layer whose own color field reads differently from the color scale layer
/// 0 establishes stays out of its domain, with one `LayerScaleExcluded`, in
/// both orders: a string field beside a quantitative color, and a numeric field
/// beside a nominal one.
#[test]
fn a_layer_whose_color_type_conflicts_is_excluded_from_the_color_domain_in_both_orders() {
    pyo3::Python::initialize();
    let quantitative = serde_json::json!({"field": "mean_z", "type": "quantitative"});
    let nominal = serde_json::json!({"field": "c"});

    let (spec, chart_encoding) = aux_layered_spec("color", [Some(quantitative.clone()), Some(nominal.clone())]);
    let (scales, warnings) = resolve_aux_layered_warned(&spec, &chart_encoding);
    match scales.color {
        Some(ColorScale::Continuous { domain, .. }) => assert_eq!(domain, (0.0, 2.0), "layer 0's column alone"),
        other => panic!("expected a continuous color scale, got {other:?}"),
    }
    assert_eq!(layer_scale_exclusions(warnings), excluded_layer_one("color"));

    let (spec, chart_encoding) = aux_layered_spec("color", [Some(nominal), Some(quantitative)]);
    let (scales, warnings) = resolve_aux_layered_warned(&spec, &chart_encoding);
    match scales.color {
        Some(ColorScale::Categorical { domain, .. }) => assert_eq!(domain, vec!["p", "q"], "layer 0's categories alone"),
        other => panic!("expected a categorical color scale, got {other:?}"),
    }
    assert_eq!(layer_scale_exclusions(warnings), excluded_layer_one("color"));
}

/// Size and the opacity family are continuous: a later layer's string field
/// stays out of the domain layer 0's numeric field establishes, with one
/// `LayerScaleExcluded` per channel.
#[test]
fn a_later_layer_with_a_string_field_is_excluded_from_a_numeric_aux_domain() {
    pyo3::Python::initialize();
    let numeric = serde_json::json!({"field": "mean_z"});
    let string = serde_json::json!({"field": "c"});
    for channel in ["size", "opacity", "fill_opacity", "stroke_opacity"] {
        let (spec, chart_encoding) = aux_layered_spec(channel, [Some(numeric.clone()), Some(string.clone())]);
        let (scales, warnings) = resolve_aux_layered_warned(&spec, &chart_encoding);
        let inner = match channel {
            "size" => scales.size.map(|s| s.inner),
            "opacity" => scales.opacity.map(|s| s.inner),
            "fill_opacity" => scales.fill_opacity.map(|s| s.inner),
            _ => scales.stroke_opacity.map(|s| s.inner),
        };
        assert_eq!(inner.unwrap().data_domain(), Some((0.0, 2.0)), "{channel}: layer 0's column alone");
        assert_eq!(layer_scale_exclusions(warnings), excluded_layer_one(channel), "{channel}");
    }
}

/// Type-compatible layers keep the union and raise no exclusion.
#[test]
fn type_compatible_layers_keep_the_union_without_an_exclusion() {
    pyo3::Python::initialize();
    let own = serde_json::json!({"field": "mean_z"});
    let (spec, chart_encoding) = aux_layered_spec("size", [Some(own.clone()), Some(own)]);
    let (scales, warnings) = resolve_aux_layered_warned(&spec, &chart_encoding);
    assert_eq!(scales.size.unwrap().inner.data_domain(), Some((0.0, 20.0)));
    assert!(layer_scale_exclusions(warnings).is_empty());
}

/// The exclusion's message names a scale, not an axis, on a non-positional
/// channel; x/y keep the axis wording.
#[test]
fn layer_scale_excluded_says_scale_for_a_non_positional_channel() {
    let warning = |channel: &str| crate::render::RenderWarning::LayerScaleExcluded {
        layer: 1,
        establishing: 0,
        channel: channel.to_string(),
        placement: crate::render::ExcludedLayerPlacement::SharedAxis,
    };
    assert_eq!(
        warning("color").to_string(),
        "layer 1: its color scale differs from the one layer 0 establishes the color scale with, \
         so it stays out of that scale; its marks draw through that scale and may fall outside its legend"
    );
    assert_eq!(
        warning("x").to_string(),
        "layer 1: its x scale differs from the one layer 0 establishes the x axis with, so it \
         stays out of that axis; its marks are placed on that axis and may fall outside it"
    );
}

/// `ferrum._core.validate_numeric_aux_scale` runs the render-time gate on a
/// raw dict with no data: it admits `linear`, `log`, `pow` and `sqrt`,
/// refuses any other type with the text a render refuses with, refuses a key
/// the scale-dict gate refuses, and refuses a channel that is not a numeric
/// auxiliary one.
#[test]
fn the_published_gate_answers_as_the_render_does() {
    use pyo3::types::{PyDict, PyDictMethods};
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let dict = |pairs: &[(&str, pyo3::Py<pyo3::PyAny>)]| {
            let d = PyDict::new(py);
            for (k, v) in pairs {
                d.set_item(*k, v).unwrap();
            }
            d
        };
        use pyo3::IntoPyObjectExt;
        let s = |v: &str| v.into_py_any(py).unwrap();
        let linear = dict(&[("type", s("linear")), ("domain", vec![0.0, 1.0].into_py_any(py).unwrap())]);
        for channel in NUMERIC_AUX_CHANNELS {
            validate_numeric_aux_scale(py, channel, &linear).unwrap();
            for supported in ["log", "pow", "sqrt"] {
                let scale = dict(&[("type", s(supported))]);
                validate_numeric_aux_scale(py, channel, &scale).unwrap();
            }
            let ordinal = dict(&[("type", s("ordinal"))]);
            let err = validate_numeric_aux_scale(py, channel, &ordinal).unwrap_err();
            assert_eq!(
                err.value(py).to_string(),
                format!(
                    "{channel} scale: type 'ordinal' is not supported; \
                     pass fm.LinearScale/LogScale/PowScale/SqrtScale(domain=[...], range=[...]) or drop scale="
                )
            );
        }
        let typo = dict(&[("type", s("linear")), ("clammp", true.into_py_any(py).unwrap())]);
        assert!(validate_numeric_aux_scale(py, "size", &typo).unwrap_err().value(py).to_string().contains("clammp"));
        assert!(validate_numeric_aux_scale(py, "color", &linear).is_err(), "color is not a numeric auxiliary channel");
    });
}

/// `zero=true` widens a descending domain to zero and keeps it descending, on
/// every numeric auxiliary channel and on the x/y axes: `[20, 0]` stays
/// `[20, 0]`, `[20, 5]` becomes `[20, 0]`, `[-5, -20]` becomes `[0, -20]`,
/// and a `domainParam` substituting `[20, 5]` becomes `[20, 0]`. Rejects
/// widening as if the pair were ascending, which collapses `[20, 0]` to
/// `[0, 0]` and turns `[20, 5]` into an ascending `[0, 5]`.
#[test]
fn zero_widens_a_descending_domain_and_keeps_it_descending() {
    use ferrum_scene::{ParamKind, ParameterSpec};
    let cases: [(serde_json::Value, (f64, f64)); 4] = [
        (serde_json::json!({"type": "linear", "domain": [20.0, 0.0], "zero": true}), (20.0, 0.0)),
        (serde_json::json!({"type": "linear", "domain": [20.0, 5.0], "zero": true}), (20.0, 0.0)),
        (serde_json::json!({"type": "linear", "domain": [-5.0, -20.0], "zero": true}), (0.0, -20.0)),
        (serde_json::json!({"type": "linear", "domainParam": "d", "zero": true}), (20.0, 0.0)),
    ];
    let params = vec![ParameterSpec {
        name: "d".into(),
        kind: ParamKind::Variable,
        value: Some(serde_json::json!([20, 5])),
        bind: None,
        select: None,
    }];
    let batch = appearance_batch();
    for (scale, expected) in &cases {
        for channel in NUMERIC_AUX_CHANNELS.iter().copied().chain(["x", "y"]) {
            let scale_spec: crate::spec::encoding::ScaleSpec = serde_json::from_value(scale.clone()).unwrap();
            let mut spec = match channel {
                "x" | "y" => {
                    let mut spec = appearance_spec(serde_json::json!({}));
                    let binding = if channel == "x" { &mut spec.encoding.x } else { &mut spec.encoding.y };
                    binding.as_mut().unwrap().scale = Some(scale_spec);
                    spec
                }
                _ => appearance_spec(serde_json::json!({ channel: {"field": "v", "scale": scale} })),
            };
            spec.params = params.clone();
            resolve_param_domains(&mut spec);
            let (scales, _) = resolve_appearance(&spec, &batch);
            let domain = match channel {
                "x" => scales.x.data_domain(),
                "y" => scales.y.data_domain(),
                "size" => scales.size.map(|s| s.inner.data_domain().unwrap()),
                "opacity" => scales.opacity.map(|s| s.inner.data_domain().unwrap()),
                "fill_opacity" => scales.fill_opacity.map(|s| s.inner.data_domain().unwrap()),
                "stroke_opacity" => scales.stroke_opacity.map(|s| s.inner.data_domain().unwrap()),
                "stroke_width" => scales.stroke_width.map(|s| s.inner.data_domain().unwrap()),
                other => panic!("{other}"),
            };
            assert_eq!(domain, Some(*expected), "{channel} {scale}");
        }
    }
}

/// An x axis with `domain=[20, 0], zero=true` draws its points, at the
/// positions a plain `[20, 0]` gives them, instead of collapsing the axis.
#[test]
fn a_descending_zero_x_domain_draws_its_points() {
    let batch = appearance_batch();
    let render = |scale: serde_json::Value| {
        let mut spec = appearance_spec(serde_json::json!({}));
        spec.encoding.x.as_mut().unwrap().scale = Some(serde_json::from_value(scale).unwrap());
        render_appearance(&spec, &batch).bytes
    };
    let zeroed = render(serde_json::json!({"type": "linear", "domain": [20.0, 0.0], "zero": true}));
    let plain = render(serde_json::json!({"type": "linear", "domain": [20.0, 0.0]}));
    assert_eq!(zeroed.matches("<circle").count(), 3, "three points");
    assert_eq!(zeroed, plain);
}

/// With no chart-level `x`, the axis binding falls back to the first layer
/// whose own `x` its mark draws. Layer 0 is a one-axis rule writing `y`, with
/// an `x` Python filled and listed as inherited: the rule draws no `x`, so the
/// fallback skips it and takes the point layer's `g`. The rejected reading,
/// the raw authored encoding, returns the rule's filled `f`.
#[test]
fn the_axis_fallback_skips_a_filled_channel_the_layers_mark_never_draws() {
    let spec: ChartSpec = serde_json::from_str(
        r#"{"data":{"kind":"named","name":"default"},"mark":"rule","encoding":{"y":{"field":"v"}},"layers":[{"mark":"rule","encoding":{"x":{"field":"f"},"y":{"field":"v"}},"data_source":"a","inherited_positional":["x"]},{"mark":"point","encoding":{"x":{"field":"g"},"y":{"field":"v"}},"data_source":"b"}]}"#,
    )
    .expect("valid spec");
    let (x, _) = positional_binding(&spec, PositionalChannel::X);
    assert_eq!(x.map(|e| e.field.clone()), Some("g".to_string()));
}

/// A geoshape draws no position channel, so a layered geoshape on its own
/// output with no `x`/`y` of its own is not refused for missing them, while a
/// point in its place, which draws both, still is.
#[test]
fn a_geoshape_layer_needs_no_position_channel() {
    let json = |mark: &str| {
        format!(
            r#"{{"data":{{"kind":"named","name":"default"}},"mark":"point","encoding":{{"x":{{"field":"a"}},"y":{{"field":"b"}}}},"layers":[{{"mark":"point","encoding":{{"x":{{"field":"a"}},"y":{{"field":"b"}}}}}},{{"mark":"{mark}","encoding":{{}},"data_source":"shapes"}}]}}"#
        )
    };
    let geoshape: ChartSpec = serde_json::from_str(&json("geoshape")).expect("valid spec");
    assert!(validate_drawing_layers_can_plot(&geoshape, &geoshape.encoding).is_ok());
    let point: ChartSpec = serde_json::from_str(&json("point")).expect("valid spec");
    assert!(matches!(
        validate_drawing_layers_can_plot(&point, &point.encoding),
        Err(RenderError::LayerMissingPositionalChannel { layer: 1, .. })
    ));
}

/// An unlayered `Overlay` member binding an auxiliary channel to a column
/// its one batch lacks refuses as that chart alone does, naming the
/// channel, though the group resolves the member's scales in its place.
/// Rejects a group scale that skips the member's binding and draws it
/// without the channel.
#[test]
fn an_unlayered_overlay_member_binding_an_absent_column_refuses() {
    use crate::spec::encoding::{Encoding, EncodingSpec};
    let chart = |color: &str| ChartSpec {
        mark: crate::spec::mark::Mark::Point,
        encoding: Encoding {
            x: Some(EncodingSpec { field: "x".into(), ..Default::default() }),
            y: Some(EncodingSpec { field: "x".into(), ..Default::default() }),
            color: Some(EncodingSpec { field: color.into(), ..Default::default() }),
            ..Default::default()
        },
        ..Default::default()
    };
    let rows = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("x", ArrowDataType::Float64, false)])),
        vec![Arc::new(Float64Array::from(vec![1.0, 2.0]))],
    )
    .unwrap();
    let outputs = HashMap::new();
    let (absent, other) = (chart("w"), chart("x"));
    let members = [
        LeafResolveInput { spec: &absent, encoding: &absent.encoding, final_batch: &rows, transform_outputs: &outputs, transform_lineages: &crate::transform::run::OutputLineages::new() },
        LeafResolveInput { spec: &other, encoding: &other.encoding, final_batch: &rows, transform_outputs: &outputs, transform_lineages: &crate::transform::run::OutputLineages::new() },
    ];
    let refusal = build_overlay_auxiliary_scales(&members, &absent.encoding, &ThemeInputs::default(), &mut Vec::new())
        .expect_err("member 0's color names a column its rows lack");
    assert_eq!(refusal.channel, "color");
    assert_eq!(refusal.source, RenderError::UnknownColumn { name: "w".to_string() });
}

/// A text mark under `CoordPolar` that binds the coord's theta channel is
/// single-axis eligible on that channel, as an arc is: a labelled pie's text
/// layer binds theta and no radius. Its radius channel alone is not enough,
/// and outside a polar coord a text mark still needs both axes.
#[test]
fn polar_text_is_single_axis_eligible_on_its_theta_channel() {
    use crate::spec::coord::CoordKind;
    use crate::spec::mark::Mark;
    use ferrum_scene::{PolarDirection, PolarThetaChannel};

    let polar = |theta| CoordKind::Polar {
        theta,
        start_angle: 0.0,
        direction: PolarDirection::Clockwise,
        inner_radius: 0.0,
        outer_radius: None,
        pad_angle: 0.0,
    };
    let on_x = polar(PolarThetaChannel::X);
    let on_y = polar(PolarThetaChannel::Y);
    for mark in [Mark::Arc, Mark::Text] {
        assert!(super::mark_single_axis_eligible(mark, Some(&on_x), true, false), "{mark}: theta on x");
        assert!(super::mark_single_axis_eligible(mark, Some(&on_y), false, true), "{mark}: theta on y");
        assert!(!super::mark_single_axis_eligible(mark, Some(&on_x), false, true), "{mark}: radius alone");
        assert!(!super::mark_single_axis_eligible(mark, None, true, false), "{mark}: no polar coord");
    }
    assert!(!super::mark_single_axis_eligible(Mark::Point, Some(&on_x), true, false), "control: a point needs both");
}

// ── `ScaleKind::tick_anchors`: which tick values are `Log` decades ─────────

#[test]
fn tick_anchors_marks_log_decades_among_the_1_2_5_ladder() {
    use crate::scale::log::LogScale;
    let log = ScaleKind::Log(LogScale::new_internal(vec![1.0, 100.0], vec![0.0, 400.0], 10.0, false, false));
    let values = log.tick_values_raw(10).expect("log scale has tick values");
    assert_eq!(values, vec![1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]);
    let anchors = log.tick_anchors(&values);
    assert_eq!(anchors, vec![true, false, false, true, false, false, true]);
}

#[test]
fn tick_anchors_empty_for_every_non_log_kind() {
    use crate::scale::linear::LinearScale;
    let linear = ScaleKind::Linear(LinearScale::new_internal(vec![0.0, 100.0], vec![0.0, 400.0], false, false));
    let values = linear.tick_values_raw(5).expect("linear scale has tick values");
    assert!(linear.tick_anchors(&values).is_empty(), "non-Log tick_anchors is empty, not all-false");
}
