//! Registry exhaustiveness + declared-vs-actual schema tests (batch-D task
//! 1). Split from `mod.rs` per the repo's directory-module convention: this
//! carries a per-kind schema fixture for every kind whose `outputs` entry is
//! non-empty (21 of the transform's ~42 kinds), which would roughly double
//! `mod.rs`'s length if kept inline.
//!
//! Five things are proven here, all against REAL production types/sites (no
//! hand-built mirrors):
//! 1. Every kind in [`super::all_transform_kinds`] has an [`super::entry_for_kind`]
//!    (the net for a new transform landing without a registry decision).
//! 2. For every kind whose declared `outputs` is non-empty, applying a real
//!    `*Spec` (constructed with the DEFAULT method/kernel and minimal valid
//!    input) against a tiny fixture batch produces a schema EQUAL TO the
//!    declared `outputs` plus a declared passthrough allowlist — the
//!    "manifest-test mold" from `chart_config.rs:957-990`, adapted from a
//!    static field-set diff to a real applied-schema diff (there is no
//!    `deny_unknown_fields` error to scrape here; the transform's actual
//!    output IS the ground truth). Equality, not subset (round-2 cycle-1
//!    fix): a subset check can't catch under-declaration, which is exactly
//!    how `contour`'s missing `contour_x2`/`contour_y2` and `swarm`'s
//!    missing additive columns passed cycle 1's suite.
//! 3. Every `TransformSpec` variant's real serde `"type"` tag matches its
//!    registry `kind` or an explicit named exception
//!    (`wire_tag_matches_registry_kind_or_named_exception`) — the module
//!    doc's wire-tag claim is self-checking, not prose.
//! 4. Every kind's declared `params` equals its spec struct's real
//!    serde-accepted field set, mined from a `deny_unknown_fields` rejection
//!    (`params_match_spec_struct_serde_fields`) — the `scale_accepted_keys`
//!    drift-guard precedent, ported to `params`. That guard's own coverage is
//!    pinned against [`super::all_transform_kinds`] (not just asserted once
//!    and left to rot) so a new kind without a call site fails loudly.
//! 5. Every declared `kde`/`qq`/`smooth`/`hex`/`raster`/`linkage`
//!    method/kernel/distribution/aggregate is accepted by its real PyO3
//!    constructor, and a bogus one is refused (the
//!    `*_accepted_and_bogus_*_rejected` tests) — the same precedent, ported
//!    to `methods`. `pivot` is the one declared vocabulary with no such
//!    guard: its `_ => sum` fallback (`pivot.rs`) never refuses a bogus op,
//!    so there is no real acceptance/refusal site to assert against (see the
//!    `"pivot"` entry's comment in `mod.rs`).
//! 6. [`KINDS_WITH_SCHEMA_FIXTURE`] pins the per-kind fixture set (item 2)
//!    against "every kind whose declared `outputs` is non-empty", so a new
//!    kind landing with a real schema and no fixture fails a test instead of
//!    silently joining the equality check's blind spot for `outputs: []`
//!    kinds.

use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};

use super::*;
use crate::transform::core::TransformSpec;

fn f64_batch(cols: &[(&str, &[f64])]) -> RecordBatch {
    let fields: Vec<Field> =
        cols.iter().map(|(name, _)| Field::new(*name, DataType::Float64, true)).collect();
    let arrays: Vec<ArrayRef> =
        cols.iter().map(|(_, vals)| Arc::new(Float64Array::from(vals.to_vec())) as ArrayRef).collect();
    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap()
}

fn value_group_batch(values: &[f64], groups: &[&str]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("v", DataType::Float64, true),
        Field::new("g", DataType::Utf8, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(values.to_vec())),
            Arc::new(StringArray::from(groups.to_vec())),
        ],
    )
    .unwrap()
}

/// Asserts `actual`'s field set is EXACTLY `expected`'s declared names plus
/// `passthrough` (fixture input columns / groupby echoes the registry's
/// `outputs` entry deliberately excludes, per the module doc's convention —
/// e.g. `swarm`'s original input columns, which it carries through
/// unchanged). Two checks, not one:
/// 1. every declared `(name, dtype)` is present in `actual` with a matching
///    `DataType` (the original subset check);
/// 2. `actual`'s field-name set equals `expected ∪ passthrough` exactly — an
///    under-declared registry entry (an extra real column not in either set)
///    fails here even though it would have passed a subset-only check. This
///    is what the cycle-1 review found `contour` (fill=false's real 6-column
///    schema vs. the 4 declared) and `swarm` (`[]` vs. its real additive
///    columns) both slipping past.
fn assert_schema_matches(
    actual: &Schema,
    expected: &[(&'static str, DataType)],
    passthrough: &[&str],
    kind: &str,
) {
    for (name, dt) in expected {
        let field = actual
            .field_with_name(name)
            .unwrap_or_else(|_| panic!("{kind}: declared output '{name}' missing from actual schema {actual:?}"));
        assert_eq!(
            field.data_type(),
            dt,
            "{kind}: declared dtype for '{name}' is {dt:?}, actual schema has {:?}",
            field.data_type()
        );
    }
    let expected_names: std::collections::BTreeSet<&str> =
        expected.iter().map(|(n, _)| *n).chain(passthrough.iter().copied()).collect();
    let actual_names: std::collections::BTreeSet<&str> =
        actual.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(
        actual_names, expected_names,
        "{kind}: registry `outputs` (+ declared passthrough {passthrough:?}) drifted from the \
         real applied schema.\nin real schema but not declared (under-declaration): {:?}\n\
         declared but not in real schema (over-declaration): {:?}",
        actual_names.difference(&expected_names).collect::<Vec<_>>(),
        expected_names.difference(&actual_names).collect::<Vec<_>>(),
    );
}

#[test]
fn every_transform_kind_has_a_registry_entry() {
    for kind in all_transform_kinds() {
        let entry = entry_for_kind(kind)
            .unwrap_or_else(|| panic!("transform kind '{kind}' has no registry entry"));
        assert!(!entry.params.is_empty(), "'{kind}': every transform spec has at least a `name` param");
    }
}

#[test]
fn unknown_kind_returns_none() {
    assert!(entry_for_kind("not_a_real_transform").is_none());
}

/// Kinds with a per-kind schema fixture below (`*_outputs_match_actual_schema`
/// / `*_match_shared_fit_schema`). Required Fix 2 (round-3 cycle-2 review):
/// pinned against "every kind whose declared `outputs` is non-empty"
/// (`schema_fixture_coverage_matches_kinds_with_declared_outputs`) so a task
/// 9-11 kind landing with a real schema and no fixture fails a test, instead
/// of silently joining the 20 `outputs: []` kinds that
/// `assert_schema_matches`'s equality check has no fixture to run against.
const KINDS_WITH_SCHEMA_FIXTURE: &[&str] = &[
    "bin",
    "bin_2d",
    "kde",
    "smooth",
    "summary",
    "error_extent",
    "box_stats",
    "violin",
    "kde_2d",
    "contour",
    "qq",
    "linkage",
    "raster",
    "hex",
    "swarm",
    "letter_value",
    "logistic",
    "glm",
    "robust",
    "regression_data",
    "loess_data",
    "density_data",
];

#[test]
fn schema_fixture_coverage_matches_kinds_with_declared_outputs() {
    let declared_nonempty: std::collections::BTreeSet<&str> = all_transform_kinds()
        .into_iter()
        .filter(|k| !entry_for_kind(k).unwrap().outputs.is_empty())
        .collect();
    let fixtures: std::collections::BTreeSet<&str> = KINDS_WITH_SCHEMA_FIXTURE.iter().copied().collect();
    assert_eq!(
        fixtures, declared_nonempty,
        "KINDS_WITH_SCHEMA_FIXTURE drifted from the kinds with non-empty declared `outputs`.\n\
         has a fixture but declared `outputs` is now empty (stale fixture-list entry): {:?}\n\
         has non-empty `outputs` but no schema fixture (Required Fix 2 gap): {:?}",
        fixtures.difference(&declared_nonempty).collect::<Vec<_>>(),
        declared_nonempty.difference(&fixtures).collect::<Vec<_>>(),
    );
}

#[test]
fn bin_outputs_match_actual_schema() {
    pyo3::Python::initialize();
    use crate::transform::bin::BinSpec;
    use crate::transform::bin_mode::BinMode;
    let spec = BinSpec {
        field: "v".into(),
        mode: BinMode::Fixed { n: 4 },
        extent: None,
        nice: false,
        cumulative: false,
        shared_extent: false,
        groupby: vec![],
        name: None,
    };
    let batch = f64_batch(&[("v", &[1.0, 2.0, 3.0, 4.0, 5.0])]);
    let out = crate::transform::bin::apply(&spec, &batch).unwrap();
    assert_schema_matches(out.schema().as_ref(), &entry_for_kind("bin").unwrap().outputs, &[], "bin");
}

#[test]
fn bin_2d_outputs_match_actual_schema() {
    pyo3::Python::initialize();
    use crate::transform::bin_2d::Bin2DSpec;
    use crate::transform::bin_mode::BinMode;
    let spec = Bin2DSpec {
        x: "x".into(),
        y: "y".into(),
        bins_x: BinMode::Fixed { n: 3 },
        bins_y: BinMode::Fixed { n: 3 },
        extent_x: None,
        extent_y: None,
        cumulative: false,
        name: None,
    };
    let batch = f64_batch(&[("x", &[1.0, 2.0, 3.0]), ("y", &[1.0, 2.0, 3.0])]);
    let out = crate::transform::bin_2d::apply(&spec, &batch).unwrap();
    assert_schema_matches(out.schema().as_ref(), &entry_for_kind("bin_2d").unwrap().outputs, &[], "bin_2d");
}

#[test]
fn kde_outputs_match_actual_schema() {
    pyo3::Python::initialize();
    use crate::transform::kde::{BandwidthSpec, KdeSpec};
    let spec = KdeSpec {
        field: "v".into(),
        bandwidth: BandwidthSpec::Scott,
        bw_adjust: 1.0,
        n: 16,
        extent: None,
        cumulative: false,
        shared_extent: false,
        kernel: "gaussian".into(),
        groupby: vec![],
        name: None,
    };
    let batch = f64_batch(&[("v", &[1.0, 2.0, 3.0, 4.0, 5.0])]);
    let out = crate::transform::kde::apply(&spec, &batch).unwrap();
    assert_schema_matches(out.schema().as_ref(), &entry_for_kind("kde").unwrap().outputs, &[], "kde");
}

#[test]
fn smooth_fitted_outputs_match_shared_fit_schema() {
    pyo3::Python::initialize();
    use crate::transform::smooth::{SmoothMethod, SmoothOutput, SmoothSpec};
    let spec = SmoothSpec {
        x: "x".into(),
        y: "y".into(),
        method: SmoothMethod::Lm,
        ci: None,
        bandwidth: 0.75,
        degree: 1,
        n: 5,
        seed: 0,
        x_bins: None,
        x_estimator: None,
        output: SmoothOutput::Fitted,
        inject_zero_ref: false,
        inject_metrics: false,
        groupby: vec![],
        name: None,
        x_range: None,
    };
    let batch = f64_batch(&[("x", &[0.0, 1.0, 2.0, 3.0, 4.0]), ("y", &[0.0, 2.0, 4.0, 6.0, 8.0])]);
    let out = crate::transform::smooth::apply(&spec, &batch).unwrap();
    assert_schema_matches(out.schema().as_ref(), &entry_for_kind("smooth").unwrap().outputs, &[], "smooth");
    // The fit-family schema-collapse (spec §4.6): Smooth/Logistic/Glm/Robust
    // declare the identical shared constant, not four independent literals.
    assert_eq!(entry_for_kind("smooth").unwrap().outputs, FIT_OUTPUT_COLUMNS.to_vec());
}

#[test]
fn summary_outputs_match_actual_schema() {
    pyo3::Python::initialize();
    use crate::transform::summary::{ErrorFn, SummarySpec};
    let spec = SummarySpec {
        field: "v".into(),
        groupby: vec![],
        error_fn: ErrorFn::Stdev,
        ci: 0.95,
        n_boot: 200,
        seed: 0,
        name: None,
    };
    let batch = f64_batch(&[("v", &[1.0, 2.0, 3.0, 4.0, 5.0])]);
    let out = crate::transform::summary::apply(&spec, &batch).unwrap();
    assert_schema_matches(out.schema().as_ref(), &entry_for_kind("summary").unwrap().outputs, &[], "summary");
}

#[test]
fn error_extent_outputs_match_actual_schema() {
    pyo3::Python::initialize();
    use crate::transform::error_extent::{ErrorExtentSpec, ErrorMethod};
    let spec = ErrorExtentSpec {
        field: "v".into(),
        groupby: vec![],
        method: ErrorMethod::Stdev,
        seed: 0,
        n_boot: 200,
        name: None,
    };
    let batch = f64_batch(&[("v", &[1.0, 2.0, 3.0, 4.0, 5.0])]);
    let out = crate::transform::error_extent::apply(&spec, &batch).unwrap();
    assert_schema_matches(
        out.schema().as_ref(),
        &entry_for_kind("error_extent").unwrap().outputs,
        &[],
        "error_extent",
    );
}

#[test]
fn box_stats_outputs_match_actual_schema() {
    pyo3::Python::initialize();
    use crate::transform::box_stats::{BoxStatsSpec, WhiskerExtent};
    let spec = BoxStatsSpec {
        field: "v".into(),
        groupby: vec![],
        whisker_extent: WhiskerExtent::default(),
        name: None,
    };
    let batch = f64_batch(&[("v", &[1.0, 2.0, 3.0, 4.0, 5.0])]);
    let out = crate::transform::box_stats::apply(&spec, &batch).unwrap();
    assert_schema_matches(out.schema().as_ref(), &entry_for_kind("box_stats").unwrap().outputs, &[], "box_stats");
}

#[test]
fn violin_outputs_match_actual_schema() {
    pyo3::Python::initialize();
    use crate::transform::kde::BandwidthSpec;
    use crate::transform::violin::ViolinSpec;
    let spec = ViolinSpec {
        field: "v".into(),
        groupby: vec![],
        bandwidth: BandwidthSpec::Scott,
        bw_adjust: 1.0,
        n: 16,
        width: 0.8,
        extent: None,
        shared_extent: false,
        name: None,
        horizontal: false,
    };
    let batch = f64_batch(&[("v", &[1.0, 2.0, 3.0, 4.0, 5.0])]);
    let out = crate::transform::violin::apply(&spec, &batch).unwrap();
    assert_schema_matches(out.schema().as_ref(), &entry_for_kind("violin").unwrap().outputs, &[], "violin");
}

/// Also exercises `contour`'s schema, since `contour::apply` requires a
/// Kde2D-shaped input batch (see `contour.rs`'s module doc) — this one
/// fixture proves both kinds' declared outputs against the real pipeline
/// (Kde2D's own output feeding Contour, not a hand-built stand-in).
#[test]
fn kde_2d_and_contour_outputs_match_actual_schema() {
    pyo3::Python::initialize();
    use crate::transform::contour::ContourSpec;
    use crate::transform::kde::BandwidthSpec;
    use crate::transform::kde_2d::Kde2DSpec;
    let kde2d_spec = Kde2DSpec {
        x: "x".into(),
        y: "y".into(),
        bandwidth: BandwidthSpec::Scott,
        n: 8,
        extent: None,
        groupby: vec![],
        name: None,
    };
    let batch = f64_batch(&[("x", &[1.0, 2.0, 3.0, 4.0]), ("y", &[1.0, 3.0, 2.0, 4.0])]);
    let kde2d_out = crate::transform::kde_2d::apply(&kde2d_spec, &batch).unwrap();
    assert_schema_matches(
        kde2d_out.schema().as_ref(),
        &entry_for_kind("kde_2d").unwrap().outputs,
        &[],
        "kde_2d",
    );

    let contour_spec = ContourSpec { thresholds: 5, fill: false, smooth: false, name: None };
    let contour_out = crate::transform::contour::apply(&contour_spec, &kde2d_out).unwrap();
    assert_schema_matches(
        contour_out.schema().as_ref(),
        &entry_for_kind("contour").unwrap().outputs,
        &[],
        "contour",
    );
}

#[test]
fn qq_outputs_match_actual_schema() {
    pyo3::Python::initialize();
    use crate::transform::qq::QQSpec;
    let spec = QQSpec {
        field: "v".into(),
        distribution: "normal".into(),
        dequantize: false,
        emit_line: true,
        seed: 0,
        name: None,
    };
    let batch = f64_batch(&[("v", &[1.0, 2.0, 3.0, 4.0, 5.0])]);
    let out = crate::transform::qq::apply(&spec, &batch).unwrap();
    assert_schema_matches(out.schema().as_ref(), &entry_for_kind("qq").unwrap().outputs, &[], "qq");
}

#[test]
fn linkage_outputs_match_actual_schema() {
    pyo3::Python::initialize();
    use crate::transform::linkage::{LinkageAxis, LinkageMethod, LinkageSpec, DistanceMetric};
    let spec = LinkageSpec {
        method: LinkageMethod::Ward,
        metric: DistanceMetric::Euclidean,
        axis: LinkageAxis::Rows,
        z_score: None,
        standard_scale: None,
        name: None,
    };
    let batch = f64_batch(&[("a", &[1.0, 2.0, 3.0]), ("b", &[4.0, 1.0, 0.0])]);
    let out = crate::transform::linkage::apply(&spec, &batch).unwrap();
    assert_schema_matches(out.schema().as_ref(), &entry_for_kind("linkage").unwrap().outputs, &[], "linkage");
}

#[test]
fn raster_outputs_match_actual_schema() {
    pyo3::Python::initialize();
    use crate::transform::raster::RasterSpec;
    let spec = RasterSpec {
        x: "x".into(),
        y: "y".into(),
        aggregate: "count".into(),
        field: None,
        resolution: Default::default(),
        min_count: None,
        log_scale: false,
        name: None,
        extent_x: None,
        extent_y: None,
    };
    let batch = f64_batch(&[("x", &[1.0, 2.0, 3.0, 4.0]), ("y", &[1.0, 3.0, 2.0, 4.0])]);
    let out = crate::transform::raster::apply(&spec, &batch).unwrap();
    assert_schema_matches(out.schema().as_ref(), &entry_for_kind("raster").unwrap().outputs, &[], "raster");
}

#[test]
fn hex_outputs_match_actual_schema() {
    pyo3::Python::initialize();
    use crate::transform::hex::HexSpec;
    let spec = HexSpec {
        x: "x".into(),
        y: "y".into(),
        bin_size: None,
        aggregate: "count".into(),
        field: None,
        name: None,
        extent_x: None,
        extent_y: None,
    };
    let batch = f64_batch(&[("x", &[1.0, 2.0, 3.0, 4.0]), ("y", &[1.0, 3.0, 2.0, 4.0])]);
    let out = crate::transform::hex::apply(&spec, &batch).unwrap();
    assert_schema_matches(out.schema().as_ref(), &entry_for_kind("hex").unwrap().outputs, &[], "hex");
}

#[test]
fn letter_value_outputs_match_actual_schema() {
    pyo3::Python::initialize();
    use crate::transform::letter_value::{KDepth, LetterValueSpec};
    let spec = LetterValueSpec {
        value: "v".into(),
        group: Some("g".into()),
        k_depth: KDepth::Tukey,
        outlier_threshold: 1.5,
        name: None,
        sort: None,
    };
    let batch = value_group_batch(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &["a", "a", "a", "a", "a", "a", "a", "a"],
    );
    let out = crate::transform::letter_value::apply(&spec, &batch).unwrap();
    assert_schema_matches(
        out.schema().as_ref(),
        &entry_for_kind("letter_value").unwrap().outputs,
        &[],
        "letter_value",
    );
}

#[test]
fn logistic_outputs_match_shared_fit_schema() {
    pyo3::Python::initialize();
    use crate::transform::logistic::LogisticSpec;
    let spec =
        LogisticSpec { x: "x".into(), y: "y".into(), n_grid: 10, ci: None, max_iter: 25, tol: 1e-8, name: None };
    // Non-separable y (not monotone in x) — a perfectly-separable fixture
    // trips `fit_irls`'s perfect-separation guard and errors instead of
    // fitting, which isn't what this schema check is after.
    let batch =
        f64_batch(&[("x", &[0.0, 1.0, 2.0, 3.0, 4.0]), ("y", &[0.0, 1.0, 0.0, 1.0, 1.0])]);
    let out = crate::transform::logistic::apply(&spec, &batch).unwrap();
    assert_schema_matches(out.schema().as_ref(), &entry_for_kind("logistic").unwrap().outputs, &[], "logistic");
}

#[test]
fn glm_outputs_match_shared_fit_schema() {
    pyo3::Python::initialize();
    use crate::transform::glm::{GlmFamily, GlmLink, GlmSpec};
    let spec = GlmSpec {
        x: "x".into(),
        y: "y".into(),
        family: GlmFamily::Gaussian,
        link: GlmLink::Identity,
        n_grid: 10,
        ci: None,
        max_iter: 25,
        tol: 1e-8,
        name: None,
    };
    let batch = f64_batch(&[("x", &[0.0, 1.0, 2.0, 3.0, 4.0]), ("y", &[0.0, 2.0, 4.0, 6.0, 8.0])]);
    let out = crate::transform::glm::apply(&spec, &batch).unwrap();
    assert_schema_matches(out.schema().as_ref(), &entry_for_kind("glm").unwrap().outputs, &[], "glm");
}

#[test]
fn robust_outputs_match_shared_fit_schema() {
    pyo3::Python::initialize();
    use crate::transform::robust::RobustSpec;
    use crate::transform::smooth::SmoothOutput;
    let spec = RobustSpec {
        x: "x".into(),
        y: "y".into(),
        n_grid: 10,
        ci: None,
        huber_c: 1.345,
        max_iter: 25,
        tol: 1e-8,
        output: SmoothOutput::Fitted,
        inject_zero_ref: false,
        inject_metrics: false,
        name: None,
        x_range: None,
    };
    let batch = f64_batch(&[("x", &[0.0, 1.0, 2.0, 3.0, 4.0]), ("y", &[0.0, 2.0, 4.0, 6.0, 8.0])]);
    let out = crate::transform::robust::apply(&spec, &batch).unwrap();
    assert_schema_matches(out.schema().as_ref(), &entry_for_kind("robust").unwrap().outputs, &[], "robust");
}

#[test]
fn regression_data_outputs_match_actual_schema() {
    pyo3::Python::initialize();
    use crate::transform::regression_data::RegressionDataSpec;
    let spec = RegressionDataSpec {
        x: "x".into(),
        y: "y".into(),
        method: "linear".into(),
        order: 1,
        groupby: None,
        as_: ("x".into(), "y".into()),
        name: None,
    };
    let batch = f64_batch(&[("x", &[0.0, 1.0, 2.0, 3.0, 4.0]), ("y", &[0.0, 2.0, 4.0, 6.0, 8.0])]);
    let out = crate::transform::regression_data::apply(&spec, &batch).unwrap();
    assert_schema_matches(
        out.schema().as_ref(),
        &entry_for_kind("regression_data").unwrap().outputs,
        &[],
        "regression_data",
    );
}

#[test]
fn loess_data_outputs_match_actual_schema() {
    pyo3::Python::initialize();
    use crate::transform::loess_data::LoessDataSpec;
    let spec = LoessDataSpec {
        x: "x".into(),
        y: "y".into(),
        bandwidth: 0.5,
        groupby: None,
        as_: ("x".into(), "y".into()),
        name: None,
    };
    let batch = f64_batch(&[
        ("x", &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        ("y", &[0.0, 2.0, 3.5, 6.0, 7.5, 10.0]),
    ]);
    let out = crate::transform::loess_data::apply(&spec, &batch).unwrap();
    assert_schema_matches(
        out.schema().as_ref(),
        &entry_for_kind("loess_data").unwrap().outputs,
        &[],
        "loess_data",
    );
}

#[test]
fn density_data_outputs_match_actual_schema() {
    pyo3::Python::initialize();
    use crate::transform::density_data::DensityDataSpec;
    let spec = DensityDataSpec {
        field: "v".into(),
        bandwidth: None,
        groupby: None,
        extent: None,
        steps: None,
        cumulative: false,
        as_: ("value".into(), "density".into()),
        name: None,
    };
    let batch = f64_batch(&[("v", &[1.0, 2.0, 3.0, 4.0, 5.0])]);
    let out = crate::transform::density_data::apply(&spec, &batch).unwrap();
    assert_schema_matches(
        out.schema().as_ref(),
        &entry_for_kind("density_data").unwrap().outputs,
        &[],
        "density_data",
    );
}

/// `swarm`'s declared outputs (fixed round-2 cycle-1: was `outputs: []`,
/// inconsistent with `violin`'s declared `__pos_x_offset__` for the identical
/// column) are additive to an ARBITRARY input schema — unlike every other
/// fixture above, the real output also carries the fixture's own `cat`/`v`
/// input columns unchanged, so this is the one call site that needs a
/// non-empty `passthrough` allowlist for `assert_schema_matches`'s equality
/// check to hold.
#[test]
fn swarm_outputs_match_actual_schema() {
    pyo3::Python::initialize();
    use crate::transform::swarm::{SwarmOrient, SwarmSide, SwarmSpec};
    let schema = Arc::new(Schema::new(vec![
        Field::new("cat", DataType::Utf8, false),
        Field::new("v", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["a", "a", "b"])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
        ],
    )
    .unwrap();
    let spec = SwarmSpec {
        category: "cat".into(),
        value: "v".into(),
        point_size: 5.0,
        spacing: 1.0,
        side: SwarmSide::Both,
        orient: SwarmOrient::Vertical,
        dodge: None,
        name: None,
    };
    let out = crate::transform::swarm::apply(&spec, &batch).unwrap();
    assert_schema_matches(
        out.schema().as_ref(),
        &entry_for_kind("swarm").unwrap().outputs,
        &["cat", "v"],
        "swarm",
    );
}

// ── PyO3-facing `transform_registry` (spec §4.6) ────────────────────────────

fn attach<F, T>(f: F) -> T
where
    F: for<'py> FnOnce(Python<'py>) -> T,
{
    pyo3::Python::initialize();
    Python::attach(f)
}

#[test]
fn transform_registry_known_kind_returns_dict_shape() {
    attach(|py| {
        let dict = transform_registry(py, "kde").unwrap();
        let dict = dict.bind(py);
        let methods: Vec<String> = dict.get_item("methods").unwrap().unwrap().extract().unwrap();
        assert!(methods.contains(&"gaussian".to_string()));
        let outputs: Vec<(String, String)> = dict.get_item("outputs").unwrap().unwrap().extract().unwrap();
        assert_eq!(outputs, vec![("value".to_string(), "Float64".to_string()), ("density".to_string(), "Float64".to_string())]);
        let params: Vec<String> = dict.get_item("params").unwrap().unwrap().extract().unwrap();
        assert!(params.contains(&"kernel".to_string()));
    });
}

#[test]
fn transform_registry_unknown_kind_raises_value_error_listing_known_kinds() {
    attach(|py| {
        let err = transform_registry(py, "not_a_real_transform").unwrap_err();
        let msg = err.value(py).to_string();
        assert!(msg.contains("not_a_real_transform"), "message should name the unknown kind: {msg}");
        assert!(msg.contains("smooth"), "message should list known kinds: {msg}");
        assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py), "expected ValueError, got {err:?}");
    });
}

/// Required fix 7 (round-2 cycle-1): walk EVERY kind from
/// `all_transform_kinds()` through `transform_registry(py, kind)`, not just
/// `"kde"`. This gives the pyfunction real coverage across all 42 entries and
/// demotes `dtype_label`'s `_ => unimplemented!` fallback (mod.rs) from a
/// runtime panic reachable at the PyO3 boundary to a test failure: a future
/// registry entry using a dtype `dtype_label` doesn't yet cover fails HERE,
/// not in a caller's Python process.
#[test]
fn transform_registry_covers_every_kind() {
    attach(|py| {
        for kind in all_transform_kinds() {
            let dict = transform_registry(py, kind)
                .unwrap_or_else(|e| panic!("transform_registry('{kind}') failed: {e}"));
            let dict = dict.bind(py);
            assert!(dict.get_item("methods").unwrap().is_some(), "{kind}: missing 'methods' key");
            assert!(dict.get_item("outputs").unwrap().is_some(), "{kind}: missing 'outputs' key");
            assert!(dict.get_item("params").unwrap().is_some(), "{kind}: missing 'params' key");
        }
    });
}

// ── Wire-tag divergence (Required Fix 4, round-2 cycle-1) ──────────────────
//
// The module doc's original claim ("the two agree everywhere except
// `TimeUnit`") was prose, unchecked against serde's actual `Serialize`
// output. `Kde2D`'s snake_case wire tag is `kde2_d`, not `kde_2d` — a second,
// real divergence the prose missed. This section makes the claim
// self-checking: build one instance of every `TransformSpec` variant,
// serialize it, and assert its real `"type"` tag against the registry kind
// (identity) or an explicit, named exception (mirroring
// `spec/encoding.rs`'s `accepted_keys_for_scale_type_matches_every_variants_
// serialized_keys`, which does the same for `ScaleSpec`).

/// One instance of every `TransformSpec` variant, keyed by its registry
/// `kind`. Field values are deliberately minimal/arbitrary — these are NEVER
/// passed through a transform's `apply()`, only through `serde_json::to_value`
/// to read the enum's derived `"type"` tag, so validity of the *content*
/// (e.g. non-empty groupby, valid method spellings) doesn't matter, only that
/// every field has SOME value of the declared type.
fn all_transform_spec_instances() -> Vec<(&'static str, TransformSpec)> {
    use crate::transform::aggregate::AggregateSpec;
    use crate::transform::bin::BinSpec;
    use crate::transform::bin_2d::Bin2DSpec;
    use crate::transform::bin_mode::BinMode;
    use crate::transform::box_stats::{BoxStatsSpec, WhiskerExtent};
    use crate::transform::calculate::CalculateSpec;
    use crate::transform::contour::ContourSpec;
    use crate::transform::data_aggregate::DataAggregateSpec;
    use crate::transform::data_bin::DataBinSpec;
    use crate::transform::data_stack::DataStackSpec;
    use crate::transform::data_window::DataWindowSpec;
    use crate::transform::density_data::DensityDataSpec;
    use crate::transform::error_extent::{ErrorExtentSpec, ErrorMethod};
    use crate::transform::filter::FilterSpec;
    use crate::transform::flatten::FlattenSpec;
    use crate::transform::fold::FoldSpec;
    use crate::transform::glm::{GlmFamily, GlmLink, GlmSpec};
    use crate::transform::hex::HexSpec;
    use crate::transform::identity::IdentitySpec;
    use crate::transform::impute::ImputeSpec;
    use crate::transform::join_aggregate::JoinAggregateSpec;
    use crate::transform::kde::{BandwidthSpec, KdeSpec};
    use crate::transform::kde_2d::Kde2DSpec;
    use crate::transform::letter_value::{KDepth, LetterValueSpec};
    use crate::transform::linkage::{DistanceMetric, LinkageAxis, LinkageMethod, LinkageSpec};
    use crate::transform::loess_data::LoessDataSpec;
    use crate::transform::logistic::LogisticSpec;
    use crate::transform::outliers::OutliersSpec;
    use crate::transform::pivot::PivotSpec;
    use crate::transform::qq::QQSpec;
    use crate::transform::raster::{RasterSpec, ResolutionSpec};
    use crate::transform::reference_line::ReferenceLineSpec;
    use crate::transform::regression_data::RegressionDataSpec;
    use crate::transform::reorder::ReorderSpec;
    use crate::transform::robust::RobustSpec;
    use crate::transform::sample::SampleSpec;
    use crate::transform::smooth::{SmoothMethod, SmoothOutput, SmoothSpec};
    use crate::transform::summary::{ErrorFn, SummarySpec};
    use crate::transform::swarm::{SwarmOrient, SwarmSide, SwarmSpec};
    use crate::transform::timeunit::TimeUnitSpec;
    use crate::transform::top_k::TopKSpec;
    use crate::transform::unpivot::UnpivotSpec;
    use crate::transform::violin::ViolinSpec;
    use TransformSpec::*;

    vec![
        ("bin", Bin(BinSpec {
            field: "v".into(), mode: BinMode::Fixed { n: 4 }, extent: None, nice: false,
            cumulative: false, shared_extent: false, groupby: vec![], name: None,
        })),
        ("bin_2d", Bin2D(Bin2DSpec {
            x: "x".into(), y: "y".into(), bins_x: BinMode::Fixed { n: 3 },
            bins_y: BinMode::Fixed { n: 3 }, extent_x: None, extent_y: None,
            cumulative: false, name: None,
        })),
        ("kde", Kde(KdeSpec {
            field: "v".into(), bandwidth: BandwidthSpec::Scott, bw_adjust: 1.0, n: 16,
            extent: None, cumulative: false, shared_extent: false, kernel: "gaussian".into(),
            groupby: vec![], name: None,
        })),
        ("smooth", Smooth(SmoothSpec {
            x: "x".into(), y: "y".into(), method: SmoothMethod::Lm, ci: None, bandwidth: 0.75,
            degree: 1, n: 5, seed: 0, x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,
            inject_zero_ref: false, inject_metrics: false, groupby: vec![], name: None, x_range: None,
        })),
        ("aggregate", Aggregate(AggregateSpec { ops: vec![], groupby: vec![], name: None })),
        ("summary", Summary(SummarySpec {
            field: "v".into(), groupby: vec![], error_fn: ErrorFn::Stdev, ci: 0.95, n_boot: 200,
            seed: 0, name: None,
        })),
        ("outliers", Outliers(OutliersSpec { field: "v".into(), groupby: vec![], extent: 1.5, name: None })),
        ("error_extent", ErrorExtent(ErrorExtentSpec {
            field: "v".into(), groupby: vec![], method: ErrorMethod::Stdev, seed: 0, n_boot: 200,
            name: None,
        })),
        ("box_stats", BoxStats(BoxStatsSpec {
            field: "v".into(), groupby: vec![], whisker_extent: WhiskerExtent::IqrMultiplier(1.5),
            name: None,
        })),
        ("violin", Violin(ViolinSpec {
            field: "v".into(), groupby: vec![], bandwidth: BandwidthSpec::Scott, bw_adjust: 1.0,
            n: 16, width: 0.8, extent: None, shared_extent: false, name: None, horizontal: false,
        })),
        ("kde_2d", Kde2D(Kde2DSpec {
            x: "x".into(), y: "y".into(), bandwidth: BandwidthSpec::Scott, n: 8, extent: None,
            groupby: vec![], name: None,
        })),
        ("contour", Contour(ContourSpec { thresholds: 6, fill: false, smooth: true, name: None })),
        ("qq", Qq(QQSpec {
            field: "v".into(), distribution: "normal".into(), dequantize: false, emit_line: true,
            seed: 0, name: None,
        })),
        ("linkage", Linkage(LinkageSpec {
            method: LinkageMethod::Ward, metric: DistanceMetric::Euclidean, axis: LinkageAxis::Rows,
            z_score: None, standard_scale: None, name: None,
        })),
        ("raster", Raster(RasterSpec {
            x: "x".into(), y: "y".into(), aggregate: "count".into(), field: None,
            resolution: ResolutionSpec::Named("screen".into()), min_count: None, log_scale: false,
            name: None, extent_x: None, extent_y: None,
        })),
        ("hex", Hex(HexSpec {
            x: "x".into(), y: "y".into(), bin_size: None, aggregate: "count".into(), field: None,
            name: None, extent_x: None, extent_y: None,
        })),
        ("swarm", Swarm(SwarmSpec {
            category: "cat".into(), value: "v".into(), point_size: 5.0, spacing: 1.0,
            side: SwarmSide::Both, orient: SwarmOrient::Vertical, dodge: None, name: None,
        })),
        ("unpivot", Unpivot(UnpivotSpec {
            id_vars: vec![], value_vars: None, var_name: "variable".into(), value_name: "value".into(),
            name: None,
        })),
        ("reorder", Reorder(ReorderSpec { by: "x".into(), drop_index: false, from: None, name: None })),
        ("reference_line", ReferenceLine(ReferenceLineSpec {
            x_field: "x".into(), y_field: "y".into(), x: (0.0, 1.0), y: (0.0, 1.0), name: None,
        })),
        ("letter_value", LetterValue(LetterValueSpec {
            value: "v".into(), group: None, k_depth: KDepth::Tukey, outlier_threshold: 1.5,
            name: None, sort: None,
        })),
        ("logistic", Logistic(LogisticSpec {
            x: "x".into(), y: "y".into(), n_grid: 10, ci: None, max_iter: 25, tol: 1e-8, name: None,
        })),
        ("glm", Glm(GlmSpec {
            x: "x".into(), y: "y".into(), family: GlmFamily::Gaussian, link: GlmLink::Identity,
            n_grid: 10, ci: None, max_iter: 25, tol: 1e-8, name: None,
        })),
        ("robust", Robust(RobustSpec {
            x: "x".into(), y: "y".into(), n_grid: 10, ci: None, huber_c: 1.345, max_iter: 25,
            tol: 1e-8, output: SmoothOutput::Fitted, inject_zero_ref: false, inject_metrics: false,
            name: None, x_range: None,
        })),
        ("identity", Identity(IdentitySpec { name: None })),
        ("filter", Filter(FilterSpec { predicate: "x > 0".into(), name: None, param: None })),
        ("calculate", Calculate(CalculateSpec { as_field: "z".into(), expr: "x + y".into(), name: None })),
        ("fold", Fold(FoldSpec { fields: vec![], as_: ("key".into(), "value".into()), name: None })),
        ("pivot", Pivot(PivotSpec {
            field: "k".into(), value: "v".into(), groupby: None, limit: None, op: "sum".into(),
            name: None,
        })),
        ("join_aggregate", JoinAggregate(JoinAggregateSpec { aggregates: vec![], groupby: None, name: None })),
        ("data_window", DataWindow(DataWindowSpec {
            ops: vec![], sort: None, groupby: None, frame: None, ignore_peers: false, name: None,
        })),
        ("density_data", DensityData(DensityDataSpec {
            field: "v".into(), bandwidth: None, groupby: None, extent: None, steps: None,
            cumulative: false, as_: ("value".into(), "density".into()), name: None,
        })),
        ("regression_data", RegressionData(RegressionDataSpec {
            x: "x".into(), y: "y".into(), method: "linear".into(), order: 1, groupby: None,
            as_: ("x".into(), "y".into()), name: None,
        })),
        ("loess_data", LoessData(LoessDataSpec {
            x: "x".into(), y: "y".into(), bandwidth: 0.5, groupby: None,
            as_: ("x".into(), "y".into()), name: None,
        })),
        ("impute", Impute(ImputeSpec {
            field: "v".into(), method: "mean".into(), value: None, groupby: None, key: None, name: None,
        })),
        ("flatten", Flatten(FlattenSpec { fields: vec![], as_: None, name: None })),
        ("sample", Sample(SampleSpec { n: 10, seed: 0, name: None })),
        ("top_k", TopK(TopKSpec {
            n: 5, field: "v".into(), op: "sum".into(), sort: "descending".into(), name: None,
        })),
        ("data_stack", DataStack(DataStackSpec {
            field: "v".into(), groupby: vec![], sort: None, as_: ("y0".into(), "y1".into()),
            offset: "zero".into(), name: None,
        })),
        ("timeunit", TimeUnit(TimeUnitSpec {
            field: "t".into(), unit: "year".into(), utc: false, as_: None, name: None,
        })),
        ("data_bin", DataBin(DataBinSpec {
            field: "v".into(), as_: None, maxbins: None, step: None, nice: true, name: None,
            extent: None,
        })),
        ("data_aggregate", DataAggregate(DataAggregateSpec { aggregates: vec![], groupby: None, name: None })),
    ]
}

#[test]
fn all_transform_spec_instances_covers_every_kind() {
    let covered: std::collections::BTreeSet<&str> =
        all_transform_spec_instances().iter().map(|(k, _)| *k).collect();
    let all: std::collections::BTreeSet<&str> = all_transform_kinds().into_iter().collect();
    assert_eq!(
        covered, all,
        "all_transform_spec_instances() is missing or has extra kinds vs all_transform_kinds() \
         — a new TransformSpec variant needs an entry here too, or this wire-tag guard silently \
         stops covering it"
    );
}

#[test]
fn wire_tag_matches_registry_kind_or_named_exception() {
    // The named exception set (round-2 cycle-1 correction): registry `kind`
    // -> real serde wire tag, for the two kinds where they disagree. Every
    // OTHER kind must serialize a `"type"` identical to its registry kind —
    // a future kind added to `entry_for_kind` with a diverging wire tag and
    // no entry here fails this test, not just the module doc's prose.
    let known_exceptions: std::collections::BTreeMap<&str, &str> =
        [("timeunit", "time_unit"), ("kde_2d", "kde2_d")].into_iter().collect();

    for (kind, spec) in all_transform_spec_instances() {
        let value = serde_json::to_value(&spec).unwrap();
        let tag = value
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| panic!("{kind}: TransformSpec did not serialize a \"type\" tag: {value:?}"));
        let expected = known_exceptions.get(kind).copied().unwrap_or(kind);
        assert_eq!(
            tag, expected,
            "{kind}: real serde wire tag is '{tag}', expected '{expected}' (the registry kind, \
             or its named exception in `known_exceptions` above)"
        );
    }
}

// ── `params` drift guard (Required Fix 5, round-2 cycle-1) ─────────────────
//
// Ports the `scale_accepted_keys` precedent's drift guard
// (`spec/encoding.rs`'s `accepted_keys_for_scale_type_matches_every_variants_
// serialized_keys`) to `params`: every declared list must equal the real
// spec struct's serde-accepted field set, mined from serde's own
// `deny_unknown_fields` rejection message via the shared
// `chart_config::accepted_fields_from_deny_unknown_error` helper (all 42
// specs carry `#[serde(deny_unknown_fields)]`) — not a second hand-mirrored
// list. A single bogus-keyed JSON object triggers the rejection before
// serde ever checks for missing required fields (the unknown-field check
// runs per-key during the map-visiting loop, ahead of the end-of-loop
// required-field check), so this works uniformly regardless of how many
// fields a struct requires.

fn assert_params_match_serde_fields<T>(kind: &'static str) -> &'static str
where
    T: for<'de> serde::Deserialize<'de>,
{
    let res: Result<T, _> = serde_json::from_str(r#"{"__registry_drift_probe__": null}"#);
    let err = res.err().unwrap_or_else(|| panic!("{kind}: expected a deny_unknown_fields rejection"));
    let msg = err.to_string();
    let serde_fields = crate::render::chart_config::accepted_fields_from_deny_unknown_error(&msg);
    let declared: std::collections::BTreeSet<String> =
        entry_for_kind(kind).unwrap().params.iter().map(|s| s.to_string()).collect();
    assert_eq!(
        serde_fields, declared,
        "{kind}: registry `params` drifted from the real spec struct's serde-accepted fields.\n\
         in serde but not registry: {:?}\nin registry but not serde: {:?}",
        serde_fields.difference(&declared).collect::<Vec<_>>(),
        declared.difference(&serde_fields).collect::<Vec<_>>(),
    );
    kind
}

/// Required Fix 2 (round-3 cycle-2 review): pins this guard's own coverage
/// against [`super::all_transform_kinds`] — each
/// `assert_params_match_serde_fields` call site below returns the `kind` it
/// checked, and this test asserts that set equals every known kind, mirroring
/// `all_transform_spec_instances_covers_every_kind`'s pin for the wire-tag
/// guard. Without this, a new `TransformSpec` variant could land with no
/// `params` drift coverage and nothing would fail.
#[test]
fn params_match_spec_struct_serde_fields() {
    let covered: std::collections::BTreeSet<&str> = [
        assert_params_match_serde_fields::<crate::transform::bin::BinSpec>("bin"),
        assert_params_match_serde_fields::<crate::transform::bin_2d::Bin2DSpec>("bin_2d"),
        assert_params_match_serde_fields::<crate::transform::kde::KdeSpec>("kde"),
        assert_params_match_serde_fields::<crate::transform::smooth::SmoothSpec>("smooth"),
        assert_params_match_serde_fields::<crate::transform::aggregate::AggregateSpec>("aggregate"),
        assert_params_match_serde_fields::<crate::transform::summary::SummarySpec>("summary"),
        assert_params_match_serde_fields::<crate::transform::outliers::OutliersSpec>("outliers"),
        assert_params_match_serde_fields::<crate::transform::error_extent::ErrorExtentSpec>("error_extent"),
        assert_params_match_serde_fields::<crate::transform::box_stats::BoxStatsSpec>("box_stats"),
        assert_params_match_serde_fields::<crate::transform::violin::ViolinSpec>("violin"),
        assert_params_match_serde_fields::<crate::transform::kde_2d::Kde2DSpec>("kde_2d"),
        assert_params_match_serde_fields::<crate::transform::contour::ContourSpec>("contour"),
        assert_params_match_serde_fields::<crate::transform::qq::QQSpec>("qq"),
        assert_params_match_serde_fields::<crate::transform::linkage::LinkageSpec>("linkage"),
        assert_params_match_serde_fields::<crate::transform::raster::RasterSpec>("raster"),
        assert_params_match_serde_fields::<crate::transform::hex::HexSpec>("hex"),
        assert_params_match_serde_fields::<crate::transform::swarm::SwarmSpec>("swarm"),
        assert_params_match_serde_fields::<crate::transform::unpivot::UnpivotSpec>("unpivot"),
        assert_params_match_serde_fields::<crate::transform::reorder::ReorderSpec>("reorder"),
        assert_params_match_serde_fields::<crate::transform::reference_line::ReferenceLineSpec>("reference_line"),
        assert_params_match_serde_fields::<crate::transform::letter_value::LetterValueSpec>("letter_value"),
        assert_params_match_serde_fields::<crate::transform::logistic::LogisticSpec>("logistic"),
        assert_params_match_serde_fields::<crate::transform::glm::GlmSpec>("glm"),
        assert_params_match_serde_fields::<crate::transform::robust::RobustSpec>("robust"),
        assert_params_match_serde_fields::<crate::transform::identity::IdentitySpec>("identity"),
        assert_params_match_serde_fields::<crate::transform::filter::FilterSpec>("filter"),
        assert_params_match_serde_fields::<crate::transform::calculate::CalculateSpec>("calculate"),
        assert_params_match_serde_fields::<crate::transform::fold::FoldSpec>("fold"),
        assert_params_match_serde_fields::<crate::transform::pivot::PivotSpec>("pivot"),
        assert_params_match_serde_fields::<crate::transform::join_aggregate::JoinAggregateSpec>("join_aggregate"),
        assert_params_match_serde_fields::<crate::transform::data_window::DataWindowSpec>("data_window"),
        assert_params_match_serde_fields::<crate::transform::density_data::DensityDataSpec>("density_data"),
        assert_params_match_serde_fields::<crate::transform::regression_data::RegressionDataSpec>(
            "regression_data",
        ),
        assert_params_match_serde_fields::<crate::transform::loess_data::LoessDataSpec>("loess_data"),
        assert_params_match_serde_fields::<crate::transform::impute::ImputeSpec>("impute"),
        assert_params_match_serde_fields::<crate::transform::flatten::FlattenSpec>("flatten"),
        assert_params_match_serde_fields::<crate::transform::sample::SampleSpec>("sample"),
        assert_params_match_serde_fields::<crate::transform::top_k::TopKSpec>("top_k"),
        assert_params_match_serde_fields::<crate::transform::data_stack::DataStackSpec>("data_stack"),
        assert_params_match_serde_fields::<crate::transform::timeunit::TimeUnitSpec>("timeunit"),
        assert_params_match_serde_fields::<crate::transform::data_bin::DataBinSpec>("data_bin"),
        assert_params_match_serde_fields::<crate::transform::data_aggregate::DataAggregateSpec>(
            "data_aggregate",
        ),
    ]
    .into_iter()
    .collect();
    let all: std::collections::BTreeSet<&str> = all_transform_kinds().into_iter().collect();
    assert_eq!(
        covered, all,
        "params_match_spec_struct_serde_fields is missing (or has an extra) kind vs \
         all_transform_kinds() — a new TransformSpec variant needs an \
         `assert_params_match_serde_fields` call site here too, or this `params` drift guard \
         silently stops covering it"
    );
}

// ── `methods` drift guard (Required Fix 6, round-2 cycle-1) ────────────────
//
// For each spec §4.6-named vocabulary Python is deleting its own mirror of
// (`kde` kernels / `_VALID_KERNELS`, `qq` distributions, `smooth`/lmplot
// methods), calls through the REAL validation site — the PyO3 constructor,
// invoked the same way a Python caller would (via the class's type object,
// not by calling the module-private Rust `new()` directly) — for every
// declared method, and confirms a bogus one is refused. A method added to
// the real parser without a matching registry entry (or vice versa) fails
// here instead of leaving Python's derived validator silently stale.

#[test]
fn kde_methods_accepted_and_bogus_kernel_rejected() {
    attach(|py| {
        let ty = py.get_type::<crate::transform::kde::PyKde>();
        for &kernel in &entry_for_kind("kde").unwrap().methods {
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("kernel", kernel).unwrap();
            let res = ty.call(("v",), Some(&kwargs));
            assert!(res.is_ok(), "declared kde kernel '{kernel}' rejected by Kde(): {:?}", res.err());
        }
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("kernel", "not_a_real_kernel").unwrap();
        let res = ty.call(("v",), Some(&kwargs));
        assert!(res.is_err(), "bogus kde kernel must be rejected by Kde()");
    });
}

#[test]
fn qq_distributions_accepted_and_bogus_distribution_rejected() {
    attach(|py| {
        let ty = py.get_type::<crate::transform::qq::PyQq>();
        for &distribution in &entry_for_kind("qq").unwrap().methods {
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("distribution", distribution).unwrap();
            let res = ty.call(("v",), Some(&kwargs));
            assert!(res.is_ok(), "declared qq distribution '{distribution}' rejected by QQ(): {:?}", res.err());
        }
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("distribution", "not_a_real_distribution").unwrap();
        let res = ty.call(("v",), Some(&kwargs));
        assert!(res.is_err(), "bogus qq distribution must be rejected by QQ()");
    });
}

#[test]
fn smooth_methods_accepted_and_bogus_method_rejected() {
    attach(|py| {
        let ty = py.get_type::<crate::transform::smooth::PySmooth>();
        for &method in &entry_for_kind("smooth").unwrap().methods {
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("method", method).unwrap();
            let res = ty.call(("x", "y"), Some(&kwargs));
            assert!(res.is_ok(), "declared smooth method '{method}' rejected by Smooth(): {:?}", res.err());
        }
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("method", "not_a_real_method").unwrap();
        let res = ty.call(("x", "y"), Some(&kwargs));
        assert!(res.is_err(), "bogus smooth method must be rejected by Smooth()");
    });
}

/// Required Fix 3 (round-3 cycle-2 review): extends the `methods` drift guard
/// to `hex`. `HexSpec::new` (hex.rs) refuses a bogus `aggregate` hard, same
/// PyO3-constructor-time shape as `kde`/`qq`/`smooth`. `hex`'s eight
/// aggregates are also a live Python-mirror site (spec §4.6's target class):
/// `src/ferrum/marks/heavy_stat.py`'s `_VALID_AGGREGATES` names the identical
/// eight strings. Non-`count` aggregates require `field` to be set (checked
/// before the aggregate-name check would even run for them), so the fixture
/// supplies it for every declared value except `count`.
#[test]
fn hex_methods_accepted_and_bogus_aggregate_rejected() {
    attach(|py| {
        let ty = py.get_type::<crate::transform::hex::PyHex>();
        for &aggregate in &entry_for_kind("hex").unwrap().methods {
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("aggregate", aggregate).unwrap();
            if aggregate != "count" {
                kwargs.set_item("field", "v").unwrap();
            }
            let res = ty.call(("x", "y"), Some(&kwargs));
            assert!(res.is_ok(), "declared hex aggregate '{aggregate}' rejected by Hex(): {:?}", res.err());
        }
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("aggregate", "not_a_real_aggregate").unwrap();
        let res = ty.call(("x", "y"), Some(&kwargs));
        assert!(res.is_err(), "bogus hex aggregate must be rejected by Hex()");
    });
}

/// Required Fix 3 (round-3 cycle-2 review): extends the `methods` drift guard
/// to `raster`, same shape as `hex` above — `RasterSpec::new` (raster.rs)
/// refuses a bogus `aggregate` hard; `mean`/`sum` require `field`.
#[test]
fn raster_methods_accepted_and_bogus_aggregate_rejected() {
    attach(|py| {
        let ty = py.get_type::<crate::transform::raster::PyRaster>();
        for &aggregate in &entry_for_kind("raster").unwrap().methods {
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("aggregate", aggregate).unwrap();
            if matches!(aggregate, "mean" | "sum") {
                kwargs.set_item("field", "v").unwrap();
            }
            let res = ty.call(("x", "y"), Some(&kwargs));
            assert!(res.is_ok(), "declared raster aggregate '{aggregate}' rejected by Raster(): {:?}", res.err());
        }
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("aggregate", "not_a_real_aggregate").unwrap();
        let res = ty.call(("x", "y"), Some(&kwargs));
        assert!(res.is_err(), "bogus raster aggregate must be rejected by Raster()");
    });
}

/// Required Fix 4 (round-3 cycle-2 review): `linkage`'s `method` is a closed
/// 7-value vocabulary refused hard at `PyLinkage::new` (linkage.rs), the same
/// PyO3-constructor-time shape as `kde`/`qq`/`smooth`/`hex`/`raster` — see the
/// module doc's `methods` convention note on why it was declared this round
/// instead of staying `[]`. `PyLinkage::new`'s signature is all-keyword with
/// defaults, so the positional-args tuple is empty.
#[test]
fn linkage_methods_accepted_and_bogus_method_rejected() {
    attach(|py| {
        let ty = py.get_type::<crate::transform::linkage::PyLinkage>();
        for &method in &entry_for_kind("linkage").unwrap().methods {
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("method", method).unwrap();
            let res = ty.call((), Some(&kwargs));
            assert!(res.is_ok(), "declared linkage method '{method}' rejected by Linkage(): {:?}", res.err());
        }
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("method", "not_a_real_method").unwrap();
        let res = ty.call((), Some(&kwargs));
        assert!(res.is_err(), "bogus linkage method must be rejected by Linkage()");
    });
}
