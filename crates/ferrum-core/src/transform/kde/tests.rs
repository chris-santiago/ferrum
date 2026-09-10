//! `Kde` test coverage.
//!
//! Split out of `kde.rs` under the CLAUDE.md Rust test-module convention
//! ("promote to `mod.rs` + `tests.rs` when the test body would exceed the
//! source module") — batch F crossed that threshold for this module, taking it
//! to 1660 test lines against 775 source lines. This is a pure relocation of
//! the former inline `mod tests` block: no test was added, removed, renamed, or
//! rewritten in the move.
//!
//! What it carries, in file order: the JSON-fixture correctness suite
//! (`stat_refs.json`), the kernel table (one test per entry in `KERNELS`, plus
//! the dispatch-arm reachability guard), bandwidth/`bw_adjust` rules, grouped
//! output shape across every supported key dtype, serde round-trips and wire
//! back-compat, and the batch-F count-scale block — including the `Bin`-oracle
//! tests, which run the REAL `Bin` transform and compare against what it
//! actually rendered rather than re-deriving a width here.
//!
//! Everything under test is `pub(crate)` or private to the parent module, so
//! this coverage can only live in-crate; `use super::*` reaches all of it.

use super::*;
use arrow::array::{Array, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use serde::Deserialize;
use std::sync::Arc;

// `../fixtures/` because this module moved down one directory in the
// promotion to `kde/mod.rs` + `kde/tests.rs`; the fixture itself is shared
// with the other stat transforms and stays at `transform/fixtures/`.
const FIXTURES: &str = include_str!("../fixtures/stat_refs.json");

#[derive(Deserialize)]
struct KdeCase {
    name: String,
    input: Vec<f64>,
    bandwidth: String,
    #[serde(default)]
    fixed_bandwidth: Option<f64>,
    n: usize,
    extent: [f64; 2],
    cumulative: bool,
    expected_bandwidth: f64,
    value: Vec<f64>,
    density: Vec<f64>,
}

#[derive(Deserialize)]
struct Fixtures {
    kde: Vec<KdeCase>,
}

fn load_kde() -> Vec<KdeCase> {
    let f: Fixtures = serde_json::from_str(FIXTURES).unwrap();
    f.kde
}

fn batch_with(name: &str, values: Vec<f64>) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::Float64, true)]));
    RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(values))]).unwrap()
}

fn col(b: &RecordBatch, name: &str) -> Vec<f64> {
    let arr = b
        .column(b.schema().index_of(name).unwrap())
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    (0..arr.len())
        .map(|i| if arr.is_null(i) { f64::NAN } else { arr.value(i) })
        .collect()
}

#[test]
fn test_kde_against_fixtures_within_tolerance() {
    for case in load_kde() {
        let bw = match (case.bandwidth.as_str(), case.fixed_bandwidth) {
            ("scott", _) => BandwidthSpec::Scott,
            ("silverman", _) => BandwidthSpec::Silverman,
            ("fixed", Some(v)) => BandwidthSpec::Fixed { value: v },
            other => panic!("unknown bandwidth spec: {other:?}"),
        };
        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: bw,
            bw_adjust: 1.0,
            shared_extent: false,
            n: case.n,
            extent: Some((case.extent[0], case.extent[1])),
            cumulative: case.cumulative,
            kernel: default_kernel(),
            groupby: vec![],
            count_scale: None,
            count_scale_extent: None,
            name: None,
        };
        let batch = batch_with("x", case.input.clone());
        let out = apply(&spec, &batch).unwrap();
        let got_value = col(&out, "value");
        let got_density = col(&out, "density");
        for i in 0..case.n {
            assert!(
                (got_value[i] - case.value[i]).abs() < 1e-9,
                "case {} value[{i}]: got {} vs expected {}",
                case.name,
                got_value[i],
                case.value[i]
            );
            assert!(
                (got_density[i] - case.density[i]).abs() < 1e-6,
                "case {} density[{i}]: got {} vs expected {} (diff {})",
                case.name,
                got_density[i],
                case.density[i],
                (got_density[i] - case.density[i]).abs()
            );
        }
        // Suppress unused field warning for expected_bandwidth
        let _ = case.expected_bandwidth;
    }
}

#[test]
fn test_kde_zero_variance_emits_nan_densities() {
    let batch = batch_with("x", vec![3.0, 3.0, 3.0]);
    let spec = KdeSpec {
        field: "x".into(),
        bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            shared_extent: false,
        n: 16,
        extent: Some((0.0, 6.0)),
        cumulative: false,
        kernel: default_kernel(),
        groupby: vec![],
        count_scale: None,
        count_scale_extent: None,
        name: None,
    };
    let out = apply(&spec, &batch).unwrap();
    let density = col(&out, "density");
    assert!(density.iter().all(|d| d.is_nan()), "expected all-NaN densities");
}

#[test]
fn test_kde_n_lt_2_emits_nan_densities() {
    let batch = batch_with("x", vec![1.0]);
    let spec = KdeSpec {
        field: "x".into(),
        bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            shared_extent: false,
        n: 8,
        extent: Some((0.0, 2.0)),
        cumulative: false,
        kernel: default_kernel(),
        groupby: vec![],
        count_scale: None,
        count_scale_extent: None,
        name: None,
    };
    let out = apply(&spec, &batch).unwrap();
    let density = col(&out, "density");
    assert!(density.iter().all(|d| d.is_nan()));
}

#[test]
fn test_kde_grouped_preserves_group_column_and_per_group_density() {
    use arrow::array::StringArray;
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, true),
        Field::new("g", DataType::Utf8, false),
    ]));
    // Two groups, well-separated means so per-group bandwidth differs from
    // global bandwidth.
    let xs = Float64Array::from(vec![
        0.0, 0.5, 1.0, 1.5, 2.0,        // group A
        10.0, 10.5, 11.0, 11.5, 12.0,    // group B
    ]);
    let gs = StringArray::from(vec!["A", "A", "A", "A", "A",
                                      "B", "B", "B", "B", "B"]);
    let batch = RecordBatch::try_new(schema, vec![Arc::new(xs), Arc::new(gs)]).unwrap();
    let spec = KdeSpec {
        field: "x".into(),
        bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            shared_extent: false,
        n: 8,
        extent: None,
        cumulative: false,
        kernel: default_kernel(),
        groupby: vec!["g".into()],
        count_scale: None,
        count_scale_extent: None,
        name: None,
    };
    let out = apply(&spec, &batch).unwrap();
    // 2 groups × 8 grid points = 16 rows
    assert_eq!(out.num_rows(), 16);
    // Schema must include the group column as the 3rd field.
    let out_schema = out.schema();
    assert_eq!(out_schema.field(0).name(), "value");
    assert_eq!(out_schema.field(1).name(), "density");
    assert_eq!(out_schema.field(2).name(), "g");
    // Per-group grids span per-group extents.
    let values = col(&out, "value");
    let groups = out.column(2).as_any().downcast_ref::<StringArray>().unwrap();
    // Rows 0..8 are group A (values 0.0..2.0); rows 8..16 are group B (10.0..12.0).
    for i in 0..8 { assert_eq!(groups.value(i), "A"); }
    for i in 8..16 { assert_eq!(groups.value(i), "B"); }
    assert!((values[0] - 0.0).abs() < 1e-9, "group A grid starts at min");
    assert!((values[7] - 2.0).abs() < 1e-9, "group A grid ends at max");
    assert!((values[8] - 10.0).abs() < 1e-9, "group B grid starts at min");
    assert!((values[15] - 12.0).abs() < 1e-9, "group B grid ends at max");
}

#[test]
fn test_kde_grouped_int64_group_column() {
    // T0.3 regression: a non-Utf8 (Int64) group column must work, not raise
    // "must be Utf8". Pre-fix this errored in kde::apply_grouped.
    use arrow::array::Int64Array;
    pyo3::Python::initialize();
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, true),
        Field::new("g", DataType::Int64, false),
    ]));
    let xs = Float64Array::from(vec![
        0.0, 0.5, 1.0, 1.5, 2.0, // group 1
        10.0, 10.5, 11.0, 11.5, 12.0, // group 2
    ]);
    let gs = Int64Array::from(vec![1_i64, 1, 1, 1, 1, 2, 2, 2, 2, 2]);
    let batch = RecordBatch::try_new(schema, vec![Arc::new(xs), Arc::new(gs)]).unwrap();
    let spec = KdeSpec {
        field: "x".into(),
        bandwidth: BandwidthSpec::Scott,
        bw_adjust: 1.0,
        shared_extent: false,
        n: 8,
        extent: None,
        cumulative: false,
        kernel: default_kernel(),
        groupby: vec!["g".into()],
        count_scale: None,
        count_scale_extent: None,
        name: None,
    };
    let out = apply(&spec, &batch).expect("Int64 groupby must succeed");
    assert_eq!(out.num_rows(), 16, "2 groups × 8 grid points");
    // Group column must round-trip as Int64, NOT String.
    assert_eq!(out.schema().field(2).name(), "g");
    assert_eq!(out.schema().field(2).data_type(), &DataType::Int64);
    let groups = out.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
    for i in 0..8 {
        assert_eq!(groups.value(i), 1, "first 8 rows are group 1");
    }
    for i in 8..16 {
        assert_eq!(groups.value(i), 2, "last 8 rows are group 2");
    }
}

#[test]
fn test_kde_grouped_boolean_group_column() {
    // T0.3 regression: a Boolean group column must work (was "must be Utf8").
    use arrow::array::BooleanArray;
    pyo3::Python::initialize();
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, true),
        Field::new("g", DataType::Boolean, false),
    ]));
    let xs = Float64Array::from(vec![
        0.0, 0.5, 1.0, 1.5, 2.0, // false group
        10.0, 10.5, 11.0, 11.5, 12.0, // true group
    ]);
    let gs = BooleanArray::from(vec![
        false, false, false, false, false, true, true, true, true, true,
    ]);
    let batch = RecordBatch::try_new(schema, vec![Arc::new(xs), Arc::new(gs)]).unwrap();
    let spec = KdeSpec {
        field: "x".into(),
        bandwidth: BandwidthSpec::Scott,
        bw_adjust: 1.0,
        shared_extent: false,
        n: 8,
        extent: None,
        cumulative: false,
        kernel: default_kernel(),
        groupby: vec!["g".into()],
        count_scale: None,
        count_scale_extent: None,
        name: None,
    };
    let out = apply(&spec, &batch).expect("Boolean groupby must succeed");
    assert_eq!(out.num_rows(), 16);
    assert_eq!(out.schema().field(2).data_type(), &DataType::Boolean);
    let groups = out.column(2).as_any().downcast_ref::<BooleanArray>().unwrap();
    // First-appearance order → false group first, then true.
    for i in 0..8 {
        assert!(!groups.value(i), "first 8 rows are the false group");
    }
    for i in 8..16 {
        assert!(groups.value(i), "last 8 rows are the true group");
    }
}

#[test]
fn test_kde_ungrouped_output_schema_unchanged() {
    // Sentinel: ungrouped output schema MUST stay [value, density] (2 cols)
    // so existing goldens stay byte-identical.
    let batch = batch_with("x", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let spec = KdeSpec {
        field: "x".into(),
        bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            shared_extent: false,
        n: 4,
        extent: Some((0.0, 6.0)),
        cumulative: false,
        kernel: default_kernel(),
        groupby: vec![],
        count_scale: None,
        count_scale_extent: None,
        name: None,
    };
    let out = apply(&spec, &batch).unwrap();
    assert_eq!(out.num_columns(), 2);
    assert_eq!(out.schema().field(0).name(), "value");
    assert_eq!(out.schema().field(1).name(), "density");
}

#[test]
fn test_kde_grouped_missing_column_errors() {
    pyo3::Python::initialize();
    let batch = batch_with("x", vec![1.0, 2.0, 3.0]);
    let spec = KdeSpec {
        field: "x".into(),
        bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            shared_extent: false,
        n: 4,
        extent: None,
        cumulative: false,
        kernel: default_kernel(),
        groupby: vec!["ghost".into()],
        count_scale: None,
        count_scale_extent: None,
        name: None,
    };
    let err = apply(&spec, &batch).unwrap_err();
    assert!(err.to_string().contains("ghost"), "err: {err}");
}

#[test]
fn test_kde_int64_column_is_coerced() {
    // Int64 input must be coerced to Float64, not rejected. The density
    // output must match a Float64 batch of the same values bit-for-bit.
    use arrow::array::Int64Array;
    pyo3::Python::initialize();
    let vals_i: Vec<i64> = (0..10).collect();
    let vals_f: Vec<f64> = vals_i.iter().map(|&v| v as f64).collect();
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
    let int_batch =
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vals_i))]).unwrap();
    let float_batch = batch_with("x", vals_f);
    let spec = KdeSpec {
        field: "x".into(),
        bandwidth: BandwidthSpec::Scott,
        bw_adjust: 1.0,
        shared_extent: false,
        n: 16,
        extent: None,
        cumulative: false,
        kernel: default_kernel(),
        groupby: vec![],
        count_scale: None,
        count_scale_extent: None,
        name: None,
    };
    let int_out = apply(&spec, &int_batch).expect("Int64 KDE should succeed");
    let float_out = apply(&spec, &float_batch).unwrap();
    assert_eq!(int_out.num_rows(), 16);
    let id = col(&int_out, "density");
    let fd = col(&float_out, "density");
    let iv = col(&int_out, "value");
    let fv = col(&float_out, "value");
    for i in 0..16 {
        assert_eq!(iv[i].to_bits(), fv[i].to_bits(), "value[{i}] mismatch");
        assert_eq!(id[i].to_bits(), fd[i].to_bits(), "density[{i}] mismatch");
    }
}

#[test]
fn test_kde_int32_column_is_coerced() {
    // Prove general numeric coercion (not just Int64).
    use arrow::array::Int32Array;
    pyo3::Python::initialize();
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, true)]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from(vec![0_i32, 1, 2, 3, 4]))],
    )
    .unwrap();
    let spec = KdeSpec {
        field: "x".into(),
        bandwidth: BandwidthSpec::Scott,
        bw_adjust: 1.0,
        shared_extent: false,
        n: 8,
        extent: None,
        cumulative: false,
        kernel: default_kernel(),
        groupby: vec![],
        count_scale: None,
        count_scale_extent: None,
        name: None,
    };
    let out = apply(&spec, &batch).expect("Int32 KDE should succeed");
    assert_eq!(out.num_rows(), 8);
}

#[test]
fn test_kde_string_column_still_errors() {
    // Negative test: a genuinely non-numeric column must STILL error.
    use arrow::array::StringArray;
    pyo3::Python::initialize();
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Utf8, true)]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(StringArray::from(vec!["a", "b", "c"]))],
    )
    .unwrap();
    let spec = KdeSpec {
        field: "x".into(),
        bandwidth: BandwidthSpec::Scott,
        bw_adjust: 1.0,
        shared_extent: false,
        n: 8,
        extent: None,
        cumulative: false,
        kernel: default_kernel(),
        groupby: vec![],
        count_scale: None,
        count_scale_extent: None,
        name: None,
    };
    let err = apply(&spec, &batch).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("must be numeric"), "err: {msg}");
}

#[test]
fn test_kde_round_trip_json() {
    let original = KdeSpec {
        field: "x".into(),
        bandwidth: BandwidthSpec::Fixed { value: 0.5 },
        bw_adjust: 1.0,
        n: 32,
        extent: Some((-1.0, 5.0)),
        cumulative: true,
        shared_extent: false,
        kernel: "gaussian".to_string(),
        groupby: vec![],
        count_scale: None,
        count_scale_extent: None,
        name: None,
    };
    let json = serde_json::to_string(&original).unwrap();
    let parsed: KdeSpec = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, original);
}

#[test]
fn test_kde_single_row_emits_nan_densities() {
    // A single data point → n < 2, so Scott/Silverman bandwidth is undefined.
    // The transform should handle gracefully by emitting NaN densities.
    let batch = batch_with("x", vec![7.0]);
    let spec = KdeSpec {
        field: "x".into(),
        bandwidth: BandwidthSpec::Silverman,
        bw_adjust: 1.0,
        shared_extent: false,
        n: 10,
        extent: Some((5.0, 9.0)),
        cumulative: false,
        kernel: default_kernel(),
        groupby: vec![],
        count_scale: None,
        count_scale_extent: None,
        name: None,
    };
    let out = apply(&spec, &batch).unwrap();
    assert_eq!(out.num_rows(), 10);
    let density = col(&out, "density");
    assert!(
        density.iter().all(|d| d.is_nan()),
        "single-row KDE should emit all-NaN densities"
    );
}

#[test]
fn test_kde_all_identical_values_emits_nan_densities() {
    // All values identical → std=0 → bandwidth=0 (both Scott and Silverman).
    // This is effectively the zero-variance case.
    let batch = batch_with("x", vec![4.0; 20]);
    let spec = KdeSpec {
        field: "x".into(),
        bandwidth: BandwidthSpec::Silverman,
        bw_adjust: 1.0,
        shared_extent: false,
        n: 16,
        extent: Some((2.0, 6.0)),
        cumulative: false,
        kernel: default_kernel(),
        groupby: vec![],
        count_scale: None,
        count_scale_extent: None,
        name: None,
    };
    let out = apply(&spec, &batch).unwrap();
    let density = col(&out, "density");
    assert!(
        density.iter().all(|d| d.is_nan()),
        "zero-variance KDE should emit all-NaN densities"
    );
}

// ---- Kernel-specific tests ----

fn kde_spec_with_kernel(kernel: &str, n: usize) -> KdeSpec {
    KdeSpec {
        field: "x".into(),
        bandwidth: BandwidthSpec::Fixed { value: 1.0 },
        bw_adjust: 1.0,
        shared_extent: false,
        n,
        extent: Some((-3.0, 3.0)),
        cumulative: false,
        kernel: kernel.to_string(),
        groupby: vec![],
        count_scale: None,
        count_scale_extent: None,
        name: None,
    }
}

fn simple_batch() -> RecordBatch {
    batch_with("x", vec![0.0, 0.5, -0.5, 1.0, -1.0])
}

/// Gaussian kernel (default) must produce non-zero density everywhere in the grid.
#[test]
fn test_kde_gaussian_produces_nonzero_density() {
    let batch = simple_batch();
    let spec = kde_spec_with_kernel("gaussian", 20);
    let out = apply(&spec, &batch).unwrap();
    let density = col(&out, "density");
    assert!(
        density.iter().all(|&d| d > 0.0),
        "Gaussian KDE: all grid densities should be positive"
    );
}

/// Epanechnikov kernel must produce non-zero density on simple data.
#[test]
fn test_kde_epanechnikov_produces_nonzero_density() {
    let batch = simple_batch();
    let spec = kde_spec_with_kernel("epanechnikov", 20);
    let out = apply(&spec, &batch).unwrap();
    let density = col(&out, "density");
    // At least some grid points must be non-zero (those inside bandwidth support).
    let nonzero_count = density.iter().filter(|&&d| d > 0.0).count();
    assert!(
        nonzero_count > 0,
        "Epanechnikov KDE: expected some non-zero densities, got all zeros"
    );
}

/// Epanechnikov kernel must produce exactly 0 for grid points further than
/// bandwidth h from all data points.
#[test]
fn test_kde_epanechnikov_zero_outside_bandwidth() {
    // Single data point at x=0, bandwidth=0.5.
    // Grid points beyond |g - 0| > 0.5 must be exactly 0.
    let batch = batch_with("x", vec![0.0, 0.01]); // need ≥ 2 points
    let spec = KdeSpec {
        field: "x".into(),
        bandwidth: BandwidthSpec::Fixed { value: 0.5 },
        bw_adjust: 1.0,
        shared_extent: false,
        n: 21,
        extent: Some((-2.0, 2.0)),
        cumulative: false,
        kernel: "epanechnikov".to_string(),
        groupby: vec![],
        count_scale: None,
        count_scale_extent: None,
        name: None,
    };
    let out = apply(&spec, &batch).unwrap();
    let values = col(&out, "value");
    let density = col(&out, "density");
    // All grid points where |g| > 0.5 AND |g - 0.01| > 0.5 must have density = 0.
    for (i, (&v, &d)) in values.iter().zip(density.iter()).enumerate() {
        let in_support = v.abs() <= 0.5 || (v - 0.01).abs() <= 0.5;
        if !in_support {
            assert_eq!(
                d, 0.0,
                "Epanechnikov: expected density=0 at grid point {i} (x={v}), got {d}"
            );
        }
    }
}

/// "epan" alias must produce the same output as "epanechnikov".
#[test]
fn test_kde_epan_alias_matches_epanechnikov() {
    let batch = simple_batch();
    let spec1 = kde_spec_with_kernel("epanechnikov", 15);
    let spec2 = kde_spec_with_kernel("epan", 15);
    let out1 = apply(&spec1, &batch).unwrap();
    let out2 = apply(&spec2, &batch).unwrap();
    let d1 = col(&out1, "density");
    let d2 = col(&out2, "density");
    for (a, b) in d1.iter().zip(d2.iter()) {
        assert_eq!(
            a.to_bits(), b.to_bits(),
            "'epan' and 'epanechnikov' must produce identical output"
        );
    }
}

/// Tophat kernel must produce non-zero density on simple data.
#[test]
fn test_kde_tophat_produces_nonzero_density() {
    let batch = simple_batch();
    let spec = kde_spec_with_kernel("tophat", 20);
    let out = apply(&spec, &batch).unwrap();
    let density = col(&out, "density");
    let nonzero_count = density.iter().filter(|&&d| d > 0.0).count();
    assert!(
        nonzero_count > 0,
        "Tophat KDE: expected some non-zero densities"
    );
}

/// "uniform" alias must produce the same output as "tophat".
#[test]
fn test_kde_uniform_alias_matches_tophat() {
    let batch = simple_batch();
    let spec1 = kde_spec_with_kernel("tophat", 15);
    let spec2 = kde_spec_with_kernel("uniform", 15);
    let out1 = apply(&spec1, &batch).unwrap();
    let out2 = apply(&spec2, &batch).unwrap();
    let d1 = col(&out1, "density");
    let d2 = col(&out2, "density");
    for (a, b) in d1.iter().zip(d2.iter()) {
        assert_eq!(
            a.to_bits(), b.to_bits(),
            "'uniform' and 'tophat' must produce identical output"
        );
    }
}

/// Cosine kernel must produce non-zero density on simple data.
#[test]
fn test_kde_cosine_produces_nonzero_density() {
    let batch = simple_batch();
    let spec = kde_spec_with_kernel("cosine", 20);
    let out = apply(&spec, &batch).unwrap();
    let density = col(&out, "density");
    let nonzero_count = density.iter().filter(|&&d| d > 0.0).count();
    assert!(
        nonzero_count > 0,
        "Cosine KDE: expected some non-zero densities"
    );
}

// ---- T11: exponential + linear kernels ----

/// Exponential kernel must produce non-zero density on simple data.
#[test]
fn test_kde_exponential_produces_nonzero_density() {
    let batch = simple_batch();
    let spec = kde_spec_with_kernel("exponential", 20);
    let out = apply(&spec, &batch).unwrap();
    let density = col(&out, "density");
    assert!(
        density.iter().all(|&d| d > 0.0),
        "Exponential KDE: unbounded support means every grid point should be non-zero"
    );
}

/// Exponential kernel discriminator: K(u) = exp(-|u|)/2 evaluated exactly
/// against a single data point at 0, h=1. This pins the dedicated formula
/// (not a silent Gaussian fallback, which would give ~0.3989 at u=0
/// instead of exactly 0.5) and its "cusp" property — the value decays by
/// the SAME multiplicative factor e^-1 per unit step on both sides of the
/// peak, unlike Gaussian's super-exponential (accelerating) falloff.
#[test]
fn test_kde_exponential_matches_exact_formula_and_has_cusp() {
    let x = vec![0.0];
    let grid = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let h = 1.0;
    let got = exponential_density_at_grid(&x, h, &grid);
    let want: Vec<f64> = grid.iter().map(|&u| 0.5 * (-(u.abs())).exp()).collect();
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!((g - w).abs() < 1e-12, "exponential[{i}]: got {g}, want {w}");
    }
    // Gaussian's value at u=0 is 1/sqrt(2*pi) ≈ 0.3989, not 0.5 — proves
    // this isn't the Gaussian formula under a different name.
    assert!((got[2] - 0.5).abs() < 1e-12, "exponential K(0) must be exactly 0.5");
    // Cusp: constant ratio e across the peak on both sides (linear decay
    // in log-space), which a smooth (Gaussian) peak does not exhibit.
    assert!((got[2] / got[1] - std::f64::consts::E).abs() < 1e-9);
    assert!((got[2] / got[3] - std::f64::consts::E).abs() < 1e-9);
}

/// Linear kernel must produce non-zero density on simple data.
#[test]
fn test_kde_linear_produces_nonzero_density() {
    let batch = simple_batch();
    let spec = kde_spec_with_kernel("linear", 20);
    let out = apply(&spec, &batch).unwrap();
    let density = col(&out, "density");
    let nonzero_count = density.iter().filter(|&&d| d > 0.0).count();
    assert!(nonzero_count > 0, "Linear KDE: expected some non-zero densities");
}

/// Linear kernel discriminator: triangular K(u) = 1 - |u| for |u| <= 1,
/// with a HARD edge (exact 0) at and beyond |u| = 1 — unlike
/// Epanechnikov's parabolic taper (which is also compactly supported but
/// non-linear) and unlike exponential/Gaussian's unbounded tails.
#[test]
fn test_kde_linear_produces_hard_edged_triangular_shape() {
    let x = vec![0.0];
    let grid = vec![-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5];
    let h = 1.0;
    let got = linear_density_at_grid(&x, h, &grid);
    let want = [0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0];
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!((g - w).abs() < 1e-12, "linear[{i}]: got {g}, want {w}");
    }
    // Discriminator vs Epanechnikov: at u=0.5, Epanechnikov gives
    // 0.75*(1-0.25) = 0.5625, not the triangular kernel's exact 0.5.
    let epan_at_half = epanechnikov_density_at_grid(&x, h, &[0.5]);
    assert!(
        (epan_at_half[0] - 0.5625).abs() < 1e-12,
        "sanity: epanechnikov(0.5) should be 0.5625, got {}",
        epan_at_half[0]
    );
    assert!(
        (got[4] - epan_at_half[0]).abs() > 1e-9,
        "linear and epanechnikov must diverge in shape away from the peak"
    );
}

/// Before this task, `kernel_density_at_grid`'s catch-all silently
/// substituted Gaussian for any unrecognised kernel string. A `KdeSpec`
/// built directly (bypassing `PyKde::new`'s validation — e.g. a spec
/// deserialized from wire JSON with a typo'd kernel) must now error at
/// `apply()` time instead of silently rendering the wrong shape.
#[test]
fn test_kde_apply_with_unvalidated_bogus_kernel_now_errors() {
    pyo3::Python::initialize();
    let batch = batch_with("x", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let spec = KdeSpec {
        field: "x".into(),
        bandwidth: BandwidthSpec::Fixed { value: 1.0 },
        bw_adjust: 1.0,
        shared_extent: false,
        n: 8,
        extent: Some((0.0, 6.0)),
        cumulative: false,
        kernel: "not_a_real_kernel".into(),
        groupby: vec![],
        count_scale: None,
        count_scale_extent: None,
        name: None,
    };
    let err = apply(&spec, &batch).unwrap_err();
    assert!(err.to_string().contains("unknown kernel"), "err: {err}");
}

/// Reverse guard (round 2, batch-D T11 cycle 2): every `KERNELS` entry
/// must reach a real dispatch arm in `kernel_density_at_grid`, not just
/// pass `PyKde::new`'s `KERNELS.contains` pre-check — that pre-check
/// only proves the FORWARD direction (a dispatch arm without a `KERNELS`
/// entry is unreachable at construction time), and it never calls
/// `kernel_density_at_grid` itself, so a name added to `KERNELS` with no
/// matching arm would otherwise pass both the pre-check and this test
/// suite and only fail at render time in a user's chart. Mirrors
/// `smooth::test_all_method_aliases_accepted_by_pysmooth_new`, adapted to
/// call the actual compute site directly since (unlike `Smooth`) `Kde`'s
/// constructor pre-check is not itself the dispatch.
#[test]
fn test_all_kernels_reach_a_dispatch_arm() {
    for &kernel in KERNELS {
        let result = kernel_density_at_grid(&[0.0, 1.0], 1.0, &[0.0], kernel);
        assert!(
            result.is_ok(),
            "KERNELS entry {kernel:?} has no matching dispatch arm in \
             kernel_density_at_grid: {:?}",
            result.err().map(|e| e.to_string())
        );
    }
}

/// Determinism pin (round 2, batch-D T11 cycle 2): the two new kernels
/// must produce byte-identical output across repeated `apply()` calls on
/// identical input, per the spec's testing section ("determinism: two
/// runs byte-equal"). Rust owns this pin — see the round-2 report for
/// why it isn't duplicated Python-side.
#[test]
fn test_kde_exponential_and_linear_apply_deterministic_across_runs() {
    for kernel in ["exponential", "linear"] {
        let batch = simple_batch();
        let spec = kde_spec_with_kernel(kernel, 50);
        let out1 = apply(&spec, &batch).unwrap();
        let out2 = apply(&spec, &batch).unwrap();
        let d1 = col(&out1, "density");
        let d2 = col(&out2, "density");
        assert_eq!(d1.len(), d2.len());
        for (i, (a, b)) in d1.iter().zip(d2.iter()).enumerate() {
            assert_eq!(
                a.to_bits(), b.to_bits(),
                "kernel={kernel} density[{i}] not byte-identical across runs: {a} vs {b}"
            );
        }
    }
}

/// Gaussian must produce the same output as before (regression against the
/// hardcoded path we replaced with the dispatch).
#[test]
fn test_kde_gaussian_regression_matches_direct_call() {
    let data = vec![-1.0, 0.0, 1.0, 2.0, 3.0];
    let h = 0.8_f64;
    let grid: Vec<f64> = (0..10).map(|i| -2.0 + i as f64 * 0.5).collect();
    let expected = gaussian_density_at_grid(&data, h, &grid);
    let via_dispatch = kernel_density_at_grid(&data, h, &grid, "gaussian").unwrap();
    for (a, b) in expected.iter().zip(via_dispatch.iter()) {
        assert_eq!(
            a.to_bits(), b.to_bits(),
            "gaussian dispatch must be byte-identical to direct call"
        );
    }
}

/// Unknown kernel name in PyKde::new must return an error.
#[test]
fn test_kde_unknown_kernel_returns_error() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        use pyo3::types::PyString;
        let bw_obj = PyString::new(py, "scott").into_any();
        let result = PyKde::new(
            "x",
            Some(&bw_obj),
            1.0,
            512,
            None,
            false,
            false,
            "laplacian",
            None,
            None,
            None,
        );
        assert!(result.is_err(), "unknown kernel 'laplacian' must return an error");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("unknown kernel"),
            "error message should mention 'unknown kernel': {msg}"
        );
    });
}

/// Gaussian kernel must round-trip through JSON including the kernel field.
#[test]
fn test_kde_kernel_field_round_trips_json() {
    let original = KdeSpec {
        field: "x".into(),
        bandwidth: BandwidthSpec::Scott,
        bw_adjust: 1.0,
        n: 64,
        extent: None,
        cumulative: false,
        shared_extent: false,
        kernel: "epanechnikov".to_string(),
        groupby: vec![],
        count_scale: None,
        count_scale_extent: None,
        name: None,
    };
    let json = serde_json::to_string(&original).unwrap();
    let parsed: KdeSpec = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.kernel, "epanechnikov");
    assert_eq!(parsed, original);
}

/// Old JSON without a "kernel" field must deserialize to gaussian (backward compat).
#[test]
fn test_kde_kernel_field_defaults_to_gaussian_on_old_json() {
    // Simulate a serialized KdeSpec without the "kernel" field.
    let json = r#"{"field":"x","bandwidth":{"kind":"scott"},"bw_adjust":1.0,"n":32}"#;
    let parsed: KdeSpec = serde_json::from_str(json).unwrap();
    assert_eq!(
        parsed.kernel, "gaussian",
        "missing 'kernel' field must default to 'gaussian' for backward compat"
    );
}

// ── Task 8: global_extent helper ─────────────────────────────────────────

/// `global_extent` returns the full-data range over the KDE field.
#[test]
fn test_kde_global_extent_returns_full_range() {
    let batch = batch_with("x", vec![3.0, 1.0, 5.0, 2.0, 4.0]);
    let spec = KdeSpec {
        field: "x".into(),
        bandwidth: BandwidthSpec::Scott,
        bw_adjust: 1.0,
        n: 32,
        extent: None,
        cumulative: false,
        shared_extent: false,
        kernel: default_kernel(),
        groupby: vec![],
        count_scale: None,
        count_scale_extent: None,
        name: None,
    };
    let ext = super::global_extent(&spec, &batch);
    assert_eq!(ext, Some((1.0, 5.0)), "global_extent should be (min, max) = (1.0, 5.0)");
}

/// `global_extent` handles integer columns via coerce_to_float64.
#[test]
fn test_kde_global_extent_integer_column() {
    use arrow::array::Int64Array;
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int64Array::from(vec![10_i64, 20, 30]))],
    ).unwrap();
    let spec = KdeSpec {
        field: "x".into(),
        bandwidth: BandwidthSpec::Scott,
        bw_adjust: 1.0,
        n: 16,
        extent: None,
        cumulative: false,
        shared_extent: false,
        kernel: default_kernel(),
        groupby: vec![],
        count_scale: None,
        count_scale_extent: None,
        name: None,
    };
    let ext = super::global_extent(&spec, &batch);
    assert_eq!(ext, Some((10.0, 30.0)), "global_extent on Int64 column should return (10.0, 30.0)");
}

/// `global_extent` returns None on missing field.
#[test]
fn test_kde_global_extent_missing_field_returns_none() {
    let batch = batch_with("x", vec![1.0, 2.0, 3.0]);
    let spec = KdeSpec {
        field: "nonexistent".into(),
        bandwidth: BandwidthSpec::Scott,
        bw_adjust: 1.0,
        n: 16,
        extent: None,
        cumulative: false,
        shared_extent: false,
        kernel: default_kernel(),
        groupby: vec![],
        count_scale: None,
        count_scale_extent: None,
        name: None,
    };
    let ext = super::global_extent(&spec, &batch);
    assert_eq!(ext, None, "global_extent on missing field must return None");
}

/// `global_extent` returns None when all values are null or NaN.
#[test]
fn test_kde_global_extent_all_null_returns_none() {
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, true)]));
    let arr = Float64Array::from(vec![None::<f64>, None, None]);
    let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
    let spec = KdeSpec {
        field: "x".into(),
        bandwidth: BandwidthSpec::Scott,
        bw_adjust: 1.0,
        n: 16,
        extent: None,
        cumulative: false,
        shared_extent: false,
        kernel: default_kernel(),
        groupby: vec![],
        count_scale: None,
        count_scale_extent: None,
        name: None,
    };
    let ext = super::global_extent(&spec, &batch);
    assert_eq!(ext, None, "global_extent on all-null field must return None");
}

// ── T3.8a regression tests ────────────────────────────────────────────────

/// D-GROUPBY-1 regression (T3.8a): a 2-column groupby on `Kde` must produce
/// `n` density grid points per distinct `(g1, g2)` combination, with both
/// group columns appended after the standard 2 kde output columns.
#[test]
fn kde_two_column_groupby_produces_per_combination_rows() {
    use arrow::array::StringArray;
    pyo3::Python::initialize();
    // 8 rows: two g1 × two g2 × 2 x-values each.
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, false),
        Field::new("g1", DataType::Utf8, false),
        Field::new("g2", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![
                1.0, 9.0, 1.0, 9.0, 1.0, 9.0, 1.0, 9.0,
            ])) as arrow::array::ArrayRef,
            Arc::new(StringArray::from(vec!["A", "A", "A", "A", "B", "B", "B", "B"]))
                as arrow::array::ArrayRef,
            Arc::new(StringArray::from(vec!["x", "x", "y", "y", "x", "x", "y", "y"]))
                as arrow::array::ArrayRef,
        ],
    )
    .unwrap();

    let n = 8_usize;
    let spec = KdeSpec {
        field: "x".into(),
        bandwidth: BandwidthSpec::Scott,
        bw_adjust: 1.0,
        n,
        extent: None,
        cumulative: false,
        shared_extent: false,
        kernel: default_kernel(),
        groupby: vec!["g1".into(), "g2".into()],
        count_scale: None,
        count_scale_extent: None,
        name: None,
    };
    let out = apply(&spec, &batch).unwrap();

    // Schema: value, density, g1, g2.
    let s = out.schema();
    assert!(s.index_of("value").is_ok(), "missing value");
    assert!(s.index_of("density").is_ok(), "missing density");
    assert!(s.index_of("g1").is_ok(), "missing g1");
    assert!(s.index_of("g2").is_ok(), "missing g2");

    // 4 distinct (g1,g2) groups × n grid points each.
    assert_eq!(
        out.num_rows(),
        4 * n,
        "expected 4 groups × {n} grid points = {} rows, got {}",
        4 * n,
        out.num_rows()
    );

    // Collect distinct (g1,g2) combos in the output.
    let g1_col = out.column(s.index_of("g1").unwrap())
        .as_any().downcast_ref::<StringArray>().unwrap();
    let g2_col = out.column(s.index_of("g2").unwrap())
        .as_any().downcast_ref::<StringArray>().unwrap();
    let combos: std::collections::BTreeSet<(String, String)> = (0..out.num_rows())
        .map(|i| (g1_col.value(i).to_string(), g2_col.value(i).to_string()))
        .collect();
    assert_eq!(combos.len(), 4, "must have 4 distinct (g1,g2) pairs");
    assert!(combos.contains(&("A".into(), "x".into())));
    assert!(combos.contains(&("A".into(), "y".into())));
    assert!(combos.contains(&("B".into(), "x".into())));
    assert!(combos.contains(&("B".into(), "y".into())));
}

/// Close-remediation F1 (batch-F rust design review): the repeated-name
/// refusal is a property of the groupby REQUEST, not of one stacking
/// strategy, so it must fire for every grouped transform — including the
/// ones that never touch `group_key::stack_grouped_batches`. `Kde` is the
/// non-stack consumer that pins the widened coverage: the reviewer's live
/// pre-fix probe had `Smooth(groupby=["g","g"])` refusing while
/// `Kde(field="v", groupby=["g","g"])` rendered happily and emitted a
/// batch with two identically-named `g` fields, which `Schema::index_of`
/// then silently resolves to the FIRST of. The error must still name the
/// calling transform (`stat_kde`), not the shared helper.
#[test]
fn kde_grouped_duplicate_groupby_name_refuses() {
    use arrow::array::StringArray;
    pyo3::Python::initialize();
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, true),
        Field::new("g", DataType::Utf8, false),
    ]));
    let xs = Float64Array::from(vec![0.0, 0.5, 1.0, 10.0, 10.5, 11.0]);
    let gs = StringArray::from(vec!["A", "A", "A", "B", "B", "B"]);
    let batch = RecordBatch::try_new(schema, vec![Arc::new(xs), Arc::new(gs)]).unwrap();
    let spec = KdeSpec {
        groupby: vec!["g".into(), "g".into()],
        ..count_scale_spec("x", 8, (0.0, 11.0), None)
    };
    let err = apply(&spec, &batch)
        .expect_err("groupby=['g', 'g'] must refuse rather than emit two 'g' fields");
    let msg = err.to_string();
    assert!(msg.contains('g'), "error must name the repeated column: {msg}");
    assert!(msg.contains("repeated"), "error must explain the duplication: {msg}");
    assert!(
        msg.contains("stat_kde"),
        "error must name the calling transform, not the shared helper: {msg}"
    );
}

/// D-GROUPBY-1 regression (T3.8a): the KdeSpec serde round-trip preserves
/// the canonical array wire form; deserializing the old bare-string form
/// yields a one-element vec identical to `vec!["col"]`.
#[test]
fn kde_groupby_serde_round_trip_and_back_compat() {
    // Minimal valid KdeSpec JSON (bandwidth and n are required fields).
    // "bandwidth":{"kind":"scott"} is the Scott variant serialized by serde.
    let base_json =
        r#"{"field":"x","bandwidth":{"kind":"scott"},"n":64,"groupby":"col"}"#;

    let spec = KdeSpec {
        field: "x".into(),
        bandwidth: BandwidthSpec::Scott,
        bw_adjust: 1.0,
        n: 64,
        extent: None,
        cumulative: false,
        shared_extent: false,
        kernel: default_kernel(),
        groupby: vec!["col".into()],
        count_scale: None,
        count_scale_extent: None,
        name: None,
    };

    // Serialize: groupby=["col"] must use array form.
    let json = serde_json::to_string(&spec).unwrap();
    assert!(json.contains(r#""groupby":["col"]"#), "must serialize as array, got: {json}");

    // Legacy bare-string deserializes to vec!["col"].
    let from_legacy: KdeSpec = serde_json::from_str(base_json).unwrap();
    assert_eq!(from_legacy.groupby, vec!["col"], "bare-string must parse to [\"col\"]");

    // Legacy null deserializes to [].
    let null_json =
        r#"{"field":"x","bandwidth":{"kind":"scott"},"n":64,"groupby":null}"#;
    let from_null: KdeSpec = serde_json::from_str(null_json).unwrap();
    assert!(from_null.groupby.is_empty(), "null groupby must parse to []");

    // Empty groupby is omitted from serialization.
    let empty = KdeSpec { groupby: vec![], ..spec.clone() };
    let empty_json = serde_json::to_string(&empty).unwrap();
    assert!(!empty_json.contains("groupby"), "empty groupby must be omitted: {empty_json}");
}

// ── Batch F (F-L09-10): count-scale ─────────────────────────────────────

fn count_scale_spec(field: &str, n: usize, extent: (f64, f64), count_scale: Option<BinMode>) -> KdeSpec {
    KdeSpec {
        field: field.into(),
        bandwidth: BandwidthSpec::Scott,
        bw_adjust: 1.0,
        n,
        extent: Some(extent),
        cumulative: false,
        shared_extent: false,
        kernel: default_kernel(),
        groupby: vec![],
        count_scale,
        count_scale_extent: None,
        name: None,
    }
}

/// Wire-default round-trip: `count_scale` is omitted from serialized JSON
/// when absent (matching every other `Option` field on `KdeSpec`), old
/// JSON with no `count_scale` key deserializes to `None` (today's wire
/// specs stay byte-identical), and a `Some` value round-trips through its
/// tagged `BinMode` shape unchanged.
#[test]
fn test_kde_count_scale_wire_default_round_trip() {
    let unscaled = count_scale_spec("x", 8, (0.0, 10.0), None);
    let json = serde_json::to_string(&unscaled).unwrap();
    assert!(!json.contains("count_scale"), "absent count_scale must be omitted: {json}");
    let parsed: KdeSpec = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, unscaled);

    // Old JSON (pre-batch-F) with no count_scale key must deserialize to None.
    let old_json = r#"{"field":"x","bandwidth":{"kind":"scott"},"n":8,"extent":[0.0,10.0]}"#;
    let from_old: KdeSpec = serde_json::from_str(old_json).unwrap();
    assert_eq!(from_old.count_scale, None, "missing count_scale key must default to None");

    // A `Some` value round-trips through its tagged BinMode wire shape.
    let scaled = count_scale_spec("x", 8, (0.0, 10.0), Some(BinMode::Fixed { n: 5 }));
    let scaled_json = serde_json::to_string(&scaled).unwrap();
    assert!(
        scaled_json.contains(r#""count_scale":{"kind":"fixed","n":5}"#),
        "expected tagged BinMode shape, got: {scaled_json}"
    );
    let scaled_parsed: KdeSpec = serde_json::from_str(&scaled_json).unwrap();
    assert_eq!(scaled_parsed, scaled);
}

/// Absent `count_scale` (the default) must leave `density` EXACTLY the
/// raw kernel-density output — a real byte-identity pin against an
/// independently-derived oracle, not a determinism check. Cross-checks
/// `apply()`'s output against a direct call to `gaussian_density_at_grid`
/// (the same primitive `apply_one_group` dispatches to for a fixed
/// bandwidth) computed OUTSIDE `apply()`, so a regression that
/// spuriously applies ANY scaling factor when `count_scale=None` would
/// be caught — unlike the prior `apply(&spec)` vs `apply(&spec)`
/// self-comparison, which could never fail regardless of what `apply()`
/// computed.
#[test]
fn test_kde_count_scale_absent_matches_raw_kernel_density_exactly() {
    let x = vec![0.0, 1.0, 2.0, 4.0];
    let h = 1.0;
    let batch = batch_with("x", x.clone());
    let spec = KdeSpec {
        field: "x".into(),
        bandwidth: BandwidthSpec::Fixed { value: h },
        bw_adjust: 1.0,
        n: 6,
        extent: Some((0.0, 4.0)),
        cumulative: false,
        shared_extent: false,
        kernel: default_kernel(),
        groupby: vec![],
        count_scale: None,
        count_scale_extent: None,
        name: None,
    };
    let out = apply(&spec, &batch).unwrap();
    let got_value = col(&out, "value");
    let got_density = col(&out, "density");

    // Independently-computed oracle: the raw kernel density on the
    // ACTUAL returned grid, called directly (bypassing `apply()`'s own
    // count-scale dispatch entirely).
    let expected_density = gaussian_density_at_grid(&x, h, &got_value);
    for (i, (g, e)) in got_density.iter().zip(expected_density.iter()).enumerate() {
        assert_eq!(
            g.to_bits(), e.to_bits(),
            "point {i}: count_scale=None must be the RAW kernel density, got {g} vs expected {e}"
        );
    }
}

/// Ungrouped count-scale: `density × n × bin_width` must have bit-parity
/// with `Bin`'s OWN bin width for the same field + bin mode — not a
/// hand-derived approximation. Runs the real `Bin` transform on the same
/// batch/field/mode and cross-checks the scaling factor against its
/// actual `bin_end - bin_start`.
#[test]
fn test_kde_count_scale_ungrouped_scales_by_n_times_bin_width() {
    let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let batch = batch_with("x", values.clone());

    // The real Bin transform, same field + mode, to get the ACTUAL bin width.
    let bin_spec = crate::transform::bin::BinSpec {
        field: "x".into(),
        mode: BinMode::Fixed { n: 5 },
        extent: None,
        nice: true,
        cumulative: false,
        shared_extent: false,
        groupby: vec![],
        agg_field: None,
        agg: None,
        name: None,
    };
    let bin_out = crate::transform::bin::apply(&bin_spec, &batch).unwrap();
    let bin_starts = bin_out
        .column(bin_out.schema().index_of("bin_start").unwrap())
        .as_any().downcast_ref::<Float64Array>().unwrap();
    let bin_ends = bin_out
        .column(bin_out.schema().index_of("bin_end").unwrap())
        .as_any().downcast_ref::<Float64Array>().unwrap();
    let expected_bin_width = bin_ends.value(0) - bin_starts.value(0);

    let unscaled = count_scale_spec("x", 12, (0.0, 19.0), None);
    let scaled = count_scale_spec("x", 12, (0.0, 19.0), Some(BinMode::Fixed { n: 5 }));
    let out_unscaled = apply(&unscaled, &batch).unwrap();
    let out_scaled = apply(&scaled, &batch).unwrap();
    let d_unscaled = col(&out_unscaled, "density");
    let d_scaled = col(&out_scaled, "density");

    let n = values.len() as f64;
    for (i, (u, s)) in d_unscaled.iter().zip(d_scaled.iter()).enumerate() {
        let expected = u * n * expected_bin_width;
        assert!(
            (s - expected).abs() < 1e-9,
            "point {i}: scaled {s} != unscaled {u} * n {n} * bin_width {expected_bin_width} = {expected}"
        );
    }
}

/// Runs the REAL grouped `Bin` transform on `batch`/`field`/`group_cols`
/// under `shared_extent` and returns the bin width (`bin_end -
/// bin_start`) actually rendered for each group. The oracle every
/// grouped count-scale test below cross-checks against — never
/// `resolve_bin_extent` called with the implementation's own arguments,
/// which would be tautological on the exact global-vs-per-group
/// dimension these tests exist to pin (quality-review cycle-3
/// finding 1/3). Parameterized on `shared_extent` (quality-review
/// cycle-4 finding A) so both `Bin::apply_grouped` branches — an
/// un-shared per-group extent (`shared_extent=false`, the `displot`
/// default) and a shared extent with still-per-group `n_bins`
/// (`shared_extent=true`) — get a real-`Bin` oracle, not just a
/// hand-traced claim.
fn actual_bin_widths_per_group(
    batch: &RecordBatch,
    field: &str,
    group_cols: &[String],
    mode: BinMode,
    shared_extent: bool,
) -> std::collections::HashMap<String, f64> {
    let bin_spec = crate::transform::bin::BinSpec {
        field: field.into(),
        mode,
        extent: None,
        nice: true,
        cumulative: false,
        shared_extent,
        groupby: group_cols.to_vec(),
        agg_field: None,
        agg: None,
        name: None,
    };
    let bin_out = crate::transform::bin::apply(&bin_spec, batch).unwrap();
    widths_by_group_from_bin_output(&bin_out, &group_cols[0])
}

/// Read `bin_end - bin_start` per group off an ALREADY-COMPUTED `Bin`
/// output batch, keyed by `group_col`'s value. Shared by
/// `actual_bin_widths_per_group` (which builds its own `BinSpec`) and
/// the faceted test below (which needs the width `Bin` renders for a
/// spec it does NOT control the construction of — the facet-pinned
/// spec `fix_transform_extents_for_facet` produces).
fn widths_by_group_from_bin_output(
    bin_out: &RecordBatch,
    group_col: &str,
) -> std::collections::HashMap<String, f64> {
    use arrow::array::StringArray;
    let s = bin_out.schema();
    let starts = bin_out.column(s.index_of("bin_start").unwrap())
        .as_any().downcast_ref::<Float64Array>().unwrap();
    let ends = bin_out.column(s.index_of("bin_end").unwrap())
        .as_any().downcast_ref::<Float64Array>().unwrap();
    let groups = bin_out.column(s.index_of(group_col).unwrap())
        .as_any().downcast_ref::<StringArray>().unwrap();
    let mut widths = std::collections::HashMap::new();
    for i in 0..bin_out.num_rows() {
        widths.entry(groups.value(i).to_string())
            .or_insert_with(|| ends.value(i) - starts.value(i));
    }
    widths
}

/// Grouped count-scale, `Fixed` mode: `bin_width` must be derived PER
/// GROUP, mirroring `Bin::apply_grouped`'s `shared_extent=false` default
/// (the `displot` default, `multiple="layer"`) — NOT one scalar shared
/// across every hue level. Reproduces the quality reviewer's own
/// hand-computed numbers: group A `[0,2,4,6,8]` and group B
/// `[10,14,18]` under `Fixed{n:4}` each render width **2** from `Bin`
/// (`nice_step(0,8,4)=2`, `nice_step(10,18,4)=2`), NOT the global width
/// **5** a single-scalar derivation over the full `[0..18]` range would
/// give (`nice_step(0,18,4)=5`) — confirmed live: this test fails
/// (asserts 5.0 where `Bin` renders 2.0) against the pre-fix
/// implementation and passes against the per-group fix.
#[test]
fn test_kde_count_scale_grouped_fixed_mirrors_bin_per_group_width() {
    use arrow::array::StringArray;
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, true),
        Field::new("g", DataType::Utf8, false),
    ]));
    let xs = Float64Array::from(vec![
        0.0, 2.0, 4.0, 6.0, 8.0, // group A (5 points)
        10.0, 14.0, 18.0,        // group B (3 points)
    ]);
    let gs = StringArray::from(vec!["A", "A", "A", "A", "A", "B", "B", "B"]);
    let batch = RecordBatch::try_new(schema, vec![Arc::new(xs), Arc::new(gs)]).unwrap();
    let group_cols = vec!["g".to_string()];
    let mode = BinMode::Fixed { n: 4 };

    let widths = actual_bin_widths_per_group(&batch, "x", &group_cols, mode.clone(), false);
    assert!((widths["A"] - 2.0).abs() < 1e-9, "group A: Bin renders width 2.0, got {}", widths["A"]);
    assert!((widths["B"] - 2.0).abs() < 1e-9, "group B: Bin renders width 2.0, got {}", widths["B"]);

    let unscaled = KdeSpec { groupby: group_cols.clone(), ..count_scale_spec("x", 10, (0.0, 18.0), None) };
    let scaled = KdeSpec { groupby: group_cols, ..count_scale_spec("x", 10, (0.0, 18.0), Some(mode)) };
    let out_unscaled = apply(&unscaled, &batch).unwrap();
    let out_scaled = apply(&scaled, &batch).unwrap();
    let d_unscaled = col(&out_unscaled, "density");
    let d_scaled = col(&out_scaled, "density");
    let groups = out_unscaled.column(2).as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(out_scaled.num_rows(), out_unscaled.num_rows());

    for i in 0..out_unscaled.num_rows() {
        let g = groups.value(i).to_string();
        let n_group = if g == "A" { 5.0 } else { 3.0 };
        let expected = d_unscaled[i] * n_group * widths[&g];
        assert!(
            (d_scaled[i] - expected).abs() < 1e-9,
            "row {i} (group {g}): scaled {} != unscaled {} * n_group {n_group} * bin_width {} = {expected}",
            d_scaled[i], d_unscaled[i], widths[&g],
        );
    }
}

/// Grouped count-scale, `Sturges` mode (the `displot` default rule):
/// `resolve_bin_count`'s Sturges arm depends on `vals.len()`, so two
/// groups of DIFFERENT sizes must render DIFFERENT bin counts (hence
/// widths) even under the same field range — exactly what `Bin` itself
/// does and what a single shared-scalar derivation could never
/// reproduce. Discriminates the exact bug finding 1 describes for the
/// most common real-world `count_scale` mode (the pre-batch-F default
/// `bins="sturges"` displot forwards).
#[test]
fn test_kde_count_scale_grouped_sturges_mirrors_bin_per_group_n_bins() {
    use arrow::array::StringArray;
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, true),
        Field::new("g", DataType::Utf8, false),
    ]));
    let a: Vec<f64> = (0..12).map(|i| i as f64).collect(); // 12 points
    let b: Vec<f64> = vec![0.0, 4.0, 8.0, 12.0];            // 4 points
    let mut xs_vals = a.clone();
    xs_vals.extend(b.iter().copied());
    let mut gs_vals = vec!["A"; a.len()];
    gs_vals.extend(vec!["B"; b.len()]);
    let xs = Float64Array::from(xs_vals);
    let gs = StringArray::from(gs_vals);
    let batch = RecordBatch::try_new(schema, vec![Arc::new(xs), Arc::new(gs)]).unwrap();
    let group_cols = vec!["g".to_string()];

    let widths = actual_bin_widths_per_group(&batch, "x", &group_cols, BinMode::Sturges, false);
    assert!(
        (widths["A"] - widths["B"]).abs() > 1e-9,
        "expected different Sturges widths per group (A={}, B={}) — otherwise this \
         test can't discriminate per-group from a shared-scalar derivation",
        widths["A"], widths["B"],
    );

    let unscaled = KdeSpec { groupby: group_cols.clone(), ..count_scale_spec("x", 8, (0.0, 12.0), None) };
    let scaled = KdeSpec { groupby: group_cols, ..count_scale_spec("x", 8, (0.0, 12.0), Some(BinMode::Sturges)) };
    let out_unscaled = apply(&unscaled, &batch).unwrap();
    let out_scaled = apply(&scaled, &batch).unwrap();
    let d_unscaled = col(&out_unscaled, "density");
    let d_scaled = col(&out_scaled, "density");
    let groups = out_unscaled.column(2).as_any().downcast_ref::<StringArray>().unwrap();

    for i in 0..out_unscaled.num_rows() {
        let g = groups.value(i).to_string();
        let n_group = if g == "A" { a.len() as f64 } else { b.len() as f64 };
        let expected = d_unscaled[i] * n_group * widths[&g];
        assert!(
            (d_scaled[i] - expected).abs() < 1e-9,
            "row {i} (group {g}): scaled {} != unscaled {} * n_group {n_group} * bin_width {} = {expected}",
            d_scaled[i], d_unscaled[i], widths[&g],
        );
    }
}

/// Grouped count-scale, `shared_extent=true` + `Sturges` mode
/// (quality-review cycle-4 finding A) — the one branch the two tests
/// above never exercise: `resolve_group_bin_width`'s `Some((lo, hi))`
/// arm, reached when `multiple in ("stack","fill")` sets `shared_extent`
/// on both `mark_histogram` and `mark_density`. Group B's own 3 points
/// span only `[5,7]` — if the extent were (wrongly) still derived
/// per-group, group B would bin over `[5,7]`, not the shared `[0,9]`;
/// this test would then still pass by accident unless it also proves
/// the shared bound is actually reaching `Bin` and `Kde` identically,
/// which the real-`Bin` oracle does by construction (both read the same
/// `shared_extent=true` semantics). Sturges still varies `n_bins` (and
/// hence width) per group SIZE even though the extent is shared — group
/// A (10 points) and group B (3 points) get different bin counts over
/// the same `[0,9]` range, pinning both halves of the branch at once.
#[test]
fn test_kde_count_scale_grouped_shared_extent_sturges_mirrors_bin_shared_extent_per_group_n_bins() {
    use arrow::array::StringArray;
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, true),
        Field::new("g", DataType::Utf8, false),
    ]));
    let a: Vec<f64> = (0..10).map(|i| i as f64).collect(); // 10 points, spans [0,9]
    let b: Vec<f64> = vec![5.0, 6.0, 7.0];                  // 3 points, spans [5,7] alone
    let mut xs_vals = a.clone();
    xs_vals.extend(b.iter().copied());
    let mut gs_vals = vec!["A"; a.len()];
    gs_vals.extend(vec!["B"; b.len()]);
    let xs = Float64Array::from(xs_vals);
    let gs = StringArray::from(gs_vals);
    let batch = RecordBatch::try_new(schema, vec![Arc::new(xs), Arc::new(gs)]).unwrap();
    let group_cols = vec!["g".to_string()];

    let widths = actual_bin_widths_per_group(&batch, "x", &group_cols, BinMode::Sturges, true);
    assert!(
        (widths["A"] - widths["B"]).abs() > 1e-9,
        "expected different Sturges n_bins per group even under a shared extent \
         (A={}, B={}) — otherwise this test can't discriminate the shared-extent, \
         still-per-group-n_bins shape from either extreme",
        widths["A"], widths["B"],
    );

    let unscaled = KdeSpec {
        groupby: group_cols.clone(),
        shared_extent: true,
        ..count_scale_spec("x", 8, (0.0, 9.0), None)
    };
    let scaled = KdeSpec {
        groupby: group_cols,
        shared_extent: true,
        ..count_scale_spec("x", 8, (0.0, 9.0), Some(BinMode::Sturges))
    };
    let out_unscaled = apply(&unscaled, &batch).unwrap();
    let out_scaled = apply(&scaled, &batch).unwrap();
    let d_unscaled = col(&out_unscaled, "density");
    let d_scaled = col(&out_scaled, "density");
    let groups = out_unscaled.column(2).as_any().downcast_ref::<StringArray>().unwrap();

    for i in 0..out_unscaled.num_rows() {
        let g = groups.value(i).to_string();
        let n_group = if g == "A" { a.len() as f64 } else { b.len() as f64 };
        let expected = d_unscaled[i] * n_group * widths[&g];
        assert!(
            (d_scaled[i] - expected).abs() < 1e-9,
            "row {i} (group {g}): scaled {} != unscaled {} * n_group {n_group} * bin_width {} = {expected}",
            d_scaled[i], d_unscaled[i], widths[&g],
        );
    }
}

/// Faceted grouped count-scale (quality-review cycle-4 finding B): a
/// faceted `displot(col=, kind="hist", kde=True, stat="count")` runs
/// each panel's `Bin` with its extent PRE-PINNED to the niced GLOBAL
/// span (`fix_transform_extents_for_facet`, `render/prepare/extent.rs`),
/// so a narrow panel's histogram still bins over the WHOLE-dataset
/// range, not its own local span. `count_scale_extent` must be pinned
/// the SAME way (`kde::count_scale_global_extent`) so a faceted KDE
/// overlay derives the SAME per-panel bin width the real `Bin` renders
/// in that panel — not the panel's own local (narrower) span, which
/// would under-scale the overlay (the reviewer's ~25%,
/// unbounded-for-narrow-panels divergence). Simulates faceting
/// directly: pins BOTH a `Bin` and a `Kde` spec against the FULL
/// pre-facet batch (mirroring `fix_transform_extents_for_facet`'s
/// actual call in `render::prepare`), then applies BOTH to a NARROW
/// panel sub-batch — reproducing the divergence's exact shape.
#[test]
fn test_kde_count_scale_faceted_grouped_matches_bin_actual_panel_width() {
    use crate::render::prepare::fix_transform_extents_for_facet;
    use crate::transform::core::TransformSpec;
    use arrow::array::StringArray;

    // Full pre-facet dataset: spans [0, 20], two hue groups.
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, true),
        Field::new("g", DataType::Utf8, false),
    ]));
    let full_xs: Vec<f64> = (0..=20).map(|i| i as f64).collect();
    let full_gs: Vec<&str> = (0..=20).map(|i| if i % 2 == 0 { "A" } else { "B" }).collect();
    let full_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Float64Array::from(full_xs)) as ArrayRef,
            Arc::new(StringArray::from(full_gs)) as ArrayRef,
        ],
    ).unwrap();

    let mode = BinMode::Fixed { n: 5 };
    let group_cols = vec!["g".to_string()];
    let bin_spec = crate::transform::bin::BinSpec {
        field: "x".into(),
        mode: mode.clone(),
        extent: None,
        nice: true,
        cumulative: false,
        shared_extent: false,
        groupby: group_cols.clone(),
        agg_field: None,
        agg: None,
        name: None,
    };
    let kde_spec = KdeSpec {
        groupby: group_cols.clone(),
        ..count_scale_spec("x", 8, (2.0, 9.0), Some(mode.clone()))
    };

    // Mirror `render::prepare`'s actual facet-pin call: pin BOTH specs
    // against the FULL pre-facet batch.
    let pinned = fix_transform_extents_for_facet(
        &[TransformSpec::Bin(bin_spec), TransformSpec::Kde(kde_spec)],
        &full_batch,
    );
    let TransformSpec::Bin(pinned_bin) = &pinned[0] else { panic!("expected a pinned Bin") };
    let TransformSpec::Kde(pinned_kde) = &pinned[1] else { panic!("expected a pinned Kde") };

    // A NARROW panel: only rows with x in [2, 9] (8 points) — a fraction
    // of the full [0, 20] dataset span, the shape that diverges under a
    // per-panel-local derivation.
    let panel_xs: Vec<f64> = (2..=9).map(|i| i as f64).collect();
    let panel_gs: Vec<&str> = (2..=9).map(|i| if i % 2 == 0 { "A" } else { "B" }).collect();
    let panel_batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(panel_xs)) as ArrayRef,
            Arc::new(StringArray::from(panel_gs.clone())) as ArrayRef,
        ],
    ).unwrap();

    // The REAL Bin transform's ACTUAL per-group width IN THIS PANEL
    // (using the facet-pinned spec, exactly as `render::prepare` would
    // apply it per panel).
    let bin_out = crate::transform::bin::apply(pinned_bin, &panel_batch).unwrap();
    let panel_widths = widths_by_group_from_bin_output(&bin_out, "g");

    // Sanity: the panel's OWN local span [2,9] must give a DIFFERENT
    // (narrower) width than the global-pinned one Bin actually
    // rendered — otherwise this test can't discriminate the facet-pin
    // fix from a false pass.
    let local_widths = actual_bin_widths_per_group(&panel_batch, "x", &group_cols, mode, false);
    assert!(
        (panel_widths["A"] - local_widths["A"]).abs() > 1e-9,
        "test setup must produce a real global-vs-local divergence to discriminate the fix \
         (global-pinned width {} == local-only width {})",
        panel_widths["A"], local_widths["A"],
    );

    // `Kde`'s own facet-pinned `count_scale_extent` must equal `Bin`'s
    // own facet-pinned `extent` — both derived via the SAME
    // niced-global routine over the SAME field.
    assert_eq!(
        pinned_kde.count_scale_extent, pinned_bin.extent,
        "Kde's facet-pinned count_scale_extent must equal Bin's facet-pinned extent \
         (both niced-global over the same field)"
    );

    // Apply Kde (scaled vs unscaled) to the SAME narrow panel and
    // confirm the applied factor equals Bin's ACTUAL per-panel-group
    // width — not the panel's own local span.
    let unscaled_kde = KdeSpec {
        count_scale: None,
        count_scale_extent: None,
        ..pinned_kde.clone()
    };
    let out_unscaled = apply(&unscaled_kde, &panel_batch).unwrap();
    let out_scaled = apply(pinned_kde, &panel_batch).unwrap();
    let d_unscaled = col(&out_unscaled, "density");
    let d_scaled = col(&out_scaled, "density");
    let groups = out_unscaled.column(2).as_any().downcast_ref::<StringArray>().unwrap();

    for i in 0..out_unscaled.num_rows() {
        let g = groups.value(i).to_string();
        let n_group = panel_gs.iter().filter(|&&v| v == g).count() as f64;
        let expected = d_unscaled[i] * n_group * panel_widths[&g];
        assert!(
            (d_scaled[i] - expected).abs() < 1e-9,
            "row {i} (group {g}): faceted scaled {} != unscaled {} * n_group {n_group} * \
             Bin's ACTUAL panel width {} = {expected}",
            d_scaled[i], d_unscaled[i], panel_widths[&g],
        );
    }
}

/// Close-remediation F7 (batch-F rust design review): `Kde::__repr__` must
/// show every constructor parameter, the completeness `Robust::__repr__`
/// gained in the same batch. Nothing pinned this before, which is how the
/// two siblings drifted apart inside one batch; asserting on the parameter
/// NAMES (not the whole rendered string) keeps the test about completeness
/// rather than about formatting.
#[test]
fn kde_repr_shows_every_constructor_parameter() {
    let spec = KdeSpec {
        groupby: vec!["g".into()],
        bw_adjust: 1.5,
        shared_extent: true,
        name: Some("overlay".into()),
        ..count_scale_spec("x", 8, (0.0, 9.0), Some(BinMode::Fixed { n: 5 }))
    };
    let repr = PyKde(TransformSpec::Kde(spec)).__repr__();
    for param in [
        "field", "bandwidth", "bw_adjust", "n=", "extent", "cumulative",
        "shared_extent", "kernel", "groupby", "count_scale", "name",
    ] {
        assert!(repr.contains(param), "repr must show `{param}`: {repr}");
    }
    assert!(
        !repr.contains("count_scale_extent"),
        "count_scale_extent has no constructor parameter — it is set only by the \
         facet pin, so it is not part of the value a user wrote: {repr}"
    );
    // Name-substring checks above pass regardless of what the expressions
    // evaluate to (the format string emits `"shared_extent="` etc. as a
    // literal no matter what follows it). Assert the rendered VALUES for the
    // spec's deliberately distinctive fields too, so a repr that renders
    // every value wrong — e.g. `shared_extent`'s True/False arms swapped —
    // is caught.
    assert!(
        repr.contains("shared_extent=True"),
        "repr must render shared_extent's actual value (True): {repr}"
    );
    assert!(
        repr.contains("bw_adjust=1.5"),
        "repr must render bw_adjust's actual value (1.5): {repr}"
    );
    assert!(
        repr.contains("name=Some(\"overlay\")"),
        "repr must render name's actual value (Some(\"overlay\")): {repr}"
    );
}

/// Close-remediation F3 (batch-F rust design review): `count_scale_extent`
/// has no constructor parameter, so the WIRE is its only route — and
/// `KdeSpec`'s `deny_unknown_fields` accepts the key. A hand-authored or
/// round-tripped spec carrying a reversed `[10, 0]` used to be consumed
/// verbatim, yielding a negative bin width that silently sign-flipped
/// every density. Both sibling extent slots refuse this shape (`Bin`'s
/// at consumption, `Kde::extent`'s at the constructor); this one now
/// refuses too, in `Bin`'s own error voice.
#[test]
fn kde_count_scale_extent_reversed_refuses_through_the_wire() {
    let json = r#"{"field":"x","bandwidth":{"kind":"scott"},"n":8,
        "count_scale":{"kind":"fixed","n":5},"count_scale_extent":[10.0,0.0]}"#;
    let spec: KdeSpec =
        serde_json::from_str(json).expect("the wire accepts the key — that is the reachability");
    assert_eq!(spec.count_scale_extent, Some((10.0, 0.0)));

    let batch = batch_with("x", (0..10).map(|i| i as f64).collect());
    let err = apply(&spec, &batch)
        .expect_err("a reversed count_scale_extent must refuse, not sign-flip the density");
    let msg = err.to_string();
    assert!(
        msg.contains("count_scale_extent") && msg.contains("lo < hi"),
        "error must name the slot and the violated constraint: {msg}"
    );
    assert!(msg.contains("stat_kde"), "error must name the transform: {msg}");
}

/// The non-finite half of the same guard: a `NaN`/infinite pinned extent
/// reaches `resolve_bin_extent` as a silent NaN width rather than a
/// refusal. `Kde::extent`'s constructor check covers both `lo < hi` and
/// finiteness; the wire-only slot must too.
#[test]
fn kde_count_scale_extent_non_finite_refuses() {
    let batch = batch_with("x", (0..10).map(|i| i as f64).collect());
    let spec = KdeSpec {
        count_scale_extent: Some((0.0, f64::INFINITY)),
        ..count_scale_spec("x", 8, (0.0, 9.0), Some(BinMode::Fixed { n: 5 }))
    };
    let err = apply(&spec, &batch)
        .expect_err("a non-finite count_scale_extent must refuse");
    assert!(
        err.to_string().contains("count_scale_extent"),
        "error must name the slot: {err}"
    );
}

/// Close-remediation mutation gate (advisory): the guard's error text
/// promises `lo < hi` with both finite, but the sibling tests above only
/// pin the reversed shape (`kde_count_scale_extent_reversed_refuses_through_the_wire`)
/// and the hi-side non-finite shape (`kde_count_scale_extent_non_finite_refuses`).
/// Two more ways to violate the same contract had no coverage: a degenerate
/// `lo == hi` extent (does not trip `lo >= hi`'s strict form being weakened
/// to `lo > hi` — an undetected mutant there would let it through, and it
/// would flow to `resolve_group_bin_width` as a pinned extent with
/// `should_nice = false`, yielding bin width `(hi - lo) / n_bins == 0.0` and
/// silently collapsing every count-scaled density to zero) and a lo-side
/// non-finite bound (`-inf`/`NaN` — `NaN >= hi` is false, so a mutant that
/// only checks `!hi.is_finite()` would still refuse the hi-side case but
/// wrongly accept these).
#[test]
fn kde_count_scale_extent_degenerate_and_lo_side_non_finite_refuse() {
    let batch = batch_with("x", (0..10).map(|i| i as f64).collect());
    for (lo, hi) in [(5.0, 5.0), (f64::NEG_INFINITY, 10.0), (f64::NAN, 10.0)] {
        let spec = KdeSpec {
            count_scale_extent: Some((lo, hi)),
            ..count_scale_spec("x", 8, (0.0, 9.0), Some(BinMode::Fixed { n: 5 }))
        };
        let err = apply(&spec, &batch)
            .expect_err(&format!("count_scale_extent=({lo}, {hi}) must refuse"));
        assert!(
            err.to_string().contains("count_scale_extent"),
            "error must name the slot for ({lo}, {hi}): {err}"
        );
    }
}

/// The bin width the REAL ungrouped `Bin` transform renders for
/// `field` under `mode`/`extent` — the ungrouped counterpart of
/// `actual_bin_widths_per_group`'s oracle.
fn actual_bin_width_ungrouped(
    batch: &RecordBatch,
    field: &str,
    mode: BinMode,
    extent: Option<(f64, f64)>,
) -> f64 {
    let bin_spec = crate::transform::bin::BinSpec {
        field: field.into(),
        mode,
        extent,
        nice: true,
        cumulative: false,
        shared_extent: false,
        groupby: vec![],
        agg_field: None,
        agg: None,
        name: None,
    };
    let bin_out = crate::transform::bin::apply(&bin_spec, batch).unwrap();
    let s = bin_out.schema();
    let starts = bin_out.column(s.index_of("bin_start").unwrap())
        .as_any().downcast_ref::<Float64Array>().unwrap();
    let ends = bin_out.column(s.index_of("bin_end").unwrap())
        .as_any().downcast_ref::<Float64Array>().unwrap();
    ends.value(0) - starts.value(0)
}

/// The Rust-side anchor for `count_scale`'s cross-language parity claim
/// (batch-F close review, "what still feels wrong" item 2). `count_scale`
/// promises "bit parity with a histogram binned the same way", but nothing
/// in `cargo test` asserted that `resolve_group_bin_width`'s width EQUALS
/// the width `Bin::apply` actually used — the end-to-end count-scale tests
/// above assert the applied FACTOR, and the parity claim itself was
/// checked only on the Python side (§10, overlay plateau vs rendered
/// bars). This walks the full matrix the two implementations can diverge
/// across — `BinMode` (Fixed / Sturges) × group shape (grouped /
/// ungrouped) × extent (pinned / data-derived) — and asserts EXACT
/// equality, not an epsilon: both sides route through
/// `bin::resolve_bin_extent`, so any drift in either one's fold, nicing,
/// or bin-count decision shows up as a bit difference here rather than as
/// a mis-scaled curve a user has to notice.
#[test]
fn resolve_group_bin_width_matches_bin_actual_width_across_the_matrix() {
    use arrow::array::StringArray;
    pyo3::Python::initialize();

    // Group A spans [0, 12] with 13 points; group B spans [1, 9] with 3.
    // Different sizes AND different spans, so Sturges' size-dependent bin
    // count and the extent fold are both exercised per group.
    let a: Vec<f64> = (0..=12).map(|i| i as f64).collect();
    let b: Vec<f64> = vec![1.0, 5.0, 9.0];
    let mut xs_vals = a.clone();
    xs_vals.extend(b.iter().copied());
    let mut gs_vals = vec!["A"; a.len()];
    gs_vals.extend(vec!["B"; b.len()]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, true),
        Field::new("g", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(xs_vals.clone())) as ArrayRef,
            Arc::new(StringArray::from(gs_vals)) as ArrayRef,
        ],
    ).unwrap();
    let group_cols = vec!["g".to_string()];

    // A pinned extent deliberately WIDER than the data, so a site that
    // silently fell back to the data-derived range would differ.
    for pinned in [None, Some((-2.0, 20.0))] {
        for mode in [BinMode::Fixed { n: 4 }, BinMode::Sturges] {
            // Ungrouped: one `clean` over every row.
            let bin_width = actual_bin_width_ungrouped(&batch, "x", mode.clone(), pinned);
            let kde_width =
                resolve_group_bin_width(&xs_vals, &mode, pinned, "stat_kde").unwrap();
            assert_eq!(
                kde_width,
                Some(bin_width),
                "ungrouped {mode:?} with extent {pinned:?}: Kde's count_scale width must \
                 equal the width Bin actually rendered",
            );

            // Grouped: one `clean` per group, one width per group.
            let bin_spec = crate::transform::bin::BinSpec {
                field: "x".into(),
                mode: mode.clone(),
                extent: pinned,
                nice: true,
                cumulative: false,
                shared_extent: false,
                groupby: group_cols.clone(),
                agg_field: None,
                agg: None,
                name: None,
            };
            let bin_out = crate::transform::bin::apply(&bin_spec, &batch).unwrap();
            let widths = widths_by_group_from_bin_output(&bin_out, "g");
            for (g, clean) in [("A", &a), ("B", &b)] {
                let kde_width =
                    resolve_group_bin_width(clean, &mode, pinned, "stat_kde").unwrap();
                assert_eq!(
                    kde_width,
                    Some(widths[g]),
                    "group {g}, {mode:?} with extent {pinned:?}: Kde's count_scale width \
                     must equal the width Bin actually rendered for that group",
                );
            }
        }
    }
}

/// Direct unit tests for `resolve_group_bin_width` (quality-review
/// cycle-3 findings 4/5): the empty-clean case returns `Ok(None)` — "no
/// scaling factor" — rather than a NaN sentinel, and the degenerate
/// all-equal case collapses to `Bin`'s unit-width bin (`1.0`) exactly.
#[test]
fn test_resolve_group_bin_width_empty_returns_none() {
    let width = resolve_group_bin_width(&[], &BinMode::Sturges, None, "stat_kde").unwrap();
    assert_eq!(width, None, "empty clean column must return None, not a NaN sentinel");
}

#[test]
fn test_resolve_group_bin_width_degenerate_all_equal_matches_bin_unit_width() {
    let clean = vec![5.0, 5.0, 5.0, 5.0];
    let width = resolve_group_bin_width(&clean, &BinMode::Fixed { n: 3 }, None, "stat_kde").unwrap();
    assert_eq!(width, Some(1.0), "degenerate all-equal field must collapse to Bin's unit-width bin");
}

/// Cumulative count-scale: the CDF scales by `n_group` alone — NOT
/// `n_group × bin_width` — per the normative spec semantics.
#[test]
fn test_kde_count_scale_cumulative_scales_by_n_group_only() {
    let values: Vec<f64> = (0..15).map(|i| i as f64).collect();
    let batch = batch_with("x", values.clone());

    let mut unscaled = count_scale_spec("x", 10, (0.0, 14.0), None);
    unscaled.cumulative = true;
    let mut scaled = count_scale_spec("x", 10, (0.0, 14.0), Some(BinMode::Fixed { n: 5 }));
    scaled.cumulative = true;

    let out_unscaled = apply(&unscaled, &batch).unwrap();
    let out_scaled = apply(&scaled, &batch).unwrap();
    let d_unscaled = col(&out_unscaled, "density");
    let d_scaled = col(&out_scaled, "density");

    let n = values.len() as f64;
    for (i, (u, s)) in d_unscaled.iter().zip(d_scaled.iter()).enumerate() {
        let expected = u * n;
        assert!(
            (s - expected).abs() < 1e-9,
            "point {i}: cumulative scaled {s} != unscaled {u} * n {n} = {expected} (bin_width must NOT apply to the CDF)"
        );
    }
}
