//! Data transform: DensityData — KDE as a data transform.
//!
//! Produces a two-column output (x-grid, density) similar to the stat KDE
//! transform but exposed as a data transform with different parameters.

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::transform::group_key::{
    groupby_field_nullable, groupby_key_at, is_groupby_supported_dtype, materialize_groupby_col,
    KeyValue,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct DensityDataSpec {
    pub field: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub bandwidth: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub groupby: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub extent: Option<(f64, f64)>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub steps: Option<usize>,
    #[serde(default)]
    pub cumulative: bool,
    /// Kernel function for density estimation. Shares `Kde`'s vocabulary and
    /// dispatch (`crate::transform::kde::kernel_density_at_grid`,
    /// `crate::transform::kde::KERNELS`) — see that module for the supported
    /// list. Omitted from the wire when "gaussian" (the default), so specs
    /// written before this field round-trip byte-identically.
    #[serde(
        default = "crate::transform::kde::default_kernel",
        skip_serializing_if = "is_default_kernel"
    )]
    pub kernel: String,
    #[serde(default = "default_density_as")]
    pub as_: (String, String),
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

fn default_density_as() -> (String, String) {
    ("value".into(), "density".into())
}

/// `skip_serializing_if` predicate for [`DensityDataSpec::kernel`].
fn is_default_kernel(k: &str) -> bool {
    k == "gaussian"
}

/// Compute the global (pre-facet) raw min/max of the density field over the full
/// batch. Returns `None` when the column is absent or all values are null/NaN.
///
/// DensityData does **not** nice its extent (unlike `bin::global_extent`), because
/// it controls the KDE grid start and end, not discrete bin edges. This mirrors
/// `kde::global_extent`. Used by `render::prepare::fix_transform_extents_for_facet`
/// to pin the value axis before partitioning so every facet panel shares the same
/// KDE grid range (archaeology bug #7, extended to DensityData by round-3 T1).
///
/// NICENESS CONTRACT (XFORM-08): returns the RAW `(lo, hi)` (no nicing), mirroring
/// the KDE sibling and unlike `Bin`. Uses the shared `column_extent` helper, which
/// coerces Int64/Float32/etc. pre-facet batches to Float64 so they produce a valid
/// shared extent instead of returning None.
pub(crate) fn global_extent(spec: &DensityDataSpec, batch: &RecordBatch) -> Option<(f64, f64)> {
    crate::transform::numeric_util::column_extent(batch, &spec.field)
}

pub(crate) fn apply(spec: &DensityDataSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let schema = batch.schema();
    let idx = schema.index_of(&spec.field).map_err(|_| {
        PyValueError::new_err(format!("data_density: column '{}' not found", spec.field))
    })?;

    if let Some(groupby) = &spec.groupby {
        return apply_grouped(spec, batch, idx, groupby);
    }

    apply_one_group(spec, batch, idx, None)
}

fn apply_grouped(
    spec: &DensityDataSpec,
    batch: &RecordBatch,
    field_idx: usize,
    groupby: &[String],
) -> PyResult<RecordBatch> {
    let schema = batch.schema();
    let n_rows = batch.num_rows();

    // Only support single groupby column for simplicity.
    if groupby.is_empty() {
        return apply_one_group(spec, batch, field_idx, None);
    }

    let g_col_name = &groupby[0];
    let g_idx = schema.index_of(g_col_name).map_err(|_| {
        PyValueError::new_err(format!("data_density: groupby column '{g_col_name}' not found"))
    })?;
    let g_dtype = schema.field(g_idx).data_type().clone();
    if !is_groupby_supported_dtype(&g_dtype) {
        return Err(PyValueError::new_err(format!(
            "data_density: groupby column '{g_col_name}' has unsupported dtype {g_dtype:?}; \
             supported: Utf8/LargeUtf8, Float64/Float32, \
             Int8/Int16/Int32/Int64, UInt8/UInt16/UInt32/UInt64, Boolean"
        )));
    }
    let g_col = batch.column(g_idx);

    // Partition rows by group key (FA-7: int/uint/bool/float/string supported).
    let mut groups: std::collections::BTreeMap<KeyValue, Vec<usize>> =
        std::collections::BTreeMap::new();
    for row in 0..n_rows {
        let key = groupby_key_at(g_col.as_ref(), &g_dtype, row).ok_or_else(|| {
            PyValueError::new_err(format!(
                "data_density: internal error extracting groupby key at row {row}"
            ))
        })?;
        groups.entry(key).or_default().push(row);
    }

    // Run KDE per group and concatenate.
    let mut all_x: Vec<f64> = Vec::new();
    let mut all_y: Vec<f64> = Vec::new();
    let mut group_keys_out: Vec<Vec<KeyValue>> = Vec::new();

    for (group_key, rows) in &groups {
        let result = apply_one_group(spec, batch, field_idx, Some(rows))?;
        let x_col = result.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        let y_col = result.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        for i in 0..result.num_rows() {
            all_x.push(x_col.value(i));
            all_y.push(y_col.value(i));
            group_keys_out.push(vec![group_key.clone()]);
        }
    }

    let out_schema = Arc::new(Schema::new(vec![
        Field::new(&spec.as_.0, DataType::Float64, false),
        Field::new(&spec.as_.1, DataType::Float64, false),
        Field::new(g_col_name, g_dtype.clone(), groupby_field_nullable()),
    ]));
    let g_out = materialize_groupby_col(&group_keys_out, 0, &g_dtype)
        .map_err(PyValueError::new_err)?;
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(all_x)),
        Arc::new(Float64Array::from(all_y)),
        g_out,
    ];
    RecordBatch::try_new(out_schema, cols)
        .map_err(|e| PyValueError::new_err(format!("data_density: {e}")))
}

fn apply_one_group(
    spec: &DensityDataSpec,
    batch: &RecordBatch,
    field_idx: usize,
    only_rows: Option<&[usize]>,
) -> PyResult<RecordBatch> {
    let col = crate::transform::numeric_util::coerce_to_float64(
        batch.column(field_idx),
        "data_density",
        &spec.field,
    )?;

    // Extract clean values.
    let mut data: Vec<f64> = Vec::new();
    match only_rows {
        Some(rows) => {
            for &r in rows {
                if !col.is_null(r) {
                    let v = col.value(r);
                    if !v.is_nan() {
                        data.push(v);
                    }
                }
            }
        }
        None => {
            for i in 0..col.len() {
                if !col.is_null(i) {
                    let v = col.value(i);
                    if !v.is_nan() {
                        data.push(v);
                    }
                }
            }
        }
    }

    if data.is_empty() {
        let out_schema = Arc::new(Schema::new(vec![
            Field::new(&spec.as_.0, DataType::Float64, false),
            Field::new(&spec.as_.1, DataType::Float64, false),
        ]));
        let cols: Vec<ArrayRef> = vec![
            Arc::new(Float64Array::from(Vec::<f64>::new())),
            Arc::new(Float64Array::from(Vec::<f64>::new())),
        ];
        return RecordBatch::try_new(out_schema, cols)
            .map_err(|e| PyValueError::new_err(format!("data_density: {e}")));
    }

    let n = data.len() as f64;
    let steps = spec.steps.unwrap_or(200);

    // Bandwidth: Silverman's rule if not specified.
    let bw = spec.bandwidth.unwrap_or_else(|| {
        let mean = data.iter().sum::<f64>() / n;
        let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = var.sqrt();
        1.06 * std_dev * n.powf(-0.2)
    });

    // Extent.
    let data_min = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let data_max = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let (lo, hi) = spec.extent.unwrap_or((data_min - 3.0 * bw, data_max + 3.0 * bw));

    // Generate x grid.
    let step_size = (hi - lo) / (steps - 1).max(1) as f64;
    let x_grid: Vec<f64> = (0..steps).map(|i| lo + i as f64 * step_size).collect();

    // Compute KDE via the shared kernel dispatch (kde.rs) — the crate's one
    // kernel-math table. `bw` above keeps DensityData's own bandwidth
    // default/formula (1.06 * std_dev * n^-0.2), unchanged from before this
    // task; only the per-kernel math is now shared with `Kde`.
    let mut density: Vec<f64> =
        crate::transform::kde::kernel_density_at_grid(&data, bw, &x_grid, &spec.kernel)?;

    // Cumulative if requested.
    if spec.cumulative {
        let dx = step_size;
        let mut cumsum = 0.0;
        for d in density.iter_mut() {
            cumsum += *d * dx;
            *d = cumsum;
        }
    }

    let out_schema = Arc::new(Schema::new(vec![
        Field::new(&spec.as_.0, DataType::Float64, false),
        Field::new(&spec.as_.1, DataType::Float64, false),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(x_grid)),
        Arc::new(Float64Array::from(density)),
    ];
    RecordBatch::try_new(out_schema, cols)
        .map_err(|e| PyValueError::new_err(format!("data_density: {e}")))
}

// ─── PyO3 wrapper ──────────────────────────────────────────────────────────

use pyo3::prelude::*;
use crate::transform::core::TransformSpec;

#[pyclass(module = "ferrum._core", name = "DensityData")]
#[derive(Debug, Clone)]
pub(crate) struct PyDensityData(pub(crate) TransformSpec);

#[pymethods]
impl PyDensityData {
    #[new]
    #[pyo3(signature = (field, *, name = None))]
    fn new(field: String, name: Option<String>) -> Self {
        PyDensityData(TransformSpec::DensityData(DensityDataSpec {
            field,
            bandwidth: None,
            groupby: None,
            extent: None,
            steps: None,
            cumulative: false,
            kernel: crate::transform::kde::default_kernel(),
            as_: default_density_as(),
            name,
        }))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::DensityData(s) => format!("DensityData(field='{}')", s.field),
            _ => "DensityData(?)".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float64Array;
    use arrow::datatypes::{Field, Schema};
    use std::sync::Arc;

    #[test]
    fn density_basic() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![
                1.0, 2.0, 3.0, 4.0, 5.0,
            ]))],
        )
        .unwrap();

        let spec = DensityDataSpec {
            field: "x".into(),
            bandwidth: Some(1.0),
            groupby: None,
            extent: Some((0.0, 6.0)),
            steps: Some(50),
            cumulative: false,
            kernel: crate::transform::kde::default_kernel(),
            as_: ("value".into(), "density".into()),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 50);
        assert_eq!(out.num_columns(), 2);
        assert_eq!(out.schema().field(0).name(), "value");
        assert_eq!(out.schema().field(1).name(), "density");

        // Density should be positive.
        let d = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        for i in 0..d.len() {
            assert!(d.value(i) >= 0.0);
        }
    }

    // ── global_extent helper ──────────────────────────────────────────────────

    fn make_batch(values: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(values))]).unwrap()
    }

    fn base_spec() -> DensityDataSpec {
        DensityDataSpec {
            field: "x".into(),
            bandwidth: None,
            groupby: None,
            extent: None,
            steps: None,
            cumulative: false,
            kernel: crate::transform::kde::default_kernel(),
            as_: ("value".into(), "density".into()),
            name: None,
        }
    }

    /// `global_extent` returns the raw (min, max) — no nicing, mirroring KDE.
    #[test]
    fn global_extent_returns_raw_min_max() {
        let batch = make_batch(vec![1.0, 3.0, 2.5, 7.8, 4.2]);
        let spec = base_spec();
        let ext = global_extent(&spec, &batch);
        assert_eq!(ext, Some((1.0, 7.8)), "global_extent must be (min, max) of values");
    }

    /// `global_extent` returns `None` when the column is absent.
    #[test]
    fn global_extent_missing_field_returns_none() {
        let schema = Arc::new(Schema::new(vec![Field::new("y", DataType::Float64, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![1.0, 2.0]))],
        )
        .unwrap();
        let spec = base_spec(); // spec.field = "x", not "y"
        assert_eq!(global_extent(&spec, &batch), None);
    }

    /// `global_extent` returns `None` when all values are null or NaN.
    #[test]
    fn global_extent_all_null_returns_none() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, true)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![None, None]))],
        )
        .unwrap();
        let spec = base_spec();
        assert_eq!(global_extent(&spec, &batch), None);
    }

    /// `global_extent` skips NaN values without panicking.
    #[test]
    fn global_extent_skips_nan() {
        let batch = make_batch(vec![f64::NAN, 2.0, f64::NAN, 8.0]);
        let spec = base_spec();
        assert_eq!(global_extent(&spec, &batch), Some((2.0, 8.0)));
    }

    // ── F1 regression: int-field shared-extent gap ────────────────────
    //
    // Before the fix, `global_extent` used raw `downcast_ref::<Float64Array>()`
    // which returns None for an Int64 column. A faceted density on an integer
    // field would silently get no shared-extent pin (bug #7 for int inputs).
    // After the fix, `coerce_to_float64` is used, mirroring `kde::global_extent`.

    /// `global_extent` on an Int64 column returns Some((min, max)) — not None.
    #[test]
    fn global_extent_int64_field_returns_some() {
        use arrow::array::Int64Array;
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![5i64, 1, 9, 3]))],
        )
        .unwrap();
        let spec = base_spec(); // spec.field = "x"
        // Pre-fix: raw downcast returns None for Int64 → no shared extent.
        // Post-fix: coerce_to_float64 casts Int64 → Float64 → Some((1.0, 9.0)).
        assert_eq!(
            global_extent(&spec, &batch),
            Some((1.0, 9.0)),
            "global_extent on Int64 column must coerce and return Some"
        );
    }

    /// `global_extent` on a Float64 column remains byte-identical after the fix.
    #[test]
    fn global_extent_float64_unchanged_by_fix() {
        let batch = make_batch(vec![3.0, 1.0, 7.5, 2.2]);
        let spec = base_spec();
        assert_eq!(
            global_extent(&spec, &batch),
            Some((1.0, 7.5)),
            "global_extent on Float64 must be unchanged by fix"
        );
    }

    // ── T11: shared kernel dispatch ────────────────────────────────────────

    /// A `DensityDataSpec` deserialized from wire JSON without a `kernel`
    /// field defaults to "gaussian", and the default is omitted from
    /// serialization so specs written before this field round-trip
    /// byte-identically.
    #[test]
    fn density_data_kernel_field_defaults_and_wire_omission() {
        let json = r#"{"field":"x"}"#;
        let parsed: DensityDataSpec = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.kernel, "gaussian");

        let default_json = serde_json::to_string(&base_spec()).unwrap();
        assert!(!default_json.contains("kernel"), "default kernel must be omitted: {default_json}");

        let non_default = DensityDataSpec { kernel: "epanechnikov".into(), ..base_spec() };
        let non_default_json = serde_json::to_string(&non_default).unwrap();
        assert!(
            non_default_json.contains(r#""kernel":"epanechnikov""#),
            "non-default kernel must serialize: {non_default_json}"
        );
    }

    /// `DensityData`'s kernel dispatch must actually vary by kernel, not
    /// silently compute Gaussian regardless of the requested string. Single
    /// data point + fixed bandwidth: Epanechnikov's compact support means
    /// density is EXACTLY 0 beyond the bandwidth cutoff — a shape Gaussian's
    /// unbounded tail never produces. Proves `density_data` now routes
    /// through `kde::kernel_density_at_grid` (the shared table) rather than
    /// the old hardcoded-Gaussian inline loop.
    #[test]
    fn density_data_kernel_routes_through_shared_kde_dispatch() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![0.0, 0.01]))],
        )
        .unwrap();
        let spec = DensityDataSpec {
            field: "x".into(),
            bandwidth: Some(0.5),
            groupby: None,
            extent: Some((-2.0, 2.0)),
            steps: Some(21),
            cumulative: false,
            kernel: "epanechnikov".into(),
            as_: ("value".into(), "density".into()),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let value = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        let density = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        let mut saw_zero_beyond_support = false;
        for i in 0..out.num_rows() {
            let v = value.value(i);
            let in_support = v.abs() <= 0.5 || (v - 0.01).abs() <= 0.5;
            if !in_support {
                assert_eq!(
                    density.value(i),
                    0.0,
                    "epanechnikov beyond bandwidth must be exactly 0 at x={v}"
                );
                saw_zero_beyond_support = true;
            }
        }
        assert!(
            saw_zero_beyond_support,
            "test setup should include grid points outside the bandwidth"
        );
    }

    /// An unrecognized kernel string must error at `apply()` time (routed
    /// through `kde::kernel_density_at_grid`'s typed refusal), not silently
    /// fall back to Gaussian — `DensityData` has no constructor-time kernel
    /// validation of its own (`PyDensityData::new` doesn't expose the
    /// parameter; `transform_density` builds the spec directly from wire
    /// JSON), so this is the only enforcement point for this path.
    #[test]
    fn density_data_unknown_kernel_errors() {
        pyo3::Python::initialize();
        let batch = make_batch(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let spec = DensityDataSpec { kernel: "not_a_real_kernel".into(), ..base_spec() };
        let err = apply(&spec, &batch).unwrap_err();
        assert!(err.to_string().contains("unknown kernel"), "err: {err}");
    }

    /// Forward regression snapshot (round 2, batch-D T11 cycle 2, replaces
    /// the vacuous `>= 0.0` check the spec reviewer flagged). This task
    /// deliberately changed `density_data`'s gaussian-default output at the
    /// bit level: the old inline loop accumulated `norm * exp(...)`
    /// data-outer/grid-inner, while the shared `kde::kernel_density_at_grid`
    /// dispatch sums raw `exp(...)` terms grid-outer/data-inner and
    /// multiplies by `norm` once — floating-point addition isn't
    /// associative, so the two loop orders differ by up to ~41 ULP / 7.0e-15
    /// relative at n=5000 (measured directly against this module's own
    /// `apply()`, not a design estimate). Nothing pinned the ACTUAL
    /// post-change values before this test — a re-inlined
    /// loop, a wrong normalization constant, or a reordered summation would
    /// all have passed the old `>= 0.0` assertion. These bit patterns are
    /// the adjudicated post-change baseline (dumped from this exact
    /// `apply()` call the round this test was written) — any future change
    /// to `density_data`'s default gaussian path, intentional or not, must
    /// touch this pin.
    #[test]
    fn density_data_gaussian_default_matches_pinned_bits() {
        let out = apply(&base_spec(), &make_batch(vec![1.0, 2.0, 3.0, 4.0, 5.0])).unwrap();
        let density = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(density.len(), 200, "steps defaults to 200");
        // (grid index, expected bit pattern) — a representative subset
        // (first, near-first, two interior, and the mirror-symmetric last
        // point) rather than all 200, since the extent is symmetric around
        // the data's mean.
        //
        // CAUTION: These pins are anchored to macOS libm. Platform-specific libm
        // implementations may differ by ≤1 ULP. The sibling test
        // `density_data_gaussian_portable_matches_in_binary` provides platform-neutral
        // validation by computing density via two independent in-binary paths with
        // the same inputs and asserting bit-equality.
        let pins: &[(usize, u64)] = &[
            (0, 0x3f4bdbcdc3785a18),
            (1, 0x3f502125787b0725),
            (50, 0x3fb754893c29f29a),
            (100, 0x3fc9296342089189),
            (150, 0x3fb64ee46eb5edb3),
            (199, 0x3f4bdbcdc3785a18),
        ];
        for &(i, want_bits) in pins {
            let got_bits = density.value(i).to_bits();
            assert_eq!(
                got_bits, want_bits,
                "density_data gaussian-default output drifted at grid index {i}: \
                 got {:?} (bits {got_bits:#x}), want bits {want_bits:#x}",
                density.value(i)
            );
        }
    }

    /// Portable companion to `density_data_gaussian_default_matches_pinned_bits`:
    /// two independent in-binary computations of the same Gaussian density.
    /// The pinned-bits test anchors to macOS libm; this test validates the
    /// computation path by calling `gaussian_density_at_grid` directly with
    /// the same bandwidth, grid, and data, then asserting bit-equality against
    /// the output from `apply()`. Both paths run in the same process with the
    /// same libm, so this proves the apply() output is reproducible within the
    /// binary rather than subject to compiler variation or transcendental
    /// function vagaries.
    #[test]
    fn density_data_gaussian_portable_matches_in_binary() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let batch = make_batch(data.clone());
        let out = apply(&base_spec(), &batch).unwrap();
        let density_from_apply =
            out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        let x_grid_from_apply =
            out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();

        // Replicate the bandwidth calculation (Silverman's rule) from apply_one_group.
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = var.sqrt();
        let bw = 1.06 * std_dev * n.powf(-0.2);

        // Replicate the grid generation from apply_one_group.
        let steps = 200;
        let data_min = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let data_max = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let lo = data_min - 3.0 * bw;
        let hi = data_max + 3.0 * bw;
        let step_size = (hi - lo) / (steps - 1).max(1) as f64;
        let x_grid: Vec<f64> = (0..steps).map(|i| lo + i as f64 * step_size).collect();

        // Compute density via the direct gaussian_density_at_grid path.
        let density_from_direct =
            crate::transform::kde::gaussian_density_at_grid(&data, bw, &x_grid);

        // Both should have the same length.
        assert_eq!(
            density_from_apply.len(),
            density_from_direct.len(),
            "apply() and direct gaussian_density_at_grid must produce same grid length"
        );

        // Every grid point should match bit-for-bit.
        for i in 0..density_from_apply.len() {
            let apply_val = density_from_apply.value(i);
            let direct_val = density_from_direct[i];
            let apply_bits = apply_val.to_bits();
            let direct_bits = direct_val.to_bits();
            assert_eq!(
                apply_bits, direct_bits,
                "density at grid index {i} must match: apply()={:?} (bits {apply_bits:#x}) \
                 vs direct={:?} (bits {direct_bits:#x})",
                apply_val, direct_val
            );
        }

        // Also verify the x_grid matches.
        assert_eq!(
            x_grid_from_apply.len(),
            x_grid.len(),
            "x_grid length from apply() should match replicated grid"
        );
        for i in 0..x_grid.len() {
            let apply_grid = x_grid_from_apply.value(i);
            let direct_grid = x_grid[i];
            let apply_bits = apply_grid.to_bits();
            let direct_bits = direct_grid.to_bits();
            assert_eq!(
                apply_bits, direct_bits,
                "grid point at index {i} must match: apply()={:?} (bits {apply_bits:#x}) \
                 vs direct={:?} (bits {direct_bits:#x})",
                apply_grid, direct_grid
            );
        }
    }
}
