//! Data transform: LoessData — LOESS smoothing as a data transform.
//!
//! Fits a locally weighted polynomial regression (LOESS/LOWESS) to produce
//! a smooth curve output.

use arrow::array::{ArrayRef, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::transform::group_key::{
    group_partition_multi, groupby_field_nullable, materialize_groupby_col, KeyValue,
};
use crate::transform::numeric_util::coerce_xy_pairs_at;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct LoessDataSpec {
    pub x: String,
    pub y: String,
    #[serde(default = "default_bandwidth")]
    pub bandwidth: f64,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub groupby: Option<Vec<String>>,
    #[serde(default = "default_loess_as")]
    pub as_: (String, String),
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

fn default_bandwidth() -> f64 {
    0.3
}
fn default_loess_as() -> (String, String) {
    ("x".into(), "y".into())
}

/// Grid size shared by [`apply_one_group`] and [`all_nan_output`] — kept as
/// one constant so the two can't drift apart.
const N_OUT: usize = 200;

pub(crate) fn apply(spec: &LoessDataSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    match &spec.groupby {
        Some(cols) if !cols.is_empty() => apply_grouped(spec, batch, cols),
        _ => apply_one_group(spec, batch, None),
    }
}

/// Fit one LOESS curve over `batch`, restricted to `only_indices` when given.
/// Ingests via [`coerce_xy_pairs_at`] so a grouped partition and the
/// ungrouped path (`only_indices = None`) share one cleaning walk (T2/M1).
fn apply_one_group(
    spec: &LoessDataSpec,
    batch: &RecordBatch,
    only_indices: Option<&[usize]>,
) -> PyResult<RecordBatch> {
    // Extract clean paired data (any numeric dtype; non-numeric raises).
    let (xs, ys) = coerce_xy_pairs_at(batch, &spec.x, &spec.y, "data_loess", only_indices)?;

    if xs.len() < 3 {
        return all_nan_output(spec);
    }

    // Sort by x.
    let mut pairs: Vec<(f64, f64)> = xs.iter().zip(ys.iter()).map(|(&x, &y)| (x, y)).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let sorted_x: Vec<f64> = pairs.iter().map(|(x, _)| *x).collect();
    let sorted_y: Vec<f64> = pairs.iter().map(|(_, y)| *y).collect();

    // Generate output grid.
    let x_min = sorted_x[0];
    let x_max = *sorted_x.last().unwrap();
    let step = (x_max - x_min) / (N_OUT - 1).max(1) as f64;
    let x_grid: Vec<f64> = (0..N_OUT).map(|i| x_min + i as f64 * step).collect();

    // LOESS: for each grid point, compute locally weighted linear regression.
    let n = sorted_x.len();
    let span = (spec.bandwidth * n as f64).ceil().max(3.0) as usize;
    let span = span.min(n);

    let mut y_grid: Vec<f64> = Vec::with_capacity(N_OUT);
    for &xg in &x_grid {
        let y_hat = loess_predict(xg, &sorted_x, &sorted_y, span);
        y_grid.push(y_hat);
    }

    let out_schema = Arc::new(Schema::new(vec![
        Field::new(&spec.as_.0, DataType::Float64, false),
        Field::new(&spec.as_.1, DataType::Float64, false),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(x_grid)),
        Arc::new(Float64Array::from(y_grid)),
    ];
    RecordBatch::try_new(out_schema, cols)
        .map_err(|e| PyValueError::new_err(format!("data_loess: {e}")))
}

/// Partition `batch` by `group_cols`, fit each partition independently via
/// [`apply_one_group`], then stack the per-group grids into one batch with
/// the group columns appended at their original dtype (XFORM-03) as
/// additional fields — the `smooth.rs:186-275` multi-column grouped template
/// (spec §4.3 / decision-doc M3a).
///
/// Each group's output grid spans that group's OWN x range (`sorted_x[0]`/
/// `sorted_x.last()` computed from the group-restricted `xs`), not a range
/// shared across groups — this mirrors `stat_smooth`'s grouped behavior (see
/// `smooth.rs::apply_one_group`), so a per-group LOESS curve only ever draws
/// over the x-values that group actually has.
///
/// A group with fewer than 3 usable (non-null, non-NaN) rows cannot support a
/// fit; `apply_one_group` returns [`all_nan_output`] for it rather than an
/// empty batch, so that group still contributes [`N_OUT`] rows and keeps its
/// group key in the concatenated output — the same choice `smooth.rs` makes
/// for its below-threshold groups. A silent 0-row drop would remove the group
/// from the color-scale domain and legend entirely; a NaN-filled row set
/// draws no visible curve (the line mark filters non-finite points) but keeps
/// the group's identity intact everywhere else.
fn apply_grouped(
    spec: &LoessDataSpec,
    batch: &RecordBatch,
    group_cols: &[String],
) -> PyResult<RecordBatch> {
    let (group_order, group_idx_map, gtypes) =
        group_partition_multi(batch, group_cols, "data_loess")?;

    let mut all_x: Vec<f64> = Vec::new();
    let mut all_y: Vec<f64> = Vec::new();
    let mut group_keys_out: Vec<Vec<KeyValue>> = Vec::new();

    for g in &group_order {
        let ixs = group_idx_map.get(g).ok_or_else(|| {
            PyValueError::new_err(format!("data_loess: missing group key {g:?} in index map"))
        })?;
        let out = apply_one_group(spec, batch, Some(ixs))?;
        let out_x = out.column(0).as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
            PyValueError::new_err("data_loess: expected Float64Array in grouped output")
        })?;
        let out_y = out.column(1).as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
            PyValueError::new_err("data_loess: expected Float64Array in grouped output")
        })?;
        for i in 0..out.num_rows() {
            all_x.push(out_x.value(i));
            all_y.push(out_y.value(i));
            group_keys_out.push(g.clone());
        }
    }

    let mut out_fields = vec![
        Field::new(&spec.as_.0, DataType::Float64, false),
        Field::new(&spec.as_.1, DataType::Float64, false),
    ];
    let mut out_cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(all_x)),
        Arc::new(Float64Array::from(all_y)),
    ];

    for (gi, (col_name, dtype)) in group_cols.iter().zip(gtypes.iter()).enumerate() {
        out_fields.push(Field::new(col_name, dtype.clone(), groupby_field_nullable()));
        let group_col_arr = materialize_groupby_col(&group_keys_out, gi, dtype)
            .map_err(PyValueError::new_err)?;
        out_cols.push(group_col_arr);
    }

    let out_schema = Arc::new(Schema::new(out_fields));
    RecordBatch::try_new(out_schema, out_cols)
        .map_err(|e| PyValueError::new_err(format!("data_loess: {e}")))
}

/// Below-fit-threshold (`xs.len() < 3`) output: mirrors
/// `smooth.rs::all_nan_output` by emitting [`N_OUT`] NaN-valued rows instead
/// of an empty batch.
///
/// **Serves both grouped and ungrouped paths:** Under `apply_grouped`, a
/// below-threshold group contributes its full [`N_OUT`]-row count and keeps
/// its group key when concatenated — it stays in the output schema, the
/// color-scale domain, and the legend, rather than silently vanishing (see
/// the `apply_grouped` doc comment for the full rationale). Under the
/// ungrouped path (`apply_one_group(..., None)`), an entire below-threshold
/// input frame likewise emits [`N_OUT`] NaN rows rather than 0 rows, which
/// means such a chart renders with a fabricated fallback 0–1 domain for the
/// axes (since there is no real x-range to draw from empty input) — no error,
/// no warning, but also no visible curve (the line mark filters non-finite
/// points before building polylines).
fn all_nan_output(spec: &LoessDataSpec) -> PyResult<RecordBatch> {
    let out_schema = Arc::new(Schema::new(vec![
        Field::new(&spec.as_.0, DataType::Float64, false),
        Field::new(&spec.as_.1, DataType::Float64, false),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(vec![f64::NAN; N_OUT])),
        Arc::new(Float64Array::from(vec![f64::NAN; N_OUT])),
    ];
    RecordBatch::try_new(out_schema, cols)
        .map_err(|e| PyValueError::new_err(format!("data_loess: {e}")))
}

/// Predict y at point x using LOESS with tri-cube kernel.
fn loess_predict(x: f64, xs: &[f64], ys: &[f64], span: usize) -> f64 {
    let n = xs.len();

    // Find the `span` nearest neighbors.
    let mut dists: Vec<(usize, f64)> = xs
        .iter()
        .enumerate()
        .map(|(i, &xi)| (i, (xi - x).abs()))
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let neighbors = &dists[..span.min(n)];

    let max_dist = neighbors.last().map(|(_, d)| *d).unwrap_or(1.0);
    let h = if max_dist > 0.0 { max_dist } else { 1.0 };

    // Weighted linear regression: y = a + b * x with tri-cube weights.
    let mut sw = 0.0;
    let mut swx = 0.0;
    let mut swxx = 0.0;
    let mut swy = 0.0;
    let mut swxy = 0.0;

    for &(i, d) in neighbors {
        let u = d / h;
        let w = if u < 1.0 {
            let t = 1.0 - u * u * u;
            t * t * t
        } else {
            0.0
        };
        let xi = xs[i];
        let yi = ys[i];
        sw += w;
        swx += w * xi;
        swxx += w * xi * xi;
        swy += w * yi;
        swxy += w * xi * yi;
    }

    if sw < 1e-15 {
        return f64::NAN;
    }

    let det = sw * swxx - swx * swx;
    if det.abs() < 1e-15 {
        return swy / sw; // Constant fit.
    }

    let a = (swxx * swy - swx * swxy) / det;
    let b = (sw * swxy - swx * swy) / det;
    a + b * x
}

// ─── PyO3 wrapper ──────────────────────────────────────────────────────────

use pyo3::prelude::*;
use crate::transform::core::TransformSpec;

#[pyclass(module = "ferrum._core", name = "LoessData")]
#[derive(Debug, Clone)]
pub(crate) struct PyLoessData(pub(crate) TransformSpec);

#[pymethods]
impl PyLoessData {
    #[new]
    #[pyo3(signature = (x, y, *, bandwidth = 0.3, name = None))]
    fn new(x: String, y: String, bandwidth: f64, name: Option<String>) -> Self {
        PyLoessData(TransformSpec::LoessData(LoessDataSpec {
            x,
            y,
            bandwidth,
            groupby: None,
            as_: default_loess_as(),
            name,
        }))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::LoessData(s) => format!("LoessData(x='{}', y='{}')", s.x, s.y),
            _ => "LoessData(?)".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Float64Array};
    use arrow::datatypes::{Field, Schema};
    use std::sync::Arc;

    #[test]
    fn loess_smooth_linear() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        // Perfect line: y = 2x
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])),
                Arc::new(Float64Array::from(vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0])),
            ],
        )
        .unwrap();

        let spec = LoessDataSpec {
            x: "x".into(),
            y: "y".into(),
            bandwidth: 0.5,
            groupby: None,
            as_: ("x".into(), "y".into()),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 200);

        // LOESS on a perfect line should recover approximately y = 2x.
        let out_x = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        let out_y = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        // Check midpoint: x~4.5, y should be ~9.0
        let mid = 100;
        let expected_y = out_x.value(mid) * 2.0;
        assert!(
            (out_y.value(mid) - expected_y).abs() < 0.5,
            "LOESS at x={} gave y={}, expected ~{}",
            out_x.value(mid),
            out_y.value(mid),
            expected_y
        );
    }

    // ── M3a groupby coverage (F-L05-03) ──────────────────────────────────────

    fn spec_with_groupby(groupby: Option<Vec<String>>) -> LoessDataSpec {
        LoessDataSpec {
            x: "x".into(),
            y: "y".into(),
            bandwidth: 0.5,
            groupby,
            as_: ("lx".into(), "ly".into()),
            name: None,
        }
    }

    #[test]
    fn loess_grouped_fits_each_group_independently() {
        use arrow::array::StringArray;
        // Group A: increasing linear (y = 2x). Group B: decreasing linear (y = 20 - 2x).
        let n = 10;
        let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let ys_a: Vec<f64> = xs.iter().map(|&x| 2.0 * x).collect();
        let ys_b: Vec<f64> = xs.iter().map(|&x| 20.0 - 2.0 * x).collect();

        let mut group_col = vec!["A".to_string(); n];
        group_col.extend(vec!["B".to_string(); n]);
        let mut x_col = xs.clone();
        x_col.extend(xs.clone());
        let mut y_col = ys_a;
        y_col.extend(ys_b);

        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, false),
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(group_col)),
                Arc::new(Float64Array::from(x_col)),
                Arc::new(Float64Array::from(y_col)),
            ],
        )
        .unwrap();

        let spec = spec_with_groupby(Some(vec!["g".into()]));
        let out = apply(&spec, &batch).unwrap();

        assert_eq!(out.num_columns(), 3);
        assert_eq!(out.schema().field(2).name(), "g");
        assert_eq!(out.schema().field(2).data_type(), &DataType::Utf8);
        // 200 grid points per group (n_out), first-appearance order (A, B).
        assert_eq!(out.num_rows(), 400);

        let ly = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        let g = out.column(2).as_any().downcast_ref::<StringArray>().unwrap();

        // Group A (rows [0, 200)) is monotone non-decreasing.
        assert_eq!(g.value(0), "A");
        for i in 1..200 {
            assert!(
                ly.value(i) >= ly.value(i - 1) - 1e-6,
                "group A LOESS output should be non-decreasing at index {i}"
            );
        }
        // Group B (rows [200, 400)) is monotone non-increasing.
        assert_eq!(g.value(200), "B");
        for i in 201..400 {
            assert!(
                ly.value(i) <= ly.value(i - 1) + 1e-6,
                "group B LOESS output should be non-increasing at index {i}"
            );
        }
    }

    #[test]
    fn loess_grouped_int_group_dtype_preserved() {
        use arrow::array::Int64Array;
        let n = 10;
        let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut group_col = vec![0_i64; n];
        group_col.extend(vec![1_i64; n]);
        let mut x_col = xs.clone();
        x_col.extend(xs.clone());
        let mut y_col: Vec<f64> = xs.iter().map(|&x| 2.0 * x).collect();
        y_col.extend(xs.iter().map(|&x| 3.0 * x));

        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Int64, false),
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(group_col)),
                Arc::new(Float64Array::from(x_col)),
                Arc::new(Float64Array::from(y_col)),
            ],
        )
        .unwrap();

        let spec = spec_with_groupby(Some(vec!["g".into()]));
        let out = apply(&spec, &batch).unwrap();

        assert_eq!(out.schema().field(2).data_type(), &DataType::Int64);
        let g = out.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(g.value(0), 0);
        assert_eq!(g.value(200), 1);
    }

    #[test]
    fn loess_grouped_multi_column_partitions_distinct_combos() {
        use arrow::array::{Int64Array, StringArray};
        let n = 10;
        let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();

        let mut g1 = vec!["A".to_string(); n];
        g1.extend(vec!["B".to_string(); n]);
        let mut g2 = vec![1_i64; n];
        g2.extend(vec![2_i64; n]);
        let mut x_col = xs.clone();
        x_col.extend(xs.clone());
        let mut y_col: Vec<f64> = xs.iter().map(|&x| x).collect();
        y_col.extend(xs.iter().map(|&x| 2.0 * x));

        let schema = Arc::new(Schema::new(vec![
            Field::new("g1", DataType::Utf8, false),
            Field::new("g2", DataType::Int64, false),
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(g1)),
                Arc::new(Int64Array::from(g2)),
                Arc::new(Float64Array::from(x_col)),
                Arc::new(Float64Array::from(y_col)),
            ],
        )
        .unwrap();

        let spec = spec_with_groupby(Some(vec!["g1".into(), "g2".into()]));
        let out = apply(&spec, &batch).unwrap();

        assert_eq!(out.num_columns(), 4);
        assert_eq!(out.schema().field(2).name(), "g1");
        assert_eq!(out.schema().field(3).name(), "g2");
        assert_eq!(out.schema().field(3).data_type(), &DataType::Int64);
        // 2 distinct combos × 200 grid points each.
        assert_eq!(out.num_rows(), 400);
    }

    #[test]
    fn loess_grouped_below_threshold_group_emits_nan_rows_with_group_key() {
        use arrow::array::StringArray;
        // Group A: 10 healthy rows (increasing linear, y = 2x). Group B: 2
        // rows — below the <3-point LOESS threshold. Per the `apply_grouped`
        // doc comment (mirroring `smooth.rs::all_nan_output`), B must still
        // contribute N_OUT rows and keep its group key, not vanish.
        let n = 10;
        let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let ys_a: Vec<f64> = xs.iter().map(|&x| 2.0 * x).collect();

        let mut group_col = vec!["A".to_string(); n];
        group_col.extend(vec!["B".to_string(); 2]);
        let mut x_col = xs.clone();
        x_col.extend(vec![0.0, 1.0]);
        let mut y_col = ys_a;
        y_col.extend(vec![5.0, 7.0]);

        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, false),
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(group_col)),
                Arc::new(Float64Array::from(x_col)),
                Arc::new(Float64Array::from(y_col)),
            ],
        )
        .unwrap();

        let spec = spec_with_groupby(Some(vec!["g".into()]));
        let out = apply(&spec, &batch).unwrap();

        // Group A contributes N_OUT real rows; group B contributes N_OUT NaN
        // rows — the group is not dropped from the output.
        assert_eq!(out.num_rows(), 2 * N_OUT);
        assert_eq!(out.num_columns(), 3);

        let ly = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        let g = out.column(2).as_any().downcast_ref::<StringArray>().unwrap();

        assert_eq!(g.value(0), "A");
        assert!(!ly.value(0).is_nan(), "healthy group A must produce real fit values");

        assert_eq!(g.value(N_OUT), "B");
        for i in N_OUT..2 * N_OUT {
            assert!(ly.value(i).is_nan(), "below-threshold group B row {i} should be NaN, got {}", ly.value(i));
            assert_eq!(g.value(i), "B", "below-threshold group B must keep its group key at row {i}");
        }
    }

    #[test]
    fn loess_groupby_none_and_empty_vec_are_both_ungrouped() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let n = 10;
        let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 2.0 * x).collect();
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(xs)), Arc::new(Float64Array::from(ys))],
        )
        .unwrap();

        let out_none = apply(&spec_with_groupby(None), &batch).unwrap();
        let out_empty = apply(&spec_with_groupby(Some(vec![])), &batch).unwrap();

        assert_eq!(out_none.num_columns(), 2, "ungrouped output has no group column");
        assert_eq!(out_empty.num_columns(), 2, "empty groupby list is also ungrouped");
        assert_eq!(out_none.num_rows(), out_empty.num_rows());
    }

    #[test]
    fn loess_ungrouped_below_threshold_emits_nan_rows() {
        // Ungrouped input with 2 rows — below the <3-point LOESS fit threshold.
        // Per the `all_nan_output` doc comment, the ungrouped path emits [`N_OUT`]
        // NaN rows rather than 0 rows, mirroring the grouped-degenerate behavior
        // and `smooth.rs` semantics. No error, no warning, but no visible curve
        // (line mark filters non-finite points).
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![0.0, 1.0])),
                Arc::new(Float64Array::from(vec![5.0, 7.0])),
            ],
        )
        .unwrap();

        let spec = spec_with_groupby(None);
        let out = apply(&spec, &batch).unwrap();

        // Below-threshold ungrouped input should emit N_OUT NaN rows, not 0 rows.
        assert_eq!(out.num_rows(), N_OUT, "ungrouped below-threshold must emit N_OUT rows");
        assert_eq!(out.num_columns(), 2, "ungrouped output has no group column");

        let out_y = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        for i in 0..out.num_rows() {
            assert!(
                out_y.value(i).is_nan(),
                "ungrouped below-threshold output row {i} should be NaN, got {}",
                out_y.value(i)
            );
        }
    }
}
