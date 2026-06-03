//! Data transform: DataWindow — window functions (ranking, lag/lead, rolling).
//!
//! Supports: row_number, rank, dense_rank, lag, lead, first_value, last_value,
//! and rolling aggregates (sum, mean, count, min, max) with frame bounds.

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::transform::group_key::{groupby_key_at, is_groupby_supported_dtype, KeyValue};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct WindowOp {
    /// Window operation name.
    pub op: String,
    /// Input field (for aggregates and lag/lead).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub field: Option<String>,
    /// Output column name.
    #[serde(rename = "as")]
    pub as_: String,
    /// Offset for lag/lead operations.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub param: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct DataWindowSpec {
    pub ops: Vec<WindowOp>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub sort: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub groupby: Option<Vec<String>>,
    /// Window frame: (preceding, following). None means unbounded.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub frame: Option<(Option<i64>, Option<i64>)>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

pub(crate) fn apply(spec: &DataWindowSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let schema = batch.schema();
    let n_rows = batch.num_rows();

    if spec.ops.is_empty() {
        return Err(PyValueError::new_err("data_window: ops must be non-empty"));
    }

    // Build sort order indices.
    let sorted_indices = build_sort_order(batch, &schema, &spec.sort)?;

    // Build group partitions.
    let partitions = build_partitions(batch, &schema, &spec.groupby, &sorted_indices)?;

    // For each op, compute the window function.
    let mut new_cols: Vec<ArrayRef> = Vec::with_capacity(spec.ops.len());
    let mut new_fields: Vec<Field> = Vec::with_capacity(spec.ops.len());

    for op in &spec.ops {
        let col_values = compute_window_op(batch, &schema, op, &partitions, &sorted_indices, spec.frame, n_rows)?;
        new_cols.push(Arc::new(Float64Array::from(col_values)));
        new_fields.push(Field::new(&op.as_, DataType::Float64, true));
    }

    // Build output: original columns + window columns.
    let mut fields: Vec<Field> = schema.fields().iter().map(|f| f.as_ref().clone()).collect();
    fields.extend(new_fields);
    let out_schema = Arc::new(Schema::new(fields));

    let mut columns: Vec<ArrayRef> = (0..batch.num_columns())
        .map(|i| batch.column(i).clone())
        .collect();
    columns.extend(new_cols);

    RecordBatch::try_new(out_schema, columns)
        .map_err(|e| PyValueError::new_err(format!("data_window: {e}")))
}

/// Build sort order. Returns indices in sorted order.
fn build_sort_order(
    batch: &RecordBatch,
    schema: &arrow::datatypes::SchemaRef,
    sort: &Option<Vec<String>>,
) -> PyResult<Vec<usize>> {
    let n_rows = batch.num_rows();
    let mut indices: Vec<usize> = (0..n_rows).collect();

    if let Some(sort_fields) = sort {
        let sort_cols: Vec<(usize, &dyn Array)> = sort_fields
            .iter()
            .map(|f| {
                let idx = schema.index_of(f).map_err(|_| {
                    PyValueError::new_err(format!("data_window: sort column '{f}' not found"))
                })?;
                Ok((idx, batch.column(idx).as_ref()))
            })
            .collect::<PyResult<_>>()?;

        indices.sort_by(|&a, &b| {
            for &(_, col) in &sort_cols {
                let cmp = compare_rows(col, a, b);
                if cmp != std::cmp::Ordering::Equal {
                    return cmp;
                }
            }
            std::cmp::Ordering::Equal
        });
    }

    Ok(indices)
}

fn compare_rows(col: &dyn Array, a: usize, b: usize) -> std::cmp::Ordering {
    if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
        let va = if arr.is_null(a) { f64::NAN } else { arr.value(a) };
        let vb = if arr.is_null(b) { f64::NAN } else { arr.value(b) };
        va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
    } else if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
        let va = if arr.is_null(a) { "" } else { arr.value(a) };
        let vb = if arr.is_null(b) { "" } else { arr.value(b) };
        va.cmp(vb)
    } else {
        std::cmp::Ordering::Equal
    }
}

/// Build partitions: groups of sorted indices that share the same group key.
fn build_partitions(
    batch: &RecordBatch,
    schema: &arrow::datatypes::SchemaRef,
    groupby: &Option<Vec<String>>,
    sorted_indices: &[usize],
) -> PyResult<Vec<Vec<usize>>> {
    let groupby = match groupby {
        Some(g) => g.clone(),
        None => return Ok(vec![sorted_indices.to_vec()]),
    };

    if groupby.is_empty() {
        return Ok(vec![sorted_indices.to_vec()]);
    }

    let group_col_indices: Vec<usize> = groupby
        .iter()
        .map(|g| {
            schema.index_of(g).map_err(|_| {
                PyValueError::new_err(format!("data_window: groupby column '{g}' not found"))
            })
        })
        .collect::<PyResult<_>>()?;

    // Validate groupby dtypes and capture them for shared key extraction (FA-7).
    let group_dtypes: Vec<DataType> = group_col_indices
        .iter()
        .zip(groupby.iter())
        .map(|(&gi, g)| {
            let dt = schema.field(gi).data_type().clone();
            if !is_groupby_supported_dtype(&dt) {
                return Err(PyValueError::new_err(format!(
                    "data_window: groupby column '{g}' has unsupported dtype {dt:?}; \
                     supported: Utf8/LargeUtf8, Float64/Float32, \
                     Int8/Int16/Int32/Int64, UInt8/UInt16/UInt32/UInt64, Boolean"
                )));
            }
            Ok(dt)
        })
        .collect::<PyResult<_>>()?;

    // Build group key per row, then partition sorted_indices by key.
    let extract = |idx: usize| -> PyResult<Vec<KeyValue>> {
        group_col_indices
            .iter()
            .enumerate()
            .map(|(gpos, &gi)| {
                groupby_key_at(batch.column(gi).as_ref(), &group_dtypes[gpos], idx).ok_or_else(
                    || {
                        PyValueError::new_err(format!(
                            "data_window: internal error extracting groupby key at row {idx}"
                        ))
                    },
                )
            })
            .collect()
    };

    let mut partitions: Vec<Vec<usize>> = Vec::new();
    let mut current_key: Option<Vec<KeyValue>> = None;
    let mut current_partition: Vec<usize> = Vec::new();

    for &idx in sorted_indices {
        let key = extract(idx)?;

        if current_key.as_ref() == Some(&key) {
            current_partition.push(idx);
        } else {
            if !current_partition.is_empty() {
                partitions.push(std::mem::take(&mut current_partition));
            }
            current_key = Some(key);
            current_partition.push(idx);
        }
    }
    if !current_partition.is_empty() {
        partitions.push(current_partition);
    }

    Ok(partitions)
}

fn compute_window_op(
    batch: &RecordBatch,
    schema: &arrow::datatypes::SchemaRef,
    op: &WindowOp,
    partitions: &[Vec<usize>],
    _sorted_indices: &[usize],
    frame: Option<(Option<i64>, Option<i64>)>,
    n_rows: usize,
) -> PyResult<Vec<f64>> {
    let mut result = vec![f64::NAN; n_rows];

    match op.op.as_str() {
        "row_number" => {
            for partition in partitions {
                for (rank, &idx) in partition.iter().enumerate() {
                    result[idx] = (rank + 1) as f64;
                }
            }
        }
        "rank" => {
            for partition in partitions {
                let field = op.field.as_deref().unwrap_or("");
                if field.is_empty() {
                    // Without a field, rank is just row_number.
                    for (rank, &idx) in partition.iter().enumerate() {
                        result[idx] = (rank + 1) as f64;
                    }
                } else {
                    let col_idx = schema.index_of(field).map_err(|_| {
                        PyValueError::new_err(format!("data_window: column '{field}' not found"))
                    })?;
                    let col = batch.column(col_idx);
                    let mut rank = 1usize;
                    let mut i = 0;
                    while i < partition.len() {
                        let mut j = i + 1;
                        while j < partition.len()
                            && compare_rows(col.as_ref(), partition[i], partition[j])
                                == std::cmp::Ordering::Equal
                        {
                            j += 1;
                        }
                        for k in i..j {
                            result[partition[k]] = rank as f64;
                        }
                        rank += j - i;
                        i = j;
                    }
                }
            }
        }
        "dense_rank" => {
            for partition in partitions {
                let field = op.field.as_deref().unwrap_or("");
                if field.is_empty() {
                    for (rank, &idx) in partition.iter().enumerate() {
                        result[idx] = (rank + 1) as f64;
                    }
                } else {
                    let col_idx = schema.index_of(field).map_err(|_| {
                        PyValueError::new_err(format!("data_window: column '{field}' not found"))
                    })?;
                    let col = batch.column(col_idx);
                    let mut rank = 1usize;
                    let mut i = 0;
                    while i < partition.len() {
                        let mut j = i + 1;
                        while j < partition.len()
                            && compare_rows(col.as_ref(), partition[i], partition[j])
                                == std::cmp::Ordering::Equal
                        {
                            j += 1;
                        }
                        for k in i..j {
                            result[partition[k]] = rank as f64;
                        }
                        rank += 1;
                        i = j;
                    }
                }
            }
        }
        "lag" | "lead" => {
            let field = op.field.as_deref().ok_or_else(|| {
                PyValueError::new_err(format!("data_window: {} requires 'field'", op.op))
            })?;
            let col_idx = schema.index_of(field).map_err(|_| {
                PyValueError::new_err(format!("data_window: column '{field}' not found"))
            })?;
            let offset = op.param.unwrap_or(1);
            let col = batch.column(col_idx);

            for partition in partitions {
                for (pos, &idx) in partition.iter().enumerate() {
                    let source_pos = if op.op == "lag" {
                        pos as i64 - offset
                    } else {
                        pos as i64 + offset
                    };
                    if source_pos >= 0 && (source_pos as usize) < partition.len() {
                        let src_idx = partition[source_pos as usize];
                        result[idx] = get_f64_value(col.as_ref(), src_idx);
                    }
                    // else remains NAN
                }
            }
        }
        "first_value" => {
            let field = op.field.as_deref().ok_or_else(|| {
                PyValueError::new_err("data_window: first_value requires 'field'")
            })?;
            let col_idx = schema.index_of(field).map_err(|_| {
                PyValueError::new_err(format!("data_window: column '{field}' not found"))
            })?;
            let col = batch.column(col_idx);

            for partition in partitions {
                if partition.is_empty() {
                    continue;
                }
                let first_val = get_f64_value(col.as_ref(), partition[0]);
                for &idx in partition {
                    result[idx] = first_val;
                }
            }
        }
        "last_value" => {
            let field = op.field.as_deref().ok_or_else(|| {
                PyValueError::new_err("data_window: last_value requires 'field'")
            })?;
            let col_idx = schema.index_of(field).map_err(|_| {
                PyValueError::new_err(format!("data_window: column '{field}' not found"))
            })?;
            let col = batch.column(col_idx);

            for partition in partitions {
                if partition.is_empty() {
                    continue;
                }
                let last_val = get_f64_value(col.as_ref(), *partition.last().unwrap());
                for &idx in partition {
                    result[idx] = last_val;
                }
            }
        }
        // Rolling aggregates: sum, mean, count, min, max
        "sum" | "mean" | "count" | "min" | "max" => {
            let field = op.field.as_deref().ok_or_else(|| {
                PyValueError::new_err(format!(
                    "data_window: rolling {} requires 'field'",
                    op.op
                ))
            })?;
            let col_idx = schema.index_of(field).map_err(|_| {
                PyValueError::new_err(format!("data_window: column '{field}' not found"))
            })?;
            let col = batch.column(col_idx);

            let (preceding, following) = frame.unwrap_or((None, None));

            for partition in partitions {
                for (pos, &idx) in partition.iter().enumerate() {
                    // Vega/Altair convention: frame=(preceding, following) where
                    // `preceding` is NEGATIVE to count rows *before* the current
                    // row (e.g. frame=(-2, 0) = trailing window of 3 rows).
                    // `following` is POSITIVE to count rows *after* the current row.
                    // start = pos + preceding  (preceding ≤ 0, clamped to 0)
                    // end   = pos + following  (following ≥ 0, inclusive → +1)
                    let start = match preceding {
                        Some(p) => (pos as i64 + p).max(0) as usize,
                        None => 0,
                    };
                    let end = match following {
                        Some(f) => ((pos as i64 + f + 1) as usize).min(partition.len()),
                        None => partition.len(),
                    };

                    let vals: Vec<f64> = (start..end)
                        .map(|p| get_f64_value(col.as_ref(), partition[p]))
                        .filter(|v| !v.is_nan())
                        .collect();

                    result[idx] = rolling_agg(&vals, &op.op);
                }
            }
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "data_window: unknown window op '{other}'"
            )));
        }
    }

    Ok(result)
}

fn get_f64_value(col: &dyn Array, row: usize) -> f64 {
    if col.is_null(row) {
        return f64::NAN;
    }
    if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
        return arr.value(row);
    }
    f64::NAN
}

fn rolling_agg(vals: &[f64], op: &str) -> f64 {
    if vals.is_empty() {
        return if op == "count" { 0.0 } else { f64::NAN };
    }
    match op {
        "sum" => vals.iter().sum(),
        "mean" => vals.iter().sum::<f64>() / vals.len() as f64,
        "count" => vals.len() as f64,
        "min" => vals.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        "max" => vals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
        _ => f64::NAN,
    }
}

// ─── PyO3 wrapper ──────────────────────────────────────────────────────────

use pyo3::prelude::*;
use crate::transform::core::TransformSpec;

#[pyclass(module = "ferrum._core", name = "DataWindow")]
#[derive(Debug, Clone)]
pub(crate) struct PyDataWindow(pub(crate) TransformSpec);

#[pymethods]
impl PyDataWindow {
    #[new]
    #[pyo3(signature = (*, name = None))]
    fn new(name: Option<String>) -> Self {
        PyDataWindow(TransformSpec::DataWindow(DataWindowSpec {
            ops: Vec::new(),
            sort: None,
            groupby: None,
            frame: None,
            name,
        }))
    }

    fn __repr__(&self) -> String {
        "DataWindow(...)".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float64Array;
    use arrow::datatypes::{Field, Schema};
    use std::sync::Arc;

    #[test]
    fn window_rolling_mean_with_frame() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]))],
        )
        .unwrap();

        let spec = DataWindowSpec {
            ops: vec![WindowOp {
                op: "mean".into(),
                field: Some("x".into()),
                as_: "rolling_mean".into(),
                param: None,
            }],
            sort: None,
            groupby: None,
            // Vega convention: preceding is NEGATIVE (rows before current),
            // following is POSITIVE (rows after current).
            // frame=(-1, 1) = 1 preceding + current + 1 following (symmetric).
            frame: Some((Some(-1), Some(1))),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 5);
        let means = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        // Row 0: window [0..=1] → mean(1,2) = 1.5
        assert!((means.value(0) - 1.5).abs() < 1e-12);
        // Row 1: window [0..=2] → mean(1,2,3) = 2.0
        assert!((means.value(1) - 2.0).abs() < 1e-12);
        // Row 2: window [1..=3] → mean(2,3,4) = 3.0
        assert!((means.value(2) - 3.0).abs() < 1e-12);
        // Row 3: window [2..=4] → mean(3,4,5) = 4.0
        assert!((means.value(3) - 4.0).abs() < 1e-12);
        // Row 4: window [3..=4] → mean(4,5) = 4.5
        assert!((means.value(4) - 4.5).abs() < 1e-12);
    }

    #[test]
    fn window_row_number() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0]))],
        )
        .unwrap();

        let spec = DataWindowSpec {
            ops: vec![WindowOp {
                op: "row_number".into(),
                field: None,
                as_: "rn".into(),
                param: None,
            }],
            sort: None,
            groupby: None,
            frame: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let rn = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(rn.value(0), 1.0);
        assert_eq!(rn.value(1), 2.0);
        assert_eq!(rn.value(2), 3.0);
    }

    #[test]
    fn window_lag_lead() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0]))],
        )
        .unwrap();

        let spec = DataWindowSpec {
            ops: vec![
                WindowOp {
                    op: "lag".into(),
                    field: Some("x".into()),
                    as_: "lag_x".into(),
                    param: Some(1),
                },
                WindowOp {
                    op: "lead".into(),
                    field: Some("x".into()),
                    as_: "lead_x".into(),
                    param: Some(1),
                },
            ],
            sort: None,
            groupby: None,
            frame: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let lag = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        let lead = out.column(2).as_any().downcast_ref::<Float64Array>().unwrap();

        assert!(lag.value(0).is_nan()); // No preceding
        assert_eq!(lag.value(1), 1.0);
        assert_eq!(lag.value(2), 2.0);
        assert_eq!(lead.value(0), 2.0);
        assert_eq!(lead.value(1), 3.0);
        assert!(lead.value(3).is_nan()); // No following
    }

    fn make_val_batch(vals: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(vals))]).unwrap()
    }

    fn mean_op() -> WindowOp {
        WindowOp { op: "mean".into(), field: Some("x".into()), as_: "m".into(), param: None }
    }

    fn get_means(out: &RecordBatch) -> Vec<f64> {
        let arr = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        (0..arr.len()).map(|i| arr.value(i)).collect()
    }

    // Trailing window: frame=(-2, 0) on [1,2,3,4,5].
    // Expected: [1.0, 1.5, 2.0, 3.0, 4.0]
    #[test]
    fn window_trailing_frame_negative_preceding() {
        let batch = make_val_batch(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let spec = DataWindowSpec {
            ops: vec![mean_op()],
            sort: None, groupby: None,
            frame: Some((Some(-2), Some(0))),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let means = get_means(&out);
        assert!((means[0] - 1.0).abs() < 1e-12, "pos=0: {}", means[0]);
        assert!((means[1] - 1.5).abs() < 1e-12, "pos=1: {}", means[1]);
        assert!((means[2] - 2.0).abs() < 1e-12, "pos=2: {}", means[2]);
        assert!((means[3] - 3.0).abs() < 1e-12, "pos=3: {}", means[3]);
        assert!((means[4] - 4.0).abs() < 1e-12, "pos=4: {}", means[4]);
    }

    // Leading window: frame=(0, 2) on [1,2,3,4,5].
    // Expected: [2.0, 3.0, 4.0, 4.5, 5.0]
    #[test]
    fn window_leading_frame_positive_following() {
        let batch = make_val_batch(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let spec = DataWindowSpec {
            ops: vec![mean_op()],
            sort: None, groupby: None,
            frame: Some((Some(0), Some(2))),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let means = get_means(&out);
        // pos=0: window [0..=2] → mean(1,2,3) = 2.0
        assert!((means[0] - 2.0).abs() < 1e-12, "pos=0: {}", means[0]);
        // pos=1: window [1..=3] → mean(2,3,4) = 3.0
        assert!((means[1] - 3.0).abs() < 1e-12, "pos=1: {}", means[1]);
        // pos=2: window [2..=4] → mean(3,4,5) = 4.0
        assert!((means[2] - 4.0).abs() < 1e-12, "pos=2: {}", means[2]);
        // pos=3: window [3..=4] → mean(4,5) = 4.5
        assert!((means[3] - 4.5).abs() < 1e-12, "pos=3: {}", means[3]);
        // pos=4: window [4..=4] → mean(5) = 5.0
        assert!((means[4] - 5.0).abs() < 1e-12, "pos=4: {}", means[4]);
    }

    // Symmetric window: frame=(-1, 1) on [1,2,3,4,5].
    // Expected: [1.5, 2.0, 3.0, 4.0, 4.5]
    #[test]
    fn window_symmetric_frame() {
        let batch = make_val_batch(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let spec = DataWindowSpec {
            ops: vec![mean_op()],
            sort: None, groupby: None,
            frame: Some((Some(-1), Some(1))),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let means = get_means(&out);
        // pos=0: window [0..=1] → mean(1,2) = 1.5
        assert!((means[0] - 1.5).abs() < 1e-12, "pos=0: {}", means[0]);
        // pos=1: window [0..=2] → mean(1,2,3) = 2.0
        assert!((means[1] - 2.0).abs() < 1e-12, "pos=1: {}", means[1]);
        // pos=2: window [1..=3] → mean(2,3,4) = 3.0
        assert!((means[2] - 3.0).abs() < 1e-12, "pos=2: {}", means[2]);
        // pos=3: window [2..=4] → mean(3,4,5) = 4.0
        assert!((means[3] - 4.0).abs() < 1e-12, "pos=3: {}", means[3]);
        // pos=4: window [3..=4] → mean(4,5) = 4.5
        assert!((means[4] - 4.5).abs() < 1e-12, "pos=4: {}", means[4]);
    }
}
