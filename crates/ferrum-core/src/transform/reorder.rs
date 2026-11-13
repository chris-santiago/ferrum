//! Reorder transform — apply a permutation to the input batch.
//!
//! `by` names an Int64 index column. On the row axis (default),
//! output.row[i] == input.row[by[i]]. On the column axis, the batch's numeric
//! columns other than the order's `Linkage` label column (the ones
//! `Linkage(axis="columns")` clusters, see [`feature_column_indices`]) are
//! permuted in place: the i-th numeric slot receives numeric column by[i], and
//! every other column keeps its position. The `order` of a `Linkage` with
//! `labels` names only the rows whose label is not null.
//! When `drop_index=true` (default), the `by` column is omitted from output.
//!
//! Used by `clustermap()` to reorder rows and columns by each Linkage's
//! `order` named output, so the unpivoted heatmap's first-appearance band
//! domains follow the dendrogram leaf order on both axes.

use arrow::array::{Array, ArrayRef, Int64Array, RecordBatch, UInt64Array};
use arrow::compute::take;
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::transform::linkage::{feature_column_indices, LinkageAxis, LinkageOrderMeta};
use crate::transform::vocabulary::Vocabulary;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ReorderSpec {
    pub by: String,
    #[serde(default = "default_drop_index")]
    pub drop_index: bool,
    /// Optional name of a sibling transform's named output to read the index
    /// column from (Phase 9 finalize). When set, Reorder uses `<from>` as the
    /// batch source for the `by` column, but applies the permutation to the
    /// chained input batch. Used by clustermap to consume Linkage's
    /// `<name>_order` secondary output across the transform chain.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub from: Option<String>,
    /// Which axis the permutation applies to. `Columns` requires `from`: a
    /// column order has one index per feature column, which the chained
    /// batch's own rows cannot supply.
    #[serde(skip_serializing_if = "is_rows", default = "default_axis")]
    pub axis: LinkageAxis,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

fn default_drop_index() -> bool { true }
fn default_axis() -> LinkageAxis { LinkageAxis::Rows }
fn is_rows(axis: &LinkageAxis) -> bool { *axis == LinkageAxis::Rows }

pub(crate) fn apply(spec: &ReorderSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    apply_with_outputs(spec, batch, None)
}

pub(crate) fn apply_with_outputs(
    spec: &ReorderSpec,
    batch: &RecordBatch,
    named_outputs: Option<&std::collections::HashMap<String, RecordBatch>>,
) -> PyResult<RecordBatch> {
    if spec.axis == LinkageAxis::Columns && spec.from.is_none() {
        return Err(PyValueError::new_err(
            "stat_reorder: axis='columns' requires from= naming a column linkage's order output"));
    }
    // Determine which batch to read the index column from.
    let index_batch: &RecordBatch = match (&spec.from, named_outputs) {
        (Some(name), Some(outs)) => outs.get(name).ok_or_else(|| PyValueError::new_err(
            format!("stat_reorder: from='{name}' not found in named outputs; available: {:?}",
                outs.keys().collect::<Vec<_>>())))?,
        (Some(name), None) => return Err(PyValueError::new_err(format!(
            "stat_reorder: from='{name}' set but no transform context available"))),
        (None, _) => batch,
    };
    let schema = index_batch.schema();
    let idx = schema.index_of(&spec.by).map_err(|_| PyValueError::new_err(
        format!("stat_reorder: index column '{}' not found", spec.by)))?;
    if schema.field(idx).data_type() != &DataType::Int64 {
        return Err(PyValueError::new_err(format!(
            "stat_reorder: '{}' must be Int64 (got {:?})",
            spec.by, schema.field(idx).data_type())));
    }
    let idx_arr = index_batch.column(idx).as_any().downcast_ref::<Int64Array>()
        .ok_or_else(|| PyValueError::new_err(format!(
            "stat_reorder: expected Int64Array for index column '{}'", spec.by)))?;
    let linkage = spec.from.as_ref().and_then(|_| LinkageOrderMeta::read(index_batch));
    let (cols, fields) = match spec.axis {
        LinkageAxis::Rows => permute_rows(spec, idx_arr, batch, linkage.as_ref())?,
        LinkageAxis::Columns => permute_columns(spec, idx_arr, batch, linkage.as_ref())?,
    };
    let (out_fields, out_cols): (Vec<Field>, Vec<ArrayRef>) = fields
        .into_iter()
        .zip(cols)
        .filter(|(f, _)| !(spec.drop_index && f.name() == &spec.by))
        .unzip();
    let out_schema = Arc::new(Schema::new(out_fields));
    RecordBatch::try_new(out_schema, out_cols)
        .map_err(|e| PyValueError::new_err(format!("stat_reorder: {e}")))
}

type Columns = (Vec<ArrayRef>, Vec<Field>);

/// Validate every index against `len` and return them as positions.
fn checked_indices(idx_arr: &Int64Array, len: usize) -> PyResult<Vec<usize>> {
    idx_arr.values().iter().map(|&v| {
        if v < 0 || (v as usize) >= len {
            return Err(PyValueError::new_err(format!(
                "stat_reorder: index {v} out of bounds (n={len})")));
        }
        Ok(v as usize)
    }).collect()
}

/// Permute the batch's rows by `idx_arr`. An order read `from` a named output
/// must be as long as the batch, except the order of a `Linkage` with
/// `labels` (`linkage` is its record): that order names only the rows whose
/// label is not null and keeps exactly those, in its order, over a batch that
/// has the rows the linkage read.
fn permute_rows(
    spec: &ReorderSpec,
    idx_arr: &Int64Array,
    batch: &RecordBatch,
    linkage: Option<&LinkageOrderMeta>,
) -> PyResult<Columns> {
    if let Some(from) = spec.from.as_deref() {
        match linkage {
            Some(meta) if meta.rows != batch.num_rows() => {
                return Err(PyValueError::new_err(format!(
                    "stat_reorder: from='{from}' orders the {} rows its linkage read but batch has {} rows; \
                     place the Reorder before any step that adds or removes rows",
                    meta.rows, batch.num_rows())));
            }
            Some(_) => {}
            None if idx_arr.len() != batch.num_rows() => {
                return Err(PyValueError::new_err(format!(
                    "stat_reorder: from='{from}' produced {} indices but batch has {} rows",
                    idx_arr.len(), batch.num_rows())));
            }
            None => {}
        }
    }
    // arrow::compute::take requires UInt64 indices.
    let take_indices: UInt64Array = checked_indices(idx_arr, batch.num_rows())?
        .into_iter().map(|i| i as u64).collect::<Vec<u64>>().into();
    let cols = batch.columns().iter()
        .map(|c| take(c, &take_indices, None)
            .map_err(|e| PyValueError::new_err(format!("stat_reorder: take: {e}"))))
        .collect::<PyResult<Vec<_>>>()?;
    let fields = batch.schema().fields().iter().map(|f| f.as_ref().clone()).collect();
    Ok((cols, fields))
}

/// Permute the numeric feature columns by `idx_arr`: the numeric columns
/// other than the order's `Linkage` label column, the columns that linkage
/// clustered. The empty order of a labelled linkage (every label null, so no
/// observation) leaves the columns in place.
fn permute_columns(
    spec: &ReorderSpec,
    idx_arr: &Int64Array,
    batch: &RecordBatch,
    linkage: Option<&LinkageOrderMeta>,
) -> PyResult<Columns> {
    let from = spec.from.as_deref().unwrap_or_default();
    let schema = batch.schema();
    let slots = feature_column_indices(&schema, linkage.map(|m| m.label.as_str()), "stat_reorder")?;
    if idx_arr.is_empty() && linkage.is_some() {
        return Ok((batch.columns().to_vec(), schema.fields().iter().map(|f| f.as_ref().clone()).collect()));
    }
    if idx_arr.len() != slots.len() {
        return Err(PyValueError::new_err(format!(
            "stat_reorder: from='{from}' produced {} indices but batch has {} numeric columns",
            idx_arr.len(), slots.len())));
    }
    let order = checked_indices(idx_arr, slots.len())?;
    let mut seen = vec![false; slots.len()];
    for &k in &order {
        if std::mem::replace(&mut seen[k], true) {
            return Err(PyValueError::new_err(format!(
                "stat_reorder: from='{from}' repeats column index {k}; a column order must be a permutation")));
        }
    }
    let mut cols = batch.columns().to_vec();
    let mut fields: Vec<Field> = schema.fields().iter().map(|f| f.as_ref().clone()).collect();
    for (&slot, &k) in slots.iter().zip(&order) {
        cols[slot] = batch.column(slots[k]).clone();
        fields[slot] = schema.field(slots[k]).clone();
    }
    Ok((cols, fields))
}

// PyO3 wrapper — mirror Unpivot pattern.
use crate::transform::core::TransformSpec;

/// Row-reordering transform driven by an integer index column.
///
/// Permutes all rows of the input batch according to the values in the
/// ``by`` column, which must be an integer column containing a permutation
/// of ``0..n-1`` (when ``from_`` names the ``order`` of a ``Linkage`` with
/// ``labels``, it names only the rows whose label is not null and the others
/// are dropped, as that ``Linkage`` counted them; the batch must have the
/// rows that ``Linkage`` read). When ``from_`` is set, the batch is consumed from a
/// named sibling transform's output (e.g. a ``Linkage`` that emits
/// ``new_idx``) rather than from the primary data source.
///
/// Parameters
/// ----------
/// by : str
///     Integer column containing the target row index permutation. Must be
///     of type Int64 or UInt64 and contain a valid permutation of
///     ``0..n-1``.
/// drop_index : bool, default True
///     When True, the ``by`` column is removed from the output. Set to
///     False to retain it for debugging.
/// from_ : str, optional
///     Name of a sibling transform whose output supplies this transform's
///     input batch. Used in clustermap pipelines where ``Linkage`` feeds
///     directly into ``Reorder``.
/// axis : {"rows", "columns"}, default "rows"
///     ``"rows"`` permutes the batch's rows. ``"columns"`` permutes its
///     numeric columns other than its ``Linkage``'s ``labels`` column (the
///     ones ``Linkage(axis="columns")`` clusters) in
///     place, so a following ``Unpivot(value_vars=None)`` emits the column
///     blocks in linkage leaf order; it requires ``from_``.
/// name : str, optional
///     Named output label for sibling ``Reorder(from_=...)`` lookup.
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> fm.Chart(df).mark_bar().encode(
/// ...     x=fm.X("cut", sort=fm.Reorder(field="price", fn_="mean")),
/// ...     y="price",
/// ... )
#[pyclass(eq, module = "ferrum._core", name = "Reorder")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyReorder(pub(crate) TransformSpec);

#[pymethods]
impl PyReorder {
    #[new]
    #[pyo3(signature = (by, *, drop_index = true, from_ = None, axis = "rows", name = None))]
    fn new(
        by: &str,
        drop_index: bool,
        from_: Option<String>,
        axis: &str,
        name: Option<String>,
    ) -> PyResult<Self> {
        if by.is_empty() {
            return Err(PyValueError::new_err("Reorder: by must be non-empty"));
        }
        let axis = LinkageAxis::parse_or_refuse("axis", axis)
            .map_err(|e| PyValueError::new_err(format!("Reorder: {e}")))?;
        if axis == LinkageAxis::Columns && from_.is_none() {
            return Err(PyValueError::new_err(
                "Reorder: axis='columns' requires from_ naming a column linkage's order output"));
        }
        Ok(PyReorder(TransformSpec::Reorder(ReorderSpec {
            by: by.into(), drop_index, from: from_, axis, name,
        })))
    }
    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Reorder(s) => format!(
                "Reorder(by='{}', drop_index={}, from_={:?}, axis='{}', name={:?})",
                s.by, s.drop_index, s.from, s.axis.as_str(), s.name),
            #[allow(unreachable_patterns)] _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int64Array};

    fn make_batch_with_idx(idx: Vec<i64>, vals: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("new_idx", DataType::Int64, false),
            Field::new("v", DataType::Float64, false),
        ]));
        RecordBatch::try_new(schema, vec![
            Arc::new(Int64Array::from(idx)),
            Arc::new(Float64Array::from(vals)),
        ]).unwrap()
    }

    #[test]
    fn reorder_5_rows_correctness() {
        let batch = make_batch_with_idx(vec![4, 3, 2, 1, 0], vec![10.0, 20.0, 30.0, 40.0, 50.0]);
        let spec = ReorderSpec { by: "new_idx".into(), drop_index: true, from: None, axis: LinkageAxis::Rows, name: None };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_columns(), 1);
        let v = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!((0..5).map(|i| v.value(i)).collect::<Vec<_>>(),
                   vec![50.0, 40.0, 30.0, 20.0, 10.0]);
    }

    #[test]
    fn reorder_identity_permutation_unchanged() {
        let batch = make_batch_with_idx(vec![0, 1, 2, 3, 4], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let spec = ReorderSpec { by: "new_idx".into(), drop_index: true, from: None, axis: LinkageAxis::Rows, name: None };
        let out = apply(&spec, &batch).unwrap();
        let v = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!((0..5).map(|i| v.value(i)).collect::<Vec<_>>(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn reorder_keeps_index_when_drop_false() {
        let batch = make_batch_with_idx(vec![1, 0], vec![10.0, 20.0]);
        let spec = ReorderSpec { by: "new_idx".into(), drop_index: false, from: None, axis: LinkageAxis::Rows, name: None };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_columns(), 2);
    }

    #[test]
    fn reorder_out_of_bounds_errors() {
        pyo3::Python::initialize();
        let batch = make_batch_with_idx(vec![0, 99], vec![1.0, 2.0]);
        let spec = ReorderSpec { by: "new_idx".into(), drop_index: true, from: None, axis: LinkageAxis::Rows, name: None };
        let err = apply(&spec, &batch).unwrap_err().to_string();
        assert!(err.contains("out of bounds"));
    }

    // ---------- column axis ----------

    use crate::transform::context::TransformContext;
    use crate::transform::core::{apply_transforms_named, FINAL_OUTPUT_KEY};
    use crate::transform::linkage::{DistanceMetric, LinkageMethod, LinkageSpec};
    use crate::transform::unpivot::UnpivotSpec;
    use arrow::array::StringArray;
    use std::collections::HashMap;

    const COLS: [&str; 4] = ["A", "B", "C", "D"];

    /// A wide clustermap-shaped batch: a Utf8 label column, then four
    /// numeric columns in which A pairs with D and B pairs with C, so the
    /// column linkage's leaf order cannot be the schema order A, B, C, D.
    fn wide_batch() -> RecordBatch {
        let data: [[f64; 4]; 4] = [
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 0.0, 10.0, 0.0],
            [11.0, 0.0, 11.0, 1.0],
            [1.1, 2.0, 3.0, 4.2],
        ];
        let mut fields = vec![Field::new("row", DataType::Utf8, false)];
        let mut cols: Vec<ArrayRef> =
            vec![Arc::new(StringArray::from(vec!["r0", "r1", "r2", "r3"]))];
        for (name, values) in COLS.iter().zip(data) {
            fields.push(Field::new(*name, DataType::Float64, false));
            cols.push(Arc::new(Float64Array::from(values.to_vec())));
        }
        RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).unwrap()
    }

    fn linkage(axis: LinkageAxis, name: &str) -> TransformSpec {
        TransformSpec::Linkage(LinkageSpec {
            method: LinkageMethod::Ward,
            metric: DistanceMetric::Euclidean,
            axis,
            z_score: None,
            standard_scale: None, labels: None,
            name: Some(name.into()),
        })
    }

    fn order_reorder(axis: LinkageAxis, from: &str) -> TransformSpec {
        TransformSpec::Reorder(ReorderSpec {
            by: "original_idx".into(), drop_index: false, from: Some(from.into()), axis, name: None,
        })
    }

    /// The clustermap center chain: both linkages, the row and column
    /// Reorders, then an Unpivot that takes its blocks from the batch.
    fn clustermap_chain() -> HashMap<String, RecordBatch> {
        let specs = vec![
            linkage(LinkageAxis::Rows, "row_link"),
            linkage(LinkageAxis::Columns, "col_link"),
            order_reorder(LinkageAxis::Rows, "row_link_order"),
            order_reorder(LinkageAxis::Columns, "col_link_order"),
            TransformSpec::Unpivot(UnpivotSpec {
                id_vars: vec!["row".into()],
                value_vars: None,
                var_name: "column".into(),
                value_name: "value".into(),
                name: None,
            }),
        ];
        apply_transforms_named(&specs, &wide_batch(), &TransformContext::default()).unwrap().0
    }

    fn first_appearance(batch: &RecordBatch, col: &str) -> Vec<String> {
        let arr = batch.column_by_name(col).unwrap().as_any().downcast_ref::<StringArray>().unwrap();
        let mut seen: Vec<String> = Vec::new();
        for v in arr.iter().flatten() {
            if !seen.iter().any(|s| s == v) {
                seen.push(v.to_string());
            }
        }
        seen
    }

    /// The heatmap's column band domain is the unpivoted `column` field in
    /// first-appearance order. Every column-dendrogram leaf (coords node
    /// `j < n`, the j-th feature column) sits at x = its display position,
    /// so leaf `j` must sit over band position `x`: the tree and the
    /// heatmap agree on which column each leaf is.
    #[test]
    fn column_dendrogram_leaves_sit_over_their_heatmap_columns() {
        let outs = clustermap_chain();
        let columns = first_appearance(&outs[FINAL_OUTPUT_KEY], "column");
        let order = outs["col_link_order"].column_by_name("original_idx").unwrap()
            .as_any().downcast_ref::<Int64Array>().unwrap().values().to_vec();
        assert_ne!(order, [0, 1, 2, 3], "fixture must cluster to a non-identity column order");

        let coords = &outs["col_link_coords"];
        let ids = coords.column_by_name("node_id").unwrap().as_any().downcast_ref::<Int64Array>().unwrap();
        let xs = coords.column_by_name("x").unwrap().as_any().downcast_ref::<Float64Array>().unwrap();
        let mut leaves = 0;
        for i in 0..coords.num_rows() {
            let j = ids.value(i) as usize;
            if j < COLS.len() {
                leaves += 1;
                let band_pos = columns.iter().position(|c| c == COLS[j]).unwrap();
                assert_eq!(xs.value(i), band_pos as f64,
                    "leaf {} sits at x={} but its heatmap column is at band {band_pos}; columns={columns:?}",
                    COLS[j], xs.value(i));
            }
        }
        assert_eq!(leaves, COLS.len());
    }

    /// The column permutation moves whole columns: every (row, column) cell
    /// keeps its original value, and the label column keeps its slot.
    #[test]
    fn column_reorder_preserves_cell_values() {
        let input = wide_batch();
        let outs = clustermap_chain();
        let long = &outs[FINAL_OUTPUT_KEY];
        let rows = long.column_by_name("row").unwrap().as_any().downcast_ref::<StringArray>().unwrap();
        let cols = long.column_by_name("column").unwrap().as_any().downcast_ref::<StringArray>().unwrap();
        let vals = long.column_by_name("value").unwrap().as_any().downcast_ref::<Float64Array>().unwrap();
        let in_rows = input.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(long.num_rows(), 16);
        for i in 0..long.num_rows() {
            let r = (0..4).find(|&r| in_rows.value(r) == rows.value(i)).unwrap();
            let src = input.column_by_name(cols.value(i)).unwrap()
                .as_any().downcast_ref::<Float64Array>().unwrap();
            assert_eq!(vals.value(i), src.value(r));
        }
    }

    fn column_spec(from: Option<&str>) -> ReorderSpec {
        ReorderSpec {
            by: "original_idx".into(), drop_index: false, from: from.map(Into::into),
            axis: LinkageAxis::Columns, name: None,
        }
    }

    fn order_outputs(order: Vec<i64>) -> HashMap<String, RecordBatch> {
        let schema = Arc::new(Schema::new(vec![Field::new("original_idx", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(order))]).unwrap();
        HashMap::from([("o".to_string(), batch)])
    }

    #[test]
    fn column_reorder_permutes_numeric_slots_and_keeps_label_in_place() {
        let outs = order_outputs(vec![3, 0, 2, 1]);
        let out = apply_with_outputs(&column_spec(Some("o")), &wide_batch(), Some(&outs)).unwrap();
        let names: Vec<String> = out.schema().fields().iter().map(|f| f.name().clone()).collect();
        assert_eq!(names, ["row", "D", "A", "C", "B"]);
    }

    #[test]
    fn column_reorder_refuses_malformed_orders() {
        pyo3::Python::initialize();
        let batch = wide_batch();
        let cases: [(Option<&str>, Vec<i64>, &str); 4] = [
            (None, vec![0, 1, 2, 3], "requires from="),
            (Some("o"), vec![0, 1, 2], "4 numeric columns"),
            (Some("o"), vec![0, 1, 2, 4], "out of bounds"),
            (Some("o"), vec![0, 1, 1, 2], "permutation"),
        ];
        for (from, order, needle) in cases {
            let outs = order_outputs(order);
            let err = apply_with_outputs(&column_spec(from), &batch, Some(&outs)).unwrap_err().to_string();
            assert!(err.contains(needle), "expected {needle:?} in {err}");
        }
    }

    #[test]
    fn column_reorder_refuses_a_decimal_column_in_its_own_voice() {
        pyo3::Python::initialize();
        use arrow::array::Decimal128Array;
        let wide = wide_batch();
        let mut fields: Vec<Field> = wide.schema().fields().iter().map(|f| f.as_ref().clone()).collect();
        let mut cols = wide.columns().to_vec();
        fields.push(Field::new("E", DataType::Decimal128(5, 2), false));
        cols.push(Arc::new(Decimal128Array::from(vec![1, 2, 3, 4]).with_precision_and_scale(5, 2).unwrap()));
        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).unwrap();
        let outs = order_outputs(vec![3, 0, 2, 1]);
        let err = apply_with_outputs(&column_spec(Some("o")), &batch, Some(&outs)).unwrap_err().to_string();
        assert_eq!(
            err,
            "ValueError: stat_reorder: column 'E' is a decimal column (Decimal128(5, 2)); a numeric column \
             must be an integer or float column, so cast it to float first"
        );
    }

    #[test]
    fn axis_serializes_only_when_columns() {
        let rows = TransformSpec::Reorder(ReorderSpec {
            by: "i".into(), drop_index: true, from: None, axis: LinkageAxis::Rows, name: None,
        });
        assert!(!serde_json::to_string(&rows).unwrap().contains("axis"));
        let cols = TransformSpec::Reorder(column_spec(Some("o")));
        let json = serde_json::to_string(&cols).unwrap();
        assert!(json.contains(r#""axis":"columns""#), "{json}");
        assert_eq!(serde_json::from_str::<TransformSpec>(&json).unwrap(), cols);
    }

    // ---------- an order shorter than the batch (a linkage that skipped rows) ----------

    fn rows_spec() -> ReorderSpec {
        ReorderSpec { by: "original_idx".into(), drop_index: true, from: Some("o".into()), axis: LinkageAxis::Rows, name: None }
    }

    fn values(batch: &RecordBatch) -> Vec<f64> {
        let v = batch.column_by_name("v").unwrap().as_any().downcast_ref::<Float64Array>().unwrap();
        (0..v.len()).map(|i| v.value(i)).collect()
    }

    fn five_rows() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0, 50.0]))]).unwrap()
    }

    #[test]
    fn an_order_shorter_than_the_batch_that_no_labelled_linkage_wrote_is_refused() {
        pyo3::Python::initialize();
        for order in [vec![3, 0, 4], vec![]] {
            let n = order.len();
            let err = apply_with_outputs(&rows_spec(), &five_rows(), Some(&order_outputs(order))).unwrap_err().to_string();
            assert!(err.contains(&format!("from='o' produced {n} indices but batch has 5 rows")), "{err}");
        }
    }

    #[test]
    fn a_full_length_order_still_permutes_every_row() {
        let out = apply_with_outputs(&rows_spec(), &five_rows(), Some(&order_outputs(vec![4, 3, 2, 1, 0]))).unwrap();
        assert_eq!(values(&out), vec![50.0, 40.0, 30.0, 20.0, 10.0]);
    }

    #[test]
    fn an_order_longer_than_the_batch_or_out_of_range_is_still_refused() {
        pyo3::Python::initialize();
        let longer = apply_with_outputs(&rows_spec(), &five_rows(), Some(&order_outputs(vec![0, 1, 2, 3, 4, 0]))).unwrap_err().to_string();
        assert!(longer.contains("produced 6 indices but batch has 5 rows"), "{longer}");
        let out_of_range = apply_with_outputs(&rows_spec(), &five_rows(), Some(&order_outputs(vec![0, 1, 2, 3, 9]))).unwrap_err().to_string();
        assert!(out_of_range.contains("out of bounds"), "{out_of_range}");
    }

    // ---------- orders written by a Linkage with labels ----------

    /// `wide_batch` with its row column replaced by `label` (null where `None`).
    fn labelled_wide_batch(label: ArrayRef) -> RecordBatch {
        let wide = wide_batch();
        let mut fields: Vec<Field> = wide.schema().fields().iter().map(|f| f.as_ref().clone()).collect();
        fields[0] = Field::new("row", label.data_type().clone(), true);
        let mut cols = wide.columns().to_vec();
        cols[0] = label;
        RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).unwrap()
    }

    fn null_labelled_wide_batch() -> RecordBatch {
        labelled_wide_batch(Arc::new(StringArray::from(vec![Some("r0"), None, Some("r2"), Some("r3")])))
    }

    fn labelled_linkage(axis: LinkageAxis, name: &str) -> TransformSpec {
        let TransformSpec::Linkage(spec) = linkage(axis, name) else { unreachable!() };
        TransformSpec::Linkage(LinkageSpec { labels: Some("row".into()), ..spec })
    }

    fn resizing_unpivot() -> TransformSpec {
        TransformSpec::Unpivot(UnpivotSpec {
            id_vars: vec!["row".into()], value_vars: None, var_name: "column".into(), value_name: "value".into(), name: None,
        })
    }

    #[test]
    fn a_linkage_order_over_a_batch_another_step_resized_is_refused() {
        pyo3::Python::initialize();
        let batch = null_labelled_wide_batch();
        let labelled = [labelled_linkage(LinkageAxis::Rows, "o"), resizing_unpivot(), order_reorder(LinkageAxis::Rows, "o_order")];
        let err = apply_transforms_named(&labelled, &batch, &TransformContext::default()).unwrap_err().to_string();
        assert!(
            err.contains("from='o_order' orders the 4 rows its linkage read but batch has 16 rows; place the Reorder before any step that adds or removes rows"),
            "{err}"
        );
        let unlabelled = [linkage(LinkageAxis::Rows, "o"), resizing_unpivot(), order_reorder(LinkageAxis::Rows, "o_order")];
        let err = apply_transforms_named(&unlabelled, &wide_batch(), &TransformContext::default()).unwrap_err().to_string();
        assert!(err.contains("from='o_order' produced 4 indices but batch has 16 rows"), "{err}");
    }

    #[test]
    fn a_labelled_linkage_order_keeps_exactly_the_rows_it_names() {
        let specs = [labelled_linkage(LinkageAxis::Rows, "o"), order_reorder(LinkageAxis::Rows, "o_order")];
        let (outs, _) = apply_transforms_named(&specs, &null_labelled_wide_batch(), &TransformContext::default()).unwrap();
        let order = outs["o_order"].column_by_name("original_idx").unwrap().as_any().downcast_ref::<Int64Array>().unwrap().values().to_vec();
        assert_eq!(order.len(), 3);
        assert!(!order.contains(&1), "the null-labelled row is not in the order");
        let out = &outs[FINAL_OUTPUT_KEY];
        let rows = out.column_by_name("row").unwrap().as_any().downcast_ref::<StringArray>().unwrap();
        let expected = ["r0", "", "r2", "r3"];
        let got: Vec<&str> = (0..rows.len()).map(|i| rows.value(i)).collect();
        assert_eq!(got, order.iter().map(|&i| expected[i as usize]).collect::<Vec<_>>(), "the kept rows, in the order's order");
    }

    #[test]
    fn a_column_order_from_a_labelled_linkage_leaves_a_numeric_label_column_in_place() {
        pyo3::Python::initialize();
        let chain = [labelled_linkage(LinkageAxis::Columns, "cl"), order_reorder(LinkageAxis::Columns, "cl_order")];
        let run = |label: ArrayRef| -> RecordBatch {
            let (outs, _) = apply_transforms_named(&chain, &labelled_wide_batch(label), &TransformContext::default()).unwrap();
            outs[FINAL_OUTPUT_KEY].clone()
        };
        let numeric = run(Arc::new(Int64Array::from(vec![10, 11, 12, 13])));
        let text = run(Arc::new(StringArray::from(vec!["r0", "r1", "r2", "r3"])));
        let names = |b: &RecordBatch| -> Vec<String> { b.schema().fields().iter().map(|f| f.name().clone()).collect() };
        assert_eq!(names(&numeric), names(&text), "the feature columns come out in the same order");
        assert_ne!(names(&numeric)[1..], COLS, "the linkage really reorders the features");
        let id = numeric.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(id.values().to_vec(), vec![10, 11, 12, 13], "the label column keeps its place and values");
    }

    #[test]
    fn an_empty_column_order_keeps_the_columns_only_when_a_labelled_linkage_wrote_it() {
        pyo3::Python::initialize();
        let every_null = labelled_wide_batch(Arc::new(StringArray::from(vec![None::<&str>; 4])));
        let chain = [labelled_linkage(LinkageAxis::Columns, "cl"), order_reorder(LinkageAxis::Columns, "cl_order")];
        let (outs, _) = apply_transforms_named(&chain, &every_null, &TransformContext::default()).unwrap();
        assert_eq!(outs[FINAL_OUTPUT_KEY], every_null, "the stamped empty order leaves every column in place");
        let err = apply_with_outputs(&column_spec(Some("o")), &wide_batch(), Some(&order_outputs(vec![]))).unwrap_err().to_string();
        assert!(err.contains("from='o' produced 0 indices but batch has 4 numeric columns"), "{err}");
    }
}
