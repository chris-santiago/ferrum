//! Data transform: DataWindow — window functions (ranking, lag/lead, rolling).
//!
//! Supports: row_number, rank, dense_rank, lag, lead, first_value, last_value,
//! and rolling aggregates (sum, mean, count, min, max) with frame bounds.
//!
//! ## Column ingestion (batch-D M1)
//!
//! Every column a spec references — the `sort` columns and each op's `field` —
//! is coerced exactly once per [`apply`] into a [`WindowColumn`]: integer /
//! unsigned / float dtypes cast to `Float64` via
//! [`numeric_util::coerce_to_float64`], `Timestamp` casts to its epoch value
//! (ferrum's Python boundary pre-casts `Date32`/`Date64` to `Timestamp(ms)`, so
//! this is how a date-sorted window arrives), and `Utf8`/`LargeUtf8` stay
//! string-compared. Every other dtype raises a message naming the transform,
//! the column and the dtype. Nothing reaches the window body as a silent
//! `Ordering::Equal` (an unsorted "sort") or `NaN` (an all-null aggregate).
//!
//! ## Peers (batch-D M4)
//!
//! Rows that compare equal on *all* `sort` columns are peers. Peer runs are
//! computed once per (op, partition) — each op that consults peers (`rank`,
//! `dense_rank`, or a rolling aggregate) recomputes its own [`PeerGroups`] for
//! each of its partitions, not once for the whole spec — and drive both the
//! tie-aware `rank`/`dense_rank` ranks and — unless `ignore_peers` is set — the
//! bounds of a rolling frame, so ties have one definition in this module.
//! Where there are no `sort` columns there is no peer relation to speak of, so
//! every row is its own peer run and ranks and frames behave exactly as they
//! did before.

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::{PyErr, PyResult};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;

use crate::transform::group_key::{groupby_key_at, is_groupby_supported_dtype, KeyValue};
use crate::transform::numeric_util::{coerce_to_float64, is_numeric_dtype};

/// The single source of truth for `DataWindow`'s free-string `op`
/// vocabulary. Read by [`compute_window_op`]'s unknown-op error message and
/// by `registry::entry_for_kind("data_window")`'s `methods` list — R1 design
/// finding 1, mirroring the `smooth::METHODS` precedent. Order matches
/// `compute_window_op`'s match arms: the four positional ops, the two
/// offset ops, the two edge ops, then the five rolling aggregates.
pub(crate) const OPS: &[&str] = &[
    "row_number",
    "rank",
    "dense_rank",
    "lag",
    "lead",
    "first_value",
    "last_value",
    "sum",
    "mean",
    "count",
    "min",
    "max",
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct WindowOp {
    /// One of [`OPS`]; `compute_window_op` refuses anything else.
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
#[serde(deny_unknown_fields)]
pub(crate) struct DataWindowSpec {
    pub ops: Vec<WindowOp>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub sort: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub groupby: Option<Vec<String>>,
    /// Window frame: (preceding, following). None means unbounded.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub frame: Option<(Option<i64>, Option<i64>)>,
    /// Whether rolling frames ignore peers — rows tied on every `sort` column.
    ///
    /// When `false` (the default), rolling frames are anchored on a peer run's
    /// bounds and then expanded to whole peer groups at both ends, so tied rows
    /// share one frame and therefore one aggregate value. When `true`, frames are
    /// purely row-offset based, making them independent of `sort` ties. Omitted
    /// from the wire when `false`, so specs written before this field round-trip
    /// byte-identically.
    #[serde(skip_serializing_if = "is_false", default)]
    pub ignore_peers: bool,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

/// `skip_serializing_if` predicate for [`DataWindowSpec::ignore_peers`]: serde
/// hands the field over by reference, which `std::ops::Not::not` cannot take.
fn is_false(v: &bool) -> bool {
    !*v
}

pub(crate) fn apply(spec: &DataWindowSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let schema = batch.schema();
    let n_rows = batch.num_rows();

    if spec.ops.is_empty() {
        return Err(PyValueError::new_err("data_window: ops must be non-empty"));
    }

    // Coerce every referenced column once, up front (M1): after this, no
    // comparison or aggregate in this module can meet an unhandled dtype.
    let columns = WindowColumns::resolve(batch, &schema, spec)?;

    // Build sort order indices.
    let sorted_indices = build_sort_order(n_rows, &columns);

    // Build group partitions.
    let partitions = build_partitions(batch, &schema, &spec.groupby, &sorted_indices)?;

    // For each op, compute the window function.
    let mut new_cols: Vec<ArrayRef> = Vec::with_capacity(spec.ops.len());
    let mut new_fields: Vec<Field> = Vec::with_capacity(spec.ops.len());

    for op in &spec.ops {
        let col_values = compute_window_op(op, &columns, &partitions, spec, n_rows)?;
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

/// The dtypes a window column may carry, named in every refusal message.
const SUPPORTED_WINDOW_DTYPES: &str = "Utf8/LargeUtf8 (compared as text), \
     Timestamp (compared as epoch), Int8/Int16/Int32/Int64, \
     UInt8/UInt16/UInt32/UInt64, Float16/Float32/Float64";

/// How a spec referenced a column. Affects only the "not found" message, which
/// the two roles have always worded differently.
#[derive(Clone, Copy)]
enum ColumnRole {
    Sort,
    Field,
}

impl ColumnRole {
    fn not_found(self, name: &str) -> PyErr {
        match self {
            ColumnRole::Sort => {
                PyValueError::new_err(format!("data_window: sort column '{name}' not found"))
            }
            ColumnRole::Field => {
                PyValueError::new_err(format!("data_window: column '{name}' not found"))
            }
        }
    }
}

/// A window column's values in the one representation this module compares and
/// aggregates on.
enum WindowValues {
    Numeric(Float64Array),
    Text(StringArray),
}

/// One `sort` or `field` column, coerced once per [`apply`].
struct WindowColumn {
    /// Column name, kept for error messages.
    name: String,
    /// The column's original Arrow dtype, kept for error messages.
    dtype: DataType,
    values: WindowValues,
}

impl WindowColumn {
    fn coerce(col: &ArrayRef, name: &str) -> PyResult<Self> {
        let dtype = col.data_type().clone();
        let values = match &dtype {
            DataType::Utf8 => WindowValues::Text(downcast_string(col, name)?),
            DataType::LargeUtf8 => {
                WindowValues::Text(downcast_string(&cast_column(col, &DataType::Utf8, name)?, name)?)
            }
            // Temporal columns arrive as `Timestamp` (the Python boundary
            // pre-casts `Date32`/`Date64`), and order by their epoch value.
            DataType::Timestamp(_, _) => WindowValues::Numeric(coerce_to_float64(
                &cast_column(col, &DataType::Int64, name)?,
                "data_window",
                name,
            )?),
            dt if is_numeric_dtype(dt) => {
                WindowValues::Numeric(coerce_to_float64(col, "data_window", name)?)
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "data_window: column '{name}' has unsupported dtype {other:?}; \
                     supported: {SUPPORTED_WINDOW_DTYPES}"
                )))
            }
        };
        Ok(Self {
            name: name.to_string(),
            dtype,
            values,
        })
    }

    /// Order two rows of this column.
    ///
    /// Nulls (and NaN) are equal to each other and sort last, which makes this
    /// a total order — required both for `sort_by` and for peer runs, whose
    /// definition would otherwise be intransitive ("null equals 5 and equals
    /// 10, but 5 differs from 10").
    fn compare(&self, a: usize, b: usize) -> Ordering {
        match &self.values {
            WindowValues::Numeric(arr) => {
                let (va, vb) = (f64_at(arr, a), f64_at(arr, b));
                match (va.is_nan(), vb.is_nan()) {
                    (true, true) => Ordering::Equal,
                    (true, false) => Ordering::Greater,
                    (false, true) => Ordering::Less,
                    // Neither side is NaN, so `partial_cmp` is total here.
                    (false, false) => va.partial_cmp(&vb).unwrap_or(Ordering::Equal),
                }
            }
            WindowValues::Text(arr) => match (arr.is_null(a), arr.is_null(b)) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                (false, false) => arr.value(a).cmp(arr.value(b)),
            },
        }
    }

    /// The values behind a column an op reads numerically (lag/lead, the frame
    /// boundary values, the rolling aggregates), or a typed refusal: a text
    /// column has no such value and must not quietly aggregate to NaN.
    fn numeric(&self, op: &str) -> PyResult<&Float64Array> {
        match &self.values {
            WindowValues::Numeric(arr) => Ok(arr),
            WindowValues::Text(_) => Err(PyValueError::new_err(format!(
                "data_window: {op} requires a numeric column, but '{}' has dtype {:?}",
                self.name, self.dtype
            ))),
        }
    }

    /// Peer runs of `partition` keyed on this column alone (`rank`/`dense_rank`
    /// with an explicit `field`).
    fn peer_runs(&self, partition: &[usize]) -> PeerGroups {
        PeerGroups::runs(partition, |a, b| self.compare(a, b) == Ordering::Equal)
    }
}

fn cast_column(col: &ArrayRef, to: &DataType, name: &str) -> PyResult<ArrayRef> {
    arrow::compute::cast(col, to).map_err(|e| {
        PyValueError::new_err(format!(
            "data_window: could not cast column '{name}' to {to:?}: {e}"
        ))
    })
}

fn downcast_string(col: &ArrayRef, name: &str) -> PyResult<StringArray> {
    col.as_any()
        .downcast_ref::<StringArray>()
        .cloned()
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "data_window: expected a Utf8 column for '{name}', got {:?}",
                col.data_type()
            ))
        })
}

/// A row's value as `f64`, reading null as NaN — the sentinel every aggregate
/// and comparison in this module already filters.
fn f64_at(arr: &Float64Array, row: usize) -> f64 {
    if arr.is_null(row) {
        f64::NAN
    } else {
        arr.value(row)
    }
}

/// Every column a [`DataWindowSpec`] references, coerced once per [`apply`].
struct WindowColumns {
    columns: Vec<WindowColumn>,
    by_name: HashMap<String, usize>,
    /// Positions in `columns` of the `sort` columns, in sort order.
    sort: Vec<usize>,
}

impl WindowColumns {
    fn resolve(
        batch: &RecordBatch,
        schema: &arrow::datatypes::SchemaRef,
        spec: &DataWindowSpec,
    ) -> PyResult<Self> {
        let mut this = Self {
            columns: Vec::new(),
            by_name: HashMap::new(),
            sort: Vec::new(),
        };
        for name in spec.sort.iter().flatten() {
            let at = this.intern(batch, schema, name, ColumnRole::Sort)?;
            this.sort.push(at);
        }
        for op in &spec.ops {
            if let Some(field) = op_field(op) {
                this.intern(batch, schema, field, ColumnRole::Field)?;
            }
        }
        Ok(this)
    }

    fn intern(
        &mut self,
        batch: &RecordBatch,
        schema: &arrow::datatypes::SchemaRef,
        name: &str,
        role: ColumnRole,
    ) -> PyResult<usize> {
        if let Some(&at) = self.by_name.get(name) {
            return Ok(at);
        }
        let idx = schema.index_of(name).map_err(|_| role.not_found(name))?;
        self.columns.push(WindowColumn::coerce(batch.column(idx), name)?);
        let at = self.columns.len() - 1;
        self.by_name.insert(name.to_string(), at);
        Ok(at)
    }

    fn get(&self, name: &str) -> PyResult<&WindowColumn> {
        self.by_name
            .get(name)
            .and_then(|&at| self.columns.get(at))
            .ok_or_else(|| ColumnRole::Field.not_found(name))
    }

    /// Order two rows by every `sort` column in turn.
    fn compare_sorted(&self, a: usize, b: usize) -> Ordering {
        for &at in &self.sort {
            let cmp = self.columns[at].compare(a, b);
            if cmp != Ordering::Equal {
                return cmp;
            }
        }
        Ordering::Equal
    }

    /// Peer runs of `partition` under the `sort` columns.
    ///
    /// With no sort columns there is no key to be tied on, so every row is its
    /// own peer — which is what keeps sort-less ranks equal to `row_number` and
    /// sort-less rolling frames row-based, exactly as before this module knew
    /// about peers.
    fn sorted_peer_runs(&self, partition: &[usize]) -> PeerGroups {
        if self.sort.is_empty() {
            return PeerGroups::singletons(partition.len());
        }
        PeerGroups::runs(partition, |a, b| self.compare_sorted(a, b) == Ordering::Equal)
    }
}

/// The `field` an op keys on, treating an empty string as absent (the shape
/// `rank`/`dense_rank` have always read as "no field").
fn op_field(op: &WindowOp) -> Option<&str> {
    op.field.as_deref().filter(|f| !f.is_empty())
}

/// Maximal runs of positions in one partition whose rows are tied on a key.
///
/// Shared by tie-aware ranks and peer-aware rolling frames so both see one
/// definition of "tied", and reused as [`PeerGroups::singletons`] wherever the
/// caller wants pure row semantics.
struct PeerGroups {
    /// Half-open `[start, end)` position bounds of each run, in order.
    bounds: Vec<(usize, usize)>,
    /// Run ordinal of each position in the partition.
    run_of_pos: Vec<usize>,
}

impl PeerGroups {
    /// Every row its own run — no two rows are peers.
    fn singletons(len: usize) -> Self {
        Self {
            bounds: (0..len).map(|pos| (pos, pos + 1)).collect(),
            run_of_pos: (0..len).collect(),
        }
    }

    /// Runs of `partition` under `same`, which is called on row indices and
    /// compares each row against the *first* row of the run it might join.
    fn runs(partition: &[usize], same: impl Fn(usize, usize) -> bool) -> Self {
        let mut bounds: Vec<(usize, usize)> = Vec::new();
        let mut run_of_pos = vec![0usize; partition.len()];
        let mut start = 0;
        while start < partition.len() {
            let mut end = start + 1;
            while end < partition.len() && same(partition[start], partition[end]) {
                end += 1;
            }
            for slot in run_of_pos.iter_mut().take(end).skip(start) {
                *slot = bounds.len();
            }
            bounds.push((start, end));
            start = end;
        }
        Self {
            bounds,
            run_of_pos,
        }
    }

    /// Grow a half-open position range to whole runs, so a frame boundary never
    /// splits a peer group. Empty ranges stay empty.
    ///
    /// Indexing is safe by construction: `run_of_pos` holds valid `bounds`
    /// indices for every position, and `start < end <= partition.len()`.
    fn expand(&self, (start, end): (usize, usize)) -> (usize, usize) {
        if start >= end {
            return (start, start);
        }
        (
            self.bounds[self.run_of_pos[start]].0,
            self.bounds[self.run_of_pos[end - 1]].1,
        )
    }
}

/// Half-open `[start, end)` position range of the frame anchored on positions
/// `anchor.0 ..= anchor.1` of a partition of `len` rows.
///
/// Vega convention: `preceding` is ≤ 0 (rows before the anchor), `following` is
/// ≥ 0 (rows after it), and `None` on either side means unbounded. A frame that
/// falls entirely outside the partition yields an empty range rather than
/// wrapping around to the whole partition.
fn frame_range(
    anchor: (usize, usize),
    (preceding, following): (Option<i64>, Option<i64>),
    len: usize,
) -> (usize, usize) {
    let lo = preceding.map_or(0, |p| anchor.0 as i64 + p);
    let hi = following.map_or(len as i64 - 1, |f| anchor.1 as i64 + f);
    if hi < lo || hi < 0 || lo >= len as i64 {
        return (0, 0);
    }
    (lo.max(0) as usize, (hi as usize + 1).min(len))
}

/// Build sort order. Returns indices in sorted order.
fn build_sort_order(n_rows: usize, columns: &WindowColumns) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n_rows).collect();
    if !columns.sort.is_empty() {
        // Stable, so rows tied on every sort column keep their input order.
        indices.sort_by(|&a, &b| columns.compare_sorted(a, b));
    }
    indices
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

/// The values of the column an op reads numerically, resolved from its
/// mandatory `field`. `requires` names the op in the missing-`field` message
/// ("rolling sum" for the aggregates, the bare op name otherwise).
fn numeric_field<'a>(
    op: &WindowOp,
    columns: &'a WindowColumns,
    requires: &str,
) -> PyResult<&'a Float64Array> {
    let field = op.field.as_deref().ok_or_else(|| {
        PyValueError::new_err(format!("data_window: {requires} requires 'field'"))
    })?;
    columns.get(field)?.numeric(&op.op)
}

fn compute_window_op(
    op: &WindowOp,
    columns: &WindowColumns,
    partitions: &[Vec<usize>],
    spec: &DataWindowSpec,
    n_rows: usize,
) -> PyResult<Vec<f64>> {
    let mut result = vec![f64::NAN; n_rows];

    match op.op.as_str() {
        "row_number" => {
            for partition in partitions {
                for (pos, &idx) in partition.iter().enumerate() {
                    result[idx] = (pos + 1) as f64;
                }
            }
        }
        // Competition ranks (1,1,3,…) and dense ranks (1,1,2,…) over peer runs.
        // With an explicit `field` the runs are keyed on that column; without
        // one they are keyed on the `sort` columns (M4) — which for a sort-less
        // spec degenerates to singleton runs, i.e. `row_number`.
        "rank" | "dense_rank" => {
            let dense = op.op == "dense_rank";
            for partition in partitions {
                let peers = match op_field(op) {
                    Some(field) => columns.get(field)?.peer_runs(partition),
                    None => columns.sorted_peer_runs(partition),
                };
                for (run, &(start, end)) in peers.bounds.iter().enumerate() {
                    let rank = if dense { run + 1 } else { start + 1 };
                    for pos in start..end {
                        result[partition[pos]] = rank as f64;
                    }
                }
            }
        }
        "lag" | "lead" => {
            let values = numeric_field(op, columns, &op.op)?;
            let offset = op.param.unwrap_or(1);

            for partition in partitions {
                for (pos, &idx) in partition.iter().enumerate() {
                    let source_pos = if op.op == "lag" {
                        pos as i64 - offset
                    } else {
                        pos as i64 + offset
                    };
                    if source_pos >= 0 && (source_pos as usize) < partition.len() {
                        result[idx] = f64_at(values, partition[source_pos as usize]);
                    }
                    // else remains NAN
                }
            }
        }
        "first_value" | "last_value" => {
            let values = numeric_field(op, columns, &op.op)?;
            let last = op.op == "last_value";

            for partition in partitions {
                let Some(&edge) = (if last { partition.last() } else { partition.first() }) else {
                    continue;
                };
                let edge_val = f64_at(values, edge);
                for &idx in partition {
                    result[idx] = edge_val;
                }
            }
        }
        // Rolling aggregates: sum, mean, count, min, max
        "sum" | "mean" | "count" | "min" | "max" => {
            let values = numeric_field(op, columns, &format!("rolling {}", op.op))?;
            let frame = spec.frame.unwrap_or((None, None));

            for partition in partitions {
                let len = partition.len();
                // One frame per "unit": a peer run (rows tied on every sort
                // column), or a single row when the caller opted out with
                // `ignore_peers`. By anchoring on the unit's own bounds and then
                // expanding both ends to whole peer groups, tied rows always
                // share one frame and one aggregate value, and a frame boundary
                // never lands inside a peer run. This is a deliberate divergence
                // from Vega's row-anchored frame (see
                // peer_aware_frames_share_one_value_across_tied_rows test).
                let units = if spec.ignore_peers {
                    PeerGroups::singletons(len)
                } else {
                    columns.sorted_peer_runs(partition)
                };
                for &(unit_start, unit_end) in &units.bounds {
                    let (start, end) =
                        units.expand(frame_range((unit_start, unit_end - 1), frame, len));
                    let vals: Vec<f64> = (start..end)
                        .map(|pos| f64_at(values, partition[pos]))
                        .filter(|v| !v.is_nan())
                        .collect();
                    let agg = rolling_agg(&vals, &op.op);
                    for pos in unit_start..unit_end {
                        result[partition[pos]] = agg;
                    }
                }
            }
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "data_window: unknown window op '{other}'; expected {}",
                OPS.join("|"),
            )));
        }
    }

    Ok(result)
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

// No PyO3 wrapper: `DataWindow` is constructed only via the dict-emitting
// `transform_window` Python function and carried through the `transforms_json`
// serde path (SEAM-02). `DataWindowSpec` above is the serde target.

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int64Array};
    use arrow::datatypes::{Field, Schema};
    use std::sync::Arc;

    /// A spec carrying this module's defaults, so a future field on
    /// `DataWindowSpec` doesn't churn every literal below — tests override the
    /// slots they care about through `..window_spec(...)`.
    fn window_spec(ops: Vec<WindowOp>) -> DataWindowSpec {
        DataWindowSpec {
            ops,
            sort: None,
            groupby: None,
            frame: None,
            ignore_peers: false,
            name: None,
        }
    }

    fn op(name: &str, field: Option<&str>, as_: &str) -> WindowOp {
        WindowOp {
            op: name.into(),
            field: field.map(str::to_string),
            as_: as_.into(),
            param: None,
        }
    }

    /// The values of the `at`-th output column (window columns are appended
    /// after the input's own, in op order).
    fn col_values(out: &RecordBatch, at: usize) -> Vec<f64> {
        let arr = out
            .column(at)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("window output columns are Float64");
        (0..arr.len()).map(|i| arr.value(i)).collect()
    }

    // ---- R1-relocated coverage (tests/bug_hunt_release_transforms.rs, 2026-08-27) ----

    /// FA-7 contract: an Int64 groupby column partitions by value through the
    /// real `build_partitions`, exercising the shared `group_key::groupby_key_at`
    /// path this module now delegates to (previously int/uint/bool columns fell
    /// through to an "unsupported dtype" error here). Two logical groups
    /// (1,1,2,2,2) must yield two adjacent-run partitions. The mirror's own
    /// float-dtype sibling test is dropped: it modeled a hand-rolled key
    /// function rather than calling this module's real partitioning, and
    /// Float64 groupby coverage already exists in `group_key.rs`.
    #[test]
    fn int_groupby_column_partitions_by_value() {
        let schema = Arc::new(Schema::new(vec![Field::new("g", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(vec![1i64, 1, 2, 2, 2]))],
        )
        .unwrap();
        let sorted_indices: Vec<usize> = (0..5).collect();
        let partitions =
            build_partitions(&batch, &schema, &Some(vec!["g".into()]), &sorted_indices).unwrap();
        assert_eq!(
            partitions,
            vec![vec![0, 1], vec![2, 3, 4]],
            "int groupby must partition into 2 adjacent-run groups (values 1 and 2)"
        );
    }

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
            // Vega convention: preceding is NEGATIVE (rows before current),
            // following is POSITIVE (rows after current).
            // frame=(-1, 1) = 1 preceding + current + 1 following (symmetric).
            frame: Some((Some(-1), Some(1))),
            ..window_spec(vec![op("mean", Some("x"), "rolling_mean")])
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

        let spec = window_spec(vec![op("row_number", None, "rn")]);
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

        let spec = window_spec(vec![
            WindowOp {
                param: Some(1),
                ..op("lag", Some("x"), "lag_x")
            },
            WindowOp {
                param: Some(1),
                ..op("lead", Some("x"), "lead_x")
            },
        ]);
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
        op("mean", Some("x"), "m")
    }

    fn get_means(out: &RecordBatch) -> Vec<f64> {
        col_values(out, 1)
    }

    // Trailing window: frame=(-2, 0) on [1,2,3,4,5].
    // Expected: [1.0, 1.5, 2.0, 3.0, 4.0]
    #[test]
    fn window_trailing_frame_negative_preceding() {
        let batch = make_val_batch(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let spec = DataWindowSpec {
            frame: Some((Some(-2), Some(0))),
            ..window_spec(vec![mean_op()])
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
            frame: Some((Some(0), Some(2))),
            ..window_spec(vec![mean_op()])
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
            frame: Some((Some(-1), Some(1))),
            ..window_spec(vec![mean_op()])
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

    // ---- M4: tie-aware ranks from the sort key (F-L05-08) ----

    fn score_batch() -> RecordBatch {
        // The finding's fixture: tied scores that must draw tied ranks.
        let schema = Arc::new(Schema::new(vec![Field::new("score", DataType::Float64, false)]));
        RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![5.0, 1.0, 3.0, 1.0, 5.0, 3.0]))],
        )
        .unwrap()
    }

    /// The docs' own field-less `{"op": "rank"}` over a sort key with ties.
    /// Sorted, the scores are 1,1,3,3,5,5 → competition ranks 1,1,3,3,5,5 and
    /// dense ranks 1,1,2,2,3,3, which read back in input order as the values
    /// asserted below.
    #[test]
    fn field_less_ranks_tie_on_the_sort_key() {
        let spec = DataWindowSpec {
            sort: Some(vec!["score".into()]),
            ..window_spec(vec![
                op("rank", None, "r"),
                op("dense_rank", None, "d"),
            ])
        };
        let out = apply(&spec, &score_batch()).unwrap();
        assert_eq!(col_values(&out, 1), vec![5.0, 1.0, 3.0, 1.0, 5.0, 3.0], "competition ranks");
        assert_eq!(col_values(&out, 2), vec![3.0, 1.0, 2.0, 1.0, 3.0, 2.0], "dense ranks");
    }

    /// Without `sort` there is no key to tie on, so both ranks stay exactly
    /// what they were before ties existed: `row_number`.
    #[test]
    fn field_less_ranks_without_sort_are_row_number() {
        let spec = window_spec(vec![op("rank", None, "r"), op("dense_rank", None, "d")]);
        let out = apply(&spec, &score_batch()).unwrap();
        assert_eq!(col_values(&out, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(col_values(&out, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    /// Multi-column peer runs: rows tied on the *first* sort column but
    /// differing on the second are not peers — a peer run requires agreement
    /// on *every* sort column, not just the first one checked.
    #[test]
    fn multi_column_sort_ties_require_agreement_on_every_column() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Float64, false),
            Field::new("b", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 1.0, 1.0, 2.0])),
                Arc::new(StringArray::from(vec!["x", "x", "y", "x"])),
            ],
        )
        .unwrap();
        let spec = DataWindowSpec {
            sort: Some(vec!["a".into(), "b".into()]),
            ..window_spec(vec![op("rank", None, "r")])
        };
        let out = apply(&spec, &batch).unwrap();
        // Output columns are [a, b, r]; peers only where BOTH sort columns
        // agree: rows 0,1 (a=1,b="x") tie at rank 1; row 2 (a=1,b="y") differs
        // on `b` alone, so it is its own run at rank 3; row 3 (a=2) differs on
        // `a`, rank 4.
        assert_eq!(col_values(&out, 2), vec![1.0, 1.0, 3.0, 4.0]);
    }

    /// An explicit `field` keeps its documented field-keyed behavior: ties come
    /// from `value`, never from the (deliberately different) sort key.
    #[test]
    fn explicit_field_ranks_key_on_that_field_not_the_sort() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Float64, false),
            Field::new("value", DataType::Float64, false),
        ]));
        // Sorted by `id`, `value` reads 7, 7, 9, 9 → competition 1,1,3,3.
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
                Arc::new(Float64Array::from(vec![7.0, 7.0, 9.0, 9.0])),
            ],
        )
        .unwrap();
        let spec = DataWindowSpec {
            sort: Some(vec!["id".into()]),
            ..window_spec(vec![
                op("rank", Some("value"), "r"),
                op("dense_rank", Some("value"), "d"),
            ])
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(col_values(&out, 2), vec![1.0, 1.0, 3.0, 3.0]);
        assert_eq!(col_values(&out, 3), vec![1.0, 1.0, 2.0, 2.0]);
    }

    // ---- M4: peer-aware vs row-based frames (F-L05-10) ----

    /// Sort key `t` = [1,1,2,3,3] (two peer runs of two around a singleton),
    /// value `x` = [10,20,30,40,50], rolling `sum` over `frame=(-1, 0)`.
    fn tied_frame_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("t", DataType::Float64, false),
            Field::new("x", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 1.0, 2.0, 3.0, 3.0])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0, 50.0])),
            ],
        )
        .unwrap()
    }

    fn tied_frame_spec(ignore_peers: bool) -> DataWindowSpec {
        DataWindowSpec {
            sort: Some(vec!["t".into()]),
            frame: Some((Some(-1), Some(0))),
            ignore_peers,
            ..window_spec(vec![op("sum", Some("x"), "s")])
        }
    }

    /// Peer-aware (the default): each peer run gets ONE frame, anchored on the
    /// run's own bounds and grown to whole runs, so tied rows share a value.
    ///   run [0,1] (t=1): positions [0-1 .. 1+0] = [0..1] → whole runs [0..1]
    ///                    → 10+20 = 30
    ///   run [2]   (t=2): [2-1 .. 2] = [1..2] → grown to [0..2] → 10+20+30 = 60
    ///   run [3,4] (t=3): [3-1 .. 4] = [2..4] → whole runs [2..4]
    ///                    → 30+40+50 = 120
    #[test]
    fn peer_aware_frames_share_one_value_across_tied_rows() {
        let out = apply(&tied_frame_spec(false), &tied_frame_batch()).unwrap();
        assert_eq!(col_values(&out, 2), vec![30.0, 30.0, 60.0, 120.0, 120.0]);
    }

    /// `ignore_peers=True` is the escape hatch: frames are pure row offsets,
    /// reproducing the pre-M4 output on the very same tied fixture.
    ///   pos0 [0..0]=10, pos1 [0..1]=30, pos2 [1..2]=50,
    ///   pos3 [2..3]=70, pos4 [3..4]=90
    #[test]
    fn ignore_peers_reproduces_row_based_frames() {
        let out = apply(&tied_frame_spec(true), &tied_frame_batch()).unwrap();
        assert_eq!(col_values(&out, 2), vec![10.0, 30.0, 50.0, 70.0, 90.0]);
    }

    /// With no tied sort values the two modes agree — the disclosed default
    /// change reaches only frames whose sort key has ties.
    #[test]
    fn untied_sort_frames_are_identical_under_both_modes() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("t", DataType::Float64, false),
            Field::new("x", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0, 50.0])),
            ],
        )
        .unwrap();
        let peer_aware = apply(&tied_frame_spec(false), &batch).unwrap();
        let row_based = apply(&tied_frame_spec(true), &batch).unwrap();
        assert_eq!(col_values(&peer_aware, 2), vec![10.0, 30.0, 50.0, 70.0, 90.0]);
        assert_eq!(col_values(&peer_aware, 2), col_values(&row_based, 2));
    }

    /// Multi-column peer runs also scope the *frame*, not just ranks: rows
    /// tied on `a` alone but differing on `b` must not be merged into one
    /// frame unit. If peer runs here mistakenly ignored `b`, all three
    /// `a=1.0` rows would form one run and rows 0-1 would read 60 (the sum of
    /// all three), not 30 (the sum of just their own two-row run).
    #[test]
    fn multi_column_sort_ties_scope_the_frame_too() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Float64, false),
            Field::new("b", DataType::Utf8, false),
            Field::new("x", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 1.0, 1.0, 2.0])),
                Arc::new(StringArray::from(vec!["x", "x", "y", "x"])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0])),
            ],
        )
        .unwrap();
        let spec = DataWindowSpec {
            sort: Some(vec!["a".into(), "b".into()]),
            frame: Some((Some(-1), Some(0))),
            ..window_spec(vec![op("sum", Some("x"), "s")])
        };
        let out = apply(&spec, &batch).unwrap();
        // Run [0,2) (a=1,b="x"): frame anchored on [0-1..=1] → [0..2) → 10+20=30.
        // Run [2,3) (a=1,b="y"): frame anchored on [1..=2] → grown to [0..3) → 60.
        // Run [3,4) (a=2,b="x"): frame anchored on [2..=3] → grown to [2..4) → 70.
        assert_eq!(col_values(&out, 3), vec![30.0, 30.0, 60.0, 70.0]);
    }

    /// A peer-aware frame under `groupby=` must not let a peer run cross the
    /// partition edge, even when the tied sort value repeats in the next
    /// group. Every row here ties on `t` globally; if peer runs (or the frame
    /// built from them) ever leaked across the `groupby` boundary, group "a"
    /// would sum group "b"'s values too (100 instead of 30).
    #[test]
    fn peer_aware_frame_does_not_cross_a_groupby_partition_edge() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, false),
            Field::new("t", DataType::Float64, false),
            Field::new("x", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "a", "b", "b"])),
                Arc::new(Float64Array::from(vec![1.0, 1.0, 1.0, 1.0])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0])),
            ],
        )
        .unwrap();
        let spec = DataWindowSpec {
            sort: Some(vec!["t".into()]),
            groupby: Some(vec!["g".into()]),
            frame: Some((Some(-1), Some(0))),
            ..window_spec(vec![op("sum", Some("x"), "s")])
        };
        let out = apply(&spec, &batch).unwrap();
        // Group "a": peer run is the whole group → sum(10,20) = 30 for both.
        // Group "b": peer run is the whole group → sum(30,40) = 70 for both.
        assert_eq!(col_values(&out, 3), vec![30.0, 30.0, 70.0, 70.0]);
    }

    /// A frame that lies entirely before its anchor selects no rows, rather
    /// than the unsigned-cast wraparound the old row-based arithmetic produced
    /// (`(pos + following + 1) as usize` went huge and clamped to the whole
    /// partition).
    #[test]
    fn frame_entirely_before_the_anchor_is_empty() {
        let batch = make_val_batch(vec![1.0, 2.0, 3.0]);
        let spec = DataWindowSpec {
            frame: Some((Some(-3), Some(-2))),
            ..window_spec(vec![mean_op()])
        };
        let out = apply(&spec, &batch).unwrap();
        let means = get_means(&out);
        assert!(means[0].is_nan(), "pos=0 window is empty: {}", means[0]);
        // pos=2: rows [-1 ..= 0] → [0..=0] → mean(1) = 1.0
        assert!((means[2] - 1.0).abs() < 1e-12, "pos=2: {}", means[2]);
    }

    // ---- M1: dtype ingestion (the window half of F-L05-02) ----

    /// The guide's own cumsum-over-`day` example: `day` is an integer column,
    /// which used to compare `Ordering::Equal` and leave the window unsorted.
    #[test]
    fn integer_sort_column_orders_the_window() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("day", DataType::Int64, false),
            Field::new("sales", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![3i64, 1, 2])),
                Arc::new(Float64Array::from(vec![30.0, 10.0, 20.0])),
            ],
        )
        .unwrap();
        let spec = DataWindowSpec {
            sort: Some(vec!["day".into()]),
            frame: Some((None, Some(0))),
            ..window_spec(vec![op("sum", Some("sales"), "cumulative")])
        };
        let out = apply(&spec, &batch).unwrap();
        // Cumulative in day order: day1=10, day2=30, day3=60.
        assert_eq!(col_values(&out, 2), vec![60.0, 10.0, 30.0]);
    }

    /// Integer *value* columns feed lag and the rolling aggregates instead of
    /// silently reading as NaN.
    #[test]
    fn integer_value_column_feeds_lag_and_rolling() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![1i64, 2, 3]))],
        )
        .unwrap();
        let spec = DataWindowSpec {
            frame: Some((Some(-1), Some(0))),
            ..window_spec(vec![op("lag", Some("x"), "lag_x"), op("sum", Some("x"), "s")])
        };
        let out = apply(&spec, &batch).unwrap();
        let lag = col_values(&out, 1);
        assert!(lag[0].is_nan());
        assert_eq!(lag[1..], [1.0, 2.0]);
        assert_eq!(col_values(&out, 2), vec![1.0, 3.0, 5.0]);
    }

    /// Nulls in a sort column order last and are peers of each other, which is
    /// what makes the comparator a total order.
    #[test]
    fn null_sort_values_order_last_and_tie_together() {
        let schema = Arc::new(Schema::new(vec![Field::new("t", DataType::Float64, true)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![None, Some(2.0), None, Some(1.0)]))],
        )
        .unwrap();
        let spec = DataWindowSpec {
            sort: Some(vec!["t".into()]),
            ..window_spec(vec![op("rank", None, "r")])
        };
        let out = apply(&spec, &batch).unwrap();
        // Sorted: 1.0, 2.0, null, null → ranks 1, 2, 3, 3.
        assert_eq!(col_values(&out, 1), vec![3.0, 2.0, 3.0, 1.0]);
    }

    /// Nulls in a *text* sort column also order last. The old `compare_rows`
    /// mapped a null string to `""`, which sorts first and makes null a peer
    /// of a genuine `""` — pinned separately from the numeric case because the
    /// two arms of `WindowColumn::compare` are independent code paths.
    #[test]
    fn null_text_sort_values_order_last() {
        let schema = Arc::new(Schema::new(vec![Field::new("g", DataType::Utf8, true)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec![Some("b"), None, Some("a")]))],
        )
        .unwrap();
        let spec = DataWindowSpec {
            sort: Some(vec!["g".into()]),
            ..window_spec(vec![op("rank", None, "r")])
        };
        let out = apply(&spec, &batch).unwrap();
        // Sorted: "a", "b", null → ranks 1, 2, 3.
        assert_eq!(col_values(&out, 1), vec![2.0, 3.0, 1.0]);
    }

    /// A `field` on an op that never reads one (`row_number`) is still
    /// interned up front and raises by name if it doesn't exist. This is a
    /// deliberate honest-refusal: a `field` on `row_number` was always
    /// meaningless, and the alternative (silently ignoring a bad `field`
    /// value) is exactly the kind of silent fallback M1 eliminates elsewhere
    /// in this module.
    #[test]
    fn row_number_with_field_on_missing_column_raises_by_name() {
        pyo3::Python::initialize();
        let batch = make_val_batch(vec![1.0, 2.0]);
        let spec = window_spec(vec![op("row_number", Some("ghost"), "rn")]);
        let msg = apply(&spec, &batch).unwrap_err().to_string();
        assert!(msg.contains("data_window"), "err: {msg}");
        assert!(msg.contains("'ghost'"), "err must name the field: {msg}");
        assert!(msg.contains("not found"), "err: {msg}");
    }

    #[test]
    fn unsupported_sort_dtype_names_transform_column_and_dtype() {
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![Field::new("flag", DataType::Boolean, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(arrow::array::BooleanArray::from(vec![true, false]))],
        )
        .unwrap();
        let spec = DataWindowSpec {
            sort: Some(vec!["flag".into()]),
            ..window_spec(vec![op("rank", None, "r")])
        };
        let msg = apply(&spec, &batch).unwrap_err().to_string();
        assert!(msg.contains("data_window"), "err: {msg}");
        assert!(msg.contains("'flag'"), "err must name the column: {msg}");
        assert!(msg.contains("Boolean"), "err must name the dtype: {msg}");
    }

    #[test]
    fn text_column_refuses_to_aggregate_instead_of_emitting_nan() {
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![Field::new("label", DataType::Utf8, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec!["a", "b"]))],
        )
        .unwrap();
        let spec = window_spec(vec![op("mean", Some("label"), "m")]);
        let msg = apply(&spec, &batch).unwrap_err().to_string();
        assert!(msg.contains("requires a numeric column"), "err: {msg}");
        assert!(msg.contains("'label'"), "err must name the column: {msg}");
        assert!(msg.contains("Utf8"), "err must name the dtype: {msg}");
    }

    /// Text columns remain sortable (and tie-able) — only numeric *reads*
    /// refuse them.
    #[test]
    fn text_sort_column_orders_and_ties() {
        let schema = Arc::new(Schema::new(vec![Field::new("g", DataType::Utf8, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec!["b", "a", "b"]))],
        )
        .unwrap();
        let spec = DataWindowSpec {
            sort: Some(vec!["g".into()]),
            ..window_spec(vec![op("rank", None, "r")])
        };
        let out = apply(&spec, &batch).unwrap();
        // Sorted: a, b, b → ranks 1, 2, 2.
        assert_eq!(col_values(&out, 1), vec![2.0, 1.0, 2.0]);
    }

    // ---- Wire stability ----

    /// A spec written before `ignore_peers` existed still parses (serde
    /// default), and a default spec still serializes without the key.
    #[test]
    fn ignore_peers_is_absent_from_pre_existing_wires() {
        let json = r#"{"ops":[{"op":"rank","as":"r"}],"sort":["score"]}"#;
        let spec: DataWindowSpec = serde_json::from_str(json).unwrap();
        assert!(!spec.ignore_peers, "absent means the peer-aware default");
        assert_eq!(serde_json::to_string(&spec).unwrap(), json);

        let opted_out = DataWindowSpec {
            ignore_peers: true,
            ..window_spec(vec![op("rank", None, "r")])
        };
        assert!(serde_json::to_string(&opted_out)
            .unwrap()
            .contains("\"ignore_peers\":true"));
    }
}
