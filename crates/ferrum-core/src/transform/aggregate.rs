use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::transform::context::TransformContext;
use crate::transform::group_key::{
    ensure_ungrouped_whole_batch_group, group_partition_multi, groupby_field_nullable,
    materialize_groupby_col, refuse_groupby_output_collision, KeyValue, NullKeyPolicy,
};
use crate::transform::numeric_util::coerce_to_float64;
use crate::transform::vocabulary::{FieldOperand, Vocabulary};

/// The aggregate-function vocabulary shared by `Aggregate`, `JoinAggregate`,
/// `DataAggregate`, `Bin(agg=)` and (as declared subsets) `Hex` and
/// `Smooth(x_estimator=)`.
///
/// Serde reads and writes it through [`Vocabulary::parse_or_refuse`]/
/// [`Vocabulary::as_str`] (`try_from`/`into`), so the wire accepts exactly the
/// spellings every PyO3 door accepts, aliases included, and refuses with the
/// same rendering.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(try_from = "String", into = "&'static str")]
pub(crate) enum AggFn {
    Mean,
    Sum,
    Count,
    Min,
    Max,
    Median,
    Variance,
    Stdev,
    Q1,
    Q3,
    Distinct,
}

impl Vocabulary for AggFn {
    /// Stable Rust has no derive-free way to prove this list names every
    /// variant; `spellings`'s exhaustive match plus
    /// `agg_fn_all_fns_round_trips_through_parse_and_serde` are the guard.
    const ALL: &'static [Self] = &[
        Self::Mean,
        Self::Sum,
        Self::Count,
        Self::Min,
        Self::Max,
        Self::Median,
        Self::Variance,
        Self::Stdev,
        Self::Q1,
        Self::Q3,
        Self::Distinct,
    ];

    /// User ruling Q4: `var` aliases `variance`, `std` aliases `stdev`.
    fn spellings(self) -> &'static [&'static str] {
        match self {
            Self::Mean => &["mean"],
            Self::Sum => &["sum"],
            Self::Count => &["count"],
            Self::Min => &["min"],
            Self::Max => &["max"],
            Self::Median => &["median"],
            Self::Variance => &["variance", "var"],
            Self::Stdev => &["stdev", "std"],
            Self::Q1 => &["q1"],
            Self::Q3 => &["q3"],
            Self::Distinct => &["distinct"],
        }
    }
}

impl AggFn {
    /// Whether this fn can run without a field: only `count`, through
    /// `Aggregate`'s `count()` shorthand (field `""` or `"*"` counts rows).
    /// Every other fn reads a column, so `AggregateOp` refuses it field-less.
    pub(crate) const fn counts_rows_without_field(self) -> bool {
        matches!(self, AggFn::Count)
    }
}

impl TryFrom<String> for AggFn {
    type Error = String;

    fn try_from(s: String) -> Result<Self, Self::Error> {
        Self::parse_or_refuse("fn", &s)
    }
}

impl From<AggFn> for &'static str {
    fn from(f: AggFn) -> Self {
        f.as_str()
    }
}

/// `Hex`'s aggregate vocabulary: a declared subset of [`AggFn`], not a
/// parallel vocabulary. Spellings (aliases included) come from the `AggFn`
/// each variant maps to, and [`Vocabulary::ALL`] follows `AggFn`'s declared
/// order, so `Hex(aggregate="stdev")` and `Bin(agg="std")` read the same
/// table and list it the same way. `HexSpec.aggregate` stores it typed; serde
/// goes through [`Vocabulary::parse_or_refuse`]/[`Vocabulary::as_str`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(try_from = "String", into = "&'static str")]
pub(crate) enum HexAggregate {
    Mean,
    Sum,
    #[default]
    Count,
    Min,
    Max,
    Median,
    Variance,
    Stdev,
}

impl Vocabulary for HexAggregate {
    const ALL: &'static [Self] = &[
        Self::Mean,
        Self::Sum,
        Self::Count,
        Self::Min,
        Self::Max,
        Self::Median,
        Self::Variance,
        Self::Stdev,
    ];

    fn spellings(self) -> &'static [&'static str] {
        self.agg_fn().spellings()
    }
}

impl HexAggregate {
    /// The `AggFn` this variant is. Exhaustive, no wildcard arm: a variant
    /// without a mapping (and so without a spelling) fails to compile.
    const fn agg_fn(self) -> AggFn {
        match self {
            Self::Count => AggFn::Count,
            Self::Mean => AggFn::Mean,
            Self::Sum => AggFn::Sum,
            Self::Min => AggFn::Min,
            Self::Max => AggFn::Max,
            Self::Median => AggFn::Median,
            Self::Stdev => AggFn::Stdev,
            Self::Variance => AggFn::Variance,
        }
    }

}

impl FieldOperand for HexAggregate {
    /// `AggFn`'s own fact, so the row-counting exception is stated once.
    fn needs_field(self) -> bool {
        !self.agg_fn().counts_rows_without_field()
    }
}

impl TryFrom<String> for HexAggregate {
    type Error = String;

    fn try_from(s: String) -> Result<Self, Self::Error> {
        Self::parse_or_refuse("aggregate", &s)
    }
}

impl From<HexAggregate> for &'static str {
    fn from(a: HexAggregate) -> Self {
        a.as_str()
    }
}

/// `Smooth(x_estimator=)`'s vocabulary: the per-bin estimator, a declared
/// subset of [`AggFn`] on the [`HexAggregate`] pattern (spellings from the
/// exhaustive [`XEstimator::agg_fn`] mapping, `AggFn`'s declared order).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum XEstimator {
    Mean,
    Sum,
    Min,
    Max,
    Median,
}

impl Vocabulary for XEstimator {
    const ALL: &'static [Self] = &[Self::Mean, Self::Sum, Self::Min, Self::Max, Self::Median];

    fn spellings(self) -> &'static [&'static str] {
        self.agg_fn().spellings()
    }
}

impl XEstimator {
    /// The `AggFn` this variant is: what `SmoothSpec.x_estimator` stores.
    /// Exhaustive, no wildcard arm.
    pub(crate) const fn agg_fn(self) -> AggFn {
        match self {
            Self::Mean => AggFn::Mean,
            Self::Median => AggFn::Median,
            Self::Sum => AggFn::Sum,
            Self::Min => AggFn::Min,
            Self::Max => AggFn::Max,
        }
    }
}

/// `ferrum._core.x_estimator_accepted_values()`: every `Smooth(x_estimator=)`
/// spelling, in [`Vocabulary::accepted_values`] order, so Python reads the
/// vocabulary the constructor parses instead of mirroring it. Registered in
/// `lib.rs`, typed in `src/ferrum/_core.pyi`; mirrors
/// `interpolate_accepted_values`.
#[pyfunction]
pub(crate) fn x_estimator_accepted_values() -> Vec<String> {
    XEstimator::all().into_iter().map(str::to_string).collect()
}

/// An aggregate op's source column, resolved once per batch.
///
/// Batch-D dtype policy (spec §4.1): a numeric op accepts every integer /
/// unsigned / float dtype by widening to Float64, and refuses anything else by
/// name. `Count` is the documented exception — it counts non-null rows at any
/// dtype, so a non-Float64 column under `Count` is kept in its original form.
///
/// Shared by `stat_aggregate`, `data_aggregate`, and `data_join_aggregate`,
/// which previously carried three drifted copies of this dispatch (one silently
/// returning `NaN`, one erroring without naming the column).
///
/// `fn_` is captured at `resolve` time and carried inside the variant rather
/// than passed again to `aggregate_rows`, so a caller cannot resolve a column
/// for one `AggFn` and then aggregate it as another — that mismatch used to be
/// representable (and silently wrong) even though every call site happened to
/// pass the same `fn_` both times.
pub(crate) enum AggColumn {
    /// Numeric source, widened to Float64.
    Numeric { arr: Float64Array, fn_: AggFn },
    /// `Count` over a non-Float64 column: non-null rows are counted as-is.
    NonNullCount(ArrayRef),
}

impl AggColumn {
    /// Resolve `col` for `fn_`, raising when a numeric op meets a non-numeric
    /// column. `ctx` is the calling transform's error-message prefix.
    pub(crate) fn resolve(col: &ArrayRef, field: &str, fn_: AggFn, ctx: &str) -> PyResult<Self> {
        if fn_ == AggFn::Count && col.data_type() != &DataType::Float64 {
            return Ok(AggColumn::NonNullCount(col.clone()));
        }
        Ok(AggColumn::Numeric { arr: coerce_to_float64(col, ctx, field)?, fn_ })
    }

    /// Aggregate one group's rows, dropping null and NaN entries, using the
    /// `AggFn` captured at `resolve` time.
    pub(crate) fn aggregate_rows(&self, rows: &[usize]) -> f64 {
        match self {
            AggColumn::NonNullCount(col) => {
                rows.iter().filter(|&&r| !col.is_null(r)).count() as f64
            }
            AggColumn::Numeric { arr, fn_ } => {
                let vals: Vec<f64> = rows
                    .iter()
                    .filter_map(|&r| {
                        if arr.is_null(r) {
                            return None;
                        }
                        let v = arr.value(r);
                        if v.is_nan() {
                            return None;
                        }
                        Some(v)
                    })
                    .collect();
                aggregate(&vals, *fn_, rows.len())
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct AggregateOp {
    pub field: String,
    #[serde(rename = "fn")]
    pub fn_: AggFn,
    #[serde(rename = "as")]
    pub as_: String,
}

impl AggregateOp {
    /// The `AggregateOp(...)` repr, spelled with the constructor's keywords
    /// and `fn_`'s canonical spelling (so the `var` alias renders
    /// `variance`). Shared by `AggregateOp.__repr__` and `Aggregate`'s `ops=`.
    fn repr(&self) -> String {
        format!("AggregateOp(field='{}', fn_='{}', as_='{}')",
            self.field, self.fn_.as_str(), self.as_)
    }

    /// The `count():Q` shorthand: a row-counting fn with field `""` or `"*"`,
    /// which needs no column lookup.
    fn is_count_star(&self) -> bool {
        self.fn_.counts_rows_without_field() && (self.field.is_empty() || self.field == "*")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct AggregateSpec {
    pub ops: Vec<AggregateOp>,
    #[serde(default)]
    pub groupby: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

pub(crate) fn apply(spec: &AggregateSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    apply_with_context(spec, batch, &TransformContext::default())
}

/// Context-aware entry point. `ctx` is used only as the GH #162
/// degrade-warning sink, for the one degrade this transform can report: an
/// all-null `groupby` key column (GH #197), raised inside the shared
/// partitioner.
pub(crate) fn apply_with_context(
    spec: &AggregateSpec,
    batch: &RecordBatch,
    ctx: &TransformContext,
) -> PyResult<RecordBatch> {
    if spec.ops.is_empty() {
        return Err(PyValueError::new_err("stat_aggregate: ops must be non-empty"));
    }
    let schema = batch.schema();

    // Validate fields and resolve each op's source column once, per the shared
    // dtype policy (numeric ops widen; non-numeric refuses; Count counts any dtype).
    // Count with an empty field (i.e., `count():Q` shorthand) uses the batch row count
    // directly — no column lookup is needed, so we skip resolution in that case.
    let mut op_columns: Vec<Option<AggColumn>> = Vec::with_capacity(spec.ops.len());
    for op in &spec.ops {
        if op.is_count_star() {
            op_columns.push(None);
            continue;
        }
        let idx = schema.index_of(&op.field).map_err(|_| {
            PyValueError::new_err(format!(
                "stat_aggregate: column '{}' not found", op.field
            ))
        })?;
        op_columns.push(Some(AggColumn::resolve(
            batch.column(idx),
            &op.field,
            op.fn_,
            "stat_aggregate",
        )?));
    }

    // One partitioning idiom for every grouped transform (GH #179): the shared
    // partitioner owns column validation, the repeated-name refusal,
    // first-appearance group order, and the null-key policy.
    let output_fields: Vec<String> = spec.ops.iter().map(|op| op.as_.clone()).collect();
    refuse_groupby_output_collision("stat_aggregate", &spec.groupby, &output_fields)?;
    let mut partition = group_partition_multi(
        batch,
        &spec.groupby,
        "stat_aggregate",
        NullKeyPolicy::Skip,
        ctx,
    )?;
    ensure_ungrouped_whole_batch_group(&mut partition, &spec.groupby);
    let group_dtypes = &partition.dtypes;

    // Materialize op columns: per (group, op) compute aggregate.
    let mut group_keys_out: Vec<Vec<KeyValue>> = Vec::with_capacity(partition.groups.len());
    let mut op_values_out: Vec<Vec<f64>> =
        (0..spec.ops.len()).map(|_| Vec::with_capacity(partition.groups.len())).collect();

    for (key, rows) in &partition.groups {
        group_keys_out.push(key.clone());
        for (op_i, op) in spec.ops.iter().enumerate() {
            // count():Q shorthand — empty field or "*" means count all rows in the group.
            if op.is_count_star() {
                op_values_out[op_i].push(rows.len() as f64);
                continue;
            }

            let resolved = op_columns[op_i]
                .as_ref()
                .expect("invariant: only count-star ops are unresolved, handled above");
            op_values_out[op_i].push(resolved.aggregate_rows(rows));
        }
    }

    // Build output schema and arrays.
    let mut fields: Vec<Field> = Vec::with_capacity(spec.groupby.len() + spec.ops.len());
    for (gi, g) in spec.groupby.iter().enumerate() {
        // FA-9: nullable so a null-keyed group materializes as an Arrow null.
        fields.push(Field::new(g, group_dtypes[gi].clone(), groupby_field_nullable()));
    }
    for op in &spec.ops {
        fields.push(Field::new(&op.as_, DataType::Float64, true));
    }
    let out_schema = Arc::new(Schema::new(fields));

    let mut cols: Vec<ArrayRef> = Vec::with_capacity(spec.groupby.len() + spec.ops.len());
    for (gi, dt) in group_dtypes.iter().enumerate() {
        let col = materialize_groupby_col(&group_keys_out, gi, dt)
            .map_err(PyValueError::new_err)?;
        cols.push(col);
    }
    for op_vec in op_values_out.into_iter() {
        cols.push(Arc::new(Float64Array::from(op_vec)));
    }

    RecordBatch::try_new(out_schema, cols)
        .map_err(|e| PyValueError::new_err(format!("stat_aggregate: {e}")))
}

pub(crate) fn aggregate(vals: &[f64], fn_: AggFn, group_size_including_nulls: usize) -> f64 {
    if vals.is_empty() {
        // Per spec §4.4: all-null group → NaN. Count is the exception.
        return if matches!(fn_, AggFn::Count | AggFn::Distinct) {
            0.0
        } else {
            f64::NAN
        };
    }
    let _ = group_size_including_nulls; // reserved for a future "count_all" variant
    match fn_ {
        AggFn::Mean => vals.iter().sum::<f64>() / vals.len() as f64,
        AggFn::Sum  => vals.iter().sum(),
        AggFn::Count => vals.len() as f64,
        AggFn::Min => vals.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        AggFn::Max => vals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
        AggFn::Median => {
            let mut sorted = vals.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = sorted.len();
            if n % 2 == 1 { sorted[n / 2] }
            else { 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]) }
        }
        AggFn::Variance => {
            if vals.len() < 2 { return f64::NAN; }
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            let sum_sq: f64 = vals.iter().map(|&x| (x - mean).powi(2)).sum();
            // Sample variance: Bessel's correction (n-1 denominator)
            sum_sq / (vals.len() - 1) as f64
        }
        AggFn::Stdev => {
            if vals.len() < 2 { return f64::NAN; }
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            let sum_sq: f64 = vals.iter().map(|&x| (x - mean).powi(2)).sum();
            (sum_sq / (vals.len() - 1) as f64).sqrt()
        }
        AggFn::Q1 => quantile(vals, 0.25),
        AggFn::Q3 => quantile(vals, 0.75),
        AggFn::Distinct => {
            use std::collections::HashSet;
            // Represent each f64 by its bits for hashing; NaN was already filtered.
            let unique: HashSet<u64> = vals.iter().map(|&v| v.to_bits()).collect();
            unique.len() as f64
        }
    }
}

/// Linear-interpolation quantile matching numpy's `np.quantile(method='linear')`.
///
/// Sorts `vals`, computes the virtual index `h = p * (n - 1)`, then linearly
/// interpolates between `sorted[floor(h)]` and `sorted[ceil(h)]`.
pub(crate) fn quantile(vals: &[f64], p: f64) -> f64 {
    debug_assert!(!vals.is_empty(), "caller must guard against empty slice");
    let mut sorted = vals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 1 { return sorted[0]; }
    let h = p * (n - 1) as f64;
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    let frac = h - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::transform::core::TransformSpec;

/// A single aggregation operation descriptor for use with ``Aggregate``.
///
/// Specifies one field→output-name aggregation with a named function.
/// Pass a list of these to ``Aggregate(ops=[...])``.
///
/// Parameters
/// ----------
/// field : str
///     Column to aggregate (must exist in the input batch).
/// fn_ : {"mean", "sum", "count", "min", "max", "median", "variance", "var", "stdev", "std", "q1", "q3", "distinct"}
///     Aggregation function to apply. ``"var"`` and ``"std"`` are declared
///     aliases of ``"variance"`` and ``"stdev"``.
/// as_ : str
///     Name of the output column produced by this operation.
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> op = fm.AggregateOp("price", "mean", "mean_price")
/// >>> agg = fm.Aggregate([op], groupby=["cut"])
#[pyclass(eq, module = "ferrum._core", name = "AggregateOp")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyAggregateOp(pub(crate) AggregateOp);

#[pymethods]
impl PyAggregateOp {
    #[new]
    #[pyo3(signature = (field, fn_, as_))]
    fn new(field: &str, fn_: &str, as_: &str) -> PyResult<Self> {
        if as_.is_empty() {
            return Err(PyValueError::new_err("AggregateOp: as_ must be non-empty"));
        }
        let parsed = AggFn::parse(fn_).ok_or_else(|| PyValueError::new_err(format!(
            "AggregateOp: unknown fn '{fn_}'; expected {}",
            AggFn::accepted_values()
        )))?;
        // count() with empty field is the `count():Q` shorthand — allowed.
        if field.is_empty() && !parsed.counts_rows_without_field() {
            return Err(PyValueError::new_err(
                "AggregateOp: field must be non-empty (use \"*\" for count of all rows)"
            ));
        }
        Ok(PyAggregateOp(AggregateOp {
            field: field.to_string(), fn_: parsed, as_: as_.to_string(),
        }))
    }

    fn __repr__(&self) -> String {
        self.0.repr()
    }
}

/// Group-by aggregation transform.
///
/// Computes one or more scalar summaries (mean, sum, count, min, max, median)
/// per group, reducing the batch to one row per unique group key combination.
/// When ``groupby`` is empty the entire batch is collapsed to a single row.
///
/// Parameters
/// ----------
/// ops : list of AggregateOp
///     Aggregation operations to perform, each specifying ``field``,
///     ``fn_``, and output ``as_``. Must be non-empty.
/// groupby : list of str, optional
///     Column names to group by. Default is ``[]`` (whole-batch aggregate).
/// name : str, optional
///     Named output label used by ``Reorder(from_=...)`` to look up this
///     transform's output. Ignored when no sibling references it.
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> op = fm.AggregateOp("price", "mean", "mean_price")
/// >>> agg = fm.Aggregate([op], groupby=["cut"])
/// >>> fm.Chart(df).mark_bar().encode(x="cut", y="mean_price",
/// ...     transform=agg)
#[pyclass(eq, module = "ferrum._core", name = "Aggregate")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyAggregate(pub(crate) TransformSpec);

#[pymethods]
impl PyAggregate {
    #[new]
    #[pyo3(signature = (ops, *, groupby = None, name = None))]
    fn new(
        ops: &Bound<'_, PyAny>,
        groupby: Option<Vec<String>>,
        name: Option<String>,
    ) -> PyResult<Self> {
        let ops_list: &Bound<'_, PyList> = ops.downcast::<PyList>()
            .map_err(|_| PyValueError::new_err("Aggregate: ops must be a list of AggregateOp"))?;
        if ops_list.is_empty() {
            return Err(PyValueError::new_err("Aggregate: ops must be non-empty"));
        }
        let mut parsed_ops = Vec::with_capacity(ops_list.len());
        for (i, item) in ops_list.iter().enumerate() {
            let op = item.extract::<PyAggregateOp>().map_err(|_| {
                PyValueError::new_err(format!("Aggregate: ops[{i}] must be an AggregateOp"))
            })?;
            parsed_ops.push(op.0);
        }
        let gb = groupby.unwrap_or_default();
        // The duplicate-groupby refusal (spec §6) is the shared rule every
        // grouped transform enforces, in the shared words — still refused here,
        // at the earliest boundary that has the facts, and again inside the
        // partitioner for the wire path that has no constructor (GH #179).
        crate::transform::group_key::refuse_repeated_groupby_names("Aggregate", &gb)?;
        Ok(PyAggregate(TransformSpec::Aggregate(AggregateSpec {
            ops: parsed_ops,
            groupby: gb,
            name,
        })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Aggregate(s) => format!(
                "Aggregate(ops=[{}], groupby={:?}, name={:?})",
                s.ops.iter().map(AggregateOp::repr).collect::<Vec<_>>().join(", "),
                s.groupby, s.name,
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Float64Array, Int64Array, RecordBatch, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn batch_value_group(values: Vec<Option<f64>>, groups: Vec<&str>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("v",     DataType::Float64, true),
            Field::new("group", DataType::Utf8,    true),
        ]));
        let v_arr  = Float64Array::from(values);
        let g_arr  = StringArray::from(groups);
        RecordBatch::try_new(schema, vec![Arc::new(v_arr), Arc::new(g_arr)]).unwrap()
    }

    fn col_f64(b: &RecordBatch, name: &str) -> Vec<f64> {
        let arr = b.column(b.schema().index_of(name).unwrap())
            .as_any().downcast_ref::<Float64Array>().unwrap();
        (0..arr.len()).map(|i| if arr.is_null(i) { f64::NAN } else { arr.value(i) }).collect()
    }

    fn col_str(b: &RecordBatch, name: &str) -> Vec<String> {
        let arr = b.column(b.schema().index_of(name).unwrap())
            .as_any().downcast_ref::<StringArray>().unwrap();
        (0..arr.len()).map(|i| arr.value(i).to_string()).collect()
    }

    #[test]
    fn test_aggregate_mean_sum_count_min_max_per_group() {
        let batch = batch_value_group(
            vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0), Some(6.0)],
            vec!["a", "a", "a", "b", "b", "b"],
        );
        let spec = AggregateSpec {
            ops: vec![
                AggregateOp { field: "v".into(), fn_: AggFn::Mean,  as_: "m".into() },
                AggregateOp { field: "v".into(), fn_: AggFn::Sum,   as_: "s".into() },
                AggregateOp { field: "v".into(), fn_: AggFn::Count, as_: "c".into() },
                AggregateOp { field: "v".into(), fn_: AggFn::Min,   as_: "lo".into() },
                AggregateOp { field: "v".into(), fn_: AggFn::Max,   as_: "hi".into() },
            ],
            groupby: vec!["group".into()],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 2);
        let groups = col_str(&out, "group");
        let m = col_f64(&out, "m");
        let s = col_f64(&out, "s");
        let c = col_f64(&out, "c");
        let lo = col_f64(&out, "lo");
        let hi = col_f64(&out, "hi");

        let a_idx = groups.iter().position(|g| g == "a").unwrap();
        let b_idx = groups.iter().position(|g| g == "b").unwrap();
        assert!((m[a_idx] - 2.0).abs() < 1e-12);
        assert!((m[b_idx] - 5.0).abs() < 1e-12);
        assert!((s[a_idx] - 6.0).abs() < 1e-12);
        assert!((s[b_idx] - 15.0).abs() < 1e-12);
        assert_eq!(c[a_idx] as u64, 3);
        assert_eq!(c[b_idx] as u64, 3);
        assert!((lo[a_idx] - 1.0).abs() < 1e-12);
        assert!((hi[b_idx] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_aggregate_median() {
        let batch = batch_value_group(
            vec![Some(1.0), Some(2.0), Some(100.0), Some(3.0), Some(4.0)],
            vec!["a", "a", "a", "b", "b"],
        );
        let spec = AggregateSpec {
            ops: vec![AggregateOp { field: "v".into(), fn_: AggFn::Median, as_: "med".into() }],
            groupby: vec!["group".into()],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let groups = col_str(&out, "group");
        let med = col_f64(&out, "med");
        let a = groups.iter().position(|g| g == "a").unwrap();
        let b = groups.iter().position(|g| g == "b").unwrap();
        assert!((med[a] - 2.0).abs() < 1e-12, "median(1,2,100) = 2");
        assert!((med[b] - 3.5).abs() < 1e-12, "median(3,4) = 3.5");
    }

    #[test]
    fn test_aggregate_no_groupby_emits_single_global_row() {
        let batch = batch_value_group(
            vec![Some(1.0), Some(2.0), Some(3.0)],
            vec!["a", "b", "c"],
        );
        let spec = AggregateSpec {
            ops: vec![AggregateOp { field: "v".into(), fn_: AggFn::Mean, as_: "m".into() }],
            groupby: vec![],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 1);
        let m = col_f64(&out, "m");
        assert!((m[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_aggregate_all_null_group_field_emits_nan() {
        let batch = batch_value_group(
            vec![None, None, Some(5.0)],
            vec!["a", "a", "b"],
        );
        let spec = AggregateSpec {
            ops: vec![AggregateOp { field: "v".into(), fn_: AggFn::Mean, as_: "m".into() }],
            groupby: vec!["group".into()],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let groups = col_str(&out, "group");
        let m = col_f64(&out, "m");
        let a = groups.iter().position(|g| g == "a").unwrap();
        let b = groups.iter().position(|g| g == "b").unwrap();
        assert!(m[a].is_nan());
        assert!((m[b] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_aggregate_missing_field_errors() {
        pyo3::Python::initialize();
        let batch = batch_value_group(vec![Some(1.0)], vec!["a"]);
        let spec = AggregateSpec {
            ops: vec![AggregateOp { field: "ghost".into(), fn_: AggFn::Mean, as_: "m".into() }],
            groupby: vec!["group".into()],
            name: None,
        };
        let err = apply(&spec, &batch).unwrap_err();
        assert!(err.to_string().contains("ghost"));
    }

    #[test]
    fn test_aggregate_round_trip_json() {
        let original = AggregateSpec {
            ops: vec![
                AggregateOp { field: "x".into(), fn_: AggFn::Sum, as_: "tot".into() },
                AggregateOp { field: "y".into(), fn_: AggFn::Mean, as_: "avg".into() },
            ],
            groupby: vec!["k".into()],
            name: None,
        };
        let json = serde_json::to_string(&original).unwrap();
        let parsed: AggregateSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn test_aggregate_count_on_utf8_column() {
        // Regression: Count on a non-Float64 (Utf8) column should count non-null rows.
        let schema = Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8, true),
            Field::new("group", DataType::Utf8, false),
        ]));
        let names = StringArray::from(vec![
            Some("alice"),
            Some("bob"),
            None,
            Some("carol"),
            Some("dave"),
            None,
        ]);
        let groups = StringArray::from(vec!["a", "a", "a", "b", "b", "b"]);
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(names), Arc::new(groups)]).unwrap();
        let spec = AggregateSpec {
            ops: vec![AggregateOp {
                field: "name".into(),
                fn_: AggFn::Count,
                as_: "n".into(),
            }],
            groupby: vec!["group".into()],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 2);
        let groups_out = col_str(&out, "group");
        let counts = col_f64(&out, "n");
        let a_idx = groups_out.iter().position(|g| g == "a").unwrap();
        let b_idx = groups_out.iter().position(|g| g == "b").unwrap();
        // Group a has 2 non-null names (alice, bob); group b has 2 non-null (carol, dave).
        assert_eq!(counts[a_idx] as u64, 2);
        assert_eq!(counts[b_idx] as u64, 2);
    }

    #[test]
    fn test_aggregate_all_null_float64_mean_returns_nan() {
        // Spec §4.4: all-null column with Mean op → NaN.
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Float64, true),
        ]));
        let arr = Float64Array::from(vec![None, None, None, None, None]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        let spec = AggregateSpec {
            ops: vec![AggregateOp {
                field: "v".into(),
                fn_: AggFn::Mean,
                as_: "m".into(),
            }],
            groupby: vec![],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 1);
        let m = col_f64(&out, "m");
        assert!(m[0].is_nan(), "all-null Mean should be NaN; got {}", m[0]);
    }

    // ── count() with empty field ──────────────────────────────────────────────

    #[test]
    fn test_aggregate_count_empty_field_uses_row_count() {
        pyo3::Python::initialize();
        // When field is "" (or "*") and fn_ is Count, use num_rows of the batch/group.
        let batch = batch_value_group(
            vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0)],
            vec!["a", "a", "b", "b"],
        );
        let spec = AggregateSpec {
            ops: vec![AggregateOp {
                field: "".into(),
                fn_: AggFn::Count,
                as_: "n".into(),
            }],
            groupby: vec!["group".into()],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 2);
        let groups = col_str(&out, "group");
        let counts = col_f64(&out, "n");
        let a = groups.iter().position(|g| g == "a").unwrap();
        let b = groups.iter().position(|g| g == "b").unwrap();
        assert_eq!(counts[a] as u64, 2);
        assert_eq!(counts[b] as u64, 2);
    }

    // ── new aggregate functions ───────────────────────────────────────────────

    #[test]
    fn test_aggregate_variance_known_values() {
        // var([2,4,4,4,5,5,7,9]) = 4.0 (sample variance, Bessel's correction)
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, true)]));
        let arr = Float64Array::from(vec![
            Some(2.0), Some(4.0), Some(4.0), Some(4.0),
            Some(5.0), Some(5.0), Some(7.0), Some(9.0),
        ]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        let spec = AggregateSpec {
            ops: vec![AggregateOp { field: "v".into(), fn_: AggFn::Variance, as_: "var".into() }],
            groupby: vec![],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let var = col_f64(&out, "var");
        assert!((var[0] - 4.571428571428571).abs() < 1e-9, "got {}", var[0]);
    }

    #[test]
    fn test_aggregate_stdev_known_values() {
        // stdev([2,4,4,4,5,5,7,9]) = sqrt(sample_var)
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, true)]));
        let arr = Float64Array::from(vec![
            Some(2.0), Some(4.0), Some(4.0), Some(4.0),
            Some(5.0), Some(5.0), Some(7.0), Some(9.0),
        ]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        let spec = AggregateSpec {
            ops: vec![AggregateOp { field: "v".into(), fn_: AggFn::Stdev, as_: "sd".into() }],
            groupby: vec![],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let sd = col_f64(&out, "sd");
        assert!((sd[0] - 2.138089935325867).abs() < 1e-9, "got {}", sd[0]);
    }

    #[test]
    fn test_aggregate_q1_q3_known_values() {
        // numpy default (linear interpolation): Q1([1,2,3,4]) = 1.75, Q3 = 3.25
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, true)]));
        let arr = Float64Array::from(vec![
            Some(1.0), Some(2.0), Some(3.0), Some(4.0),
        ]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        let spec = AggregateSpec {
            ops: vec![
                AggregateOp { field: "v".into(), fn_: AggFn::Q1, as_: "q1".into() },
                AggregateOp { field: "v".into(), fn_: AggFn::Q3, as_: "q3".into() },
            ],
            groupby: vec![],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let q1 = col_f64(&out, "q1");
        let q3 = col_f64(&out, "q3");
        assert!((q1[0] - 1.75).abs() < 1e-9, "Q1 got {}", q1[0]);
        assert!((q3[0] - 3.25).abs() < 1e-9, "Q3 got {}", q3[0]);
    }

    #[test]
    fn test_aggregate_q1_q3_odd_count() {
        // Q1([1,2,3,4,5]) = 2.0, Q3 = 4.0 (numpy linear)
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, true)]));
        let arr = Float64Array::from(vec![
            Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0),
        ]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        let spec = AggregateSpec {
            ops: vec![
                AggregateOp { field: "v".into(), fn_: AggFn::Q1, as_: "q1".into() },
                AggregateOp { field: "v".into(), fn_: AggFn::Q3, as_: "q3".into() },
            ],
            groupby: vec![],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let q1 = col_f64(&out, "q1");
        let q3 = col_f64(&out, "q3");
        assert!((q1[0] - 2.0).abs() < 1e-9, "Q1 got {}", q1[0]);
        assert!((q3[0] - 4.0).abs() < 1e-9, "Q3 got {}", q3[0]);
    }

    #[test]
    fn test_aggregate_distinct_count() {
        let batch = batch_value_group(
            vec![Some(1.0), Some(1.0), Some(2.0), Some(3.0), Some(3.0), Some(4.0)],
            vec!["a", "a", "a", "b", "b", "b"],
        );
        let spec = AggregateSpec {
            ops: vec![AggregateOp {
                field: "v".into(),
                fn_: AggFn::Distinct,
                as_: "d".into(),
            }],
            groupby: vec!["group".into()],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let groups = col_str(&out, "group");
        let d = col_f64(&out, "d");
        let a = groups.iter().position(|g| g == "a").unwrap();
        let b = groups.iter().position(|g| g == "b").unwrap();
        // group a: {1.0, 2.0} → 2 distinct; group b: {3.0, 4.0} → 2 distinct
        assert_eq!(d[a] as u64, 2);
        assert_eq!(d[b] as u64, 2);
    }

    #[test]
    fn test_aggregate_fn_serde_roundtrip_new_variants() {
        // Verify that new AggFn variants serialize and deserialize correctly via
        // their canonical serde names (lowercase, from rename_all = "lowercase").
        // Python-layer aliases ("var", "stdevp") are tested via PyAggregateOp::new.
        let cases = vec![
            (AggFn::Variance, "variance"),
            (AggFn::Stdev,    "stdev"),
            (AggFn::Q1,       "q1"),
            (AggFn::Q3,       "q3"),
            (AggFn::Distinct, "distinct"),
        ];
        for (fn_, canonical_name) in cases {
            // Serialize the enum variant.
            let json = serde_json::to_string(&fn_).unwrap();
            assert_eq!(json, format!("\"{}\"", canonical_name), "unexpected serde name for {:?}", fn_);

            // Deserialize back from the canonical name.
            let json_op = format!(r#"{{"field":"x","fn":"{}","as":"out"}}"#, canonical_name);
            let parsed: AggregateOp = serde_json::from_str(&json_op).unwrap_or_else(|e| {
                panic!("Failed to deserialize fn '{}': {}", canonical_name, e)
            });
            assert_eq!(parsed.fn_, fn_, "fn '{}' should round-trip to {:?}", canonical_name, fn_);
        }
    }

    // ── AggFn vocabulary exhaustiveness ─────────────────────

    /// Cross-checks `AggFn`'s spellings against `parse` and serde for every
    /// entry in `ALL`: the canonical spelling and every alias parse to
    /// the variant, serde writes the canonical spelling and reads every
    /// spelling back to the same variant. `spellings`'s match is exhaustive
    /// (no wildcard), so a variant without a spelling fails to COMPILE.
    #[test]
    fn agg_fn_all_fns_round_trips_through_parse_and_serde() {
        for &f in AggFn::ALL {
            let canonical = f.as_str();
            assert_eq!(
                serde_json::to_string(&f).unwrap(),
                format!("\"{canonical}\""),
                "{f:?}'s serde spelling drifted from as_str()"
            );
            for &spelling in f.spellings() {
                assert_eq!(AggFn::parse(spelling), Some(f), "'{spelling}' must parse to {f:?}");
                let read: AggFn = serde_json::from_str(&format!("\"{spelling}\"")).unwrap();
                assert_eq!(read, f, "serde must read '{spelling}' as {f:?}");
            }
        }
        let every: usize = AggFn::ALL.iter().map(|f| f.spellings().len()).sum();
        assert_eq!(AggFn::all().len(), every, "AggFn::all() lists every spelling once");
    }

    /// User ruling Q4: `stdev`/`variance` canonical, `std`/`var` declared
    /// aliases. Two-sided: the aliases are accepted AND advertised, and they
    /// normalize to the canonical spelling.
    #[test]
    fn agg_fn_std_and_var_are_declared_aliases_of_the_canonical_spellings() {
        assert_eq!(AggFn::parse("std"), Some(AggFn::Stdev));
        assert_eq!(AggFn::parse("var"), Some(AggFn::Variance));
        assert_eq!(AggFn::Stdev.as_str(), "stdev");
        assert_eq!(AggFn::Variance.as_str(), "variance");
        assert_eq!(
            AggFn::accepted_values(),
            "mean|sum|count|min|max|median|variance|var|stdev|std|q1|q3|distinct"
        );
        assert_eq!(AggFn::parse("geomean"), None);
    }

    /// The wire door refuses an unknown fn with the same `a|b|c` rendering
    /// the PyO3 doors use, not serde's derived "unknown variant" list.
    #[test]
    fn agg_fn_wire_refusal_uses_the_accepted_values_rendering() {
        let err = serde_json::from_str::<AggFn>("\"geomean\"").unwrap_err().to_string();
        assert_eq!(
            err,
            "unknown fn 'geomean'; expected \
             mean|sum|count|min|max|median|variance|var|stdev|std|q1|q3|distinct"
        );
    }

    /// `AggregateOp` built from an alias is the same op as one built from the
    /// canonical spelling, so it aggregates byte-identically.
    #[test]
    fn aggregate_op_alias_is_the_canonical_op() {
        for (alias, canonical) in [("std", "stdev"), ("var", "variance")] {
            let a = PyAggregateOp::new("v", alias, "out").unwrap();
            let c = PyAggregateOp::new("v", canonical, "out").unwrap();
            assert_eq!(a, c, "AggregateOp('{alias}') must equal AggregateOp('{canonical}')");
            assert_eq!(serde_json::to_string(&a.0).unwrap(), serde_json::to_string(&c.0).unwrap());
        }
        let err = PyAggregateOp::new("v", "geomean", "out").unwrap_err();
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            assert_eq!(
                err.value(py).to_string(),
                format!("AggregateOp: unknown fn 'geomean'; expected {}", AggFn::accepted_values())
            );
        });
    }

    /// Two-sided: the published list is exactly `XEstimator`'s spellings in
    /// `accepted_values()` order, and every published value parses back to
    /// the variant that owns it.
    #[test]
    fn x_estimator_accepted_values_pyfunction_equals_the_subset_spellings() {
        let published = x_estimator_accepted_values();
        assert_eq!(published, ["mean", "sum", "min", "max", "median"]);
        assert_eq!(published.join("|"), XEstimator::accepted_values());
        for value in &published {
            let parsed = XEstimator::parse(value).expect("every published spelling must parse");
            assert!(parsed.agg_fn().spellings().contains(&value.as_str()));
        }
        for x in XEstimator::ALL {
            for spelling in x.spellings() {
                assert!(published.iter().any(|p| p == spelling), "'{spelling}' of {x:?} unpublished");
            }
        }
    }

    /// `HexAggregate` is a declared subset of `AggFn`: every hex spelling is
    /// an `AggFn` spelling that parses to the same variant, and the three
    /// order statistics outside the subset are refused.
    #[test]
    fn hex_aggregate_is_a_declared_subset_of_agg_fn() {
        let agg_all = AggFn::all();
        for spelling in HexAggregate::all() {
            assert!(agg_all.contains(&spelling), "hex spelling '{spelling}' is not an AggFn spelling");
            let h = HexAggregate::parse(spelling).unwrap();
            assert_eq!(Some(h.agg_fn()), AggFn::parse(spelling));
        }
        for outside in ["q1", "q3", "distinct", "geomean"] {
            assert_eq!(HexAggregate::parse(outside), None, "'{outside}' must be refused by Hex");
        }
        assert_eq!(HexAggregate::parse("std"), Some(HexAggregate::Stdev));
        assert_eq!(HexAggregate::parse("var"), Some(HexAggregate::Variance));
        assert_eq!(
            HexAggregate::accepted_values(),
            "mean|sum|count|min|max|median|variance|var|stdev|std"
        );
        let needs: Vec<_> =
            HexAggregate::ALL.iter().filter(|h| !h.needs_field()).map(|h| h.spellings()[0]).collect();
        assert_eq!(needs, vec!["count"], "count is hex's only no-field aggregate");
    }

    /// A declared subset lists its spellings in `AggFn`'s declared order, so
    /// its refusal text is its parent's text filtered, never reordered.
    #[test]
    fn declared_subsets_list_in_agg_fn_order() {
        let hex: Vec<AggFn> = HexAggregate::ALL.iter().map(|h| h.agg_fn()).collect();
        assert_eq!(HexAggregate::all(), AggFn::all_where(|f| hex.contains(&f)));
        let xe: Vec<AggFn> = XEstimator::ALL.iter().map(|x| x.agg_fn()).collect();
        assert_eq!(XEstimator::all(), AggFn::all_where(|f| xe.contains(&f)));
    }

    // ── FA-3: integer groupby columns ─────────────────────────────────────────

    fn batch_value_int64_group(values: Vec<Option<f64>>, groups: Vec<i64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("v",   DataType::Float64, true),
            Field::new("grp", DataType::Int64,   false),
        ]));
        let v_arr = Float64Array::from(values);
        let g_arr = Int64Array::from(groups);
        RecordBatch::try_new(schema, vec![Arc::new(v_arr), Arc::new(g_arr)]).unwrap()
    }

    fn col_i64(b: &RecordBatch, name: &str) -> Vec<i64> {
        let arr = b.column(b.schema().index_of(name).unwrap())
            .as_any().downcast_ref::<Int64Array>().unwrap();
        (0..arr.len()).map(|i| arr.value(i)).collect()
    }

    #[test]
    fn test_aggregate_int64_groupby_sum() {
        // FA-3: sum grouped on an Int64 column produces correct per-group sums.
        let batch = batch_value_int64_group(
            vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)],
            vec![10, 10, 20, 20, 30],
        );
        let spec = AggregateSpec {
            ops: vec![AggregateOp { field: "v".into(), fn_: AggFn::Sum, as_: "s".into() }],
            groupby: vec!["grp".into()],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 3);

        // Output groupby column must retain Int64 dtype.
        assert_eq!(out.schema().field(0).data_type(), &DataType::Int64);

        let grps = col_i64(&out, "grp");
        let sums = col_f64(&out, "s");
        let g10 = grps.iter().position(|&g| g == 10).unwrap();
        let g20 = grps.iter().position(|&g| g == 20).unwrap();
        let g30 = grps.iter().position(|&g| g == 30).unwrap();

        assert!((sums[g10] - 3.0).abs() < 1e-12, "group 10 sum: {}", sums[g10]);
        assert!((sums[g20] - 7.0).abs() < 1e-12, "group 20 sum: {}", sums[g20]);
        assert!((sums[g30] - 5.0).abs() < 1e-12, "group 30 sum: {}", sums[g30]);
    }

    #[test]
    fn test_aggregate_int64_groupby_mean() {
        // FA-3: mean grouped on Int64 groupby.
        let batch = batch_value_int64_group(
            vec![Some(1.0), Some(3.0), Some(2.0), Some(4.0)],
            vec![2020, 2020, 2021, 2021],
        );
        let spec = AggregateSpec {
            ops: vec![AggregateOp { field: "v".into(), fn_: AggFn::Mean, as_: "m".into() }],
            groupby: vec!["grp".into()],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 2);

        let grps = col_i64(&out, "grp");
        let means = col_f64(&out, "m");
        let y2020 = grps.iter().position(|&g| g == 2020).unwrap();
        let y2021 = grps.iter().position(|&g| g == 2021).unwrap();
        assert!((means[y2020] - 2.0).abs() < 1e-12, "2020 mean: {}", means[y2020]);
        assert!((means[y2021] - 3.0).abs() < 1e-12, "2021 mean: {}", means[y2021]);
    }

    #[test]
    fn test_aggregate_int64_groupby_count() {
        // FA-3: count grouped on Int64 groupby.
        let batch = batch_value_int64_group(
            vec![Some(0.0), Some(0.0), Some(0.0), Some(0.0), Some(0.0)],
            vec![1, 1, 1, 2, 2],
        );
        let spec = AggregateSpec {
            ops: vec![AggregateOp { field: "v".into(), fn_: AggFn::Count, as_: "c".into() }],
            groupby: vec!["grp".into()],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 2);

        let grps = col_i64(&out, "grp");
        let counts = col_f64(&out, "c");
        let g1 = grps.iter().position(|&g| g == 1).unwrap();
        let g2 = grps.iter().position(|&g| g == 2).unwrap();
        assert_eq!(counts[g1] as u64, 3, "group 1 count: {}", counts[g1]);
        assert_eq!(counts[g2] as u64, 2, "group 2 count: {}", counts[g2]);
    }

    #[test]
    fn test_aggregate_int32_groupby_sum() {
        // FA-3: Int32 groupby also accepted.
        use arrow::array::Int32Array;
        let schema = Arc::new(Schema::new(vec![
            Field::new("v",   DataType::Float64, true),
            Field::new("grp", DataType::Int32,   false),
        ]));
        let v_arr = Float64Array::from(vec![Some(10.0), Some(20.0), Some(30.0)]);
        let g_arr = Int32Array::from(vec![1i32, 1, 2]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(v_arr), Arc::new(g_arr)]).unwrap();

        let spec = AggregateSpec {
            ops: vec![AggregateOp { field: "v".into(), fn_: AggFn::Sum, as_: "s".into() }],
            groupby: vec!["grp".into()],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 2);
        // Output dtype preserved as Int32.
        assert_eq!(out.schema().field(0).data_type(), &DataType::Int32);
    }

    #[test]
    fn test_aggregate_uint32_groupby_sum() {
        // FA-3: UInt32 groupby also accepted.
        use arrow::array::UInt32Array;
        let schema = Arc::new(Schema::new(vec![
            Field::new("v",   DataType::Float64, true),
            Field::new("grp", DataType::UInt32,  false),
        ]));
        let v_arr = Float64Array::from(vec![Some(5.0), Some(15.0), Some(25.0)]);
        let g_arr = UInt32Array::from(vec![100u32, 100, 200]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(v_arr), Arc::new(g_arr)]).unwrap();

        let spec = AggregateSpec {
            ops: vec![AggregateOp { field: "v".into(), fn_: AggFn::Sum, as_: "s".into() }],
            groupby: vec!["grp".into()],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 2);
        assert_eq!(out.schema().field(0).data_type(), &DataType::UInt32);
    }

    // ── FA-3 null-key grouping regression ────────────────────────────────────

    /// Build a batch with a nullable Float64 groupby column.
    fn batch_value_float64_group_nullable(
        values: Vec<Option<f64>>,
        groups: Vec<Option<f64>>,
    ) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("v",   DataType::Float64, true),
            Field::new("grp", DataType::Float64, true),
        ]));
        let v_arr = Float64Array::from(values);
        let g_arr = Float64Array::from(groups);
        RecordBatch::try_new(schema, vec![Arc::new(v_arr), Arc::new(g_arr)]).unwrap()
    }

    /// Build a batch with a nullable Int64 groupby column.
    fn batch_value_int64_group_nullable(
        values: Vec<Option<f64>>,
        groups: Vec<Option<i64>>,
    ) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("v",   DataType::Float64, true),
            Field::new("grp", DataType::Int64,   true),
        ]));
        let v_arr = Float64Array::from(values);
        let g_arr = Int64Array::from(groups);
        RecordBatch::try_new(schema, vec![Arc::new(v_arr), Arc::new(g_arr)]).unwrap()
    }

    // ── GH #197: the all-transforms null-groupby-key policy ─────────────────
    //
    // These three tests were written for the pre-#197 hand-rolled partitioner,
    // where `stat_aggregate` pooled every null-keyed row into one
    // `KeyValue::Null` group and emitted it as an Arrow null in the output
    // column. #197 adjudicated ONE policy for the whole grouped family, and
    // this transform's arm of it is `NullKeyPolicy::Skip`: a row with no key
    // belongs to no group, so it contributes to no output row. What they still
    // pin, unchanged, is FA-3's distinction between a null key and a genuine
    // NaN *value* key — the NaN group survives while the null rows drop, which
    // is only observable if the two are still distinct keys.

    #[test]
    fn aggregate_null_float64_groupby_rows_drop_out_of_every_group() {
        //   row 0: grp=None (null)  → belongs to no group
        //   row 1: grp=None (null)  → belongs to no group
        //   row 2: grp=Some(1.0)    → the one surviving group
        let batch = batch_value_float64_group_nullable(
            vec![Some(10.0), Some(20.0), Some(5.0)],
            vec![None,        None,        Some(1.0)],
        );
        let spec = AggregateSpec {
            ops: vec![AggregateOp { field: "v".into(), fn_: AggFn::Sum, as_: "s".into() }],
            groupby: vec!["grp".into()],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();

        assert_eq!(
            out.num_rows(), 1,
            "null-keyed rows belong to no group, so only the 1.0 group survives"
        );
        let grp_arr = out.column(out.schema().index_of("grp").unwrap())
            .as_any().downcast_ref::<Float64Array>().unwrap();
        assert!(!grp_arr.is_null(0), "no null-keyed group is emitted at all");
        assert!((grp_arr.value(0) - 1.0).abs() < 1e-12);
        let sums = col_f64(&out, "s");
        assert!(
            (sums[0] - 5.0).abs() < 1e-12,
            "the surviving group aggregates ONLY its own row, got {}",
            sums[0]
        );
    }

    #[test]
    fn aggregate_null_int64_groupby_rows_drop_out_of_every_group() {
        let batch = batch_value_int64_group_nullable(
            vec![Some(3.0), Some(7.0), Some(100.0)],
            vec![None,       None,       Some(7)],
        );
        let spec = AggregateSpec {
            ops: vec![AggregateOp { field: "v".into(), fn_: AggFn::Sum, as_: "s".into() }],
            groupby: vec!["grp".into()],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();

        assert_eq!(out.num_rows(), 1, "only the key=7 group survives");
        let grp_arr = out.column(out.schema().index_of("grp").unwrap())
            .as_any().downcast_ref::<Int64Array>().unwrap();
        assert!(!grp_arr.is_null(0), "no null-keyed group is emitted at all");
        assert_eq!(grp_arr.value(0), 7);
        let sums = col_f64(&out, "s");
        assert!((sums[0] - 100.0).abs() < 1e-12, "got {}", sums[0]);
    }

    #[test]
    fn aggregate_genuine_nan_groupby_key_survives_while_null_key_drops() {
        // FA-3 regression, re-aimed at #197's policy: a genuine NaN *value* in a
        // Float64 groupby column is a key like any other and keeps its group;
        // a null is the absence of a key and drops. Pre-FA-3 both encoded to
        // `Float(NAN.to_bits())`, so this test could not tell them apart —
        // under that bug the NaN row would drop with the null one.
        let nan = f64::NAN;
        let batch = batch_value_float64_group_nullable(
            vec![Some(1.0), Some(2.0), Some(3.0)],
            vec![None,       Some(nan), Some(2.0)],
        );
        let spec = AggregateSpec {
            ops: vec![AggregateOp { field: "v".into(), fn_: AggFn::Sum, as_: "s".into() }],
            groupby: vec!["grp".into()],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();

        assert_eq!(
            out.num_rows(), 2,
            "expected the NaN group and the 2.0 group; the null-keyed row drops"
        );
        let grp_arr = out.column(out.schema().index_of("grp").unwrap())
            .as_any().downcast_ref::<Float64Array>().unwrap();
        let sums = col_f64(&out, "s");

        assert!(
            (0..grp_arr.len()).all(|i| !grp_arr.is_null(i)),
            "no null-keyed group is emitted at all"
        );
        let nan_idx = (0..grp_arr.len())
            .find(|&i| grp_arr.value(i).is_nan())
            .expect("the genuine-NaN key keeps its own group");
        let real_idx = (0..grp_arr.len())
            .find(|&i| (grp_arr.value(i) - 2.0).abs() < 1e-12)
            .expect("real-valued group (key=2.0) must be present");
        assert!((sums[nan_idx] - 2.0).abs() < 1e-12, "got {}", sums[nan_idx]);
        assert!((sums[real_idx] - 3.0).abs() < 1e-12, "got {}", sums[real_idx]);
    }

    #[test]
    fn test_aggregate_unsupported_groupby_dtype_errors_clearly() {
        // FA-3 regression: genuinely unsupported dtypes (e.g., List) must raise a
        // clear error, not panic or silently miscount.
        use arrow::array::ListArray;
        use arrow::buffer::OffsetBuffer;
        use arrow::datatypes::Field as ArrowField;
        pyo3::Python::initialize();

        // Construct a minimal ListArray column.
        let values = arrow::array::Int32Array::from(vec![1, 2, 3]);
        let offsets = OffsetBuffer::new(arrow::buffer::ScalarBuffer::from(vec![0i32, 2, 3]));
        let field = ArrowField::new("item", DataType::Int32, false);
        let list_arr = ListArray::new(Arc::new(field), offsets, Arc::new(values), None);

        let schema = Arc::new(Schema::new(vec![
            Field::new("v",   DataType::Float64, true),
            Field::new("grp", DataType::List(Arc::new(ArrowField::new("item", DataType::Int32, false))), false),
        ]));
        let v_arr = Float64Array::from(vec![Some(1.0), Some(2.0)]);
        let batch = RecordBatch::try_new(
            schema, vec![Arc::new(v_arr), Arc::new(list_arr)]
        ).unwrap();

        let spec = AggregateSpec {
            ops: vec![AggregateOp { field: "v".into(), fn_: AggFn::Sum, as_: "s".into() }],
            groupby: vec!["grp".into()],
            name: None,
        };
        let err = apply(&spec, &batch).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("unsupported dtype"), "expected 'unsupported dtype' in: {msg}");
        assert!(msg.contains("grp"), "expected column name 'grp' in error: {msg}");
    }

    // ---- Batch-D dtype policy (spec §4.1) ----

    fn int_value_batch(values: Vec<i64>, groups: Vec<&str>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("v",     DataType::Int64, true),
            Field::new("group", DataType::Utf8,  true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(values)),
                Arc::new(StringArray::from(groups)),
            ],
        )
        .unwrap()
    }

    fn mean_of_v(groupby: Vec<String>) -> AggregateSpec {
        AggregateSpec {
            ops: vec![AggregateOp {
                field: "v".into(),
                fn_: AggFn::Mean,
                as_: "m".into(),
            }],
            groupby,
            name: None,
        }
    }

    #[test]
    fn aggregate_int_column_matches_its_float64_equivalent() {
        let ints = int_value_batch(vec![1, 3, 10, 20], vec!["a", "a", "b", "b"]);
        let floats = batch_value_group(
            vec![Some(1.0), Some(3.0), Some(10.0), Some(20.0)],
            vec!["a", "a", "b", "b"],
        );
        let spec = mean_of_v(vec!["group".into()]);

        let from_ints = apply(&spec, &ints).unwrap();
        let from_floats = apply(&spec, &floats).unwrap();
        assert_eq!(col_f64(&from_ints, "m"), col_f64(&from_floats, "m"));
        assert_eq!(col_str(&from_ints, "group"), col_str(&from_floats, "group"));
    }

    #[test]
    fn aggregate_non_numeric_under_numeric_op_raises_naming_column_and_dtype() {
        pyo3::Python::initialize();
        let batch = batch_value_group(vec![Some(1.0), Some(2.0)], vec!["a", "b"]);
        let spec = AggregateSpec {
            ops: vec![AggregateOp {
                field: "group".into(),
                fn_: AggFn::Mean,
                as_: "m".into(),
            }],
            groupby: vec![],
            name: None,
        };
        let msg = apply(&spec, &batch).unwrap_err().to_string();
        assert!(msg.contains("stat_aggregate"), "err: {msg}");
        assert!(msg.contains("'group'"), "err must name the column: {msg}");
        assert!(msg.contains("Utf8"), "err must name the dtype: {msg}");
    }

    #[test]
    fn aggregate_count_still_counts_non_null_rows_at_any_dtype() {
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec![Some("a"), None, Some("c")]))],
        )
        .unwrap();
        let spec = AggregateSpec {
            ops: vec![AggregateOp {
                field: "s".into(),
                fn_: AggFn::Count,
                as_: "n".into(),
            }],
            groupby: vec![],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(col_f64(&out, "n"), vec![2.0]);
    }

    // ── GH #179/#197: one partitioning idiom ────────────────────────────────

    /// A groupby column named like one of the aggregate outputs refuses rather
    /// than emitting two same-named fields (GH #179 residual 1: this append
    /// site had no collision guard at all).
    #[test]
    fn aggregate_groupby_colliding_with_an_op_output_refuses() {
        pyo3::Python::initialize();
        let batch = batch_value_group(vec![Some(1.0), Some(2.0)], vec!["a", "b"]);
        let spec = AggregateSpec {
            ops: vec![AggregateOp { field: "v".into(), fn_: AggFn::Sum, as_: "group".into() }],
            groupby: vec!["group".into()],
            name: None,
        };
        let err = apply(&spec, &batch)
            .expect_err("an op output named like the groupby column must refuse");
        let msg = err.to_string();
        assert!(msg.contains("collides"), "{msg}");
        assert!(msg.contains("stat_aggregate"), "{msg}");
    }

    /// DISCLOSED BEHAVIOR CHANGE (#179): group rows come out in
    /// first-appearance order. This transform previously emitted BTreeMap
    /// sorted order, so `b` before `a` in the data still rendered `a` first.
    #[test]
    fn aggregate_group_rows_are_in_first_appearance_order() {
        pyo3::Python::initialize();
        let batch = batch_value_group(vec![Some(1.0), Some(2.0)], vec!["b", "a"]);
        let spec = AggregateSpec {
            ops: vec![AggregateOp { field: "v".into(), fn_: AggFn::Sum, as_: "s".into() }],
            groupby: vec!["group".into()],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let g = out.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!((g.value(0), g.value(1)), ("b", "a"), "first-appearance, not sorted");

        // The flip is a PERMUTATION, not a recomputation: each key still
        // carries its own aggregate. A reorder that mismatched keys and values
        // would pass the order assertion above and fail here.
        let s = col_f64(&out, "s");
        assert_eq!(
            (s[0], s[1]),
            (1.0, 2.0),
            "b's sum stays with b and a's with a after the reorder"
        );
    }

    /// A repeated groupby name refuses at APPLY time now, not only at the PyO3
    /// constructor — the wire path (`transforms_json`) reaches this spec too,
    /// and the constructor's own loop is gone.
    #[test]
    fn aggregate_repeated_groupby_name_refuses_from_the_wire_path() {
        pyo3::Python::initialize();
        let batch = batch_value_group(vec![Some(1.0)], vec!["a"]);
        let spec = AggregateSpec {
            ops: vec![AggregateOp { field: "v".into(), fn_: AggFn::Sum, as_: "s".into() }],
            groupby: vec!["group".into(), "group".into()],
            name: None,
        };
        let err = apply(&spec, &batch).expect_err("groupby=['group', 'group'] must refuse");
        assert!(err.to_string().contains("repeated"), "{err}");
    }

    /// GH #197: an all-null groupby column yields no groups and warns once
    /// through the degrade channel, naming the transform and the column.
    #[test]
    fn aggregate_all_null_group_keys_yield_no_rows_and_warn() {
        pyo3::Python::initialize();
        let batch = batch_value_int64_group_nullable(
            vec![Some(1.0), Some(2.0)],
            vec![None, None],
        );
        let spec = AggregateSpec {
            ops: vec![AggregateOp { field: "v".into(), fn_: AggFn::Sum, as_: "s".into() }],
            groupby: vec!["grp".into()],
            name: None,
        };
        let ctx = crate::transform::context::TransformContext::default();
        let out = apply_with_context(&spec, &batch, &ctx).unwrap();
        assert_eq!(out.num_rows(), 0, "no usable key → no groups → no rows");
        let warnings = ctx.warnings.borrow();
        assert_eq!(warnings.len(), 1);
        let text = warnings[0].to_string();
        // "Aggregate", not the wire tag "stat_aggregate" — wave-1 close
        // remediation unified `GroupKeyAllNull`'s `transform` field onto the
        // same Python-facing vocabulary every other `TransformWarning`
        // variant already uses (`display_name_for_label`).
        assert!(text.contains("Aggregate"), "{text}");
        assert!(text.contains("grp"), "{text}");
    }

    /// An ungrouped aggregate over a ZERO-ROW batch still collapses to one
    /// summary row. The shared partitioner discovers groups from rows, so this
    /// boundary is asked for explicitly rather than inherited.
    #[test]
    fn aggregate_ungrouped_zero_row_input_still_yields_one_row() {
        pyo3::Python::initialize();
        let batch = batch_value_group(Vec::new(), Vec::new());
        let spec = AggregateSpec {
            ops: vec![AggregateOp { field: "".into(), fn_: AggFn::Count, as_: "n".into() }],
            groupby: vec![],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 1, "the whole-batch group survives an empty batch");
        assert_eq!(col_f64(&out, "n")[0], 0.0);
    }

    /// `count` over field `"*"` is the row-counting shorthand, like `""`: it
    /// counts every row of the group, nulls included, with no column lookup.
    /// Only `count` gets the shorthand; another fn over `"*"` looks for a
    /// column named `*` and refuses.
    #[test]
    fn aggregate_count_over_star_counts_every_row() {
        pyo3::Python::initialize();
        let batch = batch_value_group(vec![Some(1.0), None, Some(3.0)], vec!["a", "a", "b"]);
        let spec = |field: &str, fn_: AggFn| AggregateSpec {
            ops: vec![AggregateOp { field: field.into(), fn_, as_: "n".into() }],
            groupby: vec!["group".into()],
            name: None,
        };
        let star = apply(&spec("*", AggFn::Count), &batch).unwrap();
        assert_eq!(col_f64(&star, "n"), vec![2.0, 1.0], "group a has two rows, one null");
        let non_null = apply(&spec("v", AggFn::Count), &batch).unwrap();
        assert_eq!(col_f64(&non_null, "n"), vec![1.0, 1.0], "control: a column count skips the null");
        let err = apply(&spec("*", AggFn::Mean), &batch).unwrap_err();
        pyo3::Python::attach(|py| {
            assert_eq!(err.value(py).to_string(), "stat_aggregate: column '*' not found");
        });
    }

}
