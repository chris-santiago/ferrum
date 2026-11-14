//! Data transform: Impute: fill missing (null/NaN) values and complete series
//! over their keys. An infinite cell is the caller's value and is kept; the
//! fill statistic skips it.
//!
//! One computation, [`complete_series`], serves `transform_impute` here and the
//! encoding door (`Y(impute=)`, `X(impute=)`), with Vega-Lite's options:
//!
//! - `method`: `value` (default), `mean`, `median`, `min`, `max`.
//! - `value`: the constant for `method="value"`, default 0, of the column's own
//!   kind (a number, a timestamp at its unit and zone, text, a boolean).
//!   Refused beside a statistic.
//! - `keyvals`: keys every series must hold, added to the keys the batch holds
//!   (a list, or a `{start, stop, step}` sequence).
//! - `frame`: `(preceding, following)` row offsets around each fill, counted in
//!   key order, for the statistic methods. Refused beside `value`.
//!
//! ## Groupby + key/keyvals
//!
//! `impute` is row-preserving: every input row keeps its place relative to
//! every other input row. Rows partition via
//! [`group_key::group_partition_multi`] (empty `groupby` yields exactly one
//! implicit group holding every row, in original order, which is why the
//! ungrouped path is the same per-group computation over the one group that
//! happens to be the whole batch). A row whose groupby key is null belongs to
//! no group but still needs a value, so it falls back to the whole-column
//! statistic. That fallback is GH #197's stated exemption for this transform:
//! the call site passes `NullKeyPolicy::FillFromWholeBatch`, the one variant
//! that keeps and fills every null-keyed row rather than dropping it. The same
//! whole-column statistic is the fallback for a group whose own usable values
//! are none (each is null, NaN or infinite), since `0.0` is a real plottable
//! value and `±inf` drops the group's rows from the render entirely.
//!
//! `key=` completes every series over the key universe (see [`keys`]): each
//! key a series lacks gets one row, carrying the series' own columns, the
//! missing key and the series' fill, with every other column null. A null or
//! `NaN` cell of the value column is filled the same way, and an infinite cell
//! is kept. Each imputed row is inserted at its key's place (see [`insert`]),
//! so a line draws `1, 2, 3` and not `1, 3, 2`, and a caller's row never moves.
//! `key` without `groupby` is Vega's "no groupby means one implicit group"
//! rule. `keyvals` is meaningless without `key`, so passing one without the
//! other is a refusal, not a silent no-op.

mod fill;
mod insert;
mod keys;
mod literal;
#[cfg(test)]
mod complete_tests;

use arrow::array::{Array, ArrayRef, BooleanArray, Float64Array, RecordBatch, Scalar, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::transform::context::{
    describe_group_keys, format_key_value, TransformContext, TransformWarning,
};
use crate::transform::data_window::frame_range;
use crate::transform::group_key::{
    group_partition_multi, is_groupby_supported_dtype, materialize_groupby_col, GroupPartition,
    KeyValue, NullKeyPolicy, GROUPBY_SUPPORTED_DTYPES,
};
use crate::transform::numeric_util::{clean_float64_values, float64_column, USABLE};
use fill::{
    fill_array, resolve_literal, window_statistic, Entry, Method, Statistic, ValueKind,
};
use insert::{key_ordered, missing_keys, Missing};
use keys::KeyUniverse;

/// The door `transform_impute` speaks as in every refusal and warning.
const TRANSFORM_DOOR: &str = "data_impute";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ImputeSpec {
    /// Column to impute.
    pub field: String,
    /// Imputation method: "value", "mean", "median", "min", "max".
    #[serde(default = "default_method")]
    pub method: String,
    /// Constant for method="value", default 0: a number, a string, a bool, or
    /// a date as a tagged instant (`{"epoch_ms": <integer>}`), of the kind the
    /// column holds.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub value: Option<serde_json::Value>,
    /// Group-by columns (impute within groups).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub groupby: Option<Vec<String>>,
    /// Key column: within each group, missing values of this column (against
    /// the key universe, see `keyvals`) get an inserted row.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub key: Option<String>,
    /// Keys added to the observed ones. Requires `key` (a refusal, not a
    /// silent no-op, if `key` is unset). Each value is typed per the key
    /// column's own dtype at apply time: an integral JSON float (e.g. `1.0`)
    /// coerces onto an Int/UInt key column (a non-integral float against an
    /// Int/UInt key column is a typed refusal, since that coercion would be
    /// lossy), a tagged instant
    /// or an epoch-millisecond number is an instant on a temporal key, and any
    /// other dtype mismatch is a typed refusal, not a silent drop.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub keyvals: Option<Keyvals>,
    /// `(preceding, following)` row offsets, `None` unbounded, of the window
    /// each statistic fill reads. Refused for `method="value"`.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub frame: Option<(Option<i64>, Option<i64>)>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

/// The `keyvals` option: the keys listed, or the `{start, stop, step}` sequence
/// that makes them.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub(crate) enum Keyvals {
    List(Vec<serde_json::Value>),
    Sequence(KeySequence),
}

/// Vega's `sequence`: `start` (default 0) up to but not including `stop`, by
/// `step` (default 1, or -1 when `stop < start`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct KeySequence {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub start: Option<f64>,
    pub stop: f64,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub step: Option<f64>,
}

/// The single source of truth for `Impute`'s free-string `method`
/// vocabulary. Read by [`Method::parse`] (the enforcement site), its error
/// message, and `registry::entry_for_kind("impute")`'s `methods` list,
/// mirroring `smooth::METHODS`.
pub(crate) const METHODS: &[&str] = &["value", "mean", "median", "min", "max"];

fn default_method() -> String {
    "value".into()
}

/// The options both doors share, borrowed from whichever spec carries them.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ImputeOptions<'a> {
    pub method: &'a str,
    pub value: Option<&'a serde_json::Value>,
    pub keyvals: Option<&'a Keyvals>,
    pub frame: Option<(Option<i64>, Option<i64>)>,
}

/// The options dict a value channel carries (`Y(impute=...)`, `X(impute=...)`),
/// in the wire form Python writes: the options of [`ImputeSpec`] without the
/// column, whose names and defaults it shares.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ChannelImpute {
    #[serde(default = "default_method")]
    method: String,
    #[serde(default)]
    value: Option<serde_json::Value>,
    #[serde(default)]
    keyvals: Option<Keyvals>,
    #[serde(default)]
    frame: Option<(Option<i64>, Option<i64>)>,
}

impl ChannelImpute {
    /// Read the dict a channel carries, or say why it is not one.
    pub(crate) fn from_wire(wire: &serde_json::Value) -> Result<Self, String> {
        serde_json::from_value(wire.clone()).map_err(|e| format!("impute: {e}"))
    }

    pub(crate) fn options(&self) -> ImputeOptions<'_> {
        ImputeOptions {
            method: &self.method,
            value: self.value.as_ref(),
            keyvals: self.keyvals.as_ref(),
            frame: self.frame,
        }
    }
}

impl ImputeSpec {
    fn options(&self) -> ImputeOptions<'_> {
        ImputeOptions {
            method: &self.method,
            value: self.value.as_ref(),
            keyvals: self.keyvals.as_ref(),
            frame: self.frame,
        }
    }
}

/// Which observed keys join the key universe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ObservedKeys {
    /// Every non-null key: a `NaN` or infinite float key is a key of its own,
    /// completed into every series as `group_key` keys it (`transform_impute`).
    Every,
    /// Only the keys a key axis places: a `NaN` or infinite float key adds no
    /// key, and its row holds no place in the key order (it is treated as a
    /// null-keyed row: never completed, never a slot anchor, and after every
    /// keyed position in a frame). Instants, integers, text and booleans are
    /// always placed (the encoding door).
    Placed,
}

/// The column series are completed along.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ImputeKey<'a> {
    /// The key column.
    pub field: &'a str,
    /// Which observed keys join the universe.
    pub observed: ObservedKeys,
    /// Columns that ride with the key (a binned key's `bin_end`, an `x2`): an
    /// imputed row copies each from the first row holding its key. Rows that
    /// share a key an imputed row is made for must agree on them, and a
    /// `keyvals` key no row holds has nothing to copy, so both are refused. A
    /// key no series lacks needs no copy, so its rows may differ.
    pub companions: &'a [String],
}

/// The columns a completion reads.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ImputeColumns<'a> {
    /// The column whose missing values are filled.
    pub value: &'a str,
    /// The column series are completed along, if any.
    pub key: Option<ImputeKey<'a>>,
    /// The columns an imputed row copies from its series' first row: the
    /// columns that name the series, and any column constant within the batch
    /// the caller hands in (a facet partition's facet columns).
    pub series: &'a [String],
}

/// Which series each row belongs to. Ids may be any numbering of
/// `0..labels.len()`, each id holding at least one row.
#[derive(Debug, Clone)]
pub(crate) struct Series {
    /// Each row's series, `None` for a row in no series (a null in its groupby
    /// key), which takes the whole-column statistic and is never completed.
    pub of_row: Vec<Option<usize>>,
    /// Each series' name as a warning gives it (`g=b`), or `None` for the one
    /// series that is the whole frame.
    pub labels: Vec<Option<String>>,
    /// The encoding door leaves a row in no series as written: its value cell is
    /// never filled, and the caller names it. The transform door fills it with the
    /// whole column's fill (a null `groupby` key's documented contract).
    pub leave_unseriesed: bool,
}

impl Series {
    /// The series a `group_partition_multi` call made, named by `groupby`.
    fn of_partition(partition: &GroupPartition, groupby: &[String], n_rows: usize) -> Self {
        let mut of_row = vec![None; n_rows];
        let mut labels = Vec::with_capacity(partition.groups.len());
        for (index, (key, rows)) in partition.groups.iter().enumerate() {
            for &row in rows {
                of_row[row] = Some(index);
            }
            labels.push((!groupby.is_empty()).then(|| describe_group_keys(groupby, key)));
        }
        Self { of_row, labels, leave_unseriesed: false }
    }
}

/// Why a completion was refused. A plain value, not a Python error, so the
/// transform door and the render each wrap it in their own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ImputeRefusal {
    /// A refusal whose text is final, already prefixed with its door.
    Refused(String),
    /// Rows that share a key an imputed row copies disagree on a companion
    /// column, so the imputed row has no one value to copy. `key` is the first
    /// such key in key order.
    CompanionDiffers { door: String, key: String, companion: String },
    /// A `keyvals` key that no row holds, with a companion to copy.
    CompanionUnheld { door: String, key: String, companion: String },
}

impl std::fmt::Display for ImputeRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Refused(text) => f.write_str(text),
            Self::CompanionDiffers { door, key, companion } => write!(
                f,
                "{door}: rows with key {key} hold different '{companion}' values, so an imputed \
                 row has no one '{companion}' to copy"
            ),
            Self::CompanionUnheld { door, key, companion } => write!(
                f,
                "{door}: keyvals key {key} is held by no row, so an imputed row has no \
                 '{companion}' to copy"
            ),
        }
    }
}

impl From<String> for ImputeRefusal {
    fn from(text: String) -> Self {
        Self::Refused(text)
    }
}

pub(crate) fn apply(spec: &ImputeSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    apply_with_context(spec, batch, &TransformContext::default())
}

/// Context-aware entry point (#162): identical to [`apply`], plus
/// `ctx.warnings` as the sink for this transform's fallback shapes: a group
/// whose own usable values are none, a row whose groupby key is null, and a
/// frame that holds no usable value for a fill. All fall back to a wider
/// statistic (see the module doc).
pub(crate) fn apply_with_context(
    spec: &ImputeSpec,
    batch: &RecordBatch,
    ctx: &TransformContext,
) -> PyResult<RecordBatch> {
    let groupby: &[String] = spec.groupby.as_deref().unwrap_or(&[]);
    if let Some(key) = &spec.key {
        if groupby.contains(key) {
            return Err(PyValueError::new_err(format!(
                "{TRANSFORM_DOOR}: key column '{key}' cannot also be a groupby column"
            )));
        }
    }
    // `FillFromWholeBatch` is #197's stated exemption for this transform:
    // `Impute` is row-preserving by contract, so a null-keyed row is kept and
    // gets the whole column's statistic rather than being dropped (nothing to
    // count), and the all-null warning says so instead of claiming an empty
    // output.
    let partition = group_partition_multi(
        batch,
        groupby,
        TRANSFORM_DOOR,
        NullKeyPolicy::FillFromWholeBatch,
        ctx,
    )?;
    let series = Series::of_partition(&partition, groupby, batch.num_rows());
    let columns = ImputeColumns {
        value: &spec.field,
        key: spec.key.as_deref().map(|field| ImputeKey {
            field,
            observed: ObservedKeys::Every,
            companions: &[],
        }),
        series: groupby,
    };
    complete_series(batch, &columns, &series, &spec.options(), TRANSFORM_DOOR, ctx)
        .map_err(|refusal| PyValueError::new_err(refusal.to_string()))
}

/// The key column of a completion, read once: its place in the schema, its
/// dtype, the universe of keys, each row's place in it and the columns that
/// ride with it.
struct KeyColumn {
    index: usize,
    dtype: DataType,
    universe: KeyUniverse,
    ranks: Vec<Option<usize>>,
    /// The companion columns' places in the schema.
    companions: Vec<usize>,
}

impl KeyColumn {
    fn read(
        batch: &RecordBatch,
        key: &ImputeKey<'_>,
        value_index: usize,
        series_indices: &[usize],
        options: &ImputeOptions<'_>,
        door: &str,
    ) -> Result<Self, ImputeRefusal> {
        let schema = batch.schema();
        let name = key.field;
        let index = schema
            .index_of(name)
            .map_err(|_| format!("{door}: key column '{name}' not found"))?;
        if index == value_index {
            return Err(format!(
                "{door}: key column '{name}' cannot be the same as the imputed field '{}'",
                schema.field(value_index).name()
            )
            .into());
        }
        let dtype = schema.field(index).data_type().clone();
        if !is_groupby_supported_dtype(&dtype) {
            return Err(format!(
                "{door}: key column '{name}' has unsupported dtype {dtype:?}; \
                 supported: {GROUPBY_SUPPORTED_DTYPES}"
            )
            .into());
        }
        let companions = key
            .companions
            .iter()
            .map(|companion| {
                let at = schema
                    .index_of(companion)
                    .map_err(|_| format!("{door}: column '{companion}' not found"))?;
                let role = if at == value_index {
                    Some("the value")
                } else if at == index {
                    Some("the key")
                } else if series_indices.contains(&at) {
                    Some("a series column")
                } else {
                    None
                };
                match role {
                    Some(role) => Err(format!(
                        "{door}: internal error: column '{companion}' cannot be both the key's \
                         companion and {role}"
                    )),
                    None => Ok(at),
                }
            })
            .collect::<Result<Vec<usize>, String>>()?;
        let arr = batch.column(index);
        let universe =
            KeyUniverse::build(arr.as_ref(), &dtype, name, options.keyvals, key.observed, door)?;
        let ranks = universe.row_ranks(arr.as_ref(), &dtype);
        Ok(Self { index, dtype, universe, ranks, companions })
    }

    /// Every row holding a key an imputed row is made for must agree with its
    /// key's first row on each companion (a null equals a null, a `NaN` a
    /// `NaN`), else the first such key in key order is named. A key no series
    /// lacks is copied nowhere, so its rows are free to differ.
    fn check_companions_agree(
        &self,
        imputed: &[Imputed],
        batch: &RecordBatch,
        door: &str,
    ) -> Result<(), ImputeRefusal> {
        if self.companions.is_empty() || imputed.is_empty() {
            return Ok(());
        }
        let mut copied = vec![false; self.universe.len()];
        for row in imputed {
            copied[row.rank] = true;
        }
        let n_rows = batch.num_rows();
        let sources = (0..n_rows)
            .map(|row| {
                let first = self.ranks[row].and_then(|rank| self.universe.first_row(rank));
                u32::try_from(first.unwrap_or(row))
                    .map_err(|_| format!("{door}: internal error: more rows than a batch indexes"))
            })
            .collect::<Result<Vec<u32>, String>>()
            .map(UInt32Array::from)?;
        let schema = batch.schema();
        for &at in &self.companions {
            // `distinct` orders floats totally, which tells 0.0 from -0.0; they are
            // one position, so the sign of zero is not a disagreement.
            let column = unsigned_zero(batch.column(at));
            let door_error = |e: arrow::error::ArrowError| format!("{door}: {e}");
            let firsts = arrow::compute::take(column.as_ref(), &sources, None).map_err(door_error)?;
            let differs = arrow::compute::kernels::cmp::distinct(&column, &firsts).map_err(door_error)?;
            let worst = (0..n_rows)
                .filter(|&row| differs.value(row))
                .filter_map(|row| self.ranks[row])
                .filter(|&rank| copied[rank])
                .min();
            if let Some(rank) = worst {
                return Err(ImputeRefusal::CompanionDiffers {
                    door: door.to_string(),
                    key: self.spelled(rank),
                    companion: schema.field(at).name().clone(),
                });
            }
        }
        Ok(())
    }

    /// The key at place `rank` as a warning spells a key.
    fn spelled(&self, rank: usize) -> String {
        self.universe.key(rank).map_or_else(String::new, format_key_value)
    }

    /// The key column of the imputed rows: each row's key, at the column's dtype.
    fn imputed_column(&self, imputed: &[Imputed]) -> Result<ArrayRef, String> {
        let keys: Vec<Vec<KeyValue>> = imputed
            .iter()
            .filter_map(|row| self.universe.key(row.rank).cloned())
            .map(|key| vec![key])
            .collect();
        materialize_groupby_col(&keys, 0, &self.dtype)
    }

    /// An imputed row has a companion to copy only from a row that holds its
    /// key; the first imputed key (in key order) that no row holds is refused.
    fn check_companions_held(
        &self,
        imputed: &[Imputed],
        batch: &RecordBatch,
        door: &str,
    ) -> Result<(), ImputeRefusal> {
        let Some(&first_companion) = self.companions.first() else {
            return Ok(());
        };
        let unheld = imputed
            .iter()
            .filter(|row| self.universe.first_row(row.rank).is_none())
            .map(|row| row.rank)
            .min();
        match unheld {
            Some(rank) => Err(ImputeRefusal::CompanionUnheld {
                door: door.to_string(),
                key: self.spelled(rank),
                companion: batch.schema().field(first_companion).name().clone(),
            }),
            None => Ok(()),
        }
    }

    /// A companion column of the imputed rows: each row's value from the first
    /// row holding its key.
    fn imputed_companion(&self, column: &ArrayRef, imputed: &[Imputed]) -> Result<ArrayRef, String> {
        let sources = imputed
            .iter()
            .map(|row| {
                self.universe
                    .first_row(row.rank)
                    .and_then(|first| u32::try_from(first).ok())
                    .ok_or_else(|| "internal error: an imputed key has no first row".to_string())
            })
            .collect::<Result<Vec<u32>, String>>()
            .map(UInt32Array::from)?;
        arrow::compute::take(column.as_ref(), &sources, None).map_err(|e| e.to_string())
    }
}

/// What is being completed: the series, each one's rows, the keys each lacks
/// and the key column.
struct Layout<'a> {
    series: &'a Series,
    /// Each series' rows, in ascending batch order; none is empty.
    rows_of: Vec<Vec<usize>>,
    /// Each series' missing keys, in key order.
    missing: Vec<Vec<Missing>>,
    key: Option<KeyColumn>,
    /// The places of the series columns in the schema.
    series_indices: Vec<usize>,
}

impl<'a> Layout<'a> {
    fn read(
        batch: &RecordBatch,
        columns: &ImputeColumns<'_>,
        series: &'a Series,
        value_index: usize,
        options: &ImputeOptions<'_>,
        door: &str,
    ) -> Result<Self, ImputeRefusal> {
        let schema = batch.schema();
        let n_series = series.labels.len();
        let mut rows_of: Vec<Vec<usize>> = vec![Vec::new(); n_series];
        for (row, series_index) in series.of_row.iter().enumerate() {
            if let Some(index) = series_index {
                rows_of
                    .get_mut(*index)
                    .ok_or_else(|| {
                        format!(
                            "{door}: internal error: row {row} is in series {index}, but there \
                             are {n_series} series"
                        )
                    })?
                    .push(row);
            }
        }
        if let Some(empty) = rows_of.iter().position(Vec::is_empty) {
            return Err(format!("{door}: internal error: series {empty} holds no row").into());
        }
        let series_indices = columns
            .series
            .iter()
            .map(|name| {
                schema
                    .index_of(name)
                    .map_err(|_| format!("{door}: groupby column '{name}' not found"))
            })
            .collect::<Result<Vec<usize>, String>>()?;
        let key = columns
            .key
            .as_ref()
            .map(|key| KeyColumn::read(batch, key, value_index, &series_indices, options, door))
            .transpose()?;
        let missing = rows_of
            .iter()
            .map(|rows| match &key {
                Some(key) => missing_keys(rows, &key.ranks, key.universe.len()),
                None => Vec::new(),
            })
            .collect();
        Ok(Self { series, rows_of, missing, key, series_indices })
    }

    /// The imputed rows in the order they are inserted: by slot, then by the
    /// first row of their series, then by key. At one slot at most two series
    /// meet, the one that owns the row before it (its rows go after its last
    /// earlier row) and the one whose first row is the slot (its rows go before
    /// its first row), and the series whose first row is earlier comes first.
    fn imputed(&self) -> Vec<Imputed> {
        let mut rows: Vec<Imputed> = self
            .missing
            .iter()
            .enumerate()
            .flat_map(|(series, keys)| {
                keys.iter().enumerate().map(move |(index, m)| Imputed {
                    slot: m.slot,
                    series,
                    index,
                    rank: m.rank,
                })
            })
            .collect();
        rows.sort_by_key(|row| (row.slot, self.rows_of[row.series][0], row.rank));
        rows
    }
}

/// An imputed row, once its place is decided.
struct Imputed {
    /// The row of the batch it is inserted before.
    slot: usize,
    series: usize,
    /// Its place among its series' missing keys.
    index: usize,
    /// The key's place in the key order.
    rank: usize,
}

/// The fills a statistic method computes: one for every caller row (read only
/// where its cell is missing) and one for every missing key of every series.
struct StatisticFills {
    per_row: Vec<f64>,
    imputed: Vec<Vec<f64>>,
}

/// The value column of a completion: the callers' cells with their missing
/// ones filled, and the values of the imputed rows.
struct FilledValues {
    callers: ArrayRef,
    imputed: ArrayRef,
}

/// Complete every series of `batch` and fill its missing values.
///
/// `series` says which series each row belongs to, `columns` which columns hold
/// the value, the key (with the observed keys it admits and the companions that
/// ride with it) and the series' own values, and `options` how to fill (see the
/// module doc). The key order is the order the key's dtype places its values
/// (see [`keys`]). `door` prefixes every refusal and warning (`data_impute`
/// for the transform).
///
/// The result holds every caller row in its place, a row inserted for each key
/// a series lacks, and the value column filled at its own dtype: a numeric
/// column is read and written as `Float64`, and every other kind keeps its
/// dtype. The schema's metadata and each field's survive.
pub(crate) fn complete_series(
    batch: &RecordBatch,
    columns: &ImputeColumns<'_>,
    series: &Series,
    options: &ImputeOptions<'_>,
    door: &str,
    ctx: &TransformContext,
) -> Result<RecordBatch, ImputeRefusal> {
    let method = Method::parse(door, options.method)?;
    check_options(method, options, columns.key.is_some(), door)?;

    let schema = batch.schema();
    let n_rows = batch.num_rows();
    if series.of_row.len() != n_rows {
        return Err(format!(
            "{door}: internal error: {} series assignments for {n_rows} rows",
            series.of_row.len()
        )
        .into());
    }
    let value_index = schema
        .index_of(columns.value)
        .map_err(|_| format!("{door}: column '{}' not found", columns.value))?;
    let layout = Layout::read(batch, columns, series, value_index, options, door)?;
    let imputed = layout.imputed();
    if let Some(key) = &layout.key {
        key.check_companions_agree(&imputed, batch, door)?;
        key.check_companions_held(&imputed, batch, door)?;
    }

    let value_col = batch.column(value_index);
    // A row in no series is left as written on the encoding door.
    let left_as_written: Vec<bool> = series
        .of_row
        .iter()
        .map(|series_of_row| series_of_row.is_none() && series.leave_unseriesed)
        .collect();
    let filled = match method {
        Method::Statistic(stat) => {
            let col = float64_column(value_col, door, columns.value)?;
            let fills = statistic_fills(stat, &col, options.frame, &layout, ctx);
            FilledValues {
                callers: per_row_values(&col, &fills.per_row, &left_as_written),
                imputed: Arc::new(Float64Array::from_iter_values(
                    imputed.iter().map(|row| fills.imputed[row.series][row.index]),
                )),
            }
        }
        Method::Value => {
            constant_values(value_col, options, columns.value, imputed.len(), door, &left_as_written)?
        }
    };
    warn_rows_in_no_series(series, n_rows, method, layout.key.is_some(), ctx);

    let mut fields: Vec<Field> = schema.fields().iter().map(|f| f.as_ref().clone()).collect();
    fields[value_index] = fields[value_index]
        .clone()
        .with_data_type(filled.callers.data_type().clone())
        .with_nullable(left_as_written.contains(&true));
    let mut out_columns: Vec<ArrayRef> = batch.columns().to_vec();
    out_columns[value_index] = filled.callers.clone();
    let finish = |fields: Vec<Field>, out_columns: Vec<ArrayRef>| {
        let out_schema = Schema::new(fields).with_metadata(schema.metadata().clone());
        RecordBatch::try_new(Arc::new(out_schema), out_columns)
            .map_err(|e| ImputeRefusal::Refused(format!("{door}: {e}")))
    };
    if imputed.is_empty() {
        return finish(fields, out_columns);
    }

    let first_rows = imputed
        .iter()
        .map(|row| {
            u32::try_from(layout.rows_of[row.series][0])
                .map_err(|_| format!("{door}: internal error: more rows than a batch indexes"))
        })
        .collect::<Result<Vec<u32>, String>>()
        .map(UInt32Array::from)?;
    let picks = insertion_picks(n_rows, &imputed);
    let door_error = |e: arrow::error::ArrowError| format!("{door}: {e}");
    for (index, column) in out_columns.iter_mut().enumerate() {
        let appended: ArrayRef = if index == value_index {
            filled.imputed.clone()
        } else if let Some(key) = layout.key.as_ref().filter(|key| key.index == index) {
            key.imputed_column(&imputed)?
        } else if let Some(key) = layout.key.as_ref().filter(|key| key.companions.contains(&index)) {
            key.imputed_companion(column, &imputed).map_err(|e| format!("{door}: {e}"))?
        } else if layout.series_indices.contains(&index) {
            arrow::compute::take(column.as_ref(), &first_rows, None).map_err(door_error)?
        } else {
            // Every other column may carry a null in an imputed row.
            fields[index] = fields[index].clone().with_nullable(true);
            arrow::array::new_null_array(column.data_type(), imputed.len())
        };
        *column = arrow::compute::interleave(&[column.as_ref(), appended.as_ref()], &picks)
            .map_err(door_error)?;
    }
    finish(fields, out_columns)
}

/// The refusals of option pairs that mean nothing together.
fn check_options(
    method: Method,
    options: &ImputeOptions<'_>,
    has_key: bool,
    door: &str,
) -> Result<(), String> {
    if options.keyvals.is_some() && !has_key {
        return Err(format!(
            "{door}: keyvals requires key=; the key universe is meaningless without a key column"
        ));
    }
    match method {
        Method::Statistic(_) if options.value.is_some() => Err(format!(
            "{door}: value= applies to method='value', not '{}'; drop value= or set method='value'",
            options.method
        )),
        Method::Value if options.frame.is_some() => Err(format!(
            "{door}: frame= applies to mean, median, min and max; method 'value' fills a constant"
        )),
        _ => Ok(()),
    }
}

/// Rows no series holds are never completed. On the transform door they are
/// filled with the whole column's statistic, or with the constant; on the
/// encoding door (`Series::leave_unseriesed`) they are left as written, and the caller
/// names them. Reported once, and only when some
/// (not every) row is in no series: when every row is, `group_partition_multi`
/// already said so. A statistic names the fallback it took. A constant fill
/// fell back to nothing, so with no key there is nothing to say, and with a key
/// the notice says the rows were left out of the completion.
fn warn_rows_in_no_series(
    series: &Series,
    n_rows: usize,
    method: Method,
    has_key: bool,
    ctx: &TransformContext,
) {
    let in_none = series.of_row.iter().filter(|s| s.is_none()).count();
    if in_none == 0 || in_none == n_rows {
        return;
    }
    // The encoding door names these rows itself, once across its facet partitions.
    if series.leave_unseriesed {
        return;
    }
    let reason = match (method, has_key) {
        (Method::Statistic(_), _) => format!(
            "{in_none} row(s) have a null groupby key; the whole column's statistic was used instead"
        ),
        (Method::Value, true) => format!(
            "{in_none} row(s) have a null groupby key, so they belong to no group and no key is \
             completed for them"
        ),
        (Method::Value, false) => return,
    };
    ctx.warn(TransformWarning::FallbackApplied {
        transform: "Impute".into(),
        group: None,
        reason,
    });
}

/// A float column with every `-0.0` read as `0.0` (adding `0.0` does it and
/// leaves every other value, `NaN` included, as it was); any other column as it is.
fn unsigned_zero(column: &ArrayRef) -> ArrayRef {
    use arrow::array::Float32Array;
    if let Some(floats) = column.as_any().downcast_ref::<Float64Array>() {
        return Arc::new(floats.unary::<_, arrow::datatypes::Float64Type>(|x| x + 0.0));
    }
    if let Some(floats) = column.as_any().downcast_ref::<Float32Array>() {
        return Arc::new(floats.unary::<_, arrow::datatypes::Float32Type>(|x| x + 0.0));
    }
    column.clone()
}

/// Whether a cell is one the impute fills: null or `NaN` (an infinite cell is
/// the caller's value and is kept).
fn needs_fill(col: &Float64Array, row: usize) -> bool {
    col.is_null(row) || col.value(row).is_nan()
}

/// The value column of a statistic fill: each cell as it is, or its fill.
fn per_row_values(col: &Float64Array, fills: &[f64], left_as_written: &[bool]) -> ArrayRef {
    let values: Vec<Option<f64>> = (0..col.len())
        .map(|row| {
            if left_as_written[row] {
                (!col.is_null(row)).then(|| col.value(row))
            } else if needs_fill(col, row) {
                Some(fills[row])
            } else {
                Some(col.value(row))
            }
        })
        .collect();
    Arc::new(Float64Array::from(values))
}

/// The value column of `method="value"`: the one constant, of the column's own
/// kind, written over each missing cell and into each of the `n_imputed` rows.
fn constant_values(
    value_col: &ArrayRef,
    options: &ImputeOptions<'_>,
    field: &str,
    n_imputed: usize,
    door: &str,
    left_as_written: &[bool],
) -> Result<FilledValues, String> {
    let kind = ValueKind::of(value_col.data_type());
    let literal = resolve_literal(options.value, kind, door, field)?;
    let door_error = |e: arrow::error::ArrowError| format!("{door}: {e}");
    // A numeric column is read as `Float64`, where a `NaN` is a missing cell too.
    let (base, missing): (ArrayRef, BooleanArray) = if kind == ValueKind::Numbers {
        let col = float64_column(value_col, door, field)?;
        let missing = (0..col.len()).map(|row| Some(!left_as_written[row] && needs_fill(&col, row))).collect();
        (Arc::new(col), missing)
    } else {
        let nulls = arrow::compute::is_null(value_col).map_err(door_error)?;
        let missing: BooleanArray =
            nulls.iter().zip(left_as_written).map(|(null, left)| Some(null == Some(true) && !left)).collect();
        (value_col.clone(), missing)
    };
    let dtype = base.data_type().clone();
    let constant = Scalar::new(fill_array(&literal, &dtype, 1, door, field)?);
    Ok(FilledValues {
        callers: arrow::compute::kernels::zip::zip(&missing, &constant, &base).map_err(door_error)?,
        imputed: fill_array(&literal, &dtype, n_imputed, door, field)?,
    })
}

/// For every output row, which of the two columns `interleave` reads it from
/// (0 the caller's, 1 the imputed) and its place there: each imputed row lands
/// before the caller row at its slot.
fn insertion_picks(n_rows: usize, imputed: &[Imputed]) -> Vec<(usize, usize)> {
    let mut picks = Vec::with_capacity(n_rows + imputed.len());
    let mut next = 0;
    for slot in 0..=n_rows {
        while next < imputed.len() && imputed[next].slot == slot {
            picks.push((1, next));
            next += 1;
        }
        if slot < n_rows {
            picks.push((0, slot));
        }
    }
    picks
}

/// The statistic fill of every missing cell and every imputed row: a series'
/// own statistic, or with a `frame` the statistic of the usable values inside
/// each fill's window in key order, where a window with none takes the series'
/// statistic and a series with none takes the whole column's (each warned).
fn statistic_fills(
    stat: Statistic,
    col: &Float64Array,
    frame: Option<(Option<i64>, Option<i64>)>,
    layout: &Layout<'_>,
    ctx: &TransformContext,
) -> StatisticFills {
    // The whole-column statistic, computed once up front. It serves two
    // fallback cases below: a row that belongs to no series and a series whose
    // own usable values are none. When even the whole column has none (a
    // pathological empty-batch input), `degenerate_default` is the pre-groupby
    // implementation's degenerate return.
    let whole = stat.of(clean_float64_values(col, None));
    if whole.is_none() {
        ctx.warn(TransformWarning::FallbackApplied {
            transform: "Impute".into(),
            group: None,
            reason: format!(
                "the column has no {USABLE} values at all; used the {} method's \
                 degenerate default instead",
                stat.name()
            ),
        });
    }
    let whole_fill = whole.unwrap_or_else(|| stat.degenerate_default());

    // A frame open on both sides is the series' whole statistic.
    let frame = frame.filter(|&frame| frame != (None, None));
    // Rows that belong to no series keep the whole column's statistic.
    let mut per_row = vec![whole_fill; col.len()];
    let mut imputed: Vec<Vec<f64>> = Vec::with_capacity(layout.rows_of.len());
    for (index, rows) in layout.rows_of.iter().enumerate() {
        let label = &layout.series.labels[index];
        let own = stat.of(clean_float64_values(col, Some(rows)));
        // `label` is `None` for the one series that is the whole frame: its
        // degenerate case is already reported at `whole` above, and warning
        // here too would double-fire it with an empty group name.
        if let (None, Some(label)) = (own, label) {
            ctx.warn(TransformWarning::FallbackApplied {
                transform: "Impute".into(),
                group: Some(label.clone()),
                // `Display`'s `Some(g)` arm already renders `"group {g} "`
                // ahead of this text, so this clause must not repeat "group".
                reason: format!(
                    "has no {USABLE} values; the whole column's statistic was used instead"
                ),
            });
        }
        let fill = own.unwrap_or(whole_fill);
        for &row in rows {
            per_row[row] = fill;
        }
        let mut series_imputed = vec![fill; layout.missing[index].len()];
        if let Some(frame) = frame {
            let entries = match &layout.key {
                Some(key) => key_ordered(rows, &key.ranks, &layout.missing[index]),
                None => rows.iter().map(|&row| Entry::Caller(row)).collect(),
            };
            let mut fell_back = 0_usize;
            for (position, entry) in entries.iter().enumerate() {
                if matches!(entry, Entry::Caller(row) if !needs_fill(col, *row)) {
                    continue;
                }
                let window = frame_range((position, position), frame, entries.len());
                let value = window_statistic(stat, col, &entries, window).unwrap_or_else(|| {
                    fell_back += 1;
                    fill
                });
                match entry {
                    Entry::Caller(row) => per_row[*row] = value,
                    Entry::Imputed(i) => series_imputed[*i] = value,
                }
            }
            // A series with no usable value of its own already said so above.
            if fell_back > 0 && own.is_some() {
                ctx.warn(frame_fallback(label.as_deref(), fell_back, stat));
            }
        }
        imputed.push(series_imputed);
    }
    StatisticFills { per_row, imputed }
}

/// The notice that `rows` fills of a series had no usable value inside their
/// frame and took the series' whole statistic.
fn frame_fallback(label: Option<&str>, rows: usize, stat: Statistic) -> TransformWarning {
    let name = stat.name();
    match label {
        Some(label) => TransformWarning::FallbackApplied {
            transform: "Impute".into(),
            group: Some(label.to_string()),
            reason: format!(
                "has {rows} row(s) with no {USABLE} value within the frame; the group's whole \
                 {name} was used instead"
            ),
        },
        None => TransformWarning::FallbackApplied {
            transform: "Impute".into(),
            group: None,
            reason: format!(
                "{rows} row(s) have no {USABLE} value within the frame; the whole column's \
                 {name} was used instead"
            ),
        },
    }
}

// ─── PyO3 wrapper ──────────────────────────────────────────────────────────

use crate::transform::core::TransformSpec;
use pyo3::prelude::*;

#[pyclass(module = "ferrum._core", name = "Impute")]
#[derive(Debug, Clone)]
pub(crate) struct PyImpute(pub(crate) TransformSpec);

#[pymethods]
impl PyImpute {
    #[new]
    #[pyo3(signature = (field, *, method = "value", value = None, name = None))]
    fn new(field: String, method: &str, value: Option<f64>, name: Option<String>) -> PyResult<Self> {
        let value = value
            .map(|v| {
                serde_json::Number::from_f64(v).map(serde_json::Value::Number).ok_or_else(|| {
                    PyValueError::new_err("Impute: value must be a finite number")
                })
            })
            .transpose()?;
        Ok(PyImpute(TransformSpec::Impute(ImputeSpec {
            field,
            method: method.into(),
            value,
            groupby: None,
            key: None,
            keyvals: None,
            frame: None,
            name,
        })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Impute(s) => format!(
                "Impute(field='{}', method='{}', value={:?}, name={:?})",
                s.field,
                s.method,
                s.value.as_ref().and_then(serde_json::Value::as_f64),
                s.name,
            ),
            _ => "Impute(?)".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{BooleanArray, Float64Array, Int64Array, StringArray};
    use arrow::datatypes::{Field, Schema};
    use std::sync::Arc;

    fn spec(
        field: &str,
        method: &str,
        value: Option<f64>,
        groupby: Option<Vec<&str>>,
        key: Option<&str>,
        keyvals: Option<Vec<serde_json::Value>>,
    ) -> ImputeSpec {
        ImputeSpec {
            field: field.into(),
            method: method.into(),
            value: value.map(|v| serde_json::json!(v)),
            groupby: groupby.map(|g| g.into_iter().map(String::from).collect()),
            key: key.map(String::from),
            keyvals: keyvals.map(Keyvals::List),
            frame: None,
            name: None,
        }
    }

    #[test]
    fn impute_with_mean() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![
                Some(1.0),
                None,
                Some(3.0),
                None,
                Some(5.0),
            ]))],
        )
        .unwrap();

        let s = spec("x", "mean", None, None, None, None);
        let out = apply(&s, &batch).unwrap();
        let x = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        // mean(1,3,5) = 3.0
        assert_eq!(x.value(0), 1.0);
        assert_eq!(x.value(1), 3.0); // imputed
        assert_eq!(x.value(2), 3.0);
        assert_eq!(x.value(3), 3.0); // imputed
        assert_eq!(x.value(4), 5.0);
    }

    #[test]
    fn impute_with_constant() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![Some(1.0), None, Some(3.0)]))],
        )
        .unwrap();

        let s = spec("x", "value", Some(99.0), None, None, None);
        let out = apply(&s, &batch).unwrap();
        let x = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(x.value(1), 99.0);
    }

    // ── M3 groupby coverage (F-L05-04) ───────────────────────────────────

    /// The finding's own fixture: `transform_impute(groupby=["g"])` used to
    /// fill BOTH groups' gaps with the global mean (65). Group "a"'s own
    /// mean(100,120)=110; group "b"'s own mean(10,30)=20.
    #[test]
    fn f_l05_04_per_group_mean_not_global_blend() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, false),
            Field::new("sales", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "a", "a", "b", "b", "b"])),
                Arc::new(Float64Array::from(vec![
                    Some(100.0),
                    None,
                    Some(120.0),
                    Some(10.0),
                    None,
                    Some(30.0),
                ])),
            ],
        )
        .unwrap();

        let s = spec("sales", "mean", None, Some(vec!["g"]), None, None);
        let out = apply(&s, &batch).unwrap();
        let sales = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(
            (0..6).map(|i| sales.value(i)).collect::<Vec<_>>(),
            vec![100.0, 110.0, 120.0, 10.0, 20.0, 30.0],
            "group 'a' fills with its own mean 110, group 'b' with its own mean 20 — \
             NOT the global blended mean 65"
        );
        // Row-preserving: same row count and the "g" column is untouched.
        assert_eq!(out.num_rows(), 6);
        let g = out.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(g.value(0), "a");
        assert_eq!(g.value(5), "b");
    }

    #[test]
    fn ungrouped_impute_byte_identical_to_single_global_group() {
        // Same values as the F-L05-04 fixture but ungrouped: the global mean
        // over all 4 non-null values (100,120,10,30) is 65.
        let schema = Arc::new(Schema::new(vec![Field::new(
            "sales",
            DataType::Float64,
            true,
        )]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![
                Some(100.0),
                None,
                Some(120.0),
                Some(10.0),
                None,
                Some(30.0),
            ]))],
        )
        .unwrap();

        let s = spec("sales", "mean", None, None, None, None);
        let out = apply(&s, &batch).unwrap();
        let sales = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(
            (0..6).map(|i| sales.value(i)).collect::<Vec<_>>(),
            vec![100.0, 65.0, 120.0, 10.0, 65.0, 30.0],
        );
        assert!(!out.schema().field(0).is_nullable(), "field column stays non-nullable");
    }

    #[test]
    fn int_groupby_column_partitions_correctly() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Int64, false),
            Field::new("v", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1i64, 1, 2, 2])),
                Arc::new(Float64Array::from(vec![Some(10.0), None, Some(100.0), None])),
            ],
        )
        .unwrap();

        let s = spec("v", "mean", None, Some(vec!["g"]), None, None);
        let out = apply(&s, &batch).unwrap();
        let v = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(v.value(1), 10.0, "group 1's only value fills group 1's gap");
        assert_eq!(v.value(3), 100.0, "group 2's only value fills group 2's gap");
    }

    /// A row whose groupby key is null belongs to no group; it still gets a
    /// fill (row-preserving), falling back to the whole-column statistic.
    #[test]
    fn null_groupby_key_row_falls_back_to_whole_column_statistic() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, true),
            Field::new("v", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![Some("a"), Some("a"), None])),
                Arc::new(Float64Array::from(vec![Some(10.0), Some(20.0), None])),
            ],
        )
        .unwrap();

        let s = spec("v", "mean", None, Some(vec!["g"]), None, None);
        let out = apply(&s, &batch).unwrap();
        let v = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        // Group "a" has no gaps. The null-key row (excluded from any group)
        // falls back to the whole column's mean: mean(10, 20) = 15.
        assert_eq!(v.value(2), 15.0);
    }

    // ── key=/keyvals= generation (§4.3) ──────────────────────────────────

    fn key_fixture() -> RecordBatch {
        // Group 1: months {1, 2}; group 2: months {2, 3}. Universe
        // (batch-wide first-appearance) = [1, 2, 3].
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Int64, false),
            Field::new("month", DataType::Float64, false),
            Field::new("sales", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1i64, 1, 2, 2])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 2.0, 3.0])),
                Arc::new(Float64Array::from(vec![10.0, 30.0, 100.0, 300.0])),
            ],
        )
        .unwrap()
    }

    fn col_f64(out: &RecordBatch, idx: usize) -> Vec<f64> {
        out.column(idx).as_any().downcast_ref::<Float64Array>().unwrap().values().to_vec()
    }

    /// The "g" groupby column stays Int64 (pass-through, never coerced).
    fn col_i64(out: &RecordBatch, idx: usize) -> Vec<i64> {
        out.column(idx).as_any().downcast_ref::<Int64Array>().unwrap().values().to_vec()
    }

    /// Group 1 is missing month 3 (appended fill = mean(10,30) = 20); group 2
    /// is missing month 1 (appended fill = mean(100,300) = 200). Wrong group
    /// pairing (e.g. using the other group's fill) would fail this.
    #[test]
    fn missing_key_rows_generated_per_group_with_that_groups_fill() {
        let batch = key_fixture();
        let s = spec("sales", "mean", None, Some(vec!["g"]), Some("month"), None);
        let out = apply(&s, &batch).unwrap();
        assert_eq!(out.num_rows(), 6, "4 original + 2 appended (one per group's missing month)");

        let g = col_i64(&out, 0);
        let month = col_f64(&out, 1);
        let sales = col_f64(&out, 2);
        let mut rows: Vec<(i64, i64, f64)> =
            (0..6).map(|i| (g[i], month[i] as i64, sales[i])).collect();
        rows.sort_by_key(|&(g, m, _)| (g, m));
        assert_eq!(
            rows,
            vec![
                (1, 1, 10.0),
                (1, 2, 30.0),
                (1, 3, 20.0),  // appended: group 1's own fill
                (2, 1, 200.0), // appended: group 2's own fill
                (2, 2, 100.0),
                (2, 3, 300.0),
            ]
        );
        // Appended rows widen every non-field/non-groupby/non-key column's
        // nullability; here there are none besides those three, so nothing
        // else to check, but the field/groupby/key columns themselves must
        // stay non-null throughout.
        assert!(!out.schema().field(2).is_nullable(), "imputed field column stays non-nullable");
    }

    /// `keyvals=` adds to the batch-wide universe: a value absent from the
    /// data entirely still generates one inserted row per group.
    #[test]
    fn keyvals_add_to_the_universe_beyond_observed_values() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Int64, false),
            Field::new("month", DataType::Float64, false),
            Field::new("sales", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1i64, 1, 2, 2])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 1.0, 2.0])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 100.0, 200.0])),
            ],
        )
        .unwrap();

        // Without keyvals: both groups already have {1, 2}, the whole
        // observed universe, so nothing is missing.
        let s_default = spec("sales", "mean", None, Some(vec!["g"]), Some("month"), None);
        let out_default = apply(&s_default, &batch).unwrap();
        assert_eq!(out_default.num_rows(), 4, "no keyvals: nothing is missing");

        // With keyvals=[1,2,3]: month 3 is now in the universe, missing from
        // BOTH groups, so 2 rows get inserted.
        let s_override = spec(
            "sales",
            "mean",
            None,
            Some(vec!["g"]),
            Some("month"),
            Some(vec![serde_json::json!(1.0), serde_json::json!(2.0), serde_json::json!(3.0)]),
        );
        let out_override = apply(&s_override, &batch).unwrap();
        assert_eq!(out_override.num_rows(), 6, "keyvals=[1,2,3] adds month 3 for both groups");
        let month = col_f64(&out_override, 1);
        let sales = col_f64(&out_override, 2);
        let imputed: Vec<f64> =
            (0..6).filter(|&i| month[i] == 3.0).map(|i| sales[i]).collect();
        assert_eq!(imputed, vec![15.0, 150.0], "group 1 fill=mean(10,20)=15, group 2 fill=mean(100,200)=150");
    }

    /// `key` without `groupby`: Vega's "no groupby means one implicit group"
    /// rule — the whole batch is treated as a single group.
    #[test]
    fn key_without_groupby_treats_whole_batch_as_one_group() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("month", DataType::Float64, false),
            Field::new("sales", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
                Arc::new(Float64Array::from(vec![10.0, 20.0])),
            ],
        )
        .unwrap();

        let s = spec(
            "sales",
            "mean",
            None,
            None,
            Some("month"),
            Some(vec![serde_json::json!(1.0), serde_json::json!(2.0), serde_json::json!(3.0)]),
        );
        let out = apply(&s, &batch).unwrap();
        assert_eq!(out.num_rows(), 3, "month 3 is missing from the one implicit group");
        let month = col_f64(&out, 0);
        let sales = col_f64(&out, 1);
        assert_eq!(month[2], 3.0);
        assert_eq!(sales[2], 15.0, "the implicit group's own fill: mean(10,20)=15");
    }

    /// The key column's own dtype (Utf8 here) is preserved on appended rows,
    /// not coerced or corrupted.
    #[test]
    fn key_column_dtype_preserved_on_imputed_rows() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Int64, false),
            Field::new("month", DataType::Utf8, false),
            Field::new("sales", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1i64, 1, 2, 2])),
                Arc::new(StringArray::from(vec!["Jan", "Feb", "Feb", "Mar"])),
                Arc::new(Float64Array::from(vec![10.0, 30.0, 100.0, 300.0])),
            ],
        )
        .unwrap();

        let s = spec("sales", "mean", None, Some(vec!["g"]), Some("month"), None);
        let out = apply(&s, &batch).unwrap();
        assert_eq!(out.num_rows(), 6);
        let month = out.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        let g = col_i64(&out, 0);
        let sales = col_f64(&out, 2);
        let mut imputed: Vec<(i64, String, f64)> = (0..6)
            .filter(|&i| sales[i] == 20.0 || sales[i] == 200.0)
            .map(|i| (g[i], month.value(i).to_string(), sales[i]))
            .collect();
        imputed.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(
            imputed,
            vec![(1, "Mar".to_string(), 20.0), (2, "Jan".to_string(), 200.0)],
            "imputed rows carry a real Utf8 category, not a numeral or corrupted value"
        );
    }

    /// Ungrouped, no key: identical output shape and schema to the
    /// pre-M3 implementation (no appended rows, no schema widening).
    #[test]
    fn ungrouped_no_key_is_byte_identical_shape() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "x",
            DataType::Float64,
            true,
        )]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![Some(1.0), None, Some(3.0)]))],
        )
        .unwrap();
        let s = spec("x", "mean", None, None, None, None);
        let out = apply(&s, &batch).unwrap();
        assert_eq!(out.num_rows(), 3, "no rows appended when key is unset");
        assert_eq!(out.num_columns(), 1);
        assert!(!out.schema().field(0).is_nullable());
    }

    #[test]
    fn keyvals_type_mismatch_refuses() {
        pyo3::Python::initialize();
        let batch = key_fixture();
        let s = spec(
            "sales",
            "mean",
            None,
            Some(vec!["g"]),
            Some("month"),
            Some(vec![serde_json::json!("not-a-number")]),
        );
        let msg = apply(&s, &batch).unwrap_err().to_string();
        assert!(msg.contains("data_impute"), "err: {msg}");
        assert!(msg.contains("keyvals"), "err: {msg}");
        assert!(msg.contains("'month'"), "err must name the key column: {msg}");
    }

    #[test]
    fn key_same_as_field_refuses() {
        pyo3::Python::initialize();
        let batch = key_fixture();
        let s = spec("sales", "mean", None, Some(vec!["g"]), Some("sales"), None);
        let msg = apply(&s, &batch).unwrap_err().to_string();
        assert!(msg.contains("cannot be the same as the imputed field"), "err: {msg}");
    }

    #[test]
    fn key_same_as_groupby_column_refuses() {
        pyo3::Python::initialize();
        let batch = key_fixture();
        let s = spec("sales", "mean", None, Some(vec!["g"]), Some("g"), None);
        let msg = apply(&s, &batch).unwrap_err().to_string();
        assert!(msg.contains("cannot also be a groupby column"), "err: {msg}");
    }

    /// A `Timestamp` key and a `Timestamp` groupby column impute through the one
    /// shared key reader: the missing `(group, day)` row is appended, both date
    /// columns keep their unit and zone, and the fill is the group's own.
    #[test]
    fn a_timestamp_key_and_groupby_impute_and_keep_their_dtype() {
        pyo3::Python::initialize();
        use arrow::array::TimestampMillisecondArray;
        use arrow::datatypes::TimeUnit;
        let zoned = DataType::Timestamp(TimeUnit::Millisecond, Some(Arc::from("Europe/Paris")));
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Int64, false),
            Field::new("day", zoned.clone(), false),
            Field::new("sales", DataType::Float64, true),
        ]));
        let day = |values: Vec<i64>| {
            Arc::new(TimestampMillisecondArray::from(values).with_timezone("Europe/Paris"))
        };
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1i64, 1, 2])),
                day(vec![86_400_000, 172_800_000, 86_400_000]),
                Arc::new(Float64Array::from(vec![Some(10.0), Some(30.0), Some(100.0)])),
            ],
        )
        .unwrap();
        let s = spec("sales", "mean", None, Some(vec!["g"]), Some("day"), None);
        let out = apply(&s, &batch).unwrap();
        // Group 2 lacks the second day: one row appended after the three caller rows.
        assert_eq!(out.num_rows(), 4);
        assert_eq!(out.schema().field(1).data_type(), &zoned);
        assert_eq!(out.column(1).to_data(), day(vec![86_400_000, 172_800_000, 86_400_000, 172_800_000]).to_data());
        assert_eq!(col_i64(&out, 0), vec![1, 1, 2, 2]);
        assert_eq!(col_f64(&out, 2), vec![10.0, 30.0, 100.0, 100.0]);

        // A timestamp groupby column keeps its dtype on the same transform.
        let by_day = spec("sales", "mean", None, Some(vec!["day"]), None, None);
        let by_day = apply(&by_day, &batch).unwrap();
        assert_eq!(by_day.schema().field(1).data_type(), &zoned);
        assert_eq!(by_day.num_rows(), 3);
    }

    /// An unsupported key dtype is refused with the one shared supported-dtypes
    /// text, `Timestamp` included.
    #[test]
    fn the_unsupported_key_refusal_quotes_the_shared_supported_dtypes() {
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![
            Field::new("d", DataType::Date32, false),
            Field::new("v", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(arrow::array::Date32Array::from(vec![0, 1])),
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
            ],
        )
        .unwrap();
        let s = spec("v", "mean", None, None, Some("d"), None);
        let msg = apply(&s, &batch).unwrap_err().to_string();
        assert!(
            msg.ends_with(&format!("has unsupported dtype Date32; supported: {GROUPBY_SUPPORTED_DTYPES}")),
            "{msg}"
        );
    }

    #[test]
    fn unsupported_key_dtype_refuses() {
        pyo3::Python::initialize();
        // Date32 is not a supported groupby/key dtype.
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Int64, false),
            Field::new("d", DataType::Date32, false),
            Field::new("v", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1i64, 2])),
                Arc::new(arrow::array::Date32Array::from(vec![0, 1])),
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
            ],
        )
        .unwrap();
        let s = spec("v", "mean", None, Some(vec!["g"]), Some("d"), None);
        let msg = apply(&s, &batch).unwrap_err().to_string();
        assert!(msg.contains("unsupported dtype"), "err: {msg}");
        assert!(msg.contains("'d'"), "err: {msg}");
    }

    #[test]
    fn unknown_method_refuses_even_on_empty_batch() {
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, true)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(Vec::<f64>::new()))]).unwrap();
        let s = spec("x", "bogus", None, None, None, None);
        let msg = apply(&s, &batch).unwrap_err().to_string();
        assert!(msg.contains("unknown method 'bogus'"), "err: {msg}");
    }

    #[test]
    fn boolean_key_column_supported() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Int64, false),
            Field::new("flag", DataType::Boolean, false),
            Field::new("v", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1i64, 2])),
                Arc::new(BooleanArray::from(vec![true, false])),
                Arc::new(Float64Array::from(vec![10.0, 20.0])),
            ],
        )
        .unwrap();
        // Universe = {true, false}; group 1 only has `true`, so `false` is
        // appended for it; group 2 only has `false`, so `true` is appended.
        let s = spec("v", "mean", None, Some(vec!["g"]), Some("flag"), None);
        let out = apply(&s, &batch).unwrap();
        assert_eq!(out.num_rows(), 4);
    }

    // ── Round-2 fixes (rust-quality-reviewer required fixes 1-6) ─────────

    /// Fix 1: a group whose own cleaned value set is empty (every value is
    /// null/NaN) must fall back to the WHOLE-COLUMN statistic instead of
    /// `fill_value_for`'s degenerate return (`0.0` for mean/median, `±inf`
    /// for min/max), for every data-dependent method.
    #[test]
    fn empty_group_falls_back_to_whole_column_statistic_for_every_method() {
        let make_batch = || {
            let schema = Arc::new(Schema::new(vec![
                Field::new("g", DataType::Utf8, false),
                Field::new("sales", DataType::Float64, true),
            ]));
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec!["a", "a", "b", "b"])),
                    Arc::new(Float64Array::from(vec![Some(100.0), Some(120.0), None, None])),
                ],
            )
            .unwrap()
        };
        // Group "b" has zero non-null values, so its OWN fill would be
        // degenerate (0.0 for mean/median, ±inf for min/max) without the
        // fallback. The whole-column statistic over [100, 120] is the
        // correct fallback for every method.
        let expected: &[(&str, f64)] =
            &[("mean", 110.0), ("median", 110.0), ("min", 100.0), ("max", 120.0)];
        for &(method, fallback) in expected {
            let batch = make_batch();
            let s = spec("sales", method, None, Some(vec!["g"]), None, None);
            let out = apply(&s, &batch).unwrap();
            let sales = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
            assert_eq!(out.num_rows(), 4, "method {method}: row-preserving even for an all-null group");
            assert_eq!(sales.value(0), 100.0, "method {method}: group a's real value untouched");
            assert_eq!(sales.value(1), 120.0, "method {method}: group a's real value untouched");
            assert_eq!(
                sales.value(2),
                fallback,
                "method {method}: group b (all-null) must fall back to the whole-column \
                 statistic {fallback}, not a degenerate 0.0/±inf"
            );
            assert_eq!(
                sales.value(3),
                fallback,
                "method {method}: group b's second row same fallback"
            );
        }
    }

    /// Fix 1 (continued): the same fallback must be the value carried by
    /// an APPENDED key row generated for the empty group, not the
    /// degenerate `fill_value_for` return.
    #[test]
    fn empty_group_fallback_also_used_for_that_groups_appended_key_rows() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, false),
            Field::new("month", DataType::Int64, false),
            Field::new("sales", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "a", "b"])),
                Arc::new(Int64Array::from(vec![1i64, 2, 1])),
                Arc::new(Float64Array::from(vec![Some(100.0), Some(120.0), None])),
            ],
        )
        .unwrap();
        // Universe (batch-wide) = {1, 2}; group "b" only has month 1, so
        // month 2 is missing and gets appended for it.
        let s = spec("sales", "mean", None, Some(vec!["g"]), Some("month"), None);
        let out = apply(&s, &batch).unwrap();
        assert_eq!(out.num_rows(), 4, "3 original + 1 appended (group b's missing month 2)");
        let g = out.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let month = col_i64(&out, 1);
        let sales = col_f64(&out, 2);
        let appended = (0..4)
            .find(|&i| g.value(i) == "b" && month[i] == 2)
            .expect("appended row for group b, month 2");
        assert_eq!(
            sales[appended], 110.0,
            "group b's appended row must use the whole-column fallback (110), not 0.0"
        );
    }

    /// Fix 2: a Float32 key column's `keyvals` entries must round-trip
    /// through `f32` the same way `group_key::groupby_key_at` does, or
    /// every keyvals entry is judged missing from every group (spurious
    /// duplicate rows).
    #[test]
    fn float32_key_column_with_fractional_keyvals_matches_observed_rows() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("month", DataType::Float32, false),
            Field::new("sales", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(arrow::array::Float32Array::from(vec![0.1f32, 0.2, 0.2, 0.3])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0])),
            ],
        )
        .unwrap();
        let s = spec(
            "sales",
            "mean",
            None,
            None,
            Some("month"),
            Some(vec![serde_json::json!(0.1), serde_json::json!(0.2), serde_json::json!(0.3)]),
        );
        let out = apply(&s, &batch).unwrap();
        assert_eq!(
            out.num_rows(),
            4,
            "keyvals=[0.1,0.2,0.3] already covers every observed Float32 month; a \
             Float32-bits mismatch would spuriously append one row per keyvals entry"
        );
    }

    /// An integral JSON float against an Int key column is a numeric, not a dtype mismatch.
    #[test]
    fn keyvals_integral_float_coerces_onto_int_key_column() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("month", DataType::Int64, false),
            Field::new("sales", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1i64, 2])),
                Arc::new(Float64Array::from(vec![10.0, 20.0])),
            ],
        )
        .unwrap();
        let s = spec(
            "sales",
            "mean",
            None,
            None,
            Some("month"),
            Some(vec![serde_json::json!(1.0), serde_json::json!(2.0), serde_json::json!(3.0)]),
        );
        let out = apply(&s, &batch).unwrap();
        assert_eq!(
            out.num_rows(),
            3,
            "1.0/2.0 coerce onto Int64 and match the two observed rows; 3.0 is appended"
        );
    }

    /// Fix 4 (continued): the genuinely lossy case — a non-integral float
    /// against an Int key column — must stay a typed refusal.
    #[test]
    fn keyvals_non_integral_float_against_int_key_column_still_refuses() {
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![
            Field::new("month", DataType::Int64, false),
            Field::new("sales", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1i64, 2])),
                Arc::new(Float64Array::from(vec![10.0, 20.0])),
            ],
        )
        .unwrap();
        let s = spec("sales", "mean", None, None, Some("month"), Some(vec![serde_json::json!(1.5)]));
        let msg = apply(&s, &batch).unwrap_err().to_string();
        assert!(msg.contains("not compatible with key column 'month'"), "err: {msg}");
    }

    /// Fix 3: `keyvals` without `key` is a refusal, not a silent no-op.
    #[test]
    fn keyvals_without_key_refuses() {
        pyo3::Python::initialize();
        let batch = key_fixture();
        let s = spec(
            "sales",
            "mean",
            None,
            Some(vec!["g"]),
            None,
            Some(vec![serde_json::json!(1.0)]),
        );
        let msg = apply(&s, &batch).unwrap_err().to_string();
        assert!(msg.contains("keyvals requires key"), "err: {msg}");
    }

    /// Fix 1 (range check), part 1: an integer keyval that overflows the key
    /// column's specific integer width must refuse with an incompatible error,
    /// not silently wrap. `Int32` has max = 2,147,483,647; `3000000000` is
    /// out of range, so it must refuse.
    #[test]
    fn keyvals_out_of_range_int32_refuses() {
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![
            Field::new("month", DataType::Int32, false),
            Field::new("sales", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(arrow::array::Int32Array::from(vec![1i32, 2])),
                Arc::new(Float64Array::from(vec![10.0, 20.0])),
            ],
        )
        .unwrap();
        // 3000000000 is out of range for Int32 (max 2147483647).
        let s = spec(
            "sales",
            "mean",
            None,
            None,
            Some("month"),
            Some(vec![serde_json::json!(1), serde_json::json!(3000000000i64)]),
        );
        let msg = apply(&s, &batch).unwrap_err().to_string();
        assert!(msg.contains("not compatible with key column 'month'"), "err: {msg}");
        assert!(msg.contains("Int32"), "error must name the dtype 'Int32': {msg}");
    }

    /// Fix 1 (range check), part 2: an integer keyval that would wrap to an
    /// existing key's value when narrowed (the worse shape: it manufactures
    /// a duplicate of an already-present key) must refuse, not silently
    /// duplicate. `Int8` has max = 127; `257 % 256 = 1`, so 257 wrapped as
    /// Int8 is 1, which is already in the key column and would create a
    /// duplicate row.
    #[test]
    fn keyvals_wrap_onto_existing_key_int8_refuses() {
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![
            Field::new("month", DataType::Int8, false),
            Field::new("sales", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(arrow::array::Int8Array::from(vec![1i8, 2])),
                Arc::new(Float64Array::from(vec![10.0, 20.0])),
            ],
        )
        .unwrap();
        // 257 is out of range for Int8 (max 127). It would wrap to 1 (257 % 256 = 1 as u8,
        // reinterpreted as signed), creating a duplicate of the existing key 1.
        let s = spec(
            "sales",
            "mean",
            None,
            None,
            Some("month"),
            Some(vec![serde_json::json!(1), serde_json::json!(2), serde_json::json!(257i64)]),
        );
        let msg = apply(&s, &batch).unwrap_err().to_string();
        assert!(msg.contains("not compatible with key column 'month'"), "err: {msg}");
        assert!(msg.contains("Int8"), "error must name the dtype 'Int8': {msg}");
    }

    /// Fix 6 (extra-column coverage): §4.3's "every other column is null"
    /// on an appended row, previously unreached by any fixture in either
    /// suite.
    #[test]
    fn imputed_row_leaves_unrelated_column_null_and_widens_its_nullability() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Int64, false),
            Field::new("month", DataType::Int64, false),
            Field::new("sales", DataType::Float64, false),
            Field::new("note", DataType::Utf8, false), // neither field, key, nor groupby
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1i64, 1, 2, 2])),
                Arc::new(Int64Array::from(vec![1i64, 2, 2, 3])),
                Arc::new(Float64Array::from(vec![10.0, 30.0, 100.0, 300.0])),
                Arc::new(StringArray::from(vec!["p", "q", "r", "s"])),
            ],
        )
        .unwrap();
        // Universe (batch-wide) = {1, 2, 3}. Group 1 (months {1,2}) is
        // missing 3; group 2 (months {2,3}) is missing 1.
        let s = spec("sales", "mean", None, Some(vec!["g"]), Some("month"), None);
        let out = apply(&s, &batch).unwrap();
        assert_eq!(out.num_rows(), 6, "4 original + 2 appended");
        assert!(
            out.schema().field(3).is_nullable(),
            "'note' must widen to nullable once a row is appended"
        );
        let note = out.column(3).as_any().downcast_ref::<StringArray>().unwrap();
        let sales = col_f64(&out, 2);
        let kept: Vec<&str> = (0..6).filter(|&i| !note.is_null(i)).map(|i| note.value(i)).collect();
        assert_eq!(kept, vec!["p", "q", "r", "s"], "original rows keep their 'note', in order");
        for i in 0..6 {
            let is_original = [10.0, 30.0, 100.0, 300.0].contains(&sales[i]);
            assert_eq!(note.is_null(i), !is_original, "row {i}: only an imputed row has a null 'note'");
        }
    }

    /// The 64-bit boundary (2^63 for signed, 2^64 for unsigned)
    /// must refuse, not saturate to the max representable value. The hardening tightens
    /// the upper bound from `f <= i64::MAX as f64` (which equals 2^63, one above actual max)
    /// to `f < -(i64::MIN as f64)` (which equals 2^63). `2^63` itself is a keyval user might
    /// produce from Python via `2**63`, and it must refuse with an incompatible error,
    /// not silently key `i64::MAX`.
    #[test]
    fn keyvals_2pow63_on_int64_key_column_refuses() {
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![
            Field::new("month", DataType::Int64, false),
            Field::new("sales", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1i64, 2])),
                Arc::new(Float64Array::from(vec![10.0, 20.0])),
            ],
        )
        .unwrap();
        // 2^63 is 9223372036854775808.0, which is out of range for Int64 (max 9223372036854775807).
        // Expressed as JSON, this becomes the literal number 9223372036854775808.
        let s = spec(
            "sales",
            "mean",
            None,
            None,
            Some("month"),
            Some(vec![serde_json::json!(1), serde_json::json!(9223372036854775808.0)]),
        );
        let msg = apply(&s, &batch).unwrap_err().to_string();
        assert!(msg.contains("not compatible with key column 'month'"), "err: {msg}");
        assert!(msg.contains("Int64"), "error must name the dtype 'Int64': {msg}");
    }

    // ── #162: degrade-warning channel ────────────────────────────────────

    /// RED (pre-#162): fails on the pre-change code, which had no channel to
    /// report a group whose own cleaned value set is empty falling back to
    /// the whole-column statistic (`f_l05_04`-style fixture's sibling: here
    /// group "b" has zero non-null values).
    #[test]
    fn impute_empty_group_pushes_a_fallback_applied_warning() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, false),
            Field::new("sales", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "a", "b", "b"])),
                Arc::new(Float64Array::from(vec![Some(100.0), Some(120.0), None, None])),
            ],
        )
        .unwrap();
        let s = spec("sales", "mean", None, Some(vec!["g"]), None, None);
        let ctx = TransformContext::default();
        let out = apply_with_context(&s, &batch, &ctx).unwrap();
        assert_eq!(out.num_rows(), 4, "output is unchanged: still row-preserving");
        let warnings = ctx.warnings.borrow();
        assert_eq!(warnings.len(), 1, "got {warnings:?}");
        assert!(matches!(
            &warnings[0],
            TransformWarning::FallbackApplied { transform, group, .. }
                if transform == "Impute" && group.as_deref() == Some("g=b")
        ));
        // S1 finding (wave-1 close remediation): the rendered sentence must
        // not say "group" twice ("group g=b the group has no ...").
        let text = warnings[0].to_string();
        assert_eq!(
            text.matches("group").count(), 1,
            "\"group\" must appear exactly once in: {text:?}"
        );
    }

    /// A row whose groupby key is null falls back to the whole-column
    /// statistic and warns once (not once per row).
    #[test]
    fn impute_null_groupby_key_row_pushes_one_fallback_applied_warning() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, true),
            Field::new("v", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![Some("a"), Some("a"), None, None])),
                Arc::new(Float64Array::from(vec![Some(10.0), Some(20.0), None, None])),
            ],
        )
        .unwrap();
        let s = spec("v", "mean", None, Some(vec!["g"]), None, None);
        let ctx = TransformContext::default();
        let _ = apply_with_context(&s, &batch, &ctx).unwrap();
        let warnings = ctx.warnings.borrow();
        assert_eq!(
            warnings.len(), 1,
            "two null-keyed rows must produce ONE warning, not two; got {warnings:?}"
        );
        assert!(matches!(
            &warnings[0],
            TransformWarning::FallbackApplied { transform, group, reason }
                if transform == "Impute" && group.is_none() && reason.contains("2 row")
        ));
    }

    /// `Impute` grouped by a partly null key pushes
    /// `FallbackApplied` and no [`TransformWarning::NullKeyRowsDropped`] —
    /// `FillFromWholeBatch` is Impute's own exemption from the fit family's
    /// (`Smooth`/`Robust`/`Logistic`/`Glm`) `SkipWithWholeBatchFallback`:
    /// row-preserving, it keeps and fills every null-keyed row rather than
    /// dropping it, so there is no drop to count.
    #[test]
    fn impute_partly_null_groupby_key_pushes_fallback_applied_and_no_drop_count() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("h", DataType::Utf8, true),
            Field::new("v", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![Some("a"), Some("a"), None, None])),
                Arc::new(Float64Array::from(vec![Some(10.0), Some(20.0), None, None])),
            ],
        )
        .unwrap();
        let s = spec("v", "mean", None, Some(vec!["h"]), None, None);
        let ctx = TransformContext::default();
        let _ = apply_with_context(&s, &batch, &ctx).unwrap();
        let warnings = ctx.warnings.borrow();
        assert_eq!(warnings.len(), 1, "got {warnings:?}");
        assert!(
            matches!(&warnings[0], TransformWarning::FallbackApplied { transform, .. } if transform == "Impute"),
            "{:?}", warnings[0]
        );
        assert!(
            !warnings.iter().any(|w| matches!(w, TransformWarning::NullKeyRowsDropped { .. })),
            "Impute's row-preserving fill drops no row to count: {warnings:?}"
        );
    }

    /// A well-supported group must not warn.
    #[test]
    fn impute_healthy_group_does_not_warn() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, false),
            Field::new("v", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "a"])),
                Arc::new(Float64Array::from(vec![Some(10.0), None])),
            ],
        )
        .unwrap();
        let s = spec("v", "mean", None, Some(vec!["g"]), None, None);
        let ctx = TransformContext::default();
        let _ = apply_with_context(&s, &batch, &ctx).unwrap();
        assert!(ctx.warnings.borrow().is_empty());
    }

    /// An UNGROUPED
    /// `Impute` (no `groupby`) with a data-dependent method over a wholly
    /// null/NaN column used to double-fire — one `FallbackApplied { group:
    /// None, .. }` from the `whole_column_fv` check, and a SECOND
    /// `FallbackApplied { group: Some(""), .. }` from the per-group loop's
    /// singleton implicit whole-batch group (`group_partition_multi`'s
    /// empty-`group_cols` contract), whose `describe_group_keys(&[], &[])`
    /// formats as an empty string rather than `None`. Exactly ONE warning,
    /// with `group: None`, must fire for this genuinely-whole-frame
    /// degenerate case.
    #[test]
    fn impute_ungrouped_all_null_column_pushes_exactly_one_whole_frame_warning() {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, true)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![None::<f64>, None, None]))],
        )
        .unwrap();
        let s = spec("v", "mean", None, None, None, None);
        let ctx = TransformContext::default();
        let out = apply_with_context(&s, &batch, &ctx).unwrap();
        assert_eq!(out.num_rows(), 3, "output is unchanged: still row-preserving");
        let warnings = ctx.warnings.borrow();
        assert_eq!(
            warnings.len(), 1,
            "an ungrouped all-null column must push exactly ONE warning, not double-fire; \
             got {warnings:?}"
        );
        assert!(matches!(
            &warnings[0],
            TransformWarning::FallbackApplied { transform, group, .. }
                if transform == "Impute" && group.is_none()
        ), "the one warning must have group: None (genuinely whole-frame), not an \
            empty-string label; got {:?}", warnings[0]);
    }

    /// GH #197 EXEMPTION: `Impute` is row-preserving by contract, so an
    /// all-null `groupby` column cannot mean "empty output". Every row still
    /// gets the whole-column statistic, and the warning states THAT
    /// disposition rather than claiming an emptiness that never happened.
    #[test]
    fn impute_all_null_group_keys_fill_from_the_whole_column_and_warn() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, true),
            Field::new("v", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![None, None, None] as Vec<Option<&str>>)),
                Arc::new(Float64Array::from(vec![Some(10.0), Some(20.0), None])),
            ],
        )
        .unwrap();
        let s = spec("v", "mean", None, Some(vec!["g"]), None, None);
        let ctx = TransformContext::default();
        let out = apply_with_context(&s, &batch, &ctx).unwrap();
        assert_eq!(out.num_rows(), 3, "row-preserving: no row is dropped");
        let filled = out
            .column(out.schema().index_of("v").unwrap())
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(filled.value(2), 15.0, "the null row takes the whole column's mean");

        let warnings = ctx.warnings.borrow();
        // An all-null groupby
        // key used to ALSO push a `FallbackApplied` from the null-keyed-row
        // rollup below `GroupKeyAllNull` — the same condition reported
        // twice. Exactly one warning total now: `GroupKeyAllNull` alone
        // already says every row fell back to the whole-column statistic.
        assert_eq!(
            warnings.len(), 1,
            "an all-null groupby key must push exactly one warning, not double-report the \
             same condition; got {warnings:?}"
        );
        let all_null: Vec<&TransformWarning> = warnings
            .iter()
            .filter(|w| matches!(w, TransformWarning::GroupKeyAllNull { .. }))
            .collect();
        assert_eq!(all_null.len(), 1, "got {warnings:?}");
        assert!(
            all_null[0]
                .to_string()
                .contains("the whole batch was used as one ungrouped computation instead"),
            "{}",
            all_null[0]
        );
    }

    // Note: `impute_null_groupby_key_row_pushes_one_fallback_applied_warning`
    // above already pins the genuinely PARTIAL case (2 of 4 rows null-keyed,
    // not every row) — the all-null fix just above must not touch that path,
    // since `null_keyed_rows < n_rows` there keeps the rollup's own warning
    // live.

    /// The fill statistic skips an infinite value as it skips a null one, and
    /// the infinite cell itself is the caller's value: kept, never filled.
    #[test]
    fn the_fill_statistic_skips_an_infinity_and_an_infinite_cell_is_kept() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, true)]));
        let frame_of = |third: Option<f64>| {
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Float64Array::from(vec![Some(1.0), Some(3.0), third, None, Some(5.0)]))],
            )
            .unwrap()
        };
        for method in ["mean", "median", "min", "max"] {
            let fill_of = |batch: &RecordBatch| {
                let out = apply(&spec("x", method, None, None, None, None), batch).unwrap();
                let x = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
                x.value(3)
            };
            let twin = fill_of(&frame_of(None));
            for infinite in [f64::INFINITY, f64::NEG_INFINITY] {
                assert_eq!(fill_of(&frame_of(Some(infinite))), twin, "{method}");
            }
        }
        let out = apply(&spec("x", "mean", None, None, None, None), &frame_of(Some(f64::INFINITY))).unwrap();
        let x = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!((x.value(2), x.value(3)), (f64::INFINITY, 3.0), "the infinite cell is kept, the null cell filled");
    }

    /// A column with no usable value says so in the words every reason uses:
    /// an infinity is not a value the statistic could be read from.
    #[test]
    fn a_column_with_no_usable_value_names_usable_in_its_fallback() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, true)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![Some(f64::INFINITY), None]))],
        )
        .unwrap();
        let ctx = TransformContext::default();
        apply_with_context(&spec("x", "mean", None, None, None, None), &batch, &ctx).unwrap();
        let warnings = ctx.warnings.borrow();
        let text = warnings.iter().map(ToString::to_string).collect::<Vec<_>>().join("\n");
        assert!(text.contains("the column has no usable (non-null, finite) values at all"), "{text}");
    }

    /// A group whose only values are infinite has no usable value, so it falls
    /// back to the whole column's statistic and says so in the shared words;
    /// its null cell is filled and its infinite cell is kept.
    #[test]
    fn a_group_of_infinities_falls_back_to_the_whole_column_in_the_usable_words() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Float64, true),
            Field::new("g", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![Some(2.0), Some(4.0), Some(f64::INFINITY), None])),
                Arc::new(StringArray::from(vec!["a", "a", "b", "b"])),
            ],
        )
        .unwrap();
        let ctx = TransformContext::default();
        let out = apply_with_context(&spec("v", "mean", None, Some(vec!["g"]), None, None), &batch, &ctx).unwrap();
        let v = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!((v.value(2), v.value(3)), (f64::INFINITY, 3.0));
        let warnings = ctx.warnings.borrow();
        let text = warnings.iter().map(ToString::to_string).collect::<Vec<_>>().join("\n");
        assert!(text.contains("group g=b has no usable (non-null, finite) values; the whole column's statistic was used instead"), "{text}");
    }
}
