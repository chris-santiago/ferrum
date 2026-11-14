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
//! statistic. That fallback is this transform's exemption from the drop rule:
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

/// The two doors a completion is reached through. A door decides everything
/// the two callers differ in, so no caller can combine one door's key rule
/// with the other's row rule or voice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ImputeDoor {
    /// `transform_impute`: every non-null key joins the key universe, a row in
    /// no series (a null `groupby` key) is filled with the whole column's fill,
    /// and every refusal and warning speaks as `data_impute`.
    Transform,
    /// `Y(impute=)` and `X(impute=)`: only the keys a key axis places join the
    /// universe, a row in no series is left as written (the caller names it),
    /// and every refusal and warning speaks as `impute`.
    Encoding,
}

impl ImputeDoor {
    /// The prefix of every refusal and warning this door raises.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Transform => "data_impute",
            Self::Encoding => "impute",
        }
    }

    fn observed(self) -> ObservedKeys {
        match self {
            Self::Transform => ObservedKeys::Every,
            Self::Encoding => ObservedKeys::Placed,
        }
    }

    /// Whether a row in no series keeps its value cell as written.
    fn leaves_unseriesed(self) -> bool {
        matches!(self, Self::Encoding)
    }
}

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

/// `Impute`'s free-string `method` vocabulary as the registry publishes it
/// (`registry::entry_for_kind("impute")`'s `methods` list), mirroring
/// `smooth::METHODS`. [`Method`]'s `Vocabulary` impl declares the same
/// spellings and is what [`Method::parse`] enforces; a test pins the two
/// equal.
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
enum ObservedKeys {
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
        Self { of_row, labels }
    }
}

/// Why a completion was refused. A plain value, not a Python error, so the
/// transform door and the render each wrap it in their own. Only the encoding
/// door has companion columns, and it words the two companion refusals in the
/// channels the caller wrote (`x2`, `y2`) from the key alone; the `Display`
/// text of those two names the structure, for a caller with no channels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ImputeRefusal {
    /// A refusal whose text is final, already prefixed with its door.
    Refused(String),
    /// Rows that share a key an imputed row copies disagree on a companion
    /// column, so the imputed row has no one value to copy. `key` is the first
    /// such key in key order.
    CompanionDiffers { key: String },
    /// A `keyvals` key that no row holds, with a companion to copy.
    CompanionUnheld { key: String },
}

impl std::fmt::Display for ImputeRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Refused(text) => f.write_str(text),
            Self::CompanionDiffers { key } => write!(
                f,
                "rows with key {key} hold different companion values, so an imputed row has no \
                 one companion value to copy"
            ),
            Self::CompanionUnheld { key } => write!(
                f,
                "keyvals key {key} is held by no row, so an imputed row has no companion value \
                 to copy"
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

/// Context-aware entry point: identical to [`apply`], plus
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
                "{}: key column '{key}' cannot also be a groupby column",
                ImputeDoor::Transform.name()
            )));
        }
    }
    // `FillFromWholeBatch` is this transform's exemption from the drop rule:
    // `Impute` is row-preserving by contract, so a null-keyed row is kept and
    // gets the whole column's statistic rather than being dropped (nothing to
    // count), and the all-null warning says so instead of claiming an empty
    // output.
    let partition = group_partition_multi(
        batch,
        groupby,
        ImputeDoor::Transform.name(),
        NullKeyPolicy::FillFromWholeBatch,
        ctx,
    )?;
    let series = Series::of_partition(&partition, groupby, batch.num_rows());
    let columns = ImputeColumns {
        value: &spec.field,
        key: spec.key.as_deref().map(|field| ImputeKey { field, companions: &[] }),
        series: groupby,
    };
    complete_series(batch, &columns, &series, &spec.options(), ImputeDoor::Transform, ctx)
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
        mode: ImputeDoor,
    ) -> Result<Self, ImputeRefusal> {
        let door = mode.name();
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
            KeyUniverse::build(arr.as_ref(), &dtype, name, options.keyvals, mode.observed(), door)?;
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
                return Err(ImputeRefusal::CompanionDiffers { key: self.spelled(rank) });
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
    fn check_companions_held(&self, imputed: &[Imputed]) -> Result<(), ImputeRefusal> {
        if self.companions.is_empty() {
            return Ok(());
        }
        let unheld = imputed
            .iter()
            .filter(|row| self.universe.first_row(row.rank).is_none())
            .map(|row| row.rank)
            .min();
        match unheld {
            Some(rank) => Err(ImputeRefusal::CompanionUnheld { key: self.spelled(rank) }),
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
        mode: ImputeDoor,
    ) -> Result<Self, ImputeRefusal> {
        let door = mode.name();
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
            .map(|key| KeyColumn::read(batch, key, value_index, &series_indices, options, mode))
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
/// the value, the key (with the companions that
/// ride with it) and the series' own values, and `options` how to fill (see the
/// module doc). The key order is the order the key's dtype places its values
/// (see [`keys`]). `mode` is the door the completion is reached through: it
/// decides which observed keys join the universe, whether a row in no series
/// is left as written, and the prefix of every refusal and warning.
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
    mode: ImputeDoor,
    ctx: &TransformContext,
) -> Result<RecordBatch, ImputeRefusal> {
    let door = mode.name();
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
    let layout = Layout::read(batch, columns, series, value_index, options, mode)?;
    let imputed = layout.imputed();
    if let Some(key) = &layout.key {
        key.check_companions_agree(&imputed, batch, door)?;
        key.check_companions_held(&imputed)?;
    }

    let value_col = batch.column(value_index);
    // A row in no series is left as written on the encoding door.
    let left_as_written: Vec<bool> = series
        .of_row
        .iter()
        .map(|series_of_row| series_of_row.is_none() && mode.leaves_unseriesed())
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
    warn_rows_in_no_series(series, n_rows, method, layout.key.is_some(), mode, ctx);

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
/// encoding door (`ImputeDoor::leaves_unseriesed`) they are left as written, and the caller
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
    mode: ImputeDoor,
    ctx: &TransformContext,
) {
    let in_none = series.of_row.iter().filter(|s| s.is_none()).count();
    if in_none == 0 || in_none == n_rows {
        return;
    }
    // The encoding door names these rows itself, once across its facet partitions.
    if mode.leaves_unseriesed() {
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

/// The Python class builds the numeric-value form of the spec (`value` a finite
/// float); the other options travel in the transform dict `transform_impute`
/// writes. Its `repr` names every option of the spec it holds, whichever door
/// built it.
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
            TransformSpec::Impute(s) => {
                // The constant as the wire spells it: `Some(2.75)`, `Some("x")`, `Some(true)`.
                let value = s.value.as_ref().map(|v| format!("Some({v})"));
                format!(
                    "Impute(field='{}', method='{}', value={}, groupby={:?}, key={:?}, keyvals={:?}, \
                     frame={:?}, name={:?})",
                    s.field,
                    s.method,
                    value.as_deref().unwrap_or("None"),
                    s.groupby,
                    s.key,
                    s.keyvals,
                    s.frame,
                    s.name,
                )
            }
            _ => "Impute(?)".to_string(),
        }
    }
}

#[cfg(test)]
mod tests;
