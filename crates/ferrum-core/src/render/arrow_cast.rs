//! Centralized Arrow numeric/string column casting helpers.
//!
//! Renderer code reaches for Arrow columns at three different layers
//! (`scale_resolve` for domain inference, `draw` for per-row coordinate
//! extraction, and per-mark renderers for value lookups). Each layer
//! previously hand-rolled its own dtype dispatch chain over Float/Int/UInt
//! variants — five copies of the same downcast ladder across `draw.rs`
//! (`col_as_f64`, `col_as_str`) and `scale_resolve.rs` (`column_min_max_f64`,
//! `numeric_extent`, `distinct_values_in_order`).
//!
//! This module consolidates that surface so adding a new Arrow primitive
//! is a one-line edit, dtype support is uniform across the renderer, and
//! the F16 widening (color type inference) lands by changing one call site
//! instead of five.

use arrow::array::{
    Array, ArrayData, BooleanArray,
    Float32Array, Float64Array,
    Int16Array, Int32Array, Int64Array, Int8Array,
    LargeStringArray, StringArray,
    TimestampMillisecondArray, TimestampMicrosecondArray,
    TimestampNanosecondArray, TimestampSecondArray,
    UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow::datatypes::{DataType, TimeUnit};
use arrow::record_batch::RecordBatch;

use super::category_read::{timestamp_read, TimestampRead};
use super::RenderError;
use crate::transform::context::format_instant;

/// The label a null category is named by where something must spell it: the
/// census (`c=null`) and the temporal stack's bin for a null instant.
///
/// A null row belongs to no category, so no band, domain entry or per-row
/// category carries this label; a literal `"null"` string is a real category
/// of its own and a true null never joins it.
pub(crate) const NULL_CATEGORY: &str = "null";

/// Authoritative predicate: true iff `dtype` is in the set that
/// `col_as_f64` can read and `min_max_f64` can range over.
///
/// This is the single source of truth for "can this dtype be treated as a
/// continuous/quantitative value?".  `is_numeric`, `col_as_f64`, and
/// `min_max_f64` all branch from this predicate so they can never claim
/// different coverage (RSUP-02 / D-DTYPE-1).
///
/// Covered set: Float64/32, Int64/32/16/8, UInt64/32/16/8,
/// Timestamp(any unit).
///
/// Note: `Date32`/`Date64`/`Duration` are deliberately NOT covered. Every batch
/// reaching the render layer has already passed through
/// `_coerce.normalize_for_rust`, which pre-casts those dtypes to `Timestamp(ms)`
/// before crossing the CDI boundary, so they can never arrive here. Keeping them
/// out of the predicate means the predicate, `col_as_f64`, and `min_max_f64` all
/// agree that they are unsupported (RSUP-02).
pub(crate) fn supported_numeric_dtype(dtype: &DataType) -> bool {
    matches!(
        dtype,
        DataType::Float64 | DataType::Float32
            | DataType::Int64 | DataType::Int32 | DataType::Int16 | DataType::Int8
            | DataType::UInt64 | DataType::UInt32 | DataType::UInt16 | DataType::UInt8
            | DataType::Timestamp(_, _)
    )
}

/// True for Arrow dtypes that should route as continuous/quantitative when
/// inferring an encoding type from data alone.
///
/// Used by `scale_resolve::build_color_scale` to validate that a column
/// marked quantitative or temporal is actually a numeric dtype, replacing
/// the pre-F16 narrow `Float64|UInt64` check.
///
/// Delegates to [`supported_numeric_dtype`] so the coverage is always in
/// sync with `col_as_f64` and `min_max_f64` (D-DTYPE-1).
pub(crate) fn is_numeric(dtype: &DataType) -> bool {
    supported_numeric_dtype(dtype)
}

/// A present column joining a continuous shared positional axis must be
/// numeric or temporal (`#120`(9), N1; spec §6.8) — the one predicate and
/// [`RenderError::LayerAxisColumnUnplaceable`] text every site that
/// reconciles a layer/leaf column against a continuous shared axis reads,
/// so the rule and its message exist in exactly one place rather than a
/// copy per site. `layer` is the offending member's index (in `spec.layers`
/// for a single leaf's own internal layer union, or in the leaf span for a
/// composite Overlay group's cross-leaf union); `channel` the RESOLVED
/// axis token; `field`/`dtype` the offending column and its pre-formatted
/// Arrow dtype.
pub(crate) fn require_placeable_on_continuous_axis(
    layer: usize,
    channel: &str,
    field: &str,
    dtype: &DataType,
) -> Result<(), RenderError> {
    if supported_numeric_dtype(dtype) {
        Ok(())
    } else {
        Err(RenderError::LayerAxisColumnUnplaceable {
            layer,
            detail: Box::new(super::LayerAxisColumnDetail {
                channel: channel.to_string(),
                field: field.to_string(),
                dtype: format!("{dtype:?}"),
            }),
        })
    }
}

/// The kind of panel axis an exchanged layer's column lands on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::render) enum PanelAxisKind {
    /// A band axis; `of_dates` when its categories are instants.
    Categorical { of_dates: bool },
    Continuous,
}

impl PanelAxisKind {
    /// The axis kind as refusal messages spell it.
    pub(in crate::render) fn as_str(self) -> &'static str {
        match self {
            PanelAxisKind::Categorical { .. } => "categorical",
            PanelAxisKind::Continuous => "continuous",
        }
    }
}

/// Refuse `field` (of `dtype`), which a layer of mark `mark` whose own `coord`
/// exchanged onto the panel's `channel` axis, when an `axis` of that kind
/// cannot draw it. The one servability question behind
/// [`super::LayerCoordUnplaceableReason::ExchangedChannelUnservable`], asked
/// by every route that layers a member whose flip state differs from the
/// panel's: a categorical axis serves what [`ensure_category_keyable`] admits
/// under the owner's reading of `binding` ([`timestamp_read`]: an undeclared
/// date is a time axis, which a band axis cannot draw, unless the axis itself
/// is of dates), a continuous one what [`supported_numeric_dtype`] admits.
///
/// Carries no layer index: the caller wraps the `Err` in
/// [`super::RenderError::in_layer_context`] (mirroring `scene_build::
/// in_layer_voice`), the one place that index is known and voiced.
#[allow(clippy::too_many_arguments)]
pub(in crate::render) fn require_exchanged_column_servable(
    mark: crate::spec::mark::Mark,
    coord: &crate::spec::coord::CoordKind,
    channel: &'static str,
    field: &str,
    dtype: &DataType,
    binding: &crate::spec::encoding::EncodingSpec,
    axis: PanelAxisKind,
) -> Result<(), RenderError> {
    let servable = match axis {
        PanelAxisKind::Categorical { of_dates } => {
            let read = timestamp_read(channel, Some(binding), false, of_dates);
            ensure_category_keyable(field, dtype, read).is_ok()
        }
        PanelAxisKind::Continuous => supported_numeric_dtype(dtype),
    };
    if servable {
        return Ok(());
    }
    Err(RenderError::LayerCoordUnplaceable {
        coord: crate::spec::coord::public_spelling(coord),
        mark,
        reason: Box::new(super::LayerCoordUnplaceableReason::ExchangedChannelUnservable {
            channel,
            column: field.to_owned(),
            dtype: format!("{dtype:?}"),
            axis_kind: axis.as_str(),
        }),
    })
}

/// The array-level core [`col_as_f64`] and [`LeftOutTally::count_adjusted`]
/// both read through: any supported numeric Arrow column as
/// `Vec<Option<f64>>`, preserving null positions. `Err(dtype string)` for an
/// unsupported dtype — the caller attaches its own field-naming context.
///
/// Supported dtypes: Float64/32, Int64/32/16/8, UInt64/32/16/8,
/// Timestamp(any unit). `Date32`/`Date64`/`Duration` are pre-cast to
/// `Timestamp(ms)` by `_coerce.normalize_for_rust` before reaching the render
/// layer, so they hit the unsupported-dtype error arm here (and never occur on
/// live paths); see [`supported_numeric_dtype`].
fn array_as_f64(col: &dyn Array) -> Result<Vec<Option<f64>>, String> {
    macro_rules! collect_as {
        ($t:ty) => {{
            let a = col.as_any().downcast_ref::<$t>().expect("dtype matched");
            return Ok(a.iter().map(|v| v.map(|x| x as f64)).collect());
        }};
    }
    match col.data_type() {
        DataType::Float64 => {
            let a = col.as_any().downcast_ref::<Float64Array>().expect("Float64");
            Ok(a.iter().collect())
        }
        DataType::Float32 => collect_as!(Float32Array),
        DataType::Int64 => collect_as!(Int64Array),
        DataType::Int32 => collect_as!(Int32Array),
        DataType::Int16 => collect_as!(Int16Array),
        DataType::Int8 => collect_as!(Int8Array),
        DataType::UInt64 => collect_as!(UInt64Array),
        DataType::UInt32 => collect_as!(UInt32Array),
        DataType::UInt16 => collect_as!(UInt16Array),
        DataType::UInt8 => collect_as!(UInt8Array),
        DataType::Timestamp(TimeUnit::Nanosecond, _) => collect_as!(TimestampNanosecondArray),
        DataType::Timestamp(TimeUnit::Microsecond, _) => collect_as!(TimestampMicrosecondArray),
        DataType::Timestamp(TimeUnit::Millisecond, _) => collect_as!(TimestampMillisecondArray),
        DataType::Timestamp(TimeUnit::Second, _) => collect_as!(TimestampSecondArray),
        other => Err(format!("{other:?}")),
    }
}

/// Read any supported numeric Arrow column as `Vec<Option<f64>>`, preserving
/// null positions. Returns `Err(ScaleResolutionFailed)` for unsupported
/// dtypes — matches the prior `draw::col_as_f64` semantics exactly.
pub(crate) fn col_as_f64(batch: &RecordBatch, field: &str) -> Result<Vec<Option<f64>>, RenderError> {
    let col = batch.column_by_name(field)
        .ok_or_else(|| RenderError::UnknownColumn { name: field.to_string() })?;
    array_as_f64(col.as_ref()).map_err(|dtype| RenderError::UnsupportedDtype {
        field: field.to_string(),
        dtype,
        context: None,
    })
}

/// Read a Utf8 column as `Vec<Option<String>>`. Returns
/// `Err(ScaleResolutionFailed)` for any other dtype.
///
/// Strings should already have been normalized to Utf8 by
/// `prepare::normalize_string_views` upstream; this function only handles
/// the post-normalization case, matching prior `draw::col_as_str`.
pub(crate) fn col_as_str(batch: &RecordBatch, field: &str) -> Result<Vec<Option<String>>, RenderError> {
    let col = batch.column_by_name(field)
        .ok_or_else(|| RenderError::UnknownColumn { name: field.to_string() })?;
    if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
        Ok(a.iter().map(|o| o.map(|s| s.to_string())).collect())
    } else {
        Err(RenderError::UnsupportedDtype {
            field: field.to_string(),
            dtype: format!("{:?} (expected Utf8)", col.data_type()),
            context: None,
        })
    }
}

/// Read any column that can serve as an ordinal category key as
/// `Vec<Option<String>>`.
///
/// This is the correct helper to use when a scale is `ScaleKind::Ordinal` and
/// you need per-row category strings for `to_pixel_str`. Unlike `col_as_str`,
/// it accepts non-string numeric dtypes by stringifying them — using the same
/// formatting that `distinct_values_in_order` applies when building the ordinal
/// domain, so the per-row strings always match the domain entries.
///
/// Supported dtypes and their formatting:
/// - Utf8 / LargeUtf8 → value as-is (identity, same as `col_as_str`)
/// - Int64/32/16/8, UInt64/32/16/8 → native `.to_string()` (e.g. `2000i64` → `"2000"`)
/// - Float64/32 → `format!("{v}")` when the value has a non-zero fractional
///   part (e.g. `1.5` → `"1.5"`), or integer-formatted otherwise (e.g.
///   `2000.0` → `"2000"`), so integer-valued float columns behave identically
///   to Int64 ordinal columns.
/// - Boolean → `"true"` / `"false"` (same as `distinct_values_in_order`).
/// - Timestamp (any unit, any zone) → [`timestamp_category_label`]: the
///   value's UTC ISO text, date-only at midnight.
/// - Null (an all-null column with no further dtype) → `None` for every
///   row; there is no value to stringify, but the row count is kept.
///
/// Returns `Err(UnsupportedDtype)` only for dtypes that cannot be sensibly
/// turned into category strings (durations and the like; `Date32`/`Date64`/
/// `Duration` reach the render layer already cast to `Timestamp(ms)`).
fn col_as_ordinal_category_str(
    batch: &RecordBatch,
    field: &str,
) -> Result<Vec<Option<String>>, RenderError> {
    let col = batch.column_by_name(field)
        .ok_or_else(|| RenderError::UnknownColumn { name: field.to_string() })?;

    // Macro for integer types: native value → .to_string()
    macro_rules! collect_int_as_str {
        ($t:ty) => {
            if let Some(a) = col.as_any().downcast_ref::<$t>() {
                return Ok(a.iter().map(|v| v.map(|x| x.to_string())).collect());
            }
        };
    }

    // Utf8 / LargeUtf8 — identity (most common, checked first).
    if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
        return Ok(a.iter().map(|o| o.map(|s| s.to_string())).collect());
    }
    if let Some(a) = col.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(a.iter().map(|o| o.map(|s| s.to_string())).collect());
    }

    // Integer types — .to_string() matches distinct_values_in_order exactly.
    collect_int_as_str!(Int64Array);
    collect_int_as_str!(Int32Array);
    collect_int_as_str!(Int16Array);
    collect_int_as_str!(Int8Array);
    collect_int_as_str!(UInt64Array);
    collect_int_as_str!(UInt32Array);
    collect_int_as_str!(UInt16Array);
    collect_int_as_str!(UInt8Array);

    // Float types — delegate to the shared helper so this function and
    // `distinct_values_in_order` always produce matching strings.
    if let Some(a) = col.as_any().downcast_ref::<Float64Array>() {
        return Ok(a.iter().map(|v| v.map(float_as_ordinal_str)).collect());
    }
    if let Some(a) = col.as_any().downcast_ref::<Float32Array>() {
        // Widen to f64 before formatting to reuse the same helper.
        return Ok(a.iter().map(|v| v.map(|x| float_as_ordinal_str(x as f64))).collect());
    }

    // Boolean — "true" / "false".
    if let Some(a) = col.as_any().downcast_ref::<arrow::array::BooleanArray>() {
        return Ok(a.iter().map(|v| v.map(|b| b.to_string())).collect());
    }

    // Timestamp — one label per instant, read from the raw epoch value.
    if let Some((instants, unit)) = timestamp_instants(col.as_ref()) {
        return Ok(instants.into_iter().map(|v| v.map(|raw| timestamp_category_label(raw, unit))).collect());
    }

    // Null-typed column (every value null, with no further dtype to read):
    // one `None` per row rather than a refusal — there is no value to
    // stringify, but the row count is still knowable.
    if col.data_type() == &DataType::Null {
        return Ok(vec![None; col.len()]);
    }

    Err(RenderError::UnsupportedDtype {
        field: field.to_string(),
        dtype: format!("{:?} (cannot convert to ordinal category string)", col.data_type()),
        context: None,
    })
}

/// A `Timestamp` column's raw epoch values and their unit, or `None` for any
/// other dtype. Every zone reads as the same instants: the label is UTC.
fn timestamp_instants(col: &dyn Array) -> Option<(Vec<Option<i64>>, TimeUnit)> {
    macro_rules! read {
        ($t:ty, $unit:expr) => {
            if let Some(a) = col.as_any().downcast_ref::<$t>() {
                return Some((a.iter().collect(), $unit));
            }
        };
    }
    read!(TimestampNanosecondArray, TimeUnit::Nanosecond);
    read!(TimestampMicrosecondArray, TimeUnit::Microsecond);
    read!(TimestampMillisecondArray, TimeUnit::Millisecond);
    read!(TimestampSecondArray, TimeUnit::Second);
    None
}

/// The category label of one timestamp value: its UTC ISO text, date-only
/// (`2020-01-01`) when the value is at midnight, otherwise the group-key form
/// [`format_instant`] writes (`2020-01-01T12:00:00`, with a `.250`-style
/// fraction only when the value has one).
///
/// The rule is per value, so one instant has one label on every layer and a
/// band and a point drawn from different layers' rows of one column meet.
/// Distinct instants of one unit have distinct labels.
pub(crate) fn timestamp_category_label(raw: i64, unit: TimeUnit) -> String {
    let text = format_instant(raw, unit);
    match text.strip_suffix("T00:00:00") {
        Some(date) => date.to_string(),
        None => text,
    }
}

/// Put `values`, each a category label paired with a row of `batch` that
/// holds it, into time order when `field` is a `Timestamp` column; any other
/// dtype keeps the order given (first appearance). The same default order
/// [`distinct_values_in_order`] gives a timestamp axis, for the facet panel
/// split.
pub(crate) fn order_labels_by_instant(batch: &RecordBatch, field: &str, values: &mut [(String, usize)]) {
    let Some(col) = batch.column_by_name(field) else { return };
    if let Some((instants, _)) = timestamp_instants(col.as_ref()) {
        values.sort_by_key(|(_, row)| instants.get(*row).copied().flatten());
    }
}

/// Read a `Timestamp` column (any unit) as `Vec<Option<String>>` via the raw
/// `i64` epoch value's `.to_string()` — an **identity** mapping, not a
/// display formatter.
///
/// This is deliberately NOT `col_as_f64`'s `x as f64` widening: `f64` has a
/// 53-bit mantissa, so any epoch value whose magnitude exceeds `2^53`
/// (nanosecond timestamps hit this for any date after 1970; millisecond
/// timestamps hit it for dates far in the future, but the collision risk is
/// real *today* at nanosecond/microsecond resolution) loses precision on the
/// `i64 -> f64` cast, and two distinct instants can round to the same `f64`.
/// Downstream display formatters (e.g. `format_numeric`) compound this by
/// additionally collapsing everything to ~4 significant digits. Reading the
/// native `i64` and stringifying it directly has no such ceiling — every
/// representable epoch value round-trips to a distinct string.
///
/// Its one caller is [`col_as_object_key_str`], which needs an injective
/// row -> string map for a temporal `key=` column that keeps its epoch text
/// across releases, not the category label `col_as_ordinal_category_str` reads.
///
/// Returns `Err(UnsupportedDtype)` for any dtype other than `Timestamp`.
pub(crate) fn col_as_temporal_epoch_str(
    batch: &RecordBatch,
    field: &str,
) -> Result<Vec<Option<String>>, RenderError> {
    let col = batch.column_by_name(field)
        .ok_or_else(|| RenderError::UnknownColumn { name: field.to_string() })?;
    match timestamp_instants(col.as_ref()) {
        Some((instants, _)) => Ok(instants.into_iter().map(|v| v.map(|x| x.to_string())).collect()),
        None => Err(RenderError::UnsupportedDtype {
            field: field.to_string(),
            dtype: format!("{:?} (expected Timestamp)", col.data_type()),
            context: None,
        }),
    }
}

/// Per-row **identity key** for any column that must partition rows by value
/// or place rows on a category scale: exactly [`col_as_ordinal_category_str`],
/// so "which category is this row" has one answer on every path (a band, a
/// colour or shape legend entry, a stack bin, a facet panel).
///
/// A stack over a *category* axis must bin exactly as the band ladder does,
/// which delegating byte-for-byte guarantees; a stack over a continuous axis
/// bins by exact value, and every timestamp label is injective per instant
/// (`apply_stack`, #184).
///
/// Nulls stay `None`, as in the underlying reader; a caller that needs them
/// binned (the stack) maps them itself.
pub(crate) fn col_as_identity_key_str(
    batch: &RecordBatch,
    field: &str,
) -> Result<Vec<Option<String>>, RenderError> {
    col_as_ordinal_category_str(batch, field)
}

/// The category door for per-row reads that decide category versus time: the
/// per-row category strings of `field` under `read`
/// ([`super::category_read::timestamp_read`] decides it). Under
/// [`TimestampRead::Temporal`] a `Timestamp` column is the refusal it drew
/// before dates could be categories (`UnsupportedDtype`, "cannot convert to
/// ordinal category string"), so a builder's own error handling reproduces what
/// it did then; under [`TimestampRead::Categories`] it reads its per-value
/// labels. Every other dtype is read the same under both.
pub(crate) fn category_strings(
    batch: &RecordBatch,
    field: &str,
    read: TimestampRead,
) -> Result<Vec<Option<String>>, RenderError> {
    if read == TimestampRead::Temporal {
        if let Some(col) = batch.column_by_name(field) {
            if matches!(col.data_type(), DataType::Timestamp(_, _)) {
                return Err(RenderError::UnsupportedDtype {
                    field: field.to_string(),
                    dtype: format!("{:?} (cannot convert to ordinal category string)", col.data_type()),
                    context: None,
                });
            }
        }
    }
    col_as_ordinal_category_str(batch, field)
}

/// The category door for domain reads: the distinct categories of `field`
/// under `read`, each with the instant it names (nanoseconds since the epoch)
/// when it is a date. Dates come in time order; every other dtype in first
/// appearance. Under [`TimestampRead::Temporal`] a `Timestamp` column is the
/// refusal it drew before dates could be categories ("cannot enumerate
/// distinct values"). Built into a [`super::category_read::CategoryDomain`].
pub(crate) fn category_domain_entries(
    batch: &RecordBatch,
    field: &str,
    read: TimestampRead,
) -> Result<Vec<(String, Option<i128>)>, RenderError> {
    let col = batch.column_by_name(field).ok_or_else(|| RenderError::UnknownColumn { name: field.to_string() })?;
    if let Some((instants, unit)) = timestamp_instants(col.as_ref()) {
        if read == TimestampRead::Temporal {
            return Err(unsupported_category_key(field, col.data_type()));
        }
        let nanos_per_unit: i128 = match unit {
            TimeUnit::Second => 1_000_000_000,
            TimeUnit::Millisecond => 1_000_000,
            TimeUnit::Microsecond => 1_000,
            TimeUnit::Nanosecond => 1,
        };
        let in_time_order: std::collections::BTreeSet<i64> = instants.into_iter().flatten().collect();
        return Ok(in_time_order
            .into_iter()
            .map(|raw| (timestamp_category_label(raw, unit), Some(i128::from(raw) * nanos_per_unit)))
            .collect());
    }
    Ok(distinct_values_in_order(batch, field)?.into_iter().map(|label| (label, None)).collect())
}

/// Per-row **object-constancy key** for a `key=` column (GH #93): the raw epoch
/// text for a `Timestamp` ([`col_as_temporal_epoch_str`]), else the category
/// reader's string. Injective for every dtype it covers, and unchanged for a
/// temporal key column when the same column is also read as a category.
pub(crate) fn col_as_object_key_str(
    batch: &RecordBatch,
    field: &str,
) -> Result<Vec<Option<String>>, RenderError> {
    col_as_temporal_epoch_str(batch, field).or_else(|_| col_as_ordinal_category_str(batch, field))
}

/// A numeric extent folded over only the values on one side of zero
/// ([`sided_min_max_f64`]), or over every value when no side is asked for
/// ([`min_max_f64`]). `left_out` counts the column's non-null, finite values
/// that fell on the other side (always `0` with no side); `left_out_zero`
/// and `left_out_opposite` characterize those left-out values for
/// `RenderError::LogScaleNoValueOnSide`'s remedy text (`left_out_opposite`
/// omits any exact zero, which `left_out_zero` already reports).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SidedExtent {
    pub(crate) lo: f64,
    pub(crate) hi: f64,
    pub(crate) left_out: usize,
    pub(crate) left_out_zero: bool,
    pub(crate) left_out_opposite: Option<(f64, f64)>,
}

/// A `Log` channel's whole-frame left-out count: one content-deduplicated
/// tally shared by every fold that
/// reads a `Log` channel's data, so `+`, `fm.layer`, a static `LayerChart`
/// overlay and a faceted chart (shared or independent) all report the SAME
/// count for the SAME underlying data, rather than each fold summing
/// `left_out` over whatever columns it happened to read (which double-
/// counted a faceted-shared panel's rows against the all-panels batch, and
/// summed a column two layers both draw).
///
/// [`count`](Self::count) is the one write path for a plain (unadjusted)
/// read: a column counts only when no column already counted compares equal
/// by content (`ArrayData`'s `PartialEq`, the same "equal values are one
/// frame" rule `ferrum._compose._same_frame` already applies in Python for
/// `+`), and only when it left rows out (`extent.left_out > 0`) — a column
/// with nothing left out contributes nothing to store, only to
/// `any_on_side`. [`count_adjusted`](Self::count_adjusted) is the write path
/// for a position-adjusted layer's own caller rows plus the rows the
/// adjustment moves off the side; it shares the same content-dedupe key, so
/// a caller column `count` already stored and one `count_adjusted` later
/// reads (a demanded fit's own recorded lineage column beside a jittered
/// sibling reading the SAME raw column) fold into ONE entry. Each entry's
/// third element is the moved-off ADJUSTED value at each row `count_adjusted`
/// found moved, `None` elsewhere (and every position for a plain `count`
/// entry, which moves nothing) — doubling as the "is this row moved" mask
/// `rows()` counts and the value `zero()`/`opposite()` characterize.
/// [`merge`](Self::merge) folds another tally in with the same dedupe, for
/// composite groups whose members each build their own tally before the
/// group combines them.
#[derive(Debug, Clone, Default)]
pub(crate) struct LeftOutTally {
    columns: Vec<(ArrayData, SidedExtent, Vec<Option<f64>>)>,
    any_on_side: bool,
}

impl LeftOutTally {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Record `col`'s already-folded `extent`. A no-op for a column equal by
    /// content to one already counted (two reads of the same data, or the
    /// same column read through two layers). `any_on_side` is updated
    /// whenever `extent` holds a value, even for a column with nothing left
    /// out, so a channel whose every counted column is fully off-side can
    /// still be told apart from one with no value anywhere at all.
    pub(crate) fn count(&mut self, col: &dyn Array, extent: SidedExtent) {
        if extent.lo.is_finite() && extent.hi.is_finite() && extent.lo <= extent.hi {
            self.any_on_side = true;
        }
        if extent.left_out == 0 {
            return;
        }
        let data = col.to_data();
        if self.columns.iter().any(|(d, _, _)| d == &data) {
            return;
        }
        self.columns.push((data, extent, vec![None; col.len()]));
    }

    /// Record a position-adjusted layer's off-side rows: `caller`'s own rows
    /// (already-folded `caller_extent`, stored the same content-deduplicated
    /// way [`count`](Self::count) stores a plain read) PLUS every row whose
    /// `caller` value is on-side but whose `adjusted` value the position
    /// adjustment itself pushed off the side — a caller row is never also
    /// counted as moved (an off-side caller row stays `None` here; `count`'s
    /// own tally already has it), so a row two different layers or two
    /// different adjustments both move off still counts once, by the SAME
    /// `None`-vs-`Some` OR this performs against a caller column already
    /// stored (by `count` or an earlier `count_adjusted`). `any_on_side`
    /// follows the ADJUSTED values, not the caller's: a channel whose every
    /// drawn (adjusted) row is off-side still refuses as a frame with no
    /// value anywhere does, even when the caller's own rows are on-side.
    /// `Err` names the row-count mismatch when `caller` and `adjusted` are
    /// not the same length — [`AdjustedRows`](crate::render::position::AdjustedRows)'s
    /// own invariant keeps this from happening on a live path; the caller
    /// maps it to a typed refusal rather than silently truncating or
    /// panicking.
    pub(crate) fn count_adjusted(
        &mut self,
        caller: &dyn Array,
        caller_extent: SidedExtent,
        adjusted: &dyn Array,
        side: crate::scale::log::LogSide,
    ) -> Result<(), String> {
        if caller.len() != adjusted.len() {
            return Err(format!(
                "the adjusted column has {} row(s), the caller column {} — they must be row-aligned",
                adjusted.len(),
                caller.len()
            ));
        }
        let caller_values = array_as_f64(caller)?;
        let adjusted_values = array_as_f64(adjusted)?;
        let on_side = |v: f64| match side {
            crate::scale::log::LogSide::Positive => v > 0.0,
            crate::scale::log::LogSide::Negative => v < 0.0,
        };
        if adjusted_values.iter().flatten().any(|&v| v.is_finite() && on_side(v)) {
            self.any_on_side = true;
        }
        let moved: Vec<Option<f64>> = caller_values
            .iter()
            .zip(&adjusted_values)
            .map(|(c, a)| {
                let c = (*c)?;
                if !c.is_finite() || !on_side(c) {
                    return None;
                }
                let a = (*a)?;
                (a.is_finite() && !on_side(a)).then_some(a)
            })
            .collect();
        let data = caller.to_data();
        if let Some(idx) = self.columns.iter().position(|(d, _, _)| d == &data) {
            let (_, _, existing) = &mut self.columns[idx];
            for (slot, new) in existing.iter_mut().zip(&moved) {
                if slot.is_none() {
                    *slot = *new;
                }
            }
            return Ok(());
        }
        if caller_extent.left_out == 0 && !moved.iter().any(Option::is_some) {
            return Ok(());
        }
        self.columns.push((data, caller_extent, moved));
        Ok(())
    }

    /// Fold `other`'s counted columns into `self`, with the same
    /// content-equality dedupe [`count`](Self::count) applies — a composite
    /// group merging its members' tallies. A caller column both tallies
    /// already hold ORs their moved-row masks together (the SAME union
    /// [`count_adjusted`](Self::count_adjusted) performs against its own
    /// prior entry), rather than keeping two separate entries for one
    /// column.
    pub(crate) fn merge(&mut self, other: LeftOutTally) {
        self.any_on_side |= other.any_on_side;
        for (data, extent, moved) in other.columns {
            if let Some(idx) = self.columns.iter().position(|(d, _, _)| d == &data) {
                let (_, _, existing) = &mut self.columns[idx];
                for (slot, new) in existing.iter_mut().zip(&moved) {
                    if slot.is_none() {
                        *slot = *new;
                    }
                }
                continue;
            }
            self.columns.push((data, extent, moved));
        }
    }

    /// The whole-frame left-out row count: each distinct counted column's
    /// own `left_out` plus the count of its own moved-off rows (`Some`
    /// entries in its mask).
    pub(crate) fn rows(&self) -> usize {
        self.columns.iter().map(|(_, e, moved)| e.left_out + moved.iter().filter(|m| m.is_some()).count()).sum()
    }

    /// Whether any counted column's left-out values, or any moved-off
    /// adjusted value, is an exact zero.
    pub(crate) fn zero(&self) -> bool {
        self.columns.iter().any(|(_, e, moved)| e.left_out_zero || moved.iter().flatten().any(|v| *v == 0.0))
    }

    /// The combined `(min, max)` of every counted column's strictly
    /// opposite-signed left-out values, folded with every moved-off row's
    /// own (strictly opposite-signed) adjusted value, when any exist.
    pub(crate) fn opposite(&self) -> Option<(f64, f64)> {
        self.columns
            .iter()
            .flat_map(|(_, e, moved)| {
                e.left_out_opposite.into_iter().chain(moved.iter().flatten().filter(|v| **v != 0.0).map(|v| (*v, *v)))
            })
            .fold(None, |acc, (lo, hi)| {
                Some(match acc {
                    Some((a_lo, a_hi)) => (a_lo.min(lo), a_hi.max(hi)),
                    None => (lo, hi),
                })
            })
    }

    /// Whether any read this tally has seen held an on-side value, counted
    /// or not — the whole frame's own answer to "is there anything to map
    /// anywhere", independent of any one fold's own (possibly empty) extent.
    pub(crate) fn any_on_side(&self) -> bool {
        self.any_on_side
    }
}

/// `(min, max)` over a numeric Arrow column, including non-finite values.
/// Returns `Err(message)` for unsupported dtypes — matches the prior
/// `scale_resolve::column_min_max_f64` semantics exactly. Caller is
/// responsible for wrapping the `Err` into a typed `RenderError`.
///
/// **Caveat**: this function does not filter NaN. For Float columns the
/// `f64::min`/`max` IEEE-754 semantics treat NaN as "absent" (non-NaN wins),
/// which is the prior behavior. An all-NaN column returns `(+INF, -INF)` —
/// caller should validate.
///
/// The `None`-side call of [`sided_min_max_f64`], which owns the dtype
/// match so a `Log` channel's sided fold and every other channel's full
/// fold read one body instead of two that could drift. Every row counts
/// toward `(lo, hi)` under `None` (`left_out` is always `0`).
pub(crate) fn min_max_f64(col: &dyn Array) -> Result<(f64, f64), String> {
    sided_min_max_f64(col, None).map(|e| (e.lo, e.hi))
}

/// [`min_max_f64`] folded over only the values on `side` of zero (every
/// value when `side` is `None`). A value not on `side` (and, for `None`,
/// never) counts toward `left_out` instead of `(lo, hi)` — a `Log` scale's
/// extent reads only the side it maps, and the rows it leaves out are what
/// `RenderWarning::LogScaleRowsLeftOut`/`RenderError::LogScaleNoValueOnSide`
/// report. Returns `Err(message)` for unsupported dtypes, matching
/// `min_max_f64` exactly.
pub(crate) fn sided_min_max_f64(
    col: &dyn Array,
    side: Option<crate::scale::log::LogSide>,
) -> Result<SidedExtent, String> {
    use crate::scale::log::LogSide;
    let on_side = |v: f64| match side {
        None => true,
        Some(LogSide::Positive) => v > 0.0,
        Some(LogSide::Negative) => v < 0.0,
    };
    // A left-out value's own fold, for `RenderError::LogScaleNoValueOnSide`'s
    // remedy: `zero` is whether any left-out value is exactly `0.0`;
    // `opposite` is the `(min, max)` of the left-out values on the strictly
    // OPPOSITE sign from `side` (an exact zero counts toward `zero`, not
    // `opposite`). A no-op fold under `side: None`, since nothing is ever
    // left out there.
    struct LeftOut {
        zero: bool,
        opposite: Option<(f64, f64)>,
    }
    // A non-finite left-out value (an off-side infinity) contributes to
    // neither `zero` nor `opposite`: both only
    // characterize a FINITE left-out value, for the remedy text's mirrored-
    // domain suggestion, which means nothing for an infinity.
    let left_out_fold = |v: f64, left_out: &mut LeftOut| {
        if v == 0.0 {
            left_out.zero = true;
        } else if v.is_finite() {
            let (lo, hi) = left_out.opposite.unwrap_or((f64::INFINITY, f64::NEG_INFINITY));
            left_out.opposite = Some((lo.min(v), hi.max(v)));
        }
    };
    macro_rules! sided_min_max_int {
        ($t:ty, $native:ty) => {{
            let a = col.as_any().downcast_ref::<$t>().expect("dtype matched");
            let mut lo = <$native>::MAX;
            let mut hi = <$native>::MIN;
            let mut left_out = 0usize;
            let mut left_out_info = LeftOut { zero: false, opposite: None };
            for v in a.iter().flatten() {
                if on_side(v as f64) {
                    lo = <$native>::min(lo, v);
                    hi = <$native>::max(hi, v);
                } else {
                    left_out += 1;
                    left_out_fold(v as f64, &mut left_out_info);
                }
            }
            return Ok(SidedExtent {
                lo: lo as f64,
                hi: hi as f64,
                left_out,
                left_out_zero: left_out_info.zero,
                left_out_opposite: left_out_info.opposite,
            });
        }};
    }
    match col.data_type() {
        DataType::Float64 => {
            let a = col.as_any().downcast_ref::<Float64Array>().expect("Float64");
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            let mut left_out = 0usize;
            let mut left_out_info = LeftOut { zero: false, opposite: None };
            // Only NaN is filtered: an infinity reaches `on_side`, so an
            // off-side infinity counts toward `left_out`
            // instead of silently passing the finiteness filter the on-
            // side test ran behind. It never folds into `lo`/`hi` itself —
            // only a FINITE on-side value narrows the extent; an on-side
            // infinity clamps to the range end downstream
            // (`LogScaleData::scale`), like `LinearScale`.
            for v in a.iter().flatten().filter(|v| !v.is_nan()) {
                if on_side(v) {
                    if v.is_finite() {
                        lo = lo.min(v);
                        hi = hi.max(v);
                    }
                } else {
                    left_out += 1;
                    left_out_fold(v, &mut left_out_info);
                }
            }
            Ok(SidedExtent { lo, hi, left_out, left_out_zero: left_out_info.zero, left_out_opposite: left_out_info.opposite })
        }
        DataType::Float32 => {
            let a = col.as_any().downcast_ref::<Float32Array>().expect("Float32");
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            let mut left_out = 0usize;
            let mut left_out_info = LeftOut { zero: false, opposite: None };
            for v in a.iter().flatten().filter(|v| !v.is_nan()) {
                if on_side(v as f64) {
                    if v.is_finite() {
                        lo = lo.min(v);
                        hi = hi.max(v);
                    }
                } else {
                    left_out += 1;
                    left_out_fold(v as f64, &mut left_out_info);
                }
            }
            Ok(SidedExtent {
                lo: lo as f64,
                hi: hi as f64,
                left_out,
                left_out_zero: left_out_info.zero,
                left_out_opposite: left_out_info.opposite,
            })
        }
        DataType::Int64 => sided_min_max_int!(Int64Array, i64),
        DataType::Int32 => sided_min_max_int!(Int32Array, i32),
        DataType::Int16 => sided_min_max_int!(Int16Array, i16),
        DataType::Int8 => sided_min_max_int!(Int8Array, i8),
        DataType::UInt64 => sided_min_max_int!(UInt64Array, u64),
        DataType::UInt32 => sided_min_max_int!(UInt32Array, u32),
        DataType::UInt16 => sided_min_max_int!(UInt16Array, u16),
        DataType::UInt8 => sided_min_max_int!(UInt8Array, u8),
        DataType::Timestamp(TimeUnit::Nanosecond, _) => sided_min_max_int!(TimestampNanosecondArray, i64),
        DataType::Timestamp(TimeUnit::Microsecond, _) => sided_min_max_int!(TimestampMicrosecondArray, i64),
        DataType::Timestamp(TimeUnit::Millisecond, _) => sided_min_max_int!(TimestampMillisecondArray, i64),
        DataType::Timestamp(TimeUnit::Second, _) => sided_min_max_int!(TimestampSecondArray, i64),
        // `Date32`/`Date64`/`Duration` fall through to the error arm: every batch
        // reaching the render layer has been pre-cast to `Timestamp(ms)` by
        // `_coerce.normalize_for_rust`, so they never occur here on live paths and
        // are treated as unsupported, matching `col_as_f64` and
        // `supported_numeric_dtype` (RSUP-02).
        other => Err(format!("unsupported column dtype: {other:?}")),
    }
}

/// `(min, max)` over the *finite* values of a numeric column. Returns
/// `None` when no finite values are present or when the dtype is
/// unsupported — matches the prior `scale_resolve::numeric_extent`
/// semantics, but supports the full numeric dtype set (callers that
/// previously routed Int32/Float32/etc. to a categorical path will
/// continue to do so until F16 widens the routing predicate; this
/// function is unreachable for those dtypes today).
pub(crate) fn finite_min_max_f64(col: &dyn Array) -> Option<(f64, f64)> {
    let (lo, hi) = min_max_f64(col).ok()?;
    if lo.is_finite() && hi.is_finite() && lo <= hi {
        Some((lo, hi))
    } else {
        None
    }
}

/// Format a float value as an ordinal category string.
///
/// Whole-number floats (e.g. `2000.0`) are formatted as integers (`"2000"`)
/// so that an integer-valued Float64 ordinal column produces the same domain
/// strings as an equivalent Int64 column. Fractional values (e.g. `1.5`)
/// keep their decimal representation (`"1.5"`). Non-finite values fall back
/// to `format!("{v}")` (i.e. `"inf"`, `"-inf"`, `"NaN"`).
///
/// This helper is the single source of truth for float-ordinal stringification
/// and is shared between `distinct_values_in_order` (domain building) and
/// `col_as_ordinal_category_str` (per-row lookup), so the two can never drift.
#[inline]
pub(crate) fn float_as_ordinal_str(v: f64) -> String {
    if v.fract() == 0.0 && v.is_finite() {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

/// The refusal both category-key surfaces return for a column that cannot be
/// enumerated or keyed as ordinal categories.
///
/// One constructor so [`distinct_values_in_order`] and
/// [`ensure_category_keyable`] refuse in the same words: a caller that gates on
/// the dtype up front is standing in for the domain build it skipped, and a
/// user should not be able to tell which one spoke.
fn unsupported_category_key(field: &str, dtype: &DataType) -> RenderError {
    RenderError::UnsupportedDtype {
        field: field.to_string(),
        dtype: format!("{dtype:?} (cannot enumerate distinct values)"),
        context: None,
    }
}

/// Refuse — without reading the column — any dtype that cannot serve as an
/// ordinal category key.
///
/// [`distinct_values_in_order`] (domain build) and
/// [`col_as_ordinal_category_str`] (per-row key) accept exactly this dtype set,
/// so a categorical scale over a column outside it is unserviceable end to end.
/// This exists for the resolver branches that build a categorical *color*,
/// `shape` or `stroke_dash` domain without reading the column — an explicit
/// `scale.domain` supersedes the data-derived domain, so the per-row reader would be the first thing to
/// touch the column, and only on the marks whose builders propagate its error
/// (rule/segment refuse; point/bar paint every element one color under a legend
/// enumerating the declared range). Gating on the dtype there moves the refusal
/// back to scale resolution, where it is uniform across marks, at O(1) instead
/// of re-deriving a domain that is about to be discarded.
///
/// `category_readers_accept_exactly_the_dtypes_ensure_category_keyable_does`
/// pins this list against both readers, so widening one without the others
/// fails there rather than reopening the divergence.
pub(crate) fn ensure_category_keyable(field: &str, dtype: &DataType, read: TimestampRead) -> Result<(), RenderError> {
    let keyable = matches!(
        dtype,
        DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
            | DataType::Boolean
            // A column with no dtype of its own has no value to key: it
            // enumerates no category and keys every row to none.
            | DataType::Null
    );
    // One label per instant, in UTC (`timestamp_category_label`), where the
    // binding reads dates as categories.
    let keyable = keyable || (read == TimestampRead::Categories && matches!(dtype, DataType::Timestamp(_, _)));
    if keyable {
        Ok(())
    } else {
        Err(unsupported_category_key(field, dtype))
    }
}

/// Enumerate distinct values in encounter order, stringified, from a column
/// whose dtype is one of `Utf8`, `LargeUtf8`, `Int*`, `UInt*`, `Float32`,
/// `Float64`, `Boolean`, `Timestamp`, or `Null` (which has no value to enumerate) — i.e. exactly the set
/// [`ensure_category_keyable`] admits. Returns `Err(UnsupportedDtype)` for any
/// other dtype.
///
/// Stringification rules (must agree exactly with `col_as_ordinal_category_str`):
/// - Utf8 / LargeUtf8 → identity
/// - Int*/UInt* → `.to_string()`
/// - Float64/32 → `float_as_ordinal_str` (integer-formatted for whole values)
/// - Boolean → `"true"` / `"false"`
/// - Timestamp → [`timestamp_category_label`], ordered by time rather than by
///   encounter (every other dtype keeps encounter order)
fn distinct_values_in_order(batch: &RecordBatch, field: &str) -> Result<Vec<String>, RenderError> {
    let col = batch.column_by_name(field)
        .ok_or_else(|| RenderError::UnknownColumn { name: field.to_string() })?;
    let mut seen = std::collections::HashSet::<String>::new();
    let mut out = Vec::<String>::new();
    let mut push = |s: String| {
        if seen.insert(s.clone()) {
            out.push(s);
        }
    };
    if let Some((instants, unit)) = timestamp_instants(col.as_ref()) {
        let in_time_order: std::collections::BTreeSet<i64> = instants.into_iter().flatten().collect();
        for raw in in_time_order { push(timestamp_category_label(raw, unit)); }
        return Ok(out);
    }
    // Macro to avoid repeating the downcast-and-stringify loop for each integer type.
    macro_rules! push_int {
        ($t:ty) => {
            if let Some(a) = col.as_any().downcast_ref::<$t>() {
                for v in a.iter().flatten() { push(v.to_string()); }
                return Ok(out);
            }
        };
    }
    if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
        for v in a.iter().flatten() { push(v.to_string()); }
    } else if let Some(a) = col.as_any().downcast_ref::<LargeStringArray>() {
        // Polars produces LargeUtf8 (LargeStringArray) for string columns.
        for v in a.iter().flatten() { push(v.to_string()); }
    } else if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
        for v in a.iter().flatten() { push(v.to_string()); }
    } else if let Some(a) = col.as_any().downcast_ref::<BooleanArray>() {
        for v in a.iter().flatten() { push(v.to_string()); }
    } else if let Some(a) = col.as_any().downcast_ref::<Float64Array>() {
        // Float64 ordinal columns: integer-valued floats format as integers
        // (e.g. 2000.0 → "2000") to match Int64 domain strings. Fractional
        // values keep their decimal form (e.g. 1.5 → "1.5").
        for v in a.iter().flatten() { push(float_as_ordinal_str(v)); }
    } else if let Some(a) = col.as_any().downcast_ref::<Float32Array>() {
        // Same rule for Float32; widen to f64 first to reuse the same helper.
        for v in a.iter().flatten() { push(float_as_ordinal_str(v as f64)); }
    } else if col.data_type() == &DataType::Null {
        // Every value is null, so there is no category to enumerate.
    } else {
        // Non-Int64 integer variants — pandas/pyarrow users commonly have Int32, UInt8, etc.
        // Each arm stringifies the native value so downstream encoding treats them as nominal.
        push_int!(Int32Array);
        push_int!(Int16Array);
        push_int!(Int8Array);
        push_int!(UInt64Array);
        push_int!(UInt32Array);
        push_int!(UInt16Array);
        push_int!(UInt8Array);
        return Err(unsupported_category_key(field, col.data_type()));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        Date32Array, Date64Array,
        DurationMillisecondArray, DurationMicrosecondArray,
        DurationNanosecondArray, DurationSecondArray,
        Float64Array, Int32Array, Int8Array, StringArray, UInt8Array, UInt32Array,
        TimestampMillisecondArray,
    };
    // Date/Duration arrays are still constructed in the RSUP-02 test below — but
    // only to assert that the predicate + value functions treat them as
    // *unsupported* (they are pre-cast to Timestamp(ms) by normalize_for_rust
    // before reaching the render layer, so they never arrive here on live paths).
    use arrow::datatypes::{Field, Schema, TimeUnit};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    fn batch_f64(name: &str, values: Vec<Option<f64>>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::Float64, true)]));
        RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(values))]).unwrap()
    }

    /// `sided_min_max_f64` only skips the NaN filter for an infinite value;
    /// it counts toward `left_out` when off-side (never folded into `lo`/`hi`,
    /// which stay the finite on-side extent), and a non-finite left-out value
    /// contributes to neither `left_out_zero` nor `left_out_opposite` (those
    /// only characterize a FINITE left-out value, for the remedy text).
    #[test]
    fn sided_min_max_counts_an_off_side_infinity_but_never_folds_it() {
        use crate::scale::log::LogSide;
        let a = Float64Array::from(vec![f64::NEG_INFINITY, -5.0, 0.0, 2.0, f64::INFINITY]);
        let extent = sided_min_max_f64(&a, Some(LogSide::Positive)).unwrap();
        assert_eq!(extent.lo, 2.0);
        assert_eq!(extent.hi, 2.0);
        assert_eq!(extent.left_out, 3);
        assert!(extent.left_out_zero);
        assert_eq!(extent.left_out_opposite, Some((-5.0, -5.0)));
    }

    #[test]
    fn col_as_f64_handles_float64_with_nulls() {
        let b = batch_f64("x", vec![Some(1.0), None, Some(3.0)]);
        let out = col_as_f64(&b, "x").unwrap();
        assert_eq!(out, vec![Some(1.0), None, Some(3.0)]);
    }

    #[test]
    fn col_as_f64_widens_int32() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![Some(1), Some(2), None]))],
        ).unwrap();
        let out = col_as_f64(&b, "x").unwrap();
        assert_eq!(out, vec![Some(1.0), Some(2.0), None]);
    }

    #[test]
    fn col_as_f64_errors_on_string() {
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec!["a", "b"]))],
        ).unwrap();
        let err = col_as_f64(&b, "s").unwrap_err();
        assert!(format!("{err}").contains("unsupported dtype"), "{err}");
    }

    #[test]
    fn col_as_str_reads_utf8() {
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec![Some("a"), None, Some("b")]))],
        ).unwrap();
        let out = col_as_str(&b, "s").unwrap();
        assert_eq!(out, vec![Some("a".into()), None, Some("b".into())]);
    }

    #[test]
    fn min_max_f64_float64() {
        let b = batch_f64("x", vec![Some(3.0), Some(1.0), Some(2.0)]);
        let col = b.column(0).as_ref();
        let (lo, hi) = min_max_f64(col).unwrap();
        assert_eq!((lo, hi), (1.0, 3.0));
    }

    #[test]
    fn min_max_f64_int32() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![Some(5), Some(-3), Some(8)]))],
        ).unwrap();
        let (lo, hi) = min_max_f64(b.column(0).as_ref()).unwrap();
        assert_eq!((lo, hi), (-3.0, 8.0));
    }

    #[test]
    fn finite_min_max_f64_filters_nan() {
        let b = batch_f64("x", vec![Some(1.0), Some(f64::NAN), Some(3.0)]);
        let result = finite_min_max_f64(b.column(0).as_ref());
        // Per IEEE-754 f64::min/max, NaN is absorbed; non-NaN values dominate.
        assert_eq!(result, Some((1.0, 3.0)));
    }

    #[test]
    fn finite_min_max_f64_all_nan_returns_none() {
        let b = batch_f64("x", vec![Some(f64::NAN), Some(f64::NAN)]);
        let result = finite_min_max_f64(b.column(0).as_ref());
        // All-NaN: min_max returns (+INF, -INF); finite_min_max should None.
        assert_eq!(result, None);
    }

    #[test]
    fn distinct_values_in_order_strings() {
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec!["b", "a", "b", "c", "a"]))],
        ).unwrap();
        let out = distinct_values_in_order(&b, "s").unwrap();
        assert_eq!(out, vec!["b", "a", "c"]);
    }

    #[test]
    fn is_numeric_covers_supported_dtypes() {
        assert!(is_numeric(&DataType::Float64));
        assert!(is_numeric(&DataType::Float32));
        assert!(is_numeric(&DataType::Int8));
        assert!(is_numeric(&DataType::UInt32));
        assert!(is_numeric(&DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None)));
        assert!(!is_numeric(&DataType::Utf8));
        assert!(!is_numeric(&DataType::Boolean));
    }

    // ── distinct_values_in_order: integer-type columns ───────────────────────

    #[test]
    fn distinct_values_in_order_int32() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![
                Some(3), Some(1), Some(3), Some(2), Some(1),
            ]))],
        ).unwrap();
        let out = distinct_values_in_order(&b, "x").unwrap();
        assert_eq!(out, vec!["3", "1", "2"]);
    }

    #[test]
    fn distinct_values_in_order_uint8() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::UInt8, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(UInt8Array::from(vec![Some(10u8), Some(20u8), Some(10u8)]))],
        ).unwrap();
        let out = distinct_values_in_order(&b, "x").unwrap();
        assert_eq!(out, vec!["10", "20"]);
    }

    #[test]
    fn distinct_values_in_order_int8() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int8, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int8Array::from(vec![Some(-1i8), Some(2i8), Some(-1i8)]))],
        ).unwrap();
        let out = distinct_values_in_order(&b, "x").unwrap();
        assert_eq!(out, vec!["-1", "2"]);
    }

    #[test]
    fn distinct_values_in_order_uint32() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::UInt32, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(UInt32Array::from(vec![Some(100u32), Some(200u32), Some(100u32)]))],
        ).unwrap();
        let out = distinct_values_in_order(&b, "x").unwrap();
        assert_eq!(out, vec!["100", "200"]);
    }

    // ── col_as_f64: Date/Duration are NOT supported (pre-cast upstream) ───────
    //
    // `Date32`/`Date64`/`Duration` are cast to `Timestamp(ms)` by
    // `_coerce.normalize_for_rust` before any RecordBatch reaches the render
    // layer, so `col_as_f64` must reject them via its unsupported-dtype error
    // arm. These tests pin that contract (see RSUP-02 for the full four-way
    // agreement check).

    #[test]
    fn col_as_f64_rejects_duration() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("d", DataType::Duration(TimeUnit::Nanosecond), true),
        ]));
        let arr = DurationNanosecondArray::from(vec![Some(1_000_000_000i64), None]);
        let b = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        let err = col_as_f64(&b, "d").unwrap_err();
        assert!(format!("{err}").contains("unsupported dtype"), "{err}");
    }

    #[test]
    fn col_as_f64_rejects_date32() {
        let schema = Arc::new(Schema::new(vec![Field::new("d", DataType::Date32, true)]));
        let arr = Date32Array::from(vec![Some(1i32), None]);
        let b = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        let err = col_as_f64(&b, "d").unwrap_err();
        assert!(format!("{err}").contains("unsupported dtype"), "{err}");
    }

    #[test]
    fn col_as_f64_rejects_date64() {
        let schema = Arc::new(Schema::new(vec![Field::new("d", DataType::Date64, true)]));
        let arr = Date64Array::from(vec![Some(86_400_000i64), None]);
        let b = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        let err = col_as_f64(&b, "d").unwrap_err();
        assert!(format!("{err}").contains("unsupported dtype"), "{err}");
    }

    // ── min_max_f64: Date/Duration are NOT supported (pre-cast upstream) ─────
    //
    // Parallel to the `col_as_f64_rejects_*` tests above. `Date32`/`Date64`/
    // `Duration` are pre-cast to `Timestamp(ms)` by `_coerce.normalize_for_rust`
    // before any RecordBatch reaches the render layer. `min_max_f64` must reject
    // them via its unsupported-dtype error arm. These tests pin that contract
    // directly on `min_max_f64` (T6.2b).

    /// Regression: Date/Duration are pre-cast to Timestamp(ms) by normalize_for_rust
    /// upstream, so min_max_f64 must reject DurationNanosecond (T6.2b).
    #[test]
    fn min_max_f64_rejects_duration() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("d", DataType::Duration(TimeUnit::Nanosecond), true),
        ]));
        let arr = DurationNanosecondArray::from(vec![Some(1_000_000_000i64), None]);
        let b = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        let err = min_max_f64(b.column(0).as_ref()).unwrap_err();
        assert!(err.contains("unsupported"), "expected 'unsupported' in error, got: {err}");
    }

    /// Regression: Date/Duration are pre-cast to Timestamp(ms) by normalize_for_rust
    /// upstream, so min_max_f64 must reject Date32 (T6.2b).
    #[test]
    fn min_max_f64_rejects_date32() {
        let schema = Arc::new(Schema::new(vec![Field::new("d", DataType::Date32, true)]));
        let arr = Date32Array::from(vec![Some(1i32), None]);
        let b = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        let err = min_max_f64(b.column(0).as_ref()).unwrap_err();
        assert!(err.contains("unsupported"), "expected 'unsupported' in error, got: {err}");
    }

    /// Regression: Date/Duration are pre-cast to Timestamp(ms) by normalize_for_rust
    /// upstream, so min_max_f64 must reject Date64 (T6.2b).
    #[test]
    fn min_max_f64_rejects_date64() {
        let schema = Arc::new(Schema::new(vec![Field::new("d", DataType::Date64, true)]));
        let arr = Date64Array::from(vec![Some(86_400_000i64), None]);
        let b = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        let err = min_max_f64(b.column(0).as_ref()).unwrap_err();
        assert!(err.contains("unsupported"), "expected 'unsupported' in error, got: {err}");
    }

    /// Regression: `finite_min_max_f64` delegates to `min_max_f64` via `.ok()?`.
    /// After T6.2b, `min_max_f64` returns `Err` for Date32, Date64, and
    /// DurationNanosecond (these types are pre-cast to Timestamp(ms) by
    /// `normalize_for_rust` upstream before they reach these helpers). The `.ok()?`
    /// propagation means `finite_min_max_f64` must return `None` for all three. This
    /// test would fail if `min_max_f64` ever stopped rejecting Date/Duration, or if
    /// `finite_min_max_f64` stopped delegating to it.
    #[test]
    fn finite_min_max_f64_returns_none_for_date_duration() {
        // Date32
        let schema = Arc::new(Schema::new(vec![Field::new("d", DataType::Date32, true)]));
        let arr = Date32Array::from(vec![Some(1i32), None]);
        let b = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        assert!(
            finite_min_max_f64(b.column(0).as_ref()).is_none(),
            "Date32: expected None from finite_min_max_f64"
        );

        // Date64
        let schema = Arc::new(Schema::new(vec![Field::new("d", DataType::Date64, true)]));
        let arr = Date64Array::from(vec![Some(86_400_000i64), None]);
        let b = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        assert!(
            finite_min_max_f64(b.column(0).as_ref()).is_none(),
            "Date64: expected None from finite_min_max_f64"
        );

        // DurationNanosecond
        let schema = Arc::new(Schema::new(vec![Field::new(
            "dur",
            DataType::Duration(TimeUnit::Nanosecond),
            true,
        )]));
        let arr = DurationNanosecondArray::from(vec![Some(1_000_000_000i64), None]);
        let b = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        assert!(
            finite_min_max_f64(b.column(0).as_ref()).is_none(),
            "DurationNanosecond: expected None from finite_min_max_f64"
        );
    }

    // ── col_as_ordinal_category_str ──────────────────────────────────────────

    // ── distinct_values_in_order: Float32 / Float64 columns ─────────────────

    /// F1 regression: Float64 ordinal columns with integer-valued data must be
    /// supported by `distinct_values_in_order` (previously failed with
    /// `UnsupportedDtype`), and must produce the same strings as
    /// `col_as_ordinal_category_str` for the same data.
    #[test]
    fn distinct_values_in_order_float64_whole_values_format_as_integer() {
        // [2000.0, 2001.0, 2000.0] → ["2000", "2001"] (dedup, order-preserving)
        let schema = Arc::new(Schema::new(vec![Field::new("yr", DataType::Float64, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![
                Some(2000.0), Some(2001.0), Some(2000.0),
            ]))],
        )
        .unwrap();
        let out = distinct_values_in_order(&b, "yr").unwrap();
        assert_eq!(out, vec!["2000", "2001"]);
    }

    #[test]
    fn distinct_values_in_order_float64_fractional_preserves_decimal() {
        // [1.5, 2.5] → ["1.5", "2.5"]
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![Some(1.5), Some(2.5)]))],
        )
        .unwrap();
        let out = distinct_values_in_order(&b, "v").unwrap();
        assert_eq!(out, vec!["1.5", "2.5"]);
    }

    #[test]
    fn distinct_values_in_order_float64_mixed_whole_and_fractional() {
        // [2.0, 1.5] → ["2", "1.5"]: whole values integer-formatted, fractional kept.
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![Some(2.0), Some(1.5)]))],
        )
        .unwrap();
        let out = distinct_values_in_order(&b, "v").unwrap();
        assert_eq!(out, vec!["2", "1.5"]);
    }

    /// Invariant: domain strings produced by `distinct_values_in_order` and
    /// per-row strings produced by `col_as_ordinal_category_str` must agree
    /// for Float64 columns. A value present in the domain must always hit a
    /// domain entry when looked up — this is the prerequisite for correct
    /// ordinal scale resolution and mark drawing.
    #[test]
    fn float64_distinct_values_and_category_str_agree() {
        let values = vec![Some(2000.0), Some(2001.0), Some(1.5), Some(2000.0), None];
        let schema = Arc::new(Schema::new(vec![Field::new("yr", DataType::Float64, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(values))],
        )
        .unwrap();

        let domain = distinct_values_in_order(&b, "yr").unwrap();
        let per_row = col_as_ordinal_category_str(&b, "yr").unwrap();

        // Every non-null per-row string must be in the domain.
        for s in per_row.iter().flatten() {
            assert!(
                domain.contains(s),
                "per-row string {s:?} not found in domain {domain:?}; domain/drawer mismatch"
            );
        }
        // Domain must be ["2000", "2001", "1.5"] (encounter order, deduped);
        // the color/shape domain helper drops nulls.
        assert_eq!(domain, vec!["2000", "2001", "1.5"]);

        // A null row belongs to no category: it is not in the domain and its
        // per-row category stays `None`, which places no pixel.
        assert_eq!(per_row[4], None);
    }

    #[test]
    fn distinct_values_in_order_float32_whole_values_format_as_integer() {
        use arrow::array::Float32Array;
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float32, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float32Array::from(vec![Some(10.0f32), Some(20.0f32), Some(10.0f32)]))],
        )
        .unwrap();
        let out = distinct_values_in_order(&b, "v").unwrap();
        assert_eq!(out, vec!["10", "20"]);
    }

    #[test]
    fn distinct_values_in_order_float32_fractional() {
        use arrow::array::Float32Array;
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float32, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float32Array::from(vec![Some(1.5f32), Some(2.5f32)]))],
        )
        .unwrap();
        let out = distinct_values_in_order(&b, "v").unwrap();
        assert_eq!(out, vec!["1.5", "2.5"]);
    }

    #[test]
    fn col_as_ordinal_category_str_utf8_passthrough() {
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec![Some("a"), None, Some("b")]))],
        ).unwrap();
        let out = col_as_ordinal_category_str(&b, "s").unwrap();
        assert_eq!(out, vec![Some("a".into()), None, Some("b".into())]);
    }

    #[test]
    fn col_as_ordinal_category_str_int64_stringifies() {
        // Int64 values must stringify the same way distinct_values_in_order does.
        use arrow::array::Int64Array;
        let schema = Arc::new(Schema::new(vec![Field::new("year", DataType::Int64, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![Some(2000i64), Some(2001i64), None]))],
        ).unwrap();
        let out = col_as_ordinal_category_str(&b, "year").unwrap();
        assert_eq!(out, vec![Some("2000".into()), Some("2001".into()), None]);
    }

    #[test]
    fn col_as_ordinal_category_str_int32_stringifies() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![Some(10), Some(20), None]))],
        ).unwrap();
        let out = col_as_ordinal_category_str(&b, "x").unwrap();
        assert_eq!(out, vec![Some("10".into()), Some("20".into()), None]);
    }

    #[test]
    fn col_as_ordinal_category_str_float64_whole_values_format_as_integer() {
        // Float64 ordinal columns with integer-valued data (e.g. 2000.0) must
        // produce "2000", not "2000.0", to match Int64 domain strings.
        let schema = Arc::new(Schema::new(vec![Field::new("y", DataType::Float64, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![Some(2000.0), Some(2001.0), None]))],
        ).unwrap();
        let out = col_as_ordinal_category_str(&b, "y").unwrap();
        assert_eq!(out, vec![Some("2000".into()), Some("2001".into()), None]);
    }

    #[test]
    fn col_as_ordinal_category_str_float64_fractional_preserves_decimal() {
        // Float64 with a non-zero fractional part must keep the decimal.
        let schema = Arc::new(Schema::new(vec![Field::new("y", DataType::Float64, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![Some(1.5), Some(2.0), None]))],
        ).unwrap();
        let out = col_as_ordinal_category_str(&b, "y").unwrap();
        assert_eq!(out[0], Some("1.5".into()));
        assert_eq!(out[1], Some("2".into()));
        assert_eq!(out[2], None);
    }

    #[test]
    fn col_as_ordinal_category_str_uint8_stringifies() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::UInt8, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(UInt8Array::from(vec![Some(5u8), Some(10u8), None]))],
        ).unwrap();
        let out = col_as_ordinal_category_str(&b, "x").unwrap();
        assert_eq!(out, vec![Some("5".into()), Some("10".into()), None]);
    }

    const DAY_MS: i64 = 86_400_000;
    /// 2020-01-01T00:00:00Z in milliseconds.
    const JAN_1_2020_MS: i64 = 1_577_836_800_000;

    fn ts_ms_batch(values: Vec<Option<i64>>, tz: Option<&str>) -> RecordBatch {
        let array = TimestampMillisecondArray::from(values).with_timezone_opt(tz.map(|z| z.to_string()));
        let schema = Arc::new(Schema::new(vec![Field::new("t", array.data_type().clone(), true)]));
        RecordBatch::try_new(schema, vec![Arc::new(array)]).unwrap()
    }

    /// A midnight value reads its ISO date, any other value its full ISO
    /// date-time (a fraction only when the value has one), per value.
    #[test]
    fn a_timestamp_category_label_is_date_only_at_midnight_and_full_otherwise() {
        let b = ts_ms_batch(
            vec![
                Some(JAN_1_2020_MS),
                Some(JAN_1_2020_MS + 12 * 3_600_000),
                Some(JAN_1_2020_MS + 12 * 3_600_000 + 250),
                Some(JAN_1_2020_MS + 250),
                None,
            ],
            None,
        );
        assert_eq!(
            col_as_ordinal_category_str(&b, "t").unwrap(),
            vec![
                Some("2020-01-01".to_string()),
                Some("2020-01-01T12:00:00".to_string()),
                Some("2020-01-01T12:00:00.250".to_string()),
                Some("2020-01-01T00:00:00.250".to_string()),
                None,
            ],
        );
    }

    /// Every unit reads the same instant as the same label, and a zone never
    /// shifts it: the label is UTC.
    #[test]
    fn a_timestamp_category_label_is_the_same_in_every_unit_and_zone() {
        let seconds = JAN_1_2020_MS / 1_000;
        for (dtype, array) in [
            (
                DataType::Timestamp(TimeUnit::Second, None),
                Arc::new(TimestampSecondArray::from(vec![seconds])) as Arc<dyn Array>,
            ),
            (
                DataType::Timestamp(TimeUnit::Microsecond, None),
                Arc::new(TimestampMicrosecondArray::from(vec![seconds * 1_000_000])),
            ),
            (
                DataType::Timestamp(TimeUnit::Nanosecond, None),
                Arc::new(TimestampNanosecondArray::from(vec![seconds * 1_000_000_000])),
            ),
        ] {
            let schema = Arc::new(Schema::new(vec![Field::new("t", dtype.clone(), true)]));
            let b = RecordBatch::try_new(schema, vec![array]).unwrap();
            assert_eq!(col_as_ordinal_category_str(&b, "t").unwrap(), vec![Some("2020-01-01".to_string())], "{dtype:?}");
        }
        let paris = ts_ms_batch(vec![Some(JAN_1_2020_MS)], Some("Europe/Paris"));
        assert_eq!(col_as_ordinal_category_str(&paris, "t").unwrap(), vec![Some("2020-01-01".to_string())]);
    }

    /// The domain is the distinct labels in TIME order, whatever the rows'
    /// order, and equals the per-row labels deduplicated and sorted by time:
    /// the strings the per-row reader keys bands with.
    #[test]
    fn a_timestamp_domain_is_in_time_order_and_matches_the_per_row_labels() {
        let b = ts_ms_batch(
            vec![
                Some(JAN_1_2020_MS + 2 * DAY_MS),
                Some(JAN_1_2020_MS),
                None,
                Some(JAN_1_2020_MS + 12 * 3_600_000),
                Some(JAN_1_2020_MS),
                Some(JAN_1_2020_MS + DAY_MS),
            ],
            Some("UTC"),
        );
        let domain = distinct_values_in_order(&b, "t").unwrap();
        assert_eq!(domain, vec!["2020-01-01", "2020-01-01T12:00:00", "2020-01-02", "2020-01-03"]);
        let per_row: std::collections::HashSet<String> =
            col_as_ordinal_category_str(&b, "t").unwrap().into_iter().flatten().collect();
        assert_eq!(per_row, domain.iter().cloned().collect());
    }

    /// An instant outside the calendar range keeps a name of its own, so a
    /// category is never refused and two instants never share one.
    #[test]
    fn an_out_of_range_timestamp_is_still_a_distinct_category() {
        let b = ts_ms_batch(vec![Some(i64::MAX), Some(i64::MAX - 1), Some(0)], None);
        let domain = distinct_values_in_order(&b, "t").unwrap();
        assert_eq!(domain.len(), 3);
        assert_eq!(domain[0], "1970-01-01");
    }

    /// An exchanged date lands on a categorical axis only where the owner reads
    /// its binding as categories: declared nominal or ordinal, a categorical
    /// caller scale, or an axis of dates; an undeclared date on an axis of text
    /// is a time axis, which a band axis cannot draw, so it is refused.
    #[test]
    fn an_exchanged_date_is_servable_on_a_categorical_axis_only_when_the_owner_reads_categories() {
        use crate::spec::encoding::{DataType as Declared, EncodingSpec};
        let ts = DataType::Timestamp(TimeUnit::Millisecond, None);
        let check = |declared: Option<Declared>, of_dates: bool| {
            let binding = EncodingSpec { field: "t".into(), type_: declared, ..Default::default() };
            require_exchanged_column_servable(
                crate::spec::mark::Mark::Point,
                &crate::spec::coord::CoordKind::Flip,
                "x",
                "t",
                &ts,
                &binding,
                PanelAxisKind::Categorical { of_dates },
            )
        };
        assert!(check(None, false).is_err(), "undeclared on text");
        assert!(check(Some(Declared::Temporal), false).is_err(), "declared temporal on text");
        assert!(check(Some(Declared::Nominal), false).is_ok(), "declared nominal");
        assert!(check(Some(Declared::Ordinal), false).is_ok(), "declared ordinal");
        assert!(check(None, true).is_ok(), "undeclared on an axis of dates");
        assert!(check(Some(Declared::Temporal), true).is_ok(), "declared temporal on an axis of dates");
    }

    /// Under a Temporal read a date is the refusal it drew before dates could be
    /// categories, in the very words, from the per-row door, the domain door and
    /// the keyable gate; the identity door reads every date as its label.
    #[test]
    fn a_temporal_read_refuses_a_date_in_its_earlier_words_and_the_identity_door_reads_its_label() {
        let b = ts_ms_batch(vec![Some(JAN_1_2020_MS)], None);
        let dtype = DataType::Timestamp(TimeUnit::Millisecond, None);
        let refusal = |dtype_text: String| RenderError::UnsupportedDtype { field: "t".into(), dtype: dtype_text, context: None };
        assert_eq!(
            category_strings(&b, "t", TimestampRead::Temporal).unwrap_err(),
            refusal(format!("{dtype:?} (cannot convert to ordinal category string)")),
        );
        assert_eq!(
            category_domain_entries(&b, "t", TimestampRead::Temporal).unwrap_err(),
            refusal(format!("{dtype:?} (cannot enumerate distinct values)")),
        );
        assert_eq!(
            ensure_category_keyable("t", &dtype, TimestampRead::Temporal).unwrap_err(),
            refusal(format!("{dtype:?} (cannot enumerate distinct values)")),
        );
        // The identity door partitions rows; it never picks a scale, so a date reads as its label.
        assert_eq!(col_as_identity_key_str(&b, "t").unwrap(), vec![Some("2020-01-01".to_string())]);
        assert_eq!(
            category_strings(&b, "t", TimestampRead::Categories).unwrap(),
            col_as_identity_key_str(&b, "t").unwrap()
        );
    }

    /// A `key=` column keeps its epoch text, while the same column read as a
    /// category reads labels.
    #[test]
    fn an_object_key_reads_a_timestamp_as_its_epoch_and_every_other_dtype_as_a_category() {
        let b = ts_ms_batch(vec![Some(JAN_1_2020_MS), None], None);
        assert_eq!(col_as_object_key_str(&b, "t").unwrap(), vec![Some(JAN_1_2020_MS.to_string()), None]);
        assert_eq!(col_as_ordinal_category_str(&b, "t").unwrap()[0].as_deref(), Some("2020-01-01"));
        let ints = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("t", DataType::Int64, true)])),
            vec![Arc::new(arrow::array::Int64Array::from(vec![Some(7i64)]))],
        )
        .unwrap();
        assert_eq!(col_as_object_key_str(&ints, "t").unwrap(), vec![Some("7".to_string())]);
    }

    /// Panel and band values of a timestamp column sort by instant; any other
    /// dtype keeps the order given.
    #[test]
    fn timestamp_labels_order_by_instant_and_other_dtypes_keep_their_order() {
        let b = ts_ms_batch(vec![Some(JAN_1_2020_MS + DAY_MS), Some(JAN_1_2020_MS)], None);
        let mut values = vec![("2020-01-02".to_string(), 0usize), ("2020-01-01".to_string(), 1)];
        order_labels_by_instant(&b, "t", &mut values);
        assert_eq!(values.iter().map(|(v, _)| v.as_str()).collect::<Vec<_>>(), ["2020-01-01", "2020-01-02"]);
        let text = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("t", DataType::Utf8, true)])),
            vec![Arc::new(StringArray::from(vec!["b", "a"]))],
        )
        .unwrap();
        let mut values = vec![("b".to_string(), 0usize), ("a".to_string(), 1)];
        order_labels_by_instant(&text, "t", &mut values);
        assert_eq!(values.iter().map(|(v, _)| v.as_str()).collect::<Vec<_>>(), ["b", "a"]);
    }

    /// `col_as_temporal_epoch_str` reads the raw `i64` epoch value — no `f64`
    /// round-trip — so it stays injective at magnitudes where `col_as_f64`
    /// (53-bit mantissa) would collapse distinct instants onto the same
    /// value. `9_007_199_254_740_993` is `2^53 + 1`, the first `i64` that
    /// cannot round-trip through `f64` exactly.
    #[test]
    fn col_as_temporal_epoch_str_stays_injective_past_f64_mantissa_precision() {
        let ids: Vec<i64> = (0..5).map(|i| 9_007_199_254_740_993i64 + i).collect();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "t",
            DataType::Timestamp(arrow::datatypes::TimeUnit::Nanosecond, None),
            true,
        )]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(TimestampNanosecondArray::from(ids.clone()))],
        )
        .unwrap();
        let out = col_as_temporal_epoch_str(&b, "t").unwrap();
        let expected: Vec<Option<String>> = ids.iter().map(|v| Some(v.to_string())).collect();
        assert_eq!(out, expected);
        let unique: std::collections::HashSet<_> = out.into_iter().collect();
        assert_eq!(unique.len(), 5, "5 distinct nanosecond epochs must produce 5 distinct strings");
    }

    /// Every `TimeUnit` variant is covered, and nulls pass through.
    #[test]
    fn col_as_temporal_epoch_str_covers_every_time_unit() {
        use arrow::datatypes::TimeUnit;
        for unit in [
            TimeUnit::Second,
            TimeUnit::Millisecond,
            TimeUnit::Microsecond,
            TimeUnit::Nanosecond,
        ] {
            let schema = Arc::new(Schema::new(vec![Field::new(
                "t",
                DataType::Timestamp(unit, None),
                true,
            )]));
            let batch = match unit {
                TimeUnit::Second => RecordBatch::try_new(
                    schema,
                    vec![Arc::new(TimestampSecondArray::from(vec![Some(1_700_000_000i64), None]))],
                ),
                TimeUnit::Millisecond => RecordBatch::try_new(
                    schema,
                    vec![Arc::new(TimestampMillisecondArray::from(vec![
                        Some(1_700_000_000_000i64),
                        None,
                    ]))],
                ),
                TimeUnit::Microsecond => RecordBatch::try_new(
                    schema,
                    vec![Arc::new(TimestampMicrosecondArray::from(vec![
                        Some(1_700_000_000_000_000i64),
                        None,
                    ]))],
                ),
                TimeUnit::Nanosecond => RecordBatch::try_new(
                    schema,
                    vec![Arc::new(TimestampNanosecondArray::from(vec![
                        Some(1_700_000_000_000_000_000i64),
                        None,
                    ]))],
                ),
            }
            .unwrap();
            let out = col_as_temporal_epoch_str(&batch, "t").unwrap();
            assert!(out[0].is_some(), "{unit:?} must produce a value");
            assert_eq!(out[1], None, "{unit:?} null must pass through");
        }
    }

    /// Non-Timestamp dtypes are rejected, not silently coerced.
    #[test]
    fn col_as_temporal_epoch_str_errors_on_non_timestamp() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(arrow::array::Int64Array::from(vec![Some(1i64)]))],
        )
        .unwrap();
        assert!(col_as_temporal_epoch_str(&b, "x").is_err());
    }

    // ── RSUP-02 regression: supported_numeric_dtype / is_numeric / col_as_f64 /
    //    min_max_f64 must agree on every dtype ──────────────────────────────────
    //
    // The four functions must never disagree about a dtype: for a *supported*
    // dtype all four say "yes" (predicate true, col_as_f64/min_max_f64 Ok) and
    // agree numerically; for an *unsupported* dtype all four say "no" (predicate
    // false, both functions Err).
    //
    // Date32/Date64/Duration are on the unsupported side: `_coerce.normalize_for_rust`
    // pre-casts them to Timestamp(ms) before any RecordBatch reaches the render
    // layer, so the render functions never see them on live paths and deliberately
    // reject them. This test pins that contract (T6.2b) — it constructs Date/Duration
    // batches only to confirm all four functions agree they are not supported.

    /// Helper: build a one-element RecordBatch with a single column of the given
    /// dtype containing a plausible non-null value, ready for agreement checks.
    fn one_elem_batch_for_dtype(dtype: &DataType) -> RecordBatch {
        use std::sync::Arc as A;
        let field = Field::new("v", dtype.clone(), true);
        let schema = Arc::new(Schema::new(vec![field]));
        let col: Arc<dyn Array> = match dtype {
            DataType::Null => A::new(arrow::array::NullArray::new(1)),
            DataType::Float64 => A::new(Float64Array::from(vec![Some(1.0f64)])),
            DataType::Float32 => A::new(arrow::array::Float32Array::from(vec![Some(1.0f32)])),
            DataType::Int64 => A::new(arrow::array::Int64Array::from(vec![Some(1i64)])),
            DataType::Int32 => A::new(Int32Array::from(vec![Some(1i32)])),
            DataType::Int16 => A::new(arrow::array::Int16Array::from(vec![Some(1i16)])),
            DataType::Int8 => A::new(Int8Array::from(vec![Some(1i8)])),
            DataType::UInt64 => A::new(arrow::array::UInt64Array::from(vec![Some(1u64)])),
            DataType::UInt32 => A::new(UInt32Array::from(vec![Some(1u32)])),
            DataType::UInt16 => A::new(arrow::array::UInt16Array::from(vec![Some(1u16)])),
            DataType::UInt8 => A::new(UInt8Array::from(vec![Some(1u8)])),
            DataType::Timestamp(TimeUnit::Millisecond, tz) => {
                A::new(TimestampMillisecondArray::from(vec![Some(1i64)])
                    .with_timezone_opt(tz.clone()))
            }
            DataType::Timestamp(TimeUnit::Microsecond, tz) => {
                A::new(arrow::array::TimestampMicrosecondArray::from(vec![Some(1i64)])
                    .with_timezone_opt(tz.clone()))
            }
            DataType::Timestamp(TimeUnit::Nanosecond, tz) => {
                A::new(TimestampNanosecondArray::from(vec![Some(1i64)])
                    .with_timezone_opt(tz.clone()))
            }
            DataType::Timestamp(TimeUnit::Second, tz) => {
                A::new(TimestampSecondArray::from(vec![Some(1i64)])
                    .with_timezone_opt(tz.clone()))
            }
            DataType::Duration(TimeUnit::Nanosecond) => {
                A::new(DurationNanosecondArray::from(vec![Some(1i64)]))
            }
            DataType::Duration(TimeUnit::Microsecond) => {
                A::new(DurationMicrosecondArray::from(vec![Some(1i64)]))
            }
            DataType::Duration(TimeUnit::Millisecond) => {
                A::new(DurationMillisecondArray::from(vec![Some(1i64)]))
            }
            DataType::Duration(TimeUnit::Second) => {
                A::new(DurationSecondArray::from(vec![Some(1i64)]))
            }
            DataType::Date32 => A::new(Date32Array::from(vec![Some(1i32)])),
            DataType::Date64 => A::new(Date64Array::from(vec![Some(1i64)])),
            DataType::Utf8 => A::new(StringArray::from(vec![Some("x")])),
            DataType::LargeUtf8 => A::new(LargeStringArray::from(vec![Some("x")])),
            DataType::Boolean => A::new(BooleanArray::from(vec![Some(true)])),
            other => panic!("one_elem_batch_for_dtype: unhandled dtype {other:?}"),
        };
        RecordBatch::try_new(schema, vec![col]).unwrap()
    }

    /// RSUP-02 regression: `supported_numeric_dtype`, `is_numeric`, `col_as_f64`,
    /// and `min_max_f64` must never disagree about a dtype.
    ///
    /// For every *supported* dtype all four say "yes": the predicate is true and
    /// both value functions return `Ok`, with `min_max_f64` agreeing NUMERICALLY
    /// (same f64 scale) with the min/max of `col_as_f64`'s output.
    ///
    /// For every *unsupported* dtype all four say "no": the predicate is false and
    /// both value functions return `Err`. `Date32`/`Date64`/`Duration` are on this
    /// side — `_coerce.normalize_for_rust` pre-casts them to `Timestamp(ms)` before
    /// any RecordBatch reaches the render layer (T6.2b), so the render functions
    /// never see them on live paths and deliberately reject them. This test
    /// constructs Date/Duration batches only to confirm all four agree on rejection.
    #[test]
    fn supported_numeric_dtype_is_numeric_col_as_f64_min_max_agree() {
        let supported = [
            DataType::Float64,
            DataType::Float32,
            DataType::Int64,
            DataType::Int32,
            DataType::Int16,
            DataType::Int8,
            DataType::UInt64,
            DataType::UInt32,
            DataType::UInt16,
            DataType::UInt8,
            DataType::Timestamp(TimeUnit::Millisecond, None),
            DataType::Timestamp(TimeUnit::Microsecond, None),
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            DataType::Timestamp(TimeUnit::Second, None),
        ];

        // Date/Duration are pre-cast to Timestamp(ms) by normalize_for_rust before
        // reaching the render layer, so all four functions must reject them.
        let unsupported = [
            DataType::Duration(TimeUnit::Nanosecond),
            DataType::Duration(TimeUnit::Microsecond),
            DataType::Duration(TimeUnit::Millisecond),
            DataType::Duration(TimeUnit::Second),
            DataType::Date32,
            DataType::Date64,
            DataType::Utf8,
            DataType::Boolean,
        ];

        for dtype in &supported {
            let batch = one_elem_batch_for_dtype(dtype);

            // 1. supported_numeric_dtype and is_numeric must agree and both be true.
            assert!(
                supported_numeric_dtype(dtype),
                "supported_numeric_dtype returned false for supported {dtype:?}"
            );
            assert!(
                is_numeric(dtype),
                "is_numeric returned false for {dtype:?} (disagreement with supported_numeric_dtype)"
            );

            // 2. col_as_f64 must succeed.
            let f64_vals = col_as_f64(&batch, "v").unwrap_or_else(|e| {
                panic!("col_as_f64 returned Err for supported {dtype:?}: {e:?}")
            });

            // 3. min_max_f64 must succeed.
            let (mm_min, mm_max) = min_max_f64(batch.column(0).as_ref()).unwrap_or_else(|e| {
                panic!("min_max_f64 returned Err for supported {dtype:?}: {e}")
            });

            // 4. NUMERIC AGREEMENT: min_max_f64 values must match the min/max of
            //    col_as_f64's output. This is the key invariant — both must produce
            //    the same f64 scale.
            let non_null_vals: Vec<f64> = f64_vals.into_iter().flatten().collect();
            let expected_min = non_null_vals.iter().cloned()
                .fold(f64::INFINITY, f64::min);
            let expected_max = non_null_vals.iter().cloned()
                .fold(f64::NEG_INFINITY, f64::max);

            assert_eq!(
                mm_min, expected_min,
                "min_max_f64 min disagrees with col_as_f64 min for {dtype:?}: \
                 min_max_f64={mm_min} vs col_as_f64_min={expected_min}"
            );
            assert_eq!(
                mm_max, expected_max,
                "min_max_f64 max disagrees with col_as_f64 max for {dtype:?}: \
                 min_max_f64={mm_max} vs col_as_f64_max={expected_max}"
            );
        }

        // All four functions must agree that the unsupported dtypes are NOT
        // directly supported (RSUP-02): predicate false, both value functions Err.
        for dtype in &unsupported {
            assert!(
                !supported_numeric_dtype(dtype),
                "supported_numeric_dtype returned true for unsupported {dtype:?}"
            );
            assert!(
                !is_numeric(dtype),
                "is_numeric returned true for {dtype:?} (disagreement with supported_numeric_dtype)"
            );

            let batch = one_elem_batch_for_dtype(dtype);
            assert!(
                col_as_f64(&batch, "v").is_err(),
                "col_as_f64 returned Ok for unsupported {dtype:?}"
            );
            assert!(
                min_max_f64(batch.column(0).as_ref()).is_err(),
                "min_max_f64 returned Ok for unsupported {dtype:?}"
            );
        }
    }

    /// Drift guard for [`ensure_category_keyable`]: it must admit **exactly**
    /// the dtypes both category readers accept.
    ///
    /// The gate exists so a resolver branch can refuse a non-keyable color
    /// column without building a domain it is about to discard. That is only
    /// honest while the three agree: a dtype the gate admits but the per-row
    /// reader refuses puts the refusal back at mark-build time (fatal on
    /// rule/segment, silent on point/bar — the divergence the gate closed), and
    /// a dtype the gate refuses but the readers accept rejects a working chart.
    #[test]
    fn category_readers_accept_exactly_the_dtypes_ensure_category_keyable_does() {
        let dtypes = [
            DataType::Utf8,
            DataType::LargeUtf8,
            DataType::Int64,
            DataType::Int32,
            DataType::Int16,
            DataType::Int8,
            DataType::UInt64,
            DataType::UInt32,
            DataType::UInt16,
            DataType::UInt8,
            DataType::Float64,
            DataType::Float32,
            DataType::Boolean,
            // A column with no dtype of its own: no category to enumerate.
            DataType::Null,
            // One label per instant, any unit, any zone.
            DataType::Timestamp(TimeUnit::Second, None),
            DataType::Timestamp(TimeUnit::Millisecond, None),
            DataType::Timestamp(TimeUnit::Microsecond, None),
            DataType::Timestamp(TimeUnit::Nanosecond, Some("Europe/Paris".into())),
            // Refused: not category keys. Date and Duration columns reach the
            // render layer already cast to `Timestamp(ms)`.
            DataType::Date32,
            DataType::Date64,
            DataType::Duration(TimeUnit::Millisecond),
        ];
        let (mut admitted, mut refused) = (0, 0);
        for dtype in &dtypes {
            let batch = one_elem_batch_for_dtype(dtype);
            for read in [TimestampRead::Categories, TimestampRead::Temporal] {
                let gate = ensure_category_keyable("v", dtype, read).is_ok();
                if gate { admitted += 1 } else { refused += 1 }
                assert_eq!(
                    gate,
                    category_domain_entries(&batch, "v", read).is_ok(),
                    "{dtype:?} under {read:?}: ensure_category_keyable disagrees with the domain builder \
                     (gate={gate})"
                );
                assert_eq!(
                    gate,
                    category_strings(&batch, "v", read).is_ok(),
                    "{dtype:?} under {read:?}: ensure_category_keyable disagrees with the per-row reader \
                     (gate={gate})"
                );
                // A Timestamp is a category only where the binding reads it as one; every other dtype
                // reads the same under both.
                if matches!(dtype, DataType::Timestamp(_, _)) {
                    assert_eq!(gate, read == TimestampRead::Categories, "{dtype:?} under {read:?}");
                }
            }
        }
        assert!(
            admitted > 0 && refused > 0,
            "the table must exercise both sides of the predicate; got \
             {admitted} admitted / {refused} refused"
        );
    }

    /// [`col_as_identity_key_str`] is the category reader, byte for byte, for
    /// every dtype it claims (a timestamp included, labelled), and refuses what
    /// the category reader refuses with the category reader's message.
    ///
    /// `apply_stack`'s bins must coincide with the band ladder wherever a
    /// ladder exists: a reformat on the way through would move a stack off its
    /// bands while still "working".
    #[test]
    fn identity_key_reader_is_the_category_reader() {
        let delegating = [
            DataType::Utf8, DataType::LargeUtf8,
            DataType::Int64, DataType::Int32, DataType::Int16, DataType::Int8,
            DataType::UInt64, DataType::UInt32, DataType::UInt16, DataType::UInt8,
            DataType::Float64, DataType::Float32, DataType::Boolean,
            DataType::Timestamp(TimeUnit::Second, None),
            DataType::Timestamp(TimeUnit::Millisecond, None),
            DataType::Timestamp(TimeUnit::Microsecond, None),
            DataType::Timestamp(TimeUnit::Nanosecond, None),
        ];
        for dtype in &delegating {
            let batch = one_elem_batch_for_dtype(dtype);
            assert_eq!(
                col_as_identity_key_str(&batch, "v").unwrap(),
                col_as_ordinal_category_str(&batch, "v").unwrap(),
                "{dtype:?}: the identity reader must delegate byte-exactly, not reformat"
            );
        }

        for dtype in [DataType::Duration(TimeUnit::Millisecond), DataType::Date32, DataType::Date64] {
            let batch = one_elem_batch_for_dtype(&dtype);
            let err = col_as_identity_key_str(&batch, "v").unwrap_err();
            assert_eq!(
                format!("{err}"),
                format!("{}", col_as_ordinal_category_str(&batch, "v").unwrap_err()),
                "{dtype:?}: the identity reader must surface the category reader's refusal"
            );
        }
    }

    /// The two category readers must agree on the STRINGS, not merely on which
    /// dtypes they accept: [`distinct_values_in_order`]'s output must be exactly
    /// [`col_as_ordinal_category_str`]'s per-row output deduplicated in
    /// encounter order, for every dtype in the set.
    ///
    /// Both functions' doc comments state this ("must agree exactly with"), and
    /// `float64_distinct_values_and_category_str_agree` above pins it for
    /// `Float64` — the dtype where the formatting rule is non-trivial. It was
    /// unpinned for the other twelve, which matters because consumers now rely
    /// on the equality as a *partition* guarantee rather than a display one:
    /// `render::position::apply_stack` keys its stack bins through the per-row
    /// reader precisely so that two rows share a bin exactly when they share a
    /// band from the domain builder (#184). A dtype where the two formatted
    /// differently would silently split one band's rows across two stacks.
    #[test]
    fn category_readers_produce_the_same_strings_for_every_dtype_they_accept() {
        use arrow::array::Float32Array;
        // Each row: duplicated values in a deliberately non-sorted encounter
        // order, so a reader that sorted or deduplicated differently shows up.
        let cases: Vec<(&str, DataType, Arc<dyn Array>)> = vec![
            ("Utf8", DataType::Utf8, Arc::new(StringArray::from(vec![Some("b"), Some("a"), Some("b"), None]))),
            ("LargeUtf8", DataType::LargeUtf8, Arc::new(LargeStringArray::from(vec![Some("b"), Some("a"), Some("b"), None]))),
            ("Int64", DataType::Int64, Arc::new(arrow::array::Int64Array::from(vec![Some(-2i64), Some(7), Some(-2), None]))),
            ("Int32", DataType::Int32, Arc::new(Int32Array::from(vec![Some(-2i32), Some(7), Some(-2), None]))),
            ("Int16", DataType::Int16, Arc::new(arrow::array::Int16Array::from(vec![Some(-2i16), Some(7), Some(-2), None]))),
            ("Int8", DataType::Int8, Arc::new(Int8Array::from(vec![Some(-2i8), Some(7), Some(-2), None]))),
            ("UInt64", DataType::UInt64, Arc::new(arrow::array::UInt64Array::from(vec![Some(9u64), Some(3), Some(9), None]))),
            ("UInt32", DataType::UInt32, Arc::new(UInt32Array::from(vec![Some(9u32), Some(3), Some(9), None]))),
            ("UInt16", DataType::UInt16, Arc::new(arrow::array::UInt16Array::from(vec![Some(9u16), Some(3), Some(9), None]))),
            ("UInt8", DataType::UInt8, Arc::new(UInt8Array::from(vec![Some(9u8), Some(3), Some(9), None]))),
            // Whole and fractional together: the `float_as_ordinal_str` rule
            // both readers share is the only reason these two agree at all.
            ("Float64", DataType::Float64, Arc::new(Float64Array::from(vec![Some(2000.0), Some(1.5), Some(2000.0), None]))),
            ("Float32", DataType::Float32, Arc::new(Float32Array::from(vec![Some(2000.0f32), Some(1.5), Some(2000.0), None]))),
            ("Boolean", DataType::Boolean, Arc::new(BooleanArray::from(vec![Some(true), Some(false), Some(true), None]))),
        ];
        for (name, dtype, col) in cases {
            let schema = Arc::new(Schema::new(vec![Field::new("v", dtype, true)]));
            let batch = RecordBatch::try_new(schema, vec![col]).unwrap();

            let domain = distinct_values_in_order(&batch, "v").unwrap();
            let per_row = col_as_ordinal_category_str(&batch, "v").unwrap();

            let mut deduped: Vec<String> = Vec::new();
            for s in per_row.iter().flatten() {
                if !deduped.contains(s) {
                    deduped.push(s.clone());
                }
            }
            assert_eq!(
                domain, deduped,
                "{name}: the domain builder and the per-row reader disagree on the strings"
            );
            assert_eq!(domain.len(), 2, "{name}: the case must carry a duplicate to dedup");
            assert_eq!(per_row[3], None, "{name}: a null row must stay None for the base reader");

        }
    }

    #[test]
    fn a_literal_utf8_null_category_keeps_its_band_and_a_true_null_never_joins_it() {
        // A real `"null"` STRING is a real category with its own band; a true
        // null belongs to no category, so it neither adds a band nor is keyed
        // into the string's.
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, true),
            Field::new("val", DataType::Float64, true),
        ]));
        let b = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![Some("a"), Some("null"), None])),
                Arc::new(Float64Array::from(vec![Some(1.0), Some(2.0), Some(3.0)])),
            ],
        )
        .unwrap();
        assert_eq!(distinct_values_in_order(&b, "cat").unwrap(), vec!["a", "null"]);
        assert_eq!(
            col_as_ordinal_category_str(&b, "cat").unwrap(),
            vec![Some("a".to_string()), Some("null".to_string()), None],
            "the true null stays None, never the literal's band"
        );
    }

    // ── `LeftOutTally::count_adjusted` ──────────────────────────────────────

    use crate::scale::log::LogSide;

    /// A caller row moved off-side by TWO different adjustments (two
    /// `count_adjusted` calls against the SAME caller column, each with its
    /// own `adjusted` array) counts once: the second call's merge only
    /// fills a `None` slot, never overwrites a `Some` one, so the first
    /// adjustment's own moved-off value wins and the row is not summed
    /// twice.
    #[test]
    fn count_adjusted_a_row_moved_off_by_two_adjustments_counts_once() {
        let caller = Float64Array::from(vec![5.0]);
        let extent = sided_min_max_f64(&caller, Some(LogSide::Positive)).unwrap();
        assert_eq!(extent.left_out, 0, "the caller's own row is on-side");
        let mut tally = LeftOutTally::new();
        tally.count_adjusted(&caller, extent, &Float64Array::from(vec![-1.0]), LogSide::Positive).expect("f64 columns");
        tally.count_adjusted(&caller, extent, &Float64Array::from(vec![-2.0]), LogSide::Positive).expect("f64 columns");
        assert_eq!(tally.rows(), 1, "one caller row, moved off by two adjustments, must count once: {tally:?}");
    }

    /// A caller row that is already off-side is never ALSO counted as
    /// "moved" by the same adjustment: `count_adjusted`'s own on-side guard
    /// on the caller value excludes it before the adjusted value is even
    /// read, so the row's single count comes from `caller_extent.left_out`
    /// alone, not `left_out` PLUS a moved entry for the same row.
    #[test]
    fn count_adjusted_a_caller_off_side_row_is_never_also_counted_as_moved() {
        let caller = Float64Array::from(vec![-5.0]);
        let extent = sided_min_max_f64(&caller, Some(LogSide::Positive)).unwrap();
        assert_eq!(extent.left_out, 1, "the caller's own row is off-side");
        let mut tally = LeftOutTally::new();
        // The adjustment moves it further (still off-side) — direction does
        // not matter, since the caller-side guard fires first either way.
        tally.count_adjusted(&caller, extent, &Float64Array::from(vec![-10.0]), LogSide::Positive).expect("f64 columns");
        assert_eq!(tally.rows(), 1, "the off-side caller row must count exactly once, not once via the extent and again as moved: {tally:?}");
    }

    /// `merge` ORs two tallies' masks for the SAME caller column: one tally
    /// moved row 0 off-side, the other moved row 1 — after merging, both
    /// rows are counted, not just whichever tally's entry happened to be
    /// kept.
    #[test]
    fn count_adjusted_merge_ors_equal_caller_masks() {
        let caller = Float64Array::from(vec![5.0, 8.0]);
        let extent = sided_min_max_f64(&caller, Some(LogSide::Positive)).unwrap();
        assert_eq!(extent.left_out, 0);
        let mut tally_a = LeftOutTally::new();
        tally_a.count_adjusted(&caller, extent, &Float64Array::from(vec![-1.0, 8.0]), LogSide::Positive).expect("f64 columns");
        let mut tally_b = LeftOutTally::new();
        tally_b.count_adjusted(&caller, extent, &Float64Array::from(vec![5.0, -2.0]), LogSide::Positive).expect("f64 columns");
        tally_a.merge(tally_b);
        assert_eq!(tally_a.rows(), 2, "row 0 (moved by tally_a) and row 1 (moved by tally_b) must both survive the merge: {tally_a:?}");
    }

    /// `any_on_side` follows the ADJUSTED values, not the caller's: an
    /// all-off-side caller lifted to an on-side value by its own
    /// adjustment still sets it, and an all-on-side caller pushed off-side
    /// by its adjustment does NOT set it — the opposite of what reading the
    /// caller's own extent would report.
    #[test]
    fn count_adjusted_any_on_side_follows_the_adjusted_values_not_the_callers() {
        let lifted_caller = Float64Array::from(vec![-5.0]);
        let lifted_extent = sided_min_max_f64(&lifted_caller, Some(LogSide::Positive)).unwrap();
        assert_eq!(lifted_extent.left_out, 1, "the caller's own row is off-side");
        let mut tally = LeftOutTally::new();
        tally.count_adjusted(&lifted_caller, lifted_extent, &Float64Array::from(vec![3.0]), LogSide::Positive).expect("f64 columns");
        assert!(tally.any_on_side(), "the adjusted value is on-side even though the caller's own was not");

        let pushed_caller = Float64Array::from(vec![5.0]);
        let pushed_extent = sided_min_max_f64(&pushed_caller, Some(LogSide::Positive)).unwrap();
        assert_eq!(pushed_extent.left_out, 0, "the caller's own row is on-side");
        let mut tally = LeftOutTally::new();
        tally.count_adjusted(&pushed_caller, pushed_extent, &Float64Array::from(vec![-3.0]), LogSide::Positive).expect("f64 columns");
        assert!(!tally.any_on_side(), "the adjusted value is off-side even though the caller's own was not");
    }

    /// `count_adjusted`'s `on_side` closure is symmetric by reading
    /// (`LogSide::Negative` flips `v > 0.0` to `v < 0.0`), pinned directly
    /// here rather than left to the layered/unlayered tests' own
    /// positive-side fixtures: a caller row on-side on the negative side
    /// (`-5.0`), moved by the adjustment to an off-side value (`3.0`,
    /// positive), counts as moved; a caller row already off-side on the
    /// negative side (`5.0`) does not count twice.
    #[test]
    fn count_adjusted_on_the_negative_side() {
        let caller = Float64Array::from(vec![-5.0, 5.0]);
        let extent = sided_min_max_f64(&caller, Some(LogSide::Negative)).unwrap();
        assert_eq!(extent.left_out, 1, "row 1 (5.0) is off-side on the negative side");
        let mut tally = LeftOutTally::new();
        tally.count_adjusted(&caller, extent, &Float64Array::from(vec![3.0, 8.0]), LogSide::Negative).expect("f64 columns");
        assert_eq!(tally.rows(), 2, "row 0 (-5.0 on-side, moved to 3.0 off-side) plus row 1 (already off-side): {tally:?}");
    }

    /// A caller and adjusted column of different lengths refuse rather than
    /// silently truncating or panicking — `AdjustedRows`'s own invariant
    /// keeps this from happening on a live path, so this pins the typed
    /// `Err` the caller maps to a refusal instead.
    #[test]
    fn count_adjusted_mismatched_lengths_return_err() {
        let caller = Float64Array::from(vec![1.0, 2.0]);
        let extent = sided_min_max_f64(&caller, Some(LogSide::Positive)).unwrap();
        let mut tally = LeftOutTally::new();
        let err = tally.count_adjusted(&caller, extent, &Float64Array::from(vec![1.0]), LogSide::Positive).unwrap_err();
        assert!(err.contains("row-aligned"), "{err}");
    }
}
