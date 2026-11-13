//! A lowered statistic drawn on a log positional axis computes in that
//! axis's own units: the primitives every demanded `Bin`/`Kde`/`Kde2D`/
//! `Contour`/`DataWindow` arm shares (`crate::transform::color_bin_group`
//! keeps the demand itself — `SpaceDemand`, `StepSpace`, `RunPlan`).
//!
//! A value maps into log units through [`LogAxisSpace::forward`] and back
//! through [`LogAxisSpace::backward`], both signed and monotonically
//! increasing on either side of zero, so a pairwise extent, a cumulative
//! sum and a dodge slot order all keep data order with no reordering step.
//! An off-side value (zero, or signed opposite the space's side) maps to
//! `None` and is left out.
//!
//! A demanded slot computes from a step-private column holding the
//! caller's column mapped into log units ([`with_mapped_slot`]); the
//! caller's own column stays in the batch untouched, so every other reader
//! of it inside the step (an aggregate of the binned field, a per-row pass
//! through, a second slot reading the same column) reads data units.

use std::cell::RefCell;
use std::ops::Range;
use std::rc::Rc;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array, Float64Builder, ListArray, ListBuilder, RecordBatch};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use crate::transform::color_bin_group::SlotSpace;
use crate::transform::context::{describe_group_keys, describe_group_row, AllNullOutcome, TransformContext, TransformWarning};
use crate::transform::core::{SpaceColumn, SpaceSlot};
use crate::transform::group_key::{group_partition_multi, NullKeyPolicy, FIT_NULL_KEYS};
use crate::transform::numeric_util::{coerce_to_float64, column_extent};
use crate::transform::project::with_column;

/// Which side of zero a `Log` axis places, and at what base: the space a
/// lowered `Bin`/`Kde`/`Kde2D`/`Contour`/`DataWindow` statistic computes in
/// once a positional channel it draws resolves to a `LogScale`. Carries no
/// domain — only the base and the side matter for mapping a value into
/// (and out of) log units; the resolved domain stays the render's own
/// concern.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LogAxisSpace {
    pub(crate) base: f64,
    pub(crate) side: crate::scale::log::LogSide,
}

impl LogAxisSpace {
    fn sign(&self) -> f64 {
        match self.side {
            crate::scale::log::LogSide::Positive => 1.0,
            crate::scale::log::LogSide::Negative => -1.0,
        }
    }

    /// Whether `value` sits off this space's side: zero, or signed opposite
    /// the side — the same "zero or negative"/"zero or positive" rule
    /// [`crate::scale::log::LogSide::excluded_words`] names. `false` for a
    /// non-finite value: NaN and null are excluded by the ordinary null
    /// policy, not counted as off-side.
    pub(crate) fn off_side(&self, value: f64) -> bool {
        if !value.is_finite() {
            return false;
        }
        match self.side {
            crate::scale::log::LogSide::Positive => value <= 0.0,
            crate::scale::log::LogSide::Negative => value >= 0.0,
        }
    }

    /// Map a finite on-side `value` into this space's units: `s *
    /// log_base(|value|)`, where `s` is `+1` on the positive side and `-1`
    /// on the negative one. Signed so the map is monotonically INCREASING
    /// in `value` on either side (on the negative side, a value closer to
    /// zero — a smaller magnitude — maps to a larger signed result than a
    /// more negative one, matching data order): a pairwise extent, a
    /// cumulative sum and a dodge slot order computed on the mapped values
    /// never need reordering. `None` for a non-finite or off-side value,
    /// the same exclusion [`off_side`](Self::off_side) reports — a caller
    /// that needs to count the exclusion asks that separately.
    pub(crate) fn forward(&self, value: f64) -> Option<f64> {
        if !value.is_finite() || self.off_side(value) {
            return None;
        }
        Some(self.sign() * value.abs().log(self.base))
    }

    /// The inverse of [`forward`](Self::forward): `s * base^(s * mapped)`.
    /// Monotonically increasing in `mapped` on either side, mirroring
    /// `forward`, so a pair mapped forward and back keeps its order with no
    /// reordering step.
    pub(crate) fn backward(&self, mapped: f64) -> f64 {
        let s = self.sign();
        s * self.base.powf(s * mapped)
    }

    /// [`forward`](Self::forward) applied to both ends of a data-space
    /// extent, in order: `None` if either endpoint does not map. Needs no
    /// reordering after mapping — `forward` is monotone on this space's
    /// side, so `lo <= hi` in data space keeps `forward(lo) <=
    /// forward(hi)`.
    fn forward_extent(&self, extent: (f64, f64)) -> Option<(f64, f64)> {
        Some((self.forward(extent.0)?, self.forward(extent.1)?))
    }

    /// An extent-like spec field (a `Bin`'s or `Kde`'s `extent`, a `Kde`'s
    /// `count_scale_extent`, one axis's half of a `Kde2D`'s `extent`,
    /// `Smooth`'s or `Robust`'s `x_range`) in this space's units, for the
    /// step's arm to put into the spec its body reads. An endpoint off
    /// this space's side is refused in the step's own voice (`label`):
    /// malformed input for that scale, not silently rendered against an
    /// extent the axis can never place. A non-finite endpoint is refused
    /// the same way, so this function's own contract — `None` in gives
    /// `None` out, `Some` in gives `Some` out or a refusal — holds by
    /// construction for every caller: every caller (Bin/Kde/Kde2D/Smooth/
    /// Robust) validates finiteness at its own binding, so this arm is
    /// unreachable on a live path and none of their refusal texts come
    /// from here — this check holds the contract for any caller that does
    /// not. `param` names the caller-written keyword the
    /// refusal quotes (`"extent"` for Bin/Kde/Kde2D, `"x_range"` for
    /// Smooth/Robust), so the remedy names the parameter the caller
    /// actually wrote rather than a generic `extent=` no fit door exposes.
    pub(crate) fn extent_in_space(&self, label: &str, param: &str, extent: Option<(f64, f64)>) -> PyResult<Option<(f64, f64)>> {
        let Some((a, b)) = extent else { return Ok(None) };
        if !a.is_finite() || !b.is_finite() {
            return Err(PyValueError::new_err(format!("{label}: {param} ({a}, {b}) must be two finite numbers")));
        }
        if let Some(v) = [a, b].into_iter().find(|&v| self.off_side(v)) {
            return Err(PyValueError::new_err(format!(
                "{label}: {param} ({a}, {b}) names {v}, which a LogScale cannot place (it is zero or \
                 on the wrong side); state both endpoints on the scale's own side, or drop {param}="
            )));
        }
        Ok(self.forward_extent((a, b)))
    }
}

/// Whether a column this module maps is one scalar value per row (a
/// `Bin`/`Kde`/`Kde2D` field, a `Bin`/`Kde`/`Contour` output), one
/// `List<Float64>` per row (a `Kde2D`/`Contour` grid axis), or one
/// `List<Float64>` per row whose elements at `start..end` alone carry the
/// slot's units (a `Kde2D`'s `extent`, `[x_lo, x_hi, y_lo, y_hi]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ColumnShape {
    Scalar,
    List,
    ListPositions(usize, usize),
}

impl ColumnShape {
    /// The list positions a list column of this shape maps (every position
    /// of a whole list), `None` for a scalar column.
    fn positions(self) -> Option<Range<usize>> {
        match self {
            Self::Scalar => None,
            Self::List => Some(0..usize::MAX),
            Self::ListPositions(start, end) => Some(start..end),
        }
    }
}

/// One input slot a lowered step is demanded on, run through
/// [`run_in_log_space`]: `slot` is the step's input role, `field` the
/// caller's column that slot reads, `space` what [`crate::transform::color_bin_group::SpaceDemand::of`]
/// demanded for it, and `shape` whether `field`'s column holds one value
/// per row (`Bin`/`Kde`'s field, `Kde2D`'s `x`/`y`) or one list per row
/// (`Contour`'s `grid_x`/`grid_y`, read from a `Kde2D` surface).
pub(crate) struct DemandedField<'a> {
    pub(crate) slot: SpaceSlot,
    pub(crate) field: &'a str,
    pub(crate) space: &'a SlotSpace,
    pub(crate) shape: ColumnShape,
}

/// The step-private column a demanded scalar `slot` reading `field`
/// computes from: `__log_space_<slot>__<field>`, named per slot so two
/// slots reading one column (a `Kde2D` whose `x` and `y` name the same
/// field) each get their own, and prefixed as `own_value_column`'s private
/// columns are.
pub(crate) fn log_space_column(slot: SpaceSlot, field: &str) -> String {
    let slot = match slot {
        SpaceSlot::Field => "field".to_string(),
        SpaceSlot::X => "x".to_string(),
        SpaceSlot::Y => "y".to_string(),
        SpaceSlot::GridX => "grid_x".to_string(),
        SpaceSlot::GridY => "grid_y".to_string(),
        SpaceSlot::Sort(i) => format!("sort{i}"),
    };
    format!("__log_space_{slot}__{field}")
}

/// `field`'s column in `batch`, refused in the step's voice (`label`) when
/// the batch lacks it.
fn caller_column<'b>(batch: &'b RecordBatch, field: &str, label: &str) -> PyResult<&'b ArrayRef> {
    let schema = batch.schema();
    let idx = schema.index_of(field).map_err(|_| {
        PyValueError::new_err(format!(
            "{label}: column '{field}' not found; available: {:?}",
            schema.fields().iter().map(|f| f.name()).collect::<Vec<_>>()
        ))
    })?;
    Ok(batch.column(idx))
}

/// `batch` with `demand`'s slot mapped into its space's units, and the name
/// of the column that slot reads there. A scalar slot's caller column,
/// coerced to Float64 in the step's voice (`label`), is mapped through
/// [`LogAxisSpace::forward`] into the step-private column
/// [`log_space_column`] names (an off-side or non-finite value becomes
/// null), so the caller's column, its Arrow type and its nullability stay
/// as they were; a batch already holding that name is refused rather than
/// shadowed. A list slot (`Contour`'s `grid_x`/`grid_y`, which `Contour`
/// reads by its fixed input contract and never passes through to its
/// output) is replaced in place under its own name, its field rebuilt, an
/// off-side grid value becoming null as a scalar one does.
pub(crate) fn with_mapped_slot(batch: &RecordBatch, demand: &DemandedField, label: &str) -> PyResult<(RecordBatch, String)> {
    let space = demand.space.space;
    let caller = caller_column(batch, demand.field, label)?;
    let (name, column): (String, ArrayRef) = match demand.shape.positions() {
        None => {
            let name = log_space_column(demand.slot, demand.field);
            if batch.schema().index_of(&name).is_ok() {
                return Err(PyValueError::new_err(format!(
                    "{label}: input column '{name}' collides with the column a log axis computes in; rename the input column"
                )));
            }
            let values = coerce_to_float64(caller, label, demand.field)?;
            let mapped: Float64Array = values.iter().map(|v| v.and_then(|v| space.forward(v))).collect();
            (name, Arc::new(mapped))
        }
        Some(positions) => (demand.field.to_string(), map_list(caller, positions, |_, v| space.forward(v), demand.field, label)?),
    };
    let mapped = with_column(batch, &name, column).map_err(|e| PyValueError::new_err(format!("{label}: {e}")))?;
    Ok((mapped, name))
}

/// `column`, a `List<Float64>` with one list per row, with each present
/// element of row `i` at `positions` replaced by `map(i, value)` (`None`
/// becomes null) and every other element copied. A column of any other type
/// is returned unchanged.
fn map_list(column: &ArrayRef, positions: Range<usize>, map: impl Fn(usize, f64) -> Option<f64>, name: &str, label: &str) -> PyResult<ArrayRef> {
    let Some(list) = column.as_any().downcast_ref::<ListArray>() else { return Ok(Arc::clone(column)) };
    let mut builder = ListBuilder::new(Float64Builder::new());
    for i in 0..list.len() {
        if list.is_null(i) {
            builder.append(false);
            continue;
        }
        let values = list.value(i);
        let values = values.as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
            PyValueError::new_err(format!("{label}: internal error mapping '{name}' to/from data units"))
        })?;
        for (position, v) in values.iter().enumerate() {
            let v = if positions.contains(&position) { v.and_then(|v| map(i, v)) } else { v };
            builder.values().append_option(v);
        }
        builder.append(true);
    }
    Ok(Arc::new(builder.finish()))
}

/// `result` with its `name` column rewritten value by value: each present
/// entry of row `i` (every position `shape` maps, for a list column) becomes
/// `map(i, value)`, `None` giving null; a null entry stays null. The
/// column's own field is kept (the output schema stays the data-space
/// body's); an absent column, or a column of another type, is left as it
/// is, so one output list serves every route of a step (`per_row` and the
/// aggregating route).
fn rewrite_output(result: RecordBatch, name: &str, shape: ColumnShape, map: impl Fn(usize, f64) -> Option<f64>, label: &str) -> PyResult<RecordBatch> {
    let schema = result.schema();
    let Ok(idx) = schema.index_of(name) else { return Ok(result) };
    let column = result.column(idx);
    let rewritten: ArrayRef = match shape.positions() {
        None => {
            let Some(values) = column.as_any().downcast_ref::<Float64Array>() else { return Ok(result) };
            Arc::new(values.iter().enumerate().map(|(i, v)| v.and_then(|v| map(i, v))).collect::<Float64Array>())
        }
        Some(positions) => map_list(column, positions, map, name, label)?,
    };
    let mut columns = result.columns().to_vec();
    columns[idx] = rewritten;
    RecordBatch::try_new(schema, columns).map_err(|e| PyValueError::new_err(format!("{label}: {e}")))
}

/// `result` with its `name` column mapped back to data units through
/// [`LogAxisSpace::backward`] ([`rewrite_output`]): a scalar Float64 output
/// (bin edges, a kde grid, contour vertices) or a list one (a `Kde2D` grid
/// axis, or its `extent` positions).
fn backward_map_output(result: RecordBatch, name: &str, shape: ColumnShape, space: &LogAxisSpace, label: &str) -> PyResult<RecordBatch> {
    rewrite_output(result, name, shape, |_, v| Some(space.backward(v)), label)
}

/// Whether `values` hold no on-side value and at least one off-side one:
/// a demanded step's own body degrades (a flat `DegenerateFit`, an empty
/// grid) over data its axis already refuses to place, restating a warning
/// the axis's own off-side report already gives. `false` for values with
/// nothing off-side, and for values with nothing on OR off side (already
/// null/NaN: an ordinary missing-data case unrelated to this axis's side).
fn all_off_side(values: impl IntoIterator<Item = Option<f64>>, space: &LogAxisSpace) -> bool {
    let mut finite = values.into_iter().flatten().filter(|v| v.is_finite()).peekable();
    finite.peek().is_some() && finite.all(|v| space.off_side(v))
}

/// What a log axis left a call with no value to place ([`emptied_by_axis`]).
#[derive(Debug, PartialEq)]
pub(crate) enum AxisEmptied {
    /// No group (or no call) holds an off-side value and none on the side.
    Nothing,
    /// An ungrouped call whose rows hold none on some demanded slot.
    WholeCall,
    /// A grouped call whose every key is null, which the body fits as one
    /// batch ([`NullKeyPolicy::SkipWithWholeBatchFallback`]), and whose rows
    /// hold none on some demanded slot.
    FallbackBatch,
    /// The labels ([`describe_group_keys`]) of the step's own groups that
    /// hold none on some demanded slot, in group order.
    Groups(Vec<String>),
}

/// What `ctx`'s log axis left empty of a call reading `slots` (each a
/// caller's scalar column and the space it is demanded in) over `batch`:
/// the whole-call form of [`all_off_side`] for an ungrouped step, else the
/// same test restricted to each group of the step's own `groupby`
/// ([`TransformContext::groupby`]). Groups are formed by the shared
/// partitioner under `null_keys`, which each statistic hands over as the
/// one policy its own partition reads, so a group the body keeps (the
/// window's pooled null key) is a group here and a row the body drops is
/// not; a body that fits the whole batch when every key is null is judged
/// on that batch ([`AxisEmptied::FallbackBatch`]). The partitioner runs
/// under a context whose sinks are private, so nothing it reports reaches
/// the run. A partition that cannot be formed finds nothing: the body
/// refuses it in its own voice.
pub(crate) fn emptied_by_axis(
    ctx: &TransformContext,
    label: &str,
    batch: &RecordBatch,
    slots: &[(&Float64Array, LogAxisSpace)],
    null_keys: NullKeyPolicy,
) -> AxisEmptied {
    if ctx.groupby.is_empty() {
        let emptied = slots.iter().any(|(values, space)| all_off_side(values.iter(), space));
        return if emptied { AxisEmptied::WholeCall } else { AxisEmptied::Nothing };
    }
    if !slots.iter().any(|(values, space)| values.iter().flatten().any(|v| space.off_side(v))) {
        return AxisEmptied::Nothing;
    }
    let Ok(partition) = group_partition_multi(batch, &ctx.groupby, label, null_keys, &with_private_sinks(ctx)) else {
        return AxisEmptied::Nothing;
    };
    if partition.groups.is_empty() && null_keys == NullKeyPolicy::SkipWithWholeBatchFallback && batch.num_rows() > 0 {
        let emptied = slots.iter().any(|(values, space)| all_off_side(values.iter(), space));
        return if emptied { AxisEmptied::FallbackBatch } else { AxisEmptied::Nothing };
    }
    let labels: Vec<String> = partition
        .groups
        .iter()
        .filter(|(_, rows)| {
            slots.iter().any(|(values, space)| all_off_side(rows.iter().map(|&row| values.is_valid(row).then(|| values.value(row))), space))
        })
        .map(|(keys, _)| describe_group_keys(&ctx.groupby, keys))
        .collect();
    if labels.is_empty() {
        AxisEmptied::Nothing
    } else {
        AxisEmptied::Groups(labels)
    }
}

/// Which rows of `result` are the output of an emptied group, and the labels
/// of the groups they matched: a row belongs to a group when its own label
/// ([`describe_group_row`] under `groupby`) is one of `labels`. A group whose
/// rows the result does not carry under its label matches nothing, so it is
/// neither blanked nor reported.
fn rows_of_groups(result: &RecordBatch, groupby: &[String], labels: &[String]) -> (Vec<bool>, Vec<String>) {
    let mut matched: Vec<String> = Vec::new();
    let flags = (0..result.num_rows())
        .map(|row| {
            let Some(group) = describe_group_row(result, groupby, row).filter(|group| labels.contains(group)) else { return false };
            if !matched.contains(&group) {
                matched.push(group);
            }
            true
        })
        .collect();
    (flags, matched)
}

/// `result` with each of its `outputs` columns set to NaN (every position
/// `shape` maps, for a list column) on the rows `blank` flags: an emptied
/// group's default output must take no position on the axis. A column the
/// result lacks, or holds in another type, is left as it is.
fn blank_rows(result: RecordBatch, blank: &[bool], outputs: &[(&str, ColumnShape)], label: &str) -> PyResult<RecordBatch> {
    if !blank.contains(&true) {
        return Ok(result);
    }
    outputs.iter().try_fold(result, |result, (name, shape)| {
        rewrite_output(result, name, *shape, |row, v| Some(if blank[row] { f64::NAN } else { v }), label)
    })
}

/// Drop from `ctx`'s warnings, from index `from` on (the ones the body just
/// raised), each per-group `DegenerateFit` and `IntervalUndefined` that
/// names a group in `labels` (`None`: the ones that name no group, which is
/// how a body that fit the whole batch reports itself): the axis emptied
/// that group, so its fit failed for want of values the caller never gave
/// the statistic, and the axis's own off-side report already says why.
/// Every other warning stays.
fn drop_fit_warnings(ctx: &TransformContext, from: usize, labels: Option<&[String]>) {
    let mut index = 0;
    ctx.warnings.borrow_mut().retain(|warning| {
        index += 1;
        let group = match warning {
            TransformWarning::DegenerateFit { group, .. } | TransformWarning::IntervalUndefined { group, .. } => Some(group.as_deref()),
            _ => None,
        };
        let dropped = group.is_some_and(|group| match (group, labels) {
            (None, None) => true,
            (Some(group), Some(labels)) => labels.iter().any(|label| label == group),
            _ => false,
        });
        index <= from || !dropped
    });
}

/// `ctx` with a fresh, independent warnings sink: every other shared sink
/// (`log_space`, `color_bins`, `run_plan`, `log_input`, `all_null_partitions`)
/// stays shared with `ctx` (an `Rc::clone`, not a deep copy), but a warning
/// pushed through the returned context is never read back by the pipeline
/// runner that owns `ctx`'s own sink.
fn with_scratch_warnings(ctx: &TransformContext) -> TransformContext {
    TransformContext { warnings: Rc::new(RefCell::new(Vec::new())), ..ctx.clone() }
}

/// [`with_scratch_warnings`] with a private `all_null_partitions` sink too:
/// nothing the returned context's calls report, warning or all-null
/// partition, reaches the run.
fn with_private_sinks(ctx: &TransformContext) -> TransformContext {
    TransformContext { all_null_partitions: Rc::new(RefCell::new(Vec::new())), ..with_scratch_warnings(ctx) }
}

/// The column each demanded slot reads in the batch [`run_in_log_space`]
/// hands its body: an arm builds its mapped spec with only the demanded
/// slots renamed ([`Self::column`]).
#[derive(Debug, Default)]
pub(crate) struct SlotColumns {
    columns: Vec<(SpaceSlot, String)>,
    private: Vec<String>,
}

impl SlotColumns {
    /// The column `slot` reads in the mapped batch: its mapped column when
    /// the slot is demanded, else `caller_field` as written.
    pub(crate) fn column(&self, slot: SpaceSlot, caller_field: &str) -> String {
        self.columns.iter().find(|(s, _)| *s == slot).map_or_else(|| caller_field.to_string(), |(_, name)| name.clone())
    }
}

/// Run a lowered step's body in log space. For each of `demand`'s slots
/// this records the caller's untouched column for the off-side tally
/// (once per axis in [`SlotSpace::axes`], where
/// [`SlotSpace::records_caller_input`] says to) and maps the slot
/// ([`with_mapped_slot`]); [`emptied_by_axis`] then asks what the axis left
/// with no value to place.
///
/// An ungrouped call whose slot holds no on-side value and at least one
/// off-side one runs `body` on a zero-row slice under a scratch warnings
/// sink, its output cut to zero rows too: the step neither restates a
/// degenerate fit its axis already refuses nor leaves a phantom on-side
/// value (a `Kde`'s default grid mapped back) for the axis to read, and the
/// axis's own `LogScaleNoValueOnSide` refusal speaks for a whole frame with
/// nothing on the side. Inside a facet partition the panel is also recorded
/// as one the axis emptied ([`AllNullOutcome::OffSide`]), so it draws empty
/// instead of vanishing.
///
/// A grouped call runs `body` on the whole mapped batch (each grouped body
/// degrades a group with no usable value and never refuses it). A group of
/// its own `groupby` that holds no on-side value and at least one off-side
/// one is then emptied by the axis: the `DegenerateFit` and
/// `IntervalUndefined` the body raised for it are dropped, every `outputs`
/// column on its rows is set to NaN, and its label is recorded
/// ([`TransformContext::note_log_emptied`]) for the null census. The axis's
/// own off-side report names that cause once, so the group keeps its row,
/// its legend entry and its place, and takes no position on the axis.
///
/// Each of `outputs` (an output column, the slot whose units it holds, and
/// its shape) is mapped back to data units in that slot's space, and every
/// step-private column the result still carries (a `per_row` `Bin` passes
/// its input through) is dropped, so the output schema is the data-space
/// body's own. An extent-like spec field is mapped by each arm through
/// [`LogAxisSpace::extent_in_space`], since only the arm knows which of its
/// spec's fields carry which slot's units.
pub(crate) fn run_in_log_space(
    ctx: &TransformContext,
    label: &str,
    batch: &RecordBatch,
    demand: &[DemandedField],
    null_keys: NullKeyPolicy,
    body: impl FnOnce(&RecordBatch, &SlotColumns, &TransformContext) -> PyResult<RecordBatch>,
    outputs: &[(&str, SpaceSlot, ColumnShape)],
) -> PyResult<RecordBatch> {
    let mut mapped_batch = batch.clone();
    let mut columns = SlotColumns::default();
    let mut caller_slots: Vec<(Float64Array, LogAxisSpace)> = Vec::new();
    for d in demand {
        if d.shape == ColumnShape::Scalar {
            let caller = caller_column(batch, d.field, label)?;
            if d.space.records_caller_input {
                for &axis in &d.space.axes {
                    ctx.note_log_input(axis, d.field, Arc::clone(caller));
                }
            }
            caller_slots.push((coerce_to_float64(caller, label, d.field)?, d.space.space));
        }
        let (next, name) = with_mapped_slot(&mapped_batch, d, label)?;
        mapped_batch = next;
        if name != d.field {
            columns.private.push(name.clone());
        }
        columns.columns.push((d.slot, name));
    }
    let slots: Vec<(&Float64Array, LogAxisSpace)> = caller_slots.iter().map(|(values, space)| (values, *space)).collect();
    let emptied = emptied_by_axis(ctx, label, batch, &slots, null_keys);
    let warned_before = ctx.warnings.borrow().len();
    let mut result = if emptied == AxisEmptied::WholeCall {
        if ctx.is_partition {
            ctx.record_all_null(AllNullOutcome::OffSide);
        }
        body(&mapped_batch.slice(0, 0), &columns, &with_scratch_warnings(ctx))?.slice(0, 0)
    } else {
        body(&mapped_batch, &columns, ctx)?
    };
    for (output, slot, shape) in outputs {
        let Some(d) = demand.iter().find(|d| d.slot == *slot) else { continue };
        result = backward_map_output(result, output, *shape, &d.space.space, label)?;
    }
    let blanked: Vec<(&str, ColumnShape)> =
        outputs.iter().filter(|(_, slot, _)| demand.iter().any(|d| d.slot == *slot)).map(|(name, _, shape)| (*name, *shape)).collect();
    match &emptied {
        AxisEmptied::Groups(labels) => {
            let (flags, matched) = rows_of_groups(&result, &ctx.groupby, labels);
            if !matched.is_empty() {
                result = blank_rows(result, &flags, &blanked, label)?;
                drop_fit_warnings(ctx, warned_before, Some(&matched));
                ctx.note_log_emptied(Some(matched));
            }
        }
        AxisEmptied::FallbackBatch => {
            let all = vec![true; result.num_rows()];
            result = blank_rows(result, &all, &blanked, label)?;
            drop_fit_warnings(ctx, warned_before, None);
            ctx.note_log_emptied(None);
        }
        AxisEmptied::Nothing | AxisEmptied::WholeCall => {}
    }
    let schema = result.schema();
    let kept: Vec<usize> = (0..schema.fields().len()).filter(|&i| !columns.private.contains(schema.field(i).name())).collect();
    if kept.len() < schema.fields().len() {
        result = result.project(&kept).map_err(|e| PyValueError::new_err(format!("{label}: {e}")))?;
    }
    Ok(result)
}

/// [`run_in_log_space`] for `Smooth`, `Robust`, `Logistic` and `Glm`: the
/// demand (`x`/`y`, each the field it reads and the slot it is demanded
/// in, `None` when that slot is not demanded) and the map-back outputs
/// both come from `columns` (`TransformSpec::fit_space_columns`/
/// `fit_x_only_space_columns`), the one declaration of which output holds
/// which slot's units, so a demanded column and its map-back can never
/// drift apart — only the columns whose slot is actually demanded are
/// mapped back, so a `Residuals` output (declaring `x` alone) never has
/// its `y` touched even when a caller somehow demands that slot too.
/// `body` receives the mapped batch and the mapped `x`/`y` column names
/// (a slot not demanded keeps its caller name unchanged), plus the
/// context `run_in_log_space` itself threads through (the degenerate
/// scratch sink on an all-off-side slot).
pub(crate) fn run_fit_in_log_space(
    ctx: &TransformContext,
    label: &str,
    batch: &RecordBatch,
    columns: &[SpaceColumn],
    x: (&str, Option<&SlotSpace>),
    y: (&str, Option<&SlotSpace>),
    body: impl FnOnce(&RecordBatch, &str, &str, &TransformContext) -> PyResult<RecordBatch>,
) -> PyResult<RecordBatch> {
    let demand: Vec<DemandedField> = [(SpaceSlot::X, x.0, x.1), (SpaceSlot::Y, y.0, y.1)]
        .into_iter()
        .filter_map(|(slot, field, space)| space.map(|space| DemandedField { slot, field, space, shape: ColumnShape::Scalar }))
        .collect();
    let outputs: Vec<(&str, SpaceSlot, ColumnShape)> = columns
        .iter()
        .filter(|c| demand.iter().any(|d| d.slot == c.slot))
        .map(|c| (c.output.as_str(), c.slot, ColumnShape::Scalar))
        .collect();
    run_in_log_space(
        ctx,
        label,
        batch,
        &demand,
        FIT_NULL_KEYS,
        |mapped_batch, cols, ctx| {
            let xcol = cols.column(SpaceSlot::X, x.0);
            let ycol = cols.column(SpaceSlot::Y, y.0);
            body(mapped_batch, &xcol, &ycol, ctx)
        },
        &outputs,
    )
}

/// `Smooth`'s and `Robust`'s facet pin for `x_range`
/// (`render::prepare::extent::fix_transform_extents_for_facet_impl`): the
/// whole pre-facet `batch`'s own `field` extent, in DATA units, under a
/// demanded `space` on `slot` — the mapped column's on-side extent in log
/// units ([`numeric_util::column_extent`] over the private column
/// [`with_mapped_slot`] maps), backward-mapped to data units at both
/// ends. Mirrors `bin::global_extent_in_space`'s identical shape for
/// `Bin`'s own `extent`, generalized here so the fits' facet pin shares
/// it instead of duplicating it per fit. `None` on a degenerate extent
/// (no on-side value, or a missing/all-tied column), exactly as
/// `column_extent` itself reports.
///
/// [`numeric_util::column_extent`]: crate::transform::numeric_util::column_extent
pub(crate) fn column_extent_in_space(
    batch: &RecordBatch,
    field: &str,
    slot: SpaceSlot,
    space: &SlotSpace,
    label: &str,
) -> Option<(f64, f64)> {
    let demand = DemandedField { slot, field, space, shape: ColumnShape::Scalar };
    let (mapped_batch, mapped_name) = with_mapped_slot(batch, &demand, label).ok()?;
    let (lo, hi) = column_extent(&mapped_batch, &mapped_name)?;
    Some((space.space.backward(lo), space.space.backward(hi)))
}

/// One column `name` holding `values` in each numeric Arrow type a
/// caller's frame can carry besides a nullable Float64 (Int64, Int32,
/// UInt8, Float32), and as a Float64 field declared non-nullable: the
/// fixtures every demanded statistic's tests run beside the same values in
/// a nullable Float64 column.
#[cfg(test)]
pub(crate) fn numeric_dtype_columns(name: &str, values: &[i64]) -> Vec<(&'static str, arrow::datatypes::Field, ArrayRef)> {
    use arrow::array::{Float32Array, Int32Array, Int64Array, UInt8Array};
    use arrow::datatypes::{DataType, Field};
    let as_u8 = |v: i64| u8::try_from(v).unwrap_or_else(|_| panic!("{v} does not fit UInt8"));
    vec![
        ("Int64", Field::new(name, DataType::Int64, true), Arc::new(Int64Array::from(values.to_vec())) as ArrayRef),
        ("Int32", Field::new(name, DataType::Int32, true), Arc::new(values.iter().map(|&v| v as i32).collect::<Int32Array>())),
        ("UInt8", Field::new(name, DataType::UInt8, true), Arc::new(values.iter().map(|&v| as_u8(v)).collect::<UInt8Array>())),
        ("Float32", Field::new(name, DataType::Float32, true), Arc::new(values.iter().map(|&v| v as f32).collect::<Float32Array>())),
        (
            "non-nullable Float64",
            Field::new(name, DataType::Float64, false),
            Arc::new(values.iter().map(|&v| v as f64).collect::<Float64Array>()),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scale::log::LogSide;
    use crate::spec::layer::PositionalAxis;

    fn positive() -> LogAxisSpace {
        LogAxisSpace { base: 10.0, side: LogSide::Positive }
    }

    fn negative() -> LogAxisSpace {
        LogAxisSpace { base: 10.0, side: LogSide::Negative }
    }

    /// `forward` is strictly increasing in its argument on both sides, so
    /// data order survives the map with no reordering step. Rejects the
    /// unsigned `log(|v|)` map, which decreases on the negative side.
    #[test]
    fn forward_is_increasing_on_both_sides() {
        let p = positive();
        let mut mapped: Vec<f64> = [1.0, 2.0, 5.0, 10.0, 100.0].into_iter().map(|v| p.forward(v).unwrap()).collect();
        let sorted = {
            let mut s = mapped.clone();
            s.sort_by(f64::total_cmp);
            s
        };
        assert_eq!(mapped, sorted, "positive side: {mapped:?}");

        let n = negative();
        mapped = [-100.0, -10.0, -5.0, -2.0, -1.0].into_iter().map(|v| n.forward(v).unwrap()).collect();
        let sorted = {
            let mut s = mapped.clone();
            s.sort_by(f64::total_cmp);
            s
        };
        assert_eq!(mapped, sorted, "negative side (ascending data order): {mapped:?}");
    }

    #[test]
    fn forward_and_backward_round_trip_without_reordering() {
        let n = negative();
        let (lo, hi) = n.forward_extent((-100.0, -1.0)).unwrap();
        assert!(lo <= hi, "forward_extent must keep lo <= hi: ({lo}, {hi})");
        let (a, b) = (n.backward(lo), n.backward(hi));
        assert_eq!((a, b), (-100.0, -1.0));
    }

    #[test]
    fn off_side_values_are_excluded_on_both_sides() {
        let p = positive();
        assert!(p.forward(0.0).is_none());
        assert!(p.forward(-1.0).is_none());
        let n = negative();
        assert!(n.forward(0.0).is_none());
        assert!(n.forward(1.0).is_none());
    }

    /// `extent_in_space` refuses a non-finite endpoint before its off-side
    /// check runs, at either end, for each of NaN/+inf/-inf — a
    /// regression test for this arm itself: were the finiteness check
    /// deleted, `off_side` itself returns `false` for a non-finite value
    /// (before its own `<= 0.0` reading even runs), and `forward` mirrors
    /// that by returning `None` for one, so all six cases would return
    /// `Ok(None)` instead of a refusal — the test would panic at
    /// `.err().unwrap_or_else(..)` ("must be refused, not accepted"), never
    /// reaching the text assertion below it.
    #[test]
    fn extent_in_space_refuses_a_non_finite_endpoint_at_either_end() {
        pyo3::Python::initialize();
        let p = positive();
        for (a, b) in [
            (f64::NAN, 8.0),
            (8.0, f64::NAN),
            (f64::INFINITY, 8.0),
            (8.0, f64::INFINITY),
            (f64::NEG_INFINITY, 8.0),
            (8.0, f64::NEG_INFINITY),
        ] {
            let err = p
                .extent_in_space("stat_smooth", "x_range", Some((a, b)))
                .err()
                .unwrap_or_else(|| panic!("({a}, {b}) must be refused, not accepted"));
            assert_eq!(
                err.to_string(),
                format!("ValueError: stat_smooth: x_range ({a}, {b}) must be two finite numbers"),
                "({a}, {b})"
            );
        }
    }

    fn x_slot() -> SlotSpace {
        SlotSpace { space: positive(), axes: vec![PositionalAxis::X], records_caller_input: true }
    }

    /// Mapping a demanded scalar slot over a column of any numeric type, or
    /// a non-nullable one, holding an off-side 0 keeps the caller's column
    /// exactly as it was (its field and values) and adds the slot's
    /// nullable Float64 column with the 0 null and every other value in log
    /// units. Rejects writing the mapped values under the caller's field,
    /// which Arrow refuses for any type but a nullable Float64.
    #[test]
    fn a_mapped_slot_keeps_the_callers_column_beside_its_own() {
        pyo3::Python::initialize();
        let space = x_slot();
        let demand = DemandedField { slot: SpaceSlot::X, field: "v", space: &space, shape: ColumnShape::Scalar };
        for (dtype, field, column) in numeric_dtype_columns("v", &[0, 1, 10, 100]) {
            let batch = RecordBatch::try_new(Arc::new(arrow::datatypes::Schema::new(vec![field.clone()])), vec![Arc::clone(&column)]).unwrap();
            let (mapped, name) = with_mapped_slot(&batch, &demand, "stat_test").unwrap_or_else(|e| panic!("{dtype}: {e}"));
            assert_eq!(name, "__log_space_x__v", "{dtype}");
            assert_eq!(mapped.schema().field(0), &field, "{dtype}: the caller's field");
            assert_eq!(mapped.column(0), &column, "{dtype}: the caller's values");
            let slot_field = mapped.schema().field(1).clone();
            assert_eq!((slot_field.name().as_str(), slot_field.data_type(), slot_field.is_nullable()), (name.as_str(), &arrow::datatypes::DataType::Float64, true), "{dtype}");
            let values = mapped.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
            assert_eq!(values.iter().collect::<Vec<_>>(), [None, Some(0.0), Some(1.0), Some(2.0)], "{dtype}");
        }
    }

    /// An input column already named like the slot's own column is refused
    /// in the step's voice rather than shadowed, and a missing caller
    /// column is refused naming it and the columns the batch holds.
    #[test]
    fn a_mapped_slot_refuses_a_colliding_or_missing_column() {
        pyo3::Python::initialize();
        let space = x_slot();
        let demand = DemandedField { slot: SpaceSlot::X, field: "v", space: &space, shape: ColumnShape::Scalar };
        let batch = RecordBatch::try_from_iter([
            ("v", Arc::new(Float64Array::from(vec![1.0, 10.0])) as ArrayRef),
            ("__log_space_x__v", Arc::new(Float64Array::from(vec![5.0, 6.0])) as ArrayRef),
        ])
        .unwrap();
        let err = with_mapped_slot(&batch, &demand, "stat_test").unwrap_err().to_string();
        assert!(err.contains("stat_test: input column '__log_space_x__v' collides with the column a log axis computes in; rename the input column"), "{err}");
        let missing = DemandedField { field: "absent", ..demand };
        let err = with_mapped_slot(&batch, &missing, "stat_test").unwrap_err().to_string();
        assert!(err.contains("stat_test: column 'absent' not found; available: [\"v\", \"__log_space_x__v\"]"), "{err}");
    }

    /// Two slots reading one column map it once each, from the caller's
    /// untouched column, into columns of their own.
    #[test]
    fn two_slots_reading_one_column_map_it_once_each() {
        pyo3::Python::initialize();
        let batch = RecordBatch::try_from_iter([("v", Arc::new(Float64Array::from(vec![10.0, 100.0])) as ArrayRef)]).unwrap();
        let (x, y) = (x_slot(), SlotSpace { space: LogAxisSpace { base: 2.0, ..positive() }, axes: vec![PositionalAxis::Y], records_caller_input: true });
        let (batch, x_name) = with_mapped_slot(&batch, &DemandedField { slot: SpaceSlot::X, field: "v", space: &x, shape: ColumnShape::Scalar }, "t").unwrap();
        let (batch, y_name) = with_mapped_slot(&batch, &DemandedField { slot: SpaceSlot::Y, field: "v", space: &y, shape: ColumnShape::Scalar }, "t").unwrap();
        let read = |name: &str| batch.column_by_name(name).unwrap().as_any().downcast_ref::<Float64Array>().unwrap().values().to_vec();
        assert_eq!(read("v"), [10.0, 100.0]);
        assert_eq!(read(&x_name), [1.0, 2.0]);
        let y_values = read(&y_name);
        assert!((y_values[0] - 10f64.log2()).abs() < 1e-12 && (y_values[1] - 100f64.log2()).abs() < 1e-12, "{y_values:?}");
    }

    /// A list-shaped output mapped back on positions `start..end` maps only
    /// those elements of each list (a `Kde2D` `extent`'s two bounds on one
    /// axis) and copies the rest.
    #[test]
    fn a_list_positions_output_maps_only_its_positions() {
        let mut builder = ListBuilder::new(Float64Builder::new());
        builder.values().append_slice(&[0.0, 2.0, 5.0, 7.0]);
        builder.append(true);
        let batch = RecordBatch::try_from_iter([("extent", Arc::new(builder.finish()) as ArrayRef)]).unwrap();
        let mapped = backward_map_output(batch, "extent", ColumnShape::ListPositions(0, 2), &positive(), "t").unwrap();
        let list = mapped.column(0).as_any().downcast_ref::<ListArray>().unwrap().value(0);
        let values = list.as_any().downcast_ref::<Float64Array>().unwrap().values().to_vec();
        assert_eq!(values, [1.0, 100.0, 5.0, 7.0]);
    }

    // ---- the groups a log axis empties ----

    use crate::transform::context::{AllNullOutcome, LogEmptied, StepInstance, TransformWarning, Unfittable};

    /// `g` and `v`, `v` a nullable Float64.
    fn grouped(g: &[&str], v: &[Option<f64>]) -> RecordBatch {
        RecordBatch::try_from_iter([
            ("g", Arc::new(arrow::array::StringArray::from(g.to_vec())) as ArrayRef),
            ("v", Arc::new(Float64Array::from(v.to_vec())) as ArrayRef),
        ])
        .unwrap()
    }

    /// A context demanding `v` in the positive log space, grouped by `groupby`.
    fn demanding_v(groupby: &[&str]) -> TransformContext {
        TransformContext {
            groupby: groupby.iter().map(|g| g.to_string()).collect(),
            step: StepInstance::Chain(vec![1]),
            ..TransformContext::default()
        }
    }

    /// One row per group in first-appearance order, as a grouped statistic
    /// does: `g`, and `out` holding a default `1.0` in log units for every
    /// group (a `Kde` group with no usable value puts its grid there). Each
    /// group named in `degraded` raises the fit warning its own body raises.
    fn per_group_body(
        degraded: &'static [&'static str],
        interval: &'static [&'static str],
    ) -> impl FnOnce(&RecordBatch, &SlotColumns, &TransformContext) -> PyResult<RecordBatch> {
        move |mapped, _cols, ctx| {
            let g = mapped.column_by_name("g").unwrap().as_any().downcast_ref::<arrow::array::StringArray>().unwrap().clone();
            let mut seen: Vec<String> = Vec::new();
            for i in 0..g.len() {
                if !seen.iter().any(|s| s == g.value(i)) {
                    seen.push(g.value(i).to_string());
                }
            }
            for name in &seen {
                let label = format!("g={name}");
                if degraded.contains(&name.as_str()) {
                    ctx.degrade_group_or_refuse("Kde", Some(&label), Unfittable::new("Kde", "fewer than 2 usable (non-null/non-NaN) values"))?;
                }
                if interval.contains(&name.as_str()) {
                    ctx.warn(TransformWarning::IntervalUndefined {
                        transform: "Smooth".into(),
                        group: Some(label.clone()),
                        ci: 0.95,
                        reason: "no band".into(),
                        panel: None,
                        step: ctx.step.clone(),
                    });
                }
            }
            RecordBatch::try_from_iter([
                ("g", Arc::new(arrow::array::StringArray::from(seen.clone())) as ArrayRef),
                ("out", Arc::new(Float64Array::from(vec![1.0; seen.len()])) as ArrayRef),
            ])
            .map_err(|e| PyValueError::new_err(e.to_string()))
        }
    }

    fn run_per_group(
        ctx: &TransformContext,
        batch: &RecordBatch,
        degraded: &'static [&'static str],
        interval: &'static [&'static str],
    ) -> RecordBatch {
        pyo3::Python::initialize();
        let space = x_slot();
        let demand = [DemandedField { slot: SpaceSlot::Field, field: "v", space: &space, shape: ColumnShape::Scalar }];
        run_in_log_space(ctx, "stat_test", batch, &demand, NullKeyPolicy::Skip, per_group_body(degraded, interval), &[("out", SpaceSlot::Field, ColumnShape::Scalar)])
            .unwrap()
    }

    fn out_by_group(result: &RecordBatch) -> Vec<(String, f64)> {
        let g = result.column_by_name("g").unwrap().as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
        let out = result.column_by_name("out").unwrap().as_any().downcast_ref::<Float64Array>().unwrap();
        (0..result.num_rows()).map(|i| (g.value(i).to_string(), out.value(i))).collect()
    }

    fn fit_groups(ctx: &TransformContext) -> Vec<String> {
        ctx.warnings
            .borrow()
            .iter()
            .filter_map(|w| match w {
                TransformWarning::DegenerateFit { group, .. } | TransformWarning::IntervalUndefined { group, .. } => group.clone(),
                _ => None,
            })
            .collect()
    }

    /// A group whose every value is off the axis's side is recorded under
    /// its label, its default output is blanked (so no phantom value
    /// reaches the axis), and the fit warning its body raised for it is
    /// dropped; a group with values on the side keeps its output and its
    /// own warnings. Rejects leaving the group's `DegenerateFit` beside the
    /// axis's own report, or leaving its default grid at the axis.
    #[test]
    fn a_group_wholly_off_the_side_is_recorded_blanked_and_loses_its_fit_warnings() {
        let batch = grouped(&["a", "a", "c", "c", "c"], &[Some(10.0), Some(100.0), Some(-1.0), Some(0.0), Some(-3.0)]);
        let ctx = demanding_v(&["g"]);
        let result = run_per_group(&ctx, &batch, &["a", "c"], &["a", "c"]);
        assert_eq!(*ctx.log_emptied.borrow(), vec![LogEmptied { panel: None, groups: Some(vec!["g=c".to_string()]) }]);
        assert_eq!(fit_groups(&ctx), ["g=a", "g=a"], "only the group the axis emptied loses its warnings");
        let out = out_by_group(&result);
        assert_eq!(out[0], ("a".to_string(), 10.0), "the on-side group maps back as before");
        assert_eq!(out[1].0, "c");
        assert!(out[1].1.is_nan(), "the emptied group's output takes no position: {out:?}");
    }

    /// A group of one null and off-side values is emptied by the axis; a
    /// group of nulls only, and a group with one value on the side, are
    /// not: each keeps its own fit warning and its output, and nothing is
    /// recorded for them. Rejects treating a null or a single on-side
    /// value as the axis's doing.
    #[test]
    fn only_a_group_with_an_off_side_value_and_none_on_the_side_is_emptied() {
        let batch = grouped(
            &["m", "m", "m", "n", "n", "p", "p"],
            &[None, Some(-1.0), Some(0.0), None, None, Some(-1.0), Some(5.0)],
        );
        let ctx = demanding_v(&["g"]);
        let result = run_per_group(&ctx, &batch, &["m", "n", "p"], &[]);
        assert_eq!(*ctx.log_emptied.borrow(), vec![LogEmptied { panel: None, groups: Some(vec!["g=m".to_string()]) }]);
        assert_eq!(fit_groups(&ctx), ["g=n", "g=p"]);
        let out = out_by_group(&result);
        assert!(out[0].1.is_nan(), "{out:?}");
        assert_eq!((out[1].1, out[2].1), (10.0, 10.0), "the groups the axis did not empty keep their output");
    }

    /// A call with no group emptied is byte-identical to the body's own
    /// output and records nothing: its warnings stand, its columns are the
    /// body's mapped back, and the sink stays empty. Rejects a seam that
    /// records or filters on a chart with nothing off-side.
    #[test]
    fn a_call_no_group_of_which_is_emptied_records_nothing_and_filters_nothing() {
        let batch = grouped(&["a", "a", "b"], &[Some(10.0), Some(-1.0), Some(100.0)]);
        let ctx = demanding_v(&["g"]);
        let result = run_per_group(&ctx, &batch, &["b"], &["a"]);
        assert!(ctx.log_emptied.borrow().is_empty());
        assert_eq!(fit_groups(&ctx), ["g=a", "g=b"]);
        let out = out_by_group(&result);
        assert_eq!(out, [("a".to_string(), 10.0), ("b".to_string(), 10.0)]);
    }

    /// An ungrouped call wholly off the side keeps the cut: its output has
    /// no rows and a scratch sink takes the body's warnings, and nothing is
    /// recorded for the groups it does not have. Inside a facet partition
    /// the panel is also recorded as one the axis emptied, which
    /// [`AllNullOutcome::OffSide`] settles with no warning. Rejects running
    /// an ungrouped fit on a frame the axis refuses.
    #[test]
    fn an_ungrouped_call_wholly_off_the_side_is_cut_and_records_its_panel() {
        let batch = grouped(&["a", "a"], &[Some(-1.0), Some(0.0)]);
        let ctx = demanding_v(&[]);
        let result = run_per_group(&ctx, &batch, &["a"], &[]);
        assert_eq!(result.num_rows(), 0);
        assert!(ctx.warnings.borrow().is_empty(), "the body's own warnings go to a scratch sink");
        assert!(ctx.log_emptied.borrow().is_empty());
        assert!(ctx.all_null_partitions.borrow().is_empty(), "a whole-frame call is not a partition");

        let partition = TransformContext { is_partition: true, ..demanding_v(&[]) };
        let result = run_per_group(&partition, &batch, &["a"], &[]);
        assert_eq!(result.num_rows(), 0);
        let recorded = partition.all_null_partitions.borrow();
        assert_eq!(recorded.len(), 1);
        assert!(matches!(recorded[0].outcome, AllNullOutcome::OffSide));
        assert!(recorded[0].outcome.empties());
        assert_eq!(recorded[0].step, StepInstance::Chain(vec![1]));
    }

    /// A grouped call wholly off the side is not cut: its body runs and
    /// every group is blanked, so the axis finds nothing on its side and
    /// refuses in its own voice, while each group keeps its row and its
    /// place. Rejects cutting a grouped call to zero rows, which loses every
    /// group's key.
    #[test]
    fn a_grouped_call_wholly_off_the_side_blanks_every_group_and_keeps_its_rows() {
        let batch = grouped(&["a", "b"], &[Some(-1.0), Some(0.0)]);
        let ctx = TransformContext { is_partition: true, ..demanding_v(&["g"]) };
        let result = run_per_group(&ctx, &batch, &["a", "b"], &[]);
        let out = out_by_group(&result);
        assert_eq!(out.iter().map(|(g, _)| g.as_str()).collect::<Vec<_>>(), ["a", "b"]);
        assert!(out.iter().all(|(_, v)| v.is_nan()), "{out:?}");
        assert_eq!(
            *ctx.log_emptied.borrow(),
            vec![LogEmptied { panel: None, groups: Some(vec!["g=a".to_string(), "g=b".to_string()]) }]
        );
        assert!(fit_groups(&ctx).is_empty());
        assert!(ctx.all_null_partitions.borrow().is_empty(), "a grouped call keeps its rows, so no panel is cut");
    }

    /// An emptied group's list-shaped output (a `Kde2D` grid axis, or the
    /// positions of its `extent`) is blanked element by element at the
    /// positions the axis's units hold, and the other positions stay.
    #[test]
    fn an_emptied_groups_list_output_is_blanked_at_its_positions() {
        pyo3::Python::initialize();
        let batch = grouped(&["a", "c"], &[Some(10.0), Some(-1.0)]);
        let ctx = demanding_v(&["g"]);
        let space = x_slot();
        let demand = [DemandedField { slot: SpaceSlot::Field, field: "v", space: &space, shape: ColumnShape::Scalar }];
        let body = |mapped: &RecordBatch, _: &SlotColumns, _: &TransformContext| -> PyResult<RecordBatch> {
            let mut builder = ListBuilder::new(Float64Builder::new());
            for _ in 0..mapped.num_rows() {
                builder.values().append_slice(&[1.0, 2.0, 7.0]);
                builder.append(true);
            }
            RecordBatch::try_from_iter([
                ("g", Arc::clone(mapped.column_by_name("g").unwrap())),
                ("extent", Arc::new(builder.finish()) as ArrayRef),
            ])
            .map_err(|e| PyValueError::new_err(e.to_string()))
        };
        let result = run_in_log_space(&ctx, "stat_test", &batch, &demand, NullKeyPolicy::Skip, body, &[("extent", SpaceSlot::Field, ColumnShape::ListPositions(0, 2))]).unwrap();
        let lists = result.column_by_name("extent").unwrap().as_any().downcast_ref::<ListArray>().unwrap();
        let row = |i: usize| lists.value(i).as_any().downcast_ref::<Float64Array>().unwrap().values().to_vec();
        assert_eq!(row(0), [10.0, 100.0, 7.0]);
        let blanked = row(1);
        assert!(blanked[0].is_nan() && blanked[1].is_nan() && blanked[2] == 7.0, "{blanked:?}");
    }

    // ---- the null-key policy the seam shares with the body ----

    /// `g` (nullable text) and `v`.
    fn keyed(g: &[Option<&str>], v: &[Option<f64>]) -> RecordBatch {
        RecordBatch::try_from_iter([
            ("g", Arc::new(arrow::array::StringArray::from(g.to_vec())) as ArrayRef),
            ("v", Arc::new(Float64Array::from(v.to_vec())) as ArrayRef),
        ])
        .unwrap()
    }

    /// A body that groups by `g` under `policy` exactly as the statistics do
    /// (the shared partitioner), one output row per group holding `out` `1.0`,
    /// and a `DegenerateFit` for each group in `degraded` ("g=null" names the
    /// null key); when the policy fits the whole batch because every key is
    /// null, the one pooled row carries a null key and its fit warning names
    /// no group.
    fn keyed_body(
        policy: NullKeyPolicy,
        degraded: &'static [&'static str],
    ) -> impl FnOnce(&RecordBatch, &SlotColumns, &TransformContext) -> PyResult<RecordBatch> {
        move |mapped, _cols, ctx| {
            let groupby = vec!["g".to_string()];
            let partition = group_partition_multi(mapped, &groupby, "stat_test", policy, ctx)?;
            let mut keys: Vec<Option<String>> = Vec::new();
            for (key, _) in &partition.groups {
                let label = describe_group_keys(&groupby, key);
                if degraded.contains(&label.as_str()) {
                    ctx.degrade_group_or_refuse("Kde", Some(&label), Unfittable::new("Kde", "too few"))?;
                }
                keys.push(match &label[2..] {
                    "null" => None,
                    name => Some(name.to_string()),
                });
            }
            if keys.is_empty() && mapped.num_rows() > 0 {
                ctx.warn(TransformWarning::DegenerateFit {
                    transform: "Smooth".into(),
                    group: None,
                    reason: "too few".into(),
                    panel: None,
                    step: ctx.step.clone(),
                });
                keys.push(None);
            }
            RecordBatch::try_from_iter([
                ("g", Arc::new(arrow::array::StringArray::from(keys.clone())) as ArrayRef),
                ("out", Arc::new(Float64Array::from(vec![1.0; keys.len()])) as ArrayRef),
            ])
            .map_err(|e| PyValueError::new_err(e.to_string()))
        }
    }

    fn run_keyed(
        ctx: &TransformContext,
        batch: &RecordBatch,
        policy: NullKeyPolicy,
        degraded: &'static [&'static str],
    ) -> RecordBatch {
        pyo3::Python::initialize();
        let space = x_slot();
        let demand = [DemandedField { slot: SpaceSlot::Field, field: "v", space: &space, shape: ColumnShape::Scalar }];
        run_in_log_space(ctx, "stat_test", batch, &demand, policy, keyed_body(policy, degraded), &[("out", SpaceSlot::Field, ColumnShape::Scalar)]).unwrap()
    }

    fn out_values(result: &RecordBatch) -> Vec<f64> {
        result.column_by_name("out").unwrap().as_any().downcast_ref::<Float64Array>().unwrap().values().to_vec()
    }

    /// A step that pools its null-keyed rows into a group of their own (the
    /// ECDF window) has that group emptied by the axis when it holds only
    /// off-side values: its label is `g=null`, recorded, its warning dropped
    /// and its output blanked. Under a step that skips null keys the same
    /// rows form no group, so nothing is recorded for them. Rejects forming
    /// the groups with a policy other than the body's.
    #[test]
    fn the_seam_forms_groups_with_the_null_key_policy_of_the_body() {
        let batch = keyed(&[Some("a"), Some("a"), None, None], &[Some(10.0), Some(100.0), Some(-1.0), Some(0.0)]);
        let pooled = demanding_v(&["g"]);
        let result = run_keyed(&pooled, &batch, NullKeyPolicy::Pool, &["g=null"]);
        assert_eq!(*pooled.log_emptied.borrow(), vec![LogEmptied { panel: None, groups: Some(vec!["g=null".to_string()]) }]);
        assert!(fit_groups(&pooled).is_empty(), "the pooled null-key group's fit warning is the axis's to explain");
        let out = out_values(&result);
        assert_eq!(out[0], 10.0);
        assert!(out[1].is_nan(), "{out:?}");

        let skipped = demanding_v(&["g"]);
        let result = run_keyed(&skipped, &batch, NullKeyPolicy::Skip, &[]);
        assert!(skipped.log_emptied.borrow().is_empty(), "a skipped null key forms no group to empty");
        assert_eq!(out_values(&result), [10.0]);
    }

    /// A fit that falls back to the whole batch because every key is null
    /// treats that batch as the one group: when it holds only off-side
    /// values, the batch is recorded as emptied (no group named), the fit's
    /// own warning for it is dropped, its output is blanked, and the
    /// all-null-key warning (the caller's null key, a separate fact) stays.
    /// A fallback batch with a value on the side keeps its fit warning.
    #[test]
    fn a_whole_batch_fallback_emptied_by_the_axis_is_recorded_as_the_whole_call() {
        let batch = keyed(&[None, None, None], &[Some(-1.0), Some(0.0), Some(-3.0)]);
        let ctx = demanding_v(&["g"]);
        let result = run_keyed(&ctx, &batch, FIT_NULL_KEYS, &[]);
        assert_eq!(*ctx.log_emptied.borrow(), vec![LogEmptied { panel: None, groups: None }]);
        let warnings = ctx.warnings.borrow();
        assert!(!warnings.iter().any(|w| matches!(w, TransformWarning::DegenerateFit { .. })), "{warnings:?}");
        assert!(warnings.iter().any(|w| matches!(w, TransformWarning::GroupKeyAllNull { .. })), "the null key is still reported: {warnings:?}");
        assert!(out_values(&result).iter().all(|v| v.is_nan()));

        let batch = keyed(&[None, None, None], &[Some(-1.0), Some(5.0), None]);
        let ctx = demanding_v(&["g"]);
        run_keyed(&ctx, &batch, FIT_NULL_KEYS, &[]);
        assert!(ctx.log_emptied.borrow().is_empty());
        assert!(ctx.warnings.borrow().iter().any(|w| matches!(w, TransformWarning::DegenerateFit { group: None, .. })));
    }

    /// A fit warning raised before the call (a sibling facet panel's, which
    /// shares the run's sink) for a group label the call also empties is not
    /// the call's to drop. Rejects a drop that reads the whole sink.
    #[test]
    fn a_fit_warning_raised_before_the_call_survives_the_drop() {
        let batch = grouped(&["a", "c"], &[Some(10.0), Some(-1.0)]);
        let ctx = demanding_v(&["g"]);
        ctx.degrade_group_or_refuse("Kde", Some("g=c"), Unfittable::new("Kde", "an earlier panel's reason")).unwrap();
        run_per_group(&ctx, &batch, &["c"], &[]);
        assert_eq!(fit_groups(&ctx), ["g=c"], "the earlier panel's warning stays, this call's is dropped");
        assert_eq!(*ctx.log_emptied.borrow(), vec![LogEmptied { panel: None, groups: Some(vec!["g=c".to_string()]) }]);
    }

    /// The fit warning is dropped, and the group recorded, only for a group
    /// whose output rows were blanked: of two emptied groups, the one whose
    /// rows read a label other than the partition's keeps its warning and is
    /// not recorded, while the matched one is blanked, recorded and loses its
    /// warning, so the halves of one fact cannot disagree.
    #[test]
    fn a_group_whose_output_rows_cannot_be_matched_keeps_its_warning_and_is_not_recorded() {
        pyo3::Python::initialize();
        let batch = grouped(&["a", "b", "c"], &[Some(10.0), Some(-1.0), Some(-2.0)]);
        let ctx = demanding_v(&["g"]);
        let space = x_slot();
        let demand = [DemandedField { slot: SpaceSlot::Field, field: "v", space: &space, shape: ColumnShape::Scalar }];
        let body = |_: &RecordBatch, _: &SlotColumns, ctx: &TransformContext| -> PyResult<RecordBatch> {
            ctx.degrade_group_or_refuse("Kde", Some("g=b"), Unfittable::new("Kde", "too few"))?;
            ctx.degrade_group_or_refuse("Kde", Some("g=c"), Unfittable::new("Kde", "too few"))?;
            RecordBatch::try_from_iter([
                ("g", Arc::new(arrow::array::StringArray::from(vec!["a", "b", "renamed"])) as ArrayRef),
                ("out", Arc::new(Float64Array::from(vec![1.0, 1.0, 1.0])) as ArrayRef),
            ])
            .map_err(|e| PyValueError::new_err(e.to_string()))
        };
        let result = run_in_log_space(&ctx, "stat_test", &batch, &demand, NullKeyPolicy::Skip, body, &[("out", SpaceSlot::Field, ColumnShape::Scalar)]).unwrap();
        let out = out_values(&result);
        assert_eq!(out[0], 10.0);
        assert!(out[1].is_nan(), "the matched group is blanked: {out:?}");
        assert_eq!(out[2], 10.0, "the unmatched group keeps its output");
        assert_eq!(fit_groups(&ctx), ["g=c"], "only the matched group's warning is dropped");
        assert_eq!(*ctx.log_emptied.borrow(), vec![LogEmptied { panel: None, groups: Some(vec!["g=b".to_string()]) }]);
    }

    /// The fits' seam forms groups under the policy their partitions read
    /// ([`FIT_NULL_KEYS`]): a batch whose every key is null, which a fit
    /// fits whole, is judged as that batch, so a batch holding only off-side
    /// values is recorded as the whole call. Rejects handing the fits' seam
    /// a policy that drops null keys, which finds no group to empty.
    #[test]
    fn the_fits_seam_judges_an_all_null_key_batch_under_the_fit_policy() {
        pyo3::Python::initialize();
        let batch = keyed(&[None, None], &[Some(-1.0), Some(0.0)]);
        let ctx = demanding_v(&["g"]);
        let space = x_slot();
        let columns = [SpaceColumn { output: "out".to_string(), slot: SpaceSlot::X, input: "v".to_string(), pass_through: false }];
        let body = |mapped: &RecordBatch, _: &str, _: &str, _: &TransformContext| -> PyResult<RecordBatch> { Ok(mapped.clone()) };
        run_fit_in_log_space(&ctx, "stat_test", &batch, &columns, ("v", Some(&space)), ("w", None), body).unwrap();
        assert_eq!(*ctx.log_emptied.borrow(), vec![LogEmptied { panel: None, groups: None }]);
    }
}
