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
use crate::transform::context::TransformContext;
use crate::transform::core::SpaceSlot;
use crate::transform::numeric_util::coerce_to_float64;
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
    /// `count_scale_extent`, one axis's half of a `Kde2D`'s `extent`) in
    /// this space's units, for the step's arm to put into the spec its
    /// body reads. An endpoint off this space's side is
    /// refused in the step's own voice (`label`): malformed input for that
    /// scale, not silently rendered against an extent the axis can never
    /// place. `None` stays `None`.
    pub(crate) fn extent_in_space(&self, label: &str, extent: Option<(f64, f64)>) -> PyResult<Option<(f64, f64)>> {
        let Some((a, b)) = extent else { return Ok(None) };
        if let Some(v) = [a, b].into_iter().find(|&v| self.off_side(v)) {
            return Err(PyValueError::new_err(format!(
                "{label}: extent ({a}, {b}) names {v}, which a LogScale cannot place (it is zero or \
                 on the wrong side); state both endpoints on the scale's own side, or drop extent="
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
        Some(positions) => (demand.field.to_string(), map_list(caller, positions, |v| space.forward(v), demand.field, label)?),
    };
    let mapped = with_column(batch, &name, column).map_err(|e| PyValueError::new_err(format!("{label}: {e}")))?;
    Ok((mapped, name))
}

/// `column`, a `List<Float64>` with one list per row, with each list's
/// elements at `positions` mapped through `map` (`None` becomes null) and
/// every other element copied. A column of any other type is returned
/// unchanged.
fn map_list(column: &ArrayRef, positions: Range<usize>, map: impl Fn(f64) -> Option<f64>, name: &str, label: &str) -> PyResult<ArrayRef> {
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
            let v = if positions.contains(&position) { v.and_then(&map) } else { v };
            builder.values().append_option(v);
        }
        builder.append(true);
    }
    Ok(Arc::new(builder.finish()))
}

/// `result` with its `name` column mapped back to data units through
/// [`LogAxisSpace::backward`], keeping the column's own field (the output
/// schema stays the data-space body's): a scalar Float64 output (bin edges,
/// a kde grid, contour vertices) or a list one (a `Kde2D` grid axis, or
/// its `extent` positions). A null entry stays null; an absent column, or
/// a column of another type, is left as it is, so one output list serves
/// every route of a step (`per_row` and the aggregating route).
fn backward_map_output(result: RecordBatch, name: &str, shape: ColumnShape, space: &LogAxisSpace, label: &str) -> PyResult<RecordBatch> {
    let schema = result.schema();
    let Ok(idx) = schema.index_of(name) else { return Ok(result) };
    let column = result.column(idx);
    let mapped: ArrayRef = match shape.positions() {
        None => {
            let Some(values) = column.as_any().downcast_ref::<Float64Array>() else { return Ok(result) };
            Arc::new(values.iter().map(|v| v.map(|v| space.backward(v))).collect::<Float64Array>())
        }
        Some(positions) => map_list(column, positions, |v| Some(space.backward(v)), name, label)?,
    };
    let mut columns = result.columns().to_vec();
    columns[idx] = mapped;
    RecordBatch::try_new(schema, columns).map_err(|e| PyValueError::new_err(format!("{label}: {e}")))
}

/// Whether `column` holds no on-side value and at least one off-side one:
/// a demanded step's own body degrades (a flat `DegenerateFit`, an empty
/// grid) over data its axis already refuses to place, restating a warning
/// the axis's own `LogScaleNoValueOnSide` refusal already gives in
/// `mark_point`'s own voice. `false` for a column with nothing off-side,
/// and for a column with nothing on OR off side (already null/NaN — an
/// ordinary missing-data case unrelated to this axis's side).
fn all_off_side(column: &Float64Array, space: &LogAxisSpace) -> bool {
    let mut finite = column.iter().flatten().filter(|v| v.is_finite()).peekable();
    finite.peek().is_some() && finite.all(|v| space.off_side(v))
}

/// `ctx` with a fresh, independent warnings sink: every other shared sink
/// (`log_space`, `color_bins`, `run_plan`, `log_input`, `all_null_partitions`)
/// stays shared with `ctx` (an `Rc::clone`, not a deep copy), but a warning
/// pushed through the returned context is never read back by the pipeline
/// runner that owns `ctx`'s own sink.
fn with_scratch_warnings(ctx: &TransformContext) -> TransformContext {
    TransformContext { warnings: Rc::new(RefCell::new(Vec::new())), ..ctx.clone() }
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
/// [`SlotSpace::records_caller_input`] says to), notes whether a scalar
/// slot's column holds no on-side value and at least one off-side one, and
/// maps the slot ([`with_mapped_slot`]). `body` then runs on the mapped
/// batch with each slot's column name ([`SlotColumns`]) — on a zero-row
/// slice under a scratch warnings sink when a slot is all off-side, its
/// output cut to zero rows too, so the step neither restates a degenerate
/// fit its axis already refuses nor leaves a phantom on-side value (a
/// `Kde`'s default grid mapped back) for the axis to read. Each of
/// `outputs` (an output column, the slot whose units it holds, and its
/// shape) is mapped back to data units in that slot's space, and every
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
    body: impl FnOnce(&RecordBatch, &SlotColumns, &TransformContext) -> PyResult<RecordBatch>,
    outputs: &[(&str, SpaceSlot, ColumnShape)],
) -> PyResult<RecordBatch> {
    let mut mapped_batch = batch.clone();
    let mut columns = SlotColumns::default();
    let mut degenerate = false;
    for d in demand {
        if d.shape == ColumnShape::Scalar {
            let caller = caller_column(batch, d.field, label)?;
            if d.space.records_caller_input {
                for axis in &d.space.axes {
                    ctx.note_log_input(axis, d.field, Arc::clone(caller));
                }
            }
            degenerate |= all_off_side(&coerce_to_float64(caller, label, d.field)?, &d.space.space);
        }
        let (next, name) = with_mapped_slot(&mapped_batch, d, label)?;
        mapped_batch = next;
        if name != d.field {
            columns.private.push(name.clone());
        }
        columns.columns.push((d.slot, name));
    }
    let mut result = if degenerate {
        body(&mapped_batch.slice(0, 0), &columns, &with_scratch_warnings(ctx))?.slice(0, 0)
    } else {
        body(&mapped_batch, &columns, ctx)?
    };
    for (output, slot, shape) in outputs {
        let Some(d) = demand.iter().find(|d| d.slot == *slot) else { continue };
        result = backward_map_output(result, output, *shape, &d.space.space, label)?;
    }
    let schema = result.schema();
    let kept: Vec<usize> = (0..schema.fields().len()).filter(|&i| !columns.private.contains(schema.field(i).name())).collect();
    if kept.len() < schema.fields().len() {
        result = result.project(&kept).map_err(|e| PyValueError::new_err(format!("{label}: {e}")))?;
    }
    Ok(result)
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

    fn x_slot() -> SlotSpace {
        SlotSpace { space: positive(), axes: vec!["x"], records_caller_input: true }
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
        let (x, y) = (x_slot(), SlotSpace { space: LogAxisSpace { base: 2.0, ..positive() }, axes: vec!["y"], records_caller_input: true });
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
}
