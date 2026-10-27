//! Schema-tolerant `RecordBatch` concatenation — the one reconciliation
//! algorithm for stacking batches that carry the same *logical* output under
//! diverging schemas.
//!
//! Two sites need this and, before R1, had two different implementations of
//! it (design finding 3): `render::prepare` concatenating a named output
//! across facet panels, and `transform::smooth::apply_grouped` concatenating
//! per-group fit batches. The older of the two took the single WIDEST
//! schema — which can arbitrarily drop a field that only a narrower batch
//! carries when two batches tie on field count — and hand-rolled padding for
//! `Float64`/`Utf8` only, hard-erroring on any third dtype. This module owns
//! the surviving algorithm: **union of fields in first-appearance order,
//! read by name, padded with `new_null_array` at the field's own dtype**,
//! with one further rule for a field whose dtype itself disagrees across
//! batches: when every dtype it appears as is numeric (`render::arrow_cast::
//! supported_numeric_dtype`), the union reads it as `Float64` through
//! `render::arrow_cast::col_as_f64` — the one value every positional-scale
//! reader already takes, so this reconciliation can never narrow a value
//! (an `Int64` batch's dtype truncating a `Float64` sibling's). A `Timestamp`
//! of one unit beside `Float64` or an integer dtype reconciles the same way,
//! `col_as_f64` reading its epoch value in that one unit; `Timestamp`s of
//! TWO OR MORE DIFFERENT units disagreeing refuse instead — `col_as_f64`
//! reads each column's raw epoch count in its own unit, so widening a
//! millisecond column's values beside a second column's (read in ITS unit)
//! would put the two a thousandfold apart rather than at the same instant.
//! Any other dtype disagreement (a non-numeric one, or numeric with
//! disagreeing `Timestamp` units) still refuses, naming the column, exactly
//! as a plain `arrow::compute::concat` would.
//!
//! Both call sites reach here for the same reason: a degenerate fit on one
//! slice (rank-deficient design, zero-variance x, non-convergence) emits a
//! narrower batch than its siblings, and a plain `concat_batches` would
//! either error or misalign columns POSITIONALLY, surfacing as a confusing
//! message naming an unrelated column. A third reason reaches the numeric
//! rule specifically: a continuous position-adjustment preview (`Stack`,
//! `Dodge`, `Jitter`) whose panels drew different dtypes for the same field
//! — a `Dodge` whose no-op panel kept its input `Int64`/`Float32`/
//! `Timestamp` column untouched while a sibling panel's own dodge retyped
//! to `Float64` (`render::position::set_float64_column`).

use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};

/// Concatenate `batches` into one, tolerating schema divergence between them.
///
/// The reference schema is the UNION of every batch's fields in
/// first-appearance order. Each field is read from every batch BY NAME; a
/// batch missing a field is padded with a null array of that field's own
/// dtype (`arrow::array::new_null_array`) rather than erroring or silently
/// misaligning by position. Dtype-generic via `arrow::compute::concat`, so
/// any field present in at least one batch can be padded — not just the
/// `Float64`/`Utf8` pair the deleted hand-rolled copy handled.
///
/// A field whose dtype disagrees across the batches that carry it, where
/// every one of those dtypes is numeric (module doc), reads as `Float64`
/// through `render::arrow_cast::col_as_f64` instead of its own batch's raw
/// column — the one widening that cannot lose a value. A `Timestamp` of one
/// unit widens the same way; `Timestamp`s of two or more different units
/// refuse instead of widening (module doc — `col_as_f64` would otherwise
/// read two instants a thousandfold apart as the same scale). Any other
/// disagreement is left at its first-seen dtype, so `arrow::compute::concat`
/// below refuses it unchanged, naming the column.
///
/// Every field in the union has its output nullability forced to `true`,
/// even if every batch that carries it declared it non-nullable, since a
/// batch missing any field is padded with nulls.
///
/// The common case — every batch already sharing one schema — takes Arrow's
/// own `concat_batches` unchanged, so a caller with no divergence produces
/// byte-identical output to a plain concatenation.
///
/// Errors carry `context` as their prefix (`"{context}: {cause}"`) so each
/// caller keeps its own error voice; the callers map the returned `String`
/// into their own error type.
pub(crate) fn concat_by_name(
    context: &str,
    batches: Vec<RecordBatch>,
) -> Result<RecordBatch, String> {
    let same_schema = batches.windows(2).all(|w| w[0].schema() == w[1].schema());
    if same_schema {
        let schema = batches[0].schema();
        return arrow::compute::concat_batches(&schema, &batches)
            .map_err(|e| format!("{context}: {e}"));
    }

    let mut out_fields: Vec<Field> = Vec::new();
    for b in &batches {
        for field in b.schema().fields().iter() {
            if !out_fields.iter().any(|f| f.name() == field.name()) {
                out_fields.push(Field::new(field.name(), field.data_type().clone(), true));
            }
        }
    }

    // A field whose dtype disagrees across every batch that carries it,
    // where every one of those dtypes is numeric, reconciles to `Float64`.
    // Tracked by name (rather than re-deriving "was this reconciled" from
    // the field's own dtype) so a field that is naturally `Float64`
    // everywhere, and a field whose first-seen dtype happens to be
    // `Float64` beside a genuinely incompatible (non-numeric) sibling,
    // are never confused with one that was actually widened here.
    let mut reconciled: HashSet<String> = HashSet::new();
    for out_field in &mut out_fields {
        let dtypes: Vec<DataType> = batches
            .iter()
            .filter_map(|b| b.schema().column_with_name(out_field.name()).map(|(_, f)| f.data_type().clone()))
            .collect();
        let disagrees = dtypes.iter().any(|d| d != out_field.data_type());
        if disagrees && dtypes.iter().all(crate::render::arrow_cast::supported_numeric_dtype) {
            let mut units: Vec<arrow::datatypes::TimeUnit> = dtypes
                .iter()
                .filter_map(|d| match d {
                    DataType::Timestamp(unit, _) => Some(*unit),
                    _ => None,
                })
                .collect();
            units.sort_by_key(|u| format!("{u:?}"));
            units.dedup();
            if units.len() > 1 {
                let unit_list: Vec<String> = units.iter().map(|u| format!("{u:?}")).collect();
                return Err(format!(
                    "{context}: column '{name}': Timestamp columns of different units ({units}) \
                     cannot be concatenated as one numeric column; their epoch counts are not on \
                     the same scale — cast every part of '{name}' to one datetime unit",
                    name = out_field.name(),
                    units = unit_list.join(", "),
                ));
            }
            *out_field = Field::new(out_field.name(), DataType::Float64, true);
            reconciled.insert(out_field.name().clone());
        }
    }

    let mut out_cols: Vec<ArrayRef> = Vec::with_capacity(out_fields.len());
    for field in &out_fields {
        let mut parts: Vec<ArrayRef> = Vec::with_capacity(batches.len());
        for b in &batches {
            match b.schema().index_of(field.name()) {
                Ok(_) if reconciled.contains(field.name()) => {
                    let vals = crate::render::arrow_cast::col_as_f64(b, field.name())
                        .map_err(|e| format!("{context}: column '{}': {e}", field.name()))?;
                    parts.push(Arc::new(arrow::array::Float64Array::from(vals)) as ArrayRef);
                }
                Ok(fi) => parts.push(Arc::clone(b.column(fi))),
                Err(_) => parts.push(arrow::array::new_null_array(field.data_type(), b.num_rows())),
            }
        }
        let refs: Vec<&dyn Array> = parts.iter().map(|a| a.as_ref()).collect();
        let combined = arrow::compute::concat(&refs)
            .map_err(|e| format!("{context}: column '{}': {e}", field.name()))?;
        out_cols.push(combined);
    }

    RecordBatch::try_new(Arc::new(Schema::new(out_fields)), out_cols)
        .map_err(|e| format!("{context}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::DataType;

    fn batch(fields: &[(&str, ArrayRef)]) -> RecordBatch {
        let schema = Arc::new(Schema::new(
            fields
                .iter()
                .map(|(n, a)| Field::new(*n, a.data_type().clone(), true))
                .collect::<Vec<_>>(),
        ));
        RecordBatch::try_new(schema, fields.iter().map(|(_, a)| Arc::clone(a)).collect()).unwrap()
    }

    fn f64_col(vals: &[f64]) -> ArrayRef {
        Arc::new(Float64Array::from(vals.to_vec()))
    }

    #[test]
    fn identical_schemas_take_the_plain_concat_path() {
        let a = batch(&[("x", f64_col(&[1.0, 2.0]))]);
        let b = batch(&[("x", f64_col(&[3.0]))]);
        let out = concat_by_name("ctx", vec![a.clone(), b]).unwrap();
        assert_eq!(out.schema(), a.schema());
        assert_eq!(out.num_rows(), 3);
    }

    /// The property the deleted widest-schema pick could not hold: two
    /// batches TIE on field count while each carries a field the other
    /// lacks, so no single batch's schema is a valid reference.
    #[test]
    fn a_tie_on_field_count_still_keeps_every_field() {
        let a = batch(&[("x", f64_col(&[1.0])), ("only_a", f64_col(&[10.0]))]);
        let b = batch(&[("x", f64_col(&[2.0])), ("only_b", f64_col(&[20.0]))]);
        let out = concat_by_name("ctx", vec![a, b]).unwrap();
        let schema = out.schema();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, vec!["x", "only_a", "only_b"], "union, first-appearance order");
        assert_eq!(out.num_rows(), 2);
        let only_a = out.column(1);
        assert!(!only_a.is_null(0) && only_a.is_null(1), "the batch lacking a field pads null");
    }

    /// Dtype-generic: a dtype outside the `Float64`/`Utf8` pair the deleted
    /// copy hard-errored on reconciles like any other.
    #[test]
    fn a_third_dtype_reconciles_rather_than_erroring() {
        let a = batch(&[
            ("x", f64_col(&[1.0])),
            ("flag", Arc::new(arrow::array::BooleanArray::from(vec![true])) as ArrayRef),
        ]);
        let b = batch(&[("x", f64_col(&[2.0]))]);
        let out = concat_by_name("ctx", vec![a, b]).unwrap();
        assert_eq!(out.schema().field_with_name("flag").unwrap().data_type(), &DataType::Boolean);
        assert!(out.column(1).is_null(1));
    }

    #[test]
    fn utf8_and_float_pad_by_name_not_by_position() {
        let a = batch(&[
            ("y", f64_col(&[1.0])),
            ("label", Arc::new(StringArray::from(vec!["hi"])) as ArrayRef),
        ]);
        let b = batch(&[("y", f64_col(&[2.0]))]);
        let out = concat_by_name("ctx", vec![a, b]).unwrap();
        let labels = out.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(labels.value(0), "hi");
        assert!(labels.is_null(1));
    }

    #[test]
    fn errors_carry_the_callers_context_prefix() {
        // Same field name, incompatible dtypes across batches: `concat`
        // refuses, and the message must name the caller and the column.
        let a = batch(&[("v", f64_col(&[1.0]))]);
        let b = batch(&[
            ("v", Arc::new(StringArray::from(vec!["x"])) as ArrayRef),
            ("pad", f64_col(&[1.0])),
        ]);
        let err = concat_by_name("my caller", vec![a, b]).unwrap_err();
        assert!(err.starts_with("my caller: "), "error must carry the caller's prefix: {err}");
        assert!(err.contains("'v'"), "error must name the column: {err}");
    }

    // --- Numeric dtype disagreement reconciles to Float64, never narrows ---

    /// `Int64` beside `Float64`: the union is `Float64`, holding every
    /// value exactly — never the `Float64` side truncated into `Int64`,
    /// which a no-op `Dodge` panel's `Int64` schema taken as a cast
    /// target would do, turning a sibling panel's `2.25` into `2`.
    #[test]
    fn int64_and_float64_reconcile_to_float64_losslessly() {
        let a = batch(&[("v", Arc::new(arrow::array::Int64Array::from(vec![1, 2])) as ArrayRef)]);
        let b = batch(&[("v", f64_col(&[2.25, 0.75]))]);
        let out = concat_by_name("ctx", vec![a, b]).unwrap();
        assert_eq!(out.schema().field_with_name("v").unwrap().data_type(), &DataType::Float64);
        let v = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(v.values(), &[1.0, 2.0, 2.25, 0.75]);
    }

    /// `Float32` beside `Float64`: the `Float32` side widens to `Float64`
    /// without rounding, and the `Float64` side keeps its own bits exactly
    /// — casting to `Float32` instead would round a dodged `f64` position
    /// just past the previewed edge, for a `Float32` x.
    #[test]
    fn float32_and_float64_reconcile_without_rounding() {
        let a = batch(&[("v", Arc::new(arrow::array::Float32Array::from(vec![1.5_f32])) as ArrayRef)]);
        let b = batch(&[("v", f64_col(&[1.775]))]);
        let out = concat_by_name("ctx", vec![a, b]).unwrap();
        assert_eq!(out.schema().field_with_name("v").unwrap().data_type(), &DataType::Float64);
        let v = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(v.value(0), 1.5);
        assert_eq!(v.value(1).to_bits(), 1.775_f64.to_bits(), "the Float64 side keeps its own bits exactly");
    }

    /// `Timestamp(ms)` beside `Float64`: the timestamp's epoch-ms value
    /// reads as a plain `f64`, the same value `col_as_f64` gives a
    /// positional scale over a temporal axis, not its own "cast into the
    /// other side's dtype" value.
    #[test]
    fn timestamp_ms_and_float64_reconcile_to_the_epoch_value() {
        let ts = Arc::new(arrow::array::TimestampMillisecondArray::from(vec![1000_i64])) as ArrayRef;
        let a = batch(&[("v", ts)]);
        let b = batch(&[("v", f64_col(&[1000.5]))]);
        let out = concat_by_name("ctx", vec![a, b]).unwrap();
        assert_eq!(out.schema().field_with_name("v").unwrap().data_type(), &DataType::Float64);
        let v = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(v.values(), &[1000.0, 1000.5]);
    }

    /// `Timestamp(ms)` beside `Int64`: a single `Timestamp` unit still
    /// widens to `Float64`, reading the epoch value in that one unit —
    /// the refusal below is for TWO OR MORE different units, not for a
    /// `Timestamp` beside any other numeric dtype.
    #[test]
    fn timestamp_ms_beside_int64_still_reconciles_to_float64() {
        let ts = Arc::new(arrow::array::TimestampMillisecondArray::from(vec![1000_i64])) as ArrayRef;
        let a = batch(&[("v", ts)]);
        let b = batch(&[("v", Arc::new(arrow::array::Int64Array::from(vec![2000_i64])) as ArrayRef)]);
        let out = concat_by_name("ctx", vec![a, b]).unwrap();
        assert_eq!(out.schema().field_with_name("v").unwrap().data_type(), &DataType::Float64);
        let v = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(v.values(), &[1000.0, 2000.0]);
    }

    /// `Timestamp(ms)` beside `Timestamp(s)`: widening both to `Float64`
    /// through `col_as_f64` would read each column's raw epoch count in
    /// its OWN unit, putting the same instant a thousandfold apart
    /// (`1` second vs. `1` millisecond past the epoch). Refuses instead,
    /// naming the column and the caller's context, the same voice a
    /// non-numeric disagreement already uses.
    #[test]
    fn timestamp_different_units_refuses_naming_the_column() {
        use arrow::array::{TimestampMillisecondArray, TimestampSecondArray};
        let a = batch(&[("v", Arc::new(TimestampMillisecondArray::from(vec![1000_i64])) as ArrayRef)]);
        let b = batch(&[("v", Arc::new(TimestampSecondArray::from(vec![1_i64])) as ArrayRef)]);
        let err = concat_by_name("my caller", vec![a, b]).unwrap_err();
        assert!(err.starts_with("my caller: "), "error must carry the caller's prefix: {err}");
        assert!(err.contains("'v'"), "error must name the column: {err}");
        assert!(
            err.contains("cast every part of 'v' to one datetime unit"),
            "error must name the remedy: {err}"
        );
    }

    /// `UInt8` beside `Int64`: two DIFFERENT integer dtypes, neither of
    /// which is `Float64`, still reconcile to `Float64` — the rule is "all
    /// dtypes present are numeric", not "one side is already Float64".
    #[test]
    fn uint8_and_int64_reconcile_to_float64() {
        let a = batch(&[("v", Arc::new(arrow::array::UInt8Array::from(vec![1_u8, 2])) as ArrayRef)]);
        let b = batch(&[("v", Arc::new(arrow::array::Int64Array::from(vec![300_i64])) as ArrayRef)]);
        let out = concat_by_name("ctx", vec![a, b]).unwrap();
        assert_eq!(out.schema().field_with_name("v").unwrap().data_type(), &DataType::Float64);
        let v = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(v.values(), &[1.0, 2.0, 300.0]);
    }

    /// A non-numeric disagreement (`Utf8` vs. `Int64`) is NOT reconciled —
    /// it refuses, naming the column and the caller's context.
    #[test]
    fn utf8_and_int64_still_errors_naming_the_column() {
        let a = batch(&[("v", Arc::new(arrow::array::Int64Array::from(vec![1_i64])) as ArrayRef)]);
        let b = batch(&[("v", Arc::new(StringArray::from(vec!["x"])) as ArrayRef)]);
        let err = concat_by_name("my caller", vec![a, b]).unwrap_err();
        assert!(err.starts_with("my caller: "), "error must carry the caller's prefix: {err}");
        assert!(err.contains("'v'"), "error must name the column: {err}");
    }
}
