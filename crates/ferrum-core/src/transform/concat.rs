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
//! read by name, padded with `new_null_array` at the field's own dtype**.
//!
//! Both call sites reach here for the same reason: a degenerate fit on one
//! slice (rank-deficient design, zero-variance x, non-convergence) emits a
//! narrower batch than its siblings, and a plain `concat_batches` would
//! either error or misalign columns POSITIONALLY, surfacing as a confusing
//! message naming an unrelated column.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, RecordBatch};
use arrow::datatypes::{Field, Schema};

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

    let mut out_cols: Vec<ArrayRef> = Vec::with_capacity(out_fields.len());
    for field in &out_fields {
        let mut parts: Vec<ArrayRef> = Vec::with_capacity(batches.len());
        for b in &batches {
            match b.schema().index_of(field.name()) {
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
}
