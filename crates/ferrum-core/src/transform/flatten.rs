//! Data transform: Flatten — expand list/array columns into separate rows.
//!
//! For each row, if the specified list columns contain N elements, the row
//! is replicated N times with the list elements distributed across rows.
//! Non-list columns are repeated.

use arrow::array::{
    Array, ArrayRef, LargeListArray, ListArray, RecordBatch, StringArray, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::ops::Range;
use std::sync::Arc;

use crate::transform::numeric_util::coerce_to_float64;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct FlattenSpec {
    /// Column names to flatten (must be list/array typed or semi-colon strings).
    pub fields: Vec<String>,
    /// Optional output names for flattened columns.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub as_: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

pub(crate) fn apply(spec: &FlattenSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let schema = batch.schema();
    let n_rows = batch.num_rows();

    if spec.fields.is_empty() {
        return Err(PyValueError::new_err("data_flatten: fields must be non-empty"));
    }

    // Validate fields exist.
    let field_indices: Vec<usize> = spec
        .fields
        .iter()
        .map(|f| {
            schema.index_of(f).map_err(|_| {
                PyValueError::new_err(format!("data_flatten: column '{f}' not found"))
            })
        })
        .collect::<PyResult<_>>()?;

    // Resolve every flatten field up front, so an unflattenable column is
    // refused by name before any expansion work happens.
    let sources: Vec<FlattenSource> = field_indices
        .iter()
        .enumerate()
        .map(|(fi, &i)| FlattenSource::resolve(batch.column(i).as_ref(), &spec.fields[fi]))
        .collect::<PyResult<_>>()?;

    // The number of elements per row comes from the first flatten field.
    let lengths = sources[0].row_lengths(n_rows);

    // Build expanded output for each column.
    let non_flatten_indices: Vec<usize> = (0..schema.fields().len())
        .filter(|i| !field_indices.contains(i))
        .collect();

    // Expand non-flatten columns by repeating.
    let mut out_cols: Vec<ArrayRef> = Vec::new();
    let mut out_fields: Vec<Field> = Vec::new();

    for &i in &non_flatten_indices {
        let col = batch.column(i);
        let expanded = repeat_by_lengths(col.as_ref(), schema.field(i).name(), &lengths)?;
        out_cols.push(expanded);
        out_fields.push(schema.field(i).as_ref().clone());
    }

    // Expand flatten columns by extracting elements.
    let as_names = spec.as_.clone().unwrap_or_else(|| spec.fields.clone());
    for (fi, source) in sources.iter().enumerate() {
        let (expanded, dtype) = source.flatten(&spec.fields[fi])?;
        let col_name = as_names.get(fi).unwrap_or(&spec.fields[fi]);
        out_cols.push(expanded);
        out_fields.push(Field::new(col_name, dtype, true));
    }

    let out_schema = Arc::new(Schema::new(out_fields));
    RecordBatch::try_new(out_schema, out_cols)
        .map_err(|e| PyValueError::new_err(format!("data_flatten: {e}")))
}

/// A flatten field's expansion source, resolved once per `apply`.
///
/// Two forms are flattenable: an Arrow list column (either offset width), whose
/// row `i` contributes the element range `ranges[i]` of `values`; and a string
/// column of `';'`-delimited values. Anything else is refused by name — batch-D
/// dtype policy (spec §4.1) bans the former all-null fallback.
enum FlattenSource {
    List {
        values: ArrayRef,
        ranges: Vec<Range<usize>>,
    },
    Delimited(StringArray),
}

impl FlattenSource {
    fn resolve(col: &dyn Array, field: &str) -> PyResult<Self> {
        if let Some(list) = col.as_any().downcast_ref::<ListArray>() {
            return Ok(FlattenSource::List {
                values: list.values().clone(),
                ranges: offset_ranges(list.value_offsets(), |o| o as usize, list),
            });
        }
        if let Some(list) = col.as_any().downcast_ref::<LargeListArray>() {
            return Ok(FlattenSource::List {
                values: list.values().clone(),
                ranges: offset_ranges(list.value_offsets(), |o| o as usize, list),
            });
        }
        if matches!(col.data_type(), DataType::Utf8 | DataType::LargeUtf8) {
            // LargeUtf8 (what polars emits) is cast to Utf8 so one split path serves both.
            let utf8 = cast_to_utf8(col, field)?;
            return Ok(FlattenSource::Delimited(utf8));
        }
        Err(PyValueError::new_err(format!(
            "data_flatten: column '{}' has dtype {:?}, which cannot be flattened; \
             expected a List column or a Utf8 column of ';'-delimited values",
            field,
            col.data_type()
        )))
    }

    /// Elements contributed by each input row (a null list row contributes none;
    /// a null or empty delimited cell keeps its row as one null element).
    fn row_lengths(&self, n_rows: usize) -> Vec<usize> {
        match self {
            FlattenSource::List { ranges, .. } => {
                (0..n_rows).map(|i| ranges[i].len()).collect()
            }
            FlattenSource::Delimited(arr) => (0..n_rows)
                .map(|i| {
                    if arr.is_null(i) || arr.value(i).is_empty() {
                        1
                    } else {
                        arr.value(i).split(';').count()
                    }
                })
                .collect(),
        }
    }

    /// Materialize the flattened column and the dtype to declare for it.
    ///
    /// List elements keep their own dtype for strings and widen to Float64 for
    /// any numeric dtype; any other element dtype is refused by name. A null
    /// element (as opposed to a `NaN` element) is preserved as an Arrow null
    /// through `take_rows` + `coerce_to_float64`, the same null-preserving
    /// delta as `repeat_by_lengths`'s carried columns — see its doc comment.
    /// The pre-fix `flatten_column` substituted `f64::NAN` for a null list
    /// element; this half of the delta was not disclosed when the carried-
    /// column half shipped.
    fn flatten(&self, field: &str) -> PyResult<(ArrayRef, DataType)> {
        match self {
            FlattenSource::List { values, ranges } => {
                let indices: Vec<u64> = ranges
                    .iter()
                    .flat_map(|r| r.clone())
                    .map(|i| i as u64)
                    .collect();
                let elements = take_rows(values.as_ref(), indices, field)?;
                match elements.data_type() {
                    DataType::Utf8 | DataType::LargeUtf8 => {
                        let dtype = elements.data_type().clone();
                        Ok((elements, dtype))
                    }
                    _ => {
                        let widened =
                            coerce_to_float64(&elements, "data_flatten", field).map_err(|_| {
                                PyValueError::new_err(format!(
                                    "data_flatten: column '{}' is a list of {:?}, which cannot be \
                                     flattened; supported element dtypes: Utf8 and any numeric \
                                     (int/uint/float)",
                                    field,
                                    elements.data_type()
                                ))
                            })?;
                        Ok((Arc::new(widened), DataType::Float64))
                    }
                }
            }
            FlattenSource::Delimited(arr) => {
                let mut out: Vec<Option<String>> = Vec::with_capacity(arr.len());
                for i in 0..arr.len() {
                    if arr.is_null(i) || arr.value(i).is_empty() {
                        out.push(None);
                    } else {
                        for part in arr.value(i).split(';') {
                            out.push(Some(part.trim().to_string()));
                        }
                    }
                }
                Ok((Arc::new(StringArray::from(out)), DataType::Utf8))
            }
        }
    }
}

/// Per-row `[start, end)` element ranges of a list array; null rows are empty.
fn offset_ranges<O: Copy>(
    offsets: &[O],
    to_usize: impl Fn(O) -> usize,
    list: &dyn Array,
) -> Vec<Range<usize>> {
    (0..list.len())
        .map(|i| {
            if list.is_null(i) {
                0..0
            } else {
                to_usize(offsets[i])..to_usize(offsets[i + 1])
            }
        })
        .collect()
}

fn cast_to_utf8(col: &dyn Array, field: &str) -> PyResult<StringArray> {
    let casted = arrow::compute::cast(col, &DataType::Utf8).map_err(|e| {
        PyValueError::new_err(format!(
            "data_flatten: could not read column '{field}' as Utf8: {e}"
        ))
    })?;
    casted
        .as_any()
        .downcast_ref::<StringArray>()
        .cloned()
        .ok_or_else(|| {
            PyValueError::new_err(format!("data_flatten: column '{field}' is not Utf8"))
        })
}

/// Gather `indices` out of `col`, preserving its dtype exactly.
fn take_rows(col: &dyn Array, indices: Vec<u64>, field: &str) -> PyResult<ArrayRef> {
    arrow::compute::take(col, &UInt64Array::from(indices), None).map_err(|e| {
        PyValueError::new_err(format!(
            "data_flatten: could not expand column '{}' of dtype {:?}: {e}",
            field,
            col.data_type()
        ))
    })
}

/// Repeat each row of a carried (non-flattened) column `lengths[i]` times.
///
/// Uses Arrow's `take` kernel so every dtype round-trips at its declared type —
/// carried columns keep the schema field they came in with. **Null-preserving
/// behavior delta**: `take` carries a null cell through as an Arrow null,
/// where the pre-fix `Float64`/`Utf8`-only implementation substituted
/// `f64::NAN` for a null carried `Float64` cell. A null and a `NaN` are
/// interchangeable for every numeric/positional consumer (`is_null`-or-
/// `is_nan` checks treat them the same), but they are *not* interchangeable
/// on a categorical channel (`:N`/`:O`) or in `Color`/`Key` encodings: a null
/// is dropped from the domain, while a `NaN` value becomes its own `"NaN"`
/// category. Verified null-preservation is the correct choice — it is what
/// the output schema already promised for a nullable field — and no golden
/// or existing test moved when this delta shipped.
fn repeat_by_lengths(col: &dyn Array, field: &str, lengths: &[usize]) -> PyResult<ArrayRef> {
    let indices: Vec<u64> = lengths
        .iter()
        .enumerate()
        .flat_map(|(i, &len)| std::iter::repeat_n(i as u64, len))
        .collect();
    take_rows(col, indices, field)
}

// ─── PyO3 wrapper ──────────────────────────────────────────────────────────

use pyo3::prelude::*;
use crate::transform::core::TransformSpec;

#[pyclass(module = "ferrum._core", name = "Flatten")]
#[derive(Debug, Clone)]
pub(crate) struct PyFlatten(pub(crate) TransformSpec);

#[pymethods]
impl PyFlatten {
    #[new]
    #[pyo3(signature = (fields, *, as_ = None, name = None))]
    fn new(fields: Vec<String>, as_: Option<Vec<String>>, name: Option<String>) -> PyResult<Self> {
        if fields.is_empty() {
            return Err(PyValueError::new_err("Flatten: fields must be non-empty"));
        }
        Ok(PyFlatten(TransformSpec::Flatten(FlattenSpec { fields, as_, name })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Flatten(s) => format!("Flatten(fields={:?})", s.fields),
            _ => "Flatten(?)".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{Field, Schema};
    use std::sync::Arc;

    #[test]
    fn flatten_semicolon_delimited_strings() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Float64, false),
            Field::new("tags", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
                Arc::new(StringArray::from(vec!["a;b;c", "d;e"])),
            ],
        )
        .unwrap();

        let spec = FlattenSpec {
            fields: vec!["tags".into()],
            as_: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        // Row 0 expands to 3, row 1 to 2 → 5 total rows.
        assert_eq!(out.num_rows(), 5);

        let ids = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(ids.value(0), 1.0);
        assert_eq!(ids.value(1), 1.0);
        assert_eq!(ids.value(2), 1.0);
        assert_eq!(ids.value(3), 2.0);
        assert_eq!(ids.value(4), 2.0);

        let tags = out.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(tags.value(0), "a");
        assert_eq!(tags.value(1), "b");
        assert_eq!(tags.value(2), "c");
        assert_eq!(tags.value(3), "d");
        assert_eq!(tags.value(4), "e");
    }

    // ---- Batch-D dtype policy (spec §4.1) ----

    /// Build a one-list-column batch alongside a Float64 `id` column.
    fn list_batch(inner: ArrayRef, offsets: Vec<i32>, inner_dtype: DataType) -> RecordBatch {
        let n_rows = offsets.len() - 1;
        let list = ListArray::new(
            Arc::new(Field::new("item", inner_dtype.clone(), true)),
            arrow::buffer::OffsetBuffer::new(offsets.into()),
            inner,
            None,
        );
        let ids: Vec<f64> = (0..n_rows).map(|i| i as f64).collect();
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Float64, false),
            Field::new(
                "vals",
                DataType::List(Arc::new(Field::new("item", inner_dtype, true))),
                true,
            ),
        ]));
        RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(ids)), Arc::new(list)],
        )
        .unwrap()
    }

    fn flatten_vals(batch: &RecordBatch) -> PyResult<RecordBatch> {
        apply(
            &FlattenSpec {
                fields: vec!["vals".into()],
                as_: None,
                name: None,
            },
            batch,
        )
    }

    #[test]
    fn flatten_list_of_utf8_emits_real_strings() {
        // The user guide's list-of-strings example: before the dtype policy this
        // produced an all-null Float64 column.
        let inner: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c"]));
        let batch = list_batch(inner, vec![0, 2, 3], DataType::Utf8);
        let out = flatten_vals(&batch).unwrap();

        assert_eq!(out.num_rows(), 3);
        assert_eq!(out.schema().field(1).data_type(), &DataType::Utf8);
        let vals = out.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(vals.value(0), "a");
        assert_eq!(vals.value(1), "b");
        assert_eq!(vals.value(2), "c");
    }

    #[test]
    fn flatten_list_of_int64_coerces_to_float64() {
        let inner: ArrayRef = Arc::new(arrow::array::Int64Array::from(vec![10_i64, 20, 30]));
        let batch = list_batch(inner, vec![0, 2, 3], DataType::Int64);
        let out = flatten_vals(&batch).unwrap();

        assert_eq!(out.schema().field(1).data_type(), &DataType::Float64);
        let vals = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(vals.values(), &[10.0, 20.0, 30.0]);
    }

    #[test]
    fn flatten_list_of_unsupported_inner_dtype_raises_naming_it() {
        pyo3::Python::initialize();
        let inner: ArrayRef = Arc::new(arrow::array::BooleanArray::from(vec![true, false, true]));
        let batch = list_batch(inner, vec![0, 2, 3], DataType::Boolean);
        let msg = flatten_vals(&batch).unwrap_err().to_string();

        assert!(msg.contains("data_flatten"), "err: {msg}");
        assert!(msg.contains("'vals'"), "err must name the column: {msg}");
        assert!(msg.contains("Boolean"), "err must name the dtype: {msg}");
    }

    #[test]
    fn flatten_non_flattenable_column_raises_instead_of_nulling() {
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![1.0, 2.0]))],
        )
        .unwrap();
        let spec = FlattenSpec {
            fields: vec!["v".into()],
            as_: None,
            name: None,
        };
        let msg = apply(&spec, &batch).unwrap_err().to_string();

        assert!(msg.contains("data_flatten"), "err: {msg}");
        assert!(msg.contains("'v'"), "err must name the column: {msg}");
        assert!(msg.contains("Float64"), "err must name the dtype: {msg}");
    }

    #[test]
    fn flatten_carried_column_keeps_its_declared_dtype() {
        // An Int64 carried column used to fall through to an all-null Float64
        // array declared as Int64; it now repeats at its own dtype.
        let inner: ArrayRef = Arc::new(StringArray::from(vec!["a", "b"]));
        let list = ListArray::new(
            Arc::new(Field::new("item", DataType::Utf8, true)),
            arrow::buffer::OffsetBuffer::new(vec![0_i32, 2].into()),
            inner,
            None,
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new(
                "vals",
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                true,
            ),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(arrow::array::Int64Array::from(vec![7_i64])), Arc::new(list)],
        )
        .unwrap();
        let out = flatten_vals(&batch).unwrap();

        assert_eq!(out.num_rows(), 2);
        assert_eq!(out.schema().field(0).data_type(), &DataType::Int64);
        let ids = out
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .unwrap();
        assert_eq!(ids.values(), &[7_i64, 7]);
    }

    #[test]
    fn flatten_large_list_of_large_utf8_emits_strings() {
        // polars hands Rust LargeList(LargeUtf8) — the guide's list-of-strings
        // example. Before the fix this fell through to an all-null column.
        let inner: ArrayRef = Arc::new(arrow::array::LargeStringArray::from(vec![
            "python", "rust", "go",
        ]));
        let list = LargeListArray::new(
            Arc::new(Field::new("item", DataType::LargeUtf8, true)),
            arrow::buffer::OffsetBuffer::new(vec![0_i64, 2, 3].into()),
            inner,
            None,
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("person", DataType::LargeUtf8, false),
            Field::new(
                "vals",
                DataType::LargeList(Arc::new(Field::new("item", DataType::LargeUtf8, true))),
                true,
            ),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(arrow::array::LargeStringArray::from(vec!["Alice", "Bob"])),
                Arc::new(list),
            ],
        )
        .unwrap();
        let out = flatten_vals(&batch).unwrap();

        assert_eq!(out.num_rows(), 3);
        let people = out
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::LargeStringArray>()
            .unwrap();
        assert_eq!(
            (0..3).map(|i| people.value(i)).collect::<Vec<_>>(),
            vec!["Alice", "Alice", "Bob"]
        );
        let vals = out
            .column(1)
            .as_any()
            .downcast_ref::<arrow::array::LargeStringArray>()
            .unwrap();
        assert_eq!(
            (0..3).map(|i| vals.value(i)).collect::<Vec<_>>(),
            vec!["python", "rust", "go"]
        );
    }

    #[test]
    fn flatten_carried_float64_null_stays_null_not_nan() {
        // Both halves of the null-preservation delta pinned together: a
        // carried (non-flattened) Float64 column's null cell must survive as
        // an Arrow null, not the pre-fix NaN substitution — the two are not
        // interchangeable on a categorical channel (see repeat_by_lengths's
        // doc comment).
        let inner: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c"]));
        let list = ListArray::new(
            Arc::new(Field::new("item", DataType::Utf8, true)),
            arrow::buffer::OffsetBuffer::new(vec![0_i32, 1, 2, 3].into()),
            inner,
            None,
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("score", DataType::Float64, true),
            Field::new(
                "tags",
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                true,
            ),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![Some(1.0), None, Some(2.0)])),
                Arc::new(list),
            ],
        )
        .unwrap();
        let out = apply(
            &FlattenSpec { fields: vec!["tags".into()], as_: None, name: None },
            &batch,
        )
        .unwrap();

        let score = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert!(score.is_null(1), "carried Float64 null must stay an Arrow null, not NaN");
        assert_eq!(score.value(0), 1.0);
        assert_eq!(score.value(2), 2.0);
    }

    #[test]
    fn flatten_list_of_float64_null_element_stays_null_not_nan() {
        // The other half: a null *element* inside a flattened List(Float64)
        // must also survive as an Arrow null, not the pre-fix NaN
        // substitution. This half was not disclosed when the carried-column
        // half shipped (see the `flatten` method's doc comment).
        let inner: ArrayRef = Arc::new(Float64Array::from(vec![Some(1.0), None, Some(3.0)]));
        let batch = list_batch(inner, vec![0, 2, 3], DataType::Float64);
        let out = flatten_vals(&batch).unwrap();

        assert_eq!(out.schema().field(1).data_type(), &DataType::Float64);
        let vals = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        assert!(vals.is_null(1), "null list element must stay an Arrow null, not NaN");
        assert_eq!(vals.value(0), 1.0);
        assert_eq!(vals.value(2), 3.0);
    }
}
