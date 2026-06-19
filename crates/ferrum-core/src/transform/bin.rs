use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch, UInt64Array};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::scale::ticks::sturges_floor;
use crate::transform::group_key::{
    groupby_field_nullable, groupby_key_at, is_groupby_supported_dtype, materialize_groupby_col,
    KeyValue,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct BinSpec {
    pub field: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub bin_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub bin_width: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub extent: Option<(f64, f64)>,
    #[serde(default = "default_true")]
    pub nice: bool,
    #[serde(default)]
    pub cumulative: bool,
    /// When ``true`` and ``groupby`` is set, compute a single global extent
    /// from all rows before per-group binning so every group shares the same
    /// bin edges. Required for ``multiple="stack"`` / ``"fill"`` to produce
    /// aligned bars that can be stacked correctly.
    #[serde(default)]
    pub shared_extent: bool,
    /// When set, partition input by this Utf8 column and emit per-(bin, group)
    /// rows. Output schema gains the groupby column as the 5th field.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub groupby: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

fn default_true() -> bool { true }

// Task 9 (render/prepare.rs) will call this when dispatching facet extent pins for Bin.
#[allow(dead_code)]
/// Compute the global (optionally niced) `(lo, hi)` extent of `spec.field`
/// over the full `batch`.
///
/// Returns `None` when the field is missing, non-numeric, or all values are
/// null/NaN. When `spec.nice` is true and `spec.extent` is `None`, applies the
/// same nicing logic as `apply_one_group` so that panels aligned to this extent
/// will produce the same bin edges as the grouped path would compute.
///
/// This is the pre-facet extent Task 9 uses to pin the value axis before
/// partitioning, so every facet panel shares the same bin edges.
pub(crate) fn global_extent(spec: &BinSpec, batch: &RecordBatch) -> Option<(f64, f64)> {
    // When the caller has already pinned an explicit extent, use it directly.
    if let Some(e) = spec.extent {
        return Some(e);
    }
    let schema = batch.schema();
    let idx = schema.index_of(&spec.field).ok()?;
    let col = batch.column(idx);
    // Cast to Float64 the same way apply_one_group does.
    let arr: arrow::array::Float64Array = match col.data_type() {
        arrow::datatypes::DataType::Float64 => col
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .cloned()?,
        arrow::datatypes::DataType::Int8
        | arrow::datatypes::DataType::Int16
        | arrow::datatypes::DataType::Int32
        | arrow::datatypes::DataType::Int64
        | arrow::datatypes::DataType::UInt8
        | arrow::datatypes::DataType::UInt16
        | arrow::datatypes::DataType::UInt32
        | arrow::datatypes::DataType::UInt64
        | arrow::datatypes::DataType::Float32 => {
            let casted = arrow::compute::cast(col, &arrow::datatypes::DataType::Float64).ok()?;
            casted.as_any().downcast_ref::<arrow::array::Float64Array>().cloned()?
        }
        _ => return None,
    };
    let clean = crate::transform::numeric_util::clean_float64_values(&arr, None);
    if clean.is_empty() {
        return None;
    }
    let (lo, hi) = clean
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    if !(lo.is_finite() && hi.is_finite() && lo < hi) {
        return None;
    }
    // Apply the same nicing as apply_one_group so the global extent aligns with
    // per-group bin edges. Only when nice=true and no explicit extent is set.
    let (lo, hi) = if spec.nice {
        let target = spec
            .bin_count
            .unwrap_or_else(|| crate::scale::ticks::sturges_floor(clean.len()))
            .max(1);
        let step = crate::scale::ticks::nice_step(lo, hi, target);
        if step.is_finite() && step > 0.0 {
            ((lo / step).floor() * step, (hi / step).ceil() * step)
        } else {
            (lo, hi)
        }
    } else {
        (lo, hi)
    };
    Some((lo, hi))
}

pub(crate) fn apply(spec: &BinSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    // Phase 9 finalize: groupby support. Partition input by the groupby column,
    // run bin_one_group per partition, then stack the per-group outputs into a
    // single batch with the group column preserved as the 5th field.
    if let Some(g) = &spec.groupby {
        return apply_grouped(spec, batch, g);
    }
    apply_one_group(spec, batch, None)
}

fn apply_one_group(
    spec: &BinSpec,
    batch: &RecordBatch,
    only_indices: Option<&[usize]>,
) -> PyResult<RecordBatch> {
    let schema = batch.schema();
    let idx = schema.index_of(&spec.field).map_err(|_| {
        PyValueError::new_err(format!(
            "stat_bin: column '{}' not found; available: {:?}",
            spec.field,
            schema.fields().iter().map(|f| f.name()).collect::<Vec<_>>()
        ))
    })?;
    let field = schema.field(idx);
    // Auto-cast integer types to Float64 so that Int32/Int64 columns work without
    // requiring the caller to pre-cast (e.g. JointChart marginal histograms).
    let col_ref: ArrayRef;
    let arr: &Float64Array = match field.data_type() {
        DataType::Float64 => batch
            .column(idx)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("dtype guarantees Float64Array"),
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64
        | DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64
        | DataType::Float32 => {
            col_ref = cast(batch.column(idx), &DataType::Float64)
                .map_err(|e| PyValueError::new_err(format!(
                    "stat_bin: could not cast column '{}' to Float64: {e}", spec.field
                )))?;
            col_ref
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("cast to Float64 succeeded")
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "stat_bin: column '{}' must be numeric; got {:?}",
                spec.field, other
            )));
        }
    };

    // Drop nulls and NaN; optionally restrict to a subset of row indices (for groupby).
    let clean = crate::transform::numeric_util::clean_float64_values(arr, only_indices);

    // Empty input → empty output (per spec §6: stat_bin is the exception that allows empty)
    if clean.is_empty() {
        return empty_bin_output();
    }

    let (lo, hi) = match spec.extent {
        Some((a, b)) if a < b => (a, b),
        Some((a, b)) => return Err(PyValueError::new_err(format!(
            "stat_bin: extent must satisfy lo < hi; got ({a}, {b})"
        ))),
        None => {
            let (lo, hi) = clean.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                (a.min(v), b.max(v))
            });
            if lo == hi {
                // Spec §4.1 numeric edge: all-equal → single unit bin
                return single_unit_bin(lo, clean.len() as u64);
            }
            (lo, hi)
        }
    };

    // Optional "nice" rounding of the extent. Only applies when extent is
    // auto-derived (not when the caller explicitly set extent), and only when
    // bin_count is fixed (or both bin_count and bin_width are unset, in which
    // case Sturges runs after nicing).
    let (lo, hi) = if spec.nice && spec.extent.is_none() {
        let target = spec.bin_count.unwrap_or_else(|| sturges_floor(clean.len())).max(1);
        let step = crate::scale::ticks::nice_step(lo, hi, target);
        if step.is_finite() && step > 0.0 {
            ((lo / step).floor() * step, (hi / step).ceil() * step)
        } else {
            (lo, hi)
        }
    } else {
        (lo, hi)
    };

    let n_bins: usize = match (spec.bin_count, spec.bin_width) {
        (Some(c), _) if c > 0 => c,
        (None, Some(w)) if w > 0.0 => ((hi - lo) / w).ceil().max(1.0) as usize,
        _ => sturges_floor(clean.len()),
    };

    let edges: Vec<f64> = (0..=n_bins)
        .map(|i| lo + (hi - lo) * (i as f64) / (n_bins as f64))
        .collect();

    let mut counts = vec![0u64; n_bins];
    for v in &clean {
        if *v < lo || *v > hi { continue; }
        // Last edge is inclusive; otherwise [lo, hi) per bin.
        let pos = if *v == hi {
            n_bins - 1
        } else {
            let raw = ((*v - lo) / (hi - lo) * (n_bins as f64)).floor() as usize;
            raw.min(n_bins - 1)
        };
        counts[pos] += 1;
    }

    let total = clean.len() as f64;
    let bin_starts: Vec<f64> = (0..n_bins).map(|i| edges[i]).collect();
    let bin_ends:   Vec<f64> = (0..n_bins).map(|i| edges[i + 1]).collect();
    let densities:  Vec<f64> = counts
        .iter()
        .zip(bin_starts.iter().zip(bin_ends.iter()))
        .map(|(c, (s, e))| (*c as f64) / (total * (e - s)))
        .collect();

    let (final_counts, final_densities) = if spec.cumulative {
        let mut acc_count: u64 = 0;
        let cum_counts: Vec<u64> = counts.iter().map(|c| {
            acc_count = acc_count.saturating_add(*c);
            acc_count
        }).collect();
        let mut acc_density: f64 = 0.0;
        let cum_densities: Vec<f64> = densities
            .iter()
            .zip(bin_starts.iter().zip(bin_ends.iter()))
            .map(|(d, (s, e))| {
                acc_density += d * (e - s);
                acc_density
            })
            .collect();
        (cum_counts, cum_densities)
    } else {
        (counts, densities)
    };

    build_bin_batch(bin_starts, bin_ends, final_counts, final_densities)
}

/// Partition input batch by `group_col` (Utf8), call apply_one_group per
/// partition, then stack the results into a single batch with the group
/// column preserved as the 5th field.
///
/// When `spec.shared_extent` is `true` and `spec.extent` is `None`, a global
/// extent is computed across all rows of the field before per-group binning so
/// every group produces the same bin edges — required for `multiple="stack"` /
/// `"fill"` stacking to find matching x-key pairs.
fn apply_grouped(
    spec: &BinSpec,
    batch: &RecordBatch,
    group_col: &str,
) -> PyResult<RecordBatch> {
    use std::collections::BTreeMap;
    let schema = batch.schema();
    let gi = schema.index_of(group_col).map_err(|_|
        PyValueError::new_err(format!(
            "stat_bin: groupby column '{}' not found", group_col)))?;
    let gtype = schema.field(gi).data_type().clone();
    if !is_groupby_supported_dtype(&gtype) {
        return Err(PyValueError::new_err(format!(
            "stat_bin: groupby column '{}' has unsupported dtype {:?}; \
             supported: Utf8/LargeUtf8, Float64/Float32, \
             Int8/Int16/Int32/Int64, UInt8/UInt16/UInt32/UInt64, Boolean",
            group_col, gtype)));
    }
    let garr = batch.column(gi);

    // Group row indices by first-appearance order of the group value (FA-7:
    // int/uint/bool/float/string all supported). A null group key collapses
    // into its own group (KeyValue::Null), distinct from any real value.
    let mut group_order: Vec<KeyValue> = Vec::new();
    let mut group_idx_map: BTreeMap<KeyValue, Vec<usize>> = BTreeMap::new();
    for i in 0..garr.len() {
        let gv = groupby_key_at(garr.as_ref(), &gtype, i).ok_or_else(|| {
            PyValueError::new_err(format!(
                "stat_bin: internal error extracting groupby key at row {i}"
            ))
        })?;
        // Skip null group keys: the pre-existing behaviour excluded null rows
        // from the grouped output, and a KDE/bin over a null bucket is not
        // meaningful as its own series.
        if matches!(gv, KeyValue::Null) {
            continue;
        }
        if !group_idx_map.contains_key(&gv) {
            group_order.push(gv.clone());
        }
        group_idx_map.entry(gv).or_default().push(i);
    }

    // When shared_extent=true, compute the global extent from all rows of the
    // field so all groups receive the same bin edges. Only override when the
    // caller has not explicitly set spec.extent.
    let effective_spec: std::borrow::Cow<BinSpec> = if spec.shared_extent && spec.extent.is_none() {
        let fidx = schema.index_of(&spec.field).map_err(|_| {
            PyValueError::new_err(format!(
                "stat_bin: column '{}' not found", spec.field
            ))
        })?;
        let col = batch.column(fidx);
        // Cast to Float64 to handle integer columns (same cast as apply_one_group).
        let col_cast;
        let farr: &Float64Array = match col.data_type() {
            DataType::Float64 => col.as_any().downcast_ref::<Float64Array>()
                .ok_or_else(|| PyValueError::new_err(format!(
                    "stat_bin: expected Float64Array for column '{}'", spec.field)))?,
            DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64
            | DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64
            | DataType::Float32 => {
                col_cast = arrow::compute::cast(col, &DataType::Float64)
                    .map_err(|e| PyValueError::new_err(format!(
                        "stat_bin: could not cast column '{}' to Float64: {e}", spec.field
                    )))?;
                col_cast.as_any().downcast_ref::<Float64Array>()
                    .ok_or_else(|| PyValueError::new_err(format!(
                        "stat_bin: cast to Float64 failed for column '{}'", spec.field)))?
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "stat_bin: column '{}' must be numeric for shared_extent", spec.field
                )));
            }
        };
        let (global_lo, global_hi) = (0..farr.len()).fold(
            (f64::INFINITY, f64::NEG_INFINITY),
            |(a, b), i| {
                if farr.is_null(i) { return (a, b); }
                let v = farr.value(i);
                if v.is_nan() { return (a, b); }
                (a.min(v), b.max(v))
            },
        );
        if global_lo.is_finite() && global_hi.is_finite() && global_lo < global_hi {
            let mut patched = spec.clone();
            patched.extent = Some((global_lo, global_hi));
            std::borrow::Cow::Owned(patched)
        } else {
            std::borrow::Cow::Borrowed(spec)
        }
    } else {
        std::borrow::Cow::Borrowed(spec)
    };
    let spec_ref: &BinSpec = &effective_spec;

    // Per-group output, then stack.
    let mut all_starts: Vec<f64> = Vec::new();
    let mut all_ends: Vec<f64> = Vec::new();
    let mut all_counts: Vec<u64> = Vec::new();
    let mut all_densities: Vec<f64> = Vec::new();
    let mut group_keys_out: Vec<Vec<KeyValue>> = Vec::new();
    for g in &group_order {
        let ixs = group_idx_map.get(g)
            .ok_or_else(|| PyValueError::new_err(format!(
                "stat_bin: missing group key {g:?} in index map")))?;
        let out = apply_one_group(spec_ref, batch, Some(ixs))?;
        let n = out.num_rows();
        let starts = out.column(0).as_any().downcast_ref::<Float64Array>()
            .ok_or_else(|| PyValueError::new_err("stat_bin: expected Float64Array for bin_start"))?;
        let ends = out.column(1).as_any().downcast_ref::<Float64Array>()
            .ok_or_else(|| PyValueError::new_err("stat_bin: expected Float64Array for bin_end"))?;
        let counts = out.column(2).as_any().downcast_ref::<UInt64Array>()
            .ok_or_else(|| PyValueError::new_err("stat_bin: expected UInt64Array for count"))?;
        let densities = out.column(3).as_any().downcast_ref::<Float64Array>()
            .ok_or_else(|| PyValueError::new_err("stat_bin: expected Float64Array for density"))?;
        for i in 0..n {
            all_starts.push(starts.value(i));
            all_ends.push(ends.value(i));
            all_counts.push(counts.value(i));
            all_densities.push(densities.value(i));
            group_keys_out.push(vec![g.clone()]);
        }
    }

    build_bin_batch_grouped(
        all_starts, all_ends, all_counts, all_densities, group_keys_out, group_col, &gtype,
    )
}

fn build_bin_batch(
    starts: Vec<f64>,
    ends: Vec<f64>,
    counts: Vec<u64>,
    densities: Vec<f64>,
) -> PyResult<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("bin_start", DataType::Float64, false),
        Field::new("bin_end",   DataType::Float64, false),
        Field::new("count",     DataType::UInt64,  false),
        Field::new("density",   DataType::Float64, false),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(starts)),
        Arc::new(Float64Array::from(ends)),
        Arc::new(UInt64Array::from(counts)),
        Arc::new(Float64Array::from(densities)),
    ];
    RecordBatch::try_new(schema, cols)
        .map_err(|e| PyValueError::new_err(format!("stat_bin: {e}")))
}

fn build_bin_batch_grouped(
    starts: Vec<f64>,
    ends: Vec<f64>,
    counts: Vec<u64>,
    densities: Vec<f64>,
    group_keys: Vec<Vec<KeyValue>>,
    group_col_name: &str,
    group_dtype: &DataType,
) -> PyResult<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("bin_start", DataType::Float64, false),
        Field::new("bin_end",   DataType::Float64, false),
        Field::new("count",     DataType::UInt64,  false),
        Field::new("density",   DataType::Float64, false),
        // FA-7: preserve the original groupby dtype (nullable for FA-9 null keys,
        // though null keys are skipped upstream for stat_bin).
        Field::new(group_col_name, group_dtype.clone(), groupby_field_nullable()),
    ]));
    let group_col = materialize_groupby_col(&group_keys, 0, group_dtype)
        .map_err(PyValueError::new_err)?;
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(starts)),
        Arc::new(Float64Array::from(ends)),
        Arc::new(UInt64Array::from(counts)),
        Arc::new(Float64Array::from(densities)),
        group_col,
    ];
    RecordBatch::try_new(schema, cols)
        .map_err(|e| PyValueError::new_err(format!("stat_bin: {e}")))
}

fn empty_bin_output() -> PyResult<RecordBatch> {
    build_bin_batch(Vec::new(), Vec::new(), Vec::new(), Vec::new())
}

fn single_unit_bin(v: f64, count: u64) -> PyResult<RecordBatch> {
    let start = v - 0.5;
    let end   = v + 0.5;
    let density = (count as f64) / ((count as f64) * (end - start));
    build_bin_batch(vec![start], vec![end], vec![count], vec![density])
}

use pyo3::prelude::*;

use crate::transform::core::TransformSpec;

/// Equal-width or quantile binning of a numeric column.
///
/// Discretizes a continuous field into intervals and counts values per bin.
/// Output columns: ``bin_start``, ``bin_end`` (Float64 bin edges), ``count``
/// (UInt64), and ``density`` (Float64, integrates to 1 over the range).
/// When ``cumulative=True``, ``count`` and ``density`` become running totals.
///
/// Parameters
/// ----------
/// field : str
///     Column to bin (must be Float64).
/// bin_count : int, optional
///     Number of equal-width bins. Mutually exclusive with ``bin_width``.
///     When both are omitted, Sturges' rule determines bin count.
/// bin_width : float, optional
///     Fixed bin width. Mutually exclusive with ``bin_count``.
/// extent : (float, float), optional
///     ``(lo, hi)`` range to cover; data outside is clipped. Both must be
///     finite and ``lo < hi``.
/// nice : bool, default True
///     Round bin edges to visually clean numbers.
/// cumulative : bool, default False
///     When True, append a ``cumulative`` count column.
/// groupby : str, optional
///     Single group-key column; bins computed independently per group.
/// name : str, optional
///     Named output label for sibling ``Reorder(from_=...)`` lookup.
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> fm.Chart(df).mark_histogram(bin_count=10).encode(x="mpg")
///
/// Direct construction:
///
/// >>> fm.Chart(df).mark_bar().encode(
/// ...     x=fm.X("mpg_bin_start"), y="count",
/// ...     transform=fm.Bin("mpg", bin_count=10),
/// ... )
#[pyclass(eq, module = "ferrum._core", name = "Bin")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyBin(pub(crate) TransformSpec);

#[pymethods]
impl PyBin {
    #[new]
    #[pyo3(signature = (field, *, bin_count = None, bin_width = None, extent = None, nice = true, cumulative = false, shared_extent = false, groupby = None, name = None))]
    fn new(
        field: &str,
        bin_count: Option<usize>,
        bin_width: Option<f64>,
        extent: Option<(f64, f64)>,
        nice: bool,
        cumulative: bool,
        shared_extent: bool,
        groupby: Option<String>,
        name: Option<String>,
    ) -> PyResult<Self> {
        if field.is_empty() {
            return Err(PyValueError::new_err("Bin: field must be non-empty"));
        }
        if let Some(c) = bin_count {
            if c == 0 {
                return Err(PyValueError::new_err("Bin: bin_count must be > 0"));
            }
        }
        if let Some(w) = bin_width {
            if !w.is_finite() || w <= 0.0 {
                return Err(PyValueError::new_err(
                    "Bin: bin_width must be a positive finite number",
                ));
            }
        }
        if let Some((a, b)) = extent {
            if !a.is_finite() || !b.is_finite() || a >= b {
                return Err(PyValueError::new_err(
                    "Bin: extent must be (lo, hi) with lo < hi and both finite",
                ));
            }
        }
        Ok(PyBin(TransformSpec::Bin(BinSpec {
            field: field.to_string(),
            bin_count,
            bin_width,
            extent,
            nice,
            cumulative,
            shared_extent,
            groupby,
            name,
        })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Bin(s) => format!(
                "Bin(field='{}', bin_count={:?}, bin_width={:?}, extent={:?}, nice={}, cumulative={})",
                s.field, s.bin_count, s.bin_width, s.extent,
                if s.nice { "True" } else { "False" },
                if s.cumulative { "True" } else { "False" },
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, UInt64Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn batch_with(values: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
        ]));
        RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(values))]).unwrap()
    }

    fn col_f64<'a>(b: &'a RecordBatch, name: &str) -> &'a Float64Array {
        b.column(b.schema().index_of(name).unwrap())
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
    }

    fn col_u64<'a>(b: &'a RecordBatch, name: &str) -> &'a UInt64Array {
        b.column(b.schema().index_of(name).unwrap())
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
    }

    #[test]
    fn test_bin_basic_counts_match_numpy_histogram() {
        // numpy.histogram([1,2,3,4,5,6,7,8,9,10], bins=5, range=(1,10))
        // edges: [1.0, 2.8, 4.6, 6.4, 8.2, 10.0]
        // counts: [2, 2, 2, 2, 2]   (10 inclusive captured by upper-edge convention)
        let batch = batch_with(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let spec = BinSpec {
            field: "x".into(),
            bin_count: Some(5),
            bin_width: None,
            extent: Some((1.0, 10.0)),
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 5);
        let counts = col_u64(&out, "count");
        for i in 0..5 {
            assert_eq!(counts.value(i), 2, "bin {i} count: got {}", counts.value(i));
        }
        let starts = col_f64(&out, "bin_start");
        let ends = col_f64(&out, "bin_end");
        for i in 0..5 {
            let expected_start = 1.0 + 1.8 * i as f64;
            let expected_end = expected_start + 1.8;
            assert!((starts.value(i) - expected_start).abs() < 1e-9);
            assert!((ends.value(i) - expected_end).abs() < 1e-9);
        }
    }

    #[test]
    fn test_bin_density_normalizes_to_one() {
        let batch = batch_with(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let spec = BinSpec {
            field: "x".into(),
            bin_count: Some(5),
            bin_width: None,
            extent: Some((1.0, 10.0)),
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let densities = col_f64(&out, "density");
        let starts = col_f64(&out, "bin_start");
        let ends = col_f64(&out, "bin_end");
        let mut total: f64 = 0.0;
        for i in 0..5 {
            total += densities.value(i) * (ends.value(i) - starts.value(i));
        }
        assert!((total - 1.0).abs() < 1e-12, "density integrates to {total}");
    }

    #[test]
    fn test_bin_default_count_uses_sturges_floor() {
        // sturges_floor(8) = 4 per scale::ticks::sturges_floor
        let batch = batch_with(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let spec = BinSpec {
            field: "x".into(),
            bin_count: None,
            bin_width: None,
            extent: None,
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 4);
    }

    #[test]
    fn test_bin_all_equal_data_emits_single_unit_bin() {
        let batch = batch_with(vec![3.0, 3.0, 3.0]);
        let spec = BinSpec {
            field: "x".into(),
            bin_count: None,
            bin_width: None,
            extent: None,
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 1);
        let starts = col_f64(&out, "bin_start");
        let ends = col_f64(&out, "bin_end");
        let counts = col_u64(&out, "count");
        assert!((starts.value(0) - 2.5).abs() < 1e-12);
        assert!((ends.value(0)   - 3.5).abs() < 1e-12);
        assert_eq!(counts.value(0), 3);
    }

    #[test]
    fn test_bin_drops_nulls_and_nans() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
        ]));
        let arr = Float64Array::from(vec![Some(1.0), None, Some(2.0), Some(f64::NAN), Some(3.0)]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        let spec = BinSpec {
            field: "x".into(),
            bin_count: Some(2),
            bin_width: None,
            extent: Some((1.0, 3.0)),
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let counts = col_u64(&out, "count");
        let total: u64 = (0..out.num_rows()).map(|i| counts.value(i)).sum();
        assert_eq!(total, 3, "expected 3 non-null/non-nan values");
    }

    #[test]
    fn test_bin_missing_field_errors() {
        pyo3::Python::initialize();
        let batch = batch_with(vec![1.0, 2.0, 3.0]);
        let spec = BinSpec {
            field: "ghost".into(),
            bin_count: Some(5),
            bin_width: None,
            extent: None,
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: None,
            name: None,
        };
        let err = apply(&spec, &batch).unwrap_err();
        assert!(err.to_string().contains("ghost"), "err: {err}");
    }

    #[test]
    fn test_bin_wrong_dtype_errors() {
        pyo3::Python::initialize();
        use arrow::array::StringArray;
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec!["a", "b", "c"]))],
        ).unwrap();
        let spec = BinSpec {
            field: "x".into(),
            bin_count: Some(2),
            bin_width: None,
            extent: Some((1.0, 3.0)),
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: None,
            name: None,
        };
        let err = apply(&spec, &batch).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("numeric") || msg.contains("Utf8"), "err: {msg}");
    }

    #[test]
    fn test_bin_int64_auto_casts_to_float64() {
        pyo3::Python::initialize();
        use arrow::array::Int64Array;
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5]))],
        ).unwrap();
        let spec = BinSpec {
            field: "x".into(),
            bin_count: Some(2),
            bin_width: None,
            extent: None,
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: None,
            name: None,
        };
        let result = apply(&spec, &batch);
        assert!(result.is_ok(), "Int64 should auto-cast to Float64: {:?}", result.err());
    }

    #[test]
    fn test_bin_nice_extent_rounds_outward() {
        // x in [0.13, 9.7], 10 bins, nice=true → extent should round to a "nice"
        // outer bound (e.g. [0, 10] for step=1.0). The exact result depends on
        // nice_step's algorithm but lo ≤ 0.13 and hi ≥ 9.7 must hold, and
        // (hi - lo) must be a clean multiple of step.
        let batch = batch_with(vec![0.13, 1.5, 4.5, 7.7, 9.7]);
        let spec = BinSpec {
            field: "x".into(),
            bin_count: Some(10),
            bin_width: None,
            extent: None,
            nice: true,
            cumulative: false,
            shared_extent: false,
            groupby: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let starts = col_f64(&out, "bin_start");
        let ends   = col_f64(&out, "bin_end");
        let lo = starts.value(0);
        let hi = ends.value(out.num_rows() - 1);
        // nice_step(0.13, 9.7, 10) = 1.0 → floor(0.13/1.0)*1.0 = 0.0, ceil(9.7/1.0)*1.0 = 10.0
        assert_eq!(lo, 0.0, "nice lo should be exactly 0.0 (step=1.0 rounds 0.13 down)");
        assert_eq!(hi, 10.0, "nice hi should be exactly 10.0 (step=1.0 rounds 9.7 up)");
        // bin width = (10.0 - 0.0) / 10 bins = 1.0 exactly
        let bin_width = ends.value(0) - starts.value(0);
        assert!((bin_width - 1.0).abs() < 1e-12, "bin width should be exactly 1.0, got {bin_width}");
        // total count must equal n=5
        let counts = col_u64(&out, "count");
        let total: u64 = (0..out.num_rows()).map(|i| counts.value(i)).sum();
        assert_eq!(total, 5);
    }

    #[test]
    fn bin_cumulative_count_is_monotonic() {
        let batch = batch_with(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let spec = BinSpec {
            field: "x".into(),
            bin_count: Some(5),
            bin_width: None,
            extent: Some((1.0, 10.0)),
            nice: false,
            cumulative: true,
            shared_extent: false,
            groupby: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let counts = col_u64(&out, "count");
        let n = counts.len();
        for i in 1..n {
            assert!(counts.value(i) >= counts.value(i - 1),
                "cumulative count not monotonic at i={i}: {} < {}",
                counts.value(i), counts.value(i - 1));
        }
        assert_eq!(counts.value(n - 1), 10);
        // Cumulative density at the end should equal 1.0 (full CDF).
        let densities = col_f64(&out, "density");
        let last = densities.value(n - 1);
        assert!((last - 1.0).abs() < 1e-12, "cumulative density should reach 1.0, got {last}");
    }

    #[test]
    fn test_bin_single_row_produces_one_bin() {
        let batch = batch_with(vec![42.0]);
        let spec = BinSpec {
            field: "x".into(),
            bin_count: None,
            bin_width: None,
            extent: None,
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 1);
        let counts = col_u64(&out, "count");
        assert_eq!(counts.value(0), 1);
    }

    #[test]
    fn test_bin_all_identical_values_single_bin() {
        // 10 rows all with value=5.0 → should produce exactly 1 bin containing all 10.
        let batch = batch_with(vec![5.0; 10]);
        let spec = BinSpec {
            field: "x".into(),
            bin_count: Some(5),
            bin_width: None,
            extent: None,
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 1);
        let counts = col_u64(&out, "count");
        assert_eq!(counts.value(0), 10);
    }

    #[test]
    fn test_bin_count_one_single_bin_spanning_full_extent() {
        let batch = batch_with(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let spec = BinSpec {
            field: "x".into(),
            bin_count: Some(1),
            bin_width: None,
            extent: Some((1.0, 5.0)),
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 1);
        let counts = col_u64(&out, "count");
        assert_eq!(counts.value(0), 5);
        let starts = col_f64(&out, "bin_start");
        let ends = col_f64(&out, "bin_end");
        assert!((starts.value(0) - 1.0).abs() < 1e-12);
        assert!((ends.value(0) - 5.0).abs() < 1e-12);
    }

    // ── Task 8: global_extent helper ─────────────────────────────────────────

    /// `global_extent` returns the raw (not niced) extent when `nice=false`.
    #[test]
    fn test_bin_global_extent_no_nice() {
        let batch = batch_with(vec![1.5, 3.7, 2.2, 8.1]);
        let spec = BinSpec {
            field: "x".into(),
            bin_count: Some(4),
            bin_width: None,
            extent: None,
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: None,
            name: None,
        };
        let ext = super::global_extent(&spec, &batch);
        let Some((lo, hi)) = ext else {
            panic!("global_extent must return Some for valid data");
        };
        assert!((lo - 1.5).abs() < 1e-12, "lo should be 1.5, got {lo}");
        assert!((hi - 8.1).abs() < 1e-12, "hi should be 8.1, got {hi}");
    }

    /// `global_extent` with `nice=true` applies the same nicing as `apply_one_group`,
    /// so the extent matches what a grouped-path call would produce for this field.
    #[test]
    fn test_bin_global_extent_niced_matches_apply_one_group() {
        // x in [0.13, 9.7], 10 bins, nice=true. The apply path produces [0.0, 10.0].
        let batch = batch_with(vec![0.13, 1.5, 4.5, 7.7, 9.7]);
        let spec = BinSpec {
            field: "x".into(),
            bin_count: Some(10),
            bin_width: None,
            extent: None,
            nice: true,
            cumulative: false,
            shared_extent: false,
            groupby: None,
            name: None,
        };
        let ext = super::global_extent(&spec, &batch);
        let Some((lo, hi)) = ext else {
            panic!("global_extent must return Some for valid data");
        };
        // nice_step(0.13, 9.7, 10) = 1.0 → floor(0.13/1.0)*1.0 = 0.0, ceil(9.7/1.0)*1.0 = 10.0
        assert_eq!(lo, 0.0, "niced lo should be 0.0, got {lo}");
        assert_eq!(hi, 10.0, "niced hi should be 10.0, got {hi}");

        // Also verify that running apply with this pinned extent produces the same
        // bin edges as running apply without a pinned extent (but with nice=true),
        // i.e. the niced global extent is consistent with the apply-side nicing.
        let spec_pinned = BinSpec { extent: Some((lo, hi)), nice: false, ..spec.clone() };
        let spec_auto = BinSpec { extent: None, nice: true, ..spec.clone() };
        let out_pinned = apply(&spec_pinned, &batch).unwrap();
        let out_auto = apply(&spec_auto, &batch).unwrap();
        assert_eq!(out_pinned.num_rows(), out_auto.num_rows(), "bin count must match");
        let pinned_starts = col_f64(&out_pinned, "bin_start");
        let auto_starts = col_f64(&out_auto, "bin_start");
        for i in 0..out_pinned.num_rows() {
            assert!(
                (pinned_starts.value(i) - auto_starts.value(i)).abs() < 1e-12,
                "bin_start[{i}] mismatch: pinned={}, auto={}",
                pinned_starts.value(i),
                auto_starts.value(i)
            );
        }
    }

    /// `global_extent` returns the explicitly-set extent when `spec.extent` is `Some`.
    #[test]
    fn test_bin_global_extent_explicit_extent_passthrough() {
        let batch = batch_with(vec![1.0, 5.0, 10.0]);
        let spec = BinSpec {
            field: "x".into(),
            bin_count: Some(5),
            bin_width: None,
            extent: Some((0.0, 20.0)),
            nice: true,
            cumulative: false,
            shared_extent: false,
            groupby: None,
            name: None,
        };
        let ext = super::global_extent(&spec, &batch);
        // When extent is already set, global_extent returns it directly.
        assert_eq!(ext, Some((0.0, 20.0)), "global_extent must return the explicit extent unchanged");
    }

    /// `global_extent` returns None on missing field.
    #[test]
    fn test_bin_global_extent_missing_field_returns_none() {
        pyo3::Python::initialize();
        let batch = batch_with(vec![1.0, 2.0, 3.0]);
        let spec = BinSpec {
            field: "nonexistent".into(),
            bin_count: Some(5),
            bin_width: None,
            extent: None,
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: None,
            name: None,
        };
        let ext = super::global_extent(&spec, &batch);
        assert_eq!(ext, None, "global_extent on missing field must return None");
    }

    /// `global_extent` handles integer columns the same way as `apply_one_group`.
    #[test]
    fn test_bin_global_extent_integer_column() {
        pyo3::Python::initialize();
        use arrow::array::Int64Array;
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![1_i64, 5, 3]))],
        ).unwrap();
        let spec = BinSpec {
            field: "x".into(),
            bin_count: Some(3),
            bin_width: None,
            extent: None,
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: None,
            name: None,
        };
        let ext = super::global_extent(&spec, &batch);
        let Some((lo, hi)) = ext else {
            panic!("global_extent must return Some for valid Int64 data");
        };
        assert!((lo - 1.0).abs() < 1e-12, "lo should be 1.0, got {lo}");
        assert!((hi - 5.0).abs() < 1e-12, "hi should be 5.0, got {hi}");
    }
}
