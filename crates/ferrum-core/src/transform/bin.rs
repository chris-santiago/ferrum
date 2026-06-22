use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::scale::ticks::sturges_floor;
use crate::transform::bin_mode::{resolve_bin_count, BinMode};
use crate::transform::group_key::{
    deserialize_groupby_vec, group_partition_multi, groupby_field_nullable,
    materialize_groupby_col, parse_groupby_pyany, KeyValue,
};
use crate::transform::numeric_util::{
    clean_float64_values, coerce_to_float64, column_extent, resolve_shared_extent,
};

// ── Shared private helpers ────────────────────────────────────────────────────

/// Apply nice rounding to a raw `(lo, hi)` extent for a target bin count.
///
/// Computes the nice step size via `nice_step`, then floors `lo` and ceils
/// `hi` to the nearest multiple of that step.  When the step is non-positive
/// or non-finite (e.g. degenerate extent), the original values are returned
/// unchanged.
///
/// This is the single source for the `nice_step` + floor/ceil block that
/// previously appeared separately in `global_extent` and `apply_one_group`.
fn nice_extent(lo: f64, hi: f64, target: usize) -> (f64, f64) {
    let step = crate::scale::ticks::nice_step(lo, hi, target);
    if step.is_finite() && step > 0.0 {
        ((lo / step).floor() * step, (hi / step).ceil() * step)
    } else {
        (lo, hi)
    }
}

/// 1-D equal-width binning spec.
///
/// The bin-count decision is a typed [`BinMode`] (no overlapping
/// `bin_count`/`bin_width` `Option`s). The flat `bin_count`/`bin_width` JSON wire
/// keys are preserved by the [`BinWire`] serde shim (see `#[serde(into/from)]`),
/// so existing wire payloads round-trip byte-for-byte and the precedence
/// (`bin_count` wins over `bin_width` wins over Sturges) is resolved once at the
/// (de)serialization boundary rather than at apply time.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(into = "BinWire", from = "BinWire")]
pub(crate) struct BinSpec {
    pub field: String,
    pub mode: BinMode,
    pub extent: Option<(f64, f64)>,
    pub nice: bool,
    pub cumulative: bool,
    /// When ``true`` and ``groupby`` is set, compute a single global extent
    /// from all rows before per-group binning so every group shares the same
    /// bin edges. Required for ``multiple="stack"`` / ``"fill"`` to produce
    /// aligned bars that can be stacked correctly.
    pub shared_extent: bool,
    /// When non-empty, partition input by these columns and emit per-(bin, group)
    /// rows. Output schema gains one column per groupby field after the 4th field.
    pub groupby: Vec<String>,
    pub name: Option<String>,
}

/// Flat serde projection of [`BinSpec`] preserving the legacy wire shape.
///
/// `BinSpec` serializes/deserializes through this struct so the wire keeps the
/// flat `bin_count`/`bin_width`/`groupby` keys that Python and the test suite
/// assert on. `BinMode` is never serialized directly for the 1-D `Bin`.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BinWire {
    field: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    bin_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    bin_width: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    extent: Option<(f64, f64)>,
    #[serde(default = "default_true")]
    nice: bool,
    #[serde(default)]
    cumulative: bool,
    #[serde(default)]
    shared_extent: bool,
    /// Wire accepts: `null` → `[]`, `"col"` → `["col"]`, `["a","b"]` → `["a","b"]`.
    #[serde(
        skip_serializing_if = "Vec::is_empty",
        default,
        deserialize_with = "deserialize_groupby_vec"
    )]
    groupby: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    name: Option<String>,
}

impl From<BinWire> for BinSpec {
    fn from(w: BinWire) -> Self {
        // Resolve the flat-field precedence once at the boundary: bin_count wins,
        // then bin_width, else Sturges (mirrors the old apply-time match).
        let mode = if let Some(n) = w.bin_count {
            BinMode::Fixed { n }
        } else if let Some(width) = w.bin_width {
            BinMode::Width { w: width }
        } else {
            BinMode::Sturges
        };
        BinSpec {
            field: w.field,
            mode,
            extent: w.extent,
            nice: w.nice,
            cumulative: w.cumulative,
            shared_extent: w.shared_extent,
            groupby: w.groupby,
            name: w.name,
        }
    }
}

impl From<BinSpec> for BinWire {
    fn from(s: BinSpec) -> Self {
        let (bin_count, bin_width) = match s.mode {
            BinMode::Fixed { n } => (Some(n), None),
            BinMode::Width { w } => (None, Some(w)),
            BinMode::Sturges => (None, None),
            // unreachable for 1-D Bin; the constructor only yields Sturges/Fixed/Width
            BinMode::FreedmanDiaconis => (None, None),
        };
        BinWire {
            field: s.field,
            bin_count,
            bin_width,
            extent: s.extent,
            nice: s.nice,
            cumulative: s.cumulative,
            shared_extent: s.shared_extent,
            groupby: s.groupby,
            name: s.name,
        }
    }
}

fn default_true() -> bool { true }

/// Compute the global (optionally niced) `(lo, hi)` extent of `spec.field`
/// over the full `batch`.
///
/// Returns `None` when the field is missing, non-numeric, or all values are
/// null/NaN. When `spec.nice` is true and `spec.extent` is `None`, applies the
/// same nicing logic as `apply_one_group` so that panels aligned to this extent
/// will produce the same bin edges as the grouped path would compute.
///
/// This is the pre-facet extent that `render::prepare` uses to pin the value
/// axis before partitioning, so every facet panel shares the same bin edges
/// (see `fix_transform_extents_for_facet`).
pub(crate) fn global_extent(spec: &BinSpec, batch: &RecordBatch) -> Option<(f64, f64)> {
    // When the caller has already pinned an explicit extent, use it directly.
    if let Some(e) = spec.extent {
        return Some(e);
    }
    let (lo, hi) = column_extent(batch, &spec.field)?;
    // NICENESS CONTRACT (XFORM-08): `Bin` nices its faceted-pin extent so the
    // pinned range reproduces the same bin edges every per-group/per-panel
    // partition would compute. This is the one extent-deriving transform that
    // nices — `Kde`/`Violin`/`DensityData`/`Bin2D`/`Hex`/`Raster`/`Kde2D` all
    // return the raw `column_extent`. The divergence lives here, at the call
    // site, not in the shared fold. Only nice when `nice=true` and no explicit
    // extent was set.
    if spec.nice {
        // Sturges fallback target from the clean count. When the mode is
        // `Fixed{n}`, `clean.len()` is unused but the coercion is cheap; extracting
        // it avoids a second `column_extent` call. Width/Sturges/FD all fall back
        // to the Sturges target, preserving the old `bin_count.unwrap_or(sturges)`.
        let target = match &spec.mode {
            BinMode::Fixed { n } => (*n).max(1),
            _ => {
                let schema = batch.schema();
                let idx = schema.index_of(&spec.field).ok()?;
                let arr = coerce_to_float64(batch.column(idx), "stat_bin", &spec.field).ok()?;
                let n = clean_float64_values(&arr, None).len();
                sturges_floor(n).max(1)
            }
        };
        Some(nice_extent(lo, hi, target))
    } else {
        Some((lo, hi))
    }
}

pub(crate) fn apply(spec: &BinSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    // D-GROUPBY-1: groupby is now Vec<String>. Non-empty → grouped path.
    if !spec.groupby.is_empty() {
        return apply_grouped(spec, batch, &spec.groupby);
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
    // Auto-cast integer types to Float64 so that Int32/Int64 columns work without
    // requiring the caller to pre-cast (e.g. JointChart marginal histograms).
    // `coerce_to_float64` is the single source for this cast (XFORM-05).
    let arr = coerce_to_float64(batch.column(idx), "stat_bin", &spec.field)?;

    // Drop nulls and NaN; optionally restrict to a subset of row indices (for groupby).
    let clean = clean_float64_values(&arr, only_indices);

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
    // auto-derived (not when the caller explicitly set extent). When the mode is
    // `Fixed{n}` the target is that count; Width/Sturges/FD round to the Sturges
    // target (preserving the old `bin_count.unwrap_or(sturges)` behavior).
    let (lo, hi) = if spec.nice && spec.extent.is_none() {
        let target = match &spec.mode {
            BinMode::Fixed { n } => *n,
            _ => sturges_floor(clean.len()),
        }
        .max(1);
        nice_extent(lo, hi, target)
    } else {
        (lo, hi)
    };

    let n_bins: usize = resolve_bin_count(&spec.mode, &clean, lo, hi, "stat_bin")?;

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

/// Partition input batch by the groupby columns, call apply_one_group per
/// partition, then stack the results into a single batch with the group
/// columns appended after the 4th field.
///
/// When `spec.shared_extent` is `true` and `spec.extent` is `None`, a global
/// extent is computed across all rows of the field before per-group binning so
/// every group produces the same bin edges — required for `multiple="stack"` /
/// `"fill"` stacking to find matching x-key pairs.
fn apply_grouped(
    spec: &BinSpec,
    batch: &RecordBatch,
    group_cols: &[String],
) -> PyResult<RecordBatch> {
    // Multi-column partition via the shared helper (first-appearance order,
    // null-key skip, dtype-preservation — same policy as single-column path).
    let (group_order, group_idx_map, gtypes) =
        group_partition_multi(batch, group_cols, "stat_bin")?;

    // When shared_extent=true, inject the global raw extent so every group gets
    // the same bin edges (required for stack/fill). resolve_shared_extent returns
    // the explicit extent unchanged when set, else None unless shared is on.
    let effective_spec: std::borrow::Cow<BinSpec> =
        match resolve_shared_extent(spec.extent, spec.shared_extent, batch, &spec.field) {
            Some(e) if spec.extent.is_none() => {
                let mut patched = spec.clone();
                patched.extent = Some(e);
                std::borrow::Cow::Owned(patched)
            }
            _ => std::borrow::Cow::Borrowed(spec),
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
            group_keys_out.push(g.clone());
        }
    }

    build_bin_batch_grouped(
        all_starts, all_ends, all_counts, all_densities, group_keys_out, group_cols, &gtypes,
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
    group_col_names: &[String],
    group_dtypes: &[DataType],
) -> PyResult<RecordBatch> {
    let mut fields = vec![
        Field::new("bin_start", DataType::Float64, false),
        Field::new("bin_end",   DataType::Float64, false),
        Field::new("count",     DataType::UInt64,  false),
        Field::new("density",   DataType::Float64, false),
    ];
    // FA-7: preserve original groupby dtypes (nullable for FA-9 null keys,
    // though null keys are skipped upstream for stat_bin).
    for (name, dtype) in group_col_names.iter().zip(group_dtypes.iter()) {
        fields.push(Field::new(name, dtype.clone(), groupby_field_nullable()));
    }
    let mut cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(starts)),
        Arc::new(Float64Array::from(ends)),
        Arc::new(UInt64Array::from(counts)),
        Arc::new(Float64Array::from(densities)),
    ];
    for (gi, dtype) in group_dtypes.iter().enumerate() {
        let group_col = materialize_groupby_col(&group_keys, gi, dtype)
            .map_err(PyValueError::new_err)?;
        cols.push(group_col);
    }
    let schema = Arc::new(Schema::new(fields));
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
/// groupby : list[str] or str or None, optional
///     One or more group-key columns; bins computed independently per group.
///     Pass a single string or a one-element list for single-column groupby
///     (byte-identical to the old ``Option<String>`` behavior).
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
        groupby: Option<&Bound<'_, PyAny>>,
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
        let groupby = parse_groupby_pyany(groupby)?;
        // Resolve the flat constructor params to a typed mode with today's
        // precedence: bin_count wins → Fixed; else bin_width → Width; else Sturges.
        let mode = if let Some(n) = bin_count {
            BinMode::Fixed { n }
        } else if let Some(w) = bin_width {
            BinMode::Width { w }
        } else {
            BinMode::Sturges
        };
        Ok(PyBin(TransformSpec::Bin(BinSpec {
            field: field.to_string(),
            mode,
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
            TransformSpec::Bin(s) => {
                // Project the typed mode back to the public (bin_count, bin_width)
                // spelling for continuity with the constructor signature.
                let (bin_count, bin_width): (Option<usize>, Option<f64>) = match &s.mode {
                    BinMode::Fixed { n } => (Some(*n), None),
                    BinMode::Width { w } => (None, Some(*w)),
                    BinMode::Sturges | BinMode::FreedmanDiaconis => (None, None),
                };
                format!(
                    "Bin(field='{}', bin_count={:?}, bin_width={:?}, extent={:?}, nice={}, cumulative={})",
                    s.field, bin_count, bin_width, s.extent,
                    if s.nice { "True" } else { "False" },
                    if s.cumulative { "True" } else { "False" },
                )
            }
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
            mode: BinMode::Fixed { n: 5 },
            extent: Some((1.0, 10.0)),
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
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
            mode: BinMode::Fixed { n: 5 },
            extent: Some((1.0, 10.0)),
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
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
            mode: BinMode::Sturges,
            extent: None,
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
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
            mode: BinMode::Sturges,
            extent: None,
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
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
            mode: BinMode::Fixed { n: 2 },
            extent: Some((1.0, 3.0)),
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
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
            mode: BinMode::Fixed { n: 5 },
            extent: None,
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
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
            mode: BinMode::Fixed { n: 2 },
            extent: Some((1.0, 3.0)),
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
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
            mode: BinMode::Fixed { n: 2 },
            extent: None,
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
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
            mode: BinMode::Fixed { n: 10 },
            extent: None,
            nice: true,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
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
            mode: BinMode::Fixed { n: 5 },
            extent: Some((1.0, 10.0)),
            nice: false,
            cumulative: true,
            shared_extent: false,
            groupby: vec![],
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
            mode: BinMode::Sturges,
            extent: None,
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
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
            mode: BinMode::Fixed { n: 5 },
            extent: None,
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
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
            mode: BinMode::Fixed { n: 1 },
            extent: Some((1.0, 5.0)),
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
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
            mode: BinMode::Fixed { n: 4 },
            extent: None,
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
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
            mode: BinMode::Fixed { n: 10 },
            extent: None,
            nice: true,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
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
            mode: BinMode::Fixed { n: 5 },
            extent: Some((0.0, 20.0)),
            nice: true,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
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
            mode: BinMode::Fixed { n: 5 },
            extent: None,
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
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
            mode: BinMode::Fixed { n: 3 },
            extent: None,
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
            name: None,
        };
        let ext = super::global_extent(&spec, &batch);
        let Some((lo, hi)) = ext else {
            panic!("global_extent must return Some for valid Int64 data");
        };
        assert!((lo - 1.0).abs() < 1e-12, "lo should be 1.0, got {lo}");
        assert!((hi - 5.0).abs() < 1e-12, "hi should be 5.0, got {hi}");
    }

    // ── Unit tests for shared private helpers (R4 dedup) ─────────────────────

    /// `nice_extent` rounds (0.3, 9.7) outward to (0.0, 10.0) at bin_count 10.
    ///
    /// This pins the single source of the nice algorithm. The same inputs are
    /// exercised through `apply_one_group` and `global_extent` in the tests
    /// above; this test targets the helper directly so a future change to the
    /// algorithm cannot be made in one call site and silently missed in others.
    #[test]
    fn test_nice_extent_rounds_outward() {
        // nice_step(0.3, 9.7, 10) = 1.0 → floor(0.3/1.0)*1.0 = 0.0, ceil(9.7/1.0)*1.0 = 10.0
        let (lo, hi) = super::nice_extent(0.3, 9.7, 10);
        assert_eq!(lo, 0.0, "nice lo: got {lo}");
        assert_eq!(hi, 10.0, "nice hi: got {hi}");
    }

    /// `nice_extent` is a no-op when the step is degenerate (lo == hi would not
    /// reach here in practice; guard against zero step).
    #[test]
    fn test_nice_extent_degenerate_step_passthrough() {
        // Passing target=0 forces nice_step to return 0 or non-positive → passthrough.
        // We avoid calling with target=0 in production (max(1) guards it), but
        // the helper itself must not panic; it just returns the original values.
        let (lo, hi) = super::nice_extent(1.0, 2.0, 1);
        // Any finite result where lo <= 1.0 and hi >= 2.0 is acceptable.
        assert!(lo <= 1.0, "niced lo must not exceed raw lo");
        assert!(hi >= 2.0, "niced hi must not be below raw hi");
    }

    // `Bin`'s extent now delegates to `numeric_util::column_extent`; these tests
    // exercise that shared helper through the path `global_extent` takes.
    use crate::transform::numeric_util::column_extent;

    /// `column_extent` returns the min/max of a clean Float64 batch.
    #[test]
    fn test_column_extent_basic() {
        let batch = batch_with(vec![3.0, 1.0, 5.0, 2.0]);
        let ext = column_extent(&batch, "x");
        assert_eq!(ext, Some((1.0, 5.0)));
    }

    /// `column_extent` drops nulls and NaN values.
    #[test]
    fn test_column_extent_drops_nulls_and_nan() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, true)]));
        let arr = Float64Array::from(vec![Some(2.0), None, Some(f64::NAN), Some(8.0)]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        let ext = column_extent(&batch, "x");
        assert_eq!(ext, Some((2.0, 8.0)));
    }

    /// `column_extent` returns None for a missing field.
    #[test]
    fn test_column_extent_missing_field() {
        pyo3::Python::initialize();
        let batch = batch_with(vec![1.0, 2.0]);
        let ext = column_extent(&batch, "ghost");
        assert_eq!(ext, None);
    }

    /// `column_extent` returns None when lo == hi (degenerate extent).
    #[test]
    fn test_column_extent_degenerate_all_equal() {
        let batch = batch_with(vec![5.0, 5.0, 5.0]);
        let ext = column_extent(&batch, "x");
        assert_eq!(ext, None, "degenerate extent (lo==hi) must return None");
    }

    // ── T3.8a regression tests ────────────────────────────────────────────────

    /// Helper: build a batch with an `x` float column plus two string group columns.
    fn batch_with_two_groups(
        xs: Vec<f64>,
        g1s: Vec<&str>,
        g2s: Vec<&str>,
    ) -> RecordBatch {
        use arrow::array::StringArray;
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("g1", DataType::Utf8, false),
            Field::new("g2", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(xs)) as arrow::array::ArrayRef,
                Arc::new(StringArray::from(g1s)) as arrow::array::ArrayRef,
                Arc::new(StringArray::from(g2s)) as arrow::array::ArrayRef,
            ],
        )
        .unwrap()
    }

    fn col_str<'a>(b: &'a RecordBatch, name: &str) -> &'a arrow::array::StringArray {
        b.column(b.schema().index_of(name).unwrap())
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .unwrap()
    }

    /// D-GROUPBY-1 regression (T3.8a): a 2-column groupby on `Bin` must produce one
    /// set of bins per distinct `(g1, g2)` combination, with the two group columns
    /// appended after the standard 4 bin output columns.
    #[test]
    fn bin_two_column_groupby_produces_per_combination_rows() {
        // 8 rows: two g1 values × two g2 values × 2 x values each.
        let batch = batch_with_two_groups(
            vec![1.0, 9.0, 1.0, 9.0, 1.0, 9.0, 1.0, 9.0],
            vec!["A", "A", "A", "A", "B", "B", "B", "B"],
            vec!["x", "x", "y", "y", "x", "x", "y", "y"],
        );
        let spec = BinSpec {
            field: "x".into(),
            mode: BinMode::Fixed { n: 2 },
            extent: Some((0.0, 10.0)),
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: vec!["g1".into(), "g2".into()],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();

        // Output must have columns: bin_start, bin_end, count, density, g1, g2.
        let schema = out.schema();
        assert!(schema.index_of("bin_start").is_ok(), "missing bin_start");
        assert!(schema.index_of("bin_end").is_ok(), "missing bin_end");
        assert!(schema.index_of("count").is_ok(), "missing count");
        assert!(schema.index_of("density").is_ok(), "missing density");
        assert!(schema.index_of("g1").is_ok(), "missing g1 column");
        assert!(schema.index_of("g2").is_ok(), "missing g2 column");

        // 4 distinct groups × 2 bins each = 8 output rows.
        assert_eq!(out.num_rows(), 8, "expected 4 groups × 2 bins = 8 rows, got {}", out.num_rows());

        // Each group contributes 2 bins with count=2 (2 x-values per group).
        let counts = col_u64(&out, "count");
        for i in 0..8 {
            assert_eq!(counts.value(i), 1, "each bin should have count=1, row {i} has {}", counts.value(i));
        }

        // Verify both group columns are present and contain the right values.
        let g1 = col_str(&out, "g1");
        let g2 = col_str(&out, "g2");
        // Collect all (g1, g2) pairs seen in the output.
        let mut combos: std::collections::BTreeSet<(String, String)> =
            (0..out.num_rows())
            .map(|i| (g1.value(i).to_string(), g2.value(i).to_string()))
            .collect();
        assert_eq!(combos.len(), 4, "must have 4 distinct (g1,g2) groups in output");
        assert!(combos.contains(&("A".into(), "x".into())));
        assert!(combos.contains(&("A".into(), "y".into())));
        assert!(combos.contains(&("B".into(), "x".into())));
        assert!(combos.contains(&("B".into(), "y".into())));
    }

    /// D-GROUPBY-1 regression (T3.8a): a single-column groupby via `vec!["g"]`
    /// must produce the same output schema and row values as the old `Some("g")`
    /// path did before the migration.
    #[test]
    fn bin_single_column_groupby_matches_pre_migration_shape() {
        use arrow::array::StringArray;
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 9.0, 1.0, 9.0])) as arrow::array::ArrayRef,
                Arc::new(StringArray::from(vec!["A", "A", "B", "B"])) as arrow::array::ArrayRef,
            ],
        )
        .unwrap();

        let spec = BinSpec {
            field: "x".into(),
            mode: BinMode::Fixed { n: 2 },
            extent: Some((0.0, 10.0)),
            nice: false,
            cumulative: false,
            shared_extent: false,
            groupby: vec!["g".into()],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();

        // Schema: bin_start, bin_end, count, density, g
        let s = out.schema();
        assert!(s.index_of("bin_start").is_ok());
        assert!(s.index_of("bin_end").is_ok());
        assert!(s.index_of("count").is_ok());
        assert!(s.index_of("density").is_ok());
        assert!(s.index_of("g").is_ok());
        assert_eq!(s.fields().len(), 5, "single-groupby output has exactly 5 columns");

        // 2 groups × 2 bins = 4 rows.
        assert_eq!(out.num_rows(), 4);

        // Each bin should have count=1.
        let counts = col_u64(&out, "count");
        for i in 0..4 {
            assert_eq!(counts.value(i), 1);
        }
    }

    /// D-GROUPBY-1 regression (T3.8a): the BinSpec serde round-trip preserves
    /// the canonical array wire form; deserializing the old bare-string form
    /// yields a one-element vec identical to the direct vec!["col"] construction.
    #[test]
    fn bin_groupby_serde_round_trip_and_back_compat() {
        // Serialize a BinSpec with groupby=["col"] — must emit array form.
        let spec = BinSpec {
            field: "x".into(),
            mode: BinMode::Sturges,
            extent: None,
            nice: true,
            cumulative: false,
            shared_extent: false,
            groupby: vec!["col".into()],
            name: None,
        };
        let json = serde_json::to_string(&spec).unwrap();
        assert!(json.contains(r#""groupby":["col"]"#), "must serialize as array, got: {json}");

        // Deserialize a legacy JSON that has a bare string for groupby.
        let legacy = r#"{"field":"x","groupby":"col"}"#;
        let from_legacy: BinSpec = serde_json::from_str(legacy).unwrap();
        assert_eq!(from_legacy.groupby, vec!["col"], "bare-string legacy must parse to [\"col\"]");

        // Deserialize null → [].
        let null_json = r#"{"field":"x","groupby":null}"#;
        let from_null: BinSpec = serde_json::from_str(null_json).unwrap();
        assert!(from_null.groupby.is_empty(), "null groupby must parse to []");

        // Empty groupby must be omitted from serialization (skip_serializing_if).
        let empty_spec = BinSpec { groupby: vec![], ..spec.clone() };
        let empty_json = serde_json::to_string(&empty_spec).unwrap();
        assert!(!empty_json.contains("groupby"), "empty groupby must be omitted: {empty_json}");
    }

    // ── T3.8b regression tests: BinWire flat-wire shim + precedence ────────────

    fn sturges_spec() -> BinSpec {
        BinSpec {
            field: "x".into(),
            mode: BinMode::Sturges,
            extent: None,
            nice: true,
            cumulative: false,
            shared_extent: false,
            groupby: vec![],
            name: None,
        }
    }

    /// `Fixed{n}` round-trips through serde as a flat `"bin_count":n` wire key and
    /// parses back to `Fixed{n}` — never as a tagged `mode` object.
    #[test]
    fn bin_fixed_mode_serializes_flat_bin_count() {
        let spec = BinSpec { mode: BinMode::Fixed { n: 5 }, ..sturges_spec() };
        let json = serde_json::to_string(&spec).unwrap();
        assert!(json.contains(r#""bin_count":5"#), "flat bin_count expected, got: {json}");
        assert!(!json.contains("mode"), "mode must not appear on the wire: {json}");
        assert!(!json.contains("kind"), "no tagged enum on the wire: {json}");
        let back: BinSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back.mode, BinMode::Fixed { n: 5 });
        assert_eq!(back, spec);
    }

    /// `Width{w}` round-trips through serde as a flat `"bin_width":w` wire key and
    /// parses back to `Width{w}`.
    #[test]
    fn bin_width_mode_serializes_flat_bin_width() {
        let spec = BinSpec { mode: BinMode::Width { w: 2.0 }, ..sturges_spec() };
        let json = serde_json::to_string(&spec).unwrap();
        assert!(json.contains(r#""bin_width":2.0"#), "flat bin_width expected, got: {json}");
        assert!(!json.contains("bin_count"), "bin_count must be omitted for Width: {json}");
        let back: BinSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back.mode, BinMode::Width { w: 2.0 });
        assert_eq!(back, spec);
    }

    /// `Sturges` (neither count nor width) emits no `bin_count`/`bin_width` keys
    /// and parses back to `Sturges`.
    #[test]
    fn bin_sturges_mode_omits_both_flat_keys() {
        let spec = sturges_spec();
        let json = serde_json::to_string(&spec).unwrap();
        assert!(!json.contains("bin_count"), "Sturges must omit bin_count: {json}");
        assert!(!json.contains("bin_width"), "Sturges must omit bin_width: {json}");
        let back: BinSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back.mode, BinMode::Sturges);
        assert_eq!(back, spec);
    }

    /// A legacy wire that sets BOTH `bin_count` and `bin_width` resolves to
    /// `Fixed` (bin_count wins) — pins the precedence-at-boundary contract.
    #[test]
    fn bin_legacy_both_keys_resolves_to_fixed() {
        let legacy = r#"{"field":"x","bin_count":7,"bin_width":3.5}"#;
        let parsed: BinSpec = serde_json::from_str(legacy).unwrap();
        assert_eq!(
            parsed.mode,
            BinMode::Fixed { n: 7 },
            "bin_count must win over bin_width at the deserialization boundary"
        );
    }
}
