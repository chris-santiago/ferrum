//! Violin: per-group KDE → mirrored closed polygon vertices.
//!
//! For each group, runs KDE over the group's values to obtain a grid of (value, density)
//! pairs, normalizes density by `width / max_density` so the violin is exactly `width`
//! wide at its peak, and emits 2N vertices forming a closed polygon:
//!   - Right side bottom→top:  i = 0..n  → (+normalized_density, value)
//!   - Left side top→bottom:   i = (n-1)..=0 → (-normalized_density, value)
//!
//! Output schema = group_id (u32) + groupby cols + violin_x (f64) + violin_y (f64).

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::transform::group_key::{
    empty_groupby_col, groupby_key_at, is_groupby_supported_dtype, materialize_groupby_col,
    KeyValue,
};
use crate::transform::kde::{self, BandwidthSpec, KdeSpec};

fn default_bandwidth() -> BandwidthSpec {
    BandwidthSpec::Scott
}
fn default_violin_n() -> usize {
    256
}
fn default_violin_width() -> f64 {
    0.4
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ViolinSpec {
    pub field: String,
    #[serde(default)]
    pub groupby: Vec<String>,
    #[serde(default = "default_bandwidth")]
    pub bandwidth: BandwidthSpec,
    #[serde(default = "kde::default_bw_adjust")]
    pub bw_adjust: f64,
    #[serde(default = "default_violin_n")]
    pub n: usize,
    #[serde(default = "default_violin_width")]
    pub width: f64,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

pub(crate) fn apply(spec: &ViolinSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    apply_with_context(spec, batch, &crate::transform::context::TransformContext::default())
}

pub(crate) fn apply_with_context(
    spec: &ViolinSpec,
    batch: &RecordBatch,
    ctx: &crate::transform::context::TransformContext,
) -> PyResult<RecordBatch> {
    let schema = batch.schema();

    let v_idx = schema.index_of(&spec.field).map_err(|_| {
        PyValueError::new_err(format!("stat_violin: column '{}' not found", spec.field))
    })?;
    if schema.field(v_idx).data_type() != &DataType::Float64 {
        return Err(PyValueError::new_err(format!(
            "stat_violin: column '{}' must be Float64",
            spec.field
        )));
    }

    let mut group_dtypes: Vec<DataType> = Vec::with_capacity(spec.groupby.len());
    for g in &spec.groupby {
        let i = schema.index_of(g).map_err(|_| {
            PyValueError::new_err(format!(
                "stat_violin: groupby column '{}' not found",
                g
            ))
        })?;
        let dt = schema.field(i).data_type().clone();
        if !is_groupby_supported_dtype(&dt) {
            return Err(PyValueError::new_err(format!(
                "stat_violin: groupby column '{}' has unsupported dtype {:?}; \
                 supported: Utf8/LargeUtf8, Float64/Float32, \
                 Int8/Int16/Int32/Int64, UInt8/UInt16/UInt32/UInt64, Boolean",
                g, dt
            )));
        }
        group_dtypes.push(dt);
    }

    let n_rows = batch.num_rows();

    // Build output schema up-front so empty input still returns a valid RecordBatch.
    let mut fields: Vec<Field> = Vec::with_capacity(spec.groupby.len() + 3);
    fields.push(Field::new("group_id", DataType::UInt32, false));
    for (gi, g) in spec.groupby.iter().enumerate() {
        fields.push(Field::new(g, group_dtypes[gi].clone(), false));
    }
    fields.push(Field::new("violin_x", DataType::Float64, false));
    fields.push(Field::new("violin_y", DataType::Float64, false));
    // __pos_x_offset__ = per-vertex pixel offset on the category axis, so the
    // polygon mark's ordinal-x dispatch path can position each vertex relative
    // to its category band center. Computed below from panel width / n_groups.
    fields.push(Field::new("__pos_x_offset__", DataType::Float64, false));
    let out_schema = Arc::new(Schema::new(fields));

    if n_rows == 0 {
        let mut cols: Vec<ArrayRef> = Vec::with_capacity(spec.groupby.len() + 4);
        cols.push(Arc::new(UInt32Array::from(Vec::<u32>::new())));
        for dt in &group_dtypes {
            cols.push(empty_groupby_col(dt).map_err(PyValueError::new_err)?);
        }
        cols.push(Arc::new(Float64Array::from(Vec::<f64>::new())));
        cols.push(Arc::new(Float64Array::from(Vec::<f64>::new())));
        cols.push(Arc::new(Float64Array::from(Vec::<f64>::new())));
        return RecordBatch::try_new(out_schema, cols)
            .map_err(|e| PyValueError::new_err(format!("stat_violin: {e}")));
    }

    let v_arr = batch
        .column(v_idx)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| PyValueError::new_err("stat_violin: expected Float64Array for value column"))?;

    // Bucket rows by group key (BTreeMap → deterministic ordering).
    let mut groups: BTreeMap<Vec<KeyValue>, Vec<usize>> = BTreeMap::new();
    let group_arrays: Vec<&dyn arrow::array::Array> = spec
        .groupby
        .iter()
        .map(|g| batch.column(schema.index_of(g).expect("invariant: groupby columns validated above")).as_ref())
        .collect();

    for row in 0..n_rows {
        let mut key = Vec::with_capacity(spec.groupby.len());
        for (gi, arr) in group_arrays.iter().enumerate() {
            let kv = groupby_key_at(*arr, &group_dtypes[gi], row).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "stat_violin: internal error extracting groupby key at row {row}"
                ))
            })?;
            key.push(kv);
        }
        groups.entry(key).or_default().push(row);
    }
    if spec.groupby.is_empty() {
        let all: Vec<usize> = (0..n_rows).collect();
        groups.clear();
        groups.insert(Vec::new(), all);
    }

    // Per-group KDE → mirrored polygon vertices.
    let mut group_ids: Vec<u32> = Vec::new();
    let mut keys_out: Vec<Vec<KeyValue>> = Vec::new();
    let mut violin_x: Vec<f64> = Vec::new();
    let mut violin_y: Vec<f64> = Vec::new();

    for (gid, (key, rows)) in groups.iter().enumerate() {
        let vals: Vec<f64> = rows
            .iter()
            .filter_map(|&r| {
                if v_arr.is_null(r) {
                    return None;
                }
                let v = v_arr.value(r);
                if v.is_nan() {
                    return None;
                }
                Some(v)
            })
            .collect();

        if vals.is_empty() {
            continue;
        }

        let (lo, hi) = vals
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                (a.min(v), b.max(v))
            });
        if !(lo.is_finite() && hi.is_finite()) {
            continue;
        }
        // Degenerate group: a single observation or all-equal values produce
        // lo == hi (zero-width KDE domain). Emit a minimal zero-width polygon
        // (4 vertices at violin_x=0, violin_y=lo) so the output batch is
        // non-empty and scale_resolve can find finite values in violin_y. The
        // polygon mark's `ring.len() < 3` guard will still skip rendering it —
        // a 1-point violin is correctly invisible — but the column is populated.
        if lo >= hi {
            for _ in 0..4 {
                group_ids.push(gid as u32);
                keys_out.push(key.clone());
                violin_x.push(0.0);
                violin_y.push(lo);
            }
            continue;
        }

        // Build a synthetic 1-col Float64 batch for kde::apply.
        let synth_schema = Arc::new(Schema::new(vec![Field::new(
            spec.field.as_str(),
            DataType::Float64,
            false,
        )]));
        let synth_batch = RecordBatch::try_new(
            synth_schema,
            vec![Arc::new(Float64Array::from(vals.clone()))],
        )
        .map_err(|e| PyValueError::new_err(format!("stat_violin: synth batch: {e}")))?;

        let kde_spec = KdeSpec {
            field: spec.field.clone(),
            bandwidth: spec.bandwidth.clone(),
            bw_adjust: spec.bw_adjust,
            shared_extent: false,
            n: spec.n,
            extent: Some((lo, hi)),
            cumulative: false,
            kernel: crate::transform::kde::default_kernel(),
            groupby: None,
            name: None,
        };
        let kde_out = kde::apply(&kde_spec, &synth_batch)?;
        let value_col = kde_out
            .column(kde_out.schema().index_of("value")
                .map_err(|_| PyValueError::new_err("stat_violin: KDE output missing 'value' column"))?)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| PyValueError::new_err("stat_violin: expected Float64Array for KDE 'value' column"))?
            .clone();
        let density_col = kde_out
            .column(kde_out.schema().index_of("density")
                .map_err(|_| PyValueError::new_err("stat_violin: KDE output missing 'density' column"))?)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| PyValueError::new_err("stat_violin: expected Float64Array for KDE 'density' column"))?
            .clone();

        // Find max density (ignore NaN).
        let mut max_density = f64::NEG_INFINITY;
        for i in 0..density_col.len() {
            if !density_col.is_null(i) {
                let d = density_col.value(i);
                if d.is_finite() && d > max_density {
                    max_density = d;
                }
            }
        }
        if !(max_density.is_finite()) || max_density <= 0.0 {
            continue;
        }

        let scale = spec.width / max_density;
        let n = spec.n;

        // Right side: bottom→top  (i = 0..n) → +scale * density
        for i in 0..n {
            let d = if density_col.is_null(i) {
                0.0
            } else {
                let dv = density_col.value(i);
                if dv.is_finite() {
                    dv
                } else {
                    0.0
                }
            };
            group_ids.push(gid as u32);
            keys_out.push(key.clone());
            violin_x.push(scale * d);
            violin_y.push(value_col.value(i));
        }
        // Left side: top→bottom  (i in (0..n).rev()) → -scale * density
        for i in (0..n).rev() {
            let d = if density_col.is_null(i) {
                0.0
            } else {
                let dv = density_col.value(i);
                if dv.is_finite() {
                    dv
                } else {
                    0.0
                }
            };
            group_ids.push(gid as u32);
            keys_out.push(key.clone());
            violin_x.push(-scale * d);
            violin_y.push(value_col.value(i));
        }
    }

    let mut cols: Vec<ArrayRef> = Vec::with_capacity(spec.groupby.len() + 3);
    cols.push(Arc::new(UInt32Array::from(group_ids)));
    for (gi, dt) in group_dtypes.iter().enumerate() {
        cols.push(materialize_groupby_col(&keys_out, gi, dt).map_err(PyValueError::new_err)?);
    }
    // Convert violin_x (in normalized polygon-width units; |violin_x| ≤ spec.width)
    // to per-vertex pixel offsets on the category axis. With n_groups categories
    // sharing panel_w pixels, each band is roughly panel_w / n_groups wide, and
    // we use that as the conversion factor — a violin of `width=0.4` (default)
    // then spans 80% of its band, which matches the standard violin convention.
    // Without panel context (e.g. tests calling `apply` directly) we fall back
    // to a sensible per-band default so the column is still populated.
    let n_groups = groups.len().max(1);
    let band_pixels: f64 = match ctx.panel_pixel_size {
        Some((w, _)) if w > 0 => (w as f64) / (n_groups as f64),
        _ => 100.0,
    };
    let pos_x_offset: Vec<f64> = violin_x.iter().map(|x| x * band_pixels).collect();

    cols.push(Arc::new(Float64Array::from(violin_x)));
    cols.push(Arc::new(Float64Array::from(violin_y)));
    cols.push(Arc::new(Float64Array::from(pos_x_offset)));

    RecordBatch::try_new(out_schema, cols)
        .map_err(|e| PyValueError::new_err(format!("stat_violin: {e}")))
}

use pyo3::prelude::*;

/// Per-group KDE density curves for violin plots.
///
/// Computes a 1D Gaussian KDE for each group defined by ``groupby`` and
/// maps the density values to a half-width in [0, ``width``] for mirrored
/// ribbon rendering. The output feeds the ribbon mark in composite violin
/// charts.
///
/// Output columns: ``group_id`` (UInt32 sequential group identifier),
/// groupby columns (if any), ``violin_x`` (Float64, mirrored normalized
/// density), and ``violin_y`` (Float64, value grid points). Vertices form
/// a closed polygon: right side bottom→top then left side top→bottom.
///
/// Parameters
/// ----------
/// field : str
///     Numeric column to estimate (must be Float64).
/// groupby : list of str, default []
///     Columns to group by; a separate density curve is produced per group.
/// bandwidth : float or {"scott", "silverman"}, optional
///     Kernel bandwidth. Default is ``"scott"``.
/// n : int, default 256
///     Number of evaluation grid points per group. Must be > 0.
/// width : float, default 0.4
///     Maximum half-width of the violin in category-axis data units.
///     Must be > 0.
/// name : str, optional
///     Named output label for sibling ``Reorder(from_=...)`` lookup.
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> fm.Chart(df).mark_violin().encode(x="cut", y="price")
#[pyclass(eq, module = "ferrum._core", name = "Violin")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyViolin(pub(crate) crate::transform::core::TransformSpec);

#[pymethods]
impl PyViolin {
    #[new]
    #[pyo3(signature = (field, *, groupby = vec![], bandwidth = None, bw_adjust = 1.0, n = 256, width = 0.4, name = None))]
    fn new(
        field: &str,
        groupby: Vec<String>,
        bandwidth: Option<&Bound<'_, PyAny>>,
        bw_adjust: f64,
        n: usize,
        width: f64,
        name: Option<String>,
    ) -> PyResult<Self> {
        if !bw_adjust.is_finite() || bw_adjust <= 0.0 {
            return Err(PyValueError::new_err("Violin: bw_adjust must be a positive finite number"));
        }
        if field.is_empty() {
            return Err(PyValueError::new_err("Violin: field must be non-empty"));
        }
        if n == 0 {
            return Err(PyValueError::new_err("Violin: n must be > 0"));
        }
        if !width.is_finite() || width <= 0.0 {
            return Err(PyValueError::new_err(
                "Violin: width must be a positive finite number",
            ));
        }
        let bw = match bandwidth {
            None => BandwidthSpec::Scott,
            Some(obj) => {
                if let Ok(s) = obj.extract::<String>() {
                    match s.as_str() {
                        "scott" => BandwidthSpec::Scott,
                        "silverman" => BandwidthSpec::Silverman,
                        other => {
                            return Err(PyValueError::new_err(format!(
                                "Violin: unknown bandwidth '{other}'; expected 'scott' | 'silverman' | float"
                            )))
                        }
                    }
                } else if let Ok(v) = obj.extract::<f64>() {
                    if !v.is_finite() || v <= 0.0 {
                        return Err(PyValueError::new_err(
                            "Violin: numeric bandwidth must be a positive finite number",
                        ));
                    }
                    BandwidthSpec::Fixed { value: v }
                } else {
                    return Err(PyValueError::new_err(
                        "Violin: bandwidth must be 'scott', 'silverman', or a positive float",
                    ));
                }
            }
        };
        let mut seen = std::collections::HashSet::new();
        for g in &groupby {
            if !seen.insert(g.as_str()) {
                return Err(PyValueError::new_err(format!(
                    "Violin: duplicate groupby field '{g}'"
                )));
            }
        }
        Ok(Self(crate::transform::core::TransformSpec::Violin(
            ViolinSpec {
                field: field.to_string(),
                groupby,
                bandwidth: bw,
                bw_adjust,
                n,
                width,
                name,
            },
        )))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            crate::transform::core::TransformSpec::Violin(s) => format!(
                "Violin(field='{}', groupby={:?}, bandwidth={:?}, n={}, width={})",
                s.field, s.groupby, s.bandwidth, s.n, s.width
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, RecordBatch, StringArray, UInt32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn batch(field: &str, values: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(field, DataType::Float64, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(values))]).unwrap()
    }

    fn batch_value_group(values: Vec<f64>, groups: Vec<&str>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Float64, false),
            Field::new("group", DataType::Utf8, false),
        ]));
        let v = Float64Array::from(values);
        let g = StringArray::from(groups);
        RecordBatch::try_new(schema, vec![Arc::new(v), Arc::new(g)]).unwrap()
    }

    fn col_f64(b: &RecordBatch, name: &str) -> Vec<f64> {
        let arr = b
            .column(b.schema().index_of(name).unwrap())
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        (0..arr.len())
            .map(|i| if arr.is_null(i) { f64::NAN } else { arr.value(i) })
            .collect()
    }

    fn col_u32(b: &RecordBatch, name: &str) -> Vec<u32> {
        let arr = b
            .column(b.schema().index_of(name).unwrap())
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        (0..arr.len()).map(|i| arr.value(i)).collect()
    }

    fn col_str(b: &RecordBatch, name: &str) -> Vec<String> {
        let arr = b
            .column(b.schema().index_of(name).unwrap())
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        (0..arr.len()).map(|i| arr.value(i).to_string()).collect()
    }

    #[test]
    fn violin_polygon_vertex_count_and_symmetry() {
        pyo3::Python::initialize();
        let vals: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let b = batch("v", vals);
        let spec = ViolinSpec {
            field: "v".into(),
            groupby: vec![],
            bandwidth: BandwidthSpec::Scott,
        bw_adjust: 1.0,
            n: 64,
            width: 0.5,
            name: None,
        };
        let out = apply(&spec, &b).unwrap();
        let n = 64usize;
        assert_eq!(out.num_rows(), 2 * n, "expected 2*n = 128 rows");

        let xs = col_f64(&out, "violin_x");
        let ys = col_f64(&out, "violin_y");
        // For each i in 0..n, row i is right-side and row (2n-1-i) is left-side mirror.
        for i in 0..n {
            let mirror = 2 * n - 1 - i;
            assert!(
                (xs[i] + xs[mirror]).abs() < 1e-12,
                "x[{i}] should be negation of x[{mirror}]; got {} vs {}",
                xs[i],
                xs[mirror]
            );
            assert!(
                (ys[i] - ys[mirror]).abs() < 1e-12,
                "y[{i}] should equal y[{mirror}]; got {} vs {}",
                ys[i],
                ys[mirror]
            );
        }
    }

    #[test]
    fn violin_per_group_distinct_group_ids() {
        pyo3::Python::initialize();
        // Group "a": small values; Group "b": large values.
        let mut vals: Vec<f64> = Vec::new();
        let mut grps: Vec<&str> = Vec::new();
        for i in 0..50 {
            vals.push(i as f64);
            grps.push("a");
        }
        for i in 0..50 {
            vals.push(100.0 + i as f64);
            grps.push("b");
        }
        let b = batch_value_group(vals, grps);
        let spec = ViolinSpec {
            field: "v".into(),
            groupby: vec!["group".into()],
            bandwidth: BandwidthSpec::Scott,
        bw_adjust: 1.0,
            n: 32,
            width: 0.4,
            name: None,
        };
        let out = apply(&spec, &b).unwrap();
        assert_eq!(out.num_rows(), 2 * 32 * 2, "expected 128 rows");

        let gids = col_u32(&out, "group_id");
        let groups = col_str(&out, "group");

        // Distinct group ids present.
        let mut distinct: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
        for g in &gids {
            distinct.insert(*g);
        }
        assert!(distinct.contains(&0), "expected group_id 0 present");
        assert!(distinct.contains(&1), "expected group_id 1 present");

        // group_id=0 rows all share one group key value; group_id=1 the other.
        let g0_keys: std::collections::BTreeSet<&str> = gids
            .iter()
            .enumerate()
            .filter(|(_, gid)| **gid == 0)
            .map(|(i, _)| groups[i].as_str())
            .collect();
        let g1_keys: std::collections::BTreeSet<&str> = gids
            .iter()
            .enumerate()
            .filter(|(_, gid)| **gid == 1)
            .map(|(i, _)| groups[i].as_str())
            .collect();
        assert_eq!(g0_keys.len(), 1, "group_id=0 rows must share one key");
        assert_eq!(g1_keys.len(), 1, "group_id=1 rows must share one key");
        assert_ne!(g0_keys, g1_keys, "group_id=0 and =1 must hold different keys");
    }

    #[test]
    fn violin_bandwidth_variants_parse_and_apply() {
        pyo3::Python::initialize();
        let vals: Vec<f64> = (0..60).map(|i| (i as f64) * 0.1).collect();
        let n = 32usize;
        for bw in [
            BandwidthSpec::Scott,
            BandwidthSpec::Silverman,
            BandwidthSpec::Fixed { value: 0.5 },
        ] {
            let b = batch("v", vals.clone());
            let spec = ViolinSpec {
                field: "v".into(),
                groupby: vec![],
                bandwidth: bw.clone(),
        bw_adjust: 1.0,
                n,
                width: 0.4,
                name: None,
            };
            let out = apply(&spec, &b).unwrap();
            assert_eq!(out.num_rows(), 2 * n, "bw {:?}: expected {} rows", bw, 2 * n);
            let xs = col_f64(&out, "violin_x");
            assert!(
                xs.iter().all(|x| !x.is_nan()),
                "bw {:?}: violin_x must contain no NaN",
                bw
            );
        }
    }

    // --- Regression tests for the single-observation / all-equal-values fix ---
    //
    // Before the fix, `apply_with_context` skipped groups with `vals.len() < 2`.
    // This caused `scale_resolve` to raise `EmptyDomain` because no rows were
    // emitted for degenerate groups, leaving `violin_y` empty. The fix:
    //   1. Changed the guard from `vals.len() < 2` to `vals.is_empty()`.
    //   2. Added a `lo >= hi` branch that emits 4 degenerate polygon vertices
    //      at `(violin_x=0.0, violin_y=lo)` instead of skipping the group.

    /// A group with exactly one observation must produce non-zero output rows
    /// and populate `violin_y` with finite values.
    ///
    /// Fails with old `vals.len() < 2` guard because the single-element group
    /// would be skipped, resulting in zero output rows.
    #[test]
    fn violin_single_element_group_emits_rows() {
        pyo3::Python::initialize();
        let b = batch("v", vec![42.0]);
        let spec = ViolinSpec {
            field: "v".into(),
            groupby: vec![],
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 64,
            width: 0.4,
            name: None,
        };
        let out = apply(&spec, &b).unwrap();

        // Must produce non-zero rows (old `< 2` guard would have produced 0).
        assert!(
            out.num_rows() > 0,
            "single-element group must produce non-zero output rows; got 0"
        );

        // `violin_y` must contain at least one finite value.
        let ys = col_f64(&out, "violin_y");
        assert!(
            ys.iter().any(|y| y.is_finite()),
            "violin_y must contain at least one finite value; got {:?}",
            ys
        );
    }

    /// A group where all values are identical (lo == hi) must produce output
    /// rows whose `violin_y` values all equal the single observed value.
    ///
    /// Fails if the `lo >= hi` degenerate branch is absent, because KDE over
    /// a zero-width domain would be skipped or produce NaN density, and the
    /// group would be dropped.
    #[test]
    fn violin_all_equal_values_emits_degenerate_polygon() {
        pyo3::Python::initialize();
        let lo = 7.5_f64;
        let b = batch("v", vec![lo; 5]);
        let spec = ViolinSpec {
            field: "v".into(),
            groupby: vec![],
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 64,
            width: 0.4,
            name: None,
        };
        let out = apply(&spec, &b).unwrap();

        // Must produce output rows.
        assert!(
            out.num_rows() > 0,
            "all-equal-values group must produce output rows; got 0"
        );

        // All `violin_y` values must equal lo (the single observed value).
        let ys = col_f64(&out, "violin_y");
        for (i, &y) in ys.iter().enumerate() {
            assert!(
                (y - lo).abs() < 1e-12,
                "violin_y[{i}] should be {lo} for degenerate group; got {y}"
            );
        }
    }

    /// A batch with one normal multi-observation group and one single-observation
    /// group must produce output rows for both groups.  The result must contain
    /// two distinct `group_id` values, proving the single-observation group was
    /// not silently dropped.
    ///
    /// Fails with old `vals.len() < 2` guard because the single-observation
    /// group ("b") would be skipped and only group_id=0 would appear.
    #[test]
    fn violin_mixed_groups_both_appear_in_output() {
        pyo3::Python::initialize();
        // Group "a": 50 observations spread over [0, 49].
        // Group "b": exactly 1 observation.
        let mut vals: Vec<f64> = (0..50).map(|i| i as f64).collect();
        vals.push(99.0);
        let mut grps: Vec<&str> = vec!["a"; 50];
        grps.push("b");

        let b = batch_value_group(vals, grps);
        let spec = ViolinSpec {
            field: "v".into(),
            groupby: vec!["group".into()],
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 32,
            width: 0.4,
            name: None,
        };
        let out = apply(&spec, &b).unwrap();

        // There must be rows in the output.
        assert!(out.num_rows() > 0, "output must be non-empty");

        // Both groups must appear: two distinct group_id values.
        let gids = col_u32(&out, "group_id");
        let distinct: std::collections::BTreeSet<u32> = gids.iter().copied().collect();
        assert_eq!(
            distinct.len(),
            2,
            "expected 2 distinct group_ids (one per group); got {:?}",
            distinct
        );
        assert!(distinct.contains(&0), "group_id 0 must be present");
        assert!(distinct.contains(&1), "group_id 1 must be present");
    }

    // ── FA-7: shared group-keying — integer groupby support ──────────────────

    fn batch_value_int64_group(values: Vec<f64>, groups: Vec<i64>) -> RecordBatch {
        use arrow::array::Int64Array;
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Float64, false),
            Field::new("grp", DataType::Int64, false),
        ]));
        let v = Float64Array::from(values);
        let g = Int64Array::from(groups);
        RecordBatch::try_new(schema, vec![Arc::new(v), Arc::new(g)]).unwrap()
    }

    fn col_i64(b: &RecordBatch, name: &str) -> Vec<i64> {
        use arrow::array::Int64Array;
        let arr = b
            .column(b.schema().index_of(name).unwrap())
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        (0..arr.len()).map(|i| arr.value(i)).collect()
    }

    /// FA-7: violin grouped on an Int64 column groups correctly and preserves
    /// the Int64 output dtype. Before FA-7 the private `KeyValue` enum rejected
    /// non-Float64/Utf8 groupby columns with an error.
    #[test]
    fn violin_int64_groupby_succeeds_and_preserves_dtype() {
        pyo3::Python::initialize();
        let mut vals: Vec<f64> = Vec::new();
        let mut grps: Vec<i64> = Vec::new();
        for i in 0..50 {
            vals.push(i as f64);
            grps.push(2020);
        }
        for i in 0..50 {
            vals.push(100.0 + i as f64);
            grps.push(2021);
        }
        let b = batch_value_int64_group(vals, grps);
        let spec = ViolinSpec {
            field: "v".into(),
            groupby: vec!["grp".into()],
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 32,
            width: 0.4,
            name: None,
        };
        let out = apply(&spec, &b).unwrap();
        // Output groupby column must retain Int64 dtype.
        assert_eq!(
            out.schema().field(out.schema().index_of("grp").unwrap()).data_type(),
            &DataType::Int64
        );
        let gids = col_u32(&out, "group_id");
        let distinct: std::collections::BTreeSet<u32> = gids.iter().copied().collect();
        assert_eq!(distinct.len(), 2, "expected 2 distinct group_ids; got {:?}", distinct);
        // Both year keys must be present, materialized as Int64.
        let grp_vals: std::collections::BTreeSet<i64> = col_i64(&out, "grp").into_iter().collect();
        assert!(grp_vals.contains(&2020), "key 2020 must be present; got {:?}", grp_vals);
        assert!(grp_vals.contains(&2021), "key 2021 must be present; got {:?}", grp_vals);
    }

    /// FA-7 byte-stability guard: a Utf8 groupby still materializes a StringArray
    /// with identical key values after migrating to the shared group_key module.
    #[test]
    fn violin_utf8_groupby_byte_stable() {
        pyo3::Python::initialize();
        let mut vals: Vec<f64> = (0..50).map(|i| i as f64).collect();
        vals.extend((0..50).map(|i| 100.0 + i as f64));
        let mut grps: Vec<&str> = vec!["a"; 50];
        grps.extend(vec!["b"; 50]);
        let b = batch_value_group(vals, grps);
        let spec = ViolinSpec {
            field: "v".into(),
            groupby: vec!["group".into()],
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 32,
            width: 0.4,
            name: None,
        };
        let out = apply(&spec, &b).unwrap();
        assert_eq!(
            out.schema().field(out.schema().index_of("group").unwrap()).data_type(),
            &DataType::Utf8
        );
        let keys: std::collections::BTreeSet<String> = col_str(&out, "group").into_iter().collect();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains("a") && keys.contains("b"));
    }
}
