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
    empty_groupby_col, groupby_field_nullable, groupby_key_at, is_groupby_supported_dtype,
    materialize_groupby_col, KeyValue,
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
    /// Pinned extent for the per-group KDE grid. When `Some((lo, hi))`, every
    /// group evaluates its KDE over this range rather than its own data range,
    /// so all panels/groups share a comparable value axis. `render::prepare`
    /// sets this via the `global_extent` helper when faceting is active.
    /// Uses the same serde attributes as `KdeSpec::extent` and `BinSpec::extent`.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub extent: Option<(f64, f64)>,
    /// When true, all groups within a violin share the same x-grid extent.
    /// Mirrors `KdeSpec::shared_extent` and `BinSpec::shared_extent`.
    #[serde(default)]
    pub shared_extent: bool,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
    /// When true the violin is oriented horizontally: the value axis is x and
    /// the category axis is y. The density offset column emitted is
    /// `__pos_y_offset__` (computed from panel HEIGHT) instead of
    /// `__pos_x_offset__` (computed from panel WIDTH). Default false.
    #[serde(default)]
    pub horizontal: bool,
}

/// Compute the global `(lo, hi)` extent of `spec.field` over the full `batch`.
///
/// Returns `None` when the field is missing, non-numeric, or all values are
/// null/NaN. This is the pre-facet extent that `render::prepare` uses to pin the
/// value axis before partitioning, so every facet panel shares the same KDE grid
/// range (see `fix_transform_extents_for_facet`).
///
/// NICENESS CONTRACT (XFORM-08): returns the RAW `(lo, hi)` (no nicing), like the
/// KDE sibling and unlike `Bin`. Uses the shared `column_extent` helper, which
/// coerces Int64/Float32/etc. pre-facet batches to Float64 so they produce a
/// valid shared extent instead of returning None. (`apply` hard-errors on
/// non-Float64 per group, but `global_extent` is called on the raw pre-cast
/// batch, which may carry integer-typed columns.)
pub(crate) fn global_extent(spec: &ViolinSpec, batch: &RecordBatch) -> Option<(f64, f64)> {
    crate::transform::numeric_util::column_extent(batch, &spec.field)
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
        // FA-9: nullable so a null-keyed group materializes as an Arrow null.
        fields.push(Field::new(g, group_dtypes[gi].clone(), groupby_field_nullable()));
    }
    fields.push(Field::new("violin_x", DataType::Float64, false));
    fields.push(Field::new("violin_y", DataType::Float64, false));
    // __pos_x_offset__ (vertical violin) or __pos_y_offset__ (horizontal violin):
    // per-vertex pixel offset on the category axis, so the polygon mark's
    // ordinal-axis dispatch path can position each vertex relative to its
    // category band center. Computed below from panel width (vertical) or panel
    // height (horizontal) divided by n_groups.
    let offset_col_name = if spec.horizontal {
        "__pos_y_offset__"
    } else {
        "__pos_x_offset__"
    };
    fields.push(Field::new(offset_col_name, DataType::Float64, false));
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

    // Determine the evaluation extent for each group's internal KDE.
    //
    // Precedence (highest to lowest):
    //   1. `spec.extent` is `Some` — an explicit user or facet-pin; wins unconditionally.
    //      Every group uses this pinned range (existing behaviour, unchanged).
    //   2. `spec.shared_extent == true` AND the batch has >1 group — compute the
    //      cross-group global extent (min/max over ALL values, ignoring group partition)
    //      and pin every group's KDE to that range so groups share a comparable value axis.
    //      Mirrors `kde::apply_grouped` when `shared_extent=true`.
    //   3. Neither — per-group extent derived from each group's own data (backward-compat
    //      default when `shared_extent=false` and no explicit `extent`).
    let shared_kde_extent: Option<(f64, f64)> = if spec.extent.is_some() {
        // Case 1: explicit pin already present — apply_loop uses spec.extent directly.
        None // sentinel: per-group loop will use spec.extent via unwrap_or
    } else if spec.shared_extent && groups.len() > 1 {
        // Case 2: compute cross-group global extent over the full value column.
        // Delegates to the module-level `global_extent` helper, which applies the
        // same null/NaN-skipping fold and `lo < hi` finiteness guard.
        global_extent(spec, batch)
    } else {
        // Case 3: per-group extent (shared_extent=false, the backward-compat default).
        None
    };

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

        // Resolve the KDE evaluation extent for this group using the precedence
        // rule established above the loop:
        //   spec.extent (user/facet pin) > shared_kde_extent (cross-group global)
        //   > (lo, hi) (this group's own data range).
        let kde_extent = spec.extent
            .or(shared_kde_extent)
            .unwrap_or((lo, hi));
        let kde_spec = KdeSpec {
            field: spec.field.clone(),
            bandwidth: spec.bandwidth.clone(),
            bw_adjust: spec.bw_adjust,
            shared_extent: false,
            n: spec.n,
            extent: Some(kde_extent),
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
    // sharing panel pixels (width for vertical, height for horizontal), each band
    // is roughly panel_dim / n_groups wide, and we use that as the conversion
    // factor — a violin of `width=0.4` (default) then spans 80% of its band,
    // which matches the standard violin convention.
    // Without panel context (e.g. tests calling `apply` directly) we fall back
    // to a sensible per-band default so the column is still populated.
    let n_groups = groups.len().max(1);
    let band_pixels: f64 = if spec.horizontal {
        match ctx.panel_pixel_size {
            Some((_, h)) if h > 0 => (h as f64) / (n_groups as f64),
            _ => 100.0,
        }
    } else {
        match ctx.panel_pixel_size {
            Some((w, _)) if w > 0 => (w as f64) / (n_groups as f64),
            _ => 100.0,
        }
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
    #[pyo3(signature = (field, *, groupby = vec![], bandwidth = None, bw_adjust = 1.0, n = 256, width = 0.4, shared_extent = false, horizontal = false, name = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        field: &str,
        groupby: Vec<String>,
        bandwidth: Option<&Bound<'_, PyAny>>,
        bw_adjust: f64,
        n: usize,
        width: f64,
        shared_extent: bool,
        horizontal: bool,
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
                extent: None,
                shared_extent,
                name,
                horizontal,
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
            extent: None,
            shared_extent: false,
            name: None,
            horizontal: false,
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
            extent: None,
            shared_extent: false,
            name: None,
            horizontal: false,
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
                extent: None,
                shared_extent: false,
                name: None,
                horizontal: false,
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
            extent: None,
            shared_extent: false,
            name: None,
            horizontal: false,
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
            extent: None,
            shared_extent: false,
            name: None,
            horizontal: false,
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
            extent: None,
            shared_extent: false,
            name: None,
            horizontal: false,
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
            extent: None,
            shared_extent: false,
            name: None,
            horizontal: false,
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

    // ── FA-9: null groupby key regression ─────────────────────────────────────

    fn batch_value_int64_group_nullable(
        values: Vec<Option<f64>>,
        groups: Vec<Option<i64>>,
    ) -> RecordBatch {
        use arrow::array::Int64Array;
        let schema = Arc::new(Schema::new(vec![
            Field::new("v",   DataType::Float64, true),
            Field::new("grp", DataType::Int64,   true),
        ]));
        let v_arr = Float64Array::from(values);
        let g_arr = Int64Array::from(groups);
        RecordBatch::try_new(schema, vec![Arc::new(v_arr), Arc::new(g_arr)]).unwrap()
    }

    /// FA-9 regression: a null value in an Int64 groupby column must not cause
    /// RecordBatch::try_new to panic or error. Both the null-keyed group and the
    /// real-keyed group must appear in the output (two distinct group_id values),
    /// and the null key must materialize as an Arrow null in the groupby column.
    #[test]
    fn violin_null_int64_groupby_key_survives_and_is_distinct() {
        pyo3::Python::initialize();
        use arrow::array::Int64Array;

        // null group: 50 observations in [0, 49]
        // real group (grp=42): 50 observations in [100, 149]
        let mut values: Vec<Option<f64>> = (0..50).map(|i| Some(i as f64)).collect();
        values.extend((100..150).map(|i| Some(i as f64)));
        let mut groups: Vec<Option<i64>> = vec![None; 50];
        groups.extend(vec![Some(42); 50]);

        let batch = batch_value_int64_group_nullable(values, groups);
        let spec = ViolinSpec {
            field: "v".into(),
            groupby: vec!["grp".into()],
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 16,
            width: 0.4,
            extent: None,
            shared_extent: false,
            name: None,
            horizontal: false,
        };
        // Must not error: the non-nullable schema declaration was the bug.
        let out = apply(&spec, &batch).unwrap();
        assert!(out.num_rows() > 0, "output must be non-empty");

        // Both groups must appear: two distinct group_id values.
        let gids = col_u32(&out, "group_id");
        let distinct_gids: std::collections::BTreeSet<u32> = gids.iter().copied().collect();
        assert_eq!(
            distinct_gids.len(), 2,
            "expected 2 distinct group_ids (null + 42); got {:?}",
            distinct_gids
        );

        // FA-9: the null-keyed group materializes as an Arrow null in the output column.
        let grp_arr = out.column(out.schema().index_of("grp").unwrap())
            .as_any().downcast_ref::<Int64Array>().unwrap();
        let has_null_key = (0..grp_arr.len()).any(|i| grp_arr.is_null(i));
        assert!(has_null_key, "null-key group must materialize as Arrow null in 'grp' column");

        // The real-keyed group (grp=42) must also appear.
        let has_real_key = (0..grp_arr.len()).any(|i| !grp_arr.is_null(i) && grp_arr.value(i) == 42);
        assert!(has_real_key, "real-key group (grp=42) must be present in 'grp' column");
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
            extent: None,
            shared_extent: false,
            name: None,
            horizontal: false,
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

    // ── Task 8: extent field + global_extent helper ───────────────────────────

    /// ViolinSpec with `extent: Some((lo, hi))` must use the pinned extent for
    /// ALL groups, not the per-group data range. Both groups must produce violin_y
    /// values spanning the full pinned range rather than their own data range.
    #[test]
    fn violin_pinned_extent_applies_to_all_groups() {
        pyo3::Python::initialize();
        // Group "a": values in [0, 9]. Group "b": values in [100, 109].
        // Without pinning: group "a" would have y in ~[0,9], group "b" in ~[100,109].
        // With pinning to (0, 109): both groups must produce violin_y spanning 0..=109.
        let mut vals: Vec<f64> = (0..10).map(|i| i as f64).collect();
        vals.extend((100..110).map(|i| i as f64));
        let mut grps: Vec<&str> = vec!["a"; 10];
        grps.extend(vec!["b"; 10]);
        let b = batch_value_group(vals, grps);

        let pinned_lo = 0.0_f64;
        let pinned_hi = 109.0_f64;
        let spec = ViolinSpec {
            field: "v".into(),
            groupby: vec!["group".into()],
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 32,
            width: 0.4,
            extent: Some((pinned_lo, pinned_hi)),
            shared_extent: false,
            name: None,
            horizontal: false,
        };
        let out = apply(&spec, &b).unwrap();
        let ys = col_f64(&out, "violin_y");
        let groups_out = col_str(&out, "group");

        // For EVERY group, violin_y must span the pinned range
        // (first y value should be near pinned_lo, last near pinned_hi).
        for gname in &["a", "b"] {
            let group_ys: Vec<f64> = ys
                .iter()
                .zip(groups_out.iter())
                .filter(|(_, g)| g.as_str() == *gname)
                .map(|(&y, _)| y)
                .collect();
            assert!(!group_ys.is_empty(), "group '{gname}' must have output rows");
            let min_y = group_ys.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_y = group_ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            assert!(
                (min_y - pinned_lo).abs() < 1e-9,
                "group '{gname}': violin_y min should be {pinned_lo}, got {min_y}"
            );
            assert!(
                (max_y - pinned_hi).abs() < 1e-9,
                "group '{gname}': violin_y max should be {pinned_hi}, got {max_y}"
            );
        }
    }

    /// ViolinSpec with `extent: None` must produce per-group extents (backward compat).
    #[test]
    fn violin_no_pinned_extent_uses_per_group_range() {
        pyo3::Python::initialize();
        // Group "a": [0, 9]. Group "b": [100, 109].
        // Without pinning: each group's violin_y must be within its own data range.
        let mut vals: Vec<f64> = (0..10).map(|i| i as f64).collect();
        vals.extend((100..110).map(|i| i as f64));
        let mut grps: Vec<&str> = vec!["a"; 10];
        grps.extend(vec!["b"; 10]);
        let b = batch_value_group(vals, grps);
        let spec = ViolinSpec {
            field: "v".into(),
            groupby: vec!["group".into()],
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 32,
            width: 0.4,
            extent: None,
            shared_extent: false,
            name: None,
            horizontal: false,
        };
        let out = apply(&spec, &b).unwrap();
        let ys = col_f64(&out, "violin_y");
        let groups_out = col_str(&out, "group");

        // Group "a" rows should all have y in [0, 9] (per-group range).
        let a_ys: Vec<f64> = ys
            .iter()
            .zip(groups_out.iter())
            .filter(|(_, g)| g.as_str() == "a")
            .map(|(&y, _)| y)
            .collect();
        let a_max = a_ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        // Without pinning, group "a" should not have y values reaching 100 (group b's range).
        assert!(
            a_max < 50.0,
            "group 'a' without pinning: violin_y max should be near 9, got {a_max}"
        );
    }

    // ── Task R6: shared_extent wiring ─────────────────────────────────────────

    /// R6 — shared_extent=true pins all groups to the cross-group global extent.
    ///
    /// Groups "a" ([0,9]) and "b" ([100,109]) have disjoint per-group ranges.
    /// With shared_extent=true, every group's violin_y must span the full
    /// cross-group range [0, 109] instead of each group's own narrow range.
    #[test]
    fn violin_shared_extent_true_pins_all_groups_to_global_range() {
        pyo3::Python::initialize();
        // Group "a": values in [0, 9]. Group "b": values in [100, 109].
        let mut vals: Vec<f64> = (0..10).map(|i| i as f64).collect();
        vals.extend((100..110).map(|i| i as f64));
        let mut grps: Vec<&str> = vec!["a"; 10];
        grps.extend(vec!["b"; 10]);
        let b = batch_value_group(vals, grps);

        let spec = ViolinSpec {
            field: "v".into(),
            groupby: vec!["group".into()],
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 32,
            width: 0.4,
            extent: None,
            shared_extent: true, // <-- the field under test
            name: None,
            horizontal: false,
        };
        let out = apply(&spec, &b).unwrap();
        let ys = col_f64(&out, "violin_y");
        let groups_out = col_str(&out, "group");

        // Global extent over the full batch is [0, 109].
        let expected_lo = 0.0_f64;
        let expected_hi = 109.0_f64;

        // Every group must have violin_y values that span the cross-group range.
        for gname in &["a", "b"] {
            let group_ys: Vec<f64> = ys
                .iter()
                .zip(groups_out.iter())
                .filter(|(_, g)| g.as_str() == *gname)
                .map(|(&y, _)| y)
                .collect();
            assert!(!group_ys.is_empty(), "group '{gname}' must have output rows");
            let min_y = group_ys.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_y = group_ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            assert!(
                (min_y - expected_lo).abs() < 1e-9,
                "shared_extent=true: group '{gname}' violin_y min should be {expected_lo}, got {min_y}"
            );
            assert!(
                (max_y - expected_hi).abs() < 1e-9,
                "shared_extent=true: group '{gname}' violin_y max should be {expected_hi}, got {max_y}"
            );
        }
    }

    /// R6 regression — shared_extent=false keeps per-group extents (backward compat).
    ///
    /// Same disjoint groups as above, but with shared_extent=false each group
    /// must be pinned only to its own data range, NOT the cross-group global.
    #[test]
    fn violin_shared_extent_false_uses_per_group_range() {
        pyo3::Python::initialize();
        // Group "a": values in [0, 9]. Group "b": values in [100, 109].
        let mut vals: Vec<f64> = (0..10).map(|i| i as f64).collect();
        vals.extend((100..110).map(|i| i as f64));
        let mut grps: Vec<&str> = vec!["a"; 10];
        grps.extend(vec!["b"; 10]);
        let b = batch_value_group(vals, grps);

        let spec = ViolinSpec {
            field: "v".into(),
            groupby: vec!["group".into()],
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 32,
            width: 0.4,
            extent: None,
            shared_extent: false, // <-- per-group behavior
            name: None,
            horizontal: false,
        };
        let out = apply(&spec, &b).unwrap();
        let ys = col_f64(&out, "violin_y");
        let groups_out = col_str(&out, "group");

        // Group "a" violin_y must stay in [0, 9] — NOT reaching 100 (group b's range).
        let a_ys: Vec<f64> = ys
            .iter()
            .zip(groups_out.iter())
            .filter(|(_, g)| g.as_str() == "a")
            .map(|(&y, _)| y)
            .collect();
        let a_max = a_ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            a_max < 50.0,
            "shared_extent=false: group 'a' violin_y max should be near 9 (per-group), got {a_max}"
        );

        // Group "b" violin_y must stay in [100, 109] — NOT reaching 0 (group a's range).
        let b_ys: Vec<f64> = ys
            .iter()
            .zip(groups_out.iter())
            .filter(|(_, g)| g.as_str() == "b")
            .map(|(&y, _)| y)
            .collect();
        let b_min = b_ys.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            b_min > 50.0,
            "shared_extent=false: group 'b' violin_y min should be near 100 (per-group), got {b_min}"
        );
    }

    /// R6 — explicit `extent=Some(...)` takes precedence over shared_extent.
    ///
    /// Even when shared_extent=true, if the user (or facet-prepare) has set
    /// spec.extent explicitly, that pinned extent must win for ALL groups.
    #[test]
    fn violin_explicit_extent_wins_over_shared_extent() {
        pyo3::Python::initialize();
        // Group "a": [0, 9]. Group "b": [100, 109]. Cross-group global = [0, 109].
        // Explicit pin = (-10, 200) — wider than the global; must win.
        let mut vals: Vec<f64> = (0..10).map(|i| i as f64).collect();
        vals.extend((100..110).map(|i| i as f64));
        let mut grps: Vec<&str> = vec!["a"; 10];
        grps.extend(vec!["b"; 10]);
        let b = batch_value_group(vals, grps);

        let pinned_lo = -10.0_f64;
        let pinned_hi = 200.0_f64;
        let spec = ViolinSpec {
            field: "v".into(),
            groupby: vec!["group".into()],
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 32,
            width: 0.4,
            extent: Some((pinned_lo, pinned_hi)), // explicit pin
            shared_extent: true,                  // must NOT override the explicit pin
            name: None,
            horizontal: false,
        };
        let out = apply(&spec, &b).unwrap();
        let ys = col_f64(&out, "violin_y");
        let groups_out = col_str(&out, "group");

        for gname in &["a", "b"] {
            let group_ys: Vec<f64> = ys
                .iter()
                .zip(groups_out.iter())
                .filter(|(_, g)| g.as_str() == *gname)
                .map(|(&y, _)| y)
                .collect();
            assert!(!group_ys.is_empty(), "group '{gname}' must have output rows");
            let min_y = group_ys.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_y = group_ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            assert!(
                (min_y - pinned_lo).abs() < 1e-9,
                "explicit extent wins: group '{gname}' violin_y min should be {pinned_lo}, got {min_y}"
            );
            assert!(
                (max_y - pinned_hi).abs() < 1e-9,
                "explicit extent wins: group '{gname}' violin_y max should be {pinned_hi}, got {max_y}"
            );
        }
    }

    /// R6 — shared_extent=true with only ONE group falls back to per-group behavior.
    ///
    /// shared_extent only applies when there are multiple groups. A single-group
    /// violin with shared_extent=true must behave identically to shared_extent=false
    /// (per-group data range), since there is nothing to share.
    #[test]
    fn violin_shared_extent_single_group_behaves_as_per_group() {
        pyo3::Python::initialize();
        let vals: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let b = batch("v", vals);

        let spec_shared = ViolinSpec {
            field: "v".into(),
            groupby: vec![],
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 32,
            width: 0.4,
            extent: None,
            shared_extent: true,
            name: None,
            horizontal: false,
        };
        let spec_per_group = ViolinSpec {
            shared_extent: false,
            ..spec_shared.clone()
        };

        let out_shared = apply(&spec_shared, &b).unwrap();
        let out_per = apply(&spec_per_group, &b).unwrap();

        let ys_shared = col_f64(&out_shared, "violin_y");
        let ys_per = col_f64(&out_per, "violin_y");

        // Both must produce the same y values (byte-identical since only 1 group exists).
        assert_eq!(
            ys_shared, ys_per,
            "single-group violin: shared_extent=true must produce identical output to shared_extent=false"
        );
    }

    /// Serde round-trip: ViolinSpec with explicit `extent`, `shared_extent`, and
    /// `horizontal` round-trips through JSON with all fields preserved.
    #[test]
    fn violin_serde_round_trip_with_new_fields() {
        // Test both horizontal=false and horizontal=true round-trips.
        for horizontal in [false, true] {
            let spec = ViolinSpec {
                field: "val".into(),
                groupby: vec!["cat".into()],
                bandwidth: BandwidthSpec::Scott,
                bw_adjust: 1.5,
                n: 128,
                width: 0.3,
                extent: Some((0.0, 10.0)),
                shared_extent: true,
                name: Some("violin_out".into()),
                horizontal,
            };
            let json = serde_json::to_string(&spec).unwrap();
            let parsed: ViolinSpec = serde_json::from_str(&json).unwrap();
            assert_eq!(
                parsed, spec,
                "ViolinSpec round-trip must preserve all fields (horizontal={horizontal})"
            );
        }
    }

    /// Serde backward compat: JSON without `extent` and `shared_extent` (as
    /// emitted by old Python code) must deserialize with defaults applied.
    #[test]
    fn violin_serde_missing_new_fields_defaults_backward_compat() {
        // Minimal JSON that a pre-Task-8 Python might emit (no extent/shared_extent).
        let json = r#"{
            "field": "val",
            "groupby": [],
            "bandwidth": {"kind": "scott"},
            "bw_adjust": 1.0,
            "n": 256,
            "width": 0.4
        }"#;
        let parsed: ViolinSpec = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.extent, None, "extent must default to None");
        assert!(!parsed.shared_extent, "shared_extent must default to false");
        assert_eq!(parsed.name, None, "name must default to None");
        assert!(!parsed.horizontal, "horizontal must default to false");
    }

    /// `global_extent` returns the full-data range over the value field.
    #[test]
    fn violin_global_extent_returns_full_data_range() {
        let vals: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let b = batch("v", vals);
        let spec = ViolinSpec {
            field: "v".into(),
            groupby: vec![],
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 32,
            width: 0.4,
            extent: None,
            shared_extent: false,
            name: None,
            horizontal: false,
        };
        let ext = global_extent(&spec, &b);
        assert_eq!(ext, Some((0.0, 99.0)), "global_extent should return (0.0, 99.0)");
    }

    /// `global_extent` returns None when the field is missing.
    #[test]
    fn violin_global_extent_missing_field_returns_none() {
        let b = batch("v", vec![1.0, 2.0, 3.0]);
        let spec = ViolinSpec {
            field: "nonexistent".into(),
            groupby: vec![],
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 32,
            width: 0.4,
            extent: None,
            shared_extent: false,
            name: None,
            horizontal: false,
        };
        let ext = global_extent(&spec, &b);
        assert_eq!(ext, None, "global_extent on missing field must return None");
    }

    /// `global_extent` returns None when all values are null.
    #[test]
    fn violin_global_extent_all_null_returns_none() {
        use arrow::array::Float64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, true)]));
        let arr = Float64Array::from(vec![None::<f64>, None, None]);
        let b = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        let spec = ViolinSpec {
            field: "v".into(),
            groupby: vec![],
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 32,
            width: 0.4,
            extent: None,
            shared_extent: false,
            name: None,
            horizontal: false,
        };
        let ext = global_extent(&spec, &b);
        assert_eq!(ext, None, "global_extent on all-null field must return None");
    }

    // ── F1 regression: int-field shared-extent gap ────────────────────
    //
    // Before the fix, `global_extent` used raw `downcast_ref::<Float64Array>()`
    // which returns None for an Int64 column. A faceted violin on an integer
    // field would silently get no shared-extent pin (bug #7 for int inputs).
    // After the fix, `coerce_to_float64` is used, mirroring `kde::global_extent`.

    /// `global_extent` on an Int64 column returns Some((min, max)) — not None.
    #[test]
    fn violin_global_extent_int64_field_returns_some() {
        use arrow::array::Int64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
        let arr = Int64Array::from(vec![3i64, 1, 7, 2, 9]);
        let b = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        let spec = ViolinSpec {
            field: "v".into(),
            groupby: vec![],
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 32,
            width: 0.4,
            extent: None,
            shared_extent: false,
            name: None,
            horizontal: false,
        };
        // Pre-fix: raw downcast returns None for Int64 → no shared extent.
        // Post-fix: coerce_to_float64 casts Int64 → Float64 → Some((1.0, 9.0)).
        let ext = global_extent(&spec, &b);
        assert_eq!(ext, Some((1.0, 9.0)), "global_extent on Int64 column must coerce and return Some");
    }

    /// `global_extent` on a Float64 column remains byte-identical after the fix.
    #[test]
    fn violin_global_extent_float64_field_unchanged() {
        let vals: Vec<f64> = vec![2.5, 0.1, 8.8, 3.3];
        let b = batch("v", vals);
        let spec = ViolinSpec {
            field: "v".into(),
            groupby: vec![],
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 32,
            width: 0.4,
            extent: None,
            shared_extent: false,
            name: None,
            horizontal: false,
        };
        let ext = global_extent(&spec, &b);
        assert_eq!(ext, Some((0.1, 8.8)), "global_extent on Float64 must be unchanged by fix");
    }

    // ── T0.4: horizontal orientation fork ─────────────────────────────────────
    //
    // Before this fix, `ViolinSpec` had no `horizontal` field and
    // `apply_with_context` always emitted `__pos_x_offset__` computed from
    // panel WIDTH regardless of violin orientation. For horizontal violins the
    // density must expand along the y (category) axis, so the transform must
    // emit `__pos_y_offset__` from panel HEIGHT instead.
    //
    // These tests fail on pre-fix code because:
    //   1. `ViolinSpec` has no `horizontal` field → struct literal fails to compile.
    //   2. Even if horizontal were somehow true, the output schema would still
    //      contain `__pos_x_offset__` not `__pos_y_offset__`.

    /// T0.4 — `horizontal=false` (vertical, default) emits `__pos_x_offset__`
    /// computed from panel WIDTH, and does NOT emit `__pos_y_offset__`.
    ///
    /// Fails on pre-fix code because `ViolinSpec` lacks the `horizontal` field.
    #[test]
    fn violin_vertical_emits_pos_x_offset_from_panel_width() {
        pyo3::Python::initialize();
        let vals: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let b = batch("v", vals);

        let spec = ViolinSpec {
            field: "v".into(),
            groupby: vec![],
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 32,
            width: 0.4,
            extent: None,
            shared_extent: false,
            name: None,
            horizontal: false, // vertical (default)
        };

        // Panel is 600×400 px, 1 group → band = 600/1 = 600 px.
        let ctx = crate::transform::context::TransformContext {
            panel_pixel_size: Some((600, 400)),
            ..Default::default()
        };
        let out = apply_with_context(&spec, &b, &ctx).unwrap();
        let schema = out.schema();

        // Must have __pos_x_offset__, must NOT have __pos_y_offset__.
        assert!(
            schema.index_of("__pos_x_offset__").is_ok(),
            "vertical violin must emit __pos_x_offset__; schema: {:?}",
            schema.fields().iter().map(|f| f.name().as_str()).collect::<Vec<_>>()
        );
        assert!(
            schema.index_of("__pos_y_offset__").is_err(),
            "vertical violin must NOT emit __pos_y_offset__; schema: {:?}",
            schema.fields().iter().map(|f| f.name().as_str()).collect::<Vec<_>>()
        );

        // Spot-check: the offset is proportional to panel WIDTH (600), not height.
        // violin_x values are in [-0.4, 0.4]; offset = violin_x * 600.
        let violin_xs = col_f64(&out, "violin_x");
        let offsets = col_f64(&out, "__pos_x_offset__");
        for (i, (&vx, &off)) in violin_xs.iter().zip(offsets.iter()).enumerate() {
            let expected = vx * 600.0;
            assert!(
                (off - expected).abs() < 1e-9,
                "__pos_x_offset__[{i}]: expected {expected} (violin_x={vx} × panel_w=600), got {off}"
            );
        }
    }

    /// T0.4 — `horizontal=true` emits `__pos_y_offset__` computed from panel
    /// HEIGHT, and does NOT emit `__pos_x_offset__`.
    ///
    /// Fails on pre-fix code because `ViolinSpec` lacks the `horizontal` field,
    /// and even if it existed, the output schema would always name the column
    /// `__pos_x_offset__` and use panel WIDTH.
    #[test]
    fn violin_horizontal_emits_pos_y_offset_from_panel_height() {
        pyo3::Python::initialize();
        let vals: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let b = batch("v", vals);

        let spec = ViolinSpec {
            field: "v".into(),
            groupby: vec![],
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 32,
            width: 0.4,
            extent: None,
            shared_extent: false,
            name: None,
            horizontal: true, // horizontal orientation
        };

        // Panel is 600×400 px, 1 group → band = 400/1 = 400 px.
        let ctx = crate::transform::context::TransformContext {
            panel_pixel_size: Some((600, 400)),
            ..Default::default()
        };
        let out = apply_with_context(&spec, &b, &ctx).unwrap();
        let schema = out.schema();

        // Must have __pos_y_offset__, must NOT have __pos_x_offset__.
        assert!(
            schema.index_of("__pos_y_offset__").is_ok(),
            "horizontal violin must emit __pos_y_offset__; schema: {:?}",
            schema.fields().iter().map(|f| f.name().as_str()).collect::<Vec<_>>()
        );
        assert!(
            schema.index_of("__pos_x_offset__").is_err(),
            "horizontal violin must NOT emit __pos_x_offset__; schema: {:?}",
            schema.fields().iter().map(|f| f.name().as_str()).collect::<Vec<_>>()
        );

        // Spot-check: the offset is proportional to panel HEIGHT (400), not width.
        // violin_x values are in [-0.4, 0.4]; offset = violin_x * 400.
        let violin_xs = col_f64(&out, "violin_x");
        let offsets = col_f64(&out, "__pos_y_offset__");
        for (i, (&vx, &off)) in violin_xs.iter().zip(offsets.iter()).enumerate() {
            let expected = vx * 400.0;
            assert!(
                (off - expected).abs() < 1e-9,
                "__pos_y_offset__[{i}]: expected {expected} (violin_x={vx} × panel_h=400), got {off}"
            );
        }
    }

    /// T0.4 — `horizontal=true` fallback (no panel context) uses 100px default
    /// for panel HEIGHT and emits `__pos_y_offset__`.
    #[test]
    fn violin_horizontal_no_panel_context_fallback_emits_pos_y_offset() {
        pyo3::Python::initialize();
        let vals: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let b = batch("v", vals);

        let spec = ViolinSpec {
            field: "v".into(),
            groupby: vec![],
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 16,
            width: 0.4,
            extent: None,
            shared_extent: false,
            name: None,
            horizontal: true,
        };

        // No panel context → fallback band_pixels = 100.
        let ctx = crate::transform::context::TransformContext::default();
        let out = apply_with_context(&spec, &b, &ctx).unwrap();
        let schema = out.schema();

        assert!(
            schema.index_of("__pos_y_offset__").is_ok(),
            "horizontal violin without panel context must still emit __pos_y_offset__"
        );
        assert!(
            schema.index_of("__pos_x_offset__").is_err(),
            "horizontal violin without panel context must NOT emit __pos_x_offset__"
        );

        // Offsets must be violin_x * 100.0 (the fallback band_pixels).
        let violin_xs = col_f64(&out, "violin_x");
        let offsets = col_f64(&out, "__pos_y_offset__");
        for (i, (&vx, &off)) in violin_xs.iter().zip(offsets.iter()).enumerate() {
            let expected = vx * 100.0;
            assert!(
                (off - expected).abs() < 1e-9,
                "__pos_y_offset__[{i}]: expected {expected} (violin_x={vx} × fallback=100), got {off}"
            );
        }
    }
}
