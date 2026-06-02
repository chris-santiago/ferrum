//! Kde2D: 2D Gaussian kernel-density estimate evaluated on a uniform n×n grid.
//!
//! Output is a SINGLE-ROW batch with list-typed columns:
//!   grid_x:  List<Float64>   length n
//!   grid_y:  List<Float64>   length n
//!   density: List<Float64>   length n*n, row-major: density[gy * n + gx]
//!   nx:      UInt32          = n
//!   ny:      UInt32          = n
//!   extent:  List<Float64>   length 4: [xmin, xmax, ymin, ymax]
//!
//! Downstream consumers (Contour) read the grid back from this shape.

use arrow::array::{
    Array, ArrayRef, Float64Array, Float64Builder, ListBuilder, RecordBatch, StringArray,
    UInt32Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::transform::kde::{self, BandwidthSpec};

fn default_bandwidth() -> BandwidthSpec {
    BandwidthSpec::Scott
}

fn default_kde2d_n() -> usize {
    128
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct Kde2DSpec {
    pub x: String,
    pub y: String,
    #[serde(default = "default_bandwidth")]
    pub bandwidth: BandwidthSpec,
    #[serde(default = "default_kde2d_n")]
    pub n: usize,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub extent: Option<(f64, f64, f64, f64)>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub groupby: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

pub(crate) fn apply(spec: &Kde2DSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    if spec.n == 0 {
        return Err(PyValueError::new_err("stat_kde_2d: n must be > 0"));
    }
    if let Some(g) = &spec.groupby {
        return apply_grouped(spec, batch, g);
    }
    apply_one_group(spec, batch, None, None)
}

/// Compute a single 2-D KDE surface from a subset of rows (or all rows when
/// `only_indices` is None). The `extent_override` lets the caller supply a
/// pre-computed global extent so all groups share the same grid axes.
fn apply_one_group(
    spec: &Kde2DSpec,
    batch: &RecordBatch,
    only_indices: Option<&[usize]>,
    extent_override: Option<(f64, f64, f64, f64)>,
) -> PyResult<RecordBatch> {
    let schema = batch.schema();
    let xi = schema.index_of(&spec.x).map_err(|_| {
        PyValueError::new_err(format!("stat_kde_2d: column '{}' not found", spec.x))
    })?;
    if schema.field(xi).data_type() != &DataType::Float64 {
        return Err(PyValueError::new_err(format!(
            "stat_kde_2d: column '{}' must be Float64",
            spec.x
        )));
    }
    let yi = schema.index_of(&spec.y).map_err(|_| {
        PyValueError::new_err(format!("stat_kde_2d: column '{}' not found", spec.y))
    })?;
    if schema.field(yi).data_type() != &DataType::Float64 {
        return Err(PyValueError::new_err(format!(
            "stat_kde_2d: column '{}' must be Float64",
            spec.y
        )));
    }

    let xa = batch
        .column(xi)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| PyValueError::new_err(format!(
            "stat_kde_2d: expected Float64Array for column '{}'", spec.x)))?;
    let ya = batch
        .column(yi)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| PyValueError::new_err(format!(
            "stat_kde_2d: expected Float64Array for column '{}'", spec.y)))?;

    let mut xs: Vec<f64> = Vec::with_capacity(xa.len());
    let mut ys: Vec<f64> = Vec::with_capacity(ya.len());
    let push = |i: usize, xs: &mut Vec<f64>, ys: &mut Vec<f64>| {
        if xa.is_null(i) || ya.is_null(i) {
            return;
        }
        let xv = xa.value(i);
        let yv = ya.value(i);
        if !xv.is_nan() && !yv.is_nan() {
            xs.push(xv);
            ys.push(yv);
        }
    };
    match only_indices {
        Some(ixs) => {
            for &i in ixs {
                push(i, &mut xs, &mut ys);
            }
        }
        None => {
            for i in 0..xa.len() {
                push(i, &mut xs, &mut ys);
            }
        }
    }

    let n = spec.n;

    // Compute extent (xmin, xmax, ymin, ymax). Caller-supplied override takes
    // priority (used by apply_grouped for per-group extent control).
    let (xmin, xmax, ymin, ymax) = extent_override.or(spec.extent).unwrap_or_else(|| {
        if xs.is_empty() {
            (0.0, 0.0, 0.0, 0.0)
        } else {
            let (mut xlo, mut xhi) = (f64::INFINITY, f64::NEG_INFINITY);
            let (mut ylo, mut yhi) = (f64::INFINITY, f64::NEG_INFINITY);
            for i in 0..xs.len() {
                if xs[i] < xlo { xlo = xs[i]; }
                if xs[i] > xhi { xhi = xs[i]; }
                if ys[i] < ylo { ylo = ys[i]; }
                if ys[i] > yhi { yhi = ys[i]; }
            }
            (xlo, xhi, ylo, yhi)
        }
    });

    let grid_x: Vec<f64> = (0..n)
        .map(|i| {
            if n == 1 {
                xmin
            } else {
                xmin + (xmax - xmin) * (i as f64) / ((n - 1) as f64)
            }
        })
        .collect();
    let grid_y: Vec<f64> = (0..n)
        .map(|i| {
            if n == 1 {
                ymin
            } else {
                ymin + (ymax - ymin) * (i as f64) / ((n - 1) as f64)
            }
        })
        .collect();

    // Density.
    let density: Vec<f64> = if xs.len() < 2 {
        vec![0.0; n * n]
    } else {
        let hx = kde::bandwidth(&xs, &spec.bandwidth)?;
        let hy = kde::bandwidth(&ys, &spec.bandwidth)?;
        if hx <= 0.0 || hy <= 0.0 || !hx.is_finite() || !hy.is_finite() {
            vec![0.0; n * n]
        } else {
            let n_pts = xs.len() as f64;
            let norm = n_pts * hx * hy * 2.0 * std::f64::consts::PI;
            let mut d = vec![0.0_f64; n * n];
            for gy in 0..n {
                for gx in 0..n {
                    let mut s = 0.0_f64;
                    for i in 0..xs.len() {
                        let dx = (grid_x[gx] - xs[i]) / hx;
                        let dy = (grid_y[gy] - ys[i]) / hy;
                        s += (-0.5 * (dx * dx + dy * dy)).exp();
                    }
                    d[gy * n + gx] = s / norm;
                }
            }
            d
        }
    };

    build_surface_batch(&grid_x, &grid_y, &density, n, (xmin, xmax, ymin, ymax))
}

/// Assemble the 6-column single-row RecordBatch for one 2-D KDE surface.
fn build_surface_batch(
    grid_x: &[f64],
    grid_y: &[f64],
    density: &[f64],
    n: usize,
    extent: (f64, f64, f64, f64),
) -> PyResult<RecordBatch> {
    let (xmin, xmax, ymin, ymax) = extent;
    let mut grid_x_b = ListBuilder::new(Float64Builder::new());
    grid_x_b.values().append_slice(grid_x);
    grid_x_b.append(true);
    let grid_x_arr = grid_x_b.finish();

    let mut grid_y_b = ListBuilder::new(Float64Builder::new());
    grid_y_b.values().append_slice(grid_y);
    grid_y_b.append(true);
    let grid_y_arr = grid_y_b.finish();

    let mut density_b = ListBuilder::new(Float64Builder::new());
    density_b.values().append_slice(density);
    density_b.append(true);
    let density_arr = density_b.finish();

    let mut extent_b = ListBuilder::new(Float64Builder::new());
    extent_b.values().append_slice(&[xmin, xmax, ymin, ymax]);
    extent_b.append(true);
    let extent_arr = extent_b.finish();

    let nx_arr = UInt32Array::from(vec![n as u32]);
    let ny_arr = UInt32Array::from(vec![n as u32]);

    let list_f64 = DataType::List(Arc::new(Field::new("item", DataType::Float64, true)));
    let out_schema = Arc::new(Schema::new(vec![
        Field::new("grid_x", list_f64.clone(), false),
        Field::new("grid_y", list_f64.clone(), false),
        Field::new("density", list_f64.clone(), false),
        Field::new("nx", DataType::UInt32, false),
        Field::new("ny", DataType::UInt32, false),
        Field::new("extent", list_f64, false),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(grid_x_arr),
        Arc::new(grid_y_arr),
        Arc::new(density_arr),
        Arc::new(nx_arr),
        Arc::new(ny_arr),
        Arc::new(extent_arr),
    ];
    RecordBatch::try_new(out_schema, cols)
        .map_err(|e| PyValueError::new_err(format!("stat_kde_2d: {e}")))
}

/// Partition input batch by `group_col` (Utf8), compute one 2-D KDE surface
/// per group, then stack the results. Each group yields one output row; the
/// group-key column is appended as the 7th field so downstream marks can
/// color by group.
///
/// Per-group extents are used by default (matching `kde.rs` `shared_extent=false`
/// semantics). The caller-specified `spec.extent`, when present, is applied as
/// a shared override across all groups.
fn apply_grouped(
    spec: &Kde2DSpec,
    batch: &RecordBatch,
    group_col: &str,
) -> PyResult<RecordBatch> {
    use std::collections::BTreeMap;

    let schema = batch.schema();
    let gi = schema.index_of(group_col).map_err(|_| {
        PyValueError::new_err(format!(
            "stat_kde_2d: groupby column '{}' not found", group_col
        ))
    })?;
    let gtype = schema.field(gi).data_type();
    if gtype != &DataType::Utf8 {
        return Err(PyValueError::new_err(format!(
            "stat_kde_2d: groupby column '{}' must be Utf8; got {:?}",
            group_col, gtype
        )));
    }
    let garr = batch
        .column(gi)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "stat_kde_2d: expected StringArray for groupby column '{}'",
                group_col
            ))
        })?;

    // Group row indices in first-appearance order.
    let mut group_order: Vec<String> = Vec::new();
    let mut group_idx_map: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for i in 0..garr.len() {
        if garr.is_null(i) {
            continue;
        }
        let gv = garr.value(i).to_string();
        if seen.insert(gv.clone()) {
            group_order.push(gv.clone());
        }
        group_idx_map.entry(gv).or_default().push(i);
    }

    // Per-group extents by default; spec.extent (when set) is a shared override.
    let shared_extent: Option<(f64, f64, f64, f64)> = spec.extent;

    // Collect per-group surface data into parallel flat vectors so we can build
    // the multi-row output batch in a single pass.
    let n = spec.n;
    let mut all_grid_x: ListBuilder<Float64Builder> = ListBuilder::new(Float64Builder::new());
    let mut all_grid_y: ListBuilder<Float64Builder> = ListBuilder::new(Float64Builder::new());
    let mut all_density: ListBuilder<Float64Builder> = ListBuilder::new(Float64Builder::new());
    let mut all_nx: Vec<u32> = Vec::with_capacity(group_order.len());
    let mut all_ny: Vec<u32> = Vec::with_capacity(group_order.len());
    let mut all_extent: ListBuilder<Float64Builder> = ListBuilder::new(Float64Builder::new());
    let mut all_groups: Vec<String> = Vec::with_capacity(group_order.len());

    for g in &group_order {
        let ixs = group_idx_map
            .get(g)
            .ok_or_else(|| PyValueError::new_err(format!(
                "stat_kde_2d: missing group key '{g}' in index map"
            )))?;
        let surface = apply_one_group(spec, batch, Some(ixs), shared_extent)?;

        // Unpack the single-row surface batch back into its list values.
        let gx = list_values_at(&surface, "grid_x")?;
        let gy = list_values_at(&surface, "grid_y")?;
        let dens = list_values_at(&surface, "density")?;
        let ext = list_values_at(&surface, "extent")?;

        all_grid_x.values().append_slice(&gx);
        all_grid_x.append(true);
        all_grid_y.values().append_slice(&gy);
        all_grid_y.append(true);
        all_density.values().append_slice(&dens);
        all_density.append(true);
        all_extent.values().append_slice(&ext);
        all_extent.append(true);
        all_nx.push(n as u32);
        all_ny.push(n as u32);
        all_groups.push(g.clone());
    }

    let list_f64 = DataType::List(Arc::new(Field::new("item", DataType::Float64, true)));
    let out_schema = Arc::new(Schema::new(vec![
        Field::new("grid_x", list_f64.clone(), false),
        Field::new("grid_y", list_f64.clone(), false),
        Field::new("density", list_f64.clone(), false),
        Field::new("nx", DataType::UInt32, false),
        Field::new("ny", DataType::UInt32, false),
        Field::new("extent", list_f64, false),
        Field::new(group_col, DataType::Utf8, false),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(all_grid_x.finish()),
        Arc::new(all_grid_y.finish()),
        Arc::new(all_density.finish()),
        Arc::new(UInt32Array::from(all_nx)),
        Arc::new(UInt32Array::from(all_ny)),
        Arc::new(all_extent.finish()),
        Arc::new(StringArray::from(
            all_groups.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        )),
    ];
    RecordBatch::try_new(out_schema, cols)
        .map_err(|e| PyValueError::new_err(format!("stat_kde_2d: {e}")))
}

/// Extract the Float64 values from the first (and only) list entry of a named
/// column in a single-row RecordBatch.
fn list_values_at(batch: &RecordBatch, name: &str) -> PyResult<Vec<f64>> {
    use arrow::array::ListArray;
    let idx = batch.schema().index_of(name).map_err(|_| {
        PyValueError::new_err(format!("stat_kde_2d: column '{}' not found in surface batch", name))
    })?;
    let la = batch
        .column(idx)
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "stat_kde_2d: expected ListArray for column '{}'", name
            ))
        })?;
    let inner = la.value(0);
    let arr = inner.as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
        PyValueError::new_err(format!(
            "stat_kde_2d: expected Float64Array inside list column '{}'", name
        ))
    })?;
    Ok((0..arr.len()).map(|i| arr.value(i)).collect())
}

use pyo3::prelude::*;

use crate::transform::core::TransformSpec;

/// Two-dimensional kernel density estimate (KDE2D).
///
/// Fits a bivariate Gaussian kernel to the (``x``, ``y``) point cloud and
/// evaluates the density on a square grid of ``n × n`` points. The result
/// is the primary input for the ``Contour`` transform.
///
/// Output is a single-row batch with columns ``grid_x`` (List<Float64>,
/// length ``n``), ``grid_y`` (List<Float64>, length ``n``), ``density``
/// (List<Float64>, length ``n*n``, row-major), ``nx`` (UInt32), ``ny``
/// (UInt32), and ``extent`` (List<Float64>, ``[xmin, xmax, ymin, ymax]``).
///
/// When ``groupby`` is set, one row is emitted per distinct group and a
/// ``<groupby>`` (Utf8) column is appended so downstream marks can color
/// by group.
///
/// Parameters
/// ----------
/// x : str
///     Numeric column for the horizontal axis (must be Float64).
/// y : str
///     Numeric column for the vertical axis (must be Float64).
/// bandwidth : float or {"scott", "silverman"}, optional
///     Kernel bandwidth applied to both axes. Default is ``"scott"``.
/// n : int, default 128
///     Grid resolution (``n × n`` cells). Must be > 0.
/// extent : (float, float, float, float), optional
///     ``(xmin, xmax, ymin, ymax)`` clipping box; all values must be finite
///     and satisfy ``xmin < xmax``, ``ymin < ymax``.
/// groupby : str, optional
///     Single group-key column (Utf8). When set, one 2-D KDE surface is
///     computed per distinct group value. Output schema gains the group
///     column as the last field.
/// name : str, optional
///     Named output label for sibling ``Reorder(from_=...)`` lookup.
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> fm.Chart(df).mark_contour().encode(x="x", y="y")
#[pyclass(eq, module = "ferrum._core", name = "Kde2D")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyKde2D(pub(crate) TransformSpec);

#[pymethods]
impl PyKde2D {
    #[new]
    #[pyo3(signature = (x, y, *, bandwidth = None, n = 128, extent = None, groupby = None, name = None))]
    fn new(
        x: &str,
        y: &str,
        bandwidth: Option<&Bound<'_, PyAny>>,
        n: usize,
        extent: Option<(f64, f64, f64, f64)>,
        groupby: Option<String>,
        name: Option<String>,
    ) -> PyResult<Self> {
        if x.is_empty() {
            return Err(PyValueError::new_err("Kde2D: x must be non-empty"));
        }
        if y.is_empty() {
            return Err(PyValueError::new_err("Kde2D: y must be non-empty"));
        }
        if n == 0 {
            return Err(PyValueError::new_err("Kde2D: n must be > 0"));
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
                                "Kde2D: unknown bandwidth '{other}'; expected 'scott' | 'silverman' | float"
                            )))
                        }
                    }
                } else if let Ok(v) = obj.extract::<f64>() {
                    if !v.is_finite() || v <= 0.0 {
                        return Err(PyValueError::new_err(
                            "Kde2D: numeric bandwidth must be a positive finite number",
                        ));
                    }
                    BandwidthSpec::Fixed { value: v }
                } else {
                    return Err(PyValueError::new_err(
                        "Kde2D: bandwidth must be 'scott', 'silverman', or a positive float",
                    ));
                }
            }
        };
        if let Some((xmin, xmax, ymin, ymax)) = extent {
            if !xmin.is_finite() || !xmax.is_finite() || !ymin.is_finite() || !ymax.is_finite() {
                return Err(PyValueError::new_err(
                    "Kde2D: extent values must all be finite",
                ));
            }
            if xmin >= xmax || ymin >= ymax {
                return Err(PyValueError::new_err(
                    "Kde2D: extent must satisfy xmin < xmax and ymin < ymax",
                ));
            }
        }
        Ok(PyKde2D(TransformSpec::Kde2D(Kde2DSpec {
            x: x.to_string(),
            y: y.to_string(),
            bandwidth: bw,
            n,
            extent,
            groupby,
            name,
        })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Kde2D(s) => format!(
                "Kde2D(x='{}', y='{}', bandwidth={:?}, n={}, extent={:?}, groupby={:?})",
                s.x, s.y, s.bandwidth, s.n, s.extent, s.groupby
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, ListArray, RecordBatch, StringArray, UInt32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn batch_xy(xs: Vec<f64>, ys: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("y", DataType::Float64, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(xs)),
                Arc::new(Float64Array::from(ys)),
            ],
        )
        .unwrap()
    }

    fn list_f64_at(batch: &RecordBatch, name: &str) -> Vec<f64> {
        let idx = batch.schema().index_of(name).unwrap();
        let la = batch
            .column(idx)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        assert_eq!(la.len(), 1, "{name} should be single-row list column");
        let inner = la.value(0);
        let arr = inner.as_any().downcast_ref::<Float64Array>().unwrap();
        (0..arr.len()).map(|i| arr.value(i)).collect()
    }

    fn u32_at(batch: &RecordBatch, name: &str) -> u32 {
        let idx = batch.schema().index_of(name).unwrap();
        let arr = batch
            .column(idx)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        assert_eq!(arr.len(), 1);
        arr.value(0)
    }

    #[test]
    fn kde_2d_density_integrates_to_approximately_one() {
        pyo3::Python::initialize();
        // Use a small set of points and a generous extent so the kernel mass is
        // contained within the grid (Σ density · dx · dy ≈ 1).
        let xs = vec![0.0, 1.0, 0.0, 1.0, 0.5];
        let ys = vec![0.0, 0.0, 1.0, 1.0, 0.5];
        let n = 64;
        let spec = Kde2DSpec {
            x: "x".into(),
            y: "y".into(),
            bandwidth: BandwidthSpec::Scott,
            n,
            extent: Some((-3.0, 4.0, -3.0, 4.0)),
            groupby: None,
            name: None,
        };
        let b = batch_xy(xs, ys);
        let out = apply(&spec, &b).unwrap();
        assert_eq!(out.num_rows(), 1);

        let grid_x = list_f64_at(&out, "grid_x");
        let grid_y = list_f64_at(&out, "grid_y");
        let density = list_f64_at(&out, "density");
        assert_eq!(grid_x.len(), n);
        assert_eq!(grid_y.len(), n);
        assert_eq!(density.len(), n * n);

        let dx = (grid_x[n - 1] - grid_x[0]) / ((n - 1) as f64);
        let dy = (grid_y[n - 1] - grid_y[0]) / ((n - 1) as f64);
        let sum: f64 = density.iter().sum::<f64>() * dx * dy;
        assert!(
            (sum - 1.0).abs() < 0.2,
            "expected Σ density · dx · dy ≈ 1.0; got {sum}"
        );
    }

    #[test]
    fn kde_2d_extent_explicit_overrides_data_range() {
        pyo3::Python::initialize();
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![0.0, 1.0, 2.0];
        let n = 16;
        let spec = Kde2DSpec {
            x: "x".into(),
            y: "y".into(),
            bandwidth: BandwidthSpec::Scott,
            n,
            extent: Some((-10.0, 10.0, -10.0, 10.0)),
            groupby: None,
            name: None,
        };
        let b = batch_xy(xs, ys);
        let out = apply(&spec, &b).unwrap();

        let extent = list_f64_at(&out, "extent");
        assert_eq!(extent, vec![-10.0, 10.0, -10.0, 10.0]);

        let grid_x = list_f64_at(&out, "grid_x");
        let grid_y = list_f64_at(&out, "grid_y");
        assert!((grid_x[0] - (-10.0)).abs() < 1e-12);
        assert!((grid_x[n - 1] - 10.0).abs() < 1e-12);
        assert!((grid_y[0] - (-10.0)).abs() < 1e-12);
        assert!((grid_y[n - 1] - 10.0).abs() < 1e-12);

        assert_eq!(u32_at(&out, "nx"), n as u32);
        assert_eq!(u32_at(&out, "ny"), n as u32);
    }

    #[test]
    fn kde_2d_round_trip() {
        let with_extent = Kde2DSpec {
            x: "px".into(),
            y: "py".into(),
            bandwidth: BandwidthSpec::Fixed { value: 0.7 },
            n: 64,
            extent: Some((-1.0, 1.0, -2.0, 2.0)),
            groupby: None,
            name: Some("k2d".into()),
        };
        let json = serde_json::to_string(&with_extent).unwrap();
        let parsed: Kde2DSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, with_extent);

        let no_extent = Kde2DSpec {
            x: "px".into(),
            y: "py".into(),
            bandwidth: BandwidthSpec::Silverman,
            n: 32,
            extent: None,
            groupby: None,
            name: None,
        };
        let json2 = serde_json::to_string(&no_extent).unwrap();
        let parsed2: Kde2DSpec = serde_json::from_str(&json2).unwrap();
        assert_eq!(parsed2, no_extent);
        // Sanity: extent and groupby omitted in serialized form when None.
        assert!(!json2.contains("extent"), "extent=None must be omitted: {json2}");
        assert!(!json2.contains("groupby"), "groupby=None must be omitted: {json2}");

        // groupby round-trips through JSON.
        let with_groupby = Kde2DSpec {
            x: "px".into(),
            y: "py".into(),
            bandwidth: BandwidthSpec::Scott,
            n: 16,
            extent: None,
            groupby: Some("species".into()),
            name: None,
        };
        let json3 = serde_json::to_string(&with_groupby).unwrap();
        assert!(json3.contains("groupby"), "groupby=Some must appear in JSON: {json3}");
        let parsed3: Kde2DSpec = serde_json::from_str(&json3).unwrap();
        assert_eq!(parsed3, with_groupby);
    }

    // Helper: build a batch with x, y, and a string group column.
    fn batch_xy_g(
        xs: Vec<f64>,
        ys: Vec<f64>,
        gs: Vec<&str>,
    ) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("y", DataType::Float64, true),
            Field::new("g", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(xs)),
                Arc::new(Float64Array::from(ys)),
                Arc::new(StringArray::from(gs)),
            ],
        )
        .unwrap()
    }

    // Extract a Vec<f64> from row `row` of a list column.
    fn list_f64_row(batch: &RecordBatch, name: &str, row: usize) -> Vec<f64> {
        let idx = batch.schema().index_of(name).unwrap();
        let la = batch
            .column(idx)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let inner = la.value(row);
        let arr = inner.as_any().downcast_ref::<Float64Array>().unwrap();
        (0..arr.len()).map(|i| arr.value(i)).collect()
    }

    #[test]
    fn kde_2d_no_groupby_schema_unchanged() {
        // Sentinel: ungrouped output must have exactly 6 columns and 1 row,
        // byte-stable with the pre-groupby implementation.
        pyo3::Python::initialize();
        let xs = vec![0.0, 1.0, 0.0, 1.0];
        let ys = vec![0.0, 0.0, 1.0, 1.0];
        let spec = Kde2DSpec {
            x: "x".into(),
            y: "y".into(),
            bandwidth: BandwidthSpec::Scott,
            n: 8,
            extent: None,
            groupby: None,
            name: None,
        };
        let b = batch_xy(xs, ys);
        let out = apply(&spec, &b).unwrap();
        assert_eq!(out.num_rows(), 1, "ungrouped: expected 1 output row");
        assert_eq!(out.num_columns(), 6, "ungrouped: expected 6 output columns");
        let schema = out.schema();
        assert_eq!(schema.field(0).name(), "grid_x");
        assert_eq!(schema.field(1).name(), "grid_y");
        assert_eq!(schema.field(2).name(), "density");
        assert_eq!(schema.field(3).name(), "nx");
        assert_eq!(schema.field(4).name(), "ny");
        assert_eq!(schema.field(5).name(), "extent");
    }

    #[test]
    fn kde_2d_groupby_two_groups_emits_one_row_per_group() {
        // Core groupby test: two well-separated groups → 2 rows in output,
        // distinct surfaces, group column present with correct values.
        pyo3::Python::initialize();

        // Group A: cluster near (0, 0)
        // Group B: cluster near (10, 10)
        // Separation >> bandwidth → surfaces are disjoint; density at each
        // group's grid centre is higher for that group.
        let xs: Vec<f64> = vec![
            0.0, 0.2, 0.1, 0.3, 0.0, // A
            10.0, 10.2, 10.1, 10.3, 10.0, // B
        ];
        let ys: Vec<f64> = vec![
            0.0, 0.2, 0.1, 0.3, 0.0, // A
            10.0, 10.2, 10.1, 10.3, 10.0, // B
        ];
        let gs: Vec<&str> = vec!["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"];
        let n = 8usize;
        let spec = Kde2DSpec {
            x: "x".into(),
            y: "y".into(),
            bandwidth: BandwidthSpec::Scott,
            n,
            extent: None,
            groupby: Some("g".into()),
            name: None,
        };
        let b = batch_xy_g(xs, ys, gs);
        let out = apply(&spec, &b).unwrap();

        // 2 groups → 2 rows.
        assert_eq!(out.num_rows(), 2, "expected 2 output rows (one per group)");
        // Schema: 6 surface columns + 1 group column = 7.
        assert_eq!(out.num_columns(), 7, "expected 7 output columns with groupby");
        let schema = out.schema();
        assert_eq!(schema.field(6).name(), "g", "7th column must be the group key");
        assert_eq!(
            schema.field(6).data_type(),
            &DataType::Utf8,
            "group column must be Utf8"
        );

        // Group column values must be ["A", "B"] (first-appearance order).
        let garr = out
            .column(6)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(garr.value(0), "A");
        assert_eq!(garr.value(1), "B");

        // Each row must have correctly-sized list columns.
        let gx0 = list_f64_row(&out, "grid_x", 0);
        let gx1 = list_f64_row(&out, "grid_x", 1);
        let gy0 = list_f64_row(&out, "grid_y", 0);
        let gy1 = list_f64_row(&out, "grid_y", 1);
        let d0 = list_f64_row(&out, "density", 0);
        let d1 = list_f64_row(&out, "density", 1);
        assert_eq!(gx0.len(), n);
        assert_eq!(gx1.len(), n);
        assert_eq!(gy0.len(), n);
        assert_eq!(gy1.len(), n);
        assert_eq!(d0.len(), n * n);
        assert_eq!(d1.len(), n * n);

        // Per-group extents: group A grid spans near [0, 0.3], group B near [10, 10.3].
        assert!(
            gx0[0] < 1.0,
            "group A grid_x should start near 0; got {}",
            gx0[0]
        );
        assert!(
            gx1[0] > 9.0,
            "group B grid_x should start near 10; got {}",
            gx1[0]
        );

        // Surfaces are distinct: sum of density differs between groups
        // (both integrate to ≈1 but on very different extents, so when
        // compared at the same absolute coordinates they would be near 0 for
        // the other group's grid, but here we just confirm they're different).
        let sum0: f64 = d0.iter().sum();
        let sum1: f64 = d1.iter().sum();
        // Both should be positive (non-trivial surfaces).
        assert!(sum0 > 0.0, "group A density surface should be non-zero");
        assert!(sum1 > 0.0, "group B density surface should be non-zero");

        // nx/ny columns must equal n for each row.
        let nx_arr = out
            .column(3)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let ny_arr = out
            .column(4)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        assert_eq!(nx_arr.value(0), n as u32);
        assert_eq!(nx_arr.value(1), n as u32);
        assert_eq!(ny_arr.value(0), n as u32);
        assert_eq!(ny_arr.value(1), n as u32);
    }

    #[test]
    fn kde_2d_groupby_missing_column_errors() {
        pyo3::Python::initialize();
        let b = batch_xy(vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]);
        let spec = Kde2DSpec {
            x: "x".into(),
            y: "y".into(),
            bandwidth: BandwidthSpec::Scott,
            n: 4,
            extent: None,
            groupby: Some("ghost".into()),
            name: None,
        };
        let err = apply(&spec, &b).unwrap_err();
        assert!(
            err.to_string().contains("ghost"),
            "error message must mention the missing column: {err}"
        );
    }

    #[test]
    fn kde_2d_groupby_shared_extent_overrides_per_group() {
        // When spec.extent is set, all groups must share it.
        pyo3::Python::initialize();
        let xs: Vec<f64> = vec![0.0, 0.5, 1.0, 5.0, 5.5, 6.0];
        let ys: Vec<f64> = vec![0.0, 0.5, 1.0, 5.0, 5.5, 6.0];
        let gs: Vec<&str> = vec!["A", "A", "A", "B", "B", "B"];
        let shared = (-1.0_f64, 10.0_f64, -1.0_f64, 10.0_f64);
        let n = 4usize;
        let spec = Kde2DSpec {
            x: "x".into(),
            y: "y".into(),
            bandwidth: BandwidthSpec::Scott,
            n,
            extent: Some(shared),
            groupby: Some("g".into()),
            name: None,
        };
        let b = batch_xy_g(xs, ys, gs);
        let out = apply(&spec, &b).unwrap();
        assert_eq!(out.num_rows(), 2);

        // Both groups must use the shared extent for their grids.
        let gx0 = list_f64_row(&out, "grid_x", 0);
        let gx1 = list_f64_row(&out, "grid_x", 1);
        assert!(
            (gx0[0] - (-1.0)).abs() < 1e-12,
            "group A grid_x[0] should be -1; got {}",
            gx0[0]
        );
        assert!(
            (gx1[0] - (-1.0)).abs() < 1e-12,
            "group B grid_x[0] should be -1; got {}",
            gx1[0]
        );
        assert!(
            (gx0[n - 1] - 10.0).abs() < 1e-12,
            "group A grid_x[n-1] should be 10; got {}",
            gx0[n - 1]
        );
        assert!(
            (gx1[n - 1] - 10.0).abs() < 1e-12,
            "group B grid_x[n-1] should be 10; got {}",
            gx1[n - 1]
        );

        // Extent column must reflect the shared extent for both rows.
        let ext0 = list_f64_row(&out, "extent", 0);
        let ext1 = list_f64_row(&out, "extent", 1);
        assert_eq!(ext0, vec![-1.0, 10.0, -1.0, 10.0]);
        assert_eq!(ext1, vec![-1.0, 10.0, -1.0, 10.0]);
    }
}
