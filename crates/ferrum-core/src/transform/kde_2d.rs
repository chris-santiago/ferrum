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
    Array, ArrayRef, Float64Array, Float64Builder, ListBuilder, RecordBatch, UInt32Array,
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
    pub name: Option<String>,
}

pub(crate) fn apply(spec: &Kde2DSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    if spec.n == 0 {
        return Err(PyValueError::new_err("stat_kde_2d: n must be > 0"));
    }

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
        .unwrap();
    let ya = batch
        .column(yi)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();

    let mut xs: Vec<f64> = Vec::with_capacity(xa.len());
    let mut ys: Vec<f64> = Vec::with_capacity(ya.len());
    for i in 0..xa.len() {
        let xnull = xa.is_null(i);
        let ynull = ya.is_null(i);
        if xnull || ynull {
            continue;
        }
        let xv = xa.value(i);
        let yv = ya.value(i);
        if xv.is_nan() || yv.is_nan() {
            continue;
        }
        xs.push(xv);
        ys.push(yv);
    }

    let n = spec.n;

    // Compute extent (xmin, xmax, ymin, ymax).
    let (xmin, xmax, ymin, ymax) = match spec.extent {
        Some((a, b, c, d)) => (a, b, c, d),
        None => {
            if xs.is_empty() {
                (0.0, 0.0, 0.0, 0.0)
            } else {
                let (mut xlo, mut xhi) = (f64::INFINITY, f64::NEG_INFINITY);
                let (mut ylo, mut yhi) = (f64::INFINITY, f64::NEG_INFINITY);
                for i in 0..xs.len() {
                    if xs[i] < xlo {
                        xlo = xs[i];
                    }
                    if xs[i] > xhi {
                        xhi = xs[i];
                    }
                    if ys[i] < ylo {
                        ylo = ys[i];
                    }
                    if ys[i] > yhi {
                        yhi = ys[i];
                    }
                }
                (xlo, xhi, ylo, yhi)
            }
        }
    };

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

    // Build list-typed columns: each is a single-element list.
    let mut grid_x_b = ListBuilder::new(Float64Builder::new());
    grid_x_b.values().append_slice(&grid_x);
    grid_x_b.append(true);
    let grid_x_arr = grid_x_b.finish();

    let mut grid_y_b = ListBuilder::new(Float64Builder::new());
    grid_y_b.values().append_slice(&grid_y);
    grid_y_b.append(true);
    let grid_y_arr = grid_y_b.finish();

    let mut density_b = ListBuilder::new(Float64Builder::new());
    density_b.values().append_slice(&density);
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
    #[pyo3(signature = (x, y, *, bandwidth = None, n = 128, extent = None, name = None))]
    fn new(
        x: &str,
        y: &str,
        bandwidth: Option<&Bound<'_, PyAny>>,
        n: usize,
        extent: Option<(f64, f64, f64, f64)>,
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
            name,
        })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Kde2D(s) => format!(
                "Kde2D(x='{}', y='{}', bandwidth={:?}, n={}, extent={:?})",
                s.x, s.y, s.bandwidth, s.n, s.extent
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, ListArray, RecordBatch, UInt32Array};
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
            name: None,
        };
        let json2 = serde_json::to_string(&no_extent).unwrap();
        let parsed2: Kde2DSpec = serde_json::from_str(&json2).unwrap();
        assert_eq!(parsed2, no_extent);
        // Sanity: extent omitted in serialized form when None.
        assert!(!json2.contains("extent"), "extent=None must be omitted: {json2}");
    }
}
