use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum SmoothMethod {
    Lm,
    Loess,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct SmoothSpec {
    pub x: String,
    pub y: String,
    pub method: SmoothMethod,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub ci: Option<f64>,
    pub bandwidth: f64,
    pub degree: u8,
    pub n: usize,
    #[serde(default)]
    pub seed: u64,
}

pub(crate) fn apply(spec: &SmoothSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let (xs, ys) = extract_xy(spec, batch)?;
    if xs.len() < 2 {
        return all_nan_output(spec);
    }

    let (x_min, x_max) = xs.iter().fold((f64::INFINITY, f64::NEG_INFINITY),
        |(a, b), &v| (a.min(v), b.max(v)));

    let grid: Vec<f64> = (0..spec.n)
        .map(|i| if spec.n <= 1 { x_min } else {
            x_min + (x_max - x_min) * (i as f64) / ((spec.n - 1) as f64)
        })
        .collect();

    match spec.method {
        SmoothMethod::Lm => lm_fit(&xs, &ys, &grid, spec.ci, spec.n),
        SmoothMethod::Loess => Err(PyNotImplementedError::new_err(
            "stat_smooth(method=loess) lands in Task 13/15"
        )),
    }
}

fn extract_xy(spec: &SmoothSpec, batch: &RecordBatch) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let schema = batch.schema();
    let xi = schema.index_of(&spec.x)
        .map_err(|_| PyValueError::new_err(format!("stat_smooth: column '{}' not found", spec.x)))?;
    let yi = schema.index_of(&spec.y)
        .map_err(|_| PyValueError::new_err(format!("stat_smooth: column '{}' not found", spec.y)))?;
    if schema.field(xi).data_type() != &DataType::Float64 {
        return Err(PyValueError::new_err(format!("stat_smooth: '{}' must be Float64", spec.x)));
    }
    if schema.field(yi).data_type() != &DataType::Float64 {
        return Err(PyValueError::new_err(format!("stat_smooth: '{}' must be Float64", spec.y)));
    }
    let xa = batch.column(xi).as_any().downcast_ref::<Float64Array>().unwrap();
    let ya = batch.column(yi).as_any().downcast_ref::<Float64Array>().unwrap();
    let mut xs = Vec::with_capacity(xa.len());
    let mut ys = Vec::with_capacity(ya.len());
    for i in 0..xa.len() {
        if xa.is_null(i) || ya.is_null(i) { continue; }
        let xv = xa.value(i); let yv = ya.value(i);
        if xv.is_nan() || yv.is_nan() { continue; }
        xs.push(xv); ys.push(yv);
    }
    Ok((xs, ys))
}

fn all_nan_output(spec: &SmoothSpec) -> PyResult<RecordBatch> {
    let n = spec.n;
    let nans = vec![f64::NAN; n];
    build_smooth_batch(nans.clone(), nans.clone(), nans.clone(), nans)
}

fn build_smooth_batch(
    x: Vec<f64>, y: Vec<f64>, lo: Vec<f64>, hi: Vec<f64>,
) -> PyResult<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x",        DataType::Float64, true),
        Field::new("y",        DataType::Float64, true),
        Field::new("ci_lower", DataType::Float64, true),
        Field::new("ci_upper", DataType::Float64, true),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(x)),
        Arc::new(Float64Array::from(y)),
        Arc::new(Float64Array::from(lo)),
        Arc::new(Float64Array::from(hi)),
    ];
    RecordBatch::try_new(schema, cols).map_err(|e| PyValueError::new_err(format!("stat_smooth: {e}")))
}

fn lm_fit(xs: &[f64], ys: &[f64], grid: &[f64], ci: Option<f64>, n_grid: usize)
    -> PyResult<RecordBatch>
{
    let n = xs.len();
    let mean_x: f64 = xs.iter().sum::<f64>() / n as f64;
    let mean_y: f64 = ys.iter().sum::<f64>() / n as f64;
    let sxx: f64 = xs.iter().map(|x| (x - mean_x).powi(2)).sum();
    let sxy: f64 = xs.iter().zip(ys).map(|(x, y)| (x - mean_x) * (y - mean_y)).sum();

    if sxx == 0.0 {
        return build_smooth_batch(
            grid.to_vec(),
            vec![f64::NAN; n_grid],
            vec![f64::NAN; n_grid],
            vec![f64::NAN; n_grid],
        );
    }

    let beta = sxy / sxx;
    let alpha = mean_y - beta * mean_x;
    let y_fit: Vec<f64> = grid.iter().map(|x| alpha + beta * x).collect();

    let (lo, hi) = match ci {
        None => (vec![f64::NAN; n_grid], vec![f64::NAN; n_grid]),
        Some(level) => {
            let resid_ss: f64 = xs.iter().zip(ys)
                .map(|(x, y)| (y - (alpha + beta * x)).powi(2))
                .sum();
            let dof = (n as f64) - 2.0;
            if dof <= 0.0 { (vec![f64::NAN; n_grid], vec![f64::NAN; n_grid]) }
            else {
                let sigma2 = resid_ss / dof;
                let t_crit = student_t_critical(level, dof);
                let mut lo = Vec::with_capacity(n_grid);
                let mut hi = Vec::with_capacity(n_grid);
                for &xq in grid {
                    let se = (sigma2 * (1.0 / (n as f64) + (xq - mean_x).powi(2) / sxx)).sqrt();
                    lo.push(alpha + beta * xq - t_crit * se);
                    hi.push(alpha + beta * xq + t_crit * se);
                }
                (lo, hi)
            }
        }
    };

    build_smooth_batch(grid.to_vec(), y_fit, lo, hi)
}

/// Two-sided t-critical at level `level` (e.g., 0.95) with `dof` degrees of freedom.
/// Hill's approximation; adequate for n >= 3 and tail probabilities >= 0.5%.
fn student_t_critical(level: f64, dof: f64) -> f64 {
    let alpha = 1.0 - level;
    let p = 1.0 - alpha / 2.0;
    let z = inv_normal_cdf(p);
    let c1 = (z * z + 1.0) / (4.0 * dof);
    let c2 = (5.0 * z.powi(4) + 16.0 * z * z + 3.0) / (96.0 * dof * dof);
    z * (1.0 + c1 + c2)
}

fn inv_normal_cdf(p: f64) -> f64 {
    // Beasley-Springer / Moro algorithm.
    let a = [
        -3.969683028665376e+01,  2.209460984245205e+02,
        -2.759285104469687e+02,  1.383577518672690e+02,
        -3.066479806614716e+01,  2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,  1.615858368580409e+02,
        -1.556989798598866e+02,  6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00,  2.938163982698783e+00,
    ];
    let d = [
         7.784695709041462e-03,  3.224671290700398e-01,
         2.445134137142996e+00,  3.754408661907416e+00,
    ];
    let plow = 0.02425;
    let phigh = 1.0 - plow;
    if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    } else if p <= phigh {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q /
            (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn xy_batch(x: Vec<f64>, y: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("y", DataType::Float64, true),
        ]));
        RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(x)),
            Arc::new(Float64Array::from(y)),
        ]).unwrap()
    }

    fn col(b: &RecordBatch, name: &str) -> Vec<f64> {
        let arr = b.column(b.schema().index_of(name).unwrap())
            .as_any().downcast_ref::<Float64Array>().unwrap();
        (0..arr.len()).map(|i| if arr.is_null(i) { f64::NAN } else { arr.value(i) }).collect()
    }

    #[test]
    fn test_lm_recovers_slope_and_intercept_exactly() {
        let xs: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|x| 3.0 + 2.0 * x).collect();
        let batch = xy_batch(xs, ys);
        let spec = SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::Lm,
            ci: None,
            bandwidth: 0.0, degree: 1, n: 5, seed: 0,
        };
        let out = apply(&spec, &batch).unwrap();
        let xg = col(&out, "x");
        let yf = col(&out, "y");
        for (xq, yq) in xg.iter().zip(yf.iter()) {
            let expected = 3.0 + 2.0 * xq;
            assert!((yq - expected).abs() < 1e-10, "y(x={xq})={yq}, expected {expected}");
        }
    }

    #[test]
    fn test_lm_ci_band_brackets_fit_at_mean_x() {
        let xs: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().enumerate().map(|(i, &x)| {
            x + if i % 2 == 0 { 0.5 } else { -0.5 }
        }).collect();
        let mean_x = xs.iter().sum::<f64>() / xs.len() as f64;
        let batch = xy_batch(xs, ys);
        let spec = SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::Lm,
            ci: Some(0.95),
            bandwidth: 0.0, degree: 1, n: 51, seed: 0,
        };
        let out = apply(&spec, &batch).unwrap();
        let xg = col(&out, "x");
        let yf = col(&out, "y");
        let lo = col(&out, "ci_lower");
        let hi = col(&out, "ci_upper");
        let i = (0..xg.len()).min_by(|a, b|
            (xg[*a] - mean_x).abs().partial_cmp(&(xg[*b] - mean_x).abs()).unwrap()
        ).unwrap();
        assert!(lo[i] < yf[i] && yf[i] < hi[i], "CI must bracket fit at x={}", xg[i]);
        assert!(hi[i] - lo[i] > 0.0);
    }

    #[test]
    fn test_lm_zero_variance_x_emits_nan_line() {
        let xs = vec![5.0; 10];
        let ys: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let batch = xy_batch(xs, ys);
        let spec = SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::Lm,
            ci: Some(0.95),
            bandwidth: 0.0, degree: 1, n: 5, seed: 0,
        };
        let out = apply(&spec, &batch).unwrap();
        let yf = col(&out, "y");
        assert!(yf.iter().all(|y| y.is_nan()));
    }

    #[test]
    fn test_lm_n_lt_2_emits_all_nan() {
        let batch = xy_batch(vec![1.0], vec![1.0]);
        let spec = SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::Lm,
            ci: None,
            bandwidth: 0.0, degree: 1, n: 5, seed: 0,
        };
        let out = apply(&spec, &batch).unwrap();
        let yf = col(&out, "y");
        assert!(yf.iter().all(|y| y.is_nan()));
    }

    #[test]
    fn test_smooth_round_trip_json() {
        let original = SmoothSpec {
            x: "x".into(), y: "y".into(),
            method: SmoothMethod::Lm,
            ci: Some(0.95),
            bandwidth: 0.5, degree: 2, n: 100, seed: 42,
        };
        let json = serde_json::to_string(&original).unwrap();
        let parsed: SmoothSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, original);
    }
}
