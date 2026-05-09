use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum BandwidthSpec {
    Scott,
    Silverman,
    Fixed { value: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct KdeSpec {
    pub field: String,
    pub bandwidth: BandwidthSpec,
    pub n: usize,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub extent: Option<(f64, f64)>,
    #[serde(default)]
    pub cumulative: bool,
}

pub(crate) fn apply(spec: &KdeSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let schema = batch.schema();
    let idx = schema.index_of(&spec.field).map_err(|_| {
        PyValueError::new_err(format!("stat_kde: column '{}' not found", spec.field))
    })?;
    if schema.field(idx).data_type() != &DataType::Float64 {
        return Err(PyValueError::new_err(format!(
            "stat_kde: column '{}' must be Float64",
            spec.field
        )));
    }
    let arr = batch
        .column(idx)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let mut clean: Vec<f64> = Vec::with_capacity(arr.len());
    for i in 0..arr.len() {
        if !arr.is_null(i) {
            let v = arr.value(i);
            if !v.is_nan() {
                clean.push(v);
            }
        }
    }

    let (lo, hi) = match spec.extent {
        Some((a, b)) => (a, b),
        None => {
            if clean.is_empty() {
                (0.0, 0.0)
            } else {
                clean.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                    (a.min(v), b.max(v))
                })
            }
        }
    };

    let grid: Vec<f64> = (0..spec.n)
        .map(|i| {
            if spec.n <= 1 {
                lo
            } else {
                lo + (hi - lo) * (i as f64) / ((spec.n - 1) as f64)
            }
        })
        .collect();

    let density: Vec<f64> = if clean.len() < 2 {
        vec![f64::NAN; spec.n]
    } else {
        let h = bandwidth(&clean, &spec.bandwidth)?;
        if h <= 0.0 || !h.is_finite() {
            vec![f64::NAN; spec.n]
        } else {
            gaussian_density_at_grid(&clean, h, &grid)
        }
    };

    let density = if spec.cumulative {
        trapezoidal_cumulative(&grid, &density)
    } else {
        density
    };

    let out_schema = Arc::new(Schema::new(vec![
        Field::new("value", DataType::Float64, false),
        Field::new("density", DataType::Float64, true),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(grid)),
        Arc::new(Float64Array::from(density)),
    ];
    RecordBatch::try_new(out_schema, cols)
        .map_err(|e| PyValueError::new_err(format!("stat_kde: {e}")))
}

fn bandwidth(x: &[f64], spec: &BandwidthSpec) -> PyResult<f64> {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let sigma = var.sqrt();
    Ok(match spec {
        BandwidthSpec::Scott => sigma * n.powf(-0.2),
        BandwidthSpec::Silverman => {
            let mut sorted = x.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let q25 = percentile(&sorted, 0.25);
            let q75 = percentile(&sorted, 0.75);
            let iqr = q75 - q25;
            0.9 * sigma.min(iqr / 1.34) * n.powf(-0.2)
        }
        BandwidthSpec::Fixed { value } => *value,
    })
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    // numpy linear-interpolation quantile.
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    let h = p * (n as f64 - 1.0);
    let lo = h.floor() as usize;
    let hi = (h.ceil() as usize).min(n - 1);
    let frac = h - h.floor();
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

fn gaussian_density_at_grid(x: &[f64], h: f64, grid: &[f64]) -> Vec<f64> {
    let n = x.len() as f64;
    let norm = 1.0 / (n * h * (2.0 * std::f64::consts::PI).sqrt());
    grid.iter()
        .map(|&g| {
            let s: f64 = x
                .iter()
                .map(|&xi| {
                    let z = (g - xi) / h;
                    (-0.5 * z * z).exp()
                })
                .sum();
            norm * s
        })
        .collect()
}

fn trapezoidal_cumulative(grid: &[f64], y: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(grid.len());
    out.push(0.0);
    for i in 1..grid.len() {
        let dx = grid[i] - grid[i - 1];
        let avg = 0.5 * (y[i] + y[i - 1]);
        out.push(out[i - 1] + avg * dx);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Float64Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use serde::Deserialize;
    use std::sync::Arc;

    const FIXTURES: &str = include_str!("fixtures/stat_refs.json");

    #[derive(Deserialize)]
    struct KdeCase {
        name: String,
        input: Vec<f64>,
        bandwidth: String,
        #[serde(default)]
        fixed_bandwidth: Option<f64>,
        n: usize,
        extent: [f64; 2],
        cumulative: bool,
        expected_bandwidth: f64,
        value: Vec<f64>,
        density: Vec<f64>,
    }

    #[derive(Deserialize)]
    struct Fixtures {
        kde: Vec<KdeCase>,
    }

    fn load_kde() -> Vec<KdeCase> {
        let f: Fixtures = serde_json::from_str(FIXTURES).unwrap();
        f.kde
    }

    fn batch_with(name: &str, values: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::Float64, true)]));
        RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(values))]).unwrap()
    }

    fn col(b: &RecordBatch, name: &str) -> Vec<f64> {
        let arr = b
            .column(b.schema().index_of(name).unwrap())
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        (0..arr.len())
            .map(|i| if arr.is_null(i) { f64::NAN } else { arr.value(i) })
            .collect()
    }

    #[test]
    fn test_kde_against_fixtures_within_tolerance() {
        for case in load_kde() {
            let bw = match (case.bandwidth.as_str(), case.fixed_bandwidth) {
                ("scott", _) => BandwidthSpec::Scott,
                ("silverman", _) => BandwidthSpec::Silverman,
                ("fixed", Some(v)) => BandwidthSpec::Fixed { value: v },
                other => panic!("unknown bandwidth spec: {other:?}"),
            };
            let spec = KdeSpec {
                field: "x".into(),
                bandwidth: bw,
                n: case.n,
                extent: Some((case.extent[0], case.extent[1])),
                cumulative: case.cumulative,
            };
            let batch = batch_with("x", case.input.clone());
            let out = apply(&spec, &batch).unwrap();
            let got_value = col(&out, "value");
            let got_density = col(&out, "density");
            for i in 0..case.n {
                assert!(
                    (got_value[i] - case.value[i]).abs() < 1e-9,
                    "case {} value[{i}]: got {} vs expected {}",
                    case.name,
                    got_value[i],
                    case.value[i]
                );
                assert!(
                    (got_density[i] - case.density[i]).abs() < 1e-6,
                    "case {} density[{i}]: got {} vs expected {} (diff {})",
                    case.name,
                    got_density[i],
                    case.density[i],
                    (got_density[i] - case.density[i]).abs()
                );
            }
            // Suppress unused field warning for expected_bandwidth
            let _ = case.expected_bandwidth;
        }
    }

    #[test]
    fn test_kde_zero_variance_emits_nan_densities() {
        let batch = batch_with("x", vec![3.0, 3.0, 3.0]);
        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Scott,
            n: 16,
            extent: Some((0.0, 6.0)),
            cumulative: false,
        };
        let out = apply(&spec, &batch).unwrap();
        let density = col(&out, "density");
        assert!(density.iter().all(|d| d.is_nan()), "expected all-NaN densities");
    }

    #[test]
    fn test_kde_n_lt_2_emits_nan_densities() {
        let batch = batch_with("x", vec![1.0]);
        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Scott,
            n: 8,
            extent: Some((0.0, 2.0)),
            cumulative: false,
        };
        let out = apply(&spec, &batch).unwrap();
        let density = col(&out, "density");
        assert!(density.iter().all(|d| d.is_nan()));
    }

    #[test]
    fn test_kde_round_trip_json() {
        let original = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Fixed { value: 0.5 },
            n: 32,
            extent: Some((-1.0, 5.0)),
            cumulative: true,
        };
        let json = serde_json::to_string(&original).unwrap();
        let parsed: KdeSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, original);
    }
}
