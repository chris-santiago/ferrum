use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch, StringArray};
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
    /// When set, partition input by this Utf8 column and emit per-(grid, group)
    /// rows. Output schema gains the groupby column as the 3rd field.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub groupby: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

pub(crate) fn apply(spec: &KdeSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    if let Some(g) = &spec.groupby {
        return apply_grouped(spec, batch, g);
    }
    apply_one_group(spec, batch, None)
}

fn apply_one_group(
    spec: &KdeSpec,
    batch: &RecordBatch,
    only_indices: Option<&[usize]>,
) -> PyResult<RecordBatch> {
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
    let mut clean: Vec<f64> = Vec::new();
    let push = |i: usize, clean: &mut Vec<f64>| {
        if arr.is_null(i) { return; }
        let v = arr.value(i);
        if !v.is_nan() { clean.push(v); }
    };
    match only_indices {
        Some(ixs) => for &i in ixs { push(i, &mut clean); },
        None => for i in 0..arr.len() { push(i, &mut clean); },
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

/// Partition input batch by `group_col` (Utf8), call apply_one_group per
/// partition, then stack the results into a single batch with the group
/// column preserved as the 3rd field.
fn apply_grouped(
    spec: &KdeSpec,
    batch: &RecordBatch,
    group_col: &str,
) -> PyResult<RecordBatch> {
    use std::collections::BTreeMap;
    let schema = batch.schema();
    let gi = schema.index_of(group_col).map_err(|_|
        PyValueError::new_err(format!(
            "stat_kde: groupby column '{}' not found", group_col)))?;
    let gtype = schema.field(gi).data_type();
    if gtype != &DataType::Utf8 {
        return Err(PyValueError::new_err(format!(
            "stat_kde: groupby column '{}' must be Utf8; got {:?}", group_col, gtype)));
    }
    let garr = batch.column(gi).as_any().downcast_ref::<StringArray>().unwrap();

    // Group row indices by first-appearance order of the group value.
    let mut group_order: Vec<String> = Vec::new();
    let mut group_idx_map: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for i in 0..garr.len() {
        if garr.is_null(i) { continue; }
        let gv = garr.value(i).to_string();
        if seen.insert(gv.clone()) {
            group_order.push(gv.clone());
        }
        group_idx_map.entry(gv).or_default().push(i);
    }

    let mut all_values: Vec<f64> = Vec::new();
    let mut all_density: Vec<f64> = Vec::new();
    let mut all_groups: Vec<String> = Vec::new();
    for g in &group_order {
        let ixs = group_idx_map.get(g).unwrap();
        let out = apply_one_group(spec, batch, Some(ixs))?;
        let n = out.num_rows();
        let values = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        let density = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        for i in 0..n {
            all_values.push(values.value(i));
            all_density.push(if density.is_null(i) { f64::NAN } else { density.value(i) });
            all_groups.push(g.clone());
        }
    }

    let out_schema = Arc::new(Schema::new(vec![
        Field::new("value", DataType::Float64, false),
        Field::new("density", DataType::Float64, true),
        Field::new(group_col, DataType::Utf8, false),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(all_values)),
        Arc::new(Float64Array::from(all_density)),
        Arc::new(StringArray::from(all_groups.iter().map(|s| s.as_str()).collect::<Vec<_>>())),
    ];
    RecordBatch::try_new(out_schema, cols)
        .map_err(|e| PyValueError::new_err(format!("stat_kde: {e}")))
}

pub(crate) fn bandwidth(x: &[f64], spec: &BandwidthSpec) -> PyResult<f64> {
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

use pyo3::prelude::*;

use crate::transform::core::TransformSpec;

/// One-dimensional kernel density estimate (KDE).
///
/// Fits a Gaussian kernel to ``field`` and evaluates the density on an
/// evenly-spaced grid of ``n`` points spanning (or clipped to) ``extent``.
/// The result is used to draw smooth density curves in violin plots and
/// ridge plots.
///
/// Output columns: ``value`` (Float64 grid points) and ``density``
/// (Float64 kernel density values, integrates to 1 over the grid range).
///
/// Parameters
/// ----------
/// field : str
///     Numeric column to estimate (must be Float64).
/// bandwidth : float or {"scott", "silverman"}, optional
///     Kernel bandwidth. A float sets a fixed bandwidth; ``"scott"`` and
///     ``"silverman"`` use the corresponding automatic rules. Default is
///     ``"scott"``.
/// n : int, default 512
///     Number of evaluation grid points. Must be > 0.
/// extent : (float, float), optional
///     ``(lo, hi)`` range to evaluate over; defaults to the data min/max.
///     Both values must be finite and ``lo < hi``.
/// cumulative : bool, default False
///     When True, output is the cumulative distribution function (CDF)
///     rather than the PDF.
/// groupby : str, optional
///     Single group-key column (Utf8); KDE computed independently per
///     group. Output schema gains the group column as the 3rd field.
/// name : str, optional
///     Named output label for sibling ``Reorder(from_=...)`` lookup.
#[pyclass(eq, module = "ferrum._core", name = "Kde")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyKde(pub(crate) TransformSpec);

#[pymethods]
impl PyKde {
    #[new]
    #[pyo3(signature = (field, *, bandwidth = None, n = 512, extent = None, cumulative = false, groupby = None, name = None))]
    fn new(
        field: &str,
        bandwidth: Option<&Bound<'_, PyAny>>,
        n: usize,
        extent: Option<(f64, f64)>,
        cumulative: bool,
        groupby: Option<String>,
        name: Option<String>,
    ) -> PyResult<Self> {
        if field.is_empty() {
            return Err(PyValueError::new_err("Kde: field must be non-empty"));
        }
        if n == 0 {
            return Err(PyValueError::new_err("Kde: n must be > 0"));
        }
        let bw = match bandwidth {
            None => BandwidthSpec::Scott,
            Some(obj) => {
                if let Ok(s) = obj.extract::<String>() {
                    match s.as_str() {
                        "scott" => BandwidthSpec::Scott,
                        "silverman" => BandwidthSpec::Silverman,
                        other => return Err(PyValueError::new_err(format!(
                            "Kde: unknown bandwidth '{other}'; expected 'scott' | 'silverman' | float"
                        ))),
                    }
                } else if let Ok(v) = obj.extract::<f64>() {
                    if !v.is_finite() || v <= 0.0 {
                        return Err(PyValueError::new_err(
                            "Kde: numeric bandwidth must be a positive finite number",
                        ));
                    }
                    BandwidthSpec::Fixed { value: v }
                } else {
                    return Err(PyValueError::new_err(
                        "Kde: bandwidth must be 'scott', 'silverman', or a positive float",
                    ));
                }
            }
        };
        if let Some((a, b)) = extent {
            if !a.is_finite() || !b.is_finite() || a >= b {
                return Err(PyValueError::new_err(
                    "Kde: extent must be (lo, hi) with lo < hi and both finite",
                ));
            }
        }
        Ok(PyKde(TransformSpec::Kde(KdeSpec {
            field: field.to_string(),
            bandwidth: bw,
            n,
            extent,
            cumulative,
            groupby,
            name,
        })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Kde(s) => format!(
                "Kde(field='{}', bandwidth={:?}, n={}, extent={:?}, cumulative={})",
                s.field, s.bandwidth, s.n, s.extent,
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
                groupby: None,
                name: None,
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
            groupby: None,
            name: None,
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
            groupby: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let density = col(&out, "density");
        assert!(density.iter().all(|d| d.is_nan()));
    }

    #[test]
    fn test_kde_grouped_preserves_group_column_and_per_group_density() {
        use arrow::array::StringArray;
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("g", DataType::Utf8, false),
        ]));
        // Two groups, well-separated means so per-group bandwidth differs from
        // global bandwidth.
        let xs = Float64Array::from(vec![
            0.0, 0.5, 1.0, 1.5, 2.0,        // group A
            10.0, 10.5, 11.0, 11.5, 12.0,    // group B
        ]);
        let gs = StringArray::from(vec!["A", "A", "A", "A", "A",
                                          "B", "B", "B", "B", "B"]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(xs), Arc::new(gs)]).unwrap();
        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Scott,
            n: 8,
            extent: None,
            cumulative: false,
            groupby: Some("g".into()),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        // 2 groups × 8 grid points = 16 rows
        assert_eq!(out.num_rows(), 16);
        // Schema must include the group column as the 3rd field.
        let out_schema = out.schema();
        assert_eq!(out_schema.field(0).name(), "value");
        assert_eq!(out_schema.field(1).name(), "density");
        assert_eq!(out_schema.field(2).name(), "g");
        // Per-group grids span per-group extents.
        let values = col(&out, "value");
        let groups = out.column(2).as_any().downcast_ref::<StringArray>().unwrap();
        // Rows 0..8 are group A (values 0.0..2.0); rows 8..16 are group B (10.0..12.0).
        for i in 0..8 { assert_eq!(groups.value(i), "A"); }
        for i in 8..16 { assert_eq!(groups.value(i), "B"); }
        assert!((values[0] - 0.0).abs() < 1e-9, "group A grid starts at min");
        assert!((values[7] - 2.0).abs() < 1e-9, "group A grid ends at max");
        assert!((values[8] - 10.0).abs() < 1e-9, "group B grid starts at min");
        assert!((values[15] - 12.0).abs() < 1e-9, "group B grid ends at max");
    }

    #[test]
    fn test_kde_ungrouped_output_schema_unchanged() {
        // Sentinel: ungrouped output schema MUST stay [value, density] (2 cols)
        // so existing goldens stay byte-identical.
        let batch = batch_with("x", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Scott,
            n: 4,
            extent: Some((0.0, 6.0)),
            cumulative: false,
            groupby: None,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_columns(), 2);
        assert_eq!(out.schema().field(0).name(), "value");
        assert_eq!(out.schema().field(1).name(), "density");
    }

    #[test]
    fn test_kde_grouped_missing_column_errors() {
        pyo3::Python::initialize();
        let batch = batch_with("x", vec![1.0, 2.0, 3.0]);
        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Scott,
            n: 4,
            extent: None,
            cumulative: false,
            groupby: Some("ghost".into()),
            name: None,
        };
        let err = apply(&spec, &batch).unwrap_err();
        assert!(err.to_string().contains("ghost"), "err: {err}");
    }

    #[test]
    fn test_kde_round_trip_json() {
        let original = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Fixed { value: 0.5 },
            n: 32,
            extent: Some((-1.0, 5.0)),
            cumulative: true,
            groupby: None,
            name: None,
        };
        let json = serde_json::to_string(&original).unwrap();
        let parsed: KdeSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, original);
    }
}
