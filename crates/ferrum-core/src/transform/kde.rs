use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::transform::group_key::{
    deserialize_groupby_vec, group_partition_multi, groupby_field_nullable,
    materialize_groupby_col, parse_groupby_pyany, KeyValue,
};
use crate::transform::numeric_util::resolve_shared_extent;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum BandwidthSpec {
    Scott,
    Silverman,
    Fixed { value: f64 },
}

pub(crate) fn default_kernel() -> String { "gaussian".to_string() }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct KdeSpec {
    pub field: String,
    pub bandwidth: BandwidthSpec,
    /// Multiplier applied to the resolved bandwidth after rule evaluation.
    /// `bw_adjust = 1.0` is the identity. Works with all BandwidthSpec variants.
    #[serde(default = "default_bw_adjust")]
    pub bw_adjust: f64,
    pub n: usize,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub extent: Option<(f64, f64)>,
    #[serde(default)]
    pub cumulative: bool,
    /// When true, all groups share the same x-grid (required for stack/fill).
    #[serde(default)]
    pub shared_extent: bool,
    /// Kernel function for density estimation.
    /// Supported: "gaussian" (default), "epanechnikov" / "epan",
    /// "tophat" / "uniform", "cosine".
    #[serde(default = "default_kernel")]
    pub kernel: String,
    /// When non-empty, partition input by these columns and emit per-(value, group)
    /// rows. Output schema gains one column per groupby field after the 2nd field.
    /// Wire accepts: `null` → `[]`, `"col"` → `["col"]`, `["a","b"]` → `["a","b"]`.
    #[serde(
        skip_serializing_if = "Vec::is_empty",
        default,
        deserialize_with = "deserialize_groupby_vec"
    )]
    pub groupby: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

pub(crate) fn default_bw_adjust() -> f64 { 1.0 }

/// Compute the global `(lo, hi)` extent of `spec.field` over the full `batch`.
///
/// Returns `None` when the field is missing, non-numeric, or all values are
/// null/NaN. This is the pre-facet extent that `render::prepare` uses to pin the
/// value axis before partitioning, so every facet panel shares the same KDE grid
/// range (see `fix_transform_extents_for_facet`).
///
/// NICENESS CONTRACT (XFORM-08): returns the RAW `(lo, hi)` (no nicing), unlike
/// the `Bin` sibling. KDE controls a continuous grid start/end, not discrete bin
/// edges, so the pinned extent is the raw data range. Uses the shared
/// `column_extent` helper, which coerces integer-typed fields just as
/// `apply_one_group` does.
pub(crate) fn global_extent(spec: &KdeSpec, batch: &RecordBatch) -> Option<(f64, f64)> {
    crate::transform::numeric_util::column_extent(batch, &spec.field)
}

pub(crate) fn apply(spec: &KdeSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    if !spec.groupby.is_empty() {
        return apply_grouped(spec, batch, &spec.groupby);
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
    let arr = crate::transform::numeric_util::coerce_to_float64(
        batch.column(idx),
        "stat_kde",
        &spec.field,
    )?;
    let clean = crate::transform::numeric_util::clean_float64_values(&arr, only_indices);

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
        let h = bandwidth(&clean, &spec.bandwidth)? * spec.bw_adjust;
        if h <= 0.0 || !h.is_finite() {
            vec![f64::NAN; spec.n]
        } else {
            kernel_density_at_grid(&clean, h, &grid, &spec.kernel)
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

/// Partition input batch by the groupby columns, call apply_one_group per
/// partition, then stack the results into a single batch with the group
/// columns appended after the 2nd field.
fn apply_grouped(
    spec: &KdeSpec,
    batch: &RecordBatch,
    group_cols: &[String],
) -> PyResult<RecordBatch> {
    // Multi-column partition path (first-appearance order, null-key skip, dtype-preserve).
    let (group_order, group_idx_map, gtypes) =
        group_partition_multi(batch, group_cols, "stat_kde")?;

    // Shared vs per-group extent (stack/fill aligns to one global extent;
    // layer mode keeps per-group extents). resolve_shared_extent returns the
    // explicit extent unchanged, else the global extent when shared, else None.
    let global_extent = resolve_shared_extent(spec.extent, spec.shared_extent, batch, &spec.field);
    let shared_spec = KdeSpec {
        extent: global_extent.or(spec.extent),
        ..spec.clone()
    };

    let mut all_values: Vec<f64> = Vec::new();
    let mut all_density: Vec<f64> = Vec::new();
    let mut group_keys_out: Vec<Vec<KeyValue>> = Vec::new();
    for g in &group_order {
        let ixs = group_idx_map.get(g)
            .ok_or_else(|| PyValueError::new_err(format!(
                "stat_kde: missing group key {g:?} in index map")))?;
        let out = apply_one_group(&shared_spec, batch, Some(ixs))?;
        let n = out.num_rows();
        let values = out.column(0).as_any().downcast_ref::<Float64Array>()
            .ok_or_else(|| PyValueError::new_err("stat_kde: expected Float64Array for 'value' output column"))?;
        let density = out.column(1).as_any().downcast_ref::<Float64Array>()
            .ok_or_else(|| PyValueError::new_err("stat_kde: expected Float64Array for 'density' output column"))?;
        for i in 0..n {
            all_values.push(values.value(i));
            all_density.push(if density.is_null(i) { f64::NAN } else { density.value(i) });
            group_keys_out.push(g.clone());
        }
    }

    let mut fields = vec![
        Field::new("value", DataType::Float64, false),
        Field::new("density", DataType::Float64, true),
    ];
    // FA-7: preserve the original groupby dtypes (nullable per FA-9; null keys
    // are skipped upstream in group_partition_multi).
    for (name, dtype) in group_cols.iter().zip(gtypes.iter()) {
        fields.push(Field::new(name, dtype.clone(), groupby_field_nullable()));
    }
    let out_schema = Arc::new(Schema::new(fields));
    let mut cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(all_values)),
        Arc::new(Float64Array::from(all_density)),
    ];
    for (gi, dtype) in gtypes.iter().enumerate() {
        let group_col_arr = materialize_groupby_col(&group_keys_out, gi, dtype)
            .map_err(PyValueError::new_err)?;
        cols.push(group_col_arr);
    }
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
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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

/// Dispatch to the requested kernel's density estimator.
/// Falls back to Gaussian for any unrecognised kernel name (should not
/// happen in practice since `PyKde::new` validates the string).
fn kernel_density_at_grid(x: &[f64], h: f64, grid: &[f64], kernel: &str) -> Vec<f64> {
    match kernel {
        "epanechnikov" | "epan" => epanechnikov_density_at_grid(x, h, grid),
        "tophat" | "uniform" => tophat_density_at_grid(x, h, grid),
        "cosine" => cosine_density_at_grid(x, h, grid),
        _ => gaussian_density_at_grid(x, h, grid),
    }
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

/// Epanechnikov kernel: K(u) = (3/4)(1 - u²) for |u| ≤ 1, else 0.
/// Integrates to 1 over [-1, 1] and is optimal in MSE sense.
fn epanechnikov_density_at_grid(x: &[f64], h: f64, grid: &[f64]) -> Vec<f64> {
    let n = x.len() as f64;
    // The kernel integrates to 1 over [-h, h], so no extra h factor in norm.
    let norm = 1.0 / (n * h);
    grid.iter()
        .map(|&g| {
            let s: f64 = x
                .iter()
                .map(|&xi| {
                    let u = (g - xi) / h;
                    if u.abs() <= 1.0 {
                        0.75 * (1.0 - u * u)
                    } else {
                        0.0
                    }
                })
                .sum();
            norm * s
        })
        .collect()
}

/// Tophat (uniform) kernel: K(u) = 0.5 for |u| ≤ 1, else 0.
fn tophat_density_at_grid(x: &[f64], h: f64, grid: &[f64]) -> Vec<f64> {
    let n = x.len() as f64;
    let norm = 1.0 / (n * h);
    grid.iter()
        .map(|&g| {
            let s: f64 = x
                .iter()
                .map(|&xi| {
                    let u = (g - xi) / h;
                    if u.abs() <= 1.0 { 0.5 } else { 0.0 }
                })
                .sum();
            norm * s
        })
        .collect()
}

/// Cosine kernel: K(u) = (π/4)·cos(π·u/2) for |u| ≤ 1, else 0.
fn cosine_density_at_grid(x: &[f64], h: f64, grid: &[f64]) -> Vec<f64> {
    use std::f64::consts::PI;
    let n = x.len() as f64;
    let norm = 1.0 / (n * h);
    grid.iter()
        .map(|&g| {
            let s: f64 = x
                .iter()
                .map(|&xi| {
                    let u = (g - xi) / h;
                    if u.abs() <= 1.0 {
                        (PI / 4.0) * ((PI / 2.0) * u).cos()
                    } else {
                        0.0
                    }
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
/// groupby : list[str] or str or None, optional
///     One or more group-key columns; KDE computed independently per group.
///     Output schema gains one column per groupby field after the 2nd field.
/// name : str, optional
///     Named output label for sibling ``Reorder(from_=...)`` lookup.
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> fm.Chart(df).mark_density().encode(x="val")
///
/// Direct construction (custom bandwidth):
///
/// >>> fm.Chart(df).mark_area().encode(
/// ...     x="x_kde", y="density",
/// ...     transform=fm.Kde("val", bandwidth=0.3),
/// ... )
#[pyclass(eq, module = "ferrum._core", name = "Kde")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyKde(pub(crate) TransformSpec);

#[pymethods]
impl PyKde {
    #[new]
    #[pyo3(signature = (field, *, bandwidth = None, bw_adjust = 1.0, n = 512, extent = None, cumulative = false, shared_extent = false, kernel = "gaussian", groupby = None, name = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        field: &str,
        bandwidth: Option<&Bound<'_, PyAny>>,
        bw_adjust: f64,
        n: usize,
        extent: Option<(f64, f64)>,
        cumulative: bool,
        shared_extent: bool,
        kernel: &str,
        groupby: Option<&Bound<'_, PyAny>>,
        name: Option<String>,
    ) -> PyResult<Self> {
        if !bw_adjust.is_finite() || bw_adjust <= 0.0 {
            return Err(PyValueError::new_err("Kde: bw_adjust must be a positive finite number"));
        }
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
        let kernel_str = match kernel {
            "gaussian" | "epanechnikov" | "epan" | "tophat" | "uniform" | "cosine" => {
                kernel.to_string()
            }
            other => return Err(PyValueError::new_err(format!(
                "Kde: unknown kernel '{other}'; expected 'gaussian' | 'epanechnikov' | 'epan' | 'tophat' | 'uniform' | 'cosine'"
            ))),
        };
        let groupby = parse_groupby_pyany(groupby)?;
        Ok(PyKde(TransformSpec::Kde(KdeSpec {
            field: field.to_string(),
            bandwidth: bw,
            bw_adjust,
            n,
            extent,
            cumulative,
            shared_extent,
            kernel: kernel_str,
            groupby,
            name,
        })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Kde(s) => format!(
                "Kde(field='{}', bandwidth={:?}, n={}, extent={:?}, cumulative={}, kernel='{}')",
                s.field, s.bandwidth, s.n, s.extent,
                if s.cumulative { "True" } else { "False" },
                s.kernel,
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
                bw_adjust: 1.0,
                shared_extent: false,
                n: case.n,
                extent: Some((case.extent[0], case.extent[1])),
                cumulative: case.cumulative,
                kernel: default_kernel(),
                groupby: vec![],
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
                bw_adjust: 1.0,
                shared_extent: false,
            n: 16,
            extent: Some((0.0, 6.0)),
            cumulative: false,
            kernel: default_kernel(),
            groupby: vec![],
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
                bw_adjust: 1.0,
                shared_extent: false,
            n: 8,
            extent: Some((0.0, 2.0)),
            cumulative: false,
            kernel: default_kernel(),
            groupby: vec![],
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
                bw_adjust: 1.0,
                shared_extent: false,
            n: 8,
            extent: None,
            cumulative: false,
            kernel: default_kernel(),
            groupby: vec!["g".into()],
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
    fn test_kde_grouped_int64_group_column() {
        // T0.3 regression: a non-Utf8 (Int64) group column must work, not raise
        // "must be Utf8". Pre-fix this errored in kde::apply_grouped.
        use arrow::array::Int64Array;
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("g", DataType::Int64, false),
        ]));
        let xs = Float64Array::from(vec![
            0.0, 0.5, 1.0, 1.5, 2.0, // group 1
            10.0, 10.5, 11.0, 11.5, 12.0, // group 2
        ]);
        let gs = Int64Array::from(vec![1_i64, 1, 1, 1, 1, 2, 2, 2, 2, 2]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(xs), Arc::new(gs)]).unwrap();
        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            shared_extent: false,
            n: 8,
            extent: None,
            cumulative: false,
            kernel: default_kernel(),
            groupby: vec!["g".into()],
            name: None,
        };
        let out = apply(&spec, &batch).expect("Int64 groupby must succeed");
        assert_eq!(out.num_rows(), 16, "2 groups × 8 grid points");
        // Group column must round-trip as Int64, NOT String.
        assert_eq!(out.schema().field(2).name(), "g");
        assert_eq!(out.schema().field(2).data_type(), &DataType::Int64);
        let groups = out.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
        for i in 0..8 {
            assert_eq!(groups.value(i), 1, "first 8 rows are group 1");
        }
        for i in 8..16 {
            assert_eq!(groups.value(i), 2, "last 8 rows are group 2");
        }
    }

    #[test]
    fn test_kde_grouped_boolean_group_column() {
        // T0.3 regression: a Boolean group column must work (was "must be Utf8").
        use arrow::array::BooleanArray;
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("g", DataType::Boolean, false),
        ]));
        let xs = Float64Array::from(vec![
            0.0, 0.5, 1.0, 1.5, 2.0, // false group
            10.0, 10.5, 11.0, 11.5, 12.0, // true group
        ]);
        let gs = BooleanArray::from(vec![
            false, false, false, false, false, true, true, true, true, true,
        ]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(xs), Arc::new(gs)]).unwrap();
        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            shared_extent: false,
            n: 8,
            extent: None,
            cumulative: false,
            kernel: default_kernel(),
            groupby: vec!["g".into()],
            name: None,
        };
        let out = apply(&spec, &batch).expect("Boolean groupby must succeed");
        assert_eq!(out.num_rows(), 16);
        assert_eq!(out.schema().field(2).data_type(), &DataType::Boolean);
        let groups = out.column(2).as_any().downcast_ref::<BooleanArray>().unwrap();
        // First-appearance order → false group first, then true.
        for i in 0..8 {
            assert!(!groups.value(i), "first 8 rows are the false group");
        }
        for i in 8..16 {
            assert!(groups.value(i), "last 8 rows are the true group");
        }
    }

    #[test]
    fn test_kde_ungrouped_output_schema_unchanged() {
        // Sentinel: ungrouped output schema MUST stay [value, density] (2 cols)
        // so existing goldens stay byte-identical.
        let batch = batch_with("x", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Scott,
                bw_adjust: 1.0,
                shared_extent: false,
            n: 4,
            extent: Some((0.0, 6.0)),
            cumulative: false,
            kernel: default_kernel(),
            groupby: vec![],
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
                bw_adjust: 1.0,
                shared_extent: false,
            n: 4,
            extent: None,
            cumulative: false,
            kernel: default_kernel(),
            groupby: vec!["ghost".into()],
            name: None,
        };
        let err = apply(&spec, &batch).unwrap_err();
        assert!(err.to_string().contains("ghost"), "err: {err}");
    }

    #[test]
    fn test_kde_int64_column_is_coerced() {
        // Int64 input must be coerced to Float64, not rejected. The density
        // output must match a Float64 batch of the same values bit-for-bit.
        use arrow::array::Int64Array;
        pyo3::Python::initialize();
        let vals_i: Vec<i64> = (0..10).collect();
        let vals_f: Vec<f64> = vals_i.iter().map(|&v| v as f64).collect();
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
        let int_batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vals_i))]).unwrap();
        let float_batch = batch_with("x", vals_f);
        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            shared_extent: false,
            n: 16,
            extent: None,
            cumulative: false,
            kernel: default_kernel(),
            groupby: vec![],
            name: None,
        };
        let int_out = apply(&spec, &int_batch).expect("Int64 KDE should succeed");
        let float_out = apply(&spec, &float_batch).unwrap();
        assert_eq!(int_out.num_rows(), 16);
        let id = col(&int_out, "density");
        let fd = col(&float_out, "density");
        let iv = col(&int_out, "value");
        let fv = col(&float_out, "value");
        for i in 0..16 {
            assert_eq!(iv[i].to_bits(), fv[i].to_bits(), "value[{i}] mismatch");
            assert_eq!(id[i].to_bits(), fd[i].to_bits(), "density[{i}] mismatch");
        }
    }

    #[test]
    fn test_kde_int32_column_is_coerced() {
        // Prove general numeric coercion (not just Int64).
        use arrow::array::Int32Array;
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, true)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![0_i32, 1, 2, 3, 4]))],
        )
        .unwrap();
        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            shared_extent: false,
            n: 8,
            extent: None,
            cumulative: false,
            kernel: default_kernel(),
            groupby: vec![],
            name: None,
        };
        let out = apply(&spec, &batch).expect("Int32 KDE should succeed");
        assert_eq!(out.num_rows(), 8);
    }

    #[test]
    fn test_kde_string_column_still_errors() {
        // Negative test: a genuinely non-numeric column must STILL error.
        use arrow::array::StringArray;
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Utf8, true)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec!["a", "b", "c"]))],
        )
        .unwrap();
        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            shared_extent: false,
            n: 8,
            extent: None,
            cumulative: false,
            kernel: default_kernel(),
            groupby: vec![],
            name: None,
        };
        let err = apply(&spec, &batch).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("must be numeric"), "err: {msg}");
    }

    #[test]
    fn test_kde_round_trip_json() {
        let original = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Fixed { value: 0.5 },
            bw_adjust: 1.0,
            n: 32,
            extent: Some((-1.0, 5.0)),
            cumulative: true,
            shared_extent: false,
            kernel: "gaussian".to_string(),
            groupby: vec![],
            name: None,
        };
        let json = serde_json::to_string(&original).unwrap();
        let parsed: KdeSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn test_kde_single_row_emits_nan_densities() {
        // A single data point → n < 2, so Scott/Silverman bandwidth is undefined.
        // The transform should handle gracefully by emitting NaN densities.
        let batch = batch_with("x", vec![7.0]);
        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Silverman,
            bw_adjust: 1.0,
            shared_extent: false,
            n: 10,
            extent: Some((5.0, 9.0)),
            cumulative: false,
            kernel: default_kernel(),
            groupby: vec![],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 10);
        let density = col(&out, "density");
        assert!(
            density.iter().all(|d| d.is_nan()),
            "single-row KDE should emit all-NaN densities"
        );
    }

    #[test]
    fn test_kde_all_identical_values_emits_nan_densities() {
        // All values identical → std=0 → bandwidth=0 (both Scott and Silverman).
        // This is effectively the zero-variance case.
        let batch = batch_with("x", vec![4.0; 20]);
        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Silverman,
            bw_adjust: 1.0,
            shared_extent: false,
            n: 16,
            extent: Some((2.0, 6.0)),
            cumulative: false,
            kernel: default_kernel(),
            groupby: vec![],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let density = col(&out, "density");
        assert!(
            density.iter().all(|d| d.is_nan()),
            "zero-variance KDE should emit all-NaN densities"
        );
    }

    // ---- Kernel-specific tests ----

    fn kde_spec_with_kernel(kernel: &str, n: usize) -> KdeSpec {
        KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Fixed { value: 1.0 },
            bw_adjust: 1.0,
            shared_extent: false,
            n,
            extent: Some((-3.0, 3.0)),
            cumulative: false,
            kernel: kernel.to_string(),
            groupby: vec![],
            name: None,
        }
    }

    fn simple_batch() -> RecordBatch {
        batch_with("x", vec![0.0, 0.5, -0.5, 1.0, -1.0])
    }

    /// Gaussian kernel (default) must produce non-zero density everywhere in the grid.
    #[test]
    fn test_kde_gaussian_produces_nonzero_density() {
        let batch = simple_batch();
        let spec = kde_spec_with_kernel("gaussian", 20);
        let out = apply(&spec, &batch).unwrap();
        let density = col(&out, "density");
        assert!(
            density.iter().all(|&d| d > 0.0),
            "Gaussian KDE: all grid densities should be positive"
        );
    }

    /// Epanechnikov kernel must produce non-zero density on simple data.
    #[test]
    fn test_kde_epanechnikov_produces_nonzero_density() {
        let batch = simple_batch();
        let spec = kde_spec_with_kernel("epanechnikov", 20);
        let out = apply(&spec, &batch).unwrap();
        let density = col(&out, "density");
        // At least some grid points must be non-zero (those inside bandwidth support).
        let nonzero_count = density.iter().filter(|&&d| d > 0.0).count();
        assert!(
            nonzero_count > 0,
            "Epanechnikov KDE: expected some non-zero densities, got all zeros"
        );
    }

    /// Epanechnikov kernel must produce exactly 0 for grid points further than
    /// bandwidth h from all data points.
    #[test]
    fn test_kde_epanechnikov_zero_outside_bandwidth() {
        // Single data point at x=0, bandwidth=0.5.
        // Grid points beyond |g - 0| > 0.5 must be exactly 0.
        let batch = batch_with("x", vec![0.0, 0.01]); // need ≥ 2 points
        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Fixed { value: 0.5 },
            bw_adjust: 1.0,
            shared_extent: false,
            n: 21,
            extent: Some((-2.0, 2.0)),
            cumulative: false,
            kernel: "epanechnikov".to_string(),
            groupby: vec![],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let values = col(&out, "value");
        let density = col(&out, "density");
        // All grid points where |g| > 0.5 AND |g - 0.01| > 0.5 must have density = 0.
        for (i, (&v, &d)) in values.iter().zip(density.iter()).enumerate() {
            let in_support = v.abs() <= 0.5 || (v - 0.01).abs() <= 0.5;
            if !in_support {
                assert_eq!(
                    d, 0.0,
                    "Epanechnikov: expected density=0 at grid point {i} (x={v}), got {d}"
                );
            }
        }
    }

    /// "epan" alias must produce the same output as "epanechnikov".
    #[test]
    fn test_kde_epan_alias_matches_epanechnikov() {
        let batch = simple_batch();
        let spec1 = kde_spec_with_kernel("epanechnikov", 15);
        let spec2 = kde_spec_with_kernel("epan", 15);
        let out1 = apply(&spec1, &batch).unwrap();
        let out2 = apply(&spec2, &batch).unwrap();
        let d1 = col(&out1, "density");
        let d2 = col(&out2, "density");
        for (a, b) in d1.iter().zip(d2.iter()) {
            assert_eq!(
                a.to_bits(), b.to_bits(),
                "'epan' and 'epanechnikov' must produce identical output"
            );
        }
    }

    /// Tophat kernel must produce non-zero density on simple data.
    #[test]
    fn test_kde_tophat_produces_nonzero_density() {
        let batch = simple_batch();
        let spec = kde_spec_with_kernel("tophat", 20);
        let out = apply(&spec, &batch).unwrap();
        let density = col(&out, "density");
        let nonzero_count = density.iter().filter(|&&d| d > 0.0).count();
        assert!(
            nonzero_count > 0,
            "Tophat KDE: expected some non-zero densities"
        );
    }

    /// "uniform" alias must produce the same output as "tophat".
    #[test]
    fn test_kde_uniform_alias_matches_tophat() {
        let batch = simple_batch();
        let spec1 = kde_spec_with_kernel("tophat", 15);
        let spec2 = kde_spec_with_kernel("uniform", 15);
        let out1 = apply(&spec1, &batch).unwrap();
        let out2 = apply(&spec2, &batch).unwrap();
        let d1 = col(&out1, "density");
        let d2 = col(&out2, "density");
        for (a, b) in d1.iter().zip(d2.iter()) {
            assert_eq!(
                a.to_bits(), b.to_bits(),
                "'uniform' and 'tophat' must produce identical output"
            );
        }
    }

    /// Cosine kernel must produce non-zero density on simple data.
    #[test]
    fn test_kde_cosine_produces_nonzero_density() {
        let batch = simple_batch();
        let spec = kde_spec_with_kernel("cosine", 20);
        let out = apply(&spec, &batch).unwrap();
        let density = col(&out, "density");
        let nonzero_count = density.iter().filter(|&&d| d > 0.0).count();
        assert!(
            nonzero_count > 0,
            "Cosine KDE: expected some non-zero densities"
        );
    }

    /// Gaussian must produce the same output as before (regression against the
    /// hardcoded path we replaced with the dispatch).
    #[test]
    fn test_kde_gaussian_regression_matches_direct_call() {
        let data = vec![-1.0, 0.0, 1.0, 2.0, 3.0];
        let h = 0.8_f64;
        let grid: Vec<f64> = (0..10).map(|i| -2.0 + i as f64 * 0.5).collect();
        let expected = gaussian_density_at_grid(&data, h, &grid);
        let via_dispatch = kernel_density_at_grid(&data, h, &grid, "gaussian");
        for (a, b) in expected.iter().zip(via_dispatch.iter()) {
            assert_eq!(
                a.to_bits(), b.to_bits(),
                "gaussian dispatch must be byte-identical to direct call"
            );
        }
    }

    /// Unknown kernel name in PyKde::new must return an error.
    #[test]
    fn test_kde_unknown_kernel_returns_error() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            use pyo3::types::PyString;
            let bw_obj = PyString::new(py, "scott").into_any();
            let result = PyKde::new(
                "x",
                Some(&bw_obj),
                1.0,
                512,
                None,
                false,
                false,
                "laplacian",
                None,
                None,
            );
            assert!(result.is_err(), "unknown kernel 'laplacian' must return an error");
            let msg = result.unwrap_err().to_string();
            assert!(
                msg.contains("unknown kernel"),
                "error message should mention 'unknown kernel': {msg}"
            );
        });
    }

    /// Gaussian kernel must round-trip through JSON including the kernel field.
    #[test]
    fn test_kde_kernel_field_round_trips_json() {
        let original = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 64,
            extent: None,
            cumulative: false,
            shared_extent: false,
            kernel: "epanechnikov".to_string(),
            groupby: vec![],
            name: None,
        };
        let json = serde_json::to_string(&original).unwrap();
        let parsed: KdeSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.kernel, "epanechnikov");
        assert_eq!(parsed, original);
    }

    /// Old JSON without a "kernel" field must deserialize to gaussian (backward compat).
    #[test]
    fn test_kde_kernel_field_defaults_to_gaussian_on_old_json() {
        // Simulate a serialized KdeSpec without the "kernel" field.
        let json = r#"{"field":"x","bandwidth":{"kind":"scott"},"bw_adjust":1.0,"n":32}"#;
        let parsed: KdeSpec = serde_json::from_str(json).unwrap();
        assert_eq!(
            parsed.kernel, "gaussian",
            "missing 'kernel' field must default to 'gaussian' for backward compat"
        );
    }

    // ── Task 8: global_extent helper ─────────────────────────────────────────

    /// `global_extent` returns the full-data range over the KDE field.
    #[test]
    fn test_kde_global_extent_returns_full_range() {
        let batch = batch_with("x", vec![3.0, 1.0, 5.0, 2.0, 4.0]);
        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 32,
            extent: None,
            cumulative: false,
            shared_extent: false,
            kernel: default_kernel(),
            groupby: vec![],
            name: None,
        };
        let ext = super::global_extent(&spec, &batch);
        assert_eq!(ext, Some((1.0, 5.0)), "global_extent should be (min, max) = (1.0, 5.0)");
    }

    /// `global_extent` handles integer columns via coerce_to_float64.
    #[test]
    fn test_kde_global_extent_integer_column() {
        use arrow::array::Int64Array;
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![10_i64, 20, 30]))],
        ).unwrap();
        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 16,
            extent: None,
            cumulative: false,
            shared_extent: false,
            kernel: default_kernel(),
            groupby: vec![],
            name: None,
        };
        let ext = super::global_extent(&spec, &batch);
        assert_eq!(ext, Some((10.0, 30.0)), "global_extent on Int64 column should return (10.0, 30.0)");
    }

    /// `global_extent` returns None on missing field.
    #[test]
    fn test_kde_global_extent_missing_field_returns_none() {
        let batch = batch_with("x", vec![1.0, 2.0, 3.0]);
        let spec = KdeSpec {
            field: "nonexistent".into(),
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 16,
            extent: None,
            cumulative: false,
            shared_extent: false,
            kernel: default_kernel(),
            groupby: vec![],
            name: None,
        };
        let ext = super::global_extent(&spec, &batch);
        assert_eq!(ext, None, "global_extent on missing field must return None");
    }

    /// `global_extent` returns None when all values are null or NaN.
    #[test]
    fn test_kde_global_extent_all_null_returns_none() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, true)]));
        let arr = Float64Array::from(vec![None::<f64>, None, None]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 16,
            extent: None,
            cumulative: false,
            shared_extent: false,
            kernel: default_kernel(),
            groupby: vec![],
            name: None,
        };
        let ext = super::global_extent(&spec, &batch);
        assert_eq!(ext, None, "global_extent on all-null field must return None");
    }

    // ── T3.8a regression tests ────────────────────────────────────────────────

    /// D-GROUPBY-1 regression (T3.8a): a 2-column groupby on `Kde` must produce
    /// `n` density grid points per distinct `(g1, g2)` combination, with both
    /// group columns appended after the standard 2 kde output columns.
    #[test]
    fn kde_two_column_groupby_produces_per_combination_rows() {
        use arrow::array::StringArray;
        pyo3::Python::initialize();
        // 8 rows: two g1 × two g2 × 2 x-values each.
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("g1", DataType::Utf8, false),
            Field::new("g2", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![
                    1.0, 9.0, 1.0, 9.0, 1.0, 9.0, 1.0, 9.0,
                ])) as arrow::array::ArrayRef,
                Arc::new(StringArray::from(vec!["A", "A", "A", "A", "B", "B", "B", "B"]))
                    as arrow::array::ArrayRef,
                Arc::new(StringArray::from(vec!["x", "x", "y", "y", "x", "x", "y", "y"]))
                    as arrow::array::ArrayRef,
            ],
        )
        .unwrap();

        let n = 8_usize;
        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n,
            extent: None,
            cumulative: false,
            shared_extent: false,
            kernel: default_kernel(),
            groupby: vec!["g1".into(), "g2".into()],
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();

        // Schema: value, density, g1, g2.
        let s = out.schema();
        assert!(s.index_of("value").is_ok(), "missing value");
        assert!(s.index_of("density").is_ok(), "missing density");
        assert!(s.index_of("g1").is_ok(), "missing g1");
        assert!(s.index_of("g2").is_ok(), "missing g2");

        // 4 distinct (g1,g2) groups × n grid points each.
        assert_eq!(
            out.num_rows(),
            4 * n,
            "expected 4 groups × {n} grid points = {} rows, got {}",
            4 * n,
            out.num_rows()
        );

        // Collect distinct (g1,g2) combos in the output.
        let g1_col = out.column(s.index_of("g1").unwrap())
            .as_any().downcast_ref::<StringArray>().unwrap();
        let g2_col = out.column(s.index_of("g2").unwrap())
            .as_any().downcast_ref::<StringArray>().unwrap();
        let combos: std::collections::BTreeSet<(String, String)> = (0..out.num_rows())
            .map(|i| (g1_col.value(i).to_string(), g2_col.value(i).to_string()))
            .collect();
        assert_eq!(combos.len(), 4, "must have 4 distinct (g1,g2) pairs");
        assert!(combos.contains(&("A".into(), "x".into())));
        assert!(combos.contains(&("A".into(), "y".into())));
        assert!(combos.contains(&("B".into(), "x".into())));
        assert!(combos.contains(&("B".into(), "y".into())));
    }

    /// D-GROUPBY-1 regression (T3.8a): the KdeSpec serde round-trip preserves
    /// the canonical array wire form; deserializing the old bare-string form
    /// yields a one-element vec identical to `vec!["col"]`.
    #[test]
    fn kde_groupby_serde_round_trip_and_back_compat() {
        // Minimal valid KdeSpec JSON (bandwidth and n are required fields).
        // "bandwidth":{"kind":"scott"} is the Scott variant serialized by serde.
        let base_json =
            r#"{"field":"x","bandwidth":{"kind":"scott"},"n":64,"groupby":"col"}"#;

        let spec = KdeSpec {
            field: "x".into(),
            bandwidth: BandwidthSpec::Scott,
            bw_adjust: 1.0,
            n: 64,
            extent: None,
            cumulative: false,
            shared_extent: false,
            kernel: default_kernel(),
            groupby: vec!["col".into()],
            name: None,
        };

        // Serialize: groupby=["col"] must use array form.
        let json = serde_json::to_string(&spec).unwrap();
        assert!(json.contains(r#""groupby":["col"]"#), "must serialize as array, got: {json}");

        // Legacy bare-string deserializes to vec!["col"].
        let from_legacy: KdeSpec = serde_json::from_str(base_json).unwrap();
        assert_eq!(from_legacy.groupby, vec!["col"], "bare-string must parse to [\"col\"]");

        // Legacy null deserializes to [].
        let null_json =
            r#"{"field":"x","bandwidth":{"kind":"scott"},"n":64,"groupby":null}"#;
        let from_null: KdeSpec = serde_json::from_str(null_json).unwrap();
        assert!(from_null.groupby.is_empty(), "null groupby must parse to []");

        // Empty groupby is omitted from serialization.
        let empty = KdeSpec { groupby: vec![], ..spec.clone() };
        let empty_json = serde_json::to_string(&empty).unwrap();
        assert!(!empty_json.contains("groupby"), "empty groupby must be omitted: {empty_json}");
    }
}
