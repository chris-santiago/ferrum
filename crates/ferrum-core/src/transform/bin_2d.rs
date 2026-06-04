//! Bin2D — 2D rectangular binning.
//!
//! Input:  two Float64 columns (x, y).
//! Output: [x_lo: Float64, x_hi: Float64, y_lo: Float64, y_hi: Float64, count: Int64]
//!
//! Non-empty cells only when `cumulative=false`; all cells when `cumulative=true`.
//! Used by `jointplot(kind="hist")`.

use arrow::array::{Array, ArrayRef, Float64Array, Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::scale::ticks::sturges_floor;
use crate::transform::numeric_util::coerce_to_float64;

// ---------------------------------------------------------------------------
// Spec types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum BinSpec2DAxis {
    Sturges,
    FreedmanDiaconis,
    Fixed { n: usize },
    Width { w: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct Bin2DSpec {
    pub x: String,
    pub y: String,
    pub bins_x: BinSpec2DAxis,
    pub bins_y: BinSpec2DAxis,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub extent_x: Option<(f64, f64)>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub extent_y: Option<(f64, f64)>,
    #[serde(default)]
    pub cumulative: bool,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

// ---------------------------------------------------------------------------
// Algorithm helpers
// ---------------------------------------------------------------------------

/// Compute IQR for a sorted slice.
fn iqr_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n < 4 {
        return 0.0;
    }
    let q1 = percentile_sorted(sorted, 0.25);
    let q3 = percentile_sorted(sorted, 0.75);
    q3 - q1
}

fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    let h = p * (n as f64 - 1.0);
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] * (hi as f64 - h) + sorted[hi] * (h - lo as f64)
    }
}

/// Resolve a BinSpec2DAxis to a bin count given the data range and count.
fn resolve_bin_count(spec: &BinSpec2DAxis, vals: &[f64], lo: f64, hi: f64) -> PyResult<usize> {
    match spec {
        BinSpec2DAxis::Sturges => Ok(sturges_floor(vals.len())),
        BinSpec2DAxis::FreedmanDiaconis => {
            let mut sorted = vals.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let iqr_val = iqr_sorted(&sorted);
            let h = 2.0 * iqr_val * (vals.len() as f64).powf(-1.0 / 3.0);
            if h > 0.0 && h.is_finite() {
                Ok(((hi - lo) / h).ceil().max(1.0) as usize)
            } else {
                // IQR == 0 or degenerate → fall back to Sturges
                Ok(sturges_floor(vals.len()))
            }
        }
        BinSpec2DAxis::Fixed { n } => {
            if *n == 0 {
                return Err(PyValueError::new_err(
                    "Bin2D: Fixed bin count must be >= 1",
                ));
            }
            Ok(*n)
        }
        BinSpec2DAxis::Width { w } => {
            if !w.is_finite() || *w <= 0.0 {
                return Err(PyValueError::new_err(
                    "Bin2D: Width must be a positive finite number",
                ));
            }
            Ok(((hi - lo) / w).ceil().max(1.0) as usize)
        }
    }
}

// ---------------------------------------------------------------------------
// Main apply function
// ---------------------------------------------------------------------------

pub(crate) fn apply(spec: &Bin2DSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let schema = batch.schema();

    // Validate x column.
    let xi = schema.index_of(&spec.x).map_err(|_| {
        PyValueError::new_err(format!(
            "stat_bin_2d: column '{}' not found; available: {:?}",
            spec.x,
            schema.fields().iter().map(|f| f.name()).collect::<Vec<_>>()
        ))
    })?;

    // Validate y column.
    let yi = schema.index_of(&spec.y).map_err(|_| {
        PyValueError::new_err(format!(
            "stat_bin_2d: column '{}' not found; available: {:?}",
            spec.y,
            schema.fields().iter().map(|f| f.name()).collect::<Vec<_>>()
        ))
    })?;

    // Coerce both axes to Float64 (integer/Float32 columns are accepted).
    let xa = coerce_to_float64(batch.column(xi), "stat_bin_2d", &spec.x)?;
    let ya = coerce_to_float64(batch.column(yi), "stat_bin_2d", &spec.y)?;

    // Drop rows where either x or y is null or NaN.
    let mut xs: Vec<f64> = Vec::with_capacity(xa.len());
    let mut ys: Vec<f64> = Vec::with_capacity(ya.len());
    for i in 0..xa.len() {
        if xa.is_null(i) || ya.is_null(i) {
            continue;
        }
        let xv = xa.value(i);
        let yv = ya.value(i);
        if xv.is_nan() || yv.is_nan() {
            continue;
        }
        // If extents are provided, clamp rows outside to dropped.
        if let Some((x_lo, x_hi)) = spec.extent_x {
            if xv < x_lo || xv > x_hi {
                continue;
            }
        }
        if let Some((y_lo, y_hi)) = spec.extent_y {
            if yv < y_lo || yv > y_hi {
                continue;
            }
        }
        xs.push(xv);
        ys.push(yv);
    }

    // Empty input → empty output (correct schema, zero rows).
    if xs.is_empty() {
        return build_bin2d_batch(
            Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        );
    }

    // Compute extents.
    let (x_min, x_max) = match spec.extent_x {
        Some((a, b)) => (a, b),
        None => {
            let (lo, hi) = xs.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                (a.min(v), b.max(v))
            });
            (lo, hi)
        }
    };

    let (y_min, y_max) = match spec.extent_y {
        Some((a, b)) => (a, b),
        None => {
            let (lo, hi) = ys.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                (a.min(v), b.max(v))
            });
            (lo, hi)
        }
    };

    // Handle degenerate single-value extents → 1 bin of width 1.
    let (bin_count_x, bin_width_x) = if x_min == x_max {
        (1usize, 1.0f64)
    } else {
        let n = resolve_bin_count(&spec.bins_x, &xs, x_min, x_max)?;
        let w = (x_max - x_min) / n as f64;
        (n, w)
    };

    let (bin_count_y, bin_width_y) = if y_min == y_max {
        (1usize, 1.0f64)
    } else {
        let n = resolve_bin_count(&spec.bins_y, &ys, y_min, y_max)?;
        let w = (y_max - y_min) / n as f64;
        (n, w)
    };

    // Allocate count grid: index = iy * bin_count_x + ix.
    let n_cells = bin_count_x * bin_count_y;
    let mut counts: Vec<i64> = vec![0i64; n_cells];

    for i in 0..xs.len() {
        let xv = xs[i];
        let yv = ys[i];

        let ix = ((xv - x_min) / bin_width_x).floor() as i64;
        let iy = ((yv - y_min) / bin_width_y).floor() as i64;

        // Clamp: right-edge inclusive (a value exactly at x_max maps to last bin).
        let ix = ix.max(0).min(bin_count_x as i64 - 1) as usize;
        let iy = iy.max(0).min(bin_count_y as i64 - 1) as usize;

        counts[iy * bin_count_x + ix] += 1;
    }

    // 2D inclusive prefix sum when cumulative=true.
    if spec.cumulative {
        // Row-major: rows = y-bins, cols = x-bins.
        // prefix[r][c] = sum of counts[r'][c'] for r'<=r, c'<=c.
        for r in 0..bin_count_y {
            for c in 0..bin_count_x {
                let above = if r > 0 { counts[(r - 1) * bin_count_x + c] } else { 0 };
                let left  = if c > 0 { counts[r * bin_count_x + (c - 1)] } else { 0 };
                let diag  = if r > 0 && c > 0 { counts[(r - 1) * bin_count_x + (c - 1)] } else { 0 };
                counts[r * bin_count_x + c] += above + left - diag;
            }
        }
    }

    // Build output arrays.
    let mut out_x_lo: Vec<f64> = Vec::new();
    let mut out_x_hi: Vec<f64> = Vec::new();
    let mut out_y_lo: Vec<f64> = Vec::new();
    let mut out_y_hi: Vec<f64> = Vec::new();
    let mut out_count: Vec<i64> = Vec::new();

    for r in 0..bin_count_y {
        for c in 0..bin_count_x {
            let cnt = counts[r * bin_count_x + c];
            if !spec.cumulative && cnt == 0 {
                continue; // Skip empty cells in non-cumulative mode.
            }
            let xlo = x_min + c as f64 * bin_width_x;
            let xhi = x_min + (c + 1) as f64 * bin_width_x;
            let ylo = y_min + r as f64 * bin_width_y;
            let yhi = y_min + (r + 1) as f64 * bin_width_y;
            out_x_lo.push(xlo);
            out_x_hi.push(xhi);
            out_y_lo.push(ylo);
            out_y_hi.push(yhi);
            out_count.push(cnt);
        }
    }

    build_bin2d_batch(out_x_lo, out_x_hi, out_y_lo, out_y_hi, out_count)
}

fn build_bin2d_batch(
    x_lo: Vec<f64>,
    x_hi: Vec<f64>,
    y_lo: Vec<f64>,
    y_hi: Vec<f64>,
    count: Vec<i64>,
) -> PyResult<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x_lo",  DataType::Float64, false),
        Field::new("x_hi",  DataType::Float64, false),
        Field::new("y_lo",  DataType::Float64, false),
        Field::new("y_hi",  DataType::Float64, false),
        Field::new("count", DataType::Int64,   false),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(x_lo)),
        Arc::new(Float64Array::from(x_hi)),
        Arc::new(Float64Array::from(y_lo)),
        Arc::new(Float64Array::from(y_hi)),
        Arc::new(Int64Array::from(count)),
    ];
    RecordBatch::try_new(schema, cols)
        .map_err(|e| PyValueError::new_err(format!("stat_bin_2d: {e}")))
}

// ---------------------------------------------------------------------------
// PyO3 wrapper
// ---------------------------------------------------------------------------

use crate::transform::core::TransformSpec;

/// Two-dimensional binning over a pair of numeric columns.
///
/// Partitions ``x`` and ``y`` into a rectangular grid of cells and counts
/// (or aggregates) observations per cell. Used for 2D histograms and heat
/// maps where hex-binning is not desired.
///
/// Output columns: ``x_lo``, ``x_hi``, ``y_lo``, ``y_hi`` (Float64 bin
/// edges) and ``count`` (Int64). When ``cumulative=True`` an additional
/// ``cumulative`` column is appended.
///
/// Parameters
/// ----------
/// x : str
///     Column for the horizontal axis (must be numeric).
/// y : str
///     Column for the vertical axis (must be numeric).
/// bins_x : int, float, or {"sturges", "fd"}, optional
///     Bin count (int), bin width (float), or rule name for the x-axis.
///     Default is ``"sturges"``.
/// bins_y : int, float, or {"sturges", "fd"}, optional
///     Same as ``bins_x`` but for the y-axis. Default is ``"sturges"``.
/// extent_x : (float, float), optional
///     ``(lo, hi)`` clipping range for the x-axis.
/// extent_y : (float, float), optional
///     ``(lo, hi)`` clipping range for the y-axis.
/// cumulative : bool, default False
///     Append a running total column.
/// name : str, optional
///     Named output label for sibling ``Reorder(from_=...)`` lookup.
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> fm.Chart(df).mark_heatmap(bin_count=20).encode(x="x", y="y")
#[pyclass(eq, module = "ferrum._core", name = "Bin2D")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyBin2D(pub(crate) TransformSpec);

#[pymethods]
impl PyBin2D {
    #[new]
    #[pyo3(signature = (
        x, y, *,
        bins_x = None,
        bins_y = None,
        extent_x = None,
        extent_y = None,
        cumulative = false,
        name = None,
    ))]
    fn new(
        x: &str,
        y: &str,
        bins_x: Option<&Bound<'_, PyAny>>,
        bins_y: Option<&Bound<'_, PyAny>>,
        extent_x: Option<(f64, f64)>,
        extent_y: Option<(f64, f64)>,
        cumulative: bool,
        name: Option<String>,
    ) -> PyResult<Self> {
        let bins_x = match bins_x {
            None => BinSpec2DAxis::Sturges,
            Some(obj) => parse_bin_axis(obj)?,
        };
        let bins_y = match bins_y {
            None => BinSpec2DAxis::Sturges,
            Some(obj) => parse_bin_axis(obj)?,
        };
        Ok(PyBin2D(TransformSpec::Bin2D(Bin2DSpec {
            x: x.into(),
            y: y.into(),
            bins_x,
            bins_y,
            extent_x,
            extent_y,
            cumulative,
            name,
        })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Bin2D(s) => format!(
                "Bin2D(x='{}', y='{}', cumulative={})",
                s.x,
                s.y,
                s.cumulative,
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

fn parse_bin_axis(obj: &Bound<'_, PyAny>) -> PyResult<BinSpec2DAxis> {
    if let Ok(s) = obj.extract::<&str>() {
        return match s {
            "sturges" => Ok(BinSpec2DAxis::Sturges),
            "fd" | "freedman_diaconis" => Ok(BinSpec2DAxis::FreedmanDiaconis),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Bin2D: unknown bins value '{s}'; expected 'sturges'|'fd'|int|float"
            ))),
        };
    }
    if let Ok(n) = obj.extract::<usize>() {
        return Ok(BinSpec2DAxis::Fixed { n });
    }
    if let Ok(w) = obj.extract::<f64>() {
        return Ok(BinSpec2DAxis::Width { w });
    }
    Err(pyo3::exceptions::PyValueError::new_err(
        "Bin2D: bins must be 'sturges'|'fd'|int|float",
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int64Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn make_xy_batch(xs: Vec<f64>, ys: Vec<f64>) -> RecordBatch {
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

    fn col_f64<'a>(b: &'a RecordBatch, name: &str) -> &'a Float64Array {
        b.column(b.schema().index_of(name).unwrap())
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
    }

    fn col_i64<'a>(b: &'a RecordBatch, name: &str) -> &'a Int64Array {
        b.column(b.schema().index_of(name).unwrap())
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
    }

    // Test 1: 3×3 grid correctness — 9 evenly-spaced points in [0,1]×[0,1] with
    // Fixed(3) bins each → 9 cells, each with count=1.
    #[test]
    fn test_bin2d_3x3_grid_each_cell_count_one() {
        // Points at centres of a 3×3 grid.
        let xs = vec![
            1.0/6.0, 1.0/6.0, 1.0/6.0,
            3.0/6.0, 3.0/6.0, 3.0/6.0,
            5.0/6.0, 5.0/6.0, 5.0/6.0,
        ];
        let ys = vec![
            1.0/6.0, 3.0/6.0, 5.0/6.0,
            1.0/6.0, 3.0/6.0, 5.0/6.0,
            1.0/6.0, 3.0/6.0, 5.0/6.0,
        ];
        let batch = make_xy_batch(xs, ys);
        let spec = Bin2DSpec {
            x: "x".into(), y: "y".into(),
            bins_x: BinSpec2DAxis::Fixed { n: 3 },
            bins_y: BinSpec2DAxis::Fixed { n: 3 },
            extent_x: Some((0.0, 1.0)),
            extent_y: Some((0.0, 1.0)),
            cumulative: false,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 9, "expected 9 non-empty cells");
        let counts = col_i64(&out, "count");
        for i in 0..9 {
            assert_eq!(counts.value(i), 1, "cell {i} should have count=1");
        }
    }

    // Test 2: Sturges floor honored — 100 gaussian-like points with Sturges
    // should produce bin_count_x == sturges_floor(100) == 8.
    #[test]
    fn test_bin2d_sturges_floor_honored() {
        // 100 points uniformly spaced in [0,1].
        let xs: Vec<f64> = (0..100).map(|i| i as f64 / 99.0).collect();
        let ys: Vec<f64> = (0..100).map(|i| (i as f64 / 99.0) * 2.0 - 1.0).collect();
        let batch = make_xy_batch(xs, ys);
        let spec = Bin2DSpec {
            x: "x".into(), y: "y".into(),
            bins_x: BinSpec2DAxis::Sturges,
            bins_y: BinSpec2DAxis::Fixed { n: 1 }, // any y binning
            extent_x: None, extent_y: None,
            cumulative: true, // emit all cells so we can count distinct x_lo values
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let x_lo = col_f64(&out, "x_lo");
        // Collect distinct x_lo values.
        let mut distinct_x: Vec<f64> = (0..out.num_rows()).map(|i| x_lo.value(i)).collect();
        distinct_x.sort_by(|a, b| a.partial_cmp(b).unwrap());
        distinct_x.dedup();
        assert_eq!(
            distinct_x.len(),
            sturges_floor(100),
            "expected sturges_floor(100)={} x-bins; got {}",
            sturges_floor(100),
            distinct_x.len()
        );
    }

    // Test 3: Cumulative monotonicity — last cell count = N; non-decreasing
    // along each row and column.
    #[test]
    fn test_bin2d_cumulative_monotonicity() {
        let n = 20usize;
        let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let ys: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let batch = make_xy_batch(xs, ys);
        let bins = 4usize;
        let spec = Bin2DSpec {
            x: "x".into(), y: "y".into(),
            bins_x: BinSpec2DAxis::Fixed { n: bins },
            bins_y: BinSpec2DAxis::Fixed { n: bins },
            extent_x: None, extent_y: None,
            cumulative: true,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), bins * bins, "cumulative must emit all cells");

        let counts = col_i64(&out, "count");
        // Last cell (bottom-right in row-major after sorting) should equal n.
        let last = counts.value(bins * bins - 1);
        assert_eq!(last, n as i64, "last cell should equal total N={n}");

        // Non-decreasing along each row.
        for r in 0..bins {
            for c in 1..bins {
                let prev = counts.value(r * bins + (c - 1));
                let curr = counts.value(r * bins + c);
                assert!(curr >= prev, "row {r} not monotone at col {c}: {prev} > {curr}");
            }
        }
        // Non-decreasing along each column.
        for c in 0..bins {
            for r in 1..bins {
                let prev = counts.value((r - 1) * bins + c);
                let curr = counts.value(r * bins + c);
                assert!(curr >= prev, "col {c} not monotone at row {r}: {prev} > {curr}");
            }
        }
    }

    // Test 4: Extent-clamped — extent_x=(0,1) with a point at x=2.0 → that
    // row is dropped and total count is one less.
    #[test]
    fn test_bin2d_extent_clamps_out_of_range_rows() {
        let xs = vec![0.25, 0.75, 2.0]; // last point outside extent_x
        let ys = vec![0.5,  0.5,  0.5];
        let batch = make_xy_batch(xs, ys);
        let spec = Bin2DSpec {
            x: "x".into(), y: "y".into(),
            bins_x: BinSpec2DAxis::Fixed { n: 2 },
            bins_y: BinSpec2DAxis::Fixed { n: 1 },
            extent_x: Some((0.0, 1.0)),
            extent_y: None,
            cumulative: false,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let counts = col_i64(&out, "count");
        let total: i64 = (0..out.num_rows()).map(|i| counts.value(i)).sum();
        assert_eq!(total, 2, "x=2.0 outside extent_x=(0,1) should be dropped; total={total}");
    }

    // Int64 x AND y must be coerced to Float64, not rejected.
    #[test]
    fn test_bin2d_int64_xy_columns_are_coerced() {
        use arrow::array::Int64Array;
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int64, true),
            Field::new("y", DataType::Int64, true),
        ]));
        let xs: Vec<i64> = (0..9).map(|i| i % 3).collect();
        let ys: Vec<i64> = (0..9).map(|i| i / 3).collect();
        let int_batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(xs.clone())),
                Arc::new(Int64Array::from(ys.clone())),
            ],
        )
        .unwrap();
        let float_batch = make_xy_batch(
            xs.iter().map(|&v| v as f64).collect(),
            ys.iter().map(|&v| v as f64).collect(),
        );
        let spec = Bin2DSpec {
            x: "x".into(),
            y: "y".into(),
            bins_x: BinSpec2DAxis::Fixed { n: 3 },
            bins_y: BinSpec2DAxis::Fixed { n: 3 },
            extent_x: None,
            extent_y: None,
            cumulative: true,
            name: None,
        };
        let int_out = apply(&spec, &int_batch).expect("Int64 bin_2d should succeed");
        let float_out = apply(&spec, &float_batch).unwrap();
        assert_eq!(int_out.num_rows(), float_out.num_rows());
        let ic = col_i64(&int_out, "count");
        let fc = col_i64(&float_out, "count");
        for i in 0..int_out.num_rows() {
            assert_eq!(ic.value(i), fc.value(i), "count[{i}] mismatch");
        }
    }

    #[test]
    fn test_bin2d_string_column_still_errors() {
        use arrow::array::StringArray;
        pyo3::Python::initialize();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Utf8, true),
            Field::new("y", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "b"])),
                Arc::new(Float64Array::from(vec![0.0, 1.0])),
            ],
        )
        .unwrap();
        let spec = Bin2DSpec {
            x: "x".into(),
            y: "y".into(),
            bins_x: BinSpec2DAxis::Fixed { n: 2 },
            bins_y: BinSpec2DAxis::Fixed { n: 2 },
            extent_x: None,
            extent_y: None,
            cumulative: false,
            name: None,
        };
        let err = apply(&spec, &batch).unwrap_err();
        assert!(
            err.to_string().contains("must be numeric"),
            "err: {err}"
        );
    }

    // Test 5: Empty input → empty output with correct schema.
    #[test]
    fn test_bin2d_empty_input_empty_output() {
        let batch = make_xy_batch(vec![], vec![]);
        let spec = Bin2DSpec {
            x: "x".into(), y: "y".into(),
            bins_x: BinSpec2DAxis::Fixed { n: 3 },
            bins_y: BinSpec2DAxis::Fixed { n: 3 },
            extent_x: None, extent_y: None,
            cumulative: false,
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 0);
        assert_eq!(out.num_columns(), 5);
        let schema = out.schema();
        assert_eq!(schema.field(0).name(), "x_lo");
        assert_eq!(schema.field(4).name(), "count");
        assert_eq!(schema.field(4).data_type(), &DataType::Int64);
    }
}
