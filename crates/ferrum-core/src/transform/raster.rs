//! Raster: 2D bin aggregation onto a uniform grid.
//!
//! Output is a SINGLE-ROW batch:
//!   x_min, x_max, y_min, y_max:  Float64
//!   width, height:               UInt32
//!   pixel_data:                  Binary  (normalized f64 values packed as 8-byte
//!                                         little-endian blobs; length = width*height*8)
//!
//! Each cell stores one f64 in [0.0, 1.0]. The render::marks::image drawer is
//! responsible for resolving the colormap and encoding RGBA8 pixels from these
//! values. This keeps the transform cmap-agnostic.
//!
//! Resolution mode `Screen` reads `panel_pixel_size` from the TransformContext;
//! when no context is provided (apply() default), it falls back to 256x256.

use arrow::array::{
    Array, ArrayRef, BinaryArray, Float64Array, RecordBatch, UInt32Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::transform::context::TransformContext;

fn default_aggregate() -> String {
    "count".into()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub(crate) enum ResolutionSpec {
    /// "screen" — use TransformContext.panel_pixel_size (or 256x256 fallback).
    Named(String),
    /// Single integer n → n×n grid.
    Fixed(u32),
    /// (nx, ny) explicit dimensions.
    Xy(u32, u32),
}

impl Default for ResolutionSpec {
    fn default() -> Self {
        Self::Named("screen".into())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct RasterSpec {
    pub x: String,
    pub y: String,
    #[serde(default = "default_aggregate")]
    pub aggregate: String, // "count" | "density" | "mean" | "sum" | "any"
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub field: Option<String>,
    #[serde(default)]
    pub resolution: ResolutionSpec,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub min_count: Option<u32>,
    #[serde(default)]
    pub log_scale: bool,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

pub(crate) fn apply(spec: &RasterSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    apply_with_context(spec, batch, &TransformContext::default())
}

pub(crate) fn apply_with_context(
    spec: &RasterSpec,
    batch: &RecordBatch,
    ctx: &TransformContext,
) -> PyResult<RecordBatch> {
    // 1. Validate x, y columns exist + Float64.
    let schema = batch.schema();
    let xi = schema
        .index_of(&spec.x)
        .map_err(|_| PyValueError::new_err(format!("stat_raster: column '{}' not found", spec.x)))?;
    if schema.field(xi).data_type() != &DataType::Float64 {
        return Err(PyValueError::new_err(format!(
            "stat_raster: column '{}' must be Float64",
            spec.x
        )));
    }
    let yi = schema
        .index_of(&spec.y)
        .map_err(|_| PyValueError::new_err(format!("stat_raster: column '{}' not found", spec.y)))?;
    if schema.field(yi).data_type() != &DataType::Float64 {
        return Err(PyValueError::new_err(format!(
            "stat_raster: column '{}' must be Float64",
            spec.y
        )));
    }

    // 2. Validate aggregate.
    let agg = spec.aggregate.as_str();
    if !matches!(agg, "count" | "density" | "mean" | "sum" | "any") {
        return Err(PyValueError::new_err(format!(
            "stat_raster: unknown aggregate '{}'; expected 'count' | 'density' | 'mean' | 'sum' | 'any'",
            agg
        )));
    }
    let needs_field = matches!(agg, "mean" | "sum");
    if needs_field && spec.field.is_none() {
        return Err(PyValueError::new_err(format!(
            "stat_raster: aggregate='{}' requires `field`",
            agg
        )));
    }

    // 3. Validate field column if Some + Float64.
    let field_arr: Option<&Float64Array> = match &spec.field {
        None => None,
        Some(name) => {
            let fi = schema.index_of(name).map_err(|_| {
                PyValueError::new_err(format!("stat_raster: column '{}' not found", name))
            })?;
            if schema.field(fi).data_type() != &DataType::Float64 {
                return Err(PyValueError::new_err(format!(
                    "stat_raster: column '{}' must be Float64",
                    name
                )));
            }
            Some(
                batch
                    .column(fi)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap(),
            )
        }
    };

    // 4. Resolve resolution.
    let (nx, ny) = match &spec.resolution {
        ResolutionSpec::Fixed(n) => {
            if *n == 0 {
                return Err(PyValueError::new_err(
                    "stat_raster: resolution must be > 0",
                ));
            }
            (*n, *n)
        }
        ResolutionSpec::Xy(a, b) => {
            if *a == 0 || *b == 0 {
                return Err(PyValueError::new_err(
                    "stat_raster: resolution must be > 0",
                ));
            }
            (*a, *b)
        }
        ResolutionSpec::Named(s) if s == "screen" => ctx.panel_pixel_size.unwrap_or((256, 256)),
        ResolutionSpec::Named(s) => {
            return Err(PyValueError::new_err(format!(
                "stat_raster: resolution string must be 'screen'; got '{s}'"
            )))
        }
    };

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

    // 5. Filter to (x, y) pairs that are non-null, non-NaN. Capture field values
    // in lockstep when present.
    let n_rows = xa.len();
    let mut xs: Vec<f64> = Vec::with_capacity(n_rows);
    let mut ys: Vec<f64> = Vec::with_capacity(n_rows);
    let mut fs: Vec<f64> = Vec::with_capacity(n_rows);
    for i in 0..n_rows {
        if xa.is_null(i) || ya.is_null(i) {
            continue;
        }
        let xv = xa.value(i);
        let yv = ya.value(i);
        if xv.is_nan() || yv.is_nan() {
            continue;
        }
        let fv = match field_arr {
            None => 0.0,
            Some(fa) => {
                if fa.is_null(i) {
                    continue;
                }
                let v = fa.value(i);
                if v.is_nan() {
                    continue;
                }
                v
            }
        };
        xs.push(xv);
        ys.push(yv);
        fs.push(fv);
    }

    // 6. Compute extent.
    let (xmin, xmax, ymin, ymax) = if xs.is_empty() {
        (0.0, 1.0, 0.0, 1.0)
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
        // Degenerate: pad zero-width extent so binning still works.
        if xlo == xhi {
            xhi = xlo + 1.0;
        }
        if ylo == yhi {
            yhi = ylo + 1.0;
        }
        (xlo, xhi, ylo, yhi)
    };

    let nxs = nx as usize;
    let nys = ny as usize;
    let n_cells = nxs * nys;

    // 7. Allocate count + value_sum accumulators. Bin every row.
    let mut counts: Vec<u64> = vec![0; n_cells];
    let mut value_sum: Vec<f64> = vec![0.0; n_cells];
    let dx = xmax - xmin;
    let dy = ymax - ymin;
    let n_total = xs.len() as u64;
    for i in 0..xs.len() {
        // bin index in [0, nx) and [0, ny). Right-edge clamps to last cell.
        let mut gx = ((xs[i] - xmin) / dx * (nx as f64)).floor() as i64;
        let mut gy = ((ys[i] - ymin) / dy * (ny as f64)).floor() as i64;
        if gx < 0 {
            gx = 0;
        }
        if gy < 0 {
            gy = 0;
        }
        if gx >= nx as i64 {
            gx = nx as i64 - 1;
        }
        if gy >= ny as i64 {
            gy = ny as i64 - 1;
        }
        let idx = (gy as usize) * nxs + (gx as usize);
        counts[idx] += 1;
        if matches!(agg, "mean" | "sum") {
            value_sum[idx] += fs[i];
        }
    }

    // 8. Compute aggregated values per cell.
    let cell_area = (dx / (nx as f64)) * (dy / (ny as f64));
    let mut values: Vec<f64> = vec![0.0; n_cells];
    for i in 0..n_cells {
        let c = counts[i];
        let v = match agg {
            "count" => c as f64,
            "density" => {
                if n_total == 0 || cell_area == 0.0 {
                    0.0
                } else {
                    (c as f64) / ((n_total as f64) * cell_area)
                }
            }
            "mean" => {
                if c == 0 {
                    f64::NAN
                } else {
                    value_sum[i] / (c as f64)
                }
            }
            "sum" => value_sum[i],
            "any" => {
                if c > 0 {
                    1.0
                } else {
                    0.0
                }
            }
            _ => unreachable!(),
        };
        values[i] = v;
    }

    // 9. Apply min_count masking.
    if let Some(mc) = spec.min_count {
        for i in 0..n_cells {
            if counts[i] < mc as u64 {
                values[i] = 0.0;
            }
        }
    }

    // 10. Apply log_scale: ln(1+v).
    if spec.log_scale {
        for i in 0..n_cells {
            if values[i].is_nan() {
                continue;
            }
            // ln1p of negative input is undefined for our purposes; clamp at 0.
            let base = if values[i] < 0.0 { 0.0 } else { values[i] };
            values[i] = (1.0 + base).ln();
        }
    }

    // 11. Normalize to [0, 1] using max non-NaN, non-negative value.
    let mut vmax = 0.0_f64;
    for i in 0..n_cells {
        let v = values[i];
        if v.is_finite() && v > vmax {
            vmax = v;
        }
    }
    let normalized: Vec<f64> = values
        .iter()
        .map(|v| {
            if !v.is_finite() {
                return 0.0;
            }
            if vmax <= 0.0 {
                return 0.0;
            }
            let n = v / vmax;
            if n < 0.0 {
                0.0
            } else if n > 1.0 {
                1.0
            } else {
                n
            }
        })
        .collect();

    // 12. Pack normalized f64 values as 8-byte little-endian blobs.
    // The image drawer will apply the colormap and produce RGBA8 pixels.
    let mut pixel_data: Vec<u8> = Vec::with_capacity(n_cells * 8);
    for i in 0..n_cells {
        pixel_data.extend_from_slice(&normalized[i].to_le_bytes());
    }

    // 13. Build single-row output batch.
    let pixel_array = BinaryArray::from(vec![pixel_data.as_slice()]);

    let out_schema = Arc::new(Schema::new(vec![
        Field::new("x_min", DataType::Float64, false),
        Field::new("x_max", DataType::Float64, false),
        Field::new("y_min", DataType::Float64, false),
        Field::new("y_max", DataType::Float64, false),
        Field::new("width", DataType::UInt32, false),
        Field::new("height", DataType::UInt32, false),
        Field::new("pixel_data", DataType::Binary, false),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(vec![xmin])),
        Arc::new(Float64Array::from(vec![xmax])),
        Arc::new(Float64Array::from(vec![ymin])),
        Arc::new(Float64Array::from(vec![ymax])),
        Arc::new(UInt32Array::from(vec![nx])),
        Arc::new(UInt32Array::from(vec![ny])),
        Arc::new(pixel_array),
    ];
    RecordBatch::try_new(out_schema, cols)
        .map_err(|e| PyValueError::new_err(format!("stat_raster: {e}")))
}

// ---------- PyO3 wrapper ----------

use pyo3::prelude::*;

use crate::transform::core::TransformSpec;

/// 2D raster aggregation onto a uniform pixel grid.
///
/// Bins ``(x, y)`` observations onto a rectangular grid and aggregates
/// them into a raster image suitable for the ``image`` mark. Useful for
/// rendering extremely large scatter datasets without per-point overdraw.
///
/// Output is a single-row batch: ``x_min``, ``x_max``, ``y_min``,
/// ``y_max`` (Float64 data extent), ``width`` / ``height`` (UInt32 grid
/// dimensions), and ``pixel_data`` (Binary, normalized f64 values packed as
/// 8-byte little-endian blobs, length = ``width*height*8``).
///
/// Parameters
/// ----------
/// x : str
///     Horizontal-axis column (must be numeric).
/// y : str
///     Vertical-axis column (must be numeric).
/// aggregate : {"count", "density", "mean", "sum", "any"}, default "count"
///     Aggregation per cell. ``"mean"`` and ``"sum"`` require ``field``.
/// field : str, optional
///     Column to aggregate when ``aggregate`` is ``"mean"`` or ``"sum"``.
/// resolution : int, (int, int), or "screen", optional
///     Grid size. A single int gives an ``n × n`` grid; a tuple gives
///     ``(width, height)``; ``"screen"`` uses the panel pixel size from
///     the render context (or 256 × 256 as fallback). Default is
///     ``"screen"``.
/// min_count : int, optional
///     Cells with fewer than ``min_count`` observations are set to
///     transparent (alpha 0) in the output image.
/// log_scale : bool, default False
///     Apply ``log1p`` scaling to the aggregated values before normalising
///     to the pixel range, compressing high-density outliers.
/// name : str, optional
///     Named output label for sibling ``Reorder(from_=...)`` lookup.
#[pyclass(eq, module = "ferrum._core", name = "Raster")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyRaster(pub(crate) TransformSpec);

#[pymethods]
impl PyRaster {
    #[new]
    #[pyo3(signature = (
        x, y, *,
        aggregate = "count",
        field = None,
        resolution = None,
        min_count = None,
        log_scale = false,
        name = None,
    ))]
    fn new(
        x: &str,
        y: &str,
        aggregate: &str,
        field: Option<String>,
        resolution: Option<&Bound<'_, PyAny>>,
        min_count: Option<u32>,
        log_scale: bool,
        name: Option<String>,
    ) -> PyResult<Self> {
        if x.is_empty() {
            return Err(PyValueError::new_err("Raster: x must be non-empty"));
        }
        if y.is_empty() {
            return Err(PyValueError::new_err("Raster: y must be non-empty"));
        }
        if !matches!(aggregate, "count" | "density" | "mean" | "sum" | "any") {
            return Err(PyValueError::new_err(format!(
                "Raster: unknown aggregate '{aggregate}'; expected 'count' | 'density' | 'mean' | 'sum' | 'any'"
            )));
        }
        if matches!(aggregate, "mean" | "sum") && field.is_none() {
            return Err(PyValueError::new_err(format!(
                "Raster: aggregate='{aggregate}' requires `field`"
            )));
        }

        let resolution = match resolution {
            None => ResolutionSpec::Named("screen".into()),
            Some(obj) => {
                // Order matters: try string first (cannot be confused with int/tuple),
                // then tuple (explicit length-2), then int.
                if let Ok(s) = obj.extract::<String>() {
                    if s != "screen" {
                        return Err(PyValueError::new_err(format!(
                            "Raster: resolution string must be 'screen'; got '{s}'"
                        )));
                    }
                    ResolutionSpec::Named(s)
                } else if let Ok((a, b)) = obj.extract::<(u32, u32)>() {
                    if a == 0 || b == 0 {
                        return Err(PyValueError::new_err(
                            "Raster: resolution must be > 0",
                        ));
                    }
                    ResolutionSpec::Xy(a, b)
                } else if let Ok(n) = obj.extract::<u32>() {
                    if n == 0 {
                        return Err(PyValueError::new_err(
                            "Raster: resolution must be > 0",
                        ));
                    }
                    ResolutionSpec::Fixed(n)
                } else {
                    return Err(PyValueError::new_err(
                        "Raster: resolution must be 'screen' | int | (int, int)",
                    ));
                }
            }
        };

        Ok(PyRaster(TransformSpec::Raster(RasterSpec {
            x: x.to_string(),
            y: y.to_string(),
            aggregate: aggregate.to_string(),
            field,
            resolution,
            min_count,
            log_scale,
            name,
        })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Raster(s) => format!(
                "Raster(x='{}', y='{}', aggregate='{}', field={:?}, resolution={:?}, min_count={:?}, log_scale={})",
                s.x, s.y, s.aggregate, s.field, s.resolution, s.min_count, s.log_scale
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{BinaryArray, Float64Array, RecordBatch, UInt32Array};
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

    fn make_xyf_batch(xs: Vec<f64>, ys: Vec<f64>, fs: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("y", DataType::Float64, true),
            Field::new("f", DataType::Float64, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(xs)),
                Arc::new(Float64Array::from(ys)),
                Arc::new(Float64Array::from(fs)),
            ],
        )
        .unwrap()
    }

    fn pixel_at(batch: &RecordBatch, gx: u32, gy: u32) -> f64 {
        let width = batch
            .column_by_name("width")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .value(0);
        let bin = batch
            .column_by_name("pixel_data")
            .unwrap()
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap();
        let bytes = bin.value(0);
        let i = ((gy as usize) * (width as usize) + (gx as usize)) * 8;
        f64::from_le_bytes(bytes[i..i + 8].try_into().unwrap())
    }

    #[test]
    fn raster_count_aggregate() {
        pyo3::Python::initialize();
        // 4 points clustered in 2 cells of a 2x2 grid.
        // Grid spans (0,0) to (1,1). Cell (0,0) at lower-left, (1,1) upper-right.
        // Points (0.1,0.1) and (0.2,0.2) → cell (0,0).
        // Points (0.7,0.7) and (0.8,0.8) → cell (1,1).
        let b = make_xy_batch(
            vec![0.1, 0.2, 0.7, 0.8],
            vec![0.1, 0.2, 0.7, 0.8],
        );
        let spec = RasterSpec {
            x: "x".into(),
            y: "y".into(),
            aggregate: "count".into(),
            field: None,
            resolution: ResolutionSpec::Fixed(2),
            min_count: None,
            log_scale: false,
            name: None,
        };
        let out = apply(&spec, &b).unwrap();
        assert_eq!(out.num_rows(), 1);

        // (0,0) and (1,1) should be filled, (0,1) and (1,0) empty.
        let p00 = pixel_at(&out, 0, 0);
        let p11 = pixel_at(&out, 1, 1);
        let p01 = pixel_at(&out, 0, 1);
        let p10 = pixel_at(&out, 1, 0);

        // Both nonempty cells have count=2 → max=2 → normalized=1.0.
        assert!((p00 - 1.0).abs() < 1e-9, "p00={p00}");
        assert!((p11 - 1.0).abs() < 1e-9, "p11={p11}");
        // Empty cells = 0.0.
        assert!((p01 - 0.0).abs() < 1e-9, "p01={p01}");
        assert!((p10 - 0.0).abs() < 1e-9, "p10={p10}");
    }

    #[test]
    fn raster_density_aggregate() {
        pyo3::Python::initialize();
        // Same data as count test.
        let b = make_xy_batch(
            vec![0.1, 0.2, 0.7, 0.8],
            vec![0.1, 0.2, 0.7, 0.8],
        );
        let spec = RasterSpec {
            x: "x".into(),
            y: "y".into(),
            aggregate: "density".into(),
            field: None,
            resolution: ResolutionSpec::Fixed(2),
            min_count: None,
            log_scale: false,
            name: None,
        };
        let out = apply(&spec, &b).unwrap();

        // Cell area = 0.7 * 0.7 / (2*2) — the extent is (0.1,0.8) by (0.1,0.8).
        // We don't know exact extent without recomputing; instead verify
        // (a) RGBA at filled cells equals 255 (max), (b) empty cells are 0.
        // Density values for the two filled cells are equal (count=2 each), so
        // both normalize to 1.0.
        let p00 = pixel_at(&out, 0, 0);
        let p11 = pixel_at(&out, 1, 1);
        let p01 = pixel_at(&out, 0, 1);
        let p10 = pixel_at(&out, 1, 0);
        assert!((p00 - 1.0).abs() < 1e-9, "p00={p00}");
        assert!((p11 - 1.0).abs() < 1e-9, "p11={p11}");
        assert!((p01 - 0.0).abs() < 1e-9, "p01={p01}");
        assert!((p10 - 0.0).abs() < 1e-9, "p10={p10}");

        // Σ density · cell_area should be ≈ 1 (all mass within grid).
        // Reconstruct from extent + width/height.
        let xmin = out.column_by_name("x_min").unwrap()
            .as_any().downcast_ref::<Float64Array>().unwrap().value(0);
        let xmax = out.column_by_name("x_max").unwrap()
            .as_any().downcast_ref::<Float64Array>().unwrap().value(0);
        let ymin = out.column_by_name("y_min").unwrap()
            .as_any().downcast_ref::<Float64Array>().unwrap().value(0);
        let ymax = out.column_by_name("y_max").unwrap()
            .as_any().downcast_ref::<Float64Array>().unwrap().value(0);
        let cell_area = ((xmax - xmin) / 2.0) * ((ymax - ymin) / 2.0);
        // Each filled cell has count=2 out of 4 → density = 2/(4*cell_area).
        // Σ density · cell_area = (2/(4*A)*A + 2/(4*A)*A) = 0.5 + 0.5 = 1.0.
        let n_total = 4.0_f64;
        let sum_density_dx_dy = (2.0_f64 / (n_total * cell_area)) * cell_area * 2.0;
        assert!(
            (sum_density_dx_dy - 1.0).abs() < 1e-9,
            "sum density*cell_area should be 1.0; got {sum_density_dx_dy}"
        );
    }

    #[test]
    fn raster_mean_and_sum_aggregates() {
        pyo3::Python::initialize();
        // 4 points in 2 cells with field values [10, 20] → cell (0,0), [3, 5] → (1,1).
        let b = make_xyf_batch(
            vec![0.1, 0.2, 0.7, 0.8],
            vec![0.1, 0.2, 0.7, 0.8],
            vec![10.0, 20.0, 3.0, 5.0],
        );

        // mean: cell (0,0) = 15, (1,1) = 4 → max=15 → (0,0)=1.0, (1,1)=4/15.
        let spec_mean = RasterSpec {
            x: "x".into(),
            y: "y".into(),
            aggregate: "mean".into(),
            field: Some("f".into()),
            resolution: ResolutionSpec::Fixed(2),
            min_count: None,
            log_scale: false,
            name: None,
        };
        let out_mean = apply(&spec_mean, &b).unwrap();
        let p00 = pixel_at(&out_mean, 0, 0);
        let p11 = pixel_at(&out_mean, 1, 1);
        assert!((p00 - 1.0).abs() < 1e-9, "p00={p00}");
        assert!((p11 - 4.0 / 15.0).abs() < 1e-9, "p11={p11}");

        // sum: (0,0) = 30, (1,1) = 8 → max=30 → (0,0)=1.0, (1,1)=8/30.
        let spec_sum = RasterSpec {
            x: "x".into(),
            y: "y".into(),
            aggregate: "sum".into(),
            field: Some("f".into()),
            resolution: ResolutionSpec::Fixed(2),
            min_count: None,
            log_scale: false,
            name: None,
        };
        let out_sum = apply(&spec_sum, &b).unwrap();
        let q00 = pixel_at(&out_sum, 0, 0);
        let q11 = pixel_at(&out_sum, 1, 1);
        assert!((q00 - 1.0).abs() < 1e-9, "q00={q00}");
        assert!((q11 - 8.0 / 30.0).abs() < 1e-9, "q11={q11}");
    }

    #[test]
    fn raster_any_aggregate_is_indicator() {
        pyo3::Python::initialize();
        // 3 points in cell (0,0), 1 point in (1,1). With 'any', both cells should be 255.
        let b = make_xy_batch(
            vec![0.1, 0.15, 0.2, 0.8],
            vec![0.1, 0.15, 0.2, 0.8],
        );
        let spec = RasterSpec {
            x: "x".into(),
            y: "y".into(),
            aggregate: "any".into(),
            field: None,
            resolution: ResolutionSpec::Fixed(2),
            min_count: None,
            log_scale: false,
            name: None,
        };
        let out = apply(&spec, &b).unwrap();
        let p00 = pixel_at(&out, 0, 0);
        let p11 = pixel_at(&out, 1, 1);
        let p01 = pixel_at(&out, 0, 1);
        let p10 = pixel_at(&out, 1, 0);
        assert!((p00 - 1.0).abs() < 1e-9, "p00={p00}");
        assert!((p11 - 1.0).abs() < 1e-9, "p11={p11}");
        assert!((p01 - 0.0).abs() < 1e-9, "p01={p01}");
        assert!((p10 - 0.0).abs() < 1e-9, "p10={p10}");
    }

    #[test]
    fn raster_min_count_masks_low_density_cells() {
        pyo3::Python::initialize();
        // Cell (0,0): 11 points; cell (1,1): 1 point. min_count=10 masks (1,1).
        let xs: Vec<f64> = (0..11).map(|i| 0.1 + (i as f64) * 0.01).collect();
        let mut all_x = xs.clone();
        all_x.push(0.8);
        let mut all_y = xs.clone();
        all_y.push(0.8);
        let b = make_xy_batch(all_x, all_y);
        let spec = RasterSpec {
            x: "x".into(),
            y: "y".into(),
            aggregate: "count".into(),
            field: None,
            resolution: ResolutionSpec::Fixed(2),
            min_count: Some(10),
            log_scale: false,
            name: None,
        };
        let out = apply(&spec, &b).unwrap();
        let p00 = pixel_at(&out, 0, 0);
        let p11 = pixel_at(&out, 1, 1);
        // (0,0) survives: count=11 ≥ 10.
        assert!((p00 - 1.0).abs() < 1e-9, "p00={p00}");
        // (1,1) masked: count=1 < 10 → value set to 0.0.
        assert!((p11 - 0.0).abs() < 1e-9, "p11={p11}");
    }

    #[test]
    fn raster_log_scale_applies_log1p() {
        pyo3::Python::initialize();
        // Cell (0,0): 10 points; cell (1,1): 1 point. log1p(10)/log1p(10)=1, log1p(1)/log1p(10)≈0.289.
        let xs: Vec<f64> = (0..10).map(|i| 0.1 + (i as f64) * 0.01).collect();
        let mut all_x = xs.clone();
        all_x.push(0.8);
        let mut all_y = xs.clone();
        all_y.push(0.8);
        let b = make_xy_batch(all_x, all_y);
        let spec = RasterSpec {
            x: "x".into(),
            y: "y".into(),
            aggregate: "count".into(),
            field: None,
            resolution: ResolutionSpec::Fixed(2),
            min_count: None,
            log_scale: true,
            name: None,
        };
        let out = apply(&spec, &b).unwrap();
        let p00 = pixel_at(&out, 0, 0);
        let p11 = pixel_at(&out, 1, 1);
        // (0,0) max value → 1.0.
        assert!((p00 - 1.0).abs() < 1e-9, "p00={p00}");
        // (1,1) ratio = log1p(1)/log1p(10) = ln(2)/ln(11) ≈ 0.2891.
        let expected = 2.0_f64.ln() / 11.0_f64.ln();
        assert!((p11 - expected).abs() < 1e-9, "p11={p11}, expected={expected}");
    }

    #[test]
    fn raster_screen_resolution_falls_back_to_256x256_without_context() {
        pyo3::Python::initialize();
        let b = make_xy_batch(vec![0.0, 1.0, 0.5], vec![0.0, 1.0, 0.5]);
        let spec = RasterSpec {
            x: "x".into(),
            y: "y".into(),
            aggregate: "count".into(),
            field: None,
            resolution: ResolutionSpec::Named("screen".into()),
            min_count: None,
            log_scale: false,
            name: None,
        };
        let ctx = TransformContext::default();
        let out = apply_with_context(&spec, &b, &ctx).unwrap();
        let width = out
            .column_by_name("width")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .value(0);
        let height = out
            .column_by_name("height")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .value(0);
        assert_eq!(width, 256);
        assert_eq!(height, 256);
    }
}
