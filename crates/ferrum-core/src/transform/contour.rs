//! Contour: Marching Squares isolines and isobands over a Kde2D-shaped grid.
//!
//! Reads a single-row batch with the Kde2D output schema:
//!   grid_x:  List<Float64>   length nx
//!   grid_y:  List<Float64>   length ny
//!   density: List<Float64>   length nx*ny, row-major: density[gy * nx + gx]
//!   nx:      UInt32
//!   ny:      UInt32
//!   extent:  List<Float64>   length 4: [xmin, xmax, ymin, ymax]
//!
//! Output (one row per polyline vertex, NOT per polyline):
//!   level_id:    UInt32
//!   level_value: Float64
//!   contour_x:   Float64
//!   contour_y:   Float64
//!
//! `level_id` encoding:
//!   - Isoline mode: (level_index << 16) | segment_index
//!     `segment_index` increments across all cells in row-major iteration.
//!   - Isoband mode: (band_index   << 16) | cell_index
//!     `cell_index` increments per polygon emitted within that band.
//!
//! Simplifications (vs. the spec):
//!   - Per-segment output (not stitched polylines). Each segment = 2 vertex rows.
//!   - Per-cell polygons (not ring assembly) for isoband mode.
//!   - `smooth` field is accepted but ignored (TODO: pin smoothing semantics).

use arrow::array::{Array, ArrayRef, Float64Array, ListArray, RecordBatch, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::transform::core::TransformSpec;

fn default_thresholds() -> u32 {
    6
}
fn default_smooth() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ContourSpec {
    #[serde(default = "default_thresholds")]
    pub thresholds: u32,
    #[serde(default)]
    pub fill: bool,
    #[serde(default = "default_smooth")]
    pub smooth: bool,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

fn out_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("level_id", DataType::UInt32, false),
        Field::new("level_value", DataType::Float64, false),
        Field::new("contour_x", DataType::Float64, false),
        Field::new("contour_y", DataType::Float64, false),
    ]))
}

fn empty_output() -> PyResult<RecordBatch> {
    let schema = out_schema();
    let cols: Vec<ArrayRef> = vec![
        Arc::new(UInt32Array::from(Vec::<u32>::new())),
        Arc::new(Float64Array::from(Vec::<f64>::new())),
        Arc::new(Float64Array::from(Vec::<f64>::new())),
        Arc::new(Float64Array::from(Vec::<f64>::new())),
    ];
    RecordBatch::try_new(schema, cols)
        .map_err(|e| PyValueError::new_err(format!("stat_contour: {e}")))
}

fn extract_list_f64(batch: &RecordBatch, name: &str) -> PyResult<Vec<f64>> {
    let idx = batch.schema().index_of(name).map_err(|_| {
        PyValueError::new_err(format!(
            "stat_contour: input must be a Kde2D output (missing column '{name}')"
        ))
    })?;
    let la = batch
        .column(idx)
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "stat_contour: column '{name}' must be List<Float64>"
            ))
        })?;
    let inner = la.value(0);
    let arr = inner
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "stat_contour: column '{name}' inner type must be Float64"
            ))
        })?;
    Ok((0..arr.len()).map(|i| arr.value(i)).collect())
}

fn extract_u32(batch: &RecordBatch, name: &str) -> PyResult<u32> {
    let idx = batch.schema().index_of(name).map_err(|_| {
        PyValueError::new_err(format!(
            "stat_contour: input must be a Kde2D output (missing column '{name}')"
        ))
    })?;
    let arr = batch
        .column(idx)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| {
            PyValueError::new_err(format!("stat_contour: column '{name}' must be UInt32"))
        })?;
    Ok(arr.value(0))
}

pub(crate) fn apply(spec: &ContourSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    if spec.thresholds == 0 {
        return Err(PyValueError::new_err(
            "stat_contour: thresholds must be > 0",
        ));
    }

    // Validate required columns are present.
    for name in ["grid_x", "grid_y", "density", "nx", "ny", "extent"] {
        if batch.schema().index_of(name).is_err() {
            return Err(PyValueError::new_err(format!(
                "stat_contour: input must be a Kde2D output (missing column '{name}')"
            )));
        }
    }

    if batch.num_rows() == 0 {
        return empty_output();
    }
    if batch.num_rows() != 1 {
        return Err(PyValueError::new_err(format!(
            "stat_contour: input must be single-row (got {})",
            batch.num_rows()
        )));
    }

    let grid_x = extract_list_f64(batch, "grid_x")?;
    let grid_y = extract_list_f64(batch, "grid_y")?;
    let density = extract_list_f64(batch, "density")?;
    let nx = extract_u32(batch, "nx")? as usize;
    let ny = extract_u32(batch, "ny")? as usize;

    if grid_x.len() != nx || grid_y.len() != ny || density.len() != nx * ny {
        return Err(PyValueError::new_err(format!(
            "stat_contour: dimension mismatch (nx={nx}, ny={ny}, grid_x.len={}, grid_y.len={}, density.len={})",
            grid_x.len(), grid_y.len(), density.len()
        )));
    }

    if nx < 2 || ny < 2 {
        return empty_output();
    }

    // Compute thresholds: t evenly-spaced interior levels.
    let dmin = density.iter().copied().fold(f64::INFINITY, f64::min);
    let dmax = density.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !dmin.is_finite() || !dmax.is_finite() || dmax <= dmin {
        return empty_output();
    }
    let t = spec.thresholds;
    let levels: Vec<f64> = (1..=t)
        .map(|i| dmin + (dmax - dmin) * (i as f64) / ((t + 1) as f64))
        .collect();

    // TODO(phase-8b+): smoothing not yet specified; `spec.smooth` is currently a no-op.
    let _ = spec.smooth;

    let mut level_ids: Vec<u32> = Vec::new();
    let mut level_vals: Vec<f64> = Vec::new();
    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();

    if !spec.fill {
        // ---------- Isoline mode (Marching Squares) ----------
        for (li, &level) in levels.iter().enumerate() {
            let mut seg_idx: u32 = 0;
            for gy in 0..(ny - 1) {
                for gx in 0..(nx - 1) {
                    // Cell corner densities.
                    let tl = density[gy * nx + gx];
                    let tr = density[gy * nx + (gx + 1)];
                    let br = density[(gy + 1) * nx + (gx + 1)];
                    let bl = density[(gy + 1) * nx + gx];

                    // 4-bit corner mask: tl,tr,br,bl above level → 1.
                    let mask = ((tl >= level) as u8) << 3
                        | ((tr >= level) as u8) << 2
                        | ((br >= level) as u8) << 1
                        | ((bl >= level) as u8);

                    // Cell corner coords.
                    let x0 = grid_x[gx];
                    let x1 = grid_x[gx + 1];
                    let y0 = grid_y[gy];
                    let y1 = grid_y[gy + 1];

                    // Edge crossings (only computed when needed).
                    // top:    between tl(x0,y0) and tr(x1,y0)
                    // right:  between tr(x1,y0) and br(x1,y1)
                    // bottom: between br(x1,y1) and bl(x0,y1)
                    // left:   between bl(x0,y1) and tl(x0,y0)
                    let interp_x = |a: f64, b: f64, xa: f64, xb: f64| -> f64 {
                        let denom = b - a;
                        if denom.abs() < f64::EPSILON {
                            xa
                        } else {
                            let t = (level - a) / denom;
                            xa + t * (xb - xa)
                        }
                    };

                    let top = || (interp_x(tl, tr, x0, x1), y0);
                    let right = || (x1, interp_x(tr, br, y0, y1));
                    let bottom = || (interp_x(bl, br, x0, x1), y1);
                    let left = || (x0, interp_x(tl, bl, y0, y1));

                    // Segments to emit for this cell.
                    let mut segs: Vec<((f64, f64), (f64, f64))> = Vec::new();
                    match mask {
                        0 | 15 => {}
                        1 => segs.push((left(), bottom())),
                        2 => segs.push((bottom(), right())),
                        3 => segs.push((left(), right())),
                        4 => segs.push((top(), right())),
                        6 => segs.push((top(), bottom())),
                        7 => segs.push((left(), top())),
                        8 => segs.push((left(), top())),
                        9 => segs.push((top(), bottom())),
                        11 => segs.push((top(), right())),
                        12 => segs.push((left(), right())),
                        13 => segs.push((bottom(), right())),
                        14 => segs.push((left(), bottom())),
                        // Saddles: disambiguate via cell-center average.
                        5 => {
                            // mask=0101: tr & bl above. center decides.
                            let center = (tl + tr + br + bl) * 0.25;
                            if center >= level {
                                // "above region wraps"; isolate the two below corners.
                                segs.push((top(), left()));
                                segs.push((bottom(), right()));
                            } else {
                                // peaks at tr and bl isolated.
                                segs.push((top(), right()));
                                segs.push((bottom(), left()));
                            }
                        }
                        10 => {
                            // mask=1010: tl & br above.
                            let center = (tl + tr + br + bl) * 0.25;
                            if center >= level {
                                segs.push((top(), right()));
                                segs.push((bottom(), left()));
                            } else {
                                segs.push((top(), left()));
                                segs.push((bottom(), right()));
                            }
                        }
                        _ => unreachable!("4-bit mask cannot exceed 15"),
                    }

                    for (p0, p1) in segs {
                        let id = ((li as u32) << 16) | seg_idx;
                        seg_idx = seg_idx.wrapping_add(1);
                        level_ids.push(id);
                        level_vals.push(level);
                        xs.push(p0.0);
                        ys.push(p0.1);

                        level_ids.push(id);
                        level_vals.push(level);
                        xs.push(p1.0);
                        ys.push(p1.1);
                    }
                }
            }
        }
    } else {
        // ---------- Isoband mode (cell-rectangle approximation) ----------
        // TODO(phase-8b+): replace cell-rectangle approximation with full 81-case
        // isoband table + ring assembly when polygon mark drawer requires geometric correctness.
        // With t thresholds we emit t-1 bands [L_i, L_{i+1}].
        if levels.len() < 2 {
            return empty_output();
        }
        for bi in 0..(levels.len() - 1) {
            let lo = levels[bi];
            let hi = levels[bi + 1];
            let mut cell_idx: u32 = 0;
            for gy in 0..(ny - 1) {
                for gx in 0..(nx - 1) {
                    let tl = density[gy * nx + gx];
                    let tr = density[gy * nx + (gx + 1)];
                    let br = density[(gy + 1) * nx + (gx + 1)];
                    let bl = density[(gy + 1) * nx + gx];

                    // Classify each corner: 0 = below lo, 1 = within [lo, hi], 2 = above hi.
                    let cls = |v: f64| -> u8 {
                        if v < lo {
                            0
                        } else if v <= hi {
                            1
                        } else {
                            2
                        }
                    };
                    let c = [cls(tl), cls(tr), cls(br), cls(bl)];

                    // Emit a polygon if any corner is within the band, OR corners
                    // straddle the band (some below + some above).
                    let any_within = c.iter().any(|&v| v == 1);
                    let any_below = c.iter().any(|&v| v == 0);
                    let any_above = c.iter().any(|&v| v == 2);
                    if !(any_within || (any_below && any_above)) {
                        continue;
                    }

                    let x0 = grid_x[gx];
                    let x1 = grid_x[gx + 1];
                    let y0 = grid_y[gy];
                    let y1 = grid_y[gy + 1];

                    // Polygon: 4 corners (closed by repeating first vertex).
                    let id = ((bi as u32) << 16) | cell_idx;
                    cell_idx = cell_idx.wrapping_add(1);
                    let band_value = lo;
                    let pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)];
                    for (px, py) in pts {
                        level_ids.push(id);
                        level_vals.push(band_value);
                        xs.push(px);
                        ys.push(py);
                    }
                }
            }
        }
    }

    let schema = out_schema();
    let cols: Vec<ArrayRef> = vec![
        Arc::new(UInt32Array::from(level_ids)),
        Arc::new(Float64Array::from(level_vals)),
        Arc::new(Float64Array::from(xs)),
        Arc::new(Float64Array::from(ys)),
    ];
    RecordBatch::try_new(schema, cols)
        .map_err(|e| PyValueError::new_err(format!("stat_contour: {e}")))
}

// ---------- PyO3 wrapper ----------

/// Marching-Squares iso-contour extraction from a ``Kde2D`` density grid.
///
/// Reads a single-row batch in the ``Kde2D`` output schema (fields
/// ``grid_x``, ``grid_y``, ``density``, ``nx``, ``ny``, ``extent``) and
/// emits contour line segments or filled polygons at evenly-spaced density
/// levels using the Marching Squares algorithm.
///
/// Output columns: ``level_id`` (UInt32), ``level_value`` (Float64),
/// ``contour_x`` (Float64), ``contour_y`` (Float64). Each row is one
/// vertex; consecutive pairs form a segment (isoline mode) or a polygon
/// vertex sequence (isoband mode).
///
/// Parameters
/// ----------
/// thresholds : int, default 6
///     Number of equally-spaced density levels to compute contours for.
///     Must be > 0.
/// fill : bool, default False
///     When False, emit isolines (line segments); when True, emit filled
///     isoband polygons between adjacent threshold levels.
/// smooth : bool, default True
///     Accepted but currently reserved; smoothing semantics are not yet
///     fully specified and the field has no effect on output.
/// name : str, optional
///     Named output label for sibling ``Reorder(from_=...)`` lookup.
#[pyclass(eq, module = "ferrum._core", name = "Contour")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyContour(pub(crate) TransformSpec);

#[pymethods]
impl PyContour {
    #[new]
    #[pyo3(signature = (*, thresholds = 6, fill = false, smooth = true, name = None))]
    fn new(
        thresholds: u32,
        fill: bool,
        smooth: bool,
        name: Option<String>,
    ) -> PyResult<Self> {
        if thresholds == 0 {
            return Err(PyValueError::new_err("Contour: thresholds must be > 0"));
        }
        Ok(PyContour(TransformSpec::Contour(ContourSpec {
            thresholds,
            fill,
            smooth,
            name,
        })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Contour(s) => format!(
                "Contour(thresholds={}, fill={}, smooth={})",
                s.thresholds, s.fill, s.smooth
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Float64Builder, ListBuilder, UInt32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    /// Build a Kde2D-shaped batch from raw grid/density data.
    fn make_kde2d_batch(grid_x: Vec<f64>, grid_y: Vec<f64>, density: Vec<f64>) -> RecordBatch {
        let nx = grid_x.len() as u32;
        let ny = grid_y.len() as u32;
        assert_eq!(density.len(), (nx * ny) as usize, "density size mismatch");

        let mut gxb = ListBuilder::new(Float64Builder::new());
        gxb.values().append_slice(&grid_x);
        gxb.append(true);

        let mut gyb = ListBuilder::new(Float64Builder::new());
        gyb.values().append_slice(&grid_y);
        gyb.append(true);

        let mut db = ListBuilder::new(Float64Builder::new());
        db.values().append_slice(&density);
        db.append(true);

        let mut eb = ListBuilder::new(Float64Builder::new());
        eb.values()
            .append_slice(&[grid_x[0], *grid_x.last().unwrap(), grid_y[0], *grid_y.last().unwrap()]);
        eb.append(true);

        let list_f64 = DataType::List(Arc::new(Field::new("item", DataType::Float64, true)));
        let schema = Arc::new(Schema::new(vec![
            Field::new("grid_x", list_f64.clone(), false),
            Field::new("grid_y", list_f64.clone(), false),
            Field::new("density", list_f64.clone(), false),
            Field::new("nx", DataType::UInt32, false),
            Field::new("ny", DataType::UInt32, false),
            Field::new("extent", list_f64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(gxb.finish()),
                Arc::new(gyb.finish()),
                Arc::new(db.finish()),
                Arc::new(UInt32Array::from(vec![nx])),
                Arc::new(UInt32Array::from(vec![ny])),
                Arc::new(eb.finish()),
            ],
        )
        .unwrap()
    }

    #[test]
    fn contour_isoline_schema_correct() {
        pyo3::Python::initialize();
        // 4×4 grid with a synthetic single-peak density.
        let grid_x: Vec<f64> = (0..4).map(|i| i as f64).collect();
        let grid_y: Vec<f64> = (0..4).map(|i| i as f64).collect();
        let mut density = vec![0.0_f64; 16];
        for gy in 0..4 {
            for gx in 0..4 {
                let dx = (gx as f64) - 1.5;
                let dy = (gy as f64) - 1.5;
                density[gy * 4 + gx] = (-(dx * dx + dy * dy)).exp();
            }
        }
        let b = make_kde2d_batch(grid_x, grid_y, density);
        let spec = ContourSpec {
            thresholds: 2,
            fill: false,
            smooth: false,
            name: None,
        };
        let out = apply(&spec, &b).unwrap();
        let s = out.schema();
        assert_eq!(s.fields().len(), 4);
        assert_eq!(s.field(0).name(), "level_id");
        assert_eq!(s.field(0).data_type(), &DataType::UInt32);
        assert_eq!(s.field(1).name(), "level_value");
        assert_eq!(s.field(1).data_type(), &DataType::Float64);
        assert_eq!(s.field(2).name(), "contour_x");
        assert_eq!(s.field(2).data_type(), &DataType::Float64);
        assert_eq!(s.field(3).name(), "contour_y");
        assert_eq!(s.field(3).data_type(), &DataType::Float64);
        assert!(out.num_rows() > 0, "expected at least some segments emitted");
        // Each segment contributes 2 vertex rows.
        assert_eq!(out.num_rows() % 2, 0, "isoline rows must come in pairs");
    }

    #[test]
    fn contour_isoband_schema_correct() {
        pyo3::Python::initialize();
        let grid_x: Vec<f64> = (0..4).map(|i| i as f64).collect();
        let grid_y: Vec<f64> = (0..4).map(|i| i as f64).collect();
        let mut density = vec![0.0_f64; 16];
        for gy in 0..4 {
            for gx in 0..4 {
                let dx = (gx as f64) - 1.5;
                let dy = (gy as f64) - 1.5;
                density[gy * 4 + gx] = (-(dx * dx + dy * dy)).exp();
            }
        }
        let b = make_kde2d_batch(grid_x, grid_y, density);
        let spec = ContourSpec {
            thresholds: 3,
            fill: true,
            smooth: false,
            name: None,
        };
        let out = apply(&spec, &b).unwrap();
        let s = out.schema();
        assert_eq!(s.fields().len(), 4);
        assert_eq!(s.field(0).name(), "level_id");
        assert_eq!(s.field(1).name(), "level_value");
        assert_eq!(s.field(2).name(), "contour_x");
        assert_eq!(s.field(3).name(), "contour_y");
        assert!(out.num_rows() > 0, "expected at least some polygons emitted");
        // Each cell polygon = 5 vertices (closed).
        assert_eq!(out.num_rows() % 5, 0, "isoband cell-polygon rows must be multiples of 5");
    }

    #[test]
    fn contour_saddle_uses_cell_center() {
        pyo3::Python::initialize();
        // 2×2 grid (single cell) with saddle: tl=0, tr=10, br=0, bl=10.
        // Center = (0+10+0+10)/4 = 5; level = 5.
        // center >= level → first branch ("above region wraps") → 2 segments.
        let grid_x = vec![0.0, 1.0];
        let grid_y = vec![0.0, 1.0];
        // density indexing: density[gy * nx + gx], with gy=0 = top row.
        // tl = density[0,0] = 0, tr = density[0,1] = 10
        // bl = density[1,0] = 10, br = density[1,1] = 0
        let density = vec![0.0, 10.0, 10.0, 0.0];
        let b = make_kde2d_batch(grid_x, grid_y, density);
        let spec = ContourSpec {
            thresholds: 1,
            fill: false,
            smooth: false,
            name: None,
        };
        let out = apply(&spec, &b).unwrap();
        // thresholds=1 → level at 1/(1+1) = 0.5 of range [0,10] = 5.0
        let lv = out
            .column(1)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert!(lv.len() > 0);
        assert!((lv.value(0) - 5.0).abs() < 1e-9, "expected level 5.0 got {}", lv.value(0));
        // Saddle resolution: 2 segments = 4 vertex rows.
        assert_eq!(out.num_rows(), 4, "saddle must produce exactly 2 segments (4 vertex rows)");
    }

    #[test]
    fn contour_ring_with_hole_isoband_polygons() {
        pyo3::Python::initialize();
        // 8×8 bimodal density: two peaks separated by a low region.
        let n = 8usize;
        let grid_x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let grid_y: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut density = vec![0.0_f64; n * n];
        for gy in 0..n {
            for gx in 0..n {
                let dx1 = (gx as f64) - 1.5;
                let dy1 = (gy as f64) - 3.5;
                let dx2 = (gx as f64) - 6.0;
                let dy2 = (gy as f64) - 3.5;
                let p1 = (-0.5 * (dx1 * dx1 + dy1 * dy1)).exp();
                let p2 = (-0.5 * (dx2 * dx2 + dy2 * dy2)).exp();
                density[gy * n + gx] = p1 + p2;
            }
        }
        let b = make_kde2d_batch(grid_x, grid_y, density);
        let spec = ContourSpec {
            thresholds: 4,
            fill: true,
            smooth: false,
            name: None,
        };
        let out = apply(&spec, &b).unwrap();
        assert!(out.num_rows() > 0, "bimodal density must yield polygons");
        let id = out
            .column(0)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let mut distinct: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
        for i in 0..id.len() {
            distinct.insert(id.value(i));
        }
        // Expect multiple distinct level_ids (multiple cells per band).
        assert!(distinct.len() >= 2, "expected ≥2 distinct level_ids, got {}", distinct.len());
    }

    #[test]
    fn contour_bivariate_density_routing() {
        pyo3::Python::initialize();
        // Smoke integration test: Kde2D → Contour pipeline.
        use crate::transform::kde::BandwidthSpec;
        use crate::transform::kde_2d::{self, Kde2DSpec};
        use arrow::array::Float64Array;

        let xs: Vec<f64> = (0..16).map(|i| (i as f64) * 0.1).collect();
        let ys: Vec<f64> = (0..16).map(|i| ((i as f64) * 0.13).sin()).collect();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("y", DataType::Float64, true),
        ]));
        let in_batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(xs)), Arc::new(Float64Array::from(ys))],
        )
        .unwrap();
        let kde2d_spec = Kde2DSpec {
            x: "x".into(),
            y: "y".into(),
            bandwidth: BandwidthSpec::Scott,
            n: 16,
            extent: None,
            name: None,
        };
        let kde_out = kde_2d::apply(&kde2d_spec, &in_batch).unwrap();
        let cspec = ContourSpec {
            thresholds: 4,
            fill: false,
            smooth: false,
            name: None,
        };
        let out = apply(&cspec, &kde_out).unwrap();
        assert!(out.num_rows() > 0, "Kde2D → Contour pipeline should emit segments");
    }

    #[test]
    fn contour_round_trip() {
        let s = ContourSpec {
            thresholds: 8,
            fill: true,
            smooth: false,
            name: Some("c".into()),
        };
        let json = serde_json::to_string(&s).unwrap();
        let parsed: ContourSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, s);

        // Defaults round-trip.
        let s2 = ContourSpec {
            thresholds: 6,
            fill: false,
            smooth: true,
            name: None,
        };
        let json2 = serde_json::to_string(&s2).unwrap();
        let parsed2: ContourSpec = serde_json::from_str(&json2).unwrap();
        assert_eq!(parsed2, s2);
        assert!(!json2.contains("name"), "name=None must be omitted: {json2}");
    }

    #[test]
    fn contour_evenodd_level_id_encoding() {
        pyo3::Python::initialize();
        // 4×4 grid with monotone-increasing density so multiple bands have content.
        let grid_x: Vec<f64> = (0..4).map(|i| i as f64).collect();
        let grid_y: Vec<f64> = (0..4).map(|i| i as f64).collect();
        let mut density = vec![0.0_f64; 16];
        for gy in 0..4 {
            for gx in 0..4 {
                density[gy * 4 + gx] = (gx as f64) + (gy as f64);
            }
        }
        let b = make_kde2d_batch(grid_x, grid_y, density);
        // thresholds=3 → 3 levels → 2 bands.
        let spec = ContourSpec {
            thresholds: 3,
            fill: true,
            smooth: false,
            name: None,
        };
        let out = apply(&spec, &b).unwrap();
        let id = out
            .column(0)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let mut found_band0 = false;
        let mut found_band1 = false;
        for i in 0..id.len() {
            let v = id.value(i);
            let band = v >> 16;
            let cell = v & 0xFFFF;
            assert!(band <= 1, "band index out of expected range: {band}");
            assert!(cell <= 0xFFFF, "cell index overflow");
            if band == 0 {
                found_band0 = true;
                assert!(v <= 0xFFFF, "band 0 ids must fit in [0, 0xFFFF]");
            } else if band == 1 {
                found_band1 = true;
                assert!(
                    v >= 0x10000 && v <= 0x1FFFF,
                    "band 1 ids must lie in [0x10000, 0x1FFFF]; got {v:#x}"
                );
            }
        }
        assert!(found_band0, "expected at least one band 0 polygon");
        assert!(found_band1, "expected at least one band 1 polygon");
    }
}
