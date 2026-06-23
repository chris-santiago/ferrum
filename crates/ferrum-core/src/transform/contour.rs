//! Contour: Marching Squares isolines and isobands over a Kde2D-shaped grid.
//!
//! Reads a batch with the Kde2D output schema (one or more rows):
//!   grid_x:  List<Float64>   length nx
//!   grid_y:  List<Float64>   length ny
//!   density: List<Float64>   length nx*ny, row-major: density[gy * nx + gx]
//!   nx:      UInt32
//!   ny:      UInt32
//!   extent:  List<Float64>   length 4: [xmin, xmax, ymin, ymax]
//!   <group>: Utf8 (OPTIONAL trailing column — present only in grouped Kde2D output)
//!
//! When a trailing Utf8 column is present beyond the 6 known fields, the batch
//! is treated as grouped Kde2D output (one row per group). The contour
//! extraction runs once per row and the group key is propagated to every output
//! vertex so downstream marks can `color=<group>`.
//!
//! Output (one row per polyline vertex, NOT per polyline):
//!   level_id:    UInt32
//!   level_value: Float64
//!   contour_x:   Float64
//!   contour_y:   Float64
//!   [<group>:    Utf8]   — only when input is grouped
//!
//! `level_id` encoding:
//!   - Isoline mode: (level_index << 16) | segment_index
//!     `segment_index` increments across all cells in row-major iteration.
//!   - Isoband mode: (band_index   << 16) | cell_index
//!     `cell_index` increments per polygon emitted within that band.
//!
//! Simplifications (vs. the spec):
//!   - Per-segment output (not stitched polylines). Each segment = 2 vertex rows.
//!   - Per-cell polygons (not ring assembly) for isoband mode (Sutherland-Hodgman clipping).
//!   - `smooth=true` applies a 3×3 Gaussian kernel (center=4, edges=2, corners=1, /16)
//!     to interior grid cells; border cells are copied unchanged.

use arrow::array::{Array, ArrayRef, Float64Array, ListArray, RecordBatch, StringArray, UInt32Array};
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
#[serde(deny_unknown_fields)]
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

/// Output schema for isoband (fill=true): 4 columns, one row per polygon vertex.
fn out_schema_isoband() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("level_id", DataType::UInt32, false),
        Field::new("level_value", DataType::Float64, false),
        Field::new("contour_x", DataType::Float64, false),
        Field::new("contour_y", DataType::Float64, false),
    ]))
}

/// Output schema for isoline (fill=false): 6 columns, one row per segment.
fn out_schema_isoline() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("level_id", DataType::UInt32, false),
        Field::new("level_value", DataType::Float64, false),
        Field::new("contour_x", DataType::Float64, false),
        Field::new("contour_y", DataType::Float64, false),
        Field::new("contour_x2", DataType::Float64, false),
        Field::new("contour_y2", DataType::Float64, false),
    ]))
}

fn empty_output(fill: bool) -> PyResult<RecordBatch> {
    if fill {
        let schema = out_schema_isoband();
        let cols: Vec<ArrayRef> = vec![
            Arc::new(UInt32Array::from(Vec::<u32>::new())),
            Arc::new(Float64Array::from(Vec::<f64>::new())),
            Arc::new(Float64Array::from(Vec::<f64>::new())),
            Arc::new(Float64Array::from(Vec::<f64>::new())),
        ];
        RecordBatch::try_new(schema, cols)
            .map_err(|e| PyValueError::new_err(format!("stat_contour: {e}")))
    } else {
        let schema = out_schema_isoline();
        let cols: Vec<ArrayRef> = vec![
            Arc::new(UInt32Array::from(Vec::<u32>::new())),
            Arc::new(Float64Array::from(Vec::<f64>::new())),
            Arc::new(Float64Array::from(Vec::<f64>::new())),
            Arc::new(Float64Array::from(Vec::<f64>::new())),
            Arc::new(Float64Array::from(Vec::<f64>::new())),
            Arc::new(Float64Array::from(Vec::<f64>::new())),
        ];
        RecordBatch::try_new(schema, cols)
            .map_err(|e| PyValueError::new_err(format!("stat_contour: {e}")))
    }
}

fn extract_list_f64_row(batch: &RecordBatch, name: &str, row: usize) -> PyResult<Vec<f64>> {
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
    let inner = la.value(row);
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

fn extract_u32_row(batch: &RecordBatch, name: &str, row: usize) -> PyResult<u32> {
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
    Ok(arr.value(row))
}

/// Output accumulator for a single surface's contour extraction.
struct SurfaceOutput {
    level_ids: Vec<u32>,
    level_vals: Vec<f64>,
    xs: Vec<f64>,
    ys: Vec<f64>,
    /// Present only in isoline mode.
    x2s: Option<Vec<f64>>,
    y2s: Option<Vec<f64>>,
}

/// Extract contour lines or isobands from a single 2-D density surface.
/// The caller supplies pre-validated grid/density data; no RecordBatch access here.
fn contour_one_surface(
    spec: &ContourSpec,
    grid_x: &[f64],
    grid_y: &[f64],
    density: &[f64],
    nx: usize,
    ny: usize,
) -> Option<SurfaceOutput> {
    // Compute thresholds: t evenly-spaced interior levels.
    let dmin = density.iter().copied().fold(f64::INFINITY, f64::min);
    let dmax = density.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !dmin.is_finite() || !dmax.is_finite() || dmax <= dmin {
        return None;
    }
    let t = spec.thresholds;
    let levels: Vec<f64> = (1..=t)
        .map(|i| dmin + (dmax - dmin) * (i as f64) / ((t + 1) as f64))
        .collect();

    // R1: Gaussian smoothing. Apply a 3×3 kernel (center=4, edges=2, corners=1, /16)
    // to all interior cells. Border cells are copied unchanged.
    let density: Vec<f64> = if spec.smooth && nx >= 3 && ny >= 3 {
        let mut smoothed = density.to_vec();
        for gy in 1..(ny - 1) {
            for gx in 1..(nx - 1) {
                let tl = density[(gy - 1) * nx + (gx - 1)];
                let tm = density[(gy - 1) * nx + gx];
                let tr = density[(gy - 1) * nx + (gx + 1)];
                let ml = density[gy * nx + (gx - 1)];
                let mc = density[gy * nx + gx];
                let mr = density[gy * nx + (gx + 1)];
                let bl = density[(gy + 1) * nx + (gx - 1)];
                let bm = density[(gy + 1) * nx + gx];
                let br = density[(gy + 1) * nx + (gx + 1)];
                smoothed[gy * nx + gx] =
                    (tl + 2.0 * tm + tr + 2.0 * ml + 4.0 * mc + 2.0 * mr + bl + 2.0 * bm + br)
                        / 16.0;
            }
        }
        smoothed
    } else {
        density.to_vec()
    };

    let mut level_ids: Vec<u32> = Vec::new();
    let mut level_vals: Vec<f64> = Vec::new();
    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();

    if !spec.fill {
        // ---------- Isoline mode (Marching Squares) ----------
        let mut x2s: Vec<f64> = Vec::new();
        let mut y2s: Vec<f64> = Vec::new();

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
                        x2s.push(p1.0);
                        y2s.push(p1.1);
                    }
                }
            }
        }

        Some(SurfaceOutput {
            level_ids,
            level_vals,
            xs,
            ys,
            x2s: Some(x2s),
            y2s: Some(y2s),
        })
    } else {
        // ---------- Isoband mode (Sutherland-Hodgman polygon clipping) ----------
        if levels.len() < 2 {
            return None;
        }

        #[derive(Clone, Copy)]
        struct Vtx {
            x: f64,
            y: f64,
            d: f64,
        }

        fn sh_clip(
            poly: &[Vtx],
            inside: impl Fn(&Vtx) -> bool,
            t_at: impl Fn(&Vtx, &Vtx) -> f64,
        ) -> Vec<Vtx> {
            if poly.is_empty() {
                return Vec::new();
            }
            let mut out = Vec::with_capacity(poly.len() + 1);
            let n = poly.len();
            for i in 0..n {
                let cur = poly[i];
                let nxt = poly[(i + 1) % n];
                let cur_in = inside(&cur);
                let nxt_in = inside(&nxt);
                if cur_in {
                    out.push(cur);
                }
                if cur_in != nxt_in {
                    let t = t_at(&cur, &nxt);
                    let ix = cur.x + t * (nxt.x - cur.x);
                    let iy = cur.y + t * (nxt.y - cur.y);
                    let id = cur.d + t * (nxt.d - cur.d);
                    out.push(Vtx { x: ix, y: iy, d: id });
                }
            }
            out
        }

        for bi in 0..(levels.len() - 1) {
            let lo = levels[bi];
            let hi = levels[bi + 1];
            let mut cell_idx: u32 = 0;
            for gy in 0..(ny - 1) {
                for gx in 0..(nx - 1) {
                    let x0 = grid_x[gx];
                    let x1 = grid_x[gx + 1];
                    let y0 = grid_y[gy];
                    let y1 = grid_y[gy + 1];

                    let d_tl = density[gy * nx + gx];
                    let d_tr = density[gy * nx + (gx + 1)];
                    let d_br = density[(gy + 1) * nx + (gx + 1)];
                    let d_bl = density[(gy + 1) * nx + gx];

                    // CW winding: TL → TR → BR → BL.
                    let quad = [
                        Vtx { x: x0, y: y0, d: d_tl },
                        Vtx { x: x1, y: y0, d: d_tr },
                        Vtx { x: x1, y: y1, d: d_br },
                        Vtx { x: x0, y: y1, d: d_bl },
                    ];

                    let clipped_lo = sh_clip(
                        &quad,
                        |v| v.d >= lo,
                        |a, b| {
                            let denom = b.d - a.d;
                            if denom.abs() < f64::EPSILON { 0.0 } else { (lo - a.d) / denom }
                        },
                    );
                    if clipped_lo.len() < 3 {
                        continue;
                    }

                    let clipped = sh_clip(
                        &clipped_lo,
                        |v| v.d <= hi,
                        |a, b| {
                            let denom = b.d - a.d;
                            if denom.abs() < f64::EPSILON { 0.0 } else { (hi - a.d) / denom }
                        },
                    );
                    if clipped.len() < 3 {
                        continue;
                    }

                    let id = ((bi as u32) << 16) | cell_idx;
                    cell_idx = cell_idx.wrapping_add(1);
                    let band_value = lo;
                    for v in &clipped {
                        level_ids.push(id);
                        level_vals.push(band_value);
                        xs.push(v.x);
                        ys.push(v.y);
                    }
                    // Close the polygon.
                    level_ids.push(id);
                    level_vals.push(band_value);
                    xs.push(clipped[0].x);
                    ys.push(clipped[0].y);
                }
            }
        }

        Some(SurfaceOutput {
            level_ids,
            level_vals,
            xs,
            ys,
            x2s: None,
            y2s: None,
        })
    }
}

/// Detect the optional group column: the trailing field beyond the 6 known
/// Kde2D surface columns, if it has Utf8 type. Returns `(field_name, column_index)`
/// or `None` when no group column is present.
fn detect_group_col(batch: &RecordBatch) -> Option<(String, usize)> {
    const SURFACE_COLS: usize = 6;
    let schema = batch.schema();
    if schema.fields().len() > SURFACE_COLS {
        let idx = SURFACE_COLS; // one trailing column by convention
        let field = schema.field(idx);
        if field.data_type() == &DataType::Utf8 {
            return Some((field.name().to_string(), idx));
        }
    }
    None
}

pub(crate) fn apply(spec: &ContourSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    if spec.thresholds == 0 {
        return Err(PyValueError::new_err(
            "stat_contour: thresholds must be > 0",
        ));
    }

    // Validate required surface columns are present.
    for name in ["grid_x", "grid_y", "density", "nx", "ny", "extent"] {
        if batch.schema().index_of(name).is_err() {
            return Err(PyValueError::new_err(format!(
                "stat_contour: input must be a Kde2D output (missing column '{name}')"
            )));
        }
    }

    if batch.num_rows() == 0 {
        return empty_output(spec.fill);
    }

    // Detect the optional group column (trailing Utf8 field beyond the 6 surface cols).
    let group_col = detect_group_col(batch);

    // Accumulate output across all rows (one row = one surface, possibly one group).
    let mut all_level_ids: Vec<u32> = Vec::new();
    let mut all_level_vals: Vec<f64> = Vec::new();
    let mut all_xs: Vec<f64> = Vec::new();
    let mut all_ys: Vec<f64> = Vec::new();
    let mut all_x2s: Vec<f64> = Vec::new(); // isoline only
    let mut all_y2s: Vec<f64> = Vec::new(); // isoline only
    let mut all_group_tags: Vec<String> = Vec::new(); // grouped path only

    // FA-8: level_ids restart at 0 for every surface (per-surface seg/cell
    // counter), so multiple grouped surfaces would emit colliding ids. Offset
    // each surface's ids past the previous surface's maximum so the accumulated
    // level_id column is globally unique. For a single surface the base stays 0,
    // keeping existing (ungrouped) output byte-identical.
    let mut level_id_base: u32 = 0;

    let group_arr: Option<&StringArray> = group_col.as_ref().map(|(_, col_idx)| {
        batch
            .column(*col_idx)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("group column confirmed Utf8 by detect_group_col")
    });

    for row in 0..batch.num_rows() {
        let grid_x = extract_list_f64_row(batch, "grid_x", row)?;
        let grid_y = extract_list_f64_row(batch, "grid_y", row)?;
        let density = extract_list_f64_row(batch, "density", row)?;
        let nx = extract_u32_row(batch, "nx", row)? as usize;
        let ny = extract_u32_row(batch, "ny", row)? as usize;

        if grid_x.len() != nx || grid_y.len() != ny || density.len() != nx * ny {
            return Err(PyValueError::new_err(format!(
                "stat_contour: row {row}: dimension mismatch \
                 (nx={nx}, ny={ny}, grid_x.len={}, grid_y.len={}, density.len={})",
                grid_x.len(),
                grid_y.len(),
                density.len()
            )));
        }

        if nx < 2 || ny < 2 {
            // Surface too small to contour; skip this row (not an error).
            continue;
        }

        let surface = contour_one_surface(spec, &grid_x, &grid_y, &density, nx, ny);
        let surface = match surface {
            None => continue, // degenerate surface (flat density etc.)
            Some(s) => s,
        };

        let n_new = surface.level_ids.len();
        // FA-8: shift this surface's ids past all previously-emitted ids so they
        // stay globally unique across grouped surfaces.
        let surface_max = surface.level_ids.iter().copied().max();
        all_level_ids.extend(
            surface
                .level_ids
                .iter()
                .map(|id| id.wrapping_add(level_id_base)),
        );
        if let Some(m) = surface_max {
            level_id_base = level_id_base.wrapping_add(m).wrapping_add(1);
        }
        all_level_vals.extend(surface.level_vals);
        all_xs.extend(surface.xs);
        all_ys.extend(surface.ys);
        if let (Some(x2s), Some(y2s)) = (surface.x2s, surface.y2s) {
            all_x2s.extend(x2s);
            all_y2s.extend(y2s);
        }

        // Tag every output vertex with the group key when grouped.
        if let Some(garr) = group_arr {
            let key = garr.value(row).to_string();
            all_group_tags.extend(std::iter::repeat_n(key, n_new));
        }
    }

    build_output(
        spec.fill,
        &group_col,
        AccumulatedOutput {
            level_ids: all_level_ids,
            level_vals: all_level_vals,
            xs: all_xs,
            ys: all_ys,
            x2s: all_x2s,
            y2s: all_y2s,
            group_tags: all_group_tags,
        },
    )
}

/// Accumulated per-vertex output across all processed surfaces.
struct AccumulatedOutput {
    level_ids: Vec<u32>,
    level_vals: Vec<f64>,
    xs: Vec<f64>,
    ys: Vec<f64>,
    /// Non-empty only in isoline mode.
    x2s: Vec<f64>,
    /// Non-empty only in isoline mode.
    y2s: Vec<f64>,
    /// Non-empty only in grouped mode.
    group_tags: Vec<String>,
}

/// Assemble the final RecordBatch from accumulated output vectors.
fn build_output(
    fill: bool,
    group_col: &Option<(String, usize)>,
    out: AccumulatedOutput,
) -> PyResult<RecordBatch> {
    let AccumulatedOutput { level_ids, level_vals, xs, ys, x2s, y2s, group_tags } = out;
    let has_group = group_col.is_some();

    if fill {
        let mut schema_fields = out_schema_isoband().fields().to_vec();
        if let Some((name, _)) = group_col {
            schema_fields.push(Arc::new(Field::new(name.as_str(), DataType::Utf8, false)));
        }
        let schema = Arc::new(Schema::new(schema_fields));
        let mut cols: Vec<ArrayRef> = vec![
            Arc::new(UInt32Array::from(level_ids)),
            Arc::new(Float64Array::from(level_vals)),
            Arc::new(Float64Array::from(xs)),
            Arc::new(Float64Array::from(ys)),
        ];
        if has_group {
            cols.push(Arc::new(StringArray::from(
                group_tags.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )));
        }
        RecordBatch::try_new(schema, cols)
            .map_err(|e| PyValueError::new_err(format!("stat_contour: {e}")))
    } else {
        let mut schema_fields = out_schema_isoline().fields().to_vec();
        if let Some((name, _)) = group_col {
            schema_fields.push(Arc::new(Field::new(name.as_str(), DataType::Utf8, false)));
        }
        let schema = Arc::new(Schema::new(schema_fields));
        let mut cols: Vec<ArrayRef> = vec![
            Arc::new(UInt32Array::from(level_ids)),
            Arc::new(Float64Array::from(level_vals)),
            Arc::new(Float64Array::from(xs)),
            Arc::new(Float64Array::from(ys)),
            Arc::new(Float64Array::from(x2s)),
            Arc::new(Float64Array::from(y2s)),
        ];
        if has_group {
            cols.push(Arc::new(StringArray::from(
                group_tags.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )));
        }
        RecordBatch::try_new(schema, cols)
            .map_err(|e| PyValueError::new_err(format!("stat_contour: {e}")))
    }
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
///     When True, applies a 3x3 Gaussian kernel pass (center=4, edges=2,
///     corners=1, /16) to interior density cells before contouring.
///     Reduces jagged contour edges at the cost of slight blurring.
/// name : str, optional
///     Named output label for sibling ``Reorder(from_=...)`` lookup.
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> fm.Chart(df).mark_contour().encode(x="x", y="y")
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
    use arrow::array::{Float64Array, Float64Builder, ListBuilder, StringArray, UInt32Array};
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
        // Isoline output: 6 columns — one row per segment, (x,y) start + (x2,y2) end.
        assert_eq!(s.fields().len(), 6);
        assert_eq!(s.field(0).name(), "level_id");
        assert_eq!(s.field(0).data_type(), &DataType::UInt32);
        assert_eq!(s.field(1).name(), "level_value");
        assert_eq!(s.field(1).data_type(), &DataType::Float64);
        assert_eq!(s.field(2).name(), "contour_x");
        assert_eq!(s.field(2).data_type(), &DataType::Float64);
        assert_eq!(s.field(3).name(), "contour_y");
        assert_eq!(s.field(3).data_type(), &DataType::Float64);
        assert_eq!(s.field(4).name(), "contour_x2");
        assert_eq!(s.field(4).data_type(), &DataType::Float64);
        assert_eq!(s.field(5).name(), "contour_y2");
        assert_eq!(s.field(5).data_type(), &DataType::Float64);
        assert!(out.num_rows() > 0, "expected at least some segments emitted");
        // Each row is exactly one segment (start + end); no pairing constraint.
        // All level_ids must be unique (each segment gets a distinct id).
        let id_col = out.column(0).as_any().downcast_ref::<UInt32Array>().unwrap();
        let mut seen = std::collections::BTreeSet::new();
        for i in 0..id_col.len() {
            assert!(seen.insert(id_col.value(i)), "duplicate level_id in isoline output");
        }
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
        // S-H clipped polygons have variable vertex counts (3-8 corners + 1 closing vertex).
        // Verify that every polygon is properly closed: first vertex == last vertex per level_id.
        {
            let id_col = out.column(0).as_any().downcast_ref::<UInt32Array>().unwrap();
            let cx_col = out.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
            let cy_col = out.column(3).as_any().downcast_ref::<Float64Array>().unwrap();
            let n = out.num_rows();
            let mut i = 0usize;
            while i < n {
                let cur_id = id_col.value(i);
                let start = i;
                while i < n && id_col.value(i) == cur_id {
                    i += 1;
                }
                let end = i;
                let len = end - start;
                assert!(len >= 4, "polygon must have at least 3 vertices + closing (got {len})");
                // First vertex == last vertex (closed polygon).
                let fx = cx_col.value(start);
                let fy = cy_col.value(start);
                let lx = cx_col.value(end - 1);
                let ly = cy_col.value(end - 1);
                assert!(
                    (fx - lx).abs() < 1e-12 && (fy - ly).abs() < 1e-12,
                    "polygon {cur_id:#x}: first ({fx},{fy}) != last ({lx},{ly})"
                );
            }
        }
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
        // Saddle resolution: 2 segments = 2 rows (one row per segment in the new schema).
        assert_eq!(out.num_rows(), 2, "saddle must produce exactly 2 segments (2 rows, one per segment)");
        // Verify each row has 6 columns (isoline schema).
        assert_eq!(out.schema().fields().len(), 6, "isoline output must have 6 columns");
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
            groupby: vec![],
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

    /// Build a 2-row grouped Kde2D-shaped batch (mimics kde_2d grouped output).
    /// Each row is one group surface; a trailing Utf8 column holds the group key.
    fn make_grouped_kde2d_batch(
        rows: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)>, // (grid_x, grid_y, density) per group
        group_col: &str,
        group_vals: Vec<&str>,
    ) -> RecordBatch {
        assert_eq!(rows.len(), group_vals.len());
        let list_f64 = DataType::List(Arc::new(Field::new("item", DataType::Float64, true)));
        let schema = Arc::new(Schema::new(vec![
            Field::new("grid_x", list_f64.clone(), false),
            Field::new("grid_y", list_f64.clone(), false),
            Field::new("density", list_f64.clone(), false),
            Field::new("nx", DataType::UInt32, false),
            Field::new("ny", DataType::UInt32, false),
            Field::new("extent", list_f64, false),
            Field::new(group_col, DataType::Utf8, false),
        ]));

        let mut gxb = ListBuilder::new(Float64Builder::new());
        let mut gyb = ListBuilder::new(Float64Builder::new());
        let mut db = ListBuilder::new(Float64Builder::new());
        let mut eb = ListBuilder::new(Float64Builder::new());
        let mut nxv: Vec<u32> = Vec::new();
        let mut nyv: Vec<u32> = Vec::new();

        for (grid_x, grid_y, density) in &rows {
            let nx = grid_x.len() as u32;
            let ny = grid_y.len() as u32;
            assert_eq!(density.len(), (nx * ny) as usize);
            gxb.values().append_slice(grid_x);
            gxb.append(true);
            gyb.values().append_slice(grid_y);
            gyb.append(true);
            db.values().append_slice(density);
            db.append(true);
            eb.values().append_slice(&[
                grid_x[0],
                *grid_x.last().unwrap(),
                grid_y[0],
                *grid_y.last().unwrap(),
            ]);
            eb.append(true);
            nxv.push(nx);
            nyv.push(ny);
        }

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(gxb.finish()),
                Arc::new(gyb.finish()),
                Arc::new(db.finish()),
                Arc::new(UInt32Array::from(nxv)),
                Arc::new(UInt32Array::from(nyv)),
                Arc::new(eb.finish()),
                Arc::new(StringArray::from(group_vals)),
            ],
        )
        .unwrap()
    }

    // --- RED tests (will fail until grouped path is implemented) ---

    #[test]
    fn contour_grouped_two_groups_both_tagged() {
        // Two-group Kde2D batch → contour must emit rows for BOTH groups,
        // each tagged with the correct group key; output schema gains the
        // group column.
        pyo3::Python::initialize();
        let n = 4usize;
        let grid_x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let grid_y: Vec<f64> = (0..n).map(|i| i as f64).collect();

        // Build a simple peaked density for both groups.
        let make_density = |cx: f64, cy: f64| -> Vec<f64> {
            let mut d = vec![0.0f64; n * n];
            for gy in 0..n {
                for gx in 0..n {
                    let dx = (gx as f64) - cx;
                    let dy = (gy as f64) - cy;
                    d[gy * n + gx] = (-(dx * dx + dy * dy)).exp();
                }
            }
            d
        };

        let rows = vec![
            (grid_x.clone(), grid_y.clone(), make_density(1.0, 1.0)),
            (grid_x.clone(), grid_y.clone(), make_density(2.0, 2.0)),
        ];
        let b = make_grouped_kde2d_batch(rows, "species", vec!["A", "B"]);

        let spec = ContourSpec {
            thresholds: 2,
            fill: false,
            smooth: false,
            name: None,
        };
        let out = apply(&spec, &b).unwrap();

        // Output must have isoline columns PLUS the group column.
        let s = out.schema();
        assert_eq!(s.fields().len(), 7, "grouped isoline output must have 7 columns (6 + group)");
        assert_eq!(s.field(6).name(), "species");
        assert_eq!(s.field(6).data_type(), &DataType::Utf8);

        // Rows from both groups must be present.
        assert!(out.num_rows() > 0, "grouped contour must emit rows");
        let gcol = out.column(6).as_any().downcast_ref::<StringArray>().unwrap();
        let has_a = (0..gcol.len()).any(|i| gcol.value(i) == "A");
        let has_b = (0..gcol.len()).any(|i| gcol.value(i) == "B");
        assert!(has_a, "output must contain rows tagged 'A'");
        assert!(has_b, "output must contain rows tagged 'B'");
    }

    #[test]
    fn contour_grouped_isoband_two_groups_both_tagged() {
        // Same as above but fill=true (isoband mode).
        pyo3::Python::initialize();
        let n = 4usize;
        let grid_x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let grid_y: Vec<f64> = (0..n).map(|i| i as f64).collect();

        let make_density = |cx: f64, cy: f64| -> Vec<f64> {
            let mut d = vec![0.0f64; n * n];
            for gy in 0..n {
                for gx in 0..n {
                    let dx = (gx as f64) - cx;
                    let dy = (gy as f64) - cy;
                    d[gy * n + gx] = (-(dx * dx + dy * dy)).exp();
                }
            }
            d
        };

        let rows = vec![
            (grid_x.clone(), grid_y.clone(), make_density(1.0, 1.0)),
            (grid_x.clone(), grid_y.clone(), make_density(2.0, 2.0)),
        ];
        let b = make_grouped_kde2d_batch(rows, "hue", vec!["X", "Y"]);

        let spec = ContourSpec {
            thresholds: 3,
            fill: true,
            smooth: false,
            name: None,
        };
        let out = apply(&spec, &b).unwrap();

        let s = out.schema();
        assert_eq!(s.fields().len(), 5, "grouped isoband output must have 5 columns (4 + group)");
        assert_eq!(s.field(4).name(), "hue");
        assert_eq!(s.field(4).data_type(), &DataType::Utf8);

        assert!(out.num_rows() > 0, "grouped isoband contour must emit rows");
        let gcol = out.column(4).as_any().downcast_ref::<StringArray>().unwrap();
        let has_x = (0..gcol.len()).any(|i| gcol.value(i) == "X");
        let has_y = (0..gcol.len()).any(|i| gcol.value(i) == "Y");
        assert!(has_x, "output must contain rows tagged 'X'");
        assert!(has_y, "output must contain rows tagged 'Y'");
    }

    #[test]
    fn contour_single_row_no_group_col_byte_stable() {
        // Single-row (no group column) path must still work; no schema change.
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
        let spec = ContourSpec { thresholds: 2, fill: false, smooth: false, name: None };
        let out = apply(&spec, &b).unwrap();
        // Must be exactly 6 columns, no group column appended.
        assert_eq!(out.schema().fields().len(), 6, "ungrouped isoline must have 6 columns");
        // Must have rows.
        assert!(out.num_rows() > 0, "ungrouped path must still emit segments");
    }

    #[test]
    fn contour_smooth_changes_output() {
        pyo3::Python::initialize();
        // 6×6 grid with a narrow Gaussian peak at (3,3). Smoothing should spread
        // density to neighbours, changing the grid values and thus contour geometry.
        let n = 6usize;
        let grid_x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let grid_y: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut density = vec![0.0_f64; n * n];
        for gy in 0..n {
            for gx in 0..n {
                let dx = (gx as f64) - 3.0;
                let dy = (gy as f64) - 3.0;
                // Very narrow Gaussian: falls off to near 0 within 1 cell of center.
                density[gy * n + gx] = (-3.0 * (dx * dx + dy * dy)).exp();
            }
        }
        let b = make_kde2d_batch(grid_x, grid_y, density);

        let spec_raw = ContourSpec { thresholds: 3, fill: false, smooth: false, name: None };
        let spec_smooth = ContourSpec { thresholds: 3, fill: false, smooth: true, name: None };

        let out_raw = apply(&spec_raw, &b).unwrap();
        let out_smooth = apply(&spec_smooth, &b).unwrap();

        // Both runs must produce segments — the Gaussian has enough range.
        assert!(out_raw.num_rows() > 0, "raw contour must produce segments");
        assert!(out_smooth.num_rows() > 0, "smoothed contour must produce segments");

        // The two outputs must differ because smoothing redistributes density.
        let xs_raw = out_raw.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
        let xs_smooth = out_smooth.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
        let min_rows = xs_raw.len().min(xs_smooth.len());
        let any_diff = (0..min_rows).any(|i| (xs_raw.value(i) - xs_smooth.value(i)).abs() > 1e-12)
            || xs_raw.len() != xs_smooth.len();
        assert!(any_diff, "smooth=true must change contour geometry vs smooth=false");
    }
}
