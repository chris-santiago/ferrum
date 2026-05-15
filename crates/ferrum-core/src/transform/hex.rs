//! Hex: hexagonal binning for 2D scatter data (pointy-top axial coords).
//!
//! Output schema (one row per hexagon vertex; 6 vertex rows per non-empty hex):
//!   hex_x:  Float64  — vertex x in data space
//!   hex_y:  Float64  — vertex y in data space
//!   hex_id: Int64    — packed (q, r) cell id; same value for all 6 vertex rows of a hex
//!   value:  Float64  — aggregated value for the hex; same for all 6 vertices
//!
//! Pointy-top axial coordinates (per Red Blob Games):
//!   pixel-to-fractional-axial:
//!     q_frac = (sqrt(3)/3 * (x - x_min) - (y - y_min)/3) / bin_size
//!     r_frac = ((y - y_min) * 2/3) / bin_size
//!   axial-to-pixel:
//!     px = bin_size * (sqrt(3) * q + sqrt(3)/2 * r)
//!     py = bin_size * (3/2 * r)
//!   vertex_i (i in 0..6) angle = (60° * i) + 30°  (pointy-top start)
//!     vx = R * cos(angle), vy = R * sin(angle); R = bin_size.
//!
//! Hexes are aggregated in a BTreeMap<(i64, i64), Aggregator> for deterministic
//! emit order.

use arrow::array::{Array, ArrayRef, Float64Array, Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

fn default_aggregate() -> String {
    "count".into()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct HexSpec {
    pub x: String,
    pub y: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub bin_size: Option<f64>,
    #[serde(default = "default_aggregate")]
    pub aggregate: String, // "count" | "mean" | "sum"
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub field: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

struct Aggregator {
    count: u64,
    sum: f64,
    sum_sq: f64,
    min: f64,
    max: f64,
    /// Populated only for aggregates requiring order statistics (median).
    values: Option<Vec<f64>>,
}

impl Default for Aggregator {
    fn default() -> Self {
        Self { count: 0, sum: 0.0, sum_sq: 0.0, min: f64::INFINITY, max: f64::NEG_INFINITY, values: None }
    }
}

/// Cube-round fractional axial (q_frac, r_frac) to integer (q, r).
/// Algorithm from Red Blob Games:
///   x_cube = q_frac, z_cube = r_frac, y_cube = -x_cube - z_cube
///   round each independently, then reset the axis with the largest delta.
fn cube_round(q_frac: f64, r_frac: f64) -> (i64, i64) {
    let x_cube = q_frac;
    let z_cube = r_frac;
    let y_cube = -x_cube - z_cube;

    let mut rx = x_cube.round();
    let mut ry = y_cube.round();
    let mut rz = z_cube.round();

    let dx = (rx - x_cube).abs();
    let dy = (ry - y_cube).abs();
    let dz = (rz - z_cube).abs();

    if dx > dy && dx > dz {
        rx = -ry - rz;
    } else if dy > dz {
        ry = -rx - rz;
    } else {
        rz = -rx - ry;
    }
    (rx as i64, rz as i64)
}

pub(crate) fn apply(spec: &HexSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    // 1. Validate columns exist + Float64.
    let schema = batch.schema();
    let xi = schema
        .index_of(&spec.x)
        .map_err(|_| PyValueError::new_err(format!("stat_hex: column '{}' not found", spec.x)))?;
    if schema.field(xi).data_type() != &DataType::Float64 {
        return Err(PyValueError::new_err(format!(
            "stat_hex: column '{}' must be Float64",
            spec.x
        )));
    }
    let yi = schema
        .index_of(&spec.y)
        .map_err(|_| PyValueError::new_err(format!("stat_hex: column '{}' not found", spec.y)))?;
    if schema.field(yi).data_type() != &DataType::Float64 {
        return Err(PyValueError::new_err(format!(
            "stat_hex: column '{}' must be Float64",
            spec.y
        )));
    }

    // Validate aggregate.
    let agg = spec.aggregate.as_str();
    if !matches!(agg, "count" | "mean" | "sum" | "min" | "max" | "median" | "std" | "var") {
        return Err(PyValueError::new_err(format!(
            "stat_hex: unknown aggregate '{}'; expected one of count/mean/sum/min/max/median/std/var",
            agg
        )));
    }
    let needs_field = !matches!(agg, "count");
    if needs_field && spec.field.is_none() {
        return Err(PyValueError::new_err(format!(
            "stat_hex: aggregate='{}' requires `field`",
            agg
        )));
    }

    // Validate field column if present.
    let field_arr: Option<&Float64Array> = match &spec.field {
        None => None,
        Some(name) => {
            let fi = schema.index_of(name).map_err(|_| {
                PyValueError::new_err(format!("stat_hex: column '{}' not found", name))
            })?;
            if schema.field(fi).data_type() != &DataType::Float64 {
                return Err(PyValueError::new_err(format!(
                    "stat_hex: column '{}' must be Float64",
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

    // Filter (x, y) pairs: drop null/NaN. Capture field in lockstep when present.
    // For "mean"/"sum", drop rows where field is null/NaN as well.
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
                    if needs_field {
                        continue;
                    }
                    0.0
                } else {
                    let v = fa.value(i);
                    if v.is_nan() {
                        if needs_field {
                            continue;
                        }
                        0.0
                    } else {
                        v
                    }
                }
            }
        };
        xs.push(xv);
        ys.push(yv);
        fs.push(fv);
    }

    // Empty input → empty output with the right schema.
    if xs.is_empty() {
        return empty_output();
    }

    // Compute extent (for x_min, y_min anchor).
    let (mut xmin, mut xmax) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut ymin, mut ymax) = (f64::INFINITY, f64::NEG_INFINITY);
    for i in 0..xs.len() {
        if xs[i] < xmin {
            xmin = xs[i];
        }
        if xs[i] > xmax {
            xmax = xs[i];
        }
        if ys[i] < ymin {
            ymin = ys[i];
        }
        if ys[i] > ymax {
            ymax = ys[i];
        }
    }

    // Auto bin_size.
    let bin_size = match spec.bin_size {
        Some(bs) => bs,
        None => {
            if xmax == xmin {
                1.0
            } else {
                (xmax - xmin) / 30.0
            }
        }
    };
    if !(bin_size.is_finite() && bin_size > 0.0) {
        return Err(PyValueError::new_err(
            "stat_hex: bin_size must be a positive finite number",
        ));
    }

    // Aggregate in a BTreeMap for deterministic iteration.
    let mut hexes: BTreeMap<(i64, i64), Aggregator> = BTreeMap::new();
    let sqrt3 = 3.0_f64.sqrt();
    for i in 0..xs.len() {
        let px = xs[i] - xmin;
        let py = ys[i] - ymin;
        let q_frac = (sqrt3 / 3.0 * px - py / 3.0) / bin_size;
        let r_frac = (2.0 / 3.0 * py) / bin_size;
        let (q, r) = cube_round(q_frac, r_frac);
        let entry = hexes.entry((q, r)).or_default();
        entry.count += 1;
        if !matches!(agg, "count") {
            let v = fs[i];
            entry.sum += v;
            entry.sum_sq += v * v;
            if v < entry.min { entry.min = v; }
            if v > entry.max { entry.max = v; }
            if matches!(agg, "median") {
                entry.values.get_or_insert_with(Vec::new).push(v);
            }
        }
    }

    // Emit 6 vertices per hex.
    let mut hex_xs: Vec<f64> = Vec::with_capacity(hexes.len() * 6);
    let mut hex_ys: Vec<f64> = Vec::with_capacity(hexes.len() * 6);
    let mut hex_ids: Vec<i64> = Vec::with_capacity(hexes.len() * 6);
    let mut values: Vec<f64> = Vec::with_capacity(hexes.len() * 6);

    let r_radius = bin_size; // distance from center to vertex (pointy-top)
    for ((q, r), agg_state) in hexes.iter_mut() {
        // Center in data space.
        let cx = xmin + bin_size * (sqrt3 * (*q as f64) + (sqrt3 / 2.0) * (*r as f64));
        let cy = ymin + bin_size * (1.5 * (*r as f64));

        // hex_id = q * 65536 + r
        let hex_id = (*q) * 65536 + (*r);

        // Aggregated value.
        let n = agg_state.count as f64;
        let value = match agg {
            "count"  => n,
            "sum"    => agg_state.sum,
            "mean"   => if agg_state.count == 0 { f64::NAN } else { agg_state.sum / n },
            "min"    => if agg_state.count == 0 { f64::NAN } else { agg_state.min },
            "max"    => if agg_state.count == 0 { f64::NAN } else { agg_state.max },
            "median" => {
                if agg_state.count == 0 { f64::NAN } else {
                    let vs = agg_state.values.get_or_insert_default();
                    vs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let mid = vs.len() / 2;
                    if vs.len() % 2 == 0 { (vs[mid - 1] + vs[mid]) / 2.0 } else { vs[mid] }
                }
            }
            "std" | "var" => {
                if agg_state.count < 2 { f64::NAN } else {
                    let mean = agg_state.sum / n;
                    let variance = (agg_state.sum_sq - n * mean * mean) / (n - 1.0);
                    let variance = variance.max(0.0); // clamp floating-point noise
                    if agg == "std" { variance.sqrt() } else { variance }
                }
            }
            _ => unreachable!(),
        };

        // Pointy-top vertex angles: 30°, 90°, 150°, 210°, 270°, 330°.
        for i in 0..6 {
            let angle = std::f64::consts::PI * (60.0 * (i as f64) + 30.0) / 180.0;
            let vx = cx + r_radius * angle.cos();
            let vy = cy + r_radius * angle.sin();
            hex_xs.push(vx);
            hex_ys.push(vy);
            hex_ids.push(hex_id);
            values.push(value);
        }
    }

    let out_schema = Arc::new(Schema::new(vec![
        Field::new("hex_x", DataType::Float64, false),
        Field::new("hex_y", DataType::Float64, false),
        Field::new("hex_id", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(hex_xs)),
        Arc::new(Float64Array::from(hex_ys)),
        Arc::new(Int64Array::from(hex_ids)),
        Arc::new(Float64Array::from(values)),
    ];
    RecordBatch::try_new(out_schema, cols)
        .map_err(|e| PyValueError::new_err(format!("stat_hex: {e}")))
}

fn empty_output() -> PyResult<RecordBatch> {
    let out_schema = Arc::new(Schema::new(vec![
        Field::new("hex_x", DataType::Float64, false),
        Field::new("hex_y", DataType::Float64, false),
        Field::new("hex_id", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(Vec::<f64>::new())),
        Arc::new(Float64Array::from(Vec::<f64>::new())),
        Arc::new(Int64Array::from(Vec::<i64>::new())),
        Arc::new(Float64Array::from(Vec::<f64>::new())),
    ];
    RecordBatch::try_new(out_schema, cols)
        .map_err(|e| PyValueError::new_err(format!("stat_hex: {e}")))
}

// ---------- PyO3 wrapper ----------

use pyo3::prelude::*;

use crate::transform::core::TransformSpec;

/// Hexagonal binning over a pair of numeric columns.
///
/// Partitions the (x, y) plane into a hex grid and counts — or aggregates
/// — the observations that fall into each cell. Produces compact, visually
/// uniform density representations suitable for large scatter data.
///
/// Output columns: ``hex_x``, ``hex_y`` (Float64 cell centers), ``count``
/// (Int64 observation count), and ``value`` (Float64, only when
/// ``aggregate`` is ``"mean"`` or ``"sum"``).
///
/// Parameters
/// ----------
/// x : str
///     Column for the horizontal axis (must be numeric).
/// y : str
///     Column for the vertical axis (must be numeric).
/// bin_size : float, optional
///     Hex cell radius in data units. When omitted, a reasonable default
///     is chosen automatically from the data range.
/// aggregate : {"count", "mean", "sum"}, default "count"
///     Aggregation to apply per cell. ``"mean"`` and ``"sum"`` require
///     ``field``.
/// field : str, optional
///     Column to aggregate when ``aggregate`` is ``"mean"`` or ``"sum"``.
/// name : str, optional
///     Named output label for sibling ``Reorder(from_=...)`` lookup.
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> fm.Chart(df).mark_hexbin(bins=20).encode(x="x", y="y")
#[pyclass(eq, module = "ferrum._core", name = "Hex")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyHex(pub(crate) TransformSpec);

#[pymethods]
impl PyHex {
    #[new]
    #[pyo3(signature = (
        x, y, *,
        bin_size = None,
        aggregate = "count",
        field = None,
        name = None,
    ))]
    fn new(
        x: &str,
        y: &str,
        bin_size: Option<f64>,
        aggregate: &str,
        field: Option<String>,
        name: Option<String>,
    ) -> PyResult<Self> {
        if x.is_empty() || y.is_empty() {
            return Err(PyValueError::new_err(
                "Hex: x and y fields must be non-empty",
            ));
        }
        match aggregate {
            "count" => {}
            "mean" | "sum" | "min" | "max" | "median" | "std" | "var" => {
                if field.is_none() {
                    return Err(PyValueError::new_err(format!(
                        "Hex: aggregate='{aggregate}' requires field"
                    )));
                }
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "Hex: unknown aggregate '{other}'; expected count|mean|sum|min|max|median|std|var"
                )))
            }
        }
        if let Some(bs) = bin_size {
            if !bs.is_finite() || bs <= 0.0 {
                return Err(PyValueError::new_err(
                    "Hex: bin_size must be a positive finite number",
                ));
            }
        }
        Ok(PyHex(TransformSpec::Hex(HexSpec {
            x: x.to_string(),
            y: y.to_string(),
            bin_size,
            aggregate: aggregate.to_string(),
            field,
            name,
        })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Hex(s) => format!(
                "Hex(x='{}', y='{}', bin_size={:?}, aggregate='{}', field={:?})",
                s.x, s.y, s.bin_size, s.aggregate, s.field
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int64Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::collections::HashMap;
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
            Field::new("v", DataType::Float64, true),
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

    fn col_f64(b: &RecordBatch, name: &str) -> Vec<f64> {
        let i = b.schema().index_of(name).unwrap();
        let a = b.column(i).as_any().downcast_ref::<Float64Array>().unwrap();
        (0..a.len()).map(|j| a.value(j)).collect()
    }
    fn col_i64(b: &RecordBatch, name: &str) -> Vec<i64> {
        let i = b.schema().index_of(name).unwrap();
        let a = b.column(i).as_any().downcast_ref::<Int64Array>().unwrap();
        (0..a.len()).map(|j| a.value(j)).collect()
    }

    #[test]
    fn hex_count_aggregate() {
        pyo3::Python::initialize();
        // 10 points: 5 clustered tightly around (0.5, 0.5), 5 tightly around (5.0, 5.0).
        let xs = vec![
            0.5, 0.51, 0.49, 0.5, 0.5,
            5.0, 5.01, 4.99, 5.0, 5.0,
        ];
        let ys = vec![
            0.5, 0.51, 0.49, 0.5, 0.5,
            5.0, 5.01, 4.99, 5.0, 5.0,
        ];
        let b = make_xy_batch(xs, ys);
        let spec = HexSpec {
            x: "x".into(),
            y: "y".into(),
            bin_size: Some(1.0),
            aggregate: "count".into(),
            field: None,
            name: None,
        };
        let out = apply(&spec, &b).unwrap();

        // Group rows by hex_id; assert exactly 6 per hex and at least 2 hexes.
        let ids = col_i64(&out, "hex_id");
        let vals = col_f64(&out, "value");
        let mut by_id: HashMap<i64, Vec<f64>> = HashMap::new();
        for (id, v) in ids.iter().zip(vals.iter()) {
            by_id.entry(*id).or_default().push(*v);
        }
        assert!(by_id.len() >= 2, "expected ≥ 2 distinct hexes; got {}", by_id.len());
        // Each hex contributes 6 rows; total rows = 6 * N hexes.
        assert_eq!(out.num_rows(), 6 * by_id.len(), "rows = 6 * hex count");
        for (id, vs) in &by_id {
            assert_eq!(vs.len(), 6, "hex {id} should have 6 vertex rows");
            // All 6 share the same value.
            for v in vs {
                assert_eq!(v.to_bits(), vs[0].to_bits(), "hex {id} value not constant");
            }
        }
        // The two clusters' counts must sum to 10 (no points dropped).
        let total_count: f64 = by_id.values().map(|vs| vs[0]).sum();
        assert_eq!(total_count, 10.0);
    }

    #[test]
    fn hex_mean_and_sum_require_field() {
        pyo3::Python::initialize();
        let b = make_xyf_batch(
            vec![0.0, 1.0, 2.0],
            vec![0.0, 1.0, 2.0],
            vec![10.0, 20.0, 30.0],
        );
        // mean without field → error.
        let spec_no_field = HexSpec {
            x: "x".into(),
            y: "y".into(),
            bin_size: Some(1.0),
            aggregate: "mean".into(),
            field: None,
            name: None,
        };
        assert!(apply(&spec_no_field, &b).is_err());

        // sum without field → error.
        let spec_sum_no = HexSpec {
            x: "x".into(),
            y: "y".into(),
            bin_size: Some(1.0),
            aggregate: "sum".into(),
            field: None,
            name: None,
        };
        assert!(apply(&spec_sum_no, &b).is_err());

        // mean with field → success.
        let spec_mean = HexSpec {
            x: "x".into(),
            y: "y".into(),
            bin_size: Some(1.0),
            aggregate: "mean".into(),
            field: Some("v".into()),
            name: None,
        };
        let out = apply(&spec_mean, &b).unwrap();
        // 4 columns, vertex rows = 6 * (number of non-empty hexes).
        assert_eq!(out.num_columns(), 4);
        assert!(out.num_rows() > 0);
        assert_eq!(out.num_rows() % 6, 0);
    }

    #[test]
    fn hex_cube_rounding_correctness() {
        pyo3::Python::initialize();
        // A single point at (x_min, y_min) = (0, 0). Fractional axial → (0, 0).
        // hex_id = 0 * 65536 + 0 = 0.
        let b = make_xy_batch(vec![0.0], vec![0.0]);
        let spec = HexSpec {
            x: "x".into(),
            y: "y".into(),
            bin_size: Some(1.0),
            aggregate: "count".into(),
            field: None,
            name: None,
        };
        let out = apply(&spec, &b).unwrap();
        let ids = col_i64(&out, "hex_id");
        assert_eq!(out.num_rows(), 6, "single point → 1 hex × 6 vertices");
        for id in &ids {
            assert_eq!(*id, 0, "hex (0,0) packs to hex_id=0");
        }
        // Sanity-check cube_round directly on (0.4, 0.4):
        // x_cube=0.4, z_cube=0.4, y_cube=-0.8 → rx=0, ry=-1, rz=0;
        // dx=0.4, dy=0.2, dz=0.4 → dx>dy true, dx>dz false; dy>dz false → rz = -rx-ry = 1.
        let (q, r) = cube_round(0.4, 0.4);
        assert_eq!((q, r), (0, 1), "cube_round(0.4, 0.4) → (q=0, r=1) per RBG");
    }

    #[test]
    fn hex_six_vertices_per_non_empty_hex() {
        pyo3::Python::initialize();
        // 50 deterministic points along a slope: (i, i*1.7).
        let xs: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let ys: Vec<f64> = (0..50).map(|i| (i as f64) * 1.7).collect();
        let b = make_xy_batch(xs, ys);
        let spec = HexSpec {
            x: "x".into(),
            y: "y".into(),
            bin_size: None, // auto: (49 - 0) / 30 ≈ 1.633
            aggregate: "count".into(),
            field: None,
            name: None,
        };
        let out = apply(&spec, &b).unwrap();
        let ids = col_i64(&out, "hex_id");
        let vals = col_f64(&out, "value");

        let mut by_id: HashMap<i64, Vec<f64>> = HashMap::new();
        for (id, v) in ids.iter().zip(vals.iter()) {
            by_id.entry(*id).or_default().push(*v);
        }
        assert!(!by_id.is_empty(), "must have at least one non-empty hex");
        for (id, vs) in &by_id {
            assert_eq!(
                vs.len(),
                6,
                "hex_id {id} has {} rows; must be exactly 6",
                vs.len()
            );
            // All 6 rows share the same value (bit-identical).
            let first = vs[0].to_bits();
            for (k, v) in vs.iter().enumerate() {
                assert_eq!(
                    v.to_bits(),
                    first,
                    "hex {id} row {k} value {} != {}",
                    v,
                    vs[0]
                );
            }
        }
        // Total emitted rows = 6 * N hexes.
        assert_eq!(out.num_rows(), 6 * by_id.len());
    }
}
