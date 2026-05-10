//! Violin: per-group KDE → mirrored closed polygon vertices.
//!
//! For each group, runs KDE over the group's values to obtain a grid of (value, density)
//! pairs, normalizes density by `width / max_density` so the violin is exactly `width`
//! wide at its peak, and emits 2N vertices forming a closed polygon:
//!   - Right side bottom→top:  i = 0..n  → (+normalized_density, value)
//!   - Left side top→bottom:   i = (n-1)..=0 → (-normalized_density, value)
//!
//! Output schema = group_id (u32) + groupby cols + violin_x (f64) + violin_y (f64).

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch, StringArray, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::transform::kde::{self, BandwidthSpec, KdeSpec};

fn default_bandwidth() -> BandwidthSpec {
    BandwidthSpec::Scott
}
fn default_violin_n() -> usize {
    256
}
fn default_violin_width() -> f64 {
    0.4
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ViolinSpec {
    pub field: String,
    #[serde(default)]
    pub groupby: Vec<String>,
    #[serde(default = "default_bandwidth")]
    pub bandwidth: BandwidthSpec,
    #[serde(default = "default_violin_n")]
    pub n: usize,
    #[serde(default = "default_violin_width")]
    pub width: f64,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum KeyValue {
    Str(String),
    Float(u64),
}

pub(crate) fn apply(spec: &ViolinSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let schema = batch.schema();

    let v_idx = schema.index_of(&spec.field).map_err(|_| {
        PyValueError::new_err(format!("stat_violin: column '{}' not found", spec.field))
    })?;
    if schema.field(v_idx).data_type() != &DataType::Float64 {
        return Err(PyValueError::new_err(format!(
            "stat_violin: column '{}' must be Float64",
            spec.field
        )));
    }

    let mut group_dtypes: Vec<DataType> = Vec::with_capacity(spec.groupby.len());
    for g in &spec.groupby {
        let i = schema.index_of(g).map_err(|_| {
            PyValueError::new_err(format!(
                "stat_violin: groupby column '{}' not found",
                g
            ))
        })?;
        let dt = schema.field(i).data_type().clone();
        if dt != DataType::Float64 && !matches!(dt, DataType::Utf8) {
            return Err(PyValueError::new_err(format!(
                "stat_violin: groupby column '{}' must be Float64 or Utf8",
                g
            )));
        }
        group_dtypes.push(dt);
    }

    let n_rows = batch.num_rows();

    // Build output schema up-front so empty input still returns a valid RecordBatch.
    let mut fields: Vec<Field> = Vec::with_capacity(spec.groupby.len() + 3);
    fields.push(Field::new("group_id", DataType::UInt32, false));
    for (gi, g) in spec.groupby.iter().enumerate() {
        fields.push(Field::new(g, group_dtypes[gi].clone(), false));
    }
    fields.push(Field::new("violin_x", DataType::Float64, false));
    fields.push(Field::new("violin_y", DataType::Float64, false));
    let out_schema = Arc::new(Schema::new(fields));

    if n_rows == 0 {
        let mut cols: Vec<ArrayRef> = Vec::with_capacity(spec.groupby.len() + 3);
        cols.push(Arc::new(UInt32Array::from(Vec::<u32>::new())));
        for gi in 0..spec.groupby.len() {
            match group_dtypes[gi] {
                DataType::Float64 => {
                    cols.push(Arc::new(Float64Array::from(Vec::<f64>::new())));
                }
                DataType::Utf8 => {
                    cols.push(Arc::new(StringArray::from(Vec::<String>::new())));
                }
                _ => unreachable!(),
            }
        }
        cols.push(Arc::new(Float64Array::from(Vec::<f64>::new())));
        cols.push(Arc::new(Float64Array::from(Vec::<f64>::new())));
        return RecordBatch::try_new(out_schema, cols)
            .map_err(|e| PyValueError::new_err(format!("stat_violin: {e}")));
    }

    let v_arr = batch
        .column(v_idx)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();

    // Bucket rows by group key (BTreeMap → deterministic ordering).
    let mut groups: BTreeMap<Vec<KeyValue>, Vec<usize>> = BTreeMap::new();
    let group_arrays: Vec<&dyn arrow::array::Array> = spec
        .groupby
        .iter()
        .map(|g| batch.column(schema.index_of(g).unwrap()).as_ref())
        .collect();

    for row in 0..n_rows {
        let mut key = Vec::with_capacity(spec.groupby.len());
        for (gi, arr) in group_arrays.iter().enumerate() {
            match group_dtypes[gi] {
                DataType::Float64 => {
                    let a = arr.as_any().downcast_ref::<Float64Array>().unwrap();
                    if a.is_null(row) {
                        key.push(KeyValue::Float(f64::NAN.to_bits()));
                    } else {
                        key.push(KeyValue::Float(a.value(row).to_bits()));
                    }
                }
                DataType::Utf8 => {
                    let a = arr.as_any().downcast_ref::<StringArray>().unwrap();
                    if a.is_null(row) {
                        key.push(KeyValue::Str(String::new()));
                    } else {
                        key.push(KeyValue::Str(a.value(row).to_string()));
                    }
                }
                _ => unreachable!(),
            }
        }
        groups.entry(key).or_default().push(row);
    }
    if spec.groupby.is_empty() {
        let all: Vec<usize> = (0..n_rows).collect();
        groups.clear();
        groups.insert(Vec::new(), all);
    }

    // Per-group KDE → mirrored polygon vertices.
    let mut group_ids: Vec<u32> = Vec::new();
    let mut keys_out: Vec<Vec<KeyValue>> = Vec::new();
    let mut violin_x: Vec<f64> = Vec::new();
    let mut violin_y: Vec<f64> = Vec::new();

    for (gid, (key, rows)) in groups.iter().enumerate() {
        let vals: Vec<f64> = rows
            .iter()
            .filter_map(|&r| {
                if v_arr.is_null(r) {
                    return None;
                }
                let v = v_arr.value(r);
                if v.is_nan() {
                    return None;
                }
                Some(v)
            })
            .collect();

        if vals.len() < 2 {
            continue;
        }

        let (lo, hi) = vals
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                (a.min(v), b.max(v))
            });
        if !(lo.is_finite() && hi.is_finite()) || lo >= hi {
            continue;
        }

        // Build a synthetic 1-col Float64 batch for kde::apply.
        let synth_schema = Arc::new(Schema::new(vec![Field::new(
            spec.field.as_str(),
            DataType::Float64,
            false,
        )]));
        let synth_batch = RecordBatch::try_new(
            synth_schema,
            vec![Arc::new(Float64Array::from(vals.clone()))],
        )
        .map_err(|e| PyValueError::new_err(format!("stat_violin: synth batch: {e}")))?;

        let kde_spec = KdeSpec {
            field: spec.field.clone(),
            bandwidth: spec.bandwidth.clone(),
            n: spec.n,
            extent: Some((lo, hi)),
            cumulative: false,
            name: None,
        };
        let kde_out = kde::apply(&kde_spec, &synth_batch)?;
        let value_col = kde_out
            .column(kde_out.schema().index_of("value").unwrap())
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .clone();
        let density_col = kde_out
            .column(kde_out.schema().index_of("density").unwrap())
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .clone();

        // Find max density (ignore NaN).
        let mut max_density = f64::NEG_INFINITY;
        for i in 0..density_col.len() {
            if !density_col.is_null(i) {
                let d = density_col.value(i);
                if d.is_finite() && d > max_density {
                    max_density = d;
                }
            }
        }
        if !(max_density.is_finite()) || max_density <= 0.0 {
            continue;
        }

        let scale = spec.width / max_density;
        let n = spec.n;

        // Right side: bottom→top  (i = 0..n) → +scale * density
        for i in 0..n {
            let d = if density_col.is_null(i) {
                0.0
            } else {
                let dv = density_col.value(i);
                if dv.is_finite() {
                    dv
                } else {
                    0.0
                }
            };
            group_ids.push(gid as u32);
            keys_out.push(key.clone());
            violin_x.push(scale * d);
            violin_y.push(value_col.value(i));
        }
        // Left side: top→bottom  (i in (0..n).rev()) → -scale * density
        for i in (0..n).rev() {
            let d = if density_col.is_null(i) {
                0.0
            } else {
                let dv = density_col.value(i);
                if dv.is_finite() {
                    dv
                } else {
                    0.0
                }
            };
            group_ids.push(gid as u32);
            keys_out.push(key.clone());
            violin_x.push(-scale * d);
            violin_y.push(value_col.value(i));
        }
    }

    let mut cols: Vec<ArrayRef> = Vec::with_capacity(spec.groupby.len() + 3);
    cols.push(Arc::new(UInt32Array::from(group_ids)));
    for gi in 0..spec.groupby.len() {
        match group_dtypes[gi] {
            DataType::Float64 => {
                let v: Vec<f64> = keys_out
                    .iter()
                    .map(|k| match &k[gi] {
                        KeyValue::Float(bits) => f64::from_bits(*bits),
                        KeyValue::Str(_) => unreachable!(),
                    })
                    .collect();
                cols.push(Arc::new(Float64Array::from(v)));
            }
            DataType::Utf8 => {
                let v: Vec<String> = keys_out
                    .iter()
                    .map(|k| match &k[gi] {
                        KeyValue::Str(s) => s.clone(),
                        KeyValue::Float(_) => unreachable!(),
                    })
                    .collect();
                cols.push(Arc::new(StringArray::from(v)));
            }
            _ => unreachable!(),
        }
    }
    cols.push(Arc::new(Float64Array::from(violin_x)));
    cols.push(Arc::new(Float64Array::from(violin_y)));

    RecordBatch::try_new(out_schema, cols)
        .map_err(|e| PyValueError::new_err(format!("stat_violin: {e}")))
}

use pyo3::prelude::*;

#[pyclass(eq, module = "ferrum._core", name = "Violin")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyViolin(pub(crate) crate::transform::core::TransformSpec);

#[pymethods]
impl PyViolin {
    #[new]
    #[pyo3(signature = (field, *, groupby = vec![], bandwidth = None, n = 256, width = 0.4, name = None))]
    fn new(
        field: &str,
        groupby: Vec<String>,
        bandwidth: Option<&Bound<'_, PyAny>>,
        n: usize,
        width: f64,
        name: Option<String>,
    ) -> PyResult<Self> {
        if field.is_empty() {
            return Err(PyValueError::new_err("Violin: field must be non-empty"));
        }
        if n == 0 {
            return Err(PyValueError::new_err("Violin: n must be > 0"));
        }
        if !width.is_finite() || width <= 0.0 {
            return Err(PyValueError::new_err(
                "Violin: width must be a positive finite number",
            ));
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
                                "Violin: unknown bandwidth '{other}'; expected 'scott' | 'silverman' | float"
                            )))
                        }
                    }
                } else if let Ok(v) = obj.extract::<f64>() {
                    if !v.is_finite() || v <= 0.0 {
                        return Err(PyValueError::new_err(
                            "Violin: numeric bandwidth must be a positive finite number",
                        ));
                    }
                    BandwidthSpec::Fixed { value: v }
                } else {
                    return Err(PyValueError::new_err(
                        "Violin: bandwidth must be 'scott', 'silverman', or a positive float",
                    ));
                }
            }
        };
        let mut seen = std::collections::HashSet::new();
        for g in &groupby {
            if !seen.insert(g.as_str()) {
                return Err(PyValueError::new_err(format!(
                    "Violin: duplicate groupby field '{g}'"
                )));
            }
        }
        Ok(Self(crate::transform::core::TransformSpec::Violin(
            ViolinSpec {
                field: field.to_string(),
                groupby,
                bandwidth: bw,
                n,
                width,
                name,
            },
        )))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            crate::transform::core::TransformSpec::Violin(s) => format!(
                "Violin(field='{}', groupby={:?}, bandwidth={:?}, n={}, width={})",
                s.field, s.groupby, s.bandwidth, s.n, s.width
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, RecordBatch, StringArray, UInt32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn batch(field: &str, values: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(field, DataType::Float64, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(values))]).unwrap()
    }

    fn batch_value_group(values: Vec<f64>, groups: Vec<&str>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Float64, false),
            Field::new("group", DataType::Utf8, false),
        ]));
        let v = Float64Array::from(values);
        let g = StringArray::from(groups);
        RecordBatch::try_new(schema, vec![Arc::new(v), Arc::new(g)]).unwrap()
    }

    fn col_f64(b: &RecordBatch, name: &str) -> Vec<f64> {
        let arr = b
            .column(b.schema().index_of(name).unwrap())
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        (0..arr.len())
            .map(|i| if arr.is_null(i) { f64::NAN } else { arr.value(i) })
            .collect()
    }

    fn col_u32(b: &RecordBatch, name: &str) -> Vec<u32> {
        let arr = b
            .column(b.schema().index_of(name).unwrap())
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        (0..arr.len()).map(|i| arr.value(i)).collect()
    }

    fn col_str(b: &RecordBatch, name: &str) -> Vec<String> {
        let arr = b
            .column(b.schema().index_of(name).unwrap())
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        (0..arr.len()).map(|i| arr.value(i).to_string()).collect()
    }

    #[test]
    fn violin_polygon_vertex_count_and_symmetry() {
        pyo3::Python::initialize();
        let vals: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let b = batch("v", vals);
        let spec = ViolinSpec {
            field: "v".into(),
            groupby: vec![],
            bandwidth: BandwidthSpec::Scott,
            n: 64,
            width: 0.5,
            name: None,
        };
        let out = apply(&spec, &b).unwrap();
        let n = 64usize;
        assert_eq!(out.num_rows(), 2 * n, "expected 2*n = 128 rows");

        let xs = col_f64(&out, "violin_x");
        let ys = col_f64(&out, "violin_y");
        // For each i in 0..n, row i is right-side and row (2n-1-i) is left-side mirror.
        for i in 0..n {
            let mirror = 2 * n - 1 - i;
            assert!(
                (xs[i] + xs[mirror]).abs() < 1e-12,
                "x[{i}] should be negation of x[{mirror}]; got {} vs {}",
                xs[i],
                xs[mirror]
            );
            assert!(
                (ys[i] - ys[mirror]).abs() < 1e-12,
                "y[{i}] should equal y[{mirror}]; got {} vs {}",
                ys[i],
                ys[mirror]
            );
        }
    }

    #[test]
    fn violin_per_group_distinct_group_ids() {
        pyo3::Python::initialize();
        // Group "a": small values; Group "b": large values.
        let mut vals: Vec<f64> = Vec::new();
        let mut grps: Vec<&str> = Vec::new();
        for i in 0..50 {
            vals.push(i as f64);
            grps.push("a");
        }
        for i in 0..50 {
            vals.push(100.0 + i as f64);
            grps.push("b");
        }
        let b = batch_value_group(vals, grps);
        let spec = ViolinSpec {
            field: "v".into(),
            groupby: vec!["group".into()],
            bandwidth: BandwidthSpec::Scott,
            n: 32,
            width: 0.4,
            name: None,
        };
        let out = apply(&spec, &b).unwrap();
        assert_eq!(out.num_rows(), 2 * 32 * 2, "expected 128 rows");

        let gids = col_u32(&out, "group_id");
        let groups = col_str(&out, "group");

        // Distinct group ids present.
        let mut distinct: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
        for g in &gids {
            distinct.insert(*g);
        }
        assert!(distinct.contains(&0), "expected group_id 0 present");
        assert!(distinct.contains(&1), "expected group_id 1 present");

        // group_id=0 rows all share one group key value; group_id=1 the other.
        let g0_keys: std::collections::BTreeSet<&str> = gids
            .iter()
            .enumerate()
            .filter(|(_, gid)| **gid == 0)
            .map(|(i, _)| groups[i].as_str())
            .collect();
        let g1_keys: std::collections::BTreeSet<&str> = gids
            .iter()
            .enumerate()
            .filter(|(_, gid)| **gid == 1)
            .map(|(i, _)| groups[i].as_str())
            .collect();
        assert_eq!(g0_keys.len(), 1, "group_id=0 rows must share one key");
        assert_eq!(g1_keys.len(), 1, "group_id=1 rows must share one key");
        assert_ne!(g0_keys, g1_keys, "group_id=0 and =1 must hold different keys");
    }

    #[test]
    fn violin_bandwidth_variants_parse_and_apply() {
        pyo3::Python::initialize();
        let vals: Vec<f64> = (0..60).map(|i| (i as f64) * 0.1).collect();
        let n = 32usize;
        for bw in [
            BandwidthSpec::Scott,
            BandwidthSpec::Silverman,
            BandwidthSpec::Fixed { value: 0.5 },
        ] {
            let b = batch("v", vals.clone());
            let spec = ViolinSpec {
                field: "v".into(),
                groupby: vec![],
                bandwidth: bw.clone(),
                n,
                width: 0.4,
                name: None,
            };
            let out = apply(&spec, &b).unwrap();
            assert_eq!(out.num_rows(), 2 * n, "bw {:?}: expected {} rows", bw, 2 * n);
            let xs = col_f64(&out, "violin_x");
            assert!(
                xs.iter().all(|x| !x.is_nan()),
                "bw {:?}: violin_x must contain no NaN",
                bw
            );
        }
    }
}
