//! Swarm: greedy beeswarm placement.
//!
//! Output schema: input schema PLUS appended `swarm_x: Float64, swarm_y: Float64`.
//! `swarm_x` is the per-category horizontal offset (relative to category center,
//! in data-space along the category axis); `swarm_y` is the original `value`.
//!
//! Algorithm: per category, sort points by `value` ascending (stable on row index
//! for byte-deterministic placements). For each point, try candidate offsets in
//! the order [0, +d, -d, +2d, -2d, ...] (or restricted to one side); accept the
//! first that doesn't overlap any already-placed point in the same category.
//!
//! `d = point_size + spacing`, converted from pixels to data-space along the
//! value axis using `panel_pixel_size` from the TransformContext. Without context,
//! falls back to `radius_data = 1.0`.
//!
//! TODO: warn-once when no context is provided so default radius doesn't silently
//! mis-place; out of scope for this task.

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::transform::context::TransformContext;

fn default_point_size() -> f64 {
    5.0
}
fn default_spacing() -> f64 {
    1.0
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub(crate) enum SwarmSide {
    #[default]
    Both,
    Left,
    Right,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct SwarmSpec {
    pub category: String, // grouping field (Utf8 or Float64)
    pub value: String,    // Float64
    #[serde(default = "default_point_size")]
    pub point_size: f64,
    #[serde(default = "default_spacing")]
    pub spacing: f64,
    #[serde(default)]
    pub side: SwarmSide,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

pub(crate) fn apply(spec: &SwarmSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    apply_with_context(spec, batch, &TransformContext::default())
}

pub(crate) fn apply_with_context(
    spec: &SwarmSpec,
    batch: &RecordBatch,
    ctx: &TransformContext,
) -> PyResult<RecordBatch> {
    // 1. Validate columns.
    let schema = batch.schema();
    let cat_idx = schema
        .index_of(&spec.category)
        .map_err(|_| PyValueError::new_err(format!(
            "stat_swarm: column '{}' not found", spec.category
        )))?;
    let cat_dt = schema.field(cat_idx).data_type().clone();
    if cat_dt != DataType::Utf8 && cat_dt != DataType::Float64 {
        return Err(PyValueError::new_err(format!(
            "stat_swarm: category column '{}' must be Utf8 or Float64; got {:?}",
            spec.category, cat_dt
        )));
    }
    let val_idx = schema
        .index_of(&spec.value)
        .map_err(|_| PyValueError::new_err(format!(
            "stat_swarm: column '{}' not found", spec.value
        )))?;
    if schema.field(val_idx).data_type() != &DataType::Float64 {
        return Err(PyValueError::new_err(format!(
            "stat_swarm: value column '{}' must be Float64", spec.value
        )));
    }

    let n = batch.num_rows();
    let val_arr = batch
        .column(val_idx)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();

    // 2. Compute value range for radius_data.
    let mut vmin = f64::INFINITY;
    let mut vmax = f64::NEG_INFINITY;
    for i in 0..n {
        if val_arr.is_null(i) {
            continue;
        }
        let v = val_arr.value(i);
        if v.is_nan() {
            continue;
        }
        if v < vmin {
            vmin = v;
        }
        if v > vmax {
            vmax = v;
        }
    }

    // radius_data: how far points need to be in value-axis units.
    let radius_data: f64 = match ctx.panel_pixel_size {
        Some((_, h)) if h > 0 => {
            let value_range = vmax - vmin;
            if value_range.is_finite() && value_range > 0.0 {
                (spec.point_size + spec.spacing) * (value_range / h as f64)
            } else {
                1.0
            }
        }
        _ => 1.0,
    };

    // 3. Group by category. BTreeMap for deterministic iteration order.
    let mut groups: BTreeMap<String, Vec<(usize, f64)>> = BTreeMap::new();
    let cat_keys: Vec<Option<String>> = match cat_dt {
        DataType::Utf8 => {
            let arr = batch
                .column(cat_idx)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            (0..n)
                .map(|i| {
                    if arr.is_null(i) {
                        None
                    } else {
                        Some(arr.value(i).to_string())
                    }
                })
                .collect()
        }
        DataType::Float64 => {
            let arr = batch
                .column(cat_idx)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            (0..n)
                .map(|i| {
                    if arr.is_null(i) {
                        None
                    } else {
                        // Use bit-pattern repr to handle NaN/0.0/-0.0 deterministically.
                        Some(arr.value(i).to_bits().to_string())
                    }
                })
                .collect()
        }
        _ => unreachable!(),
    };

    for i in 0..n {
        if val_arr.is_null(i) {
            continue;
        }
        let v = val_arr.value(i);
        if v.is_nan() {
            continue;
        }
        let key = match &cat_keys[i] {
            None => continue,
            Some(s) => s.clone(),
        };
        groups.entry(key).or_default().push((i, v));
    }

    // 4. Per-category greedy placement.
    let mut swarm_x: Vec<Option<f64>> = vec![None; n];
    let mut swarm_y: Vec<Option<f64>> = vec![None; n];

    let d = radius_data; // step size and overlap radius reference
    let two_r_sq = (2.0 * d) * (2.0 * d);

    for entries in groups.values_mut() {
        // Sort by value ascending; tiebreak by row index (stable, deterministic).
        entries.sort_by(|(idx_a, va), (idx_b, vb)| {
            va.partial_cmp(vb)
                .unwrap_or(Ordering::Equal)
                .then_with(|| idx_a.cmp(idx_b))
        });

        // Already-placed: (offset, value).
        let mut placed: Vec<(f64, f64)> = Vec::with_capacity(entries.len());

        for &(row_idx, v) in entries.iter() {
            // Generate candidate offsets in side-appropriate order.
            let mut placed_offset: Option<f64> = None;
            // Worst case: side=Left/Right needs 2*N steps (every other slot blocked);
            // side=Both needs ~N steps (alternating). Use a generous upper bound.
            let max_steps = 4 * entries.len() + 4;
            for step in 0..=max_steps {
                // Candidates per step.
                let candidates: Vec<f64> = match spec.side {
                    SwarmSide::Both => {
                        if step == 0 {
                            vec![0.0]
                        } else {
                            vec![(step as f64) * d, -(step as f64) * d]
                        }
                    }
                    SwarmSide::Left => {
                        if step == 0 {
                            vec![0.0]
                        } else {
                            vec![-(step as f64) * d]
                        }
                    }
                    SwarmSide::Right => {
                        if step == 0 {
                            vec![0.0]
                        } else {
                            vec![(step as f64) * d]
                        }
                    }
                };
                for cand in candidates {
                    let mut overlaps = false;
                    for &(po, pv) in placed.iter() {
                        let dx = cand - po;
                        let dy = v - pv;
                        if dx * dx + dy * dy < two_r_sq {
                            overlaps = true;
                            break;
                        }
                    }
                    if !overlaps {
                        placed_offset = Some(cand);
                        break;
                    }
                }
                if placed_offset.is_some() {
                    break;
                }
            }
            // Fallback: should never trigger with the generous max_steps above; use a
            // sign-correct sentinel for Left so we don't violate the side invariant.
            let fallback = match spec.side {
                SwarmSide::Left => -((max_steps as f64) * d),
                SwarmSide::Right | SwarmSide::Both => (max_steps as f64) * d,
            };
            let off = placed_offset.unwrap_or(fallback);
            placed.push((off, v));
            swarm_x[row_idx] = Some(off);
            swarm_y[row_idx] = Some(v);
        }
    }

    // 5. Build output: input columns + swarm_x + swarm_y.
    let mut out_cols: Vec<ArrayRef> = batch.columns().to_vec();
    let sx_arr: Float64Array = swarm_x.iter().copied().collect();
    let sy_arr: Float64Array = swarm_y.iter().copied().collect();
    out_cols.push(Arc::new(sx_arr));
    out_cols.push(Arc::new(sy_arr));

    let mut new_fields: Vec<Field> = schema
        .fields()
        .iter()
        .map(|f| (**f).clone())
        .collect();
    new_fields.push(Field::new("swarm_x", DataType::Float64, true));
    new_fields.push(Field::new("swarm_y", DataType::Float64, true));
    let out_schema = Arc::new(Schema::new(new_fields));

    RecordBatch::try_new(out_schema, out_cols)
        .map_err(|e| PyValueError::new_err(format!("stat_swarm: {e}")))
}

// ---------- PyO3 wrapper ----------

use pyo3::prelude::*;

use crate::transform::core::TransformSpec;

#[pyclass(eq, module = "ferrum._core", name = "Swarm")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PySwarm(pub(crate) TransformSpec);

#[pymethods]
impl PySwarm {
    #[new]
    #[pyo3(signature = (
        category, value, *,
        point_size = 5.0,
        spacing = 1.0,
        side = "both",
        name = None,
    ))]
    fn new(
        category: &str,
        value: &str,
        point_size: f64,
        spacing: f64,
        side: &str,
        name: Option<String>,
    ) -> PyResult<Self> {
        if category.is_empty() || value.is_empty() {
            return Err(PyValueError::new_err(
                "Swarm: category and value fields must be non-empty",
            ));
        }
        if !point_size.is_finite() || point_size <= 0.0 {
            return Err(PyValueError::new_err(
                "Swarm: point_size must be a positive finite number",
            ));
        }
        if !spacing.is_finite() || spacing < 0.0 {
            return Err(PyValueError::new_err(
                "Swarm: spacing must be a non-negative finite number",
            ));
        }
        let parsed_side = match side {
            "both" => SwarmSide::Both,
            "left" => SwarmSide::Left,
            "right" => SwarmSide::Right,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Swarm: unknown side '{other}'; expected both|left|right"
                )))
            }
        };
        Ok(PySwarm(TransformSpec::Swarm(SwarmSpec {
            category: category.to_string(),
            value: value.to_string(),
            point_size,
            spacing,
            side: parsed_side,
            name,
        })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Swarm(s) => format!(
                "Swarm(category='{}', value='{}', point_size={}, spacing={}, side={:?}, name={:?})",
                s.category, s.value, s.point_size, s.spacing, s.side, s.name
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, RecordBatch, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn make_batch(cats: Vec<&str>, vals: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Utf8, false),
            Field::new("v", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(cats)),
                Arc::new(Float64Array::from(vals)),
            ],
        )
        .unwrap()
    }

    fn ctx_with_panel(w: u32, h: u32) -> TransformContext {
        TransformContext {
            panel_pixel_size: Some((w, h)),
            ..Default::default()
        }
    }

    fn col_f64<'a>(batch: &'a RecordBatch, name: &str) -> &'a Float64Array {
        batch
            .column_by_name(name)
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
    }

    #[test]
    fn swarm_side_both_alternates_around_zero() {
        pyo3::Python::initialize();
        // 5 points all clustered very close in value → forces alternating offsets.
        let cats = vec!["A", "A", "A", "A", "A"];
        let vals = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let b = make_batch(cats, vals);
        let spec = SwarmSpec {
            category: "cat".into(),
            value: "v".into(),
            point_size: 5.0,
            spacing: 1.0,
            side: SwarmSide::Both,
            name: None,
        };
        // Big panel → small radius.
        let out = apply_with_context(&spec, &b, &ctx_with_panel(400, 400)).unwrap();
        assert_eq!(out.num_rows(), 5);
        assert_eq!(out.num_columns(), 4); // cat, v, swarm_x, swarm_y

        let sx = col_f64(&out, "swarm_x");
        let mut has_pos = false;
        let mut has_neg = false;
        let mut has_zero = false;
        for i in 0..5 {
            let x = sx.value(i);
            if x > 0.0 {
                has_pos = true;
            } else if x < 0.0 {
                has_neg = true;
            } else {
                has_zero = true;
            }
        }
        assert!(has_zero, "expected center point at offset 0");
        assert!(has_pos, "Both side should produce positive offsets");
        assert!(has_neg, "Both side should produce negative offsets");
    }

    #[test]
    fn swarm_side_left_only_non_positive_offsets() {
        pyo3::Python::initialize();
        let cats = vec!["A", "A", "A", "A", "A"];
        let vals = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let b = make_batch(cats.clone(), vals.clone());
        // Left
        let spec_l = SwarmSpec {
            category: "cat".into(),
            value: "v".into(),
            point_size: 5.0,
            spacing: 1.0,
            side: SwarmSide::Left,
            name: None,
        };
        let out_l = apply_with_context(&spec_l, &b, &ctx_with_panel(400, 400)).unwrap();
        let sx_l = col_f64(&out_l, "swarm_x");
        for i in 0..5 {
            assert!(
                sx_l.value(i) <= 0.0,
                "Left side: swarm_x[{i}] = {} must be <= 0",
                sx_l.value(i)
            );
        }
        // Right
        let spec_r = SwarmSpec {
            category: "cat".into(),
            value: "v".into(),
            point_size: 5.0,
            spacing: 1.0,
            side: SwarmSide::Right,
            name: None,
        };
        let out_r = apply_with_context(&spec_r, &b, &ctx_with_panel(400, 400)).unwrap();
        let sx_r = col_f64(&out_r, "swarm_x");
        for i in 0..5 {
            assert!(
                sx_r.value(i) >= 0.0,
                "Right side: swarm_x[{i}] = {} must be >= 0",
                sx_r.value(i)
            );
        }
    }

    #[test]
    fn swarm_deterministic_tiebreak() {
        pyo3::Python::initialize();
        // Several points with identical values to force ties on value.
        let cats = vec!["A", "A", "A", "B", "B", "A"];
        let vals = vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0];
        let b1 = make_batch(cats.clone(), vals.clone());
        let b2 = make_batch(cats, vals);
        let spec = SwarmSpec {
            category: "cat".into(),
            value: "v".into(),
            point_size: 5.0,
            spacing: 1.0,
            side: SwarmSide::Both,
            name: None,
        };
        let ctx = ctx_with_panel(400, 400);
        let o1 = apply_with_context(&spec, &b1, &ctx).unwrap();
        let o2 = apply_with_context(&spec, &b2, &ctx).unwrap();

        let sx1 = col_f64(&o1, "swarm_x");
        let sx2 = col_f64(&o2, "swarm_x");
        let sy1 = col_f64(&o1, "swarm_y");
        let sy2 = col_f64(&o2, "swarm_y");
        assert_eq!(sx1.len(), sx2.len());
        for i in 0..sx1.len() {
            assert_eq!(
                sx1.value(i).to_bits(),
                sx2.value(i).to_bits(),
                "swarm_x[{i}] differs"
            );
            assert_eq!(
                sy1.value(i).to_bits(),
                sy2.value(i).to_bits(),
                "swarm_y[{i}] differs"
            );
        }
    }

    #[test]
    fn swarm_no_overlapping_placements() {
        pyo3::Python::initialize();
        // 20 points all at value=1.0 in one category. Generous panel → small radius.
        let cats: Vec<&str> = (0..20).map(|_| "A").collect();
        let vals: Vec<f64> = (0..20).map(|_| 1.0).collect();
        let b = make_batch(cats, vals);
        let spec = SwarmSpec {
            category: "cat".into(),
            value: "v".into(),
            point_size: 5.0,
            spacing: 1.0,
            side: SwarmSide::Both,
            name: None,
        };
        let out = apply_with_context(&spec, &b, &ctx_with_panel(400, 400)).unwrap();
        let sx = col_f64(&out, "swarm_x");
        let sy = col_f64(&out, "swarm_y");
        // Reproduce the radius the transform used.
        // value_range = 0 here (all v=1.0), so transform falls back to radius_data = 1.0.
        let radius = 1.0_f64;
        let two_r_sq = (2.0 * radius) * (2.0 * radius);
        let n = sx.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = sx.value(i) - sx.value(j);
                let dy = sy.value(i) - sy.value(j);
                let d2 = dx * dx + dy * dy;
                assert!(
                    d2 >= two_r_sq - 1e-12,
                    "points {i} and {j} overlap: d^2 = {d2} < {two_r_sq}"
                );
            }
        }
    }
}
