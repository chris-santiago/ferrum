//! Swarm: greedy beeswarm placement.
//!
//! Output schema: input schema PLUS appended `swarm_x: Float64, swarm_y: Float64,
//! `__pos_x_offset__: Float64` (vertical orient) or `__pos_y_offset__: Float64`
//! (horizontal orient).
//! `swarm_x` is the per-category cross-axis offset in *value-axis data units*
//! (so callers can do their own conversion if needed); `swarm_y` is the original
//! `value`. The position-offset column converts the data-unit offset to *pixels*
//! on the cross-axis so the standard rendering path (`render/position.rs`) picks
//! it up automatically when the desugar encodes the chart's original category
//! field on the appropriate axis.
//!
//! Algorithm: per category, sort points by `value` ascending (stable on row index
//! for byte-deterministic placements). For each point, try candidate offsets in
//! the order [0, +d, -d, +2d, -2d, ...] (or restricted to one side); accept the
//! first that doesn't overlap any already-placed point in the same category.
//!
//! `d = point_size + spacing`, converted from pixels to data-space along the
//! value axis using `panel_pixel_size` from the TransformContext. Without context,
//! falls back to `radius_data = 1.0`; callers must supply panel_pixel_size for
//! accurate point placement.

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::transform::context::TransformContext;
use crate::transform::group_key::{groupby_key_at, is_groupby_supported_dtype, KeyValue};

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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub(crate) enum SwarmOrient {
    #[default]
    Vertical,
    Horizontal,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct SwarmSpec {
    pub category: String, // grouping field (Utf8/LargeUtf8, Float, integer, or Boolean)
    pub value: String,    // Float64
    #[serde(default = "default_point_size")]
    pub point_size: f64,
    #[serde(default = "default_spacing")]
    pub spacing: f64,
    #[serde(default)]
    pub side: SwarmSide,
    #[serde(default)]
    pub orient: SwarmOrient,
    /// When set, partition within each category by this field and offset each
    /// sub-group's swarm so the groups appear side by side within the band.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub dodge: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

pub(crate) fn apply(spec: &SwarmSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    apply_with_context(spec, batch, &TransformContext::default())
}

/// Partition by (category, dodge) and run swarm per sub-group, offsetting
/// each group's cross-axis position so the groups sit side by side.
fn apply_dodged(
    spec: &SwarmSpec,
    batch: &RecordBatch,
    ctx: &TransformContext,
    dodge_field: &str,
) -> PyResult<RecordBatch> {
    use std::collections::BTreeSet;
    let schema = batch.schema();
    let dodge_idx = schema.index_of(dodge_field).map_err(|_| {
        PyValueError::new_err(format!("stat_swarm: dodge column '{}' not found", dodge_field))
    })?;
    if schema.field(dodge_idx).data_type() != &DataType::Utf8 {
        return Err(PyValueError::new_err(format!(
            "stat_swarm: dodge column '{}' must be Utf8", dodge_field
        )));
    }
    let dodge_arr = batch.column(dodge_idx).as_any().downcast_ref::<StringArray>()
        .ok_or_else(|| PyValueError::new_err(format!("stat_swarm: dodge column '{dodge_field}' is Utf8 but downcast failed")))?;
    let dodge_values: Vec<Option<String>> = (0..batch.num_rows())
        .map(|i| if dodge_arr.is_null(i) { None } else { Some(dodge_arr.value(i).to_string()) })
        .collect();

    // Collect unique dodge groups in sorted order for deterministic sub-band assignment.
    let dodge_groups: Vec<String> = {
        let mut set: BTreeSet<String> = BTreeSet::new();
        for s in dodge_values.iter().flatten() { set.insert(s.clone()); }
        set.into_iter().collect()
    };
    let n_groups = dodge_groups.len().max(1);

    // Sub-band width in cross-axis pixels: separate groups by (point_size + spacing).
    let sub_band_px = spec.point_size + spec.spacing;
    // Center offsets for each dodge group, symmetric around 0.
    let centers: Vec<f64> = (0..n_groups)
        .map(|i| (i as f64 - (n_groups as f64 - 1.0) / 2.0) * sub_band_px)
        .collect();

    // Run a sub-spec swarm per dodge group using the restricted SwarmSpec.
    // The beeswarm offsets from this are then shifted by the group's center.
    let mut combined_out: Option<RecordBatch> = None;
    for (gi, group_val) in dodge_groups.iter().enumerate() {
        let center_px = centers[gi];
        // Filter batch to rows belonging to this dodge group.
        let mask: Vec<bool> = dodge_values.iter().map(|v| {
            v.as_deref() == Some(group_val.as_str())
        }).collect();
        let indices: Vec<u32> = mask.iter().enumerate()
            .filter(|(_, &b)| b)
            .map(|(i, _)| i as u32)
            .collect();
        if indices.is_empty() { continue; }
        let idx_arr = arrow::array::UInt32Array::from(indices.clone());
        let sub_batch = arrow::compute::take_record_batch(batch, &idx_arr)
            .map_err(|e| PyValueError::new_err(format!("stat_swarm dodge filter: {e}")))?;

        // Run swarm on sub-batch.
        let sub_spec = SwarmSpec { dodge: None, ..spec.clone() };
        let sub_out = apply_with_context(&sub_spec, &sub_batch, ctx)?;

        // Add the dodge group center offset to __pos_x_offset__ / __pos_y_offset__.
        let offset_col_name = match spec.orient {
            SwarmOrient::Vertical => "__pos_x_offset__",
            SwarmOrient::Horizontal => "__pos_y_offset__",
        };
        let oci = sub_out.schema().index_of(offset_col_name)
            .map_err(|_| PyValueError::new_err(format!("stat_swarm: offset col '{offset_col_name}' missing")))?;
        let orig_offsets = sub_out.column(oci).as_any().downcast_ref::<Float64Array>()
            .ok_or_else(|| PyValueError::new_err(format!("stat_swarm: offset col '{offset_col_name}' not Float64")))?;
        let shifted: Float64Array = orig_offsets.iter()
            .map(|v| Some(v.unwrap_or(0.0) + center_px))
            .collect();
        let mut cols: Vec<ArrayRef> = sub_out.columns().to_vec();
        cols[oci] = Arc::new(shifted);
        let shifted_out = RecordBatch::try_new(sub_out.schema(), cols)
            .map_err(|e| PyValueError::new_err(format!("stat_swarm dodge shift: {e}")))?;

        // Merge into combined output by concatenating batches.
        combined_out = Some(match combined_out {
            None => shifted_out,
            Some(existing) => arrow::compute::concat_batches(&existing.schema(), &[existing, shifted_out])
                .map_err(|e| PyValueError::new_err(format!("stat_swarm dodge concat: {e}")))?,
        });
    }

    combined_out.ok_or_else(|| PyValueError::new_err("stat_swarm: dodge produced no output"))
}

pub(crate) fn apply_with_context(
    spec: &SwarmSpec,
    batch: &RecordBatch,
    ctx: &TransformContext,
) -> PyResult<RecordBatch> {
    if let Some(dodge_field) = &spec.dodge {
        return apply_dodged(spec, batch, ctx, dodge_field);
    }
    // 1. Validate columns.
    let schema = batch.schema();
    let cat_idx = schema
        .index_of(&spec.category)
        .map_err(|_| PyValueError::new_err(format!(
            "stat_swarm: column '{}' not found", spec.category
        )))?;
    let cat_dt = schema.field(cat_idx).data_type().clone();
    if !is_groupby_supported_dtype(&cat_dt) {
        return Err(PyValueError::new_err(format!(
            "stat_swarm: category column '{}' has unsupported dtype {:?}; \
             supported: Utf8/LargeUtf8, Float64/Float32, \
             Int8/Int16/Int32/Int64, UInt8/UInt16/UInt32/UInt64, Boolean",
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
        .ok_or_else(|| PyValueError::new_err(format!(
            "stat_swarm: expected Float64Array for value column '{}'", spec.value)))?;

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
    // Vertical: value axis = y = panel height. Horizontal: value axis = x = panel width.
    let radius_data: f64 = match (spec.orient, ctx.panel_pixel_size) {
        (SwarmOrient::Vertical, Some((_, h))) if h > 0 => {
            let value_range = vmax - vmin;
            if value_range.is_finite() && value_range > 0.0 {
                (spec.point_size + spec.spacing) * (value_range / h as f64)
            } else {
                1.0
            }
        }
        (SwarmOrient::Horizontal, Some((w, _))) if w > 0 => {
            let value_range = vmax - vmin;
            if value_range.is_finite() && value_range > 0.0 {
                (spec.point_size + spec.spacing) * (value_range / w as f64)
            } else {
                1.0
            }
        }
        _ => {
            // No pixel context available (e.g. headless / test render without
            // a panel size). Fall back to radius_data = 1.0 so placement still
            // runs; callers that need accurate geometry must supply panel_pixel_size
            // via TransformContext.
            1.0
        }
    };

    // 3. Group by category. BTreeMap for deterministic iteration order.
    // FA-7: the shared group_key helper handles Utf8/Float64 and now also
    // int/uint/bool category columns. A null category key (KeyValue::Null) is
    // skipped below, matching the pre-existing "None category → skip row" rule.
    let cat_col = batch.column(cat_idx);
    let cat_keys: Vec<Option<KeyValue>> = (0..n)
        .map(|i| match groupby_key_at(cat_col.as_ref(), &cat_dt, i) {
            Some(KeyValue::Null) | None => None,
            Some(kv) => Some(kv),
        })
        .collect();

    let mut groups: BTreeMap<KeyValue, Vec<(usize, f64)>> = BTreeMap::new();
    for (i, cat_key) in cat_keys.iter().enumerate() {
        if val_arr.is_null(i) {
            continue;
        }
        let v = val_arr.value(i);
        if v.is_nan() {
            continue;
        }
        let key = match cat_key {
            None => continue,
            Some(kv) => kv.clone(),
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

    // 5. Build output: input columns + swarm_x + swarm_y + position-offset column.
    //
    // Vertical: emits `__pos_x_offset__` (cross-axis = x). The inverse of the
    // radius_data conversion: pixel_offset = swarm_x * (h / value_range).
    // Horizontal: emits `__pos_y_offset__` (cross-axis = y), using w instead of h.
    // Each placement slot (`radius_data` step) renders at exactly
    // `(point_size + spacing)` pixels — the visual spacing the algorithm intends.
    // For null offsets (skipped rows), emit 0.0 so the position-offset reader
    // never sees a None it doesn't expect (`render/position.rs` requires non-null).
    let pixel_per_data: f64 = match (spec.orient, ctx.panel_pixel_size) {
        (SwarmOrient::Vertical, Some((_, h))) if h > 0 => {
            let value_range = vmax - vmin;
            if value_range.is_finite() && value_range > 0.0 {
                (h as f64) / value_range
            } else {
                1.0
            }
        }
        (SwarmOrient::Horizontal, Some((w, _))) if w > 0 => {
            let value_range = vmax - vmin;
            if value_range.is_finite() && value_range > 0.0 {
                (w as f64) / value_range
            } else {
                1.0
            }
        }
        _ => 1.0,
    };
    let pos_offset: Float64Array = swarm_x
        .iter()
        .map(|opt| Some(opt.map(|o| o * pixel_per_data).unwrap_or(0.0)))
        .collect();

    let mut out_cols: Vec<ArrayRef> = batch.columns().to_vec();
    let sx_arr: Float64Array = swarm_x.iter().copied().collect();
    let sy_arr: Float64Array = swarm_y.iter().copied().collect();
    out_cols.push(Arc::new(sx_arr));
    out_cols.push(Arc::new(sy_arr));
    out_cols.push(Arc::new(pos_offset));

    let offset_col_name = match spec.orient {
        SwarmOrient::Vertical => "__pos_x_offset__",
        SwarmOrient::Horizontal => "__pos_y_offset__",
    };

    let mut new_fields: Vec<Field> = schema
        .fields()
        .iter()
        .map(|f| (**f).clone())
        .collect();
    new_fields.push(Field::new("swarm_x", DataType::Float64, true));
    new_fields.push(Field::new("swarm_y", DataType::Float64, true));
    new_fields.push(Field::new(offset_col_name, DataType::Float64, false));
    let out_schema = Arc::new(Schema::new(new_fields));

    RecordBatch::try_new(out_schema, out_cols)
        .map_err(|e| PyValueError::new_err(format!("stat_swarm: {e}")))
}

// ---------- PyO3 wrapper ----------

use pyo3::prelude::*;

use crate::transform::core::TransformSpec;

/// Beeswarm (strip-plot) offset computation for non-overlapping point layout.
///
/// Takes a categorical-axis column and a numeric value column and computes
/// per-point x-offsets (for a vertical beeswarm) so that circles of radius
/// ``point_size / 2`` do not overlap. Points are placed symmetrically
/// around the category centre, or on one side only when ``side`` is set.
///
/// Output: the original columns plus ``x_offset`` (Float64 per-point
/// horizontal displacement in data units).
///
/// Parameters
/// ----------
/// category : str
///     Categorical column that defines the swarm axis position.
/// value : str
///     Numeric column whose values are spread along the value axis.
/// point_size : float, default 5.0
///     Diameter of each point in data units. Must be > 0.
/// spacing : float, default 1.0
///     Minimum gap between point edges. Must be ≥ 0.
/// side : {"both", "left", "right"}, default "both"
///     Which side of the category centre to place displaced points.
/// name : str, optional
///     Named output label for sibling ``Reorder(from_=...)`` lookup.
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> fm.Chart(df).mark_swarm().encode(x="val", y="category")
#[pyclass(eq, module = "ferrum._core", name = "Swarm")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PySwarm(pub(crate) TransformSpec);

#[pymethods]
impl PySwarm {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        category, value, *,
        point_size = 5.0,
        spacing = 1.0,
        side = "both",
        orient = "vertical",
        dodge = None,
        name = None,
    ))]
    fn new(
        category: &str,
        value: &str,
        point_size: f64,
        spacing: f64,
        side: &str,
        orient: &str,
        dodge: Option<String>,
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
        let parsed_orient = match orient {
            "vertical" => SwarmOrient::Vertical,
            "horizontal" => SwarmOrient::Horizontal,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Swarm: unknown orient '{other}'; expected vertical|horizontal"
                )))
            }
        };
        Ok(PySwarm(TransformSpec::Swarm(SwarmSpec {
            category: category.to_string(),
            value: value.to_string(),
            point_size,
            spacing,
            side: parsed_side,
            orient: parsed_orient,
            dodge,
            name,
        })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Swarm(s) => format!(
                "Swarm(category='{}', value='{}', point_size={}, spacing={}, side={:?}, orient={:?}, name={:?})",
                s.category, s.value, s.point_size, s.spacing, s.side, s.orient, s.name
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
            orient: SwarmOrient::Vertical,
            name: None,
            dodge: None,
        };
        // Big panel → small radius.
        let out = apply_with_context(&spec, &b, &ctx_with_panel(400, 400)).unwrap();
        assert_eq!(out.num_rows(), 5);
        assert_eq!(out.num_columns(), 5); // cat, v, swarm_x, swarm_y, __pos_x_offset__

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
            orient: SwarmOrient::Vertical,
            name: None,
            dodge: None,
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
            orient: SwarmOrient::Vertical,
            name: None,
            dodge: None,
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
            orient: SwarmOrient::Vertical,
            name: None,
            dodge: None,
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
            orient: SwarmOrient::Vertical,
            name: None,
            dodge: None,
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

    #[test]
    fn swarm_horizontal_emits_pos_y_offset() {
        pyo3::Python::initialize();
        // Horizontal swarm: category on y, value on x.
        // Use clustered values so points must be displaced (offset != 0).
        let cats = vec!["A", "A", "A", "A", "A"];
        let vals = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let b = make_batch(cats, vals);
        let spec = SwarmSpec {
            category: "cat".into(),
            value: "v".into(),
            point_size: 5.0,
            spacing: 1.0,
            side: SwarmSide::Both,
            orient: SwarmOrient::Horizontal,
            name: None,
            dodge: None,
        };
        let out = apply_with_context(&spec, &b, &ctx_with_panel(400, 400)).unwrap();
        // Output must have 5 columns: cat, v, swarm_x, swarm_y, __pos_y_offset__
        assert_eq!(out.num_columns(), 5);
        // __pos_y_offset__ must be present; __pos_x_offset__ must NOT be present.
        assert!(
            out.schema().index_of("__pos_y_offset__").is_ok(),
            "horizontal swarm must emit __pos_y_offset__"
        );
        assert!(
            out.schema().index_of("__pos_x_offset__").is_err(),
            "horizontal swarm must NOT emit __pos_x_offset__"
        );
        // With all values equal, points compete for the same slot and must be displaced.
        let py_offset = col_f64(&out, "__pos_y_offset__");
        let has_nonzero = (0..py_offset.len()).any(|i| py_offset.value(i).abs() > 1e-12);
        assert!(has_nonzero, "horizontal swarm should have at least one non-zero offset");
    }
}
