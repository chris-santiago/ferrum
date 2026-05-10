//! Phase 9c — position-adjustment render pass.
//!
//! Rewrites a layer's RecordBatch *data values* (or injects synthetic offset
//! columns, for ordinal x) per the PositionAdjust on the layer. Runs AFTER
//! scale_resolve (so we know ordinal bandwidth or continuous-x median spacing)
//! but BEFORE mark drawing. The adjusted RecordBatch is then passed to
//! `draw::dispatch_mark` in place of the original.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};

use crate::render::scale_resolve::{ResolvedScales, ScaleKind};
use crate::spec::position::PositionAdjust;

/// Apply a position adjustment to a layer batch, returning a new batch with
/// rewritten coordinate columns (or, for ordinal-x Dodge / Jitter into bands,
/// with two synthetic `__pos_x_offset__` / `__pos_y_offset__` Float64 columns
/// appended). Returns a clone of the input unchanged if `position` is None
/// or Identity, or if the adjustment doesn't apply (e.g., Dodge with no
/// group channel set or ≤ 1 distinct groups).
pub(crate) fn apply_position(
    batch: &RecordBatch,
    position: Option<&PositionAdjust>,
    scales: &ResolvedScales,
    encoding: &crate::spec::encoding::Encoding,
) -> Result<RecordBatch, crate::render::RenderError> {
    let Some(p) = position else { return Ok(batch.clone()); };
    match p {
        PositionAdjust::Identity => Ok(batch.clone()),
        PositionAdjust::Dodge { by, padding } => {
            apply_dodge(batch, by.as_deref(), *padding, scales, encoding)
        }
        PositionAdjust::Jitter { axis, width, seed } => {
            apply_jitter(batch, axis, *width, *seed, scales, encoding)
        }
        PositionAdjust::Stack { by, offset } => {
            apply_stack(batch, by.as_deref(), offset, scales, encoding)
        }
    }
}

// ---------------------------------------------------------------------------
// Dodge
// ---------------------------------------------------------------------------

fn apply_dodge(
    batch: &RecordBatch,
    by_field: Option<&str>,
    padding: f64,
    scales: &ResolvedScales,
    encoding: &crate::spec::encoding::Encoding,
) -> Result<RecordBatch, crate::render::RenderError> {
    // Resolve the `by` column. Default to the color encoding's field if `by` is None.
    let by_col_name = match by_field {
        Some(s) => s.to_string(),
        None => match &encoding.color {
            Some(c) => c.field.clone(),
            None => return Ok(batch.clone()),
        },
    };
    let by_col_idx = match batch.schema().index_of(&by_col_name) {
        Ok(i) => i,
        Err(_) => return Ok(batch.clone()),
    };
    let by_arr = batch
        .column(by_col_idx)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            crate::render::RenderError::Other(format!(
                "Dodge: by-column '{by_col_name}' must be Utf8"
            ))
        })?;

    // Resolve x column (the axis being dodged).
    let x_field = encoding.x.as_ref().ok_or_else(|| {
        crate::render::RenderError::Other("Dodge: x encoding required".into())
    })?;
    let x_col_idx = batch.schema().index_of(&x_field.field).map_err(|_| {
        crate::render::RenderError::Other(format!(
            "Dodge: x column '{}' not found",
            x_field.field
        ))
    })?;
    let is_ordinal_x = batch.schema().field(x_col_idx).data_type() != &DataType::Float64;
    if is_ordinal_x {
        return apply_dodge_ordinal(batch, by_arr, padding, scales);
    }
    let x_arr = batch
        .column(x_col_idx)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| {
            crate::render::RenderError::Other("Dodge: x must be Float64".into())
        })?;

    // 1. Compute median spacing of unique x values (bandwidth proxy for continuous x).
    let mut uniques: Vec<f64> = (0..x_arr.len())
        .filter(|i| !x_arr.is_null(*i))
        .map(|i| x_arr.value(i))
        .collect();
    uniques.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    uniques.dedup();
    if uniques.len() < 2 {
        return Ok(batch.clone());
    }
    let mut diffs: Vec<f64> = uniques.windows(2).map(|w| w[1] - w[0]).collect();
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let bandwidth = diffs[diffs.len() / 2];

    // 2. Determine group order from `by` channel (first-appearance order).
    let mut groups_in_order: Vec<String> = Vec::new();
    let mut seen: HashMap<String, usize> = HashMap::new();
    for i in 0..by_arr.len() {
        let g = by_arr.value(i).to_string();
        if !seen.contains_key(&g) {
            seen.insert(g.clone(), groups_in_order.len());
            groups_in_order.push(g);
        }
    }
    let n_groups = groups_in_order.len();
    if n_groups <= 1 {
        return Ok(batch.clone());
    }

    let pad_total = bandwidth * padding * 2.0;
    let sub_band = (bandwidth - pad_total) / n_groups as f64;

    let mut new_x = Vec::with_capacity(x_arr.len());
    for i in 0..x_arr.len() {
        let g = by_arr.value(i);
        let group_idx = *seen.get(g).unwrap();
        let offset =
            -bandwidth / 2.0 + bandwidth * padding + sub_band * (group_idx as f64 + 0.5);
        new_x.push(x_arr.value(i) + offset);
    }

    let mut cols: Vec<ArrayRef> = batch.columns().to_vec();
    cols[x_col_idx] = Arc::new(Float64Array::from(new_x));
    let schema = batch.schema();
    RecordBatch::try_new(schema, cols)
        .map_err(|e| crate::render::RenderError::Other(format!("Dodge: {e}")))
}

/// Ordinal-x Dodge — operates in pixel space because the categorical x cannot
/// be rewritten in data space. Injects two synthetic Float64 columns named
/// `__pos_x_offset__` and `__pos_y_offset__` (the latter is always 0 for Dodge).
/// Mark drawers (bar/point/box/swarm/violin/errorbar/errorband/ribbon) read
/// these columns post-scale-resolve and add them to the rendered position.
fn apply_dodge_ordinal(
    batch: &RecordBatch,
    by_arr: &StringArray,
    padding: f64,
    scales: &ResolvedScales,
) -> Result<RecordBatch, crate::render::RenderError> {
    let schema = batch.schema();
    let bandwidth_px = match &scales.x {
        ScaleKind::Ordinal(s) => s.bandwidth(),
        _ => return Ok(batch.clone()),
    };

    let mut group_order: Vec<String> = Vec::new();
    let mut group_idx: HashMap<String, usize> = HashMap::new();
    for i in 0..by_arr.len() {
        let g = by_arr.value(i).to_string();
        if !group_idx.contains_key(&g) {
            group_idx.insert(g.clone(), group_order.len());
            group_order.push(g);
        }
    }
    let n_groups = group_order.len();
    if n_groups <= 1 {
        return Ok(batch.clone());
    }

    let pad_total = bandwidth_px * padding * 2.0;
    let sub_band = (bandwidth_px - pad_total) / n_groups as f64;

    let n = by_arr.len();
    let mut x_offsets: Vec<f64> = Vec::with_capacity(n);
    let mut y_offsets: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let g = by_arr.value(i);
        let gi = *group_idx.get(g).unwrap();
        let off = -bandwidth_px / 2.0 + bandwidth_px * padding + sub_band * (gi as f64 + 0.5);
        x_offsets.push(off);
        y_offsets.push(0.0);
    }

    let mut cols: Vec<ArrayRef> = batch.columns().to_vec();
    cols.push(Arc::new(Float64Array::from(x_offsets)));
    cols.push(Arc::new(Float64Array::from(y_offsets)));

    let mut fields: Vec<Field> = schema.fields().iter().map(|f| f.as_ref().clone()).collect();
    fields.push(Field::new("__pos_x_offset__", DataType::Float64, false));
    fields.push(Field::new("__pos_y_offset__", DataType::Float64, false));
    let new_schema = Arc::new(Schema::new(fields));

    RecordBatch::try_new(new_schema, cols)
        .map_err(|e| crate::render::RenderError::Other(format!("Dodge ordinal: {e}")))
}

// ---------------------------------------------------------------------------
// Jitter
// ---------------------------------------------------------------------------

fn apply_jitter(
    batch: &RecordBatch,
    axis: &crate::spec::position::JitterAxis,
    width: f64,
    seed: Option<u64>,
    _scales: &ResolvedScales,
    encoding: &crate::spec::encoding::Encoding,
) -> Result<RecordBatch, crate::render::RenderError> {
    use crate::spec::position::JitterAxis;
    use rand::{RngCore, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use twox_hash::xxh3;

    let x_idx = encoding
        .x
        .as_ref()
        .and_then(|e| batch.schema().index_of(&e.field).ok());
    let y_idx = encoding
        .y
        .as_ref()
        .and_then(|e| batch.schema().index_of(&e.field).ok());

    let do_x = matches!(axis, JitterAxis::X | JitterAxis::Both);
    let do_y = matches!(axis, JitterAxis::Y | JitterAxis::Both);

    let x_arr = x_idx.and_then(|j| batch.column(j).as_any().downcast_ref::<Float64Array>());
    let y_arr = y_idx.and_then(|j| batch.column(j).as_any().downcast_ref::<Float64Array>());

    let n = batch.num_rows();
    let mut new_x: Vec<f64> = Vec::with_capacity(n);
    let mut new_y: Vec<f64> = Vec::with_capacity(n);

    for i in 0..n {
        let xv = x_arr.map(|a| if a.is_null(i) { f64::NAN } else { a.value(i) }).unwrap_or(f64::NAN);
        let yv = y_arr.map(|a| if a.is_null(i) { f64::NAN } else { a.value(i) }).unwrap_or(f64::NAN);

        let row_seed = match seed {
            Some(s) => s.wrapping_add(i as u64),
            None => {
                let key = format!("{xv}|{yv}");
                xxh3::hash64(key.as_bytes())
            }
        };
        let mut rng = ChaCha8Rng::seed_from_u64(row_seed);
        let u = (rng.next_u64() as f64) / (u64::MAX as f64);
        let noise_x = (u - 0.5) * width;
        let u2 = (rng.next_u64() as f64) / (u64::MAX as f64);
        let noise_y = (u2 - 0.5) * width;

        new_x.push(if do_x { xv + noise_x } else { xv });
        new_y.push(if do_y { yv + noise_y } else { yv });
    }

    let mut cols: Vec<ArrayRef> = batch.columns().to_vec();
    if let Some(j) = x_idx {
        if do_x {
            cols[j] = Arc::new(Float64Array::from(new_x));
        }
    }
    if let Some(j) = y_idx {
        if do_y {
            cols[j] = Arc::new(Float64Array::from(new_y));
        }
    }
    let schema = batch.schema();
    RecordBatch::try_new(schema, cols)
        .map_err(|e| crate::render::RenderError::Other(format!("Jitter: {e}")))
}

// ---------------------------------------------------------------------------
// Stack
// ---------------------------------------------------------------------------

fn apply_stack(
    batch: &RecordBatch,
    by_field: Option<&str>,
    offset: &crate::spec::position::StackOffset,
    _scales: &ResolvedScales,
    encoding: &crate::spec::encoding::Encoding,
) -> Result<RecordBatch, crate::render::RenderError> {
    use crate::spec::position::StackOffset;
    use std::collections::BTreeMap;

    let by_name = match by_field {
        Some(s) => s.to_string(),
        None => match &encoding.color {
            Some(c) => c.field.clone(),
            None => return Ok(batch.clone()),
        },
    };
    let by_idx = batch.schema().index_of(&by_name).ok();
    let by_arr_opt =
        by_idx.and_then(|i| batch.column(i).as_any().downcast_ref::<StringArray>());
    let Some(by_arr) = by_arr_opt else { return Ok(batch.clone()); };

    let x_field = encoding.x.as_ref().ok_or_else(|| {
        crate::render::RenderError::Other("Stack: x encoding required".into())
    })?;
    let y_field = encoding.y.as_ref().ok_or_else(|| {
        crate::render::RenderError::Other("Stack: y encoding required".into())
    })?;
    let xi = batch.schema().index_of(&x_field.field).map_err(|_| {
        crate::render::RenderError::Other(format!(
            "Stack: x col '{}' not found",
            x_field.field
        ))
    })?;
    let yi = batch.schema().index_of(&y_field.field).map_err(|_| {
        crate::render::RenderError::Other(format!(
            "Stack: y col '{}' not found",
            y_field.field
        ))
    })?;
    let ya = batch
        .column(yi)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| crate::render::RenderError::Other("Stack: y must be Float64".into()))?;

    // x may be Float64 (continuous) or Utf8 (ordinal). Build a stable u64 key
    // for the BTreeMap from either case.
    let x_col = batch.column(xi);
    let x_keys: Vec<u64> = if x_col.data_type() == &DataType::Float64 {
        let xa = x_col.as_any().downcast_ref::<Float64Array>().unwrap();
        (0..xa.len()).map(|i| xa.value(i).to_bits()).collect()
    } else if let Some(xs) = x_col.as_any().downcast_ref::<StringArray>() {
        // Stable hash of the string for grouping; we never decode back, only bin.
        use twox_hash::xxh3;
        (0..xs.len())
            .map(|i| xxh3::hash64(xs.value(i).as_bytes()))
            .collect()
    } else {
        return Err(crate::render::RenderError::Other(
            "Stack: x column must be Float64 or Utf8".into(),
        ));
    };

    // Group order from `by` channel (first-appearance).
    let mut group_idx_map: HashMap<String, usize> = HashMap::new();
    let mut group_order: Vec<String> = Vec::new();
    for i in 0..ya.len() {
        let g = by_arr.value(i).to_string();
        if !group_idx_map.contains_key(&g) {
            group_idx_map.insert(g.clone(), group_order.len());
            group_order.push(g);
        }
    }

    // bins: x_key → Vec<(group_idx, row_idx, y)>
    let mut bins: BTreeMap<u64, Vec<(usize, usize, f64)>> = BTreeMap::new();
    for i in 0..ya.len() {
        let g = by_arr.value(i).to_string();
        let gi = *group_idx_map.get(&g).unwrap();
        let yv = if ya.is_null(i) { 0.0 } else { ya.value(i) };
        bins.entry(x_keys[i]).or_default().push((gi, i, yv));
    }

    let totals: HashMap<u64, f64> = bins
        .iter()
        .map(|(k, rows)| (*k, rows.iter().map(|(_, _, y)| y).sum::<f64>()))
        .collect();

    let mut new_y = vec![0.0_f64; ya.len()];
    for (xkey, rows) in bins.iter_mut() {
        rows.sort_by_key(|(gi, _, _)| *gi);
        let total = totals.get(xkey).copied().unwrap_or(0.0);
        let mut acc = 0.0_f64;
        for (_, row_idx, y) in rows.iter() {
            let normalized = match offset {
                StackOffset::Zero => *y,
                StackOffset::Normalize => {
                    if total != 0.0 {
                        y / total
                    } else {
                        0.0
                    }
                }
                StackOffset::Center => *y,
            };
            acc += normalized;
            new_y[*row_idx] = acc;
        }
        if matches!(offset, StackOffset::Center) {
            let mid = acc / 2.0;
            for (_, row_idx, _) in rows.iter() {
                new_y[*row_idx] -= mid;
            }
        }
    }

    let mut cols: Vec<ArrayRef> = batch.columns().to_vec();
    cols[yi] = Arc::new(Float64Array::from(new_y));
    let schema = batch.schema();
    RecordBatch::try_new(schema, cols)
        .map_err(|e| crate::render::RenderError::Other(format!("Stack: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::position::{JitterAxis, PositionAdjust, StackOffset};
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn dummy_scales() -> ResolvedScales {
        // Construct a minimal ResolvedScales — Linear x/y with [0,100] domain & range.
        use crate::scale::linear::LinearScale;
        let lx = LinearScale::new_internal(vec![0.0, 10.0], vec![0.0, 100.0], false, false);
        let ly = LinearScale::new_internal(vec![0.0, 100.0], vec![0.0, 100.0], false, false);
        ResolvedScales {
            x: ScaleKind::Linear(lx),
            y: ScaleKind::Linear(ly),
            color: None,
            size: None,
            shape: None,
            opacity: None,
            x2: None,
            y2: None,
        }
    }

    fn enc_xy(xf: &str, yf: &str, color: Option<&str>) -> Encoding {
        Encoding {
            x: Some(EncodingSpec { field: xf.into(), type_: None, ..Default::default() }),
            y: Some(EncodingSpec { field: yf.into(), type_: None, ..Default::default() }),
            color: color.map(|c| EncodingSpec {
                field: c.into(),
                type_: None,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn batch_xyg() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 1.0, 2.0, 2.0])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0])),
                Arc::new(StringArray::from(vec!["a", "b", "a", "b"])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn identity_returns_clone() {
        let b = batch_xyg();
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let out = apply_position(&b, Some(&PositionAdjust::Identity), &s, &enc).unwrap();
        assert_eq!(out.num_rows(), b.num_rows());
        assert_eq!(out.num_columns(), b.num_columns());
    }

    #[test]
    fn none_position_returns_clone() {
        let b = batch_xyg();
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let out = apply_position(&b, None, &s, &enc).unwrap();
        assert_eq!(out.num_rows(), b.num_rows());
    }

    #[test]
    fn dodge_continuous_x_rewrites_x_column() {
        let b = batch_xyg();
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Dodge { by: Some("g".into()), padding: 0.0 };
        let out = apply_position(&b, Some(&pos), &s, &enc).unwrap();
        // Two unique x values: 1.0, 2.0 → bandwidth = 1.0.
        // Two groups (a, b) → sub_band = 0.5; offsets a=-0.25, b=+0.25.
        let xa = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert!((xa.value(0) - 0.75).abs() < 1e-9, "row0 x={}", xa.value(0));
        assert!((xa.value(1) - 1.25).abs() < 1e-9, "row1 x={}", xa.value(1));
        assert!((xa.value(2) - 1.75).abs() < 1e-9, "row2 x={}", xa.value(2));
        assert!((xa.value(3) - 2.25).abs() < 1e-9, "row3 x={}", xa.value(3));
    }

    #[test]
    fn dodge_single_group_is_noop() {
        // All rows in group "a" → n_groups == 1 → return clone.
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        let b = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
                Arc::new(Float64Array::from(vec![10.0, 20.0])),
                Arc::new(StringArray::from(vec!["a", "a"])),
            ],
        )
        .unwrap();
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Dodge { by: Some("g".into()), padding: 0.05 };
        let out = apply_position(&b, Some(&pos), &s, &enc).unwrap();
        let xa = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(xa.value(0), 1.0);
        assert_eq!(xa.value(1), 2.0);
    }

    #[test]
    fn jitter_explicit_seed_deterministic() {
        let b = batch_xyg();
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Jitter { axis: JitterAxis::X, width: 0.5, seed: Some(42) };
        let a = apply_position(&b, Some(&pos), &s, &enc).unwrap();
        let bb = apply_position(&b, Some(&pos), &s, &enc).unwrap();
        let ax = a.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        let bx = bb.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        for i in 0..4 {
            assert_eq!(ax.value(i).to_bits(), bx.value(i).to_bits());
        }
    }

    #[test]
    fn jitter_none_seed_is_deterministic_via_hash() {
        let b = batch_xyg();
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Jitter { axis: JitterAxis::X, width: 0.5, seed: None };
        let a = apply_position(&b, Some(&pos), &s, &enc).unwrap();
        let bb = apply_position(&b, Some(&pos), &s, &enc).unwrap();
        let ax = a.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        let bx = bb.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        for i in 0..4 {
            assert_eq!(ax.value(i).to_bits(), bx.value(i).to_bits());
        }
    }

    #[test]
    fn stack_zero_accumulates_y() {
        let b = batch_xyg();
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Stack { by: Some("g".into()), offset: StackOffset::Zero };
        let out = apply_position(&b, Some(&pos), &s, &enc).unwrap();
        let ya = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        // Group order: a=0, b=1. At x=1: a=10 → 10, b=20 → 30. At x=2: a=30 → 30, b=40 → 70.
        assert_eq!(ya.value(0), 10.0);
        assert_eq!(ya.value(1), 30.0);
        assert_eq!(ya.value(2), 30.0);
        assert_eq!(ya.value(3), 70.0);
    }

    #[test]
    fn stack_normalize_sums_to_one_per_x() {
        let b = batch_xyg();
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Stack {
            by: Some("g".into()),
            offset: StackOffset::Normalize,
        };
        let out = apply_position(&b, Some(&pos), &s, &enc).unwrap();
        let ya = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        // For each x bin the top of the highest stack should be 1.0.
        // x=1: top group (b) reaches 1.0 → row 1.
        // x=2: top group (b) reaches 1.0 → row 3.
        assert!((ya.value(1) - 1.0).abs() < 1e-9);
        assert!((ya.value(3) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn stack_center_symmetric_around_zero() {
        let b = batch_xyg();
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Stack { by: Some("g".into()), offset: StackOffset::Center };
        let out = apply_position(&b, Some(&pos), &s, &enc).unwrap();
        let ya = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        // x=1: total=30, mid=15. a row goes 0..10 → top at 10-15=-5.
        // b row goes 10..30 → top at 30-15=15.
        assert!((ya.value(0) + 5.0).abs() < 1e-9);
        assert!((ya.value(1) - 15.0).abs() < 1e-9);
    }
}

/// Read per-row pixel offsets from synthetic `__pos_x_offset__` /
/// `__pos_y_offset__` columns. Returns `(Vec<f64>, Vec<f64>)` of zeros-by-
/// default when the columns are absent. Mark drawers call this near the top
/// of `draw()` and add the per-row offset to their resolved pixel position
/// before emitting SVG.
pub(crate) fn read_position_offsets(batch: &RecordBatch) -> (Vec<f64>, Vec<f64>) {
    let n = batch.num_rows();
    let xo = batch
        .schema()
        .index_of("__pos_x_offset__")
        .ok()
        .and_then(|i| {
            batch.column(i).as_any().downcast_ref::<Float64Array>().map(|a| {
                (0..a.len()).map(|j| a.value(j)).collect::<Vec<f64>>()
            })
        })
        .unwrap_or_else(|| vec![0.0; n]);
    let yo = batch
        .schema()
        .index_of("__pos_y_offset__")
        .ok()
        .and_then(|i| {
            batch.column(i).as_any().downcast_ref::<Float64Array>().map(|a| {
                (0..a.len()).map(|j| a.value(j)).collect::<Vec<f64>>()
            })
        })
        .unwrap_or_else(|| vec![0.0; n]);
    (xo, yo)
}
