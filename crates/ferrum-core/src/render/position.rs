//! Phase 9c — position-adjustment render pass.
//!
//! Rewrites a layer's RecordBatch *data values* (or injects synthetic offset
//! columns, for ordinal x) per the PositionAdjust on the layer. Runs AFTER
//! scale_resolve (so we know ordinal bandwidth or continuous-x median spacing)
//! but BEFORE mark drawing. The adjusted RecordBatch is then passed to
//! `draw::dispatch_mark` in place of the original.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};

use crate::render::scale_resolve::{ResolvedScales, ScaleKind};
use crate::spec::chart::ChartSpec;
use crate::spec::position::{PositionAdjust, StackOffset};

/// Return the batch the y-scale should resolve against, accounting for a
/// Stack position adjustment.
///
/// When a layer (or the single-layer spec) carries a `Stack` adjustment
/// whose encoded y matches `y_field`, the rendered y values are the
/// *cumulative* values from `apply_stack`, not the original column.
/// Resolving the y-scale from the raw batch would clip stacked tops
/// outside the domain — `LinearScale` returns NaN for out-of-domain
/// inputs and `bar.rs` drops every row whose top falls past it. Returning
/// the post-stack batch here keeps stacked bars visible.
///
/// Borrowed `primary_batch` is returned when no Stack matches; the owned
/// stacked batch is returned (boxed by `Cow`) otherwise. On a stack
/// failure the caller's primary batch is returned — the scale resolves
/// from raw data and the downstream `apply_stack` re-attempt during
/// drawing will surface the same error to the user.
///
/// Pre-F15 this logic lived in `scale_resolve::resolve_scales_with_outputs`
/// via a private `find_stack_for_y` helper. The Stack handling belongs
/// alongside the other Stack code; scale resolution shouldn't have to
/// know which specific position adjustment is in play.
pub(crate) fn axis_batch_for_y<'a>(
    spec: &'a ChartSpec,
    y_field: &str,
    primary_batch: &'a RecordBatch,
) -> Cow<'a, RecordBatch> {
    let Some((by, offset, layer_enc, mark)) = find_stack_for_y(spec, y_field) else {
        return Cow::Borrowed(primary_batch);
    };
    match apply_stack(primary_batch, by, offset, layer_enc, mark) {
        Ok(b) => Cow::Owned(b),
        Err(_) => Cow::Borrowed(primary_batch),
    }
}

/// Find the first Stack position adjustment in the spec whose layer (or
/// the chart itself, in the single-layer case) encodes the given y-field.
/// Multi-Stack layers are not merged here — the first match wins.
fn find_stack_for_y<'a>(
    spec: &'a ChartSpec,
    y_field: &str,
) -> Option<(
    Option<&'a str>,
    &'a StackOffset,
    &'a crate::spec::encoding::Encoding,
    crate::spec::mark::Mark,
)> {
    if let Some(layers) = spec.layers.as_ref() {
        for layer in layers {
            let layer_y = layer
                .encoding
                .y
                .as_ref()
                .map(|e| e.field.as_str())
                .or_else(|| spec.encoding.y.as_ref().map(|e| e.field.as_str()));
            if layer_y != Some(y_field) {
                continue;
            }
            if let Some(PositionAdjust::Stack { by, offset }) =
                layer.position.as_ref().or(spec.position.as_ref())
            {
                return Some((by.as_deref(), offset, &layer.encoding, layer.mark));
            }
        }
    }
    if let Some(PositionAdjust::Stack { by, offset }) = spec.position.as_ref() {
        let spec_y = spec.encoding.y.as_ref().map(|e| e.field.as_str());
        if spec_y == Some(y_field) {
            return Some((by.as_deref(), offset, &spec.encoding, spec.mark));
        }
    }
    None
}

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
    mark: crate::spec::mark::Mark,
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
            apply_stack(batch, by.as_deref(), offset, encoding, mark)
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
            crate::render::RenderError::PositionAdjustFailed { adjustment: "Dodge", reason: format!("by-column '{by_col_name}' must be Utf8") }
        })?;

    // Resolve x column (the axis being dodged).
    let x_field = encoding.x.as_ref().ok_or_else(|| {
        crate::render::RenderError::PositionAdjustFailed { adjustment: "Dodge", reason: "x encoding required".into() }
    })?;
    let x_col_idx = batch.schema().index_of(&x_field.field).map_err(|_| {
        crate::render::RenderError::PositionAdjustFailed { adjustment: "Dodge", reason: format!("x column '{}' not found",
            x_field.field) }
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
            crate::render::RenderError::PositionAdjustFailed { adjustment: "Dodge", reason: "x must be Float64".into() }
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
        .map_err(|e| crate::render::RenderError::PositionAdjustFailed { adjustment: "Dodge", reason: format!("{e}") })
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
        .map_err(|e| crate::render::RenderError::PositionAdjustFailed { adjustment: "Dodge", reason: format!("ordinal: {e}") })
}

// ---------------------------------------------------------------------------
// Jitter
// ---------------------------------------------------------------------------

fn apply_jitter(
    batch: &RecordBatch,
    axis: &crate::spec::position::JitterAxis,
    width: f64,
    seed: Option<u64>,
    scales: &ResolvedScales,
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

    // Per-axis ordinality check — for ordinal axes we must NOT overwrite the
    // string column with float noise. Instead we emit a pixel-offset column
    // (`__pos_x_offset__` / `__pos_y_offset__`) that the mark renderers add
    // post-scale. The pixel offset is `(u - 0.5) * width * bandwidth_px`, so
    // `width=1.0` spans the full band; `width=0.4` (default) keeps points
    // well within their band.
    let x_is_ordinal = matches!(scales.x, ScaleKind::Ordinal(_));
    let y_is_ordinal = matches!(scales.y, ScaleKind::Ordinal(_));
    let x_bandwidth = if let ScaleKind::Ordinal(s) = &scales.x { s.bandwidth() } else { 1.0 };
    let y_bandwidth = if let ScaleKind::Ordinal(s) = &scales.y { s.bandwidth() } else { 1.0 };

    let x_arr = x_idx.and_then(|j| batch.column(j).as_any().downcast_ref::<Float64Array>());
    let y_arr = y_idx.and_then(|j| batch.column(j).as_any().downcast_ref::<Float64Array>());

    let n = batch.num_rows();
    let mut new_x: Vec<f64> = Vec::with_capacity(n);
    let mut new_y: Vec<f64> = Vec::with_capacity(n);
    let mut x_pixel_offsets: Vec<f64> = Vec::with_capacity(n);
    let mut y_pixel_offsets: Vec<f64> = Vec::with_capacity(n);

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
        let noise_x_data = (u - 0.5) * width;
        let u2 = (rng.next_u64() as f64) / (u64::MAX as f64);
        let noise_y_data = (u2 - 0.5) * width;

        new_x.push(if do_x && !x_is_ordinal { xv + noise_x_data } else { xv });
        new_y.push(if do_y && !y_is_ordinal { yv + noise_y_data } else { yv });
        x_pixel_offsets.push(if do_x && x_is_ordinal { (u - 0.5) * width * x_bandwidth } else { 0.0 });
        y_pixel_offsets.push(if do_y && y_is_ordinal { (u2 - 0.5) * width * y_bandwidth } else { 0.0 });
    }

    let mut cols: Vec<ArrayRef> = batch.columns().to_vec();
    if let (Some(j), true) = (x_idx, do_x && !x_is_ordinal) {
        cols[j] = Arc::new(Float64Array::from(new_x));
    }
    if let (Some(j), true) = (y_idx, do_y && !y_is_ordinal) {
        cols[j] = Arc::new(Float64Array::from(new_y));
    }

    let need_offsets = (do_x && x_is_ordinal) || (do_y && y_is_ordinal);
    if !need_offsets {
        let schema = batch.schema();
        return RecordBatch::try_new(schema, cols)
            .map_err(|e| crate::render::RenderError::PositionAdjustFailed { adjustment: "Jitter", reason: format!("{e}") });
    }

    cols.push(Arc::new(Float64Array::from(x_pixel_offsets)));
    cols.push(Arc::new(Float64Array::from(y_pixel_offsets)));
    let mut fields: Vec<Field> = batch.schema().fields().iter().map(|f| f.as_ref().clone()).collect();
    fields.push(Field::new("__pos_x_offset__", DataType::Float64, false));
    fields.push(Field::new("__pos_y_offset__", DataType::Float64, false));
    let new_schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(new_schema, cols)
        .map_err(|e| crate::render::RenderError::PositionAdjustFailed { adjustment: "Jitter", reason: format!("ordinal: {e}") })
}

// ---------------------------------------------------------------------------
// Stack
// ---------------------------------------------------------------------------

/// Position-adjust a layer's batch for a stacked layout.
///
/// Computes per-row segment bounds within each x-bin and writes them
/// back to the batch as the y column (top) plus a synthetic
/// ``__stack_y_base__`` column. The y output depends on the calling
/// ``mark``:
///
/// - ``Bar``, ``Area``, ``Ribbon``, ``Rect`` — y = top of segment, so
///   the renderer draws a rect from ``__stack_y_base__`` to y.
/// - ``Text``, ``Point``, ``Rule``, ``Tick`` — y = midpoint of segment
///   (Schwabish SB-followup 2026-05-12), so an annotation lands at the
///   visual centre of the stacked-bar segment for the same row. This
///   is what enables ``mark_text`` segment labels on stacked bars (e.g.
///   ``class_prediction_error_chart(show_counts=True)``) without
///   duplicating the cumsum in Python.
/// - Other marks — fall back to the segment-top output (unchanged).
pub(crate) fn apply_stack(
    batch: &RecordBatch,
    by_field: Option<&str>,
    offset: &crate::spec::position::StackOffset,
    encoding: &crate::spec::encoding::Encoding,
    mark: crate::spec::mark::Mark,
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
        crate::render::RenderError::PositionAdjustFailed { adjustment: "Stack", reason: "x encoding required".into() }
    })?;
    let y_field = encoding.y.as_ref().ok_or_else(|| {
        crate::render::RenderError::PositionAdjustFailed { adjustment: "Stack", reason: "y encoding required".into() }
    })?;
    let xi = batch.schema().index_of(&x_field.field).map_err(|_| {
        crate::render::RenderError::PositionAdjustFailed { adjustment: "Stack", reason: format!("x col '{}' not found",
            x_field.field) }
    })?;
    let yi = batch.schema().index_of(&y_field.field).map_err(|_| {
        crate::render::RenderError::PositionAdjustFailed { adjustment: "Stack", reason: format!("y col '{}' not found",
            y_field.field) }
    })?;
    // Stack accepts Float64 directly; for UInt64 (e.g. Bin's `count` column),
    // we transparently widen to f64 so stacked histograms over Bin's groupby
    // output work without an explicit cast.
    let y_col = batch.column(yi);
    let ya_vals: Vec<f64> = if let Some(a) = y_col.as_any().downcast_ref::<Float64Array>() {
        (0..a.len()).map(|i| if a.is_null(i) { 0.0 } else { a.value(i) }).collect()
    } else if let Some(a) = y_col.as_any().downcast_ref::<arrow::array::UInt64Array>() {
        (0..a.len()).map(|i| if a.is_null(i) { 0.0 } else { a.value(i) as f64 }).collect()
    } else {
        return Err(crate::render::RenderError::PositionAdjustFailed {
            adjustment: "Stack",
            reason: format!("y must be Float64 or UInt64; got {:?}", y_col.data_type()),
        });
    };
    let ya_len = ya_vals.len();

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
        return Err(crate::render::RenderError::PositionAdjustFailed {
            adjustment: "Stack",
            reason: "x column must be Float64 or Utf8".into(),
        });
    };

    // Group order from `by` channel (first-appearance).
    let mut group_idx_map: HashMap<String, usize> = HashMap::new();
    let mut group_order: Vec<String> = Vec::new();
    for i in 0..ya_len {
        let g = by_arr.value(i).to_string();
        if !group_idx_map.contains_key(&g) {
            group_idx_map.insert(g.clone(), group_order.len());
            group_order.push(g);
        }
    }

    // bins: x_key → Vec<(group_idx, row_idx, y)>
    let mut bins: BTreeMap<u64, Vec<(usize, usize, f64)>> = BTreeMap::new();
    for i in 0..ya_len {
        let g = by_arr.value(i).to_string();
        let gi = *group_idx_map.get(&g).unwrap();
        bins.entry(x_keys[i]).or_default().push((gi, i, ya_vals[i]));
    }

    let totals: HashMap<u64, f64> = bins
        .iter()
        .map(|(k, rows)| (*k, rows.iter().map(|(_, _, y)| y).sum::<f64>()))
        .collect();

    // new_y holds the cumulative TOP of each segment; new_y_base holds the
    // cumulative BOTTOM (the previous segment's top within the same bin, or
    // 0 / -mid for the first segment). Bar / area renderers draw each
    // segment from new_y_base[i] to new_y[i] so segments don't overlap.
    let mut new_y = vec![0.0_f64; ya_len];
    let mut new_y_base = vec![0.0_f64; ya_len];
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
            new_y_base[*row_idx] = acc;
            acc += normalized;
            new_y[*row_idx] = acc;
        }
        if matches!(offset, StackOffset::Center) {
            let mid = acc / 2.0;
            for (_, row_idx, _) in rows.iter() {
                new_y[*row_idx] -= mid;
                new_y_base[*row_idx] -= mid;
            }
        }
    }

    // Schwabish SB-followup (2026-05-12): annotation-style marks
    // (text/point/rule/tick) on a stacked layer land at the visual
    // CENTRE of each segment. Rect-style marks (bar/area/ribbon/rect)
    // keep the segment-top semantics so they draw from base → top.
    use crate::spec::mark::Mark;
    let y_output: Vec<f64> = match mark {
        Mark::Text | Mark::Point | Mark::Rule | Mark::Tick => (0..ya_len)
            .map(|i| 0.5 * (new_y[i] + new_y_base[i]))
            .collect(),
        _ => new_y.clone(),
    };

    let mut cols: Vec<ArrayRef> = batch.columns().to_vec();
    cols[yi] = Arc::new(Float64Array::from(y_output));
    // Rebuild schema with the y column promoted to Float64 (in case the
    // input was UInt64, e.g. Bin's `count` column).
    let mut new_fields: Vec<Field> = batch
        .schema()
        .fields()
        .iter()
        .map(|f| f.as_ref().clone())
        .collect();
    new_fields[yi] = Field::new(new_fields[yi].name(), DataType::Float64, true);

    // Append a synthetic __stack_y_base__ column so mark drawers can emit
    // per-segment rects (base → top) instead of drawing every segment from
    // y=0. Bar / area renderers look this up via `col_as_f64` when present.
    cols.push(Arc::new(Float64Array::from(new_y_base)));
    new_fields.push(Field::new("__stack_y_base__", DataType::Float64, true));

    let new_schema = Arc::new(Schema::new(new_fields));
    RecordBatch::try_new(new_schema, cols)
        .map_err(|e| crate::render::RenderError::PositionAdjustFailed { adjustment: "Stack", reason: format!("{e}") })
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
        let out = apply_position(&b, Some(&PositionAdjust::Identity), &s, &enc, crate::spec::mark::Mark::Bar).unwrap();
        assert_eq!(out.num_rows(), b.num_rows());
        assert_eq!(out.num_columns(), b.num_columns());
    }

    #[test]
    fn none_position_returns_clone() {
        let b = batch_xyg();
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let out = apply_position(&b, None, &s, &enc, crate::spec::mark::Mark::Bar).unwrap();
        assert_eq!(out.num_rows(), b.num_rows());
    }

    #[test]
    fn dodge_continuous_x_rewrites_x_column() {
        let b = batch_xyg();
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Dodge { by: Some("g".into()), padding: 0.0 };
        let out = apply_position(&b, Some(&pos), &s, &enc, crate::spec::mark::Mark::Bar).unwrap();
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
        let out = apply_position(&b, Some(&pos), &s, &enc, crate::spec::mark::Mark::Bar).unwrap();
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
        let a = apply_position(&b, Some(&pos), &s, &enc, crate::spec::mark::Mark::Bar).unwrap();
        let bb = apply_position(&b, Some(&pos), &s, &enc, crate::spec::mark::Mark::Bar).unwrap();
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
        let a = apply_position(&b, Some(&pos), &s, &enc, crate::spec::mark::Mark::Bar).unwrap();
        let bb = apply_position(&b, Some(&pos), &s, &enc, crate::spec::mark::Mark::Bar).unwrap();
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
        let out = apply_position(&b, Some(&pos), &s, &enc, crate::spec::mark::Mark::Bar).unwrap();
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
        let out = apply_position(&b, Some(&pos), &s, &enc, crate::spec::mark::Mark::Bar).unwrap();
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
        let out = apply_position(&b, Some(&pos), &s, &enc, crate::spec::mark::Mark::Bar).unwrap();
        let ya = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        // x=1: total=30, mid=15. a row goes 0..10 → top at 10-15=-5.
        // b row goes 10..30 → top at 30-15=15.
        assert!((ya.value(0) + 5.0).abs() < 1e-9);
        assert!((ya.value(1) - 15.0).abs() < 1e-9);
    }

    #[test]
    fn stack_text_emits_segment_midpoint_y() {
        // Schwabish SB-followup (2026-05-12): annotation-style marks
        // (text/point/rule/tick) on a stacked layer land at the
        // visual centre of each segment instead of the top.
        let b = batch_xyg();
        let enc = enc_xy("x", "y", Some("g"));
        let s = dummy_scales();
        let pos = PositionAdjust::Stack { by: Some("g".into()), offset: StackOffset::Zero };
        let out = apply_position(&b, Some(&pos), &s, &enc, crate::spec::mark::Mark::Text).unwrap();
        let ya = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        // x=1: a segment 0..10 → mid 5;  b segment 10..30 → mid 20.
        // x=2: a segment 0..30 → mid 15; b segment 30..70 → mid 50.
        assert!((ya.value(0) - 5.0).abs()  < 1e-9, "row 0 mid={}", ya.value(0));
        assert!((ya.value(1) - 20.0).abs() < 1e-9, "row 1 mid={}", ya.value(1));
        assert!((ya.value(2) - 15.0).abs() < 1e-9, "row 2 mid={}", ya.value(2));
        assert!((ya.value(3) - 50.0).abs() < 1e-9, "row 3 mid={}", ya.value(3));
        // __stack_y_base__ still carries the segment bottoms unchanged.
        let base = out.schema().index_of("__stack_y_base__").unwrap();
        let ba = out.column(base).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(ba.value(0), 0.0);
        assert_eq!(ba.value(1), 10.0);
        assert_eq!(ba.value(2), 0.0);
        assert_eq!(ba.value(3), 30.0);
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
