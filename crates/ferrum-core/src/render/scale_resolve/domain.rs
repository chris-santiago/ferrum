//! Domain computation: numeric extent union, sort application, field location.

use std::collections::HashMap;

use arrow::array::Array;
use arrow::record_batch::RecordBatch;

use super::column_min_max_f64;
use crate::render::{RenderError, RenderWarning};
use crate::spec::chart::ChartSpec;
use crate::transform::core::FINAL_OUTPUT_KEY;

/// Result of looking up a field across the primary batch and named
/// transform outputs. Carries both the source batch and the resolved
/// column so callers don't have to re-`column_by_name(...).expect(...)`
/// after the lookup — the "field is present in this batch" invariant
/// lives in the type, not in a comment.
pub(crate) struct LocatedColumn<'a> {
    pub(crate) batch: &'a RecordBatch,
    pub(crate) col: &'a dyn Array,
}

/// Pick the batch whose schema contains `field` and return both the batch
/// and the resolved column. Prefer `primary_batch` for back-compat
/// single-batch behavior; fall back to named outputs.
///
/// The named-output fallback iterates keys in **sorted order** rather than
/// raw `HashMap` order. When the field appears in several named outputs
/// (e.g. boxen publishes `group` in every `lv_depth_K` slice plus
/// `lv_outliers`), the categorical domain is built from `located.batch` via
/// first-appearance order, so the *choice* of batch must be deterministic.
/// `HashMap` iteration order is unspecified and varies per process, which
/// previously made boxen renders non-deterministic (byte-identical-golden
/// violation). Sorting the keys pins the selection.
pub(in crate::render) fn locate_field<'a>(
    field: &str,
    primary_batch: &'a RecordBatch,
    transform_outputs: &'a HashMap<String, RecordBatch>,
) -> Option<LocatedColumn<'a>> {
    if let Some(c) = primary_batch.column_by_name(field) {
        return Some(LocatedColumn { batch: primary_batch, col: c.as_ref() });
    }
    let mut keys: Vec<&String> = transform_outputs.keys().collect();
    keys.sort_unstable();
    for key in keys {
        let batch = &transform_outputs[key];
        if let Some(c) = batch.column_by_name(field) {
            return Some(LocatedColumn { batch, col: c.as_ref() });
        }
    }
    None
}

/// Aggregate operation used by data-aware sort forms (channel shorthand and
/// sort-field object). Vega-Lite's default `op` is `"sum"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SortOp {
    Sum,
    Mean,
    Max,
    Min,
    Count,
    Median,
}

impl SortOp {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "sum" => Some(Self::Sum),
            "mean" | "average" => Some(Self::Mean),
            "max" => Some(Self::Max),
            "min" => Some(Self::Min),
            "count" => Some(Self::Count),
            "median" => Some(Self::Median),
            _ => None,
        }
    }

    /// Reduce a category's per-row values (already filtered to non-null) to a
    /// single ordering key. An empty slice yields `f64::NEG_INFINITY` so empty
    /// categories sort to the bottom under ascending order (and to the top under
    /// descending), which keeps them out of the visually-meaningful extremes.
    fn aggregate(self, values: &[f64]) -> f64 {
        if matches!(self, Self::Count) {
            return values.len() as f64;
        }
        if values.is_empty() {
            return f64::NEG_INFINITY;
        }
        match self {
            Self::Sum => values.iter().sum(),
            Self::Mean => values.iter().sum::<f64>() / values.len() as f64,
            Self::Max => values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            Self::Min => values.iter().copied().fold(f64::INFINITY, f64::min),
            Self::Count => unreachable!("handled above"),
            Self::Median => {
                let mut sorted = values.to_vec();
                sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let n = sorted.len();
                if n % 2 == 1 {
                    sorted[n / 2]
                } else {
                    (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
                }
            }
        }
    }
}

/// Sort order for data-aware forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SortOrder {
    Ascending,
    Descending,
}

/// Data needed to resolve the *data-aware* sort forms (channel shorthand
/// `"x"`/`"-x"`/`"y"`/`"-y"` and the sort-field object). The alphabetical and
/// explicit-array forms ignore this context entirely.
///
/// `category_field` is the categorical axis field whose distinct values make up
/// the domain. `value_field_for_other_channel` resolves the field bound to the
/// *opposite* positional channel (used by the channel-shorthand form: `"-y"` on
/// an x-categorical axis sorts by the aggregate of the y field). `batch` is the
/// record batch the domain was built from — the same batch carries both the
/// category column and the candidate value columns.
pub(in crate::render) struct SortContext<'a> {
    pub(in crate::render) category_field: &'a str,
    pub(in crate::render) batch: &'a RecordBatch,
    /// Field bound to the x channel (if any), for resolving `"x"`/`"-x"`.
    pub(in crate::render) x_field: Option<&'a str>,
    /// Field bound to the y channel (if any), for resolving `"y"`/`"-y"`.
    pub(in crate::render) y_field: Option<&'a str>,
}

/// Apply `encoding.sort` to an ordinal domain in place.
///
/// Accepted forms (mirrors the Vega-Lite `sort` field):
/// - `"ascending"` — sort alphabetically ascending (locale-independent byte order).
/// - `"descending"` — sort alphabetically descending.
/// - Channel shorthand `"x"`/`"-x"`/`"y"`/`"-y"` — sort the categorical domain by
///   the per-category aggregate of the *other* positional channel's field. The
///   leading `-` selects descending; the default aggregate is `sum`.
/// - Sort-field object `{"field": "...", "op": "sum"|"mean"|"max"|"min"|
///   "count"|"median", "order": "ascending"|"descending"}` — sort the domain by
///   the per-category aggregate of `field`. `op` defaults to `sum`, `order` to
///   `ascending`.
/// - JSON array of strings — replace domain with that explicit order. Values not
///   present in the original domain are silently ignored; values in the original
///   domain that are absent from the array are appended at the end in their
///   original relative order so no data disappears from the scale.
/// - Absent — no-op (preserves insertion order).
///
/// When a data-aware form references a missing field, an unsupported dtype, or
/// an invalid `op`, the domain is left in insertion order and a
/// [`RenderWarning::SortSpecIgnored`] is pushed onto `warnings` — never a silent
/// no-op and never a panic.
pub(in crate::render) fn apply_sort_to_domain(
    domain: &mut Vec<String>,
    sort: Option<&serde_json::Value>,
    ctx: &SortContext<'_>,
    warnings: &mut Vec<RenderWarning>,
) {
    match sort {
        None => {}
        Some(serde_json::Value::String(s)) if s == "ascending" => {
            domain.sort_unstable();
        }
        Some(serde_json::Value::String(s)) if s == "descending" => {
            domain.sort_unstable_by(|a, b| b.cmp(a));
        }
        Some(serde_json::Value::String(s))
            if matches!(s.as_str(), "x" | "-x" | "y" | "-y") =>
        {
            apply_channel_shorthand_sort(domain, s, ctx, warnings);
        }
        Some(serde_json::Value::String(other)) => {
            // A bare string that isn't a recognized keyword/shorthand. Treating
            // it as a field name would be ambiguous with the keyword forms, so
            // surface it rather than silently no-op.
            warnings.push(RenderWarning::SortSpecIgnored {
                reason: format!("unrecognized sort spec {other:?}"),
            });
        }
        Some(serde_json::Value::Array(arr)) => {
            let explicit: Vec<String> = arr
                .iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect();
            // Keep only values that appear in the domain, in explicit order.
            // Then append any domain values not covered by the explicit list.
            let mut reordered: Vec<String> = explicit
                .iter()
                .filter(|v| domain.contains(v))
                .cloned()
                .collect();
            for v in domain.iter() {
                if !explicit.contains(v) {
                    reordered.push(v.clone());
                }
            }
            *domain = reordered;
        }
        Some(serde_json::Value::Object(obj)) => {
            apply_sort_field_object(domain, obj, ctx, warnings);
        }
        Some(other) => {
            warnings.push(RenderWarning::SortSpecIgnored {
                reason: format!("unsupported sort spec {other}"),
            });
        }
    }
}

/// Resolve the channel-shorthand form (`"x"`/`"-x"`/`"y"`/`"-y"`). The domain is
/// reordered by the per-category aggregate (default op = sum) of the field bound
/// to the named channel.
fn apply_channel_shorthand_sort(
    domain: &mut Vec<String>,
    shorthand: &str,
    ctx: &SortContext<'_>,
    warnings: &mut Vec<RenderWarning>,
) {
    let (order, channel) = if let Some(ch) = shorthand.strip_prefix('-') {
        (SortOrder::Descending, ch)
    } else {
        (SortOrder::Ascending, shorthand)
    };
    let value_field = match channel {
        "x" => ctx.x_field,
        "y" => ctx.y_field,
        _ => None,
    };
    // R3 (SortSpecIgnored family): `channel` here is NOT a resolved/post-flip
    // token the way every other R3-covered `channel` field is — it is the
    // literal substring of the `shorthand` STRING the user wrote in their own
    // `sort=` spec (e.g. `sort="-x"` → `channel == "x"`), echoed back verbatim.
    // `prepare::user_facing_channel` must NOT be applied to it: doing so would
    // translate the user's own written text AWAY from what they wrote (and
    // directly contradict the `sort={shorthand:?}` clause right next to it in
    // the same sentence). This message already satisfies "names the channel as
    // the user wrote it" by construction, flipped or not — see
    // `sort_shorthand_missing_channel_names_users_own_text_{unflipped,under_flip}`
    // below. (Separately: whether `ctx.x_field`/`ctx.y_field` — which DO come
    // from the post-flip rendering encoding, `positional.rs`'s
    // `PositionalFields` — resolve against the axis the user actually intended
    // by `"-x"`/`"-y"` under `CoordFlip` is a resolution-semantics question,
    // not a message-naming one; out of R3's scope, logged as a follow-up.)
    let Some(value_field) = value_field else {
        warnings.push(RenderWarning::SortSpecIgnored {
            reason: format!("sort={shorthand:?} but no field is bound to channel {channel:?}"),
        });
        return;
    };
    sort_domain_by_aggregate(domain, value_field, SortOp::Sum, order, ctx, warnings);
}

/// Resolve the sort-field object form
/// `{"field": ..., "op": ..., "order": ...}`.
fn apply_sort_field_object(
    domain: &mut Vec<String>,
    obj: &serde_json::Map<String, serde_json::Value>,
    ctx: &SortContext<'_>,
    warnings: &mut Vec<RenderWarning>,
) {
    let Some(field) = obj.get("field").and_then(|v| v.as_str()) else {
        warnings.push(RenderWarning::SortSpecIgnored {
            reason: "sort object missing string \"field\"".to_string(),
        });
        return;
    };
    // `op` defaults to sum (Vega-Lite default). `count` ignores the field's
    // values but still requires the category column.
    let op = match obj.get("op") {
        None => SortOp::Sum,
        Some(serde_json::Value::String(s)) => match SortOp::parse(s) {
            Some(op) => op,
            None => {
                warnings.push(RenderWarning::SortSpecIgnored {
                    reason: format!("unsupported sort op {s:?}"),
                });
                return;
            }
        },
        Some(other) => {
            warnings.push(RenderWarning::SortSpecIgnored {
                reason: format!("sort \"op\" must be a string, got {other}"),
            });
            return;
        }
    };
    let order = match obj.get("order").and_then(|v| v.as_str()) {
        Some("descending") => SortOrder::Descending,
        // Absent, "ascending", or anything else → ascending (Vega-Lite default).
        _ => SortOrder::Ascending,
    };
    sort_domain_by_aggregate(domain, field, op, order, ctx, warnings);
}

/// Shared driver for both data-aware forms: compute the per-category aggregate
/// of `value_field` (using the category column from `ctx`) and reorder `domain`
/// accordingly. On any data failure, leave the domain untouched and warn.
fn sort_domain_by_aggregate(
    domain: &mut [String],
    value_field: &str,
    op: SortOp,
    order: SortOrder,
    ctx: &SortContext<'_>,
    warnings: &mut Vec<RenderWarning>,
) {
    let categories = match crate::render::arrow_cast::col_as_str(ctx.batch, ctx.category_field) {
        Ok(c) => c,
        Err(e) => {
            warnings.push(RenderWarning::SortSpecIgnored {
                reason: format!("cannot read category field {:?}: {e}", ctx.category_field),
            });
            return;
        }
    };
    // `count` does not read the value column; every other op does.
    let values: Vec<Option<f64>> = if matches!(op, SortOp::Count) {
        vec![None; categories.len()]
    } else {
        match crate::render::arrow_cast::col_as_f64(ctx.batch, value_field) {
            Ok(v) => v,
            Err(e) => {
                warnings.push(RenderWarning::SortSpecIgnored {
                    reason: format!("cannot read sort value field {value_field:?}: {e}"),
                });
                return;
            }
        }
    };
    if categories.len() != values.len() {
        warnings.push(RenderWarning::SortSpecIgnored {
            reason: format!(
                "row-count mismatch sorting {:?} by {value_field:?}",
                ctx.category_field
            ),
        });
        return;
    }

    // Bucket each row's value under its category, skipping null categories and
    // (for value-reading ops) null/non-finite values.
    let mut buckets: HashMap<&str, Vec<f64>> = HashMap::new();
    for (cat, val) in categories.iter().zip(values.iter()) {
        let Some(cat) = cat.as_deref() else { continue };
        let entry = buckets.entry(cat).or_default();
        if matches!(op, SortOp::Count) {
            // Count of rows in the category — the pushed sentinel is unused by
            // `SortOp::Count::aggregate` (it uses slice length).
            entry.push(0.0);
        } else if let Some(v) = val {
            if v.is_finite() {
                entry.push(*v);
            }
        }
    }

    let key_for = |cat: &str| -> f64 {
        match buckets.get(cat) {
            Some(vals) => op.aggregate(vals),
            None => f64::NEG_INFINITY,
        }
    };
    domain.sort_by(|a, b| {
        let ka = key_for(a);
        let kb = key_for(b);
        let ord = ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal);
        match order {
            SortOrder::Ascending => ord,
            SortOrder::Descending => ord.reverse(),
        }
    });
}

/// Compute the unioned numeric/temporal extent for an axis field across
/// the relevant batches.
///
/// Extent rule:
///   - If `primary_batch` contains the field, use it as the starting
///     extent (preserves single-batch / faceted-panel semantics —
///     `FINAL_OUTPUT_KEY` is NOT unioned by default, so per-panel scales
///     remain independent when nothing else references a named output).
///   - Additionally union the field's extent across every named output
///     that some layer references via `data_source`. Required when a
///     layer's `data_source` points at a named transform whose output
///     extends past the primary batch — e.g. `ReferenceLine` for the
///     y=x diagonal in `calibration_chart`, whose endpoints [0, 1] must
///     be reachable even when the primary calibration curve sits inside
///     (0.05, 0.95).
///   - When the field is absent from `primary_batch`, fall back to
///     unioning across all named outputs that contain it (Phase 8b
///     composite-mark rule — e.g. boxplot whisker fields living in the
///     `box` named output).
///   - When `include_final` is set (faceted chart, this channel resolves
///     `ResolveMode::Shared`), the global all-panels batch at
///     [`FINAL_OUTPUT_KEY`] is *also* unioned even though the field is
///     present in `primary_batch` (the per-panel partition). This makes a
///     faceted raw field's positional domain span every panel, so marks
///     scale through the same global domain the shared axis displays
///     (T4 fix). Non-faceted and `Independent`-mode callers pass `false`,
///     keeping the per-panel-only behavior byte-identical.
///   - The paired field (x2/y2) follows the same lookup discipline.
pub(in crate::render) fn numeric_domain_union(
    channel: &str,
    field: &str,
    paired_field: Option<&str>,
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    spec: &ChartSpec,
    include_final: bool,
) -> Result<(f64, f64), RenderError> {
    let layer_data_sources: std::collections::HashSet<&str> = match &spec.layers {
        Some(layers) => layers.iter().filter_map(|l| l.data_source.as_deref()).collect(),
        None => std::collections::HashSet::new(),
    };
    let (mut mn, mut mx) = (f64::INFINITY, f64::NEG_INFINITY);
    let mut accumulate = |c: &dyn Array, source_field: &str| -> Result<(), RenderError> {
        let (a, b) = column_min_max_f64(c).map_err(|_| RenderError::UnsupportedDtype {
            field: source_field.to_string(),
            dtype: format!("{:?}", c.data_type()),
            context: None,
        })?;
        if a < mn { mn = a; }
        if b > mx { mx = b; }
        Ok(())
    };

    let mut union_field = |f: &str| -> Result<(), RenderError> {
        let primary_has = primary_batch.column_by_name(f).is_some();
        if let Some(c) = primary_batch.column_by_name(f) {
            accumulate(c.as_ref(), f)?;
        }
        for (key, batch) in transform_outputs.iter() {
            let key_is_referenced = layer_data_sources.contains(key.as_str());
            // Faceted + Shared: also union the global all-panels batch so the
            // per-panel domain spans every panel (T4). `FINAL_OUTPUT_KEY` is the
            // concatenated all-panels output already present at the per-panel
            // callsite; without this it is excluded whenever `primary_has`.
            let key_is_final = include_final && key == FINAL_OUTPUT_KEY;
            if !primary_has || key_is_referenced || key_is_final {
                if let Some(c) = batch.column_by_name(f) {
                    accumulate(c.as_ref(), f)?;
                }
            }
        }
        Ok(())
    };

    union_field(field)?;
    if let Some(p) = paired_field {
        union_field(p)?;
    }

    // Also union domains from other layers' fields for the same channel.
    // When two Charts with different DataFrames are composed via `+`, the RHS
    // columns are renamed (e.g. "x__rhs_...") and routed through a named
    // Identity transform. The primary field (from the chart-level encoding)
    // only covers the LHS data. We must also include the RHS layer's field so
    // the shared scale domain spans both layers' data ranges.
    if let Some(layers) = &spec.layers {
        for layer in layers {
            // An `independent_y` layer resolves its own y-slot (secondary-y-axis,
            // GH #52) and must not widen the primary/shared y domain — that would
            // couple the left axis to a scale it does not own. Its x stays shared,
            // so only the y channel skips. Byte-stable for all-shared charts: the
            // flag defaults false, so this never fires without an independent layer.
            if channel == "y" && layer.independent_y {
                continue;
            }
            let layer_field = match channel {
                "x" => layer.encoding.x.as_ref().map(|e| e.field.as_str()),
                "y" => layer.encoding.y.as_ref().map(|e| e.field.as_str()),
                _ => None,
            };
            if let Some(lf) = layer_field {
                if lf != field {
                    // Ignore errors — the field may not exist in all batches
                    // (e.g. when it lives in a named output that isn't loaded yet).
                    let _ = union_field(lf);
                }
            }
        }
    }

    if !mn.is_finite() || !mx.is_finite() {
        // All values were null/NaN — return a default domain so the chart
        // renders with axes but no marks, instead of raising an error.
        return Ok((0.0, 1.0));
    }
    // Degenerate domain: a single row or all-equal values produce mn == mx.
    // A zero-span domain collapses every data point to the same pixel, which
    // causes `to_pixel_f64` to return NaN (0/0 in the linear formula) and the
    // mark renderer silently drops every row. Expand to a symmetric band so the
    // mark renders at the centre of the plot area. Guard fires only here because
    // `numeric_domain_union` is called exclusively from the auto-inferred
    // (no explicit ScaleSpec) path in `build_axis_scale`; explicit domains go
    // through `build_from_scale_spec` → `resolve_continuous_domain_and_range`.
    if mn == mx {
        if mn == 0.0 {
            mn = -1.0;
            mx = 1.0;
        } else {
            mn -= 0.5;
            mx += 0.5;
        }
    }
    Ok((mn, mx))
}

/// Distinct positional (x:N / y:N) categories for an ordinal/band axis, unioning
/// the per-panel partition with the global all-panels batch when faceted+Shared.
///
/// - `panel_batch` is the located per-panel batch the non-shared path uses.
/// - When `include_final` is false this is byte-identical to
///   [`crate::render::arrow_cast::distinct_positional_categories`] on
///   `panel_batch` — non-faceted and `Independent`-mode callers see no change.
/// - When `include_final` is true (faceted, this channel resolves
///   `ResolveMode::Shared`) the domain is the union of the panel's categories
///   with the global set at [`FINAL_OUTPUT_KEY`], **in global first-appearance
///   order**. Deriving the order from the single global batch (a superset of
///   every panel) is what makes the categorical domain identical in every panel
///   — the shared-axis invariant. Per-panel-only categories (which cannot occur,
///   since the global batch is the concat of all panels) would still be honored
///   by the trailing union, never dropped.
///
/// Order semantics for the non-shared path are unchanged: it returns the
/// per-panel first-appearance order untouched.
pub(in crate::render) fn distinct_positional_categories_shared(
    panel_batch: &RecordBatch,
    field: &str,
    transform_outputs: &HashMap<String, RecordBatch>,
    include_final: bool,
) -> Result<Vec<String>, RenderError> {
    let panel_domain =
        crate::render::arrow_cast::distinct_positional_categories(panel_batch, field)?;
    if !include_final {
        return Ok(panel_domain);
    }
    let Some(global_batch) = transform_outputs.get(FINAL_OUTPUT_KEY) else {
        return Ok(panel_domain);
    };
    if global_batch.column_by_name(field).is_none() {
        return Ok(panel_domain);
    }
    // Global first-appearance order is the shared domain: a superset of every
    // panel, so it is identical across panels and preserves the global order.
    let mut domain =
        crate::render::arrow_cast::distinct_positional_categories(global_batch, field)?;
    // Defensive union: append any panel category absent from the global set so a
    // category can never disappear from a panel's scale (no-op in practice).
    for cat in panel_domain {
        if !domain.contains(&cat) {
            domain.push(cat);
        }
    }
    Ok(domain)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::StringArray;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn group_batch(groups: Vec<&str>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("group", DataType::Utf8, true)]));
        let arr: Vec<Option<&str>> = groups.into_iter().map(Some).collect();
        RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(arr))]).unwrap()
    }

    /// Regression guard for the boxen render non-determinism bug: when a field
    /// (e.g. `group`) is absent from the primary batch but present in several
    /// named outputs whose rows differ in order, `locate_field` must pick the
    /// *same* batch every time. Raw `HashMap` iteration order is unspecified and
    /// varied per process, producing distinct categorical domains (hence
    /// distinct SVGs) across repeated renders of the same chart.
    #[test]
    fn locate_field_named_output_fallback_is_deterministic() {
        // Mimic boxen's fan-out: several named outputs each carry `group`, but
        // in different row orders. Sorted-key selection lands on "lv".
        let primary = RecordBatch::new_empty(Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float64,
            true,
        )])));

        // Build the map fresh on every iteration so HashMap seeding differs, and
        // assert the located batch's first row is stable regardless.
        let mut first_seen: Option<String> = None;
        for _ in 0..64 {
            let mut outputs: HashMap<String, RecordBatch> = HashMap::new();
            outputs.insert("lv".to_string(), group_batch(vec!["A", "B", "Z"]));
            outputs.insert("lv_depth_1".to_string(), group_batch(vec!["Z", "A", "B"]));
            outputs.insert("lv_depth_2".to_string(), group_batch(vec!["B", "Z", "A"]));
            outputs.insert("lv_outliers".to_string(), group_batch(vec!["Z", "B", "A"]));

            let located = locate_field("group", &primary, &outputs)
                .expect("group present in named outputs");
            let col = located.col.as_any().downcast_ref::<StringArray>().unwrap();
            let first = col.value(0).to_string();
            match &first_seen {
                None => first_seen = Some(first),
                Some(prev) => assert_eq!(
                    prev, &first,
                    "locate_field must select the same named output across runs"
                ),
            }
        }
        // Sorted keys: "lv" < "lv_depth_1" < … so the "lv" batch (A, B, Z) wins.
        assert_eq!(first_seen.as_deref(), Some("A"));
    }

    // --- R3: SortSpecIgnored (channel-shorthand family) ---------------------

    /// R3, unflipped shape: `SortContext.x_field`/`y_field` reflect an ordinary
    /// (unflipped) chart — category on x, y unbound. `sort="-y"` references the
    /// unbound channel, so the warning fires; its message must echo exactly the
    /// literal shorthand text the user wrote.
    #[test]
    fn sort_shorthand_missing_channel_names_users_own_text_unflipped() {
        let batch = group_batch(vec!["a", "b"]);
        let ctx = SortContext {
            category_field: "cat",
            batch: &batch,
            x_field: Some("price"),
            y_field: None,
        };
        let mut domain = vec!["a".to_string(), "b".to_string()];
        let mut warnings = Vec::new();
        apply_sort_to_domain(&mut domain, Some(&serde_json::json!("-y")), &ctx, &mut warnings);
        assert_eq!(warnings.len(), 1);
        match &warnings[0] {
            RenderWarning::SortSpecIgnored { reason } => {
                assert_eq!(reason, "sort=\"-y\" but no field is bound to channel \"y\"");
            }
            other => panic!("expected SortSpecIgnored, got {other:?}"),
        }
        assert_eq!(domain, vec!["a".to_string(), "b".to_string()], "insertion order preserved on failure");
    }

    /// R3, post-`CoordFlip` shape: `SortContext.x_field`/`y_field` are
    /// populated from the RESOLVED (post-flip) rendering encoding on the
    /// prepare path (`positional.rs`'s `PositionalFields`) — the same
    /// "downstream of the swap" condition every other R3 family is un-flipped
    /// under. This family is the deliberate exception: `channel` in the
    /// message is parsed straight out of the `shorthand` STRING the user
    /// wrote (`apply_channel_shorthand_sort`'s doc comment explains why), not
    /// derived from which resolved slot matched — so the message must still
    /// echo the user's own literal text, byte-identical in shape to the
    /// unflipped case above, even though the underlying `ctx` here is
    /// genuinely post-flip.
    #[test]
    fn sort_shorthand_missing_channel_names_users_own_text_under_flip() {
        let batch = group_batch(vec!["a", "b"]);
        let ctx = SortContext {
            category_field: "cat",
            batch: &batch,
            x_field: Some("weight"), // post-flip x holds what was the user's y
            y_field: None,           // post-flip y holds what was the user's x (unset)
        };
        let mut domain = vec!["a".to_string(), "b".to_string()];
        let mut warnings = Vec::new();
        apply_sort_to_domain(&mut domain, Some(&serde_json::json!("-y")), &ctx, &mut warnings);
        assert_eq!(warnings.len(), 1);
        match &warnings[0] {
            RenderWarning::SortSpecIgnored { reason } => {
                assert_eq!(
                    reason, "sort=\"-y\" but no field is bound to channel \"y\"",
                    "must echo the user's own written shorthand — never a resolved/internal token"
                );
            }
            other => panic!("expected SortSpecIgnored, got {other:?}"),
        }
    }
}
