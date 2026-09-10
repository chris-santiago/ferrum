//! Domain computation: numeric extent union, sort application, field location.

use std::collections::HashMap;

use arrow::array::Array;
use arrow::record_batch::RecordBatch;

use super::column_min_max_f64;
use crate::render::{RenderError, RenderWarning};
use crate::spec::chart::ChartSpec;
use crate::spec::mark::Mark;
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
/// The explicit continuous extent an encoding's own `scale=` declares, or
/// `None` when it declares none.
///
/// The single answer to "did the user pin this channel's domain at the
/// encoding level" — the question the documented cascade turns on. Asked by
/// the color resolver (`color.rs::scale_explicit_domain`, which seeds a
/// colorbar's extent) and by the chart-level scale-domain config
/// (`prepare::apply_axis_domain_config_for`, whose four fields lose to it
/// silently, D3/spec §4.2).
///
/// Delegates to [`ScaleSpec::positional_extent`](crate::spec::encoding::ScaleSpec::positional_extent),
/// which is compiler-enforced exhaustive over the scale variants and already
/// encodes the distinctions that matter here: `Quantile`/`Threshold`/
/// `BinOrdinal` domains are binning artifacts rather than extents, ordinal
/// families have no continuous extent at all, and a continuous scale carrying
/// no `domain` (e.g. `LinearScale(clamp=True)`) declares nothing — so it must
/// NOT out-rank a chart-level `domain_max`. Testing `scale.is_some()` instead
/// gets that last case wrong, which is exactly how the config fields went
/// silently inert for a common encoding shape.
pub(in crate::render) fn encoding_explicit_extent(
    enc: &crate::spec::encoding::EncodingSpec,
) -> Option<(f64, f64)> {
    let extent = enc.scale.as_ref()?.positional_extent()?;
    (extent.len() >= 2).then(|| (extent[0], extent[extent.len() - 1]))
}

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
///   - Every OTHER layer's fields for this channel are unioned too, read in
///     the channel's own coordinate space — see the `spec.layers` loop below.
///   - **Bar zero-anchor (T5r, catplot aggregation; narrowed cycle 2 per
///     spec-review issue 1 and rust-quality-review findings 1-3).** When a
///     `Mark::Bar` layer binds this channel's VALUE slot (never its position
///     slot — `"y"` unflipped, `"x"` under `CoordFlip`, in the channel's own
///     coordinate space, same `coord_flipped` normalization the layer-union
///     loop already applies) with data that actually contributed to the
///     union, AND that layer does not itself bind the paired span field on
///     this channel (`y2`/`x2`), AND the layer does not explicitly opt out
///     (`mark_style.zero == Some(false)`), the resolved extent is widened to
///     include zero — mirroring `resolve_continuous_domain_and_range`'s
///     `zero: true` block (`d[0] > 0.0 → 0.0`, `d[1] < 0.0 → 0.0`) exactly,
///     just applied on the AUTO-INFERRED path instead of an explicit
///     `ScaleSpec`. All four conditions mirror the single-chart injection's
///     guards (`_spec_build.py:394-399`: `axis == "y"`, `not _y2_bound` —
///     the D3 rule, `:379-385` — and `_mark_zero`/`zero_anchor()`); dropping
///     any of them widens a channel Python's own rule would never anchor —
///     e.g. without the value-slot guard, a histogram bar layer (`x`/`x2` =
///     bin range, `y` = count) had its POSITION axis wrongly zero-anchored,
///     squeezing `displot(kind="hist")` bars into a corner of the panel
///     whenever the data doesn't straddle zero. The data-contribution
///     requirement additionally guards against a layer whose declared value
///     field is absent from every batch (wrong `data_source`, not loaded
///     yet) forcing a zero-anchor on the strength of its NAME alone. See
///     `MarkKwargsSpec::zero`'s doc comment for the live, user-reachable
///     path that sets the opt-out today (`catplot(..., mark={"bar": {"zero":
///     False}})` and any other layered figure function's `mark=`
///     passthrough). A single (non-layered) `mark_bar` never
///     reaches this function at all: Python's
///     `_spec_build.py::_build_encoding_specs` already injects an explicit
///     `scale={"zero": True}` onto that chart's y encoding, which routes
///     through `build_from_scale_spec` and returns before this function is
///     even called. This rule exists for the case Python's injection cannot
///     serve without silently clipping other layers: a layered spec
///     (`catplot(kind="bar")`'s bar + CI-whisker `rule`), where an explicit
///     per-layer `ScaleSpec` would replace this cross-layer union with a
///     single-column domain and clip whichever layer's field extends
///     furthest (measured: 3 of 4 upper CI whiskers vanished off-panel).
///     Scoped to `spec.layers` only (never `spec.mark`), so a genuinely
///     unlayered spec's auto path is untouched — byte-identical, because a
///     single-chart bar never reaches the auto path in the first place.
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
    // T5r bar zero-anchor: set when a `Mark::Bar` layer binds this channel's
    // primary (non-paired) field, in the channel's own coordinate space. See
    // this function's doc comment ("Bar zero-anchor") for the full rationale.
    let mut bar_zero_wanted = false;
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

    // Returns whether `f` was found (and accumulated) in ANY batch —
    // `Ok(false)` is the normal "field lives in a batch not loaded yet /
    // wrong data_source" outcome, not an error. Callers that only care about
    // the accumulated extent ignore the bool; the bar zero-anchor rule below
    // (T5r cycle 2) needs it so a layer's merely-DECLARED value field, never
    // found in any batch, cannot force a zero-anchor on the strength of its
    // name alone (spec-review finding 3).
    let mut union_field = |f: &str| -> Result<bool, RenderError> {
        let mut found = false;
        let primary_has = primary_batch.column_by_name(f).is_some();
        if let Some(c) = primary_batch.column_by_name(f) {
            accumulate(c.as_ref(), f)?;
            found = true;
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
                    found = true;
                }
            }
        }
        Ok(found)
    };

    let field_found = union_field(field)?;
    if let Some(p) = paired_field {
        union_field(p)?;
    }

    // Also union domains from other layers' fields for the same channel.
    // When two Charts with different DataFrames are composed via `+`, the RHS
    // columns are renamed (e.g. "x__rhs_...") and routed through a named
    // Identity transform. The primary field (from the chart-level encoding)
    // only covers the LHS data. We must also include the RHS layer's field so
    // the shared scale domain spans both layers' data ranges.
    //
    // Also covers each OTHER layer's paired channel field (x2 for "x", y2 for
    // "y") — batch-A Task 13. Before this, a layer's OWN x2/y2 was never
    // unioned in here (only its bare x/y was), which this module's own
    // doc comment above already names as required "when a layer's
    // data_source points at a named transform whose output extends past the
    // primary batch" (the `calibration_chart` `ReferenceLine` example) — that
    // prior example happens to need only bare x/y (its endpoints are two
    // ROWS of one field, not one row's x2/y2), so the gap stayed latent until
    // `mark_qq(line=True)`'s reference-diagonal `rule` layer, whose single-row
    // endpoints live in x/y AND x2/y2 (`qq_line_x_start`/`_end`,
    // `qq_line_y_start`/`_end`), all four on a named `data_source` distinct
    // from the primary/chart field. Without this, `qq_line_y_end` was never
    // reachable by any domain union, so `to_pixel_f64` returned `None` for
    // the far endpoint and the reference diagonal silently drew 0 nodes even
    // though `marks/rule.rs`'s four-numeric branch resolved correctly.
    if let Some(layers) = &spec.layers {
        // COORDINATE SPACE (spec §4.4, "Extended 2026-09-02, T13 quality
        // review", ruling 1). `channel`/`field` name the axis being resolved,
        // i.e. the POST-flip space: the caller derived them from the rendering
        // encoding `prepare::build_layers` already swapped. `spec.layers`, by
        // contrast, is the spec as AUTHORED — `prepare` swaps its own
        // `LayerPrepared` copies, never the spec — so under `CoordFlip` a
        // layer's `y`/`y2` is what post-flip occupies the x axis. Reading the
        // layer's like-named channel therefore unions the WRONG axis: a rule
        // layer's `y2` widening the y domain when its endpoint lives on x is a
        // visible axis blowout (the y axis running to the far endpoint while
        // the endpoint itself is unreachable on x). Move the layer encoding
        // into the channel's own space first, via the same
        // `Encoding::flip_positional` `LayerPrepared::flip_coords` applies, and
        // both sides of every read below live in one space.
        let coord_flipped = matches!(spec.coord, Some(crate::spec::coord::CoordKind::Flip));
        for layer in layers {
            // An `independent_y` layer resolves its own y-slot (secondary-y-axis,
            // GH #52) and must not widen the primary/shared y domain — that would
            // couple the left axis to a scale it does not own. Its x stays shared,
            // so only the y channel skips. Byte-stable for all-shared charts: the
            // flag defaults false, so this never fires without an independent layer.
            // Names the axis SLOT (the layer owns a second y axis), not an
            // encoding channel, so it reads `channel` directly and is unaffected
            // by the coordinate-space normalization below.
            if channel == "y" && layer.independent_y {
                continue;
            }
            let flipped = coord_flipped.then(|| {
                let mut e = layer.encoding.clone();
                e.flip_positional();
                e
            });
            let encoding = flipped.as_ref().unwrap_or(&layer.encoding);
            let layer_fields: [Option<&str>; 2] = match channel {
                "x" => [
                    encoding.x.as_ref().map(|e| e.field.as_str()),
                    encoding.x2.as_ref().map(|e| e.field.as_str()),
                ],
                "y" => [
                    encoding.y.as_ref().map(|e| e.field.as_str()),
                    encoding.y2.as_ref().map(|e| e.field.as_str()),
                ],
                _ => [None, None],
            };
            // Union every declared field for this layer on this channel, and
            // remember whether the VALUE slot (index 0) actually resolved to
            // data in some batch — `value_field_contributed` below. One pass
            // replaces what used to be a separate "union everything, then
            // separately ask `.is_some()`" step: a layer's field can be
            // absent from every batch (wrong `data_source`, not loaded yet),
            // and the bar zero-anchor rule below must not fire on the
            // strength of the field's declared NAME alone when it never
            // contributed a single value (spec-review finding 3).
            let mut value_field_contributed = false;
            for (i, lf) in layer_fields.into_iter().enumerate() {
                let Some(lf) = lf else { continue };
                let contributed = if lf == field {
                    field_found
                } else {
                    // Ignore errors — the field may not exist in all batches
                    // (e.g. when it lives in a named output that isn't loaded yet).
                    union_field(lf).unwrap_or(false)
                };
                if i == 0 {
                    value_field_contributed = contributed;
                }
            }
            // Bar zero-anchor detection (T5r; narrowed cycle 2 — spec-review
            // issue 1; extended cycle 2 — rust-quality-review findings 1-3).
            // Mirrors the single-chart injection's guards
            // (`_spec_build.py:394-399`), not just its arithmetic:
            //   (a) `axis == "y"` there — the anchor is the bar's EXTENT
            //       axis only, never its POSITION axis. Ferrum always
            //       authors a bar's value on `y` (category/position on `x`,
            //       or the reverse under `CoordFlip`, which is exactly what
            //       `coord_flipped`/`encoding` above already normalizes) —
            //       so the value channel is `"y"` unflipped, `"x"` flipped.
            //       Dropping this guard is what fired the anchor on a
            //       histogram bar's `x` (bin-range position) channel too,
            //       squeezing `displot(kind="hist")` bars into a fraction of
            //       the panel when the data doesn't straddle zero. A bar
            //       authored with its value on the OTHER channel directly
            //       (no `CoordFlip`) — e.g. `shap_waterfall`'s floating
            //       `x`/`x2` bars (`marks/diagnostic/_explanation.py:285-296`,
            //       `x=x0`, `x2=x1`, `y=feature`) — is excluded by this same
            //       guard for the identical reason Python's own `axis=="y"`
            //       rule already excludes it: ferrum's zero-anchor has never
            //       covered a directly-authored x-as-value bar, layered or
            //       not, so this mirrors existing scope rather than widening
            //       it under this fix.
            //   (b) `not _y2_bound` there (the D3 rule, `:379-385`) — when
            //       this layer's OWN value channel carries a paired span
            //       field (`y2`, or `x2` under flip — `layer_fields[1]`),
            //       the extent is taken literally from `[value, span]` and
            //       force-anchoring to zero would distort a domain that was
            //       never meant to touch zero. `shap_waterfall` is ALSO the
            //       concrete instance of this shape (its bar layer's value
            //       channel carries `x2`) — doubly protected here: guard (a)
            //       already excludes it (its value channel is `x`, never the
            //       recognized value slot unflipped), and guard (b) would
            //       independently exclude it too if ferrum ever authored a
            //       value-on-`y`-with-`y2` floating bar (the shape
            //       `y2_bound_bar_layer_never_zero_extends` below pins).
            //   (c) the layer's own opt-out — `layer.mark_style.zero ==
            //       Some(false)`, the per-layer mirror of
            //       `mark_bar(zero=False)` / `_mark_zero`
            //       (`spec::mark_style::MarkKwargsSpec::zero`, T5r cycle 2).
            //       Live and user-reachable today via the generic per-layer
            //       `mark=` passthrough on layered figure functions — e.g.
            //       `fm.catplot(..., kind="bar", mark={"bar": {"zero":
            //       False}})` resolves the value axis to the raw data range
            //       (measured: `58..61.5`, not `0..61.5`) — see that field's
            //       doc comment for the plumbing and the one path that
            //       stays inert.
            // The Python rule's remaining guard (`not _y_is_categorical`)
            // needs no mirror: `numeric_domain_union` is only ever called
            // for a quantitative/temporal axis (`build_axis_scale`'s
            // ordinal/nominal branch never reaches here), so an ordinal
            // position channel is excluded structurally, not by an extra
            // flag.
            let is_value_channel = if coord_flipped { channel == "x" } else { channel == "y" };
            let layer_wants_zero =
                layer.mark_style.as_ref().and_then(|m| m.zero).unwrap_or(true);
            if is_value_channel
                && layer.mark == Mark::Bar
                && layer_wants_zero
                && value_field_contributed
                && layer_fields[1].is_none()
            {
                bar_zero_wanted = true;
            }
        }
    }

    if !mn.is_finite() || !mx.is_finite() {
        // All values were null/NaN — return a default domain so the chart
        // renders with axes but no marks, instead of raising an error.
        return Ok((0.0, 1.0));
    }
    if bar_zero_wanted {
        // Mirrors `resolve_continuous_domain_and_range`'s `zero: true` block
        // exactly (same two guarded assignments), applied here instead of
        // there because a layered bar spec never reaches that function (see
        // this function's doc comment). Runs BEFORE the degenerate-domain
        // widening below so a single-row/all-equal bar value (e.g. one
        // category) resolves to a real `[0, v]` span instead of the
        // synthetic `±0.5` band that widening produces for non-bar data.
        if mn > 0.0 {
            mn = 0.0;
        }
        if mx < 0.0 {
            mx = 0.0;
        }
    }
    // Degenerate domain: a single row or all-equal values produce mn == mx.
    // As of GH #104, `scale::core::degenerate_ratio` already makes every
    // continuous scale's `scale()` finite (a centered midpoint) for a
    // zero-span domain, so this expansion is no longer needed to keep marks
    // from vanishing — `to_pixel_f64` no longer returns NaN there. It stays
    // for a reason #104 doesn't touch: this is the *domain itself* (the
    // axis's displayed extent, tick generation, "nice" rounding), not the
    // per-value pixel projection. A literal `[d, d]` domain still collapses
    // `nice_ticks`/tick generation to a single degenerate tick and gives the
    // axis no visible span to lay out against; expanding to a symmetric band
    // around `mn == mx` gives the auto-inferred axis a real (if synthetic)
    // extent to tick and label, which #104's per-value guard has no way to
    // provide. Guard fires only here because `numeric_domain_union` is called
    // exclusively from the auto-inferred (no explicit ScaleSpec) path in
    // `build_axis_scale`; explicit domains go through `build_from_scale_spec`
    // → `resolve_continuous_domain_and_range`, where #104's per-scale guard
    // is what keeps a degenerate explicit domain finite instead.
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

    // ── Task 13 (spec §4.4, "Extended 2026-09-02, T13 quality review",
    // ruling 1): the layer union reads each layer's fields in the coordinate
    // space of the channel being resolved ─────────────────────────────────────
    //
    // Two properties, pinned together because one is the other's control:
    //   1. a layer's `x2`/`y2` past its bare `x`/`y` widens THAT channel's
    //      domain (the batch-A T13 union fix — `mark_qq(line=True)`'s reference
    //      diagonal was unreachable without it), and
    //   2. under `CoordFlip` it widens the OTHER axis, because `spec.layers`
    //      is the AUTHORED encoding while `channel`/`field` name the rendered
    //      (post-flip) axis.
    //
    // Both `*_SPEC_JSON`s are the VERBATIM `chart.to_dict()` of the public-API
    // call in their doc comment (captured 2026-09-02), parsed through the same
    // serde path `ChartSpec.from_json` uses, so they cannot drift from what
    // Python lowers.

    /// `fm.Chart(df).mark_point().encode(x="b", y="a")` layered (`+`) with
    /// `fm.Chart(df).mark_rule().encode(y="a", y2="big")`.
    const LAYERED_Y2_SPEC_JSON: &str = r#"{"data": {"kind": "named", "name": "default"}, "mark": "point", "encoding": {"x": {"field": "b"}, "y": {"field": "a"}}, "layers": [{"mark": "point", "encoding": {"x": {"field": "b"}, "y": {"field": "a"}}}, {"mark": "rule", "encoding": {"y": {"field": "a"}, "y2": {"field": "big"}}}]}"#;

    /// The same layering with `.coord(fm.CoordFlip())` on the primary chart —
    /// the reviewer's A/B probe. `prepare::build_layers` swaps the RENDERING
    /// encodings, so the resolved x axis holds `a` and the y axis holds `b`,
    /// while `layers` below still reads exactly as authored.
    const LAYERED_Y2_FLIPPED_SPEC_JSON: &str = r#"{"data": {"kind": "named", "name": "default"}, "mark": "point", "encoding": {"x": {"field": "b"}, "y": {"field": "a"}}, "layers": [{"mark": "point", "encoding": {"x": {"field": "b"}, "y": {"field": "a"}}}, {"mark": "rule", "encoding": {"y": {"field": "a"}, "y2": {"field": "big"}}}], "coord": {"kind": "flip"}}"#;

    /// `pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "big": [1000.0, 2000.0]})`
    /// — three disjoint numeric ranges, so which field reached the union is
    /// readable straight off the returned extent.
    fn abc_batch() -> RecordBatch {
        use arrow::array::Float64Array;
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Float64, false),
            Field::new("b", DataType::Float64, false),
            Field::new("big", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
                Arc::new(Float64Array::from(vec![3.0, 4.0])),
                Arc::new(Float64Array::from(vec![1000.0, 2000.0])),
            ],
        )
        .unwrap()
    }

    /// Resolve one axis's auto-inferred extent for a lowered spec, exactly as
    /// `build_axis_scale` does (no named transform outputs, no facet union).
    fn axis_extent(spec_json: &str, channel: &str, field: &str) -> (f64, f64) {
        let spec: ChartSpec =
            serde_json::from_str(spec_json).expect("lowered spec JSON must deserialize");
        let batch = abc_batch();
        numeric_domain_union(channel, field, None, &batch, &HashMap::new(), &spec, false)
            .expect("every field in the fixture is Float64")
    }

    /// Unflipped: the rule layer's `y2` (1000..2000) extends past its bare `y`
    /// (`a`, 1..2), so the y domain must reach 2000 — otherwise the layer's own
    /// far endpoint projects to `None` and the reference line silently draws
    /// nothing. The x axis is the control: `big` belongs to the y channel, so
    /// it must not leak there.
    #[test]
    fn layer_paired_endpoint_widens_its_own_channel_only() {
        assert_eq!(
            axis_extent(LAYERED_Y2_SPEC_JSON, "y", "a"),
            (1.0, 2000.0),
            "the layer's y2 must widen the y domain past its bare y"
        );
        assert_eq!(
            axis_extent(LAYERED_Y2_SPEC_JSON, "x", "b"),
            (3.0, 4.0),
            "a y-channel field must never widen the x domain"
        );
    }

    /// Flipped: the SAME authored `y2` now lives on the rendered x axis, so it
    /// must widen x and leave y at `b`'s own extent — the exact mirror of the
    /// unflipped case above.
    ///
    /// RED (verified in place): read `layer.encoding` instead of its
    /// flip-normalized copy and this fails at the first assertion with
    /// `(1.0, 4.0)` — x picks up the sibling layer's authored `x` (`b`) and
    /// still never reaches the rule's own far endpoint on the axis that
    /// endpoint actually occupies, while y (unasserted once x fails, `(1.0,
    /// 2000.0)` when checked alone) blows out to 2000 on an axis no mark uses
    /// there. That pair is the reviewer's live A/B probe.
    #[test]
    fn coord_flipped_layer_paired_endpoint_widens_the_other_axis() {
        assert_eq!(
            axis_extent(LAYERED_Y2_FLIPPED_SPEC_JSON, "x", "a"),
            (1.0, 2000.0),
            "under CoordFlip the authored y2 occupies the x axis and must widen it"
        );
        assert_eq!(
            axis_extent(LAYERED_Y2_FLIPPED_SPEC_JSON, "y", "b"),
            (3.0, 4.0),
            "under CoordFlip the y axis holds the authored x field; an authored \
             y/y2 must not widen it"
        );
    }

    // ── T5r: bar zero-anchor on the layered/auto-inferred path ──────────────
    //
    // `catplot(kind="bar")` renders a `bar` layer (one mean per category) plus
    // a `rule` "whisker" layer (the bootstrap CI), both reading a named
    // `Summary` transform output via `data_source`. Before this fix the
    // auto-inferred cross-layer union (this function) had no zero rule at
    // all, so the value axis stopped at the union of `mean`/`lower`/`upper`
    // and bars looked truncated at the panel bottom. An explicit per-layer
    // `scale={"zero": True}` (the mechanism a plain single-chart `mark_bar`
    // uses, injected by `_spec_build.py`) was measured and rejected: it
    // routes through `build_from_scale_spec`, which resolves the domain from
    // ONE column, so it silently clipped the whisker layer's own extent (3 of
    // 4 upper CI whiskers vanished off-panel) — see the coder's report,
    // `.sdd/task-5-report.md` concern 1, and this function's own doc comment
    // ("Bar zero-anchor") above.

    /// Verbatim `chart.to_dict()` JSON for `fm.catplot(df, x="cat", y="val",
    /// kind="bar")` (captured 2026-09-10, T5r, with `df` mapping `cat` to
    /// two groups and `val` to values whose bootstrap mean/CI land around
    /// 50-90). Two layers: `bar` on the transform's `mean` output, `whisker`
    /// (a `rule`) on `lower`/`upper`, both via `data_source: "summary"`.
    const CATPLOT_BAR_SPEC_JSON: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"cat"},"y":{"field":"val"}},"transforms":[{"type":"summary","field":"val","groupby":["cat"],"error_fn":"ci","ci":0.95,"n_boot":1000,"seed":0,"name":"summary"}],"layers":[{"mark":"bar","encoding":{"x":{"field":"cat"},"y":{"field":"mean","title":"val"}},"data_source":"summary","position":{"type":"identity"},"name":"bar"},{"mark":"rule","encoding":{"x":{"field":"cat"},"y":{"field":"lower","title":"val"},"y2":{"field":"upper"}},"mark_style":{"stroke":"theme:label","stroke_dash":[]},"data_source":"summary","position":{"type":"identity"},"name":"whisker"}]}"#;

    /// The same call with `orient="h"` (captured 2026-09-10, T5r). The
    /// authored encoding is untouched (the value field is still authored on
    /// `y`); only `coord` gains `{"kind": "flip"}`, so the RENDERED x axis is
    /// where the bar's value lives and where the zero-anchor must land.
    const CATPLOT_BAR_FLIPPED_SPEC_JSON: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"cat"},"y":{"field":"val"}},"transforms":[{"type":"summary","field":"val","groupby":["cat"],"error_fn":"ci","ci":0.95,"n_boot":1000,"seed":0,"name":"summary"}],"layers":[{"mark":"bar","encoding":{"x":{"field":"cat"},"y":{"field":"mean","title":"val"}},"data_source":"summary","position":{"type":"identity"},"name":"bar"},{"mark":"rule","encoding":{"x":{"field":"cat"},"y":{"field":"lower","title":"val"},"y2":{"field":"upper"}},"mark_style":{"stroke":"theme:label","stroke_dash":[]},"data_source":"summary","position":{"type":"identity"},"name":"whisker"}],"coord":{"kind":"flip"}}"#;

    /// `CATPLOT_BAR_SPEC_JSON` with the `bar` layer's mark swapped to
    /// `"point"` — the ONLY difference from the spec above, isolating mark
    /// type as the sole variable. Proves the anchor is gated on `Mark::Bar`
    /// specifically, not merely on "a layered spec with a `rule` sibling".
    const CATPLOT_NONBAR_SPEC_JSON: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"cat"},"y":{"field":"val"}},"transforms":[{"type":"summary","field":"val","groupby":["cat"],"error_fn":"ci","ci":0.95,"n_boot":1000,"seed":0,"name":"summary"}],"layers":[{"mark":"point","encoding":{"x":{"field":"cat"},"y":{"field":"mean","title":"val"}},"data_source":"summary","position":{"type":"identity"},"name":"bar"},{"mark":"rule","encoding":{"x":{"field":"cat"},"y":{"field":"lower","title":"val"},"y2":{"field":"upper"}},"mark_style":{"stroke":"theme:label","stroke_dash":[]},"data_source":"summary","position":{"type":"identity"},"name":"whisker"}]}"#;

    /// A non-layered `mark_bar` spec — `spec.layers` is absent entirely. This
    /// case never reaches `numeric_domain_union` in real rendering (Python's
    /// `_spec_build.py::_build_encoding_specs` already injects an explicit
    /// `scale={"zero": True}` onto a single-chart bar's y encoding, which
    /// resolves through `build_from_scale_spec` instead), but the function
    /// itself must not ALSO add a zero rule keyed on `spec.mark` — only on
    /// `spec.layers` — so this pins the auto path stays untouched if it were
    /// ever reached directly.
    const SINGLE_BAR_SPEC_JSON: &str =
        r#"{"data":{"kind":"named","name":"default"},"mark":"bar","encoding":{"x":{"field":"cat"},"y":{"field":"val"}}}"#;

    /// The shape a real `Summary` transform emits: `mean`, `lower`, `upper`,
    /// all positive and well clear of zero (50..90), so a resolved domain
    /// that reaches zero can only be explained by the bar zero-anchor, never
    /// by the data itself.
    fn catplot_summary_batch() -> RecordBatch {
        use arrow::array::Float64Array;
        let schema = Arc::new(Schema::new(vec![
            Field::new("mean", DataType::Float64, false),
            Field::new("lower", DataType::Float64, false),
            Field::new("upper", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![50.0, 80.0])),
                Arc::new(Float64Array::from(vec![40.0, 70.0])),
                Arc::new(Float64Array::from(vec![60.0, 90.0])),
            ],
        )
        .unwrap()
    }

    /// Resolve one axis's auto-inferred extent for a catplot-shaped lowered
    /// spec: the primary batch carries an unrelated `group` column (mirroring
    /// the real primary batch, which the Summary transform's `mean`/`lower`/
    /// `upper` output columns are absent from), and those three fields live
    /// in the named `"summary"` transform output every layer's `data_source`
    /// references — the same wiring `catplot`'s lowered spec carries.
    fn catplot_axis_extent(spec_json: &str, channel: &str, field: &str) -> (f64, f64) {
        let spec: ChartSpec =
            serde_json::from_str(spec_json).expect("lowered spec JSON must deserialize");
        let primary = group_batch(vec!["a", "b"]);
        let mut outputs = HashMap::new();
        outputs.insert("summary".to_string(), catplot_summary_batch());
        numeric_domain_union(channel, field, None, &primary, &outputs, &spec, false)
            .expect("every field in the fixture is Float64")
    }

    /// Before this fix, the union of `mean` (50..80), `lower` (40..70), and
    /// `upper` (60..90) is `(40.0, 90.0)` — the domain the coder's report
    /// found truncated relative to zero. The `bar` layer's own field
    /// (`mean`) participates in the y-channel union, so the zero-anchor
    /// widens the lower bound to 0, while the whisker's own `upper` extent
    /// (90) stays reachable — the exact property the report's rejected
    /// explicit-`ScaleSpec` fix could not preserve.
    ///
    /// RED (verified in place): commenting out the `if bar_zero_wanted`
    /// block in `numeric_domain_union` makes this fail with `(40.0, 90.0)`.
    #[test]
    fn layered_bar_and_whisker_value_domain_includes_zero() {
        assert_eq!(
            catplot_axis_extent(CATPLOT_BAR_SPEC_JSON, "y", "mean"),
            (0.0, 90.0),
            "a bar-mark layer on the y channel must widen the domain to include \
             zero while keeping the whisker's own extent (90) reachable"
        );
    }

    /// Control: the identical layer shape with the bar layer's mark swapped
    /// to `"point"` must NOT gain the zero-anchor — the union is exactly what
    /// `layered_bar_and_whisker_value_domain_includes_zero` computed BEFORE
    /// the zero-anchor widened it, `(40.0, 90.0)`.
    #[test]
    fn nonbar_layered_spec_value_domain_unchanged() {
        assert_eq!(
            catplot_axis_extent(CATPLOT_NONBAR_SPEC_JSON, "y", "mean"),
            (40.0, 90.0),
            "a layered spec with no bar-mark layer must not gain a zero-anchor"
        );
    }

    /// CoordFlip twin: the authored encoding is unchanged (the bar's value
    /// field is still authored on `y`), but under `.coord(CoordFlip())` that
    /// value is RENDERED on the x axis, so the zero-anchor must land there —
    /// same union arithmetic as the unflipped case (`(40.0, 90.0)` before the
    /// anchor), mirroring `coord_flipped_layer_paired_endpoint_widens_the_other_axis`
    /// above.
    ///
    /// RED (verified in place): commenting out the `if bar_zero_wanted`
    /// block makes this fail with `(40.0, 90.0)` too — proving the anchor
    /// follows the coordinate-space normalization, not just the unflipped
    /// case. Also genuinely discriminating on the flip normalization itself
    /// (rust-quality-review finding, re-measured T5r cycle 2): dropping
    /// `flip_positional` from the detection (reading `layer.encoding`
    /// directly) makes this fail too, with `(50.0, 80.0)` — the bar's `cat`
    /// field is absent from every batch in this fixture, so the mutant's
    /// `value_field_contributed` check for the (wrongly-read) unflipped
    /// value slot fails and the anchor never fires. See
    /// `coord_flipped_bar_position_channel_never_zero_extends` below for the
    /// companion fixture that discriminates the same mutant with real data
    /// on both channels instead of an absent field.
    #[test]
    fn coord_flipped_bar_widens_the_rendered_x_axis() {
        assert_eq!(
            catplot_axis_extent(CATPLOT_BAR_FLIPPED_SPEC_JSON, "x", "mean"),
            (0.0, 90.0),
            "under CoordFlip the bar's authored y (the value field) occupies \
             the rendered x axis and must carry the zero-anchor there"
        );
    }

    /// Single-chart (non-layered) `mark_bar`: this function must not add a
    /// zero-anchor on its own when `spec.layers` is absent — pinned to catch
    /// a future widening of the bar-zero rule from `spec.layers` to
    /// `spec.mark`, which would silently diverge from the explicit-scale
    /// path's `nice`/padding treatment that single-chart bars actually get
    /// in real rendering.
    #[test]
    fn single_chart_bar_auto_path_unchanged() {
        let spec: ChartSpec = serde_json::from_str(SINGLE_BAR_SPEC_JSON)
            .expect("lowered spec JSON must deserialize");
        let batch = {
            use arrow::array::Float64Array;
            let schema = Arc::new(Schema::new(vec![Field::new("val", DataType::Float64, false)]));
            RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(vec![50.0, 80.0]))])
                .unwrap()
        };
        let extent = numeric_domain_union("y", "val", None, &batch, &HashMap::new(), &spec, false)
            .expect("val is Float64");
        assert_eq!(
            extent,
            (50.0, 80.0),
            "a non-layered mark_bar spec must not gain a zero-anchor from this \
             function; Python's explicit scale={{\"zero\": true}} injection owns \
             that case"
        );
    }

    // ── T5r cycle 2 (spec-review issue 1): the two dropped guards ───────────
    //
    // The cycle-1 rule fired for whatever channel was being resolved, as long
    // as a `Mark::Bar` layer bound it — no distinction between the bar's
    // VALUE axis and its POSITION axis, and no exclusion for a layer whose
    // own value channel already carries a paired span field. Both guards
    // mirror `_spec_build.py:394-399`'s single-chart injection exactly
    // (`axis == "y"`, `not _y2_bound`). Missing them is what let a
    // `displot(kind="hist")` histogram bar's `x`/`x2` bin-range POSITION
    // widen to zero — a live regression in this same batch's T4 feature,
    // caught by spec review, not by the cycle-1 test suite (none of the four
    // cycle-1 tests exercised a bar layer's position channel or a `y2`-bound
    // bar).

    /// Verbatim `chart.to_dict()` JSON for `fm.displot(df, x="val",
    /// kind="hist", kde=True)` (captured 2026-09-10, T5r cycle 2) — the exact
    /// shape the spec reviewer measured live: a `bar` histogram layer whose
    /// POSITION channel is `x`/`x2` (the bin range) and whose VALUE channel
    /// is `y` (`count`). The sibling `kde` line layer reads a named
    /// `"kde_overlay"` output this test's fixture omits — its own field,
    /// `value`, unions to nothing and is silently skipped (`union_field`'s
    /// documented "ignore errors" behavior), so omitting it doesn't affect
    /// the assertion below.
    const DISPLOT_HIST_KDE_SPEC_JSON: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"bin_start","type":"quantitative","title":"val"},"y":{"field":"count","type":"quantitative"},"x2":{"field":"bin_end","type":"quantitative"}},"transforms":[{"type":"kde","field":"val","bandwidth":{"kind":"scott"},"bw_adjust":1.0,"n":512,"cumulative":false,"shared_extent":false,"kernel":"gaussian","count_scale":{"kind":"sturges"},"name":"kde_overlay"},{"type":"bin","field":"val","nice":true,"cumulative":false,"shared_extent":false}],"layers":[{"mark":"bar","encoding":{"x":{"field":"bin_start","type":"quantitative","title":"val"},"y":{"field":"count","type":"quantitative"},"x2":{"field":"bin_end","type":"quantitative"}},"position":{"type":"identity"},"name":"histogram"},{"mark":"line","encoding":{"x":{"field":"value"},"y":{"field":"density"}},"data_source":"kde_overlay","position":{"type":"identity"},"name":"kde"}],"position":{"type":"identity"}}"#;

    /// Bin edges offset well clear of zero (50..89), matching the spec
    /// reviewer's reproduction data — a domain that reaches zero can only be
    /// explained by the (wrong) position-channel anchor, never by the data.
    fn displot_hist_batch() -> RecordBatch {
        use arrow::array::Float64Array;
        let schema = Arc::new(Schema::new(vec![
            Field::new("bin_start", DataType::Float64, false),
            Field::new("bin_end", DataType::Float64, false),
            Field::new("count", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![50.0, 70.0])),
                Arc::new(Float64Array::from(vec![70.0, 89.0])),
                Arc::new(Float64Array::from(vec![4.0, 5.0])),
            ],
        )
        .unwrap()
    }

    /// Before the value-channel guard (cycle 1), this incorrectly returned
    /// `(0.0, 89.0)` — the bar layer's POSITION channel (`x`/`x2`, the bin
    /// range) got the anchor only its VALUE channel (`y`, `count`) is
    /// supposed to get. Measured live by the spec reviewer:
    /// `displot(x="val", kind="hist", kde=True)` with data in `[50, 89]`
    /// rendered its x axis `0…90`, squeezing every bar into the right ~44%
    /// of the panel.
    ///
    /// This fixture's position channel also carries `x2` (`bin_end`), so
    /// guard (b) (span-suppression) independently protects it too —
    /// dropping guard (a) alone does NOT turn this test red (verified: the
    /// span guard still catches it). `bar_layer_own_position_channel_without_span_never_zero_extends`
    /// below isolates guard (a) with a position channel that carries no
    /// span field at all, so only guard (a) protects it.
    #[test]
    fn bar_position_channel_never_zero_extends() {
        let spec: ChartSpec = serde_json::from_str(DISPLOT_HIST_KDE_SPEC_JSON)
            .expect("lowered spec JSON must deserialize");
        let batch = displot_hist_batch();
        let extent = numeric_domain_union(
            "x",
            "bin_start",
            Some("bin_end"),
            &batch,
            &HashMap::new(),
            &spec,
            false,
        )
        .expect("bin_start/bin_end are Float64");
        assert_eq!(
            extent,
            (50.0, 89.0),
            "a bar layer's POSITION channel (the bin range) must never gain \
             the value-channel zero-anchor"
        );
    }

    /// Verbatim `chart.to_dict()` JSON for
    /// `fm.Chart(df).mark_bar().encode(x="cat", y="lo", y2="hi") +
    /// fm.Chart(df).mark_point().encode(x="cat", y="lo")` (captured
    /// 2026-09-10, T5r cycle 2) — a bar layer whose own VALUE channel
    /// carries a paired span field, reachable from the public `Chart` API
    /// (`.encode(y2=...)` is a generically honored channel on any mark).
    /// Mirrors the D3 rule `_spec_build.py:379-385` already applies to the
    /// single-chart case: an explicit `[y, y2]` extent is taken literally
    /// and must not be force-anchored to zero.
    const Y2_BOUND_BAR_SPEC_JSON: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"bar","encoding":{"x":{"field":"cat"},"y":{"field":"lo"},"y2":{"field":"hi"}},"layers":[{"mark":"bar","encoding":{"x":{"field":"cat"},"y":{"field":"lo"},"y2":{"field":"hi"}}},{"mark":"point","encoding":{"x":{"field":"cat"},"y":{"field":"lo"}}}]}"#;

    fn y2_bound_bar_batch() -> RecordBatch {
        use arrow::array::Float64Array;
        let schema = Arc::new(Schema::new(vec![
            Field::new("lo", DataType::Float64, false),
            Field::new("hi", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![50.0, 60.0])),
                Arc::new(Float64Array::from(vec![80.0, 90.0])),
            ],
        )
        .unwrap()
    }

    /// The D3 rule: a bar layer that binds its OWN value channel's paired
    /// span field (`y2` here) must not be force-anchored to zero — the
    /// extent stays taken literally from `[lo, hi]` (50..90), exactly the
    /// pre-anchor union.
    ///
    /// RED (verified in place): removing the `layer_fields[1].is_none()`
    /// guard makes this fail with `(0.0, 90.0)`.
    #[test]
    fn y2_bound_bar_layer_never_zero_extends() {
        let spec: ChartSpec = serde_json::from_str(Y2_BOUND_BAR_SPEC_JSON)
            .expect("lowered spec JSON must deserialize");
        let batch = y2_bound_bar_batch();
        let extent =
            numeric_domain_union("y", "lo", Some("hi"), &batch, &HashMap::new(), &spec, false)
                .expect("lo/hi are Float64");
        assert_eq!(
            extent,
            (50.0, 90.0),
            "a bar layer that binds its own value channel's paired span \
             field must not be force-anchored to zero"
        );
    }

    // ── T5r cycle 2 (rust-quality-review): discriminating flip test, the
    // remaining two arithmetic arms, and the `zero=False` opt-out ──────────

    /// `DISPLOT_HIST_KDE_SPEC_JSON` with `.coord(fm.CoordFlip())` appended —
    /// same shape as `LAYERED_Y2_FLIPPED_SPEC_JSON` above ("the same
    /// layering with `.coord(fm.CoordFlip())`"), not itself reachable
    /// through `displot`'s own `orient=` (histograms have none), but a real,
    /// generically composable shape (`CoordFlip` applies to any chart).
    /// Under this flip the histogram's VALUE channel (`y`=`count`,
    /// authored) is RENDERED on `x`, and its POSITION channel (`x`/`x2` =
    /// the bin range, authored) is rendered on `y`/`y2`.
    const DISPLOT_HIST_KDE_FLIPPED_SPEC_JSON: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{"x":{"field":"bin_start","type":"quantitative","title":"val"},"y":{"field":"count","type":"quantitative"},"x2":{"field":"bin_end","type":"quantitative"}},"transforms":[{"type":"kde","field":"val","bandwidth":{"kind":"scott"},"bw_adjust":1.0,"n":512,"cumulative":false,"shared_extent":false,"kernel":"gaussian","count_scale":{"kind":"sturges"},"name":"kde_overlay"},{"type":"bin","field":"val","nice":true,"cumulative":false,"shared_extent":false}],"layers":[{"mark":"bar","encoding":{"x":{"field":"bin_start","type":"quantitative","title":"val"},"y":{"field":"count","type":"quantitative"},"x2":{"field":"bin_end","type":"quantitative"}},"position":{"type":"identity"},"name":"histogram"},{"mark":"line","encoding":{"x":{"field":"value"},"y":{"field":"density"}},"data_source":"kde_overlay","position":{"type":"identity"},"name":"kde"}],"position":{"type":"identity"},"coord":{"kind":"flip"}}"#;

    /// Companion to `coord_flipped_bar_widens_the_rendered_x_axis` above,
    /// with a bar layer that carries REAL numeric data on BOTH channels
    /// (`count` and the `bin_start`/`bin_end` bin range) rather than relying
    /// on the sibling test's category field (`"cat"`) being absent from
    /// every fixture batch to make a coordinate-space mutant visible.
    ///
    /// RED (verified in place, measured — not predicted): reverting the
    /// `let flipped = ...` / `let encoding = ...` pair above to read
    /// `&layer.encoding` directly (dropping the `flip_positional`
    /// normalization) makes the `x` assertion below fail with `(4.0, 89.0)`
    /// — `count`'s own 4..5 range unions with the RAW (unflipped) `x`/`x2`
    /// fields (`bin_start`/`bin_end`, 50..89) because the mutant reads them
    /// onto the wrong slot, and the span-suppression guard (still correctly
    /// seeing an `x2`-shaped pair, just the WRONG pair) blocks the anchor —
    /// so the x axis ends up neither zero-anchored nor at `count`'s own
    /// range. The `y` assertion does not independently discriminate this
    /// mutant (`is_value_channel` is computed from `coord_flipped`
    /// directly, not from the encoding read, so it stays correctly `false`
    /// for `channel == "y"` either way) — kept for its own sake, as the
    /// position-channel-untouched pin under `CoordFlip`. The SAME mutant is
    /// also caught by the pre-existing (non-bar) T13 test
    /// `coord_flipped_layer_paired_endpoint_widens_the_other_axis`, which
    /// exercises the identical normalization for the x2/y2 union.
    #[test]
    fn coord_flipped_bar_position_channel_never_zero_extends() {
        let spec: ChartSpec = serde_json::from_str(DISPLOT_HIST_KDE_FLIPPED_SPEC_JSON)
            .expect("lowered spec JSON must deserialize");
        let batch = displot_hist_batch();
        let x_extent =
            numeric_domain_union("x", "count", None, &batch, &HashMap::new(), &spec, false)
                .expect("count is Float64");
        assert_eq!(
            x_extent,
            (0.0, 5.0),
            "under CoordFlip the histogram's VALUE channel (count, authored \
             y) is rendered on x and must carry the zero-anchor"
        );
        let y_extent = numeric_domain_union(
            "y",
            "bin_start",
            Some("bin_end"),
            &batch,
            &HashMap::new(),
            &spec,
            false,
        )
        .expect("bin_start/bin_end are Float64");
        assert_eq!(
            y_extent,
            (50.0, 89.0),
            "under CoordFlip the histogram's POSITION channel (the bin \
             range, authored x/x2) is rendered on y/y2 and must never gain \
             the value-channel zero-anchor"
        );
    }

    /// Verbatim `chart.to_dict()` JSON for `fm.Chart(df).mark_rank1d(orient="vertical")`
    /// (captured 2026-09-10, T5r cycle 2) — a REAL single-bar-layer
    /// production shape (`marks/diagnostic/_ranking.py:43`,
    /// `desugar_rank1d`'s vertical orientation: `x=feature`, `y=score`).
    /// Reused (with different fixture batches) for the `mx < 0.0` arm and
    /// the zero-before-degenerate-widening ordering test below.
    const SINGLE_BAR_LAYER_SPEC_JSON: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{},"layers":[{"mark":"bar","encoding":{"x":{"field":"feature"},"y":{"field":"score"}},"name":"bar"}]}"#;

    /// Isolates guard (a) alone: `SINGLE_BAR_LAYER_SPEC_JSON`'s bar layer
    /// has NO span field on either channel, so guard (b)
    /// (`layer_fields[1].is_none()`) is trivially satisfied regardless of
    /// which channel is being resolved — unlike
    /// `bar_position_channel_never_zero_extends` and the `x`/`y` pair in
    /// `coord_flipped_bar_position_channel_never_zero_extends`, whose
    /// position channels carry `x2`/`y2` and so are ALSO (redundantly)
    /// protected by guard (b). Querying the POSITION channel (`x` =
    /// `feature`) directly with real, contributing Float64 data means ONLY
    /// guard (a) stands between this layer and a wrong zero-anchor.
    ///
    /// RED (verified in place): replacing `is_value_channel` with `true` in
    /// the detection condition makes this fail with `(0.0, 90.0)`.
    #[test]
    fn bar_layer_own_position_channel_without_span_never_zero_extends() {
        let spec: ChartSpec = serde_json::from_str(SINGLE_BAR_LAYER_SPEC_JSON)
            .expect("lowered spec JSON must deserialize");
        let batch = {
            use arrow::array::Float64Array;
            let schema =
                Arc::new(Schema::new(vec![Field::new("feature", DataType::Float64, false)]));
            RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(vec![50.0, 90.0]))])
                .unwrap()
        };
        let extent =
            numeric_domain_union("x", "feature", None, &batch, &HashMap::new(), &spec, false)
                .expect("feature is Float64 in this fixture");
        assert_eq!(
            extent,
            (50.0, 90.0),
            "a bar layer's own POSITION channel must never gain the \
             value-channel zero-anchor, even with no span field to fall \
             back on for protection"
        );
    }

    /// Every value negative (`-3.0`, `-8.0`) — a bar chart of negative
    /// scores anchors from ABOVE, not below: `mx < 0.0 → mx = 0.0`, `mn`
    /// (already `<= 0.0`) untouched.
    ///
    /// RED (verified in place): deleting the `if mx < 0.0 { mx = 0.0; }`
    /// arm makes this fail with `(-8.0, -3.0)`.
    #[test]
    fn negative_bar_values_anchor_from_above() {
        let spec: ChartSpec = serde_json::from_str(SINGLE_BAR_LAYER_SPEC_JSON)
            .expect("lowered spec JSON must deserialize");
        let batch = {
            use arrow::array::Float64Array;
            let schema = Arc::new(Schema::new(vec![Field::new("score", DataType::Float64, false)]));
            RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(vec![-3.0, -8.0]))])
                .unwrap()
        };
        let extent = numeric_domain_union("y", "score", None, &batch, &HashMap::new(), &spec, false)
            .expect("score is Float64");
        assert_eq!(
            extent,
            (-8.0, 0.0),
            "an all-negative bar layer must anchor from above (mx -> 0.0), \
             not below"
        );
    }

    /// A single row (one category, one score `5.0`): pre-anchor `mn == mx
    /// == 5.0`, the degenerate case. Running the zero-anchor BEFORE the
    /// degenerate-domain widening (as `numeric_domain_union` does) resolves
    /// this to the real `[0, 5]` span; running it AFTER (widening first,
    /// `4.5..5.5`, THEN zero-extending `4.5 -> 0.0`) would instead yield the
    /// wrong `(0.0, 5.5)`.
    ///
    /// RED (verified in place): moving the `if bar_zero_wanted { ... }`
    /// block below the "Degenerate domain" widening block makes this fail
    /// with `(0.0, 5.5)`.
    #[test]
    fn single_row_bar_value_zero_extends_before_degenerate_widening() {
        let spec: ChartSpec = serde_json::from_str(SINGLE_BAR_LAYER_SPEC_JSON)
            .expect("lowered spec JSON must deserialize");
        let batch = {
            use arrow::array::Float64Array;
            let schema = Arc::new(Schema::new(vec![Field::new("score", DataType::Float64, false)]));
            RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(vec![5.0]))]).unwrap()
        };
        let extent = numeric_domain_union("y", "score", None, &batch, &HashMap::new(), &spec, false)
            .expect("score is Float64");
        assert_eq!(
            extent,
            (0.0, 5.0),
            "the zero-anchor must run BEFORE the degenerate-domain widening, \
             not after"
        );
    }

    /// `SINGLE_BAR_LAYER_SPEC_JSON` with `"mark_style": {"zero": false}`
    /// hand-added to the bar layer — the per-layer mirror of the
    /// single-chart `mark_bar(zero=False)` opt-out (`MarkKwargsSpec::zero`,
    /// T5r cycle 2). Hand-authored rather than captured from a live call on
    /// THIS fixture's own shape (`mark_rank1d`), but the field itself is
    /// live and user-reachable on the wire this test pins: see
    /// `MarkKwargsSpec::zero`'s doc comment for the verbatim, measured
    /// `fm.catplot(..., kind="bar", mark={"bar": {"zero": False}})` shape
    /// that reaches this exact `"mark_style":{"zero":false}` key today.
    const ZERO_FALSE_BAR_LAYER_SPEC_JSON: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{},"layers":[{"mark":"bar","encoding":{"x":{"field":"feature"},"y":{"field":"score"}},"mark_style":{"zero":false},"name":"bar"}]}"#;

    /// Otherwise-eligible bar layer (value channel, no paired span, data
    /// contributes) but explicitly opts out via `mark_style.zero == false`
    /// — the extent must stay the raw data range, not zero-anchored.
    ///
    /// RED (verified in place): dropping the `layer_wants_zero` guard makes
    /// this fail with `(0.0, 80.0)`.
    #[test]
    fn zero_false_layer_opts_out_of_anchor() {
        let spec: ChartSpec = serde_json::from_str(ZERO_FALSE_BAR_LAYER_SPEC_JSON)
            .expect("lowered spec JSON must deserialize");
        let batch = {
            use arrow::array::Float64Array;
            let schema = Arc::new(Schema::new(vec![Field::new("score", DataType::Float64, false)]));
            RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(vec![50.0, 80.0]))])
                .unwrap()
        };
        let extent = numeric_domain_union("y", "score", None, &batch, &HashMap::new(), &spec, false)
            .expect("score is Float64");
        assert_eq!(
            extent,
            (50.0, 80.0),
            "mark_style.zero == Some(false) must suppress the layered bar \
             zero-anchor, mirroring mark_bar(zero=False)"
        );
    }

    /// A `point` layer with a real field (`existing`) plus a sibling `bar`
    /// layer whose declared value field (`phantom`) is absent from every
    /// batch (no `data_source` resolves it, and it isn't in the primary
    /// batch either) — the bar layer's NAME alone must not force a
    /// zero-anchor when its own field never contributed a value
    /// (rust-quality-review finding 3).
    const PHANTOM_FIELD_BAR_LAYER_SPEC_JSON: &str = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{},"layers":[{"mark":"point","encoding":{"y":{"field":"existing"}}},{"mark":"bar","encoding":{"x":{"field":"cat"},"y":{"field":"phantom"}}}]}"#;

    /// RED (verified in place): replacing `value_field_contributed` with
    /// `layer_fields[0].is_some()` (the cycle-1 shape, "declared" instead of
    /// "contributed") makes this fail with `(0.0, 20.0)` — the bar layer's
    /// merely-declared `phantom` field forces the anchor even though it
    /// never resolves to data anywhere.
    #[test]
    fn phantom_field_bar_layer_never_zero_extends() {
        let spec: ChartSpec = serde_json::from_str(PHANTOM_FIELD_BAR_LAYER_SPEC_JSON)
            .expect("lowered spec JSON must deserialize");
        let batch = {
            use arrow::array::Float64Array;
            let schema =
                Arc::new(Schema::new(vec![Field::new("existing", DataType::Float64, false)]));
            RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(vec![10.0, 20.0]))])
                .unwrap()
        };
        let extent =
            numeric_domain_union("y", "existing", None, &batch, &HashMap::new(), &spec, false)
                .expect("existing is Float64");
        assert_eq!(
            extent,
            (10.0, 20.0),
            "a bar layer whose declared value field never resolves to data \
             must not force a zero-anchor from its name alone"
        );
    }
}
