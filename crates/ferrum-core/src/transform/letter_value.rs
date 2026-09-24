//! Letter-value transform (Phase 9b).
//!
//! Computes the letter-value statistics used by boxen plots
//! (Hofmann/Wickham/Kafadar 2017). For each group, emits one row per
//! depth k ∈ [1, K] with the lower/upper quantiles at probabilities
//! `2^-k` and `1 - 2^-k`. The number of depths K is chosen by one of
//! four strategies.
//!
//! Primary output schema:
//! `[group: <original group-column dtype>|Null, depth: Int32, lower: Float64, upper: Float64, level: Utf8]`
//! The `group` column's dtype matches `spec.group`'s ORIGINAL input dtype
//! (any dtype [`crate::transform::group_key::validate_groupby_columns`]
//! accepts — the same set every groupby-taking sibling accepts), not a
//! forced Utf8 stringification; an ungrouped call (`spec.group = None`)
//! still emits an all-null Utf8 `group` column, matching prior behavior.
//!
//! Secondary named outputs:
//! - `outliers` (or `<name>_outliers`): `[group: <original dtype>|Null, value: Float64, is_outlier: Bool]`,
//!   filtered to rows where the input value falls outside the outermost letter-value
//!   fence (the flag is therefore always `true` on emitted rows — it survives in the
//!   schema for wire-compatibility with the pre-filter contract).
//! - `depth_1`..`depth_12` (or `<name>_depth_K`): `[group, lower, upper, level]` rows
//!   filtered to that depth. When the actual K < 12, the unused outputs are emitted
//!   as zero-row batches so downstream renderers can safely consume them.

use arrow::array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int32Array, RecordBatch, StringArray,
};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::transform::context::{describe_group_keys, TransformContext, TransformWarning};
use crate::transform::group_key::{
    group_partition_multi, groupby_key_at, materialize_groupby_col, validate_groupby_columns,
    KeyValue, NullKeyPolicy,
};

/// Maximum depth for per-depth named outputs.
pub(crate) const K_MAX_NAMED: usize = 12;

/// Letter labels by depth. depth 1 → "M" (median), 2 → "F" (fourths), ...,
/// 8 → "Z", then the conventional Tukey letter-value continuation past Z:
/// 9 → "Y", 10 → "X", 11 → "W", 12 → "V". Sized to `K_MAX_NAMED` so every
/// depth the transform can emit gets a real letter — no depth ever falls
/// back to the generic `depth_N` spelling on a user-facing axis.
const LETTERS: [&str; K_MAX_NAMED] =
    ["M", "F", "E", "D", "C", "B", "A", "Z", "Y", "X", "W", "V"];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum KDepth {
    Tukey,
    Proportion { p: f64 },
    Trustworthy,
    Full,
}

/// Aggregate operation used to order boxen groups along the categorical axis.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SortOp {
    Sum,
    Mean,
    Max,
    Min,
    Count,
    Median,
}

/// Direction of the group ordering.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SortOrder {
    Asc,
    Desc,
}

/// Optional group ordering for the letter-value transform.
///
/// When present, output groups are emitted in the order produced by ranking
/// each group's per-group aggregate of the (non-null) value column. Because
/// the categorical axis domain is built from the first-appearance order of the
/// `group` column in the primary batch, emitting groups in this order is what
/// makes the boxen categorical axis come out sorted — no separate layer sort or
/// domain step is required.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct LetterValueSort {
    pub op: SortOp,
    pub order: SortOrder,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct LetterValueSpec {
    pub value: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub group: Option<String>,
    pub k_depth: KDepth,
    #[serde(default = "default_outlier_threshold")]
    pub outlier_threshold: f64,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub sort: Option<LetterValueSort>,
}

fn default_outlier_threshold() -> f64 { 1.5 }

/// The K depth a strategy computes for group size `n`, BEFORE the
/// [`K_MAX_NAMED`] cap. [`k_for_depth`] applies the cap; comparing this
/// against the capped result is how [`compute_depths`] detects (and reports,
/// #162) that a group's depth was silently reduced from what the strategy
/// actually wanted.
fn k_for_depth_uncapped(strategy: &KDepth, n: usize) -> usize {
    if n == 0 { return 0; }
    let n_f = n as f64;
    match strategy {
        KDepth::Tukey => {
            let raw = n_f.log2().floor() as i64 - 3;
            raw.max(1) as usize
        }
        KDepth::Proportion { p } => {
            let target = (p * n_f).ceil() as i64;
            let target = target.max(1);
            let mut k = 1usize;
            // Uncapped: this loop already stops itself once `k` PASSES
            // `K_MAX_NAMED` (one past it, at most), so it cannot run away for
            // a pathologically large `target`; `k_for_depth`'s `.min(K_MAX_NAMED)`
            // still owns the actual cap.
            while (1i64 << (k as i64 - 1)) < target {
                k += 1;
                if k > K_MAX_NAMED { break; }
            }
            k.max(1)
        }
        KDepth::Trustworthy => {
            // K = floor(log2(n / 3.32)).
            let val = (n_f / 3.32_f64).log2().floor() as i64;
            val.max(1) as usize
        }
        KDepth::Full => {
            let raw = n_f.log2().floor() as i64;
            raw.max(1) as usize
        }
    }
}

/// Compute the K depth for a given strategy and group size n, capped at
/// [`K_MAX_NAMED`] — the number of per-depth named outputs and Tukey letters
/// this transform can emit.
pub(crate) fn k_for_depth(strategy: &KDepth, n: usize) -> usize {
    k_for_depth_uncapped(strategy, n).min(K_MAX_NAMED)
}

use crate::transform::numeric_util::{coerce_to_float64, quantile_sorted};

struct DepthRow {
    group: KeyValue,
    depth: i32,
    lower: f64,
    upper: f64,
    level: String,
}

/// Extract `(group key, value)` pairs (dropping null/NaN values) plus the
/// group column's dtype for output materialization.
///
/// `spec.group = None` (ungrouped) reports [`DataType::Utf8`] — matching the
/// prior all-null Utf8 `group` output column — and every row's key is
/// [`KeyValue::Null`]. A `spec.group = Some(col)` is resolved and dtype-checked
/// through the shared [`validate_groupby_columns`] (the same request-level
/// validator `group_partition_multi` itself calls, and every groupby-taking
/// sibling transform uses), rather than a hand-duplicated existence lookup
/// and dtype gate — one rule, one wording, enforced once. Every supported
/// dtype (the same set every groupby-taking sibling transform accepts) is
/// then read through the shared [`groupby_key_at`] extractor, so a null group
/// VALUE (as opposed to no group column at all) also maps to
/// `KeyValue::Null` and the two null-group and no-group cases pool together,
/// exactly as the prior `Option<String>` representation did.
fn extract_values(spec: &LetterValueSpec, batch: &RecordBatch) -> PyResult<(Vec<(KeyValue, f64)>, DataType)> {
    let schema = batch.schema();
    let vi = schema.index_of(&spec.value)
        .map_err(|_| PyValueError::new_err(format!("stat_letter_value: column '{}' not found", spec.value)))?;
    let varr = coerce_to_float64(batch.column(vi), "stat_letter_value", &spec.value)?;

    let group_cols: Vec<String> = spec.group.iter().cloned().collect();
    let resolved = validate_groupby_columns(batch, &group_cols, "stat_letter_value")?;
    let group_info: Option<(ArrayRef, DataType)> =
        resolved.first().map(|(gi, dt)| (batch.column(*gi).clone(), dt.clone()));
    let group_dtype = group_info.as_ref().map(|(_, dt)| dt.clone()).unwrap_or(DataType::Utf8);

    let mut out = Vec::with_capacity(varr.len());
    for i in 0..varr.len() {
        if varr.is_null(i) { continue; }
        let v = varr.value(i);
        if v.is_nan() { continue; }
        let key = match &group_info {
            None => KeyValue::Null,
            Some((arr, dt)) => groupby_key_at(arr.as_ref(), dt, i).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "stat_letter_value: internal error extracting group key at row {i}"
                ))
            })?,
        };
        out.push((key, v));
    }
    Ok((out, group_dtype))
}

pub(crate) fn apply(spec: &LetterValueSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    apply_with_context(spec, batch, &TransformContext::default())
}

/// Context-aware entry point (#162): identical to [`apply`], plus
/// `ctx.warnings` as the sink for [`compute_depths`]'s per-group
/// [`K_MAX_NAMED`] depth-cap warning.
pub(crate) fn apply_with_context(
    spec: &LetterValueSpec,
    batch: &RecordBatch,
    ctx: &TransformContext,
) -> PyResult<RecordBatch> {
    if !(spec.outlier_threshold.is_finite() && spec.outlier_threshold >= 0.0) {
        return Err(PyValueError::new_err(
            "stat_letter_value: outlier_threshold must be finite and >= 0",
        ));
    }
    let (depths, group_dtype) = compute_depths(spec, batch, ctx)?;
    build_primary_batch(&depths, &group_dtype)
}

/// Determine the order in which groups are emitted.
///
/// With `sort = None` the order is lexicographic over the (optional) group key
/// — identical to the historical `BTreeMap` iteration order. With `sort =
/// Some`, groups are ranked by the requested per-group aggregate over the
/// non-null, non-NaN value rows [`compute_depths`] assigns to each group (so
/// `count` counts only the rows that boxen actually plots). Ties are broken
/// lexicographically for byte-deterministic output.
///
/// This ranking has no equivalent in [`group_partition_multi`] —
/// no other groupby-taking transform in the crate ranks its groups by an
/// aggregate of the value column — so it stays letter_value's own
/// post-partition step rather than something to route through the shared
/// toolkit; [`compute_depths`] applies it AFTER the shared partitioner has
/// already resolved which rows belong to which group.
fn group_order(
    sort: &Option<LetterValueSort>,
    grouped: &BTreeMap<KeyValue, Vec<f64>>,
) -> Vec<KeyValue> {
    // BTreeMap keys are already ordered by KeyValue's Ord impl (Null first,
    // then the typed variants in their natural order — lexicographic for a
    // Utf8 group column, numeric for an Int/UInt/Float/Bool one); this is
    // the `sort = None` answer and the tie-break baseline.
    let lexicographic: Vec<KeyValue> = grouped.keys().cloned().collect();
    let Some(sort) = sort else {
        return lexicographic;
    };

    let aggregate = |vs: &[f64]| -> f64 {
        match sort.op {
            SortOp::Count => vs.len() as f64,
            SortOp::Sum => vs.iter().sum(),
            SortOp::Mean => {
                if vs.is_empty() { f64::NAN } else { vs.iter().sum::<f64>() / vs.len() as f64 }
            }
            SortOp::Max => vs.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            SortOp::Min => vs.iter().copied().fold(f64::INFINITY, f64::min),
            SortOp::Median => {
                if vs.is_empty() {
                    f64::NAN
                } else {
                    let mut s = vs.to_vec();
                    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    quantile_sorted(&s, 0.5)
                }
            }
        }
    };

    // Rank lexicographic order by aggregate; stable sort keeps lexicographic
    // order among ties, then we reverse for descending.
    let mut ordered = lexicographic;
    ordered.sort_by(|a, b| {
        let av = aggregate(grouped.get(a).map(Vec::as_slice).unwrap_or(&[]));
        let bv = aggregate(grouped.get(b).map(Vec::as_slice).unwrap_or(&[]));
        let cmp = av.partial_cmp(&bv).unwrap_or(std::cmp::Ordering::Equal);
        match sort.order {
            SortOrder::Asc => cmp,
            SortOrder::Desc => cmp.reverse(),
        }
        // Tie-break: BTreeMap order is already lexicographic and `sort_by` is
        // stable, so equal aggregates retain ascending-lexicographic order
        // under both directions.
    });
    ordered
}

/// Resolve the value column, partition `batch` by `spec.group` through the
/// shared [`group_partition_multi`] (the same partitioner
/// `box_stats`/`violin`/`outliers`/`smooth`/... use — existence check, dtype
/// validation, and row-to-group assignment all live there once, not
/// re-hand-rolled here), then compute each group's letter-value depths.
///
/// **`NullKeyPolicy::Pool`, not the more common `Skip`.** MOST
/// `group_partition_multi` callers in the crate use `Skip` (a null-keyed row
/// belongs to no group and is dropped), because those transforms append the
/// group key as an output column and a row with no key has no row to write
/// it to (GH #197). But `Pool` is not exceptional in the crate — three
/// siblings already use it: `data_stack.rs` and `join_aggregate.rs` (named
/// directly in [`NullKeyPolicy::Pool`]'s own doc, `group_key/mod.rs`) are
/// row-preserving transforms whose contract is one output value per INPUT
/// row and which therefore append NO group column at all — for them,
/// skipping a null-keyed row would leave it silently holding a default
/// value instead of a computed one. `data_window.rs` takes the same
/// exemption for the same row-preserving reason. letter_value's reason for
/// `Pool` is different from all three: unlike them, it DOES append a group
/// column, and its pre-existing, tested contract is different from `Skip`
/// on two points `Skip` would break silently:
///   - **Null-group pooling.** A null-valued group row has always joined a
///     single null-keyed group (materialized as an Arrow null in the `group`
///     output column) rather than being dropped — this is what lets a boxen
///     chart draw a "no category" box instead of silently losing those rows.
///     `Pool` reproduces this exactly: `group_partition_multi`'s partitioning
///     loop only `continue`s past a null-keyed row when
///     `null_keys.skips_null_rows()` is true, which is `!matches!(self,
///     Pool)` — so under `Pool` every null-valued-group row still gets
///     pushed into the `[KeyValue::Null]` group instead of being excluded.
///   - **Emission order.** `group_partition_multi` returns groups in
///     FIRST-APPEARANCE order (GH #179's disclosed policy for its Skip
///     callers). letter_value's `sort = None` contract is
///     LEXICOGRAPHIC/numeric order
///     (`test_letter_value_sort_none_is_lexicographic`, unchanged since
///     before this migration), and `sort = Some(...)` ranks groups by a
///     per-group value aggregate `group_partition_multi` has no equivalent
///     for. Neither depends on partition-return order: [`group_order`] is
///     applied AFTER partitioning, as a deliberate post-partition reorder
///     over the same `BTreeMap<KeyValue, Vec<f64>>` shape used pre-migration
///     — the partitioner supplies WHICH rows belong to which group and its
///     null policy; `group_order` decides emission order. `Skip` +
///     first-appearance order was ruled out, not left untried: swapping to
///     it would have to also drop the null-pooling contract (no test pins
///     it, but nothing asked to change it) and would have made
///     first-appearance order the ordering [`group_order`] has to override
///     unconditionally for both its branches — replacing a real policy this
///     module already owns with a policy borrowed from a different transform
///     shape, not a simplification.
fn compute_depths(
    spec: &LetterValueSpec,
    batch: &RecordBatch,
    ctx: &TransformContext,
) -> PyResult<(Vec<DepthRow>, DataType)> {
    let schema = batch.schema();
    let vi = schema.index_of(&spec.value)
        .map_err(|_| PyValueError::new_err(format!("stat_letter_value: column '{}' not found", spec.value)))?;
    let varr = coerce_to_float64(batch.column(vi), "stat_letter_value", &spec.value)?;

    let group_cols: Vec<String> = spec.group.iter().cloned().collect();
    let partition = group_partition_multi(batch, &group_cols, "stat_letter_value", NullKeyPolicy::Pool, ctx)?;
    let group_dtype = partition.dtypes.first().cloned().unwrap_or(DataType::Utf8);

    // Per-group (non-null, non-NaN) sorted values. The partitioner decides
    // WHICH rows belong to which group; filtering by the VALUE column's own
    // null/NaN rows is this transform's job, same shape as
    // `box_stats::apply_with_context`'s per-group `filter_map` over its
    // partition's row indices.
    //
    // A group whose rows are ALL null/NaN filters down to an empty `vs`
    // here — a group the pre-migration code could never see, since it built
    // this map from already-filtered `(key, value)` pairs rather than raw
    // partitioned rows. Such a group must NOT be inserted:
    // `group_order`'s `sort=Some(mean|median)` aggregate over an empty slice
    // is `NaN`, and `NaN.partial_cmp(_)` is `None`, which the comparator
    // degrades to `Ordering::Equal` against every other group — measured
    // (not just predicted) to silently INVERT the real groups' requested
    // order (`letter_value_all_nan_group_does_not_corrupt_sort_order`).
    // Skipping the insert here restores the exact invariant the
    // pre-migration `pairs`-based map always had: every key in `groups` has
    // at least one real value.
    let mut groups: BTreeMap<KeyValue, Vec<f64>> = BTreeMap::new();
    for (key, rows) in &partition.groups {
        let g = key.first().cloned().unwrap_or(KeyValue::Null);
        let mut vs: Vec<f64> = rows.iter().filter_map(|&r| {
            if varr.is_null(r) { return None; }
            let v = varr.value(r);
            if v.is_nan() { return None; }
            Some(v)
        }).collect();
        if vs.is_empty() {
            continue;
        }
        vs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        groups.insert(g, vs);
    }
    let order = group_order(&spec.sort, &groups);

    let mut rows = Vec::new();
    for g in &order {
        let vs = match groups.get(g) {
            Some(vs) if !vs.is_empty() => vs,
            _ => continue,
        };
        let raw_k = k_for_depth_uncapped(&spec.k_depth, vs.len());
        let k = k_for_depth(&spec.k_depth, vs.len());
        if raw_k > K_MAX_NAMED {
            // #162's named case: the strategy computed a depth deeper than
            // this transform can emit (12 Tukey letters / named per-depth
            // outputs) and it was silently capped.
            // `g` is the group KEY (e.g. `KeyValue::Str("big")` or
            // `KeyValue::Int(6)`); `TransformWarning::group`'s documented
            // contract is a `"col=value"` label (`context.rs`'s doc), built
            // here with the same `describe_group_keys` formatter every
            // sibling adopter (`smooth.rs`, `robust.rs`) uses, so the label
            // text cannot drift from theirs. `g` is `KeyValue::Null` in
            // exactly the two cases with no real "col=value" to report: no
            // group column at all, or this group's own key being null —
            // `spec.group.is_some()` disambiguates the latter from the
            // former, but both fall back to `group: None`.
            let group = if matches!(g, KeyValue::Null) {
                None
            } else {
                spec.group.as_ref().map(|col| {
                    describe_group_keys(std::slice::from_ref(col), std::slice::from_ref(g))
                })
            };
            ctx.warn(TransformWarning::ParameterClamped {
                transform: "LetterValue".into(),
                group,
                param: "k_depth".into(),
                requested: raw_k.to_string(),
                applied: k.to_string(),
            });
        }
        for depth in 1..=k {
            let p = 0.5_f64.powi(depth as i32);
            let lower = quantile_sorted(vs, p);
            let upper = quantile_sorted(vs, 1.0 - p);
            // `k` is clamped to `K_MAX_NAMED` (== `LETTERS.len()`) by
            // `k_for_depth`, so `depth` can never exceed `LETTERS.len()` here;
            // the `depth_{depth}` fallback is defensive only — kept so a
            // future divergence between the two constants degrades to a
            // readable label instead of an out-of-bounds panic.
            let letter = if depth >= 1 && depth <= LETTERS.len() {
                LETTERS[depth - 1].to_string()
            } else {
                format!("depth_{depth}")
            };
            rows.push(DepthRow {
                group: g.clone(),
                depth: depth as i32,
                lower,
                upper,
                level: letter,
            });
        }
    }
    Ok((rows, group_dtype))
}

fn build_primary_batch(rows: &[DepthRow], group_dtype: &DataType) -> PyResult<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("group", group_dtype.clone(), true),
        Field::new("depth", DataType::Int32, false),
        Field::new("lower", DataType::Float64, false),
        Field::new("upper", DataType::Float64, false),
        Field::new("level", DataType::Utf8, false),
    ]));
    let group_keys: Vec<Vec<KeyValue>> = rows.iter().map(|r| vec![r.group.clone()]).collect();
    let group_col =
        materialize_groupby_col(&group_keys, 0, group_dtype).map_err(PyValueError::new_err)?;
    let depths: Vec<i32> = rows.iter().map(|r| r.depth).collect();
    let lowers: Vec<f64> = rows.iter().map(|r| r.lower).collect();
    let uppers: Vec<f64> = rows.iter().map(|r| r.upper).collect();
    let levels: Vec<String> = rows.iter().map(|r| r.level.clone()).collect();
    let cols: Vec<ArrayRef> = vec![
        group_col,
        Arc::new(Int32Array::from(depths)),
        Arc::new(Float64Array::from(lowers)),
        Arc::new(Float64Array::from(uppers)),
        Arc::new(StringArray::from(levels.iter().map(|s| s.as_str()).collect::<Vec<_>>())),
    ];
    RecordBatch::try_new(schema, cols)
        .map_err(|e| PyValueError::new_err(format!("stat_letter_value: {e}")))
}

/// The outlier batch's key under default construction; a spec `name` prefixes
/// it as `"{name}_outliers"`.
pub(crate) const OUTLIERS_KEY: &str = "outliers";

/// The outlier batch's column schema — read by [`secondary_outputs`]'s schema
/// builder and by `registry::entry_for_kind("letter_value")` (R1 design
/// finding (d)).
pub(crate) const OUTLIERS_COLUMNS: &[(&str, DataType)] = &[
    ("group", DataType::Utf8),
    ("value", DataType::Float64),
    ("is_outlier", DataType::Boolean),
];

/// Every per-depth band batch's column schema. `depth` itself is implicit in
/// the output's key, so it is not a column here.
pub(crate) const DEPTH_COLUMNS: &[(&str, DataType)] = &[
    ("group", DataType::Utf8),
    ("lower", DataType::Float64),
    ("upper", DataType::Float64),
    ("level", DataType::Utf8),
];

/// The key of the depth-`k` band batch under default construction; a spec
/// `name` prefixes it as `"{name}_depth_{k}"`. One key per `k` in
/// `1..=K_MAX_NAMED` is always published, empty when that depth has no rows.
pub(crate) fn depth_key(k: usize) -> String {
    format!("depth_{k}")
}

/// Compute the secondary named outputs: outliers + depth_1..depth_12.
pub(crate) fn secondary_outputs(
    spec: &LetterValueSpec,
    input_batch: &RecordBatch,
    primary_batch: &RecordBatch,
) -> PyResult<Vec<(String, RecordBatch)>> {
    let group_dtype = primary_batch.schema().field(0).data_type().clone();
    let groups_col = primary_batch.column(0);
    let depths_col = primary_batch.column(1).as_any().downcast_ref::<Int32Array>()
        .ok_or_else(|| PyValueError::new_err("stat_letter_value: expected Int32Array for 'depth' column"))?;
    let lowers_col = primary_batch.column(2).as_any().downcast_ref::<Float64Array>()
        .ok_or_else(|| PyValueError::new_err("stat_letter_value: expected Float64Array for 'lower' column"))?;
    let uppers_col = primary_batch.column(3).as_any().downcast_ref::<Float64Array>()
        .ok_or_else(|| PyValueError::new_err("stat_letter_value: expected Float64Array for 'upper' column"))?;

    // Reads row `i`'s group key from the primary batch's "group" column at
    // its own (possibly non-Utf8) dtype, via the shared extractor rather than
    // a hardcoded `StringArray` downcast.
    let group_at = |i: usize| -> PyResult<KeyValue> {
        groupby_key_at(groups_col.as_ref(), &group_dtype, i).ok_or_else(|| {
            PyValueError::new_err(format!(
                "stat_letter_value: internal error extracting group key at row {i}"
            ))
        })
    };

    // Per-group max-depth (lower, upper) extents — the outermost letter value.
    let mut tail: BTreeMap<KeyValue, (i32, f64, f64)> = BTreeMap::new();
    for i in 0..primary_batch.num_rows() {
        let g = group_at(i)?;
        let d = depths_col.value(i);
        let lo = lowers_col.value(i);
        let hi = uppers_col.value(i);
        let entry = tail.entry(g).or_insert((d, lo, hi));
        if d >= entry.0 {
            *entry = (d, lo, hi);
        }
    }

    // ---- outliers output ----
    // For each non-null input row: classify as outlier if value falls outside
    //   [lower_K - threshold*(upper_K - lower_K), upper_K + threshold*(upper_K - lower_K)].
    // Only rows that ARE outliers are emitted (schema unchanged: `is_outlier`
    // survives as a column but is always `true` on the rows that reach it) —
    // the batch is a filtered outlier list, not a flagged copy of every input
    // row, so a boxen chart's outlier layer draws only the points it names.
    let (input_pairs, _) = extract_values(spec, input_batch)?;
    let mut o_groups: Vec<KeyValue> = Vec::new();
    let mut o_vals: Vec<f64> = Vec::new();
    let mut o_flags: Vec<bool> = Vec::new();
    for (g, v) in &input_pairs {
        let (lo, hi) = match tail.get(g) {
            Some((_, lo, hi)) => (*lo, *hi),
            None => (f64::NAN, f64::NAN),
        };
        let span = hi - lo;
        let low_fence = lo - spec.outlier_threshold * span;
        let high_fence = hi + spec.outlier_threshold * span;
        let is_outlier = !lo.is_nan() && !hi.is_nan() && (*v < low_fence || *v > high_fence);
        if !is_outlier {
            continue;
        }
        o_groups.push(g.clone());
        o_vals.push(*v);
        o_flags.push(is_outlier);
    }
    // Column names/nullability still come from the shared consts (no drift
    // risk); only the "group" entry's declared dtype is overridden to this
    // call's actual group dtype, since the consts describe the schema under
    // DEFAULT (ungrouped, Utf8) construction — see their doc comments and
    // `registry::letter_value_secondary_outputs`.
    let outliers_schema = Arc::new(Schema::new(
        OUTLIERS_COLUMNS
            .iter()
            .map(|(name, dt)| {
                let dt = if *name == "group" { group_dtype.clone() } else { dt.clone() };
                Field::new(*name, dt, *name == "group")
            })
            .collect::<Vec<_>>(),
    ));
    let o_group_keys: Vec<Vec<KeyValue>> = o_groups.iter().map(|k| vec![k.clone()]).collect();
    let o_group_col =
        materialize_groupby_col(&o_group_keys, 0, &group_dtype).map_err(PyValueError::new_err)?;
    let outliers_batch = RecordBatch::try_new(
        outliers_schema,
        vec![
            o_group_col,
            Arc::new(Float64Array::from(o_vals)) as ArrayRef,
            Arc::new(BooleanArray::from(o_flags)) as ArrayRef,
        ],
    ).map_err(|e| PyValueError::new_err(format!("stat_letter_value outliers: {e}")))?;

    // ---- per-depth outputs (depth_1..depth_12) ----
    // Each depth_K output is the primary rows filtered to depth==K, projecting only
    // [group, lower, upper, level] (no depth column — implicit in the output name).
    let depth_schema = Arc::new(Schema::new(
        DEPTH_COLUMNS
            .iter()
            .map(|(name, dt)| {
                let dt = if *name == "group" { group_dtype.clone() } else { dt.clone() };
                Field::new(*name, dt, *name == "group")
            })
            .collect::<Vec<_>>(),
    ));
    let level_col = primary_batch.column(4).as_any().downcast_ref::<StringArray>()
        .ok_or_else(|| PyValueError::new_err("stat_letter_value: expected StringArray for 'level' column"))?;
    let mut by_depth: BTreeMap<i32, (Vec<KeyValue>, Vec<f64>, Vec<f64>, Vec<String>)> = BTreeMap::new();
    for i in 0..primary_batch.num_rows() {
        let d = depths_col.value(i);
        let g = group_at(i)?;
        let entry = by_depth.entry(d).or_insert_with(|| (Vec::new(), Vec::new(), Vec::new(), Vec::new()));
        entry.0.push(g);
        entry.1.push(lowers_col.value(i));
        entry.2.push(uppers_col.value(i));
        entry.3.push(level_col.value(i).to_string());
    }

    let key_outliers = crate::transform::registry::secondary_key(spec.name.as_deref(), OUTLIERS_KEY);

    let mut out: Vec<(String, RecordBatch)> = Vec::new();
    out.push((key_outliers, outliers_batch));
    for k in 1..=K_MAX_NAMED {
        let (groups_v, lowers_v, uppers_v, levels_v) = by_depth
            .get(&(k as i32))
            .map(|t| t.clone())
            .unwrap_or_else(|| (Vec::new(), Vec::new(), Vec::new(), Vec::new()));
        let group_keys_v: Vec<Vec<KeyValue>> = groups_v.iter().map(|k| vec![k.clone()]).collect();
        let group_col_v = materialize_groupby_col(&group_keys_v, 0, &group_dtype)
            .map_err(PyValueError::new_err)?;
        let cols: Vec<ArrayRef> = vec![
            group_col_v,
            Arc::new(Float64Array::from(lowers_v)) as ArrayRef,
            Arc::new(Float64Array::from(uppers_v)) as ArrayRef,
            Arc::new(StringArray::from(levels_v.iter().map(|s| s.as_str()).collect::<Vec<_>>())) as ArrayRef,
        ];
        let batch = RecordBatch::try_new(depth_schema.clone(), cols)
            .map_err(|e| PyValueError::new_err(format!("stat_letter_value depth_{k}: {e}")))?;
        out.push((crate::transform::registry::secondary_key(spec.name.as_deref(), &depth_key(k)), batch));
    }
    Ok(out)
}

// ---------- PyO3 wrapper ----------

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::transform::core::TransformSpec;

/// Parse the optional `sort={"op": ..., "order": ...}` kwarg into a
/// [`LetterValueSort`]. Returns `Ok(None)` when no dict is supplied.
fn parse_sort(sort: Option<&Bound<'_, PyDict>>) -> PyResult<Option<LetterValueSort>> {
    let Some(d) = sort else { return Ok(None) };
    let op_str: String = match d.get_item("op")? {
        Some(v) => v.extract()?,
        None => "sum".to_string(),
    };
    let order_str: String = match d.get_item("order")? {
        Some(v) => v.extract()?,
        None => "ascending".to_string(),
    };
    let op = match op_str.as_str() {
        "sum" => SortOp::Sum,
        "mean" => SortOp::Mean,
        "max" => SortOp::Max,
        "min" => SortOp::Min,
        "count" => SortOp::Count,
        "median" => SortOp::Median,
        other => {
            return Err(PyValueError::new_err(format!(
                "LetterValue: unknown sort op '{other}'; expected \
                 'sum'|'mean'|'max'|'min'|'count'|'median'"
            )))
        }
    };
    let order = match order_str.as_str() {
        "ascending" | "asc" => SortOrder::Asc,
        "descending" | "desc" => SortOrder::Desc,
        other => {
            return Err(PyValueError::new_err(format!(
                "LetterValue: unknown sort order '{other}'; expected \
                 'ascending'|'descending'"
            )))
        }
    };
    Ok(Some(LetterValueSort { op, order }))
}

/// Letter-value (boxen) summary statistics for a numeric column.
///
/// Computes a hierarchical sequence of medians and quantiles extending
/// outward from the center until the estimated uncertainty of the extreme
/// values is too large. The depth of the hierarchy is controlled by
/// ``k_depth``. Used to draw boxen plots that reveal the full distributional
/// shape for large datasets.
///
/// Output columns: ``letter``, ``value_lo``, ``value_hi`` (Float64 edges
/// of each letter-value interval) plus ``is_outlier`` (Bool) for the
/// ``Outliers``-equivalent rows, optionally preceded by the ``group``
/// column.
///
/// Parameters
/// ----------
/// value : str
///     Numeric column to summarize (any integer, unsigned, or float dtype;
///     widened to Float64). A non-numeric column raises, naming the
///     transform, column, and dtype.
/// group : str, optional
///     Single group-key column; letter-values computed independently per
///     group.
/// k_depth : {"tukey", "proportion", "trustworthy", "full"}, \
///           default "proportion"
///     Algorithm for choosing the number of letter-value levels:
///     ``"tukey"`` — fixed integer depth; ``"proportion"`` — fraction of
///     sample size (see ``k_proportion``); ``"trustworthy"`` — based on
///     statistical trustworthiness; ``"full"`` — all levels to median.
/// k_proportion : float, default 0.007
///     Fraction of sample size used when ``k_depth="proportion"``. Must
///     be in (0, 1).
/// outlier_threshold : float, default 1.5
///     Tukey fence multiplier for labelling outliers. Must be ≥ 0.
/// name : str, optional
///     Named output label for sibling ``Reorder(from_=...)`` lookup.
/// sort : dict, optional
///     Group ordering for the categorical axis, as ``{"op": ..., "order":
///     ...}``. ``op`` is one of ``"sum" | "mean" | "max" | "min" | "count" |
///     "median"`` and ``order`` is ``"ascending" | "descending"`` (``"asc"``
///     / ``"desc"`` also accepted). The aggregate is computed over the value
///     column per group; ``count`` counts non-null value rows. Groups are then
///     emitted in ranked order (ties broken lexicographically), which sets the
///     boxen categorical axis domain directly. When omitted, groups are emitted
///     in lexicographic order.
///
/// Examples
/// --------
/// >>> import ferrum as fm
/// >>> fm.Chart(df).mark_boxen().encode(x="cut", y="price")
#[pyclass(eq, module = "ferrum._core", name = "LetterValue")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyLetterValue(pub(crate) TransformSpec);

#[pymethods]
impl PyLetterValue {
    #[new]
    #[pyo3(signature = (
        value, *,
        group = None,
        k_depth = "proportion",
        k_proportion = 0.007,
        outlier_threshold = 1.5,
        name = None,
        sort = None,
    ))]
    fn new(
        value: &str,
        group: Option<String>,
        k_depth: &str,
        k_proportion: f64,
        outlier_threshold: f64,
        name: Option<String>,
        sort: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        if value.is_empty() {
            return Err(PyValueError::new_err("LetterValue: value must be non-empty"));
        }
        if !outlier_threshold.is_finite() || outlier_threshold < 0.0 {
            return Err(PyValueError::new_err(
                "LetterValue: outlier_threshold must be finite and >= 0",
            ));
        }
        let kd = match k_depth {
            "tukey" => KDepth::Tukey,
            "proportion" => {
                if !(k_proportion.is_finite() && k_proportion > 0.0 && k_proportion < 1.0) {
                    return Err(PyValueError::new_err(
                        "LetterValue: k_proportion must be in (0, 1)",
                    ));
                }
                KDepth::Proportion { p: k_proportion }
            }
            "trustworthy" => KDepth::Trustworthy,
            "full" => KDepth::Full,
            other => return Err(PyValueError::new_err(format!(
                "LetterValue: unknown k_depth '{other}'; expected 'tukey'|'proportion'|'trustworthy'|'full'"
            ))),
        };
        let sort = parse_sort(sort)?;
        Ok(PyLetterValue(TransformSpec::LetterValue(LetterValueSpec {
            value: value.to_string(),
            group,
            k_depth: kd,
            outlier_threshold,
            name,
            sort,
        })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::LetterValue(s) => format!(
                "LetterValue(value='{}', group={:?}, k_depth={:?}, outlier_threshold={}, sort={:?}, name={:?})",
                s.value, s.group, s.k_depth, s.outlier_threshold, s.sort, s.name,
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, StringArray};

    fn batch1col(name: &str, vs: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::Float64, true)]));
        RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(vs))]).unwrap()
    }

    fn batch_grouped(groups: Vec<&str>, values: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, true),
            Field::new("v", DataType::Float64, true),
        ]));
        let g_arr: Vec<Option<&str>> = groups.iter().map(|s| Some(*s)).collect();
        RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(g_arr)),
            Arc::new(Float64Array::from(values)),
        ]).unwrap()
    }

    // ── #162: degrade-warning channel ────────────────────────────────────

    /// RED (pre-#162): fails on the pre-change code, which had no channel to
    /// report `KDepth::Full`'s computed depth silently capping at
    /// `K_MAX_NAMED` (12). `log2(10000)` floors to 13, one past the cap.
    #[test]
    fn letter_value_depth_cap_pushes_a_parameter_clamped_warning() {
        let values: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
        let batch = batch1col("x", values);
        let spec = LetterValueSpec {
            value: "x".into(),
            group: None,
            k_depth: KDepth::Full,
            outlier_threshold: 1.5,
            name: None,
            sort: None,
        };
        let ctx = TransformContext::default();
        let _ = apply_with_context(&spec, &batch, &ctx).unwrap();
        let warnings = ctx.warnings.borrow();
        assert_eq!(warnings.len(), 1, "got {warnings:?}");
        assert!(matches!(
            &warnings[0],
            TransformWarning::ParameterClamped { transform, group, param, requested, applied }
                if transform == "LetterValue" && group.is_none() && param == "k_depth"
                   && requested == "13" && applied == "12"
        ));
    }

    /// A grouped call names which group's own depth was capped.
    #[test]
    fn letter_value_grouped_depth_cap_names_the_group() {
        let groups: Vec<&str> = std::iter::repeat("big").take(10_000).collect();
        let values: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
        let batch = batch_grouped(groups, values);
        let spec = LetterValueSpec {
            value: "v".into(),
            group: Some("g".into()),
            k_depth: KDepth::Full,
            outlier_threshold: 1.5,
            name: None,
            sort: None,
        };
        let ctx = TransformContext::default();
        let _ = apply_with_context(&spec, &batch, &ctx).unwrap();
        let warnings = ctx.warnings.borrow();
        assert_eq!(warnings.len(), 1);
        // Rust-quality-reviewer finding 2 (post-#162 close): the bare group
        // VALUE ("big") broke `TransformWarning::group`'s documented
        // "col=value" contract (`context.rs`'s doc, matched by every sibling
        // adopter); must be "g=big", naming the column too.
        assert!(matches!(
            &warnings[0],
            TransformWarning::ParameterClamped { group, .. } if group.as_deref() == Some("g=big")
        ), "expected group label \"g=big\" (col=value), got {:?}", warnings[0]);
    }

    /// A well-supported depth (Tukey on a small group) must not warn.
    #[test]
    fn letter_value_tukey_default_does_not_warn() {
        let batch = batch1col("x", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let spec = LetterValueSpec {
            value: "x".into(),
            group: None,
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: None,
            sort: None,
        };
        let ctx = TransformContext::default();
        let _ = apply_with_context(&spec, &batch, &ctx).unwrap();
        assert!(ctx.warnings.borrow().is_empty());
    }

    #[test]
    fn test_letter_value_round_trip_json() {
        let spec = LetterValueSpec {
            value: "x".into(),
            group: Some("g".into()),
            k_depth: KDepth::Proportion { p: 0.007 },
            outlier_threshold: 1.5,
            name: Some("lv".into()),
            sort: None,
        };
        let wrapped = TransformSpec::LetterValue(spec.clone());
        let json = serde_json::to_string(&wrapped).unwrap();
        let parsed: TransformSpec = serde_json::from_str(&json).unwrap();
        match parsed {
            TransformSpec::LetterValue(s) => assert_eq!(s, spec),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_letter_value_sort_round_trip_and_deny_unknown() {
        // A valid sort dict round-trips losslessly (byte-identity for the
        // ferrum-produced wire: LetterValueSort serializes exactly {op, order}).
        let spec = LetterValueSpec {
            value: "v".into(),
            group: Some("g".into()),
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: None,
            sort: Some(LetterValueSort { op: SortOp::Sum, order: SortOrder::Desc }),
        };
        let wrapped = TransformSpec::LetterValue(spec.clone());
        let json = serde_json::to_string(&wrapped).unwrap();
        let parsed: TransformSpec = serde_json::from_str(&json).unwrap();
        match parsed {
            TransformSpec::LetterValue(s) => assert_eq!(s, spec),
            _ => panic!("wrong variant"),
        }

        // An EXTRA key inside the sort dict must now error (no-silent-drop:
        // matches the SEAM-04 deny_unknown_fields contract on the parent spec).
        // `TransformSpec` is internally tagged (`tag = "type"`), so the spec
        // fields — including `sort` — sit at the object root.
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        value["sort"]["bogus"] = serde_json::json!("oops");
        let polluted = serde_json::to_string(&value).unwrap();
        let err = serde_json::from_str::<TransformSpec>(&polluted);
        assert!(
            err.is_err(),
            "deny_unknown_fields must reject an unknown key in the sort dict; got: {err:?}",
        );
    }

    #[test]
    fn test_letter_value_basic_depth1_is_median() {
        let vs: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let batch = batch1col("x", vs);
        let spec = LetterValueSpec {
            value: "x".into(), group: None,
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: None,
            sort: None,
        };
        let out = apply(&spec, &batch).unwrap();
        // First row is depth=1 (median). lower == upper == median of 1..=100 = 50.5
        let depth = out.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        let lower = out.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
        let upper = out.column(3).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(depth.value(0), 1);
        assert!((lower.value(0) - 50.5).abs() < 1e-9);
        assert!((upper.value(0) - 50.5).abs() < 1e-9);
    }

    #[test]
    fn test_letter_value_tukey_k_for_n100() {
        // Tukey: K = floor(log2(100)) - 3 = 6 - 3 = 3.
        let k = k_for_depth(&KDepth::Tukey, 100);
        assert_eq!(k, 3);
    }

    #[test]
    fn test_letter_value_proportion_k_for_p007_n200() {
        // target = ceil(0.007 * 200) = 2; smallest K with 2^(K-1) >= 2 → K=2.
        let k = k_for_depth(&KDepth::Proportion { p: 0.007 }, 200);
        assert_eq!(k, 2);
    }

    #[test]
    fn test_letter_value_full_k_for_n20() {
        // Full: K = floor(log2(20)) = 4.
        let k = k_for_depth(&KDepth::Full, 20);
        assert_eq!(k, 4);
    }

    #[test]
    fn test_letter_value_grouped_3_groups() {
        let groups: Vec<&str> = (0..30).map(|i| ["a","b","c"][i % 3]).collect();
        let values: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let batch = batch_grouped(groups, values);
        let spec = LetterValueSpec {
            value: "v".into(), group: Some("g".into()),
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: None,
            sort: None,
        };
        let out = apply(&spec, &batch).unwrap();
        // n=10 per group → Tukey K = max(1, floor(log2(10)) - 3) = max(1, 3-3) = 1.
        // 3 groups × 1 depth = 3 rows.
        assert_eq!(out.num_rows(), 3);
        let g_arr = out.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let mut seen = std::collections::BTreeSet::new();
        for i in 0..out.num_rows() {
            seen.insert(g_arr.value(i).to_string());
        }
        assert_eq!(seen.len(), 3);
    }

    #[test]
    fn test_letter_value_outliers_classifies_extreme_values() {
        // Build n=50 normal-ish values plus 2 extremes; the extremes should be
        // flagged. Threshold 1.5 means: |value - lv_K| > 1.5 * IQR(lv_K).
        let mut vs: Vec<f64> = (1..=50).map(|i| i as f64).collect();
        vs.push(-1000.0);
        vs.push(1000.0);
        let batch = batch1col("x", vs);
        let spec = LetterValueSpec {
            value: "x".into(), group: None,
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: None,
            sort: None,
        };
        let primary = apply(&spec, &batch).unwrap();
        let secs = secondary_outputs(&spec, &batch, &primary).unwrap();
        let outliers = secs.iter().find(|(k, _)| k == "outliers").unwrap();
        let sch = outliers.1.schema();
        assert_eq!(sch.field(0).name(), "group");
        assert_eq!(sch.field(1).name(), "value");
        assert_eq!(sch.field(2).name(), "is_outlier");
        // The two extreme values must be flagged is_outlier=true.
        let val_col = outliers.1.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        let flag_col = outliers.1.column(2).as_any().downcast_ref::<BooleanArray>().unwrap();
        let mut found_extreme = 0;
        for i in 0..outliers.1.num_rows() {
            let v = val_col.value(i);
            if (v - 1000.0).abs() < 1e-9 || (v + 1000.0).abs() < 1e-9 {
                assert!(flag_col.value(i), "extreme value {v} should be flagged");
                found_extreme += 1;
            }
        }
        assert_eq!(found_extreme, 2);
    }

    #[test]
    fn test_letter_value_outliers_batch_contains_only_true_flags() {
        // Regression for F-L03-02: the outliers batch must be a FILTERED list
        // of outlier rows, not every input row with a flag column. n=50
        // in-range values + 2 extremes ⇒ exactly 2 rows survive, and every
        // surviving row's flag is true (there is no false-flag row left to
        // find).
        let mut vs: Vec<f64> = (1..=50).map(|i| i as f64).collect();
        vs.push(-1000.0);
        vs.push(1000.0);
        let batch = batch1col("x", vs);
        let spec = LetterValueSpec {
            value: "x".into(), group: None,
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: None,
            sort: None,
        };
        let primary = apply(&spec, &batch).unwrap();
        let secs = secondary_outputs(&spec, &batch, &primary).unwrap();
        let outliers = secs.iter().find(|(k, _)| k == "outliers").unwrap();
        assert_eq!(
            outliers.1.num_rows(),
            2,
            "outliers batch must contain only the 2 rows that are actually outliers, not all 52 input rows"
        );
        let flag_col = outliers.1.column(2).as_any().downcast_ref::<BooleanArray>().unwrap();
        for i in 0..outliers.1.num_rows() {
            assert!(flag_col.value(i), "every surviving row must be flagged true");
        }
    }

    #[test]
    fn test_letter_value_named_outputs_count_twelve() {
        // Structural check: one secondary output per declared key
        // (outliers + depth_1..depth_K_MAX_NAMED). The behavioral pin that
        // K_MAX_NAMED is actually 12 — i.e. that the raised cap lets
        // `k_for_depth` produce depths beyond the old 8 — lives in
        // `test_letter_value_k_for_depth_tukey_uses_raised_cap_at_n8192` and
        // `test_letter_value_tukey_n8192_reaches_depth_9_with_continuation_letter`
        // below; this test only pins that every declared key is published.
        let vs: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let batch = batch1col("x", vs);
        let spec = LetterValueSpec {
            value: "x".into(), group: None,
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: None,
            sort: None,
        };
        let primary = apply(&spec, &batch).unwrap();
        let secs = secondary_outputs(&spec, &batch, &primary).unwrap();
        assert_eq!(secs.len(), K_MAX_NAMED + 1, "expected outliers + depth_1..depth_{K_MAX_NAMED}");
        for k in 1..=K_MAX_NAMED {
            let key = format!("depth_{k}");
            assert!(secs.iter().any(|(name, _)| name == &key), "missing {key}");
        }
    }

    #[test]
    fn test_letter_value_k_for_depth_tukey_uses_raised_cap_at_n8192() {
        // Regression for F-L03-11: at n=8192, Tukey's raw depth is
        // floor(log2(8192)) - 3 = 13 - 3 = 10, which is inert under the old
        // K_MAX_NAMED=8 clamp (k_for_depth would return 8). The raised cap
        // (12) lets the real depth through unclamped.
        assert_eq!(k_for_depth(&KDepth::Tukey, 8192), 10);
    }

    #[test]
    fn test_letter_value_tukey_n8192_reaches_depth_9_with_continuation_letter() {
        // Companion to the k_for_depth pin above: proves the raised cap
        // carries actual ROWS through `secondary_outputs`, not just keys.
        // At n=8192 (k=10), depth_9 must be a non-empty band (it was
        // unreachable — the key didn't even exist — under the old K_MAX_NAMED=8
        // cap) while depth_12 (> k=10) stays a published-but-empty band, and
        // depth_9's `level` must use the real Tukey continuation letter "Y"
        // rather than the generic `depth_9` fallback spelling.
        let vs: Vec<f64> = (0..8192).map(|i| i as f64).collect();
        let batch = batch1col("x", vs);
        let spec = LetterValueSpec {
            value: "x".into(), group: None,
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: None,
            sort: None,
        };
        let primary = apply(&spec, &batch).unwrap();
        let secs = secondary_outputs(&spec, &batch, &primary).unwrap();

        let depth_9 = secs.iter().find(|(k, _)| k == "depth_9")
            .unwrap_or_else(|| panic!("depth_9 output missing — K_MAX_NAMED not raised"));
        assert_eq!(depth_9.1.num_rows(), 1, "depth_9 must carry the row a k=10 group actually produces");
        let level_col = depth_9.1.column(3).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(level_col.value(0), "Y", "depth_9's level must be the Tukey continuation letter, not a 'depth_9' fallback spelling");

        let depth_12 = secs.iter().find(|(k, _)| k == "depth_12")
            .unwrap_or_else(|| panic!("depth_12 output missing — K_MAX_NAMED not raised to 12"));
        assert_eq!(depth_12.1.num_rows(), 0, "depth_12 exceeds this group's actual depth (10) and must stay a published-but-empty band");
    }

    #[test]
    fn test_letter_value_outliers_zero_when_no_outliers() {
        // Edge case from the spec's acceptance criteria ("boxen with zero
        // outliers"): every value sits well within the outermost letter-value
        // fence, so the filtered outliers batch must be zero rows, not a
        // batch of all-false-flag rows.
        let vs: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let batch = batch1col("x", vs);
        let spec = LetterValueSpec {
            value: "x".into(), group: None,
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: None,
            sort: None,
        };
        let primary = apply(&spec, &batch).unwrap();
        let secs = secondary_outputs(&spec, &batch, &primary).unwrap();
        let outliers = secs.iter().find(|(k, _)| k == "outliers").unwrap();
        assert_eq!(outliers.1.num_rows(), 0, "no input value falls outside the fence; the batch must be empty");
    }

    #[test]
    fn test_letter_value_named_outputs_use_name_prefix() {
        let vs: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let batch = batch1col("x", vs);
        let spec = LetterValueSpec {
            value: "x".into(), group: None,
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: Some("lv".into()),
            sort: None,
        };
        let primary = apply(&spec, &batch).unwrap();
        let secs = secondary_outputs(&spec, &batch, &primary).unwrap();
        assert!(secs.iter().any(|(name, _)| name == "lv_outliers"));
        for k in 1..=K_MAX_NAMED {
            let key = format!("lv_depth_{k}");
            assert!(secs.iter().any(|(name, _)| name == &key), "missing {key}");
        }
    }

    /// First-appearance order of the `group` column in the primary batch.
    /// This is exactly what the categorical-axis domain is built from, so it is
    /// the order that controls the boxen axis.
    fn group_first_appearance(batch: &RecordBatch) -> Vec<String> {
        let g = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for i in 0..batch.num_rows() {
            let s = g.value(i).to_string();
            if seen.insert(s.clone()) {
                out.push(s);
            }
        }
        out
    }

    #[test]
    fn test_letter_value_sort_none_is_lexicographic() {
        // Groups appear in source as c, a, b but unsorted output is lexicographic.
        let groups: Vec<&str> = (0..30).map(|i| ["c", "a", "b"][i % 3]).collect();
        let values: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let batch = batch_grouped(groups, values);
        let spec = LetterValueSpec {
            value: "v".into(), group: Some("g".into()),
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: None,
            sort: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(group_first_appearance(&out), vec!["a", "b", "c"]);
    }

    #[test]
    fn test_letter_value_sort_sum_desc() {
        // a: [0,1,2] sum=3 ; b: [10,11,12] sum=33 ; c: [5,6,7] sum=18.
        // Descending by sum → b, c, a.
        let groups: Vec<&str> = vec![
            "a", "a", "a", "b", "b", "b", "c", "c", "c",
        ];
        let values: Vec<f64> = vec![0.0, 1.0, 2.0, 10.0, 11.0, 12.0, 5.0, 6.0, 7.0];
        let batch = batch_grouped(groups, values);
        let spec = LetterValueSpec {
            value: "v".into(), group: Some("g".into()),
            k_depth: KDepth::Full,
            outlier_threshold: 1.5,
            name: None,
            sort: Some(LetterValueSort { op: SortOp::Sum, order: SortOrder::Desc }),
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(group_first_appearance(&out), vec!["b", "c", "a"]);
    }

    #[test]
    fn test_letter_value_sort_tie_breaks_lexicographically() {
        // a and b both sum to 3; c sums to 30. Ascending → a, b (tie, lex), c.
        let groups: Vec<&str> = vec![
            "b", "b", "a", "a", "c", "c",
        ];
        let values: Vec<f64> = vec![1.0, 2.0, 1.0, 2.0, 15.0, 15.0];
        let batch = batch_grouped(groups, values);
        let spec = LetterValueSpec {
            value: "v".into(), group: Some("g".into()),
            k_depth: KDepth::Full,
            outlier_threshold: 1.5,
            name: None,
            sort: Some(LetterValueSort { op: SortOp::Sum, order: SortOrder::Asc }),
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(group_first_appearance(&out), vec!["a", "b", "c"]);
    }

    #[test]
    fn test_letter_value_sort_count_uses_extracted_rows() {
        // a has 5 non-null rows, b has 2, c has 3 (one b value is NaN → dropped).
        // count descending → a, c, b.
        let groups: Vec<&str> = vec![
            "a", "a", "a", "a", "a", "b", "b", "b", "c", "c", "c",
        ];
        let values: Vec<f64> = vec![
            1.0, 1.0, 1.0, 1.0, 1.0, // a: 5
            2.0, 2.0, f64::NAN,      // b: NaN dropped → 2
            3.0, 3.0, 3.0,           // c: 3
        ];
        let batch = batch_grouped(groups, values);
        let spec = LetterValueSpec {
            value: "v".into(), group: Some("g".into()),
            k_depth: KDepth::Full,
            outlier_threshold: 1.5,
            name: None,
            sort: Some(LetterValueSort { op: SortOp::Count, order: SortOrder::Desc }),
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(group_first_appearance(&out), vec!["a", "c", "b"]);
    }

    // ── #185: letter_value accepts every dtype its siblings accept ────────

    fn batch_value_int64_group(values: Vec<f64>, groups: Vec<i64>) -> RecordBatch {
        use arrow::array::Int64Array;
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Float64, true),
            Field::new("cyl", DataType::Int64, true),
        ]));
        RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(values)),
            Arc::new(Int64Array::from(groups)),
        ]).unwrap()
    }

    /// RED (pre-#185): a group column of any dtype other than Utf8 refused
    /// with `"stat_letter_value: group column 'cyl' must be Utf8; got Int64"`
    /// — the inverted contract every sibling groupby-taking transform (box_stats,
    /// outliers, swarm, kde_2d, smooth, ...) does NOT share. GREEN: the call
    /// renders, and the output `group` column carries the ORIGINAL Int64 dtype
    /// (not a `.to_string()` shortcut, which would silently re-widen it to Utf8
    /// one layer down — the exact drift #185 exists to close).
    #[test]
    fn letter_value_int64_group_succeeds_and_preserves_dtype() {
        use arrow::array::Int64Array;
        // Group 6: [10,20,30,40,50]; group 8: [1,2,3,4,5].
        let batch = batch_value_int64_group(
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            vec![6, 6, 6, 6, 6, 8, 8, 8, 8, 8],
        );
        let spec = LetterValueSpec {
            value: "v".into(), group: Some("cyl".into()),
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: None,
            sort: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(
            out.schema().field(0).data_type(),
            &DataType::Int64,
            "the 'group' output column must keep the input's Int64 dtype, not widen to Utf8"
        );
        let groups = out.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let mut seen = std::collections::BTreeSet::new();
        for i in 0..out.num_rows() {
            seen.insert(groups.value(i));
        }
        assert_eq!(seen, std::collections::BTreeSet::from([6, 8]));
    }

    /// A group column of an unsupported dtype (e.g. Date32, which no groupby
    /// consumer in the crate accepts) still refuses, naming the column and
    /// dtype. Pins the WORDING word-for-word, not just a substring match:
    /// `extract_values`/`compute_depths` no longer hand-roll their own dtype
    /// gate — they delegate to `group_key::validate_groupby_columns` (via
    /// `extract_values`) and `group_key::group_partition_multi` (via
    /// `compute_depths`, which calls `validate_groupby_columns` internally
    /// too), so the refusal is that SAME shared function's exact text
    /// (`"groupby column '<col>' has unsupported dtype <dt>; supported:
    /// ..."`), not a hand-duplicated near-copy that happened to say "group
    /// column" instead of "groupby column". `apply` reaches this through
    /// `compute_depths`, so this also proves the migration to
    /// `group_partition_multi` (#185 close-review finding 2) still refuses
    /// by name.
    #[test]
    fn letter_value_unsupported_group_dtype_refuses_by_name() {
        pyo3::Python::initialize();
        use arrow::array::Date32Array;
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Float64, true),
            Field::new("d", DataType::Date32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
                Arc::new(Date32Array::from(vec![Some(1), Some(2)])),
            ],
        ).unwrap();
        let spec = LetterValueSpec {
            value: "v".into(), group: Some("d".into()),
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: None,
            sort: None,
        };
        let err = apply(&spec, &batch).expect_err("a Date32 group column must refuse");
        assert_eq!(
            err.to_string(),
            "ValueError: stat_letter_value: groupby column 'd' has unsupported dtype Date32; \
             supported: Utf8/LargeUtf8, Float64/Float32, Int8/Int16/Int32/Int64, \
             UInt8/UInt16/UInt32/UInt64, Boolean"
        );
    }

    /// Companion to the primary-batch refusal above: `extract_values` (the
    /// path `secondary_outputs` uses for its outlier classification) refuses
    /// with the SAME `validate_groupby_columns` wording, byte-for-byte —
    /// proof the two call sites did not drift onto their own copies of the
    /// message after the shared-toolkit migration.
    #[test]
    fn letter_value_extract_values_unsupported_dtype_matches_compute_depths_wording() {
        pyo3::Python::initialize();
        use arrow::array::Date32Array;
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Float64, true),
            Field::new("d", DataType::Date32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
                Arc::new(Date32Array::from(vec![Some(1), Some(2)])),
            ],
        ).unwrap();
        let spec = LetterValueSpec {
            value: "v".into(), group: Some("d".into()),
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: None,
            sort: None,
        };
        let err = extract_values(&spec, &batch).expect_err("a Date32 group column must refuse");
        assert_eq!(
            err.to_string(),
            "ValueError: stat_letter_value: groupby column 'd' has unsupported dtype Date32; \
             supported: Utf8/LargeUtf8, Float64/Float32, Int8/Int16/Int32/Int64, \
             UInt8/UInt16/UInt32/UInt64, Boolean"
        );
    }

    /// The Int64 group dtype survives into the SECONDARY outputs too (outliers
    /// + depth_K bands), not just the primary batch — `secondary_outputs` reads
    /// the group column back off the primary batch via the shared
    /// `groupby_key_at` extractor rather than assuming it is a `StringArray`.
    #[test]
    fn letter_value_secondary_outputs_preserve_int64_group_dtype() {
        use arrow::array::Int64Array;
        let mut vs: Vec<f64> = (1..=50).map(|i| i as f64).collect();
        vs.push(-1000.0);
        vs.push(1000.0);
        let n = vs.len();
        let groups = vec![7i64; n];
        let batch = batch_value_int64_group(vs, groups);
        let spec = LetterValueSpec {
            value: "v".into(), group: Some("cyl".into()),
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: None,
            sort: None,
        };
        let primary = apply(&spec, &batch).unwrap();
        let secs = secondary_outputs(&spec, &batch, &primary).unwrap();

        let outliers = secs.iter().find(|(k, _)| k == "outliers").unwrap();
        assert_eq!(outliers.1.schema().field(0).data_type(), &DataType::Int64);
        assert_eq!(outliers.1.num_rows(), 2, "the two extreme values are still classified as outliers");
        let o_groups = outliers.1.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        for i in 0..outliers.1.num_rows() {
            assert_eq!(o_groups.value(i), 7);
        }

        let depth_1 = secs.iter().find(|(k, _)| k == "depth_1").unwrap();
        assert_eq!(depth_1.1.schema().field(0).data_type(), &DataType::Int64);
        assert_eq!(depth_1.1.num_rows(), 1);
        let d_groups = depth_1.1.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(d_groups.value(0), 7);
    }

    /// Evidence for the `NullKeyPolicy::Pool` design choice in
    /// `compute_depths`'s doc comment: a null-valued group row still forms
    /// its OWN group (materialized as a null `group` entry), rather than
    /// being dropped the way `NullKeyPolicy::Skip` would drop it. 5 rows key
    /// to `"a"`, 5 key to a null group column value; both must survive as
    /// distinct groups.
    #[test]
    fn letter_value_null_group_value_pools_into_one_null_keyed_group() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, true),
            Field::new("v", DataType::Float64, true),
        ]));
        let groups: Vec<Option<&str>> = vec![
            Some("a"), Some("a"), Some("a"), Some("a"), Some("a"),
            None, None, None, None, None,
        ];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0];
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(groups)),
                Arc::new(Float64Array::from(values)),
            ],
        ).unwrap();
        let spec = LetterValueSpec {
            value: "v".into(), group: Some("g".into()),
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: None,
            sort: None,
        };
        let out = apply(&spec, &batch).unwrap();
        let g_arr = out.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let mut seen: std::collections::HashSet<Option<String>> = std::collections::HashSet::new();
        for i in 0..out.num_rows() {
            seen.insert(if g_arr.is_null(i) { None } else { Some(g_arr.value(i).to_string()) });
        }
        assert_eq!(
            seen,
            std::collections::HashSet::from([Some("a".to_string()), None]),
            "the null-valued-group rows must survive as their own group (a null 'group' \
             entry), not be silently dropped (NullKeyPolicy::Skip's behavior) or scattered"
        );
    }

    /// FA-7-style byte-stability guard for #185: a Utf8 group column produces
    /// the identical output it did before the KeyValue migration — the same
    /// group set, the same per-group median, and the field stays Utf8.
    #[test]
    fn letter_value_utf8_group_byte_stable_after_keyvalue_migration() {
        let groups: Vec<&str> = (0..30).map(|i| ["a", "b", "c"][i % 3]).collect();
        let values: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let batch = batch_grouped(groups, values);
        let spec = LetterValueSpec {
            value: "v".into(), group: Some("g".into()),
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: None,
            sort: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.schema().field(0).data_type(), &DataType::Utf8);
        assert_eq!(out.num_rows(), 3);
        assert_eq!(group_first_appearance(&out), vec!["a", "b", "c"]);
    }

    /// Regression for stage1-rust-quality-reviewer Bug Risk 3 (cycle 3):
    /// `compute_depths` used to insert EVERY partition group into `groups`,
    /// including one whose value rows are all null/NaN (`vs` empty) — a
    /// group the pre-migration code could never see, since it built the map
    /// from already-filtered `(key, value)` pairs. `group_order`'s
    /// sort-by-aggregate comparator has no filter for empty groups: an empty
    /// group's `median`/`mean` aggregate is `NaN`, `NaN.partial_cmp(_)` is
    /// `None`, and the comparator degrades to `Ordering::Equal` against
    /// every other group.
    ///
    /// MEASURED, not predicted: with three real groups (a/c/e, medians
    /// 10/50/90) and three all-NaN groups (b/d/f) interleaved
    /// lexicographically between them, `sort=Some(median, Desc)` on the
    /// pre-fix (cycle-2) code deterministically produced `["a", "c", "e"]`
    /// — ascending order, the exact OPPOSITE of the requested descending
    /// order — reproduced identically across three separate runs, with no
    /// panic on this toolchain. That is the real symptom this test pins:
    /// not a panic, a silently INVERTED sort.
    #[test]
    fn letter_value_all_nan_group_does_not_corrupt_sort_order() {
        let groups: Vec<&str> = vec![
            "a", "a", "a",
            "b", "b", "b",
            "c", "c", "c",
            "d", "d", "d",
            "e", "e", "e",
            "f", "f", "f",
        ];
        let values: Vec<f64> = vec![
            9.0, 10.0, 11.0,
            f64::NAN, f64::NAN, f64::NAN,
            49.0, 50.0, 51.0,
            f64::NAN, f64::NAN, f64::NAN,
            89.0, 90.0, 91.0,
            f64::NAN, f64::NAN, f64::NAN,
        ];
        let batch = batch_grouped(groups, values);
        let spec = LetterValueSpec {
            value: "v".into(), group: Some("g".into()),
            k_depth: KDepth::Tukey,
            outlier_threshold: 1.5,
            name: None,
            sort: Some(LetterValueSort { op: SortOp::Median, order: SortOrder::Desc }),
        };
        let out = apply(&spec, &batch).unwrap();
        // The all-NaN groups (b/d/f) never had any value rows and must not
        // be emitted at all -- only the three real groups.
        assert_eq!(out.num_rows(), 3, "only a/c/e's single Tukey depth row each should be emitted");
        assert_eq!(
            group_first_appearance(&out),
            vec!["e", "c", "a"],
            "descending-by-median order must be e(90), c(50), a(10) -- \
             an all-NaN group's NaN aggregate must not collapse the comparator \
             to Equal and invert the real groups' relative order"
        );
    }
}
