//! Transform metadata registry (batch-D spec §4.6, decision doc M6a).
//!
//! One Rust-owned table of per-transform-kind metadata — method/kernel/
//! distribution vocabularies, primary output schemas, and wire parameter
//! names — co-located with [`super::core::for_each_transform`] so that a new
//! transform variant is a registry decision, not an afterthought (the
//! `scale_accepted_keys` move, applied to the transform layer). Exposed to
//! Python as `ferrum._core.transform_registry(kind)`. Batch-D T12 deleted
//! Python's hand-mirrored `_VALID_KERNELS`, qq distribution lists, and
//! hardcoded fit-output-column literals in favor of lookups against this
//! table; `lmplot`'s method set stays a pinned routed-subset literal until
//! T13 routes the remaining methods.
//!
//! **`kind` is the transform's *module name*** — the second column of
//! `for_each_transform!`'s table (`"bin"`, `"bin_2d"`, `"regression_data"`,
//! …), not necessarily the wire `"type"` tag serde derives for
//! `TransformSpec`'s internally-tagged JSON form. **Corrected 2026-09-04:**
//! the two do NOT agree everywhere except
//! `TimeUnit` — the real divergence set, proven by
//! `wire_tag_matches_registry_kind_or_named_exception` (`tests.rs`) against
//! `TransformSpec`'s actual `Serialize` output rather than prose, is
//! `{timeunit → time_unit, kde_2d → kde2_d}`. `Bin2D` looks like a third
//! divergence but isn't one in practice: its module name (`bin_2d`) and wire
//! tag (`bin_2d`) agree only because the variant carries an explicit
//! `#[serde(rename = "bin_2d")]` — without it, serde's snake_case derivation
//! would produce `bin2_d`, the same digit-adjacent-variant-name pattern that
//! makes `Kde2D` diverge for real (no override exists on that variant). Every
//! transform this task's Python consumers need to query (`kde`, `qq`,
//! `smooth`, `regression_data`, …) has module name == wire tag, so both
//! divergences are inert for this task's consumers — but `kde_2d` is a kind
//! whose desugar consumers do exist, so a future consumer keying this
//! registry off a serialized spec's `"type"` field must translate through
//! the named exception set in `tests.rs`, not assume identity.
//!
//! **`outputs` convention.** A kind's `outputs` list is its *closed* primary
//! output schema under default construction — literal column names/dtypes
//! that do not depend on arbitrary caller-supplied strings or on the input
//! batch's own (pass-through) columns. "Under default construction" is
//! load-bearing for kinds whose schema shape is itself parameter-dependent:
//! `Contour`'s declared 6-column isoline schema holds for `fill=false` (the
//! default) — `fill=true` emits a different 4-column isoband schema — and
//! the four `density_data`/`regression_data`/`loess_data` output names hold
//! only under each transform's default `as_` tuple; a caller-supplied `as_`
//! changes the real column names without changing this table. `outputs` is
//! `vec![]` for:
//! - pass-through-only transforms (`Outliers`, `Filter`, `Reorder`,
//!   `Identity`, `TopK`, `Sample`) — the output schema is whatever the input
//!   schema was;
//! - pass-through-*plus*-additive transforms whose additive columns cannot be
//!   declared as a stable name/dtype pair together with the transform's
//!   arbitrary (unknown) pass-through columns — `DataWindow` (each
//!   `WindowOp.as_` names its own column, from a `Vec<Op>` of caller-supplied
//!   length) and `JoinAggregate` (same shape: each `AggSpec.as_` names its
//!   own column). **`DataStack` corrected 2026-09-05:** it was previously
//!   grouped here under "no *stable* default name to declare", which is false — `DataStackSpec.as_` is a single top-level
//!   `(String, String)` with a real serde default, `("y0", "y1")`
//!   (`data_stack.rs`). The accurate reason it stays `[]` is different: its
//!   additive pair, unlike `Swarm`/`Violin` below, is a directly
//!   caller-settable parameter layered on top of the transform's ENTIRE
//!   arbitrary input schema (every input column passes through unchanged),
//!   and declaring it correctly would need the same non-empty per-call
//!   passthrough allowlist that `swarm_outputs_match_actual_schema`
//!   (`tests.rs`) proves for `Swarm` — a fixture this table doesn't have for
//!   `DataStack` today. This is "not yet proven", not "structurally
//!   undeclarable"; left open rather than reopening the accepted `[]`
//!   decision;
//! - variable-arity transforms (`Aggregate`, `DataAggregate`) whose column
//!   count/names come from a `Vec<Op>` of caller-supplied length;
//! - transforms whose sole output column name and/or dtype is driven by a
//!   REQUIRED (no-default) user parameter, or whose declared default is
//!   `None` meaning "derive from another required parameter" rather than a
//!   static literal (`ReferenceLine`, `Calculate`, `TimeUnit`, `DataBin`,
//!   `Flatten`, `Pivot`) — there is no default name/dtype to declare;
//! - `Unpivot` and `Fold`. **Corrected 2026-09-05:** these were previously
//!   grouped in the bullet above under "REQUIRED
//!   (no-default) user parameter", which is false for their column NAMES —
//!   `UnpivotSpec.var_name`/`value_name` default to `"variable"`/`"value"`
//!   and `FoldSpec.as_` defaults to `("key", "value")` (`unpivot.rs`,
//!   `fold.rs`). Two real reasons keep them at `[]` instead: (1) both also
//!   pass through a caller-selected, variable-length column set (`Unpivot`'s
//!   `id_vars`, `Fold`'s non-folded columns) — the same variable-arity shape
//!   as `Aggregate`/`DataAggregate` above; and (2) both transforms' *value*
//!   column dtype is `unified_dtype`, computed at `apply()` time from
//!   whichever columns the caller selects to melt/fold, so even under the
//!   default name the dtype is never statically knowable.
//!
//! `Swarm` and `Violin` are both pass-through-plus-additive (input schema is
//! unknown/arbitrary) but their *additive* columns are name-stable under
//! default construction (`swarm_x`/`swarm_y`/`__pos_x_offset__` for `Swarm`'s
//! default `orient="vertical"`; `group_id`/`violin_x`/`violin_y`/
//! `__pos_x_offset__` for `Violin`), so both declare those additive columns
//! rather than `[]` — the "no closed schema" carve-out above applies only
//! when even the additive names are caller-controlled, not whenever *some*
//! part of the schema (the pass-through part) is unknown.
//!
//! **`secondary_outputs` convention.** `outputs` covers a kind's *primary*
//! batch (spec §4.6's wording). Three kinds also publish additional NAMED
//! batches from `TransformSpec::secondary_outputs` (`core.rs`): `Qq` (the
//! reference line), `LetterValue` (the outlier rows plus one band batch per
//! depth), and `Linkage` (the linkage batch re-published under its own key,
//! plus dendrogram order/coords/segments). **Declared 2026-09-06:** those
//! were out of the registry by contract, which left the least-guessable
//! column names in the crate — `qq_line_x_start`, `lv_depth_{k}`, the
//! dendrogram columns — as unchecked Python literals, i.e. the exact drift
//! class the registry closed for primary columns. They now ride the same
//! mold: each entry is `(key, columns)` under the kind's DEFAULT
//! construction (`name = None`); a caller-supplied `name` prefixes EVERY key
//! (`"{name}_order"`, `"{name}_depth_3"`, `"{name}_qq_line"`, …) the same way
//! a caller-supplied `as_` renames primary columns, and the same "under
//! default construction" carve-out applies. This is a true universal across
//! all three kinds — `Qq`'s key was unprefixed under a `name` until
//! **2026-09-06**, when `qq::secondary_outputs` was made to prefix
//! `QQ_LINE_KEY` exactly like `letter_value::OUTLIERS_KEY`/
//! `linkage::LINKAGE_KEY` already did. `Qq`'s entry holds for its default
//! `emit_line=true`; at `emit_line=false` it emits no secondary batch at
//! all. Every declared entry is proven against the real
//! `secondary_outputs()` output by the per-kind
//! `*_secondary_outputs_match_actual_batches` fixtures (`tests.rs`), and
//! `KINDS_WITH_SECONDARY_FIXTURE` there pins that coverage against "every
//! kind with a non-empty declared `secondary_outputs`";
//! `a_spec_name_prefixes_every_declared_secondary_key` (`tests.rs`) proves
//! the prefixing rule itself across all three kinds directly.
//!
//! **`methods` convention.** Only the kind's single most prominent free-string
//! method/kernel/distribution/operation vocabulary — the parameter most
//! directly named `method`/`kernel`/`distribution`/`op`/`aggregate`/`offset`.
//! `Glm`/`Logistic`/`Robust` are deliberately `methods: []`: `Glm`'s
//! `family`/`link` are two orthogonal closed vocabularies, and cramming both
//! into one flat list would be ambiguous to any consumer without inventing a
//! new registry shape — out of scope for this task (no consumer needs it
//! yet; `parse_family`/`parse_link` in `glm.rs` remain the source of truth).
//! **`Linkage` declared 2026-09-05:** it carries a closed 7-value `method`
//! vocabulary (`LinkageMethod`, `linkage.rs`) behind a PyO3-constructor-time
//! hard validation site shaped identically to `kde`/`qq`/`smooth`/`hex`/
//! `raster` (`PyLinkage::new`, `linkage.rs`), so the rule above predicts a
//! non-empty list for it and there is no ambiguity to carve out — it was
//! simply missed initially, not a deliberate `[]`.
//!
//! **Every non-empty `methods` entry READS its enforcement site's own const,
//! and every one has a drift guard, completed 2026-09-06.** Before that,
//! only 4 of 14 did: the other ten were hand-copied literals, seven of them
//! with no acceptance/drift test at all,
//! and several enforcement sites spelled their own vocabulary more than once
//! (`raster` three times, `linkage` baked into an error string). Each of the
//! ten now has a `pub(crate) const` beside its enforcement site —
//! `error_extent::METHODS`, `linkage::METHODS`, `raster::AGGREGATES`,
//! `hex::AGGREGATES`, `pivot::OPS`, `data_window::OPS`,
//! `regression_data::METHODS`, `impute::METHODS`, `top_k::OPS`,
//! `data_stack::OFFSETS` — which the site's validation and its error message
//! both read, and which this table clones. `methods_drift_guard_covers_every_declared_vocabulary`
//! (`tests.rs`) pins the guard set against "every kind with a non-empty
//! `methods`", so a future kind cannot join this table without one.
//! `pivot` and `top_k` were the two kinds that genuinely could not be guarded
//! before, because each had a silent `_ => sum()` fallback rather than a
//! refusal site (which is also how the earlier undetected `pivot` vocabulary
//! drift went unnoticed, and a latent silent-wrong-answer path in its own
//! right); both now refuse an unrecognized op, so both are guardable and
//! guarded.
//!
//! **`params`** is the spec struct's real WIRE-accepted field set — the set
//! `params_match_spec_struct_serde_fields` (`tests.rs`) proves by mining
//! serde's own `deny_unknown_fields` rejection, not a reading of the Rust
//! field names. **Corrected 2026-09-05:** this is NOT "identical to the
//! Rust field name at every site in this crate" — the
//! `"bin"` entry below is the one site where they diverge: `BinSpec` carries
//! `#[serde(into = "BinWire", from = "BinWire")]`, so the wire it actually
//! (de)serializes through is `BinWire`'s flat `bin_count`/`bin_width` keys,
//! not the Rust-side typed `mode: BinMode` field the struct exposes to
//! in-crate callers (a manual audit once read the Rust field name and missed
//! this; the drift guard catches it). No other top-level spec struct in
//! this crate carries an `into`/`from` shim — the `#[serde(rename)]`s on
//! `aggregate.rs`/`data_window.rs`/`join_aggregate.rs` are all on NESTED
//! op-structs, not top-level params. The declared order matches each entry's
//! real field-declaration order at every site checked, but this is NOT a
//! checked contract: `params_match_spec_struct_serde_fields` mines the
//! accepted-field list out of a `deny_unknown_fields` error message into a
//! `BTreeSet` (`chart_config::accepted_fields_from_deny_unknown_error`), so
//! it compares field SETS, not sequences — an entry's declared order could
//! silently drift from its struct's real field order without failing any
//! test. Dropped the "in declaration order" claim rather than build an
//! ordered-sequence comparison, since the shared `BTreeSet`-returning helper
//! is also relied on by two pre-existing consumers
//! (`render::binding::theme_dict_tests`, `render::chart_config::tests`), and
//! widening its return shape would need to account for both.

use std::sync::Arc;

use arrow::datatypes::{DataType, Field};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// One named secondary output batch: the key it is published under (under
/// default construction — see the module doc's `secondary_outputs`
/// convention) and that batch's own column schema.
pub(crate) type SecondaryOutput = (String, Vec<(&'static str, DataType)>);

/// One transform kind's registry entry. See the module doc for the
/// `outputs`/`secondary_outputs`/`methods` inclusion conventions.
pub(crate) struct TransformRegistryEntry {
    pub methods: Vec<&'static str>,
    pub outputs: Vec<(&'static str, DataType)>,
    pub secondary_outputs: Vec<SecondaryOutput>,
    pub params: Vec<&'static str>,
}

/// The fit-family primary output schema shared by `Smooth` (`output="fitted"`,
/// the default), `Logistic`, `Glm`, and `Robust` (`output="fitted"`, the
/// default) — declared once here per the batch-D M6 schema-collapse (spec
/// §4.6: "the four identical `x, y, ci_lower, ci_upper` schema declarations
/// collapse onto one shared constant") and consumed both by this registry's
/// `outputs` entries and by [`fit_output_fields`], which each of the four
/// transforms' schema builders now calls instead of repeating the four
/// `Field::new` calls inline.
pub(crate) const FIT_OUTPUT_COLUMNS: &[(&str, DataType)] = &[
    ("x", DataType::Float64),
    ("y", DataType::Float64),
    ("ci_lower", DataType::Float64),
    ("ci_upper", DataType::Float64),
];

/// `[x, y, ci_lower, ci_upper]`, all nullable Float64 — the shared
/// `Field` list for `Smooth`/`Logistic`/`Glm`/`Robust`'s fitted-curve output
/// batches. See [`FIT_OUTPUT_COLUMNS`].
pub(crate) fn fit_output_fields() -> Vec<Field> {
    FIT_OUTPUT_COLUMNS
        .iter()
        .map(|(name, dt)| Field::new(*name, dt.clone(), true))
        .collect()
}

/// List-of-Float64 dtype used by `Kde2D`'s grid/density/extent columns.
/// `Arc::new` is not a `const fn`, so this can't live in a `const` table
/// alongside [`FIT_OUTPUT_COLUMNS`] — built fresh per call instead (this
/// whole module deals in owned `Vec`s, not `const` slices, for exactly this
/// reason: several kinds' dtypes are not const-constructible).
fn list_f64_dtype() -> DataType {
    DataType::List(Arc::new(Field::new("item", DataType::Float64, true)))
}

/// The full set of transform kinds (module names), derived from
/// [`super::core::for_each_transform`] so a new transform automatically
/// appears here without a second hand-maintained list — the
/// `every_transform_kind_has_a_registry_entry` test (`tests.rs`) is the net
/// that catches a kind landing in the dispatch table without a matching
/// [`entry_for_kind`] arm.
pub(crate) fn all_transform_kinds() -> Vec<&'static str> {
    macro_rules! kind_list_arm {
        ($($V:ident => $m:ident : $py:ident,)*) => {
            vec![ $( stringify!($m) ),* ]
        };
    }
    crate::transform::core::for_each_transform!(kind_list_arm)
}

/// `linkage`'s four secondary batches under default construction. Built here
/// rather than inline because `DataType` is not `const`-constructible into a
/// nested `Vec` (the same reason [`list_f64_dtype`] is a function).
fn linkage_secondary_outputs() -> Vec<SecondaryOutput> {
    use crate::transform::linkage;
    vec![
        (linkage::LINKAGE_KEY.to_string(), linkage::LINKAGE_COLUMNS.to_vec()),
        (linkage::ORDER_KEY.to_string(), linkage::ORDER_COLUMNS.to_vec()),
        (linkage::COORDS_KEY.to_string(), linkage::COORDS_COLUMNS.to_vec()),
        (linkage::SEGMENTS_KEY.to_string(), linkage::SEGMENTS_COLUMNS.to_vec()),
    ]
}

/// `letter_value`'s outlier batch plus one band batch per named depth, under
/// default construction. `depth_key` is `letter_value`'s own key formatter, so
/// the declared keys cannot drift from the emitted ones.
fn letter_value_secondary_outputs() -> Vec<SecondaryOutput> {
    use crate::transform::letter_value;
    let mut out =
        vec![(letter_value::OUTLIERS_KEY.to_string(), letter_value::OUTLIERS_COLUMNS.to_vec())];
    out.extend(
        (1..=letter_value::K_MAX_NAMED)
            .map(|k| (letter_value::depth_key(k), letter_value::DEPTH_COLUMNS.to_vec())),
    );
    out
}

/// Look up one transform kind's registry entry by module name (see the
/// module doc for what `kind` means). `None` for an unknown kind.
pub(crate) fn entry_for_kind(kind: &str) -> Option<TransformRegistryEntry> {
    let fit_outputs = || FIT_OUTPUT_COLUMNS.to_vec();
    Some(match kind {
        // `BinSpec` carries `#[serde(into = "BinWire", from = "BinWire")]`
        // (bin.rs) — the wire it actually (de)serializes through is the flat
        // `BinWire` shim's `bin_count`/`bin_width` keys, not the Rust-side
        // typed `mode: BinMode` field the struct exposes to callers within
        // this crate. `params_match_spec_struct_serde_fields` (tests.rs)
        // deserializes `BinSpec` directly (as any real caller's `transforms_json`
        // payload does) and caught this: an earlier manual param audit
        // read the Rust field name instead of the real wire projection.
        // batch-E T7 (spec §6.5, corrected in cycle-2 review): `agg_field`/
        // `agg` are optional and unset by default, but `agg_value` is
        // declared HERE unconditionally rather than carved out under the
        // "default construction" convention — unlike Contour's fill=true
        // isoband schema (a strict SUBSET of the declared isoline columns,
        // so every column Python actually reads is already declared),
        // `agg_value` would otherwise be the first Python-consumed bin/
        // registry column declared NOWHERE, leaving T12's lowering no
        // `registry_output_columns`-checked path to the name and forcing a
        // bare string literal — the exact drift class this registry exists
        // to close. `agg_value` is CONDITIONALLY present at apply-time (only
        // when `agg` is set); `bin_outputs_match_actual_schema` (tests.rs)
        // asserts the no-agg default schema by explicitly excluding it from
        // the declared list, and `bin_outputs_match_actual_schema_with_agg`
        // asserts the with-agg schema against the full declared list
        // unchanged — the "conditional declared column" allowance now lives
        // on the no-agg side instead of being synthesized on the with-agg
        // side.
        "bin" => TransformRegistryEntry {
            methods: crate::transform::aggregate::AggFn::all(),
            outputs: vec![
                ("bin_start", DataType::Float64),
                ("bin_end", DataType::Float64),
                ("count", DataType::UInt64),
                ("density", DataType::Float64),
                ("agg_value", DataType::Float64),
            ],
            secondary_outputs: vec![],
            params: vec![
                "field", "bin_count", "bin_width", "extent", "nice", "cumulative",
                "shared_extent", "groupby", "agg_field", "agg", "name",
            ],
        },
        "bin_2d" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![
                ("x_lo", DataType::Float64),
                ("x_hi", DataType::Float64),
                ("y_lo", DataType::Float64),
                ("y_hi", DataType::Float64),
                ("count", DataType::Int64),
            ],
            secondary_outputs: vec![],
            params: vec![
                "x", "y", "bins_x", "bins_y", "extent_x", "extent_y", "cumulative", "name",
            ],
        },
        // `methods` reads the SAME const the parser's pre-check and error
        // string read (`crate::transform::kde::KERNELS`) — batch-D T11,
        // mirroring `smooth`'s `METHODS` precedent below. See `KERNELS`'s doc
        // comment in `kde.rs`.
        "kde" => TransformRegistryEntry {
            methods: crate::transform::kde::KERNELS.to_vec(),
            outputs: vec![("value", DataType::Float64), ("density", DataType::Float64)],
            secondary_outputs: vec![],
            params: vec![
                "field", "bandwidth", "bw_adjust", "n", "extent", "cumulative", "shared_extent",
                "kernel", "groupby", "name",
            ],
        },
        // `methods` reads the SAME const the parser's pre-check and error
        // string read (`crate::transform::smooth::METHODS`), which is what
        // makes the drift guard structural rather than test-detected. See
        // `METHODS`'s doc comment in `smooth.rs`.
        "smooth" => TransformRegistryEntry {
            methods: crate::transform::smooth::METHODS.to_vec(),
            outputs: fit_outputs(),
            secondary_outputs: vec![],
            params: vec![
                "x", "y", "method", "ci", "bandwidth", "degree", "k", "n", "seed", "x_bins",
                "x_estimator", "output", "inject_zero_ref", "inject_metrics", "groupby", "name",
                "x_range",
            ],
        },
        "aggregate" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![],
            secondary_outputs: vec![],
            params: vec!["ops", "groupby", "name"],
        },
        "summary" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![
                ("mean", DataType::Float64),
                ("lower", DataType::Float64),
                ("upper", DataType::Float64),
            ],
            secondary_outputs: vec![],
            params: vec!["field", "groupby", "error_fn", "ci", "n_boot", "seed", "name"],
        },
        "outliers" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![],
            secondary_outputs: vec![],
            params: vec!["field", "groupby", "extent", "name"],
        },
        "error_extent" => TransformRegistryEntry {
            methods: crate::transform::error_extent::METHODS.to_vec(),
            outputs: vec![
                ("mean", DataType::Float64),
                ("lower", DataType::Float64),
                ("upper", DataType::Float64),
            ],
            secondary_outputs: vec![],
            params: vec!["field", "groupby", "method", "seed", "n_boot", "name"],
        },
        "box_stats" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![
                ("q1", DataType::Float64),
                ("median", DataType::Float64),
                ("q3", DataType::Float64),
                ("lower_whisker", DataType::Float64),
                ("upper_whisker", DataType::Float64),
            ],
            secondary_outputs: vec![],
            params: vec!["field", "groupby", "whisker_extent", "name"],
        },
        "violin" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![
                ("group_id", DataType::UInt32),
                ("violin_x", DataType::Float64),
                ("violin_y", DataType::Float64),
                ("__pos_x_offset__", DataType::Float64),
            ],
            secondary_outputs: vec![],
            params: vec![
                "field", "groupby", "bandwidth", "bw_adjust", "n", "width", "extent",
                "shared_extent", "name", "horizontal",
            ],
        },
        "kde_2d" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![
                ("grid_x", list_f64_dtype()),
                ("grid_y", list_f64_dtype()),
                ("density", list_f64_dtype()),
                ("nx", DataType::UInt32),
                ("ny", DataType::UInt32),
                ("extent", list_f64_dtype()),
            ],
            secondary_outputs: vec![],
            params: vec!["x", "y", "bandwidth", "n", "extent", "groupby", "name"],
        },
        // Default construction is `fill=false` (contour.rs's `default_fill`
        // equivalent is the bool default), whose real schema is the 6-column
        // isoline form (`out_schema_isoline`, contour.rs) — `contour_x2`/
        // `contour_y2` are the segment's end vertex, not optional. `fill=true`
        // emits the 4-column isoband form instead; see the module doc's
        // "under default construction" note.
        "contour" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![
                ("level_id", DataType::UInt32),
                ("level_value", DataType::Float64),
                ("contour_x", DataType::Float64),
                ("contour_y", DataType::Float64),
                ("contour_x2", DataType::Float64),
                ("contour_y2", DataType::Float64),
            ],
            secondary_outputs: vec![],
            params: vec!["thresholds", "fill", "smooth", "name"],
        },
        // `methods` reads the SAME const the parser's pre-check and error
        // string read (`crate::transform::qq::DISTRIBUTIONS`) — batch-D T11,
        // mirroring `smooth`'s `METHODS`/`kde`'s `KERNELS` precedent. See
        // `DISTRIBUTIONS`'s doc comment in `qq.rs`.
        "qq" => TransformRegistryEntry {
            methods: crate::transform::qq::DISTRIBUTIONS.to_vec(),
            outputs: vec![("theoretical", DataType::Float64), ("sample", DataType::Float64)],
            // Holds for the default `emit_line=true`; `emit_line=false`
            // publishes no secondary batch at all.
            secondary_outputs: vec![(
                crate::transform::qq::QQ_LINE_KEY.to_string(),
                crate::transform::qq::QQ_LINE_COLUMNS.to_vec(),
            )],
            params: vec!["field", "distribution", "dequantize", "emit_line", "seed", "name"],
        },
        // `method` is a closed 7-value vocabulary (`LinkageMethod`) validated
        // hard at `PyLinkage::new` (linkage.rs) — same PyO3-constructor-time
        // shape as `kde`/`qq`/`smooth`/`hex`/`raster`. Declared 2026-09-05;
        // see the module doc's `methods` convention note.
        "linkage" => TransformRegistryEntry {
            methods: crate::transform::linkage::METHODS.to_vec(),
            outputs: crate::transform::linkage::LINKAGE_COLUMNS.to_vec(),
            // The first key re-publishes the primary batch under a stable
            // name; the other three are the dendrogram geometry.
            secondary_outputs: linkage_secondary_outputs(),
            params: vec!["method", "metric", "axis", "z_score", "standard_scale", "name"],
        },
        "raster" => TransformRegistryEntry {
            methods: crate::transform::raster::AGGREGATES.to_vec(),
            outputs: vec![
                ("x_min", DataType::Float64),
                ("x_max", DataType::Float64),
                ("y_min", DataType::Float64),
                ("y_max", DataType::Float64),
                ("width", DataType::UInt32),
                ("height", DataType::UInt32),
                ("pixel_data", DataType::Binary),
            ],
            secondary_outputs: vec![],
            params: vec![
                "x", "y", "aggregate", "field", "resolution", "min_count", "log_scale", "name",
                "extent_x", "extent_y",
            ],
        },
        "hex" => TransformRegistryEntry {
            methods: crate::transform::hex::AGGREGATES.to_vec(),
            outputs: vec![
                ("hex_x", DataType::Float64),
                ("hex_y", DataType::Float64),
                ("hex_id", DataType::Int64),
                ("value", DataType::Float64),
            ],
            secondary_outputs: vec![],
            params: vec![
                "x", "y", "bin_size", "aggregate", "field", "name", "extent_x", "extent_y",
            ],
        },
        // Additive columns under default construction (`orient="vertical"`).
        // `orient="horizontal"` emits `__pos_y_offset__` instead of
        // `__pos_x_offset__` — same "under default construction" carve-out as
        // `contour`. See the module doc's `Swarm`/`Violin` note: unlike
        // `DataWindow`/`DataStack`/`JoinAggregate`, these additive names are
        // NOT caller-controlled, so they're declarable even though the
        // pass-through part of the schema (the input's own columns) isn't.
        "swarm" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![
                ("swarm_x", DataType::Float64),
                ("swarm_y", DataType::Float64),
                ("__pos_x_offset__", DataType::Float64),
            ],
            secondary_outputs: vec![],
            params: vec![
                "category", "value", "point_size", "spacing", "side", "orient", "dodge", "name",
            ],
        },
        "unpivot" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![],
            secondary_outputs: vec![],
            params: vec!["id_vars", "value_vars", "var_name", "value_name", "name"],
        },
        "reorder" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![],
            secondary_outputs: vec![],
            params: vec!["by", "drop_index", "from", "name"],
        },
        "reference_line" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![],
            secondary_outputs: vec![],
            params: vec!["x_field", "y_field", "x", "y", "name"],
        },
        "letter_value" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![
                ("group", DataType::Utf8),
                ("depth", DataType::Int32),
                ("lower", DataType::Float64),
                ("upper", DataType::Float64),
                ("level", DataType::Utf8),
            ],
            secondary_outputs: letter_value_secondary_outputs(),
            params: vec!["value", "group", "k_depth", "outlier_threshold", "name", "sort"],
        },
        "logistic" => TransformRegistryEntry {
            methods: vec![],
            outputs: fit_outputs(),
            secondary_outputs: vec![],
            params: vec!["x", "y", "n_grid", "ci", "max_iter", "tol", "name"],
        },
        "glm" => TransformRegistryEntry {
            methods: vec![],
            outputs: fit_outputs(),
            secondary_outputs: vec![],
            params: vec!["x", "y", "family", "link", "n_grid", "ci", "max_iter", "tol", "name"],
        },
        "robust" => TransformRegistryEntry {
            methods: vec![],
            outputs: fit_outputs(),
            secondary_outputs: vec![],
            params: vec![
                "x", "y", "n_grid", "ci", "huber_c", "max_iter", "tol", "output",
                "inject_zero_ref", "inject_metrics", "name", "x_range",
            ],
        },
        "identity" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![],
            secondary_outputs: vec![],
            params: vec!["name"],
        },
        "filter" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![],
            secondary_outputs: vec![],
            params: vec!["predicate", "name", "param"],
        },
        "calculate" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![],
            secondary_outputs: vec![],
            params: vec!["as_field", "expr", "name"],
        },
        "fold" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![],
            secondary_outputs: vec![],
            params: vec!["fields", "as_", "name"],
        },
        // `pivot` was the one declared vocabulary with no drift guard:
        // `aggregate_values` (pivot.rs) had a silent
        // `_ => clean.iter().sum()` fallback for an unrecognized `op`, so
        // there was no real refusal site to assert a bogus value against —
        // which is exactly where the earlier undetected vocabulary drift
        // (missing `first`/`last`) came from. That fallback is now a typed
        // refusal, so this entry reads `pivot::OPS` and
        // `pivot_methods_accepted_and_bogus_op_rejected` (`tests.rs`) guards
        // it like every other kind's.
        "pivot" => TransformRegistryEntry {
            methods: crate::transform::pivot::OPS.to_vec(),
            outputs: vec![],
            secondary_outputs: vec![],
            params: vec!["field", "value", "groupby", "limit", "op", "name"],
        },
        "join_aggregate" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![],
            secondary_outputs: vec![],
            params: vec!["aggregates", "groupby", "name"],
        },
        "data_window" => TransformRegistryEntry {
            methods: crate::transform::data_window::OPS.to_vec(),
            outputs: vec![],
            secondary_outputs: vec![],
            params: vec!["ops", "sort", "groupby", "frame", "ignore_peers", "name"],
        },
        // `methods` declares `kde`'s vocabulary, per the module doc's
        // `methods` convention above: `kernel` is the literally-named
        // `kernel` parameter with no orthogonal-vocabulary ambiguity, so the
        // rule predicts a non-empty list here.
        // `density_data` has no PyO3-constructor-time acceptance/refusal
        // site (`PyDensityData::new` doesn't expose `kernel` at all —
        // `DensityDataSpec` is built directly from wire JSON by
        // `transform_density`), so unlike `kde`/`qq`/`smooth`/`hex`/
        // `raster`/`linkage` its real enforcement point is `apply()`-time,
        // via `kernel_density_at_grid`'s typed error — same shape as
        // `pivot`'s "no hard-refusal site" note above, except `density_data`
        // DOES have a real refusal site, just at `apply()` rather than
        // `__init__`. `density_data_methods_accepted_and_bogus_kernel_rejected`
        // (`registry/tests.rs`) asserts against that site directly.
        "density_data" => TransformRegistryEntry {
            methods: crate::transform::kde::KERNELS.to_vec(),
            outputs: vec![("value", DataType::Float64), ("density", DataType::Float64)],
            secondary_outputs: vec![],
            params: vec![
                "field", "bandwidth", "groupby", "extent", "steps", "cumulative", "kernel", "as_",
                "name",
            ],
        },
        "regression_data" => TransformRegistryEntry {
            methods: crate::transform::regression_data::METHODS.to_vec(),
            outputs: vec![("x", DataType::Float64), ("y", DataType::Float64)],
            secondary_outputs: vec![],
            params: vec!["x", "y", "method", "order", "groupby", "as_", "name"],
        },
        "loess_data" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![("x", DataType::Float64), ("y", DataType::Float64)],
            secondary_outputs: vec![],
            params: vec!["x", "y", "bandwidth", "groupby", "as_", "name"],
        },
        "impute" => TransformRegistryEntry {
            methods: crate::transform::impute::METHODS.to_vec(),
            outputs: vec![],
            secondary_outputs: vec![],
            params: vec!["field", "method", "value", "groupby", "key", "keyvals", "name"],
        },
        "flatten" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![],
            secondary_outputs: vec![],
            params: vec!["fields", "as_", "name"],
        },
        "sample" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![],
            secondary_outputs: vec![],
            params: vec!["n", "seed", "name"],
        },
        "top_k" => TransformRegistryEntry {
            methods: crate::transform::top_k::OPS.to_vec(),
            outputs: vec![],
            secondary_outputs: vec![],
            params: vec!["n", "field", "op", "sort", "groupby", "name"],
        },
        "data_stack" => TransformRegistryEntry {
            methods: crate::transform::data_stack::OFFSETS.to_vec(),
            outputs: vec![],
            secondary_outputs: vec![],
            params: vec!["field", "groupby", "sort", "as_", "offset", "name"],
        },
        "timeunit" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![],
            secondary_outputs: vec![],
            params: vec!["field", "unit", "utc", "as_", "name"],
        },
        "data_bin" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![],
            secondary_outputs: vec![],
            params: vec!["field", "as_", "maxbins", "step", "nice", "name", "extent"],
        },
        "data_aggregate" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![],
            secondary_outputs: vec![],
            params: vec!["aggregates", "groupby", "name"],
        },
        _ => return None,
    })
}

/// `DataType` -> the dtype string reported to Python. Deliberately narrow:
/// covers exactly the dtypes this registry's `outputs` entries use. A dtype
/// this table starts using without a matching arm here fails loudly (not
/// silently mislabeled) via the `_ => unimplemented!` fallback, which only
/// fires for a dtype no `entry_for_kind` arm currently produces.
fn dtype_label(dt: &DataType) -> String {
    match dt {
        DataType::Float64 => "Float64".to_string(),
        DataType::Int64 => "Int64".to_string(),
        DataType::Int32 => "Int32".to_string(),
        DataType::UInt64 => "UInt64".to_string(),
        DataType::UInt32 => "UInt32".to_string(),
        DataType::Utf8 => "Utf8".to_string(),
        DataType::Boolean => "Boolean".to_string(),
        DataType::Binary => "Binary".to_string(),
        DataType::List(field) => format!("List<{}>", dtype_label(field.data_type())),
        other => unimplemented!(
            "transform registry: no Python-facing label defined for dtype {other:?}; \
             add one to `dtype_label` in transform/registry/mod.rs"
        ),
    }
}

/// `ferrum._core.transform_registry(kind)` — spec §4.6, batch-D task 1.
/// Returns `{"methods": [...], "outputs": [(name, dtype), ...],
/// "secondary_outputs": [(key, [(name, dtype), ...]), ...], "params": [...]}`
/// for a known transform kind (module name — see the module doc); raises
/// `ValueError` listing every known kind otherwise.
#[pyfunction]
pub fn transform_registry(py: Python<'_>, kind: &str) -> PyResult<Py<PyDict>> {
    let entry = entry_for_kind(kind).ok_or_else(|| {
        let mut known = all_transform_kinds();
        known.sort_unstable();
        PyValueError::new_err(format!(
            "unknown transform kind '{kind}'; known kinds: {}",
            known.join(", ")
        ))
    })?;
    let dict = PyDict::new(py);
    dict.set_item("methods", entry.methods)?;
    let outputs: Vec<(&'static str, String)> = entry
        .outputs
        .iter()
        .map(|(name, dt)| (*name, dtype_label(dt)))
        .collect();
    dict.set_item("outputs", outputs)?;
    let secondary: Vec<(String, Vec<(&'static str, String)>)> = entry
        .secondary_outputs
        .iter()
        .map(|(key, columns)| {
            (
                key.clone(),
                columns.iter().map(|(name, dt)| (*name, dtype_label(dt))).collect(),
            )
        })
        .collect();
    dict.set_item("secondary_outputs", secondary)?;
    dict.set_item("params", entry.params)?;
    Ok(dict.into())
}

#[cfg(test)]
mod tests;
