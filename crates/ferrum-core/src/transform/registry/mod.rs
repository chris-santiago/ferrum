//! Transform metadata registry (batch-D spec §4.6, decision doc M6a).
//!
//! One Rust-owned table of per-transform-kind metadata — method/kernel/
//! distribution vocabularies, primary output schemas, and wire parameter
//! names — co-located with [`super::core::for_each_transform`] so that a new
//! transform variant is a registry decision, not an afterthought (the
//! `scale_accepted_keys` move, applied to the transform layer). Exposed to
//! Python as `ferrum._core.transform_registry(kind)`; later tasks (12, 13)
//! delete Python's hand-mirrored `_VALID_KERNELS`, qq distribution lists,
//! `lmplot`'s method set, and hardcoded fit-output-column literals in favor
//! of lookups against this table.
//!
//! **`kind` is the transform's *module name*** — the second column of
//! `for_each_transform!`'s table (`"bin"`, `"bin_2d"`, `"regression_data"`,
//! …), not necessarily the wire `"type"` tag serde derives for
//! `TransformSpec`'s internally-tagged JSON form. **Corrected 2026-09-04**
//! (round-2 cycle-1 review): the two do NOT agree everywhere except
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
//!   own column). **`DataStack` corrected 2026-09-05** (round-3 cycle-2
//!   review): it was previously grouped here under "no *stable* default name
//!   to declare", which is false — `DataStackSpec.as_` is a single top-level
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
//!   decision (out of scope for this round);
//! - variable-arity transforms (`Aggregate`, `DataAggregate`) whose column
//!   count/names come from a `Vec<Op>` of caller-supplied length;
//! - transforms whose sole output column name and/or dtype is driven by a
//!   REQUIRED (no-default) user parameter, or whose declared default is
//!   `None` meaning "derive from another required parameter" rather than a
//!   static literal (`ReferenceLine`, `Calculate`, `TimeUnit`, `DataBin`,
//!   `Flatten`, `Pivot`) — there is no default name/dtype to declare;
//! - `Unpivot` and `Fold`. **Corrected 2026-09-05** (round-3 cycle-2 review):
//!   these were previously grouped in the bullet above under "REQUIRED
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
//! **Named *secondary* outputs are out of `outputs` by contract, not by
//! omission.** `Qq` (`secondary_outputs`: the reference-line's `qq_line_x`/
//! `qq_line_y` batch), `LetterValue` (`secondary_outputs`: the outlier-row
//! batch), and `Linkage` (`secondary_outputs`: dendrogram coordinates) each
//! emit an additional NAMED batch alongside their primary rows via their
//! module's `secondary_outputs()` function. `outputs` here covers only the
//! *primary* output schema (spec §4.6's wording) — a consumer needing the
//! secondary batch's columns reads them from `qq.rs`/`letter_value.rs`/
//! `linkage.rs` directly; Python's mirrors of those secondary column names
//! stay as literals rather than becoming registry lookups (this is not an
//! incomplete mirror deletion — see the batch-D task-12 handoff note).
//!
//! **`methods` convention.** Only the kind's single most prominent free-string
//! method/kernel/distribution/operation vocabulary — the parameter most
//! directly named `method`/`kernel`/`distribution`/`op`/`aggregate`/`offset`.
//! `Glm`/`Logistic`/`Robust` are deliberately `methods: []`: `Glm`'s
//! `family`/`link` are two orthogonal closed vocabularies, and cramming both
//! into one flat list would be ambiguous to any consumer without inventing a
//! new registry shape — out of scope for this task (no consumer needs it
//! yet; `parse_family`/`parse_link` in `glm.rs` remain the source of truth).
//! **`Linkage` declared 2026-09-05** (round-3 cycle-2 review): it carries a
//! closed 7-value `method` vocabulary (`LinkageMethod`, `linkage.rs`) behind
//! a PyO3-constructor-time hard validation site shaped identically to
//! `kde`/`qq`/`smooth`/`hex`/`raster` (`PyLinkage::new`, `linkage.rs`), so the
//! rule above predicts a non-empty list for it and there is no ambiguity to
//! carve out — it was simply missed in round 1/2, not a deliberate `[]`.
//!
//! **`params`** is the spec struct's real WIRE-accepted field set — the set
//! `params_match_spec_struct_serde_fields` (`tests.rs`) proves by mining
//! serde's own `deny_unknown_fields` rejection, not a reading of the Rust
//! field names. **Corrected 2026-09-05** (round-3 cycle-2 review): this is
//! NOT "identical to the Rust field name at every site in this crate" — the
//! `"bin"` entry below is the one site where they diverge: `BinSpec` carries
//! `#[serde(into = "BinWire", from = "BinWire")]`, so the wire it actually
//! (de)serializes through is `BinWire`'s flat `bin_count`/`bin_width` keys,
//! not the Rust-side typed `mode: BinMode` field the struct exposes to
//! in-crate callers (round-2's own manual audit read the Rust field name and
//! missed this; the drift guard caught it). No other top-level spec struct in
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
//! (`render::binding::theme_dict_tests`, `render::chart_config::tests`) and
//! widening its return shape is out of this round's doc-and-test-only scope.

use std::sync::Arc;

use arrow::datatypes::{DataType, Field};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// One transform kind's registry entry. See the module doc for the
/// `outputs`/`methods` inclusion conventions.
pub(crate) struct TransformRegistryEntry {
    pub methods: Vec<&'static str>,
    pub outputs: Vec<(&'static str, DataType)>,
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
        // payload does) and caught this: round-1 cycle-1's manual param audit
        // read the Rust field name instead of the real wire projection.
        "bin" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![
                ("bin_start", DataType::Float64),
                ("bin_end", DataType::Float64),
                ("count", DataType::UInt64),
                ("density", DataType::Float64),
            ],
            params: vec![
                "field", "bin_count", "bin_width", "extent", "nice", "cumulative",
                "shared_extent", "groupby", "name",
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
            params: vec![
                "x", "y", "bins_x", "bins_y", "extent_x", "extent_y", "cumulative", "name",
            ],
        },
        "kde" => TransformRegistryEntry {
            methods: vec!["gaussian", "epanechnikov", "epan", "tophat", "uniform", "cosine"],
            outputs: vec![("value", DataType::Float64), ("density", DataType::Float64)],
            params: vec![
                "field", "bandwidth", "bw_adjust", "n", "extent", "cumulative", "shared_extent",
                "kernel", "groupby", "name",
            ],
        },
        "smooth" => TransformRegistryEntry {
            methods: vec!["lm", "linear", "loess", "quadratic", "cubic", "log", "sqrt"],
            outputs: fit_outputs(),
            params: vec![
                "x", "y", "method", "ci", "bandwidth", "degree", "n", "seed", "x_bins",
                "x_estimator", "output", "inject_zero_ref", "inject_metrics", "groupby", "name",
                "x_range",
            ],
        },
        "aggregate" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![],
            params: vec!["ops", "groupby", "name"],
        },
        "summary" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![
                ("mean", DataType::Float64),
                ("lower", DataType::Float64),
                ("upper", DataType::Float64),
            ],
            params: vec!["field", "groupby", "error_fn", "ci", "n_boot", "seed", "name"],
        },
        "outliers" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![],
            params: vec!["field", "groupby", "extent", "name"],
        },
        "error_extent" => TransformRegistryEntry {
            methods: vec!["ci", "stderr", "stdev", "iqr"],
            outputs: vec![
                ("mean", DataType::Float64),
                ("lower", DataType::Float64),
                ("upper", DataType::Float64),
            ],
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
            params: vec!["thresholds", "fill", "smooth", "name"],
        },
        "qq" => TransformRegistryEntry {
            methods: vec!["normal", "uniform", "exponential"],
            outputs: vec![("theoretical", DataType::Float64), ("sample", DataType::Float64)],
            params: vec!["field", "distribution", "dequantize", "emit_line", "seed", "name"],
        },
        // `method` is a closed 7-value vocabulary (`LinkageMethod`) validated
        // hard at `PyLinkage::new` (linkage.rs) — same PyO3-constructor-time
        // shape as `kde`/`qq`/`smooth`/`hex`/`raster`. Declared 2026-09-05
        // (round-3 cycle-2 review); see the module doc's `methods` convention
        // note.
        "linkage" => TransformRegistryEntry {
            methods: vec!["single", "complete", "average", "weighted", "centroid", "median", "ward"],
            outputs: vec![
                ("node_id", DataType::Int64),
                ("left", DataType::Int64),
                ("right", DataType::Int64),
                ("distance", DataType::Float64),
                ("n_obs", DataType::Int64),
            ],
            params: vec!["method", "metric", "axis", "z_score", "standard_scale", "name"],
        },
        "raster" => TransformRegistryEntry {
            methods: vec!["count", "density", "mean", "sum", "any"],
            outputs: vec![
                ("x_min", DataType::Float64),
                ("x_max", DataType::Float64),
                ("y_min", DataType::Float64),
                ("y_max", DataType::Float64),
                ("width", DataType::UInt32),
                ("height", DataType::UInt32),
                ("pixel_data", DataType::Binary),
            ],
            params: vec![
                "x", "y", "aggregate", "field", "resolution", "min_count", "log_scale", "name",
                "extent_x", "extent_y",
            ],
        },
        "hex" => TransformRegistryEntry {
            methods: vec!["count", "mean", "sum", "min", "max", "median", "std", "var"],
            outputs: vec![
                ("hex_x", DataType::Float64),
                ("hex_y", DataType::Float64),
                ("hex_id", DataType::Int64),
                ("value", DataType::Float64),
            ],
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
            params: vec![
                "category", "value", "point_size", "spacing", "side", "orient", "dodge", "name",
            ],
        },
        "unpivot" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![],
            params: vec!["id_vars", "value_vars", "var_name", "value_name", "name"],
        },
        "reorder" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![],
            params: vec!["by", "drop_index", "from", "name"],
        },
        "reference_line" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![],
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
            params: vec!["value", "group", "k_depth", "outlier_threshold", "name", "sort"],
        },
        "logistic" => TransformRegistryEntry {
            methods: vec![],
            outputs: fit_outputs(),
            params: vec!["x", "y", "n_grid", "ci", "max_iter", "tol", "name"],
        },
        "glm" => TransformRegistryEntry {
            methods: vec![],
            outputs: fit_outputs(),
            params: vec!["x", "y", "family", "link", "n_grid", "ci", "max_iter", "tol", "name"],
        },
        "robust" => TransformRegistryEntry {
            methods: vec![],
            outputs: fit_outputs(),
            params: vec![
                "x", "y", "n_grid", "ci", "huber_c", "max_iter", "tol", "output",
                "inject_zero_ref", "inject_metrics", "name", "x_range",
            ],
        },
        "identity" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![],
            params: vec!["name"],
        },
        "filter" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![],
            params: vec!["predicate", "name", "param"],
        },
        "calculate" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![],
            params: vec!["as_field", "expr", "name"],
        },
        "fold" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![],
            params: vec!["fields", "as_", "name"],
        },
        // No methods-acceptance drift guard exists for `pivot` (round-3
        // cycle-2 review): `aggregate_values` (pivot.rs) has a silent
        // `_ => clean.iter().sum()` fallback for an unrecognized `op`, so
        // there is no real refusal site to assert a bogus value against —
        // unlike `kde`/`qq`/`smooth`/`hex`/`raster`/`linkage`, which all
        // reject a bogus method hard at PyO3-constructor time. This is
        // exactly where round-1's undetected `pivot` vocabulary drift (missing
        // `first`/`last`) came from, so a future edit to this list still
        // needs a manual cross-check against `pivot.rs`'s real `match` arms
        // until `aggregate_values` gains a hard-refusal path.
        "pivot" => TransformRegistryEntry {
            methods: vec!["sum", "mean", "count", "min", "max", "first", "last"],
            outputs: vec![],
            params: vec!["field", "value", "groupby", "limit", "op", "name"],
        },
        "join_aggregate" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![],
            params: vec!["aggregates", "groupby", "name"],
        },
        "data_window" => TransformRegistryEntry {
            methods: vec![
                "row_number", "rank", "dense_rank", "lag", "lead", "first_value", "last_value",
                "sum", "mean", "count", "min", "max",
            ],
            outputs: vec![],
            params: vec!["ops", "sort", "groupby", "frame", "ignore_peers", "name"],
        },
        "density_data" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![("value", DataType::Float64), ("density", DataType::Float64)],
            params: vec![
                "field", "bandwidth", "groupby", "extent", "steps", "cumulative", "as_", "name",
            ],
        },
        "regression_data" => TransformRegistryEntry {
            methods: vec!["linear", "poly", "exp", "log", "pow"],
            outputs: vec![("x", DataType::Float64), ("y", DataType::Float64)],
            params: vec!["x", "y", "method", "order", "groupby", "as_", "name"],
        },
        "loess_data" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![("x", DataType::Float64), ("y", DataType::Float64)],
            params: vec!["x", "y", "bandwidth", "groupby", "as_", "name"],
        },
        "impute" => TransformRegistryEntry {
            methods: vec!["value", "mean", "median", "min", "max"],
            outputs: vec![],
            params: vec!["field", "method", "value", "groupby", "key", "name"],
        },
        "flatten" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![],
            params: vec!["fields", "as_", "name"],
        },
        "sample" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![],
            params: vec!["n", "seed", "name"],
        },
        "top_k" => TransformRegistryEntry {
            methods: vec!["sum", "mean", "count", "min", "max"],
            outputs: vec![],
            params: vec!["n", "field", "op", "sort", "name"],
        },
        "data_stack" => TransformRegistryEntry {
            methods: vec!["zero", "normalize", "center"],
            outputs: vec![],
            params: vec!["field", "groupby", "sort", "as_", "offset", "name"],
        },
        "timeunit" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![],
            params: vec!["field", "unit", "utc", "as_", "name"],
        },
        "data_bin" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![],
            params: vec!["field", "as_", "maxbins", "step", "nice", "name", "extent"],
        },
        "data_aggregate" => TransformRegistryEntry {
            methods: vec![],
            outputs: vec![],
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
/// Returns `{"methods": [...], "outputs": [(name, dtype), ...], "params":
/// [...]}` for a known transform kind (module name — see the module doc);
/// raises `ValueError` listing every known kind otherwise.
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
    dict.set_item("params", entry.params)?;
    Ok(dict.into())
}

#[cfg(test)]
mod tests;
