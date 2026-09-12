//! TransformContext: render-time info passed to transforms that need viewport
//! or cross-transform output access.
//!
//! Used by:
//! - Raster (resolution="screen"), Swarm (point radius unit conversion), and
//!   Violin (band_pixels) — all three read `panel_pixel_size`, populated by
//!   the production pipeline as a pre-layout estimate (see the field's own
//!   doc for the estimate's exact semantics and limits).
//! - Reorder with `from=<named_output>` (Phase 9 finalize) — reads `named_outputs`
//!   to look up the index column from a sibling named transform output.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use arrow::record_batch::RecordBatch;

/// A transform degraded (rather than refused) at a specific site — the
/// degrade-warning channel (GH #162). Mirrors [`crate::render::RenderWarning`]'s
/// contract exactly: one Rust variant per distinct degrade shape, an owned
/// `String` for every text field (never `&'static str` — see that enum's doc
/// for why: `Deserialize<'de>` would otherwise require `'de: 'static`), and a
/// [`std::fmt::Display`] impl that is the sole user-facing text — no Rust
/// type or field names leak into it.
///
/// Instances are pushed onto [`TransformContext::warnings`] by the transform
/// that degraded and drained by `render::prepare::run_transforms_with_facet`
/// into the render pipeline's own `Vec<RenderWarning>` (wrapped via
/// [`crate::render::RenderWarning::Transform`]), which
/// `render::binding::emit_warnings` delivers to Python as a `UserWarning` —
/// the same single delivery point every other render warning already uses.
/// Emitting one of these must never change the transform's OUTPUT: the
/// degrade this reports (an all-NaN fit, a clamped parameter, a
/// whole-column-fallback fill, …) already happened before this batch;
/// batch-D's "the group's own degenerate output still contributes its rows
/// and its key" doctrine (`group_key.rs`'s doc) is unchanged by this
/// channel — it only makes the degradation observable.
/// `tag = "kind"`, not `"type"`: `RenderWarning::Transform` wraps this enum
/// as a newtype variant (mirroring `RenderWarning::Layout(LayoutWarning)`),
/// and `RenderWarning` itself already owns the outer `"type"` tag — the same
/// collision `LayoutWarning` avoids by using `"kind"` (see `RenderWarning`'s
/// own doc note on this).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum TransformWarning {
    /// A fit (the whole frame, or one `groupby`/grouped-two-column-fit
    /// partition) could not be computed — fewer than the minimum usable
    /// points, zero-variance x, a non-convergent or singular design — and
    /// degraded to an all-NaN/no-op output instead of failing the whole
    /// render. See `smooth.rs`'s "Degenerate-input policy" module doc for
    /// the canonical rationale this variant reports: a group's (or a facet
    /// panel's) own degenerate output still contributes its rows and keeps
    /// its key, so the group stays in the legend rather than silently
    /// vanishing.
    ///
    /// `transform` names the producing transform (its Python-facing name,
    /// e.g. `"Smooth"`); `group` is a human-readable label for the
    /// degenerate slice (`"g=A"`, `"g1=A, g2=2"`) when it is one groupby
    /// group or grouped-fit partition, `None` for a genuine whole-frame
    /// degrade; `reason` is a short user-facing clause naming why (e.g.
    /// `"fewer than 2 usable points"`, `"zero x variance"`).
    DegenerateFit {
        transform: String,
        group: Option<String>,
        reason: String,
    },
    /// A caller-requested parameter was silently clamped to a value the data
    /// (or the transform's own bounds) could actually support — e.g.
    /// `Smooth(method="gam", k=200)` on data with fewer distinct x values
    /// than requested, or `LetterValue`'s per-strategy computed depth
    /// exceeding its named-output ceiling. `requested`/`applied` are already
    /// formatted as user-facing strings, so the message never leaks a Rust
    /// numeric type.
    ParameterClamped {
        transform: String,
        group: Option<String>,
        param: String,
        requested: String,
        applied: String,
    },
    /// A per-group (or per-row) fill/statistic could not be computed from
    /// that group's own data and fell back to a wider one instead of the
    /// method's own degenerate default — e.g. `Impute`'s whole-column
    /// statistic standing in for an all-null group or a null-keyed row.
    /// `reason` names what fell back and to what (e.g. `"group has no
    /// non-null values; used the whole column's mean instead"`).
    FallbackApplied {
        transform: String,
        group: Option<String>,
        reason: String,
    },
}

impl std::fmt::Display for TransformWarning {
    /// User-facing warning text forwarded to Python's ``warnings.warn`` by
    /// `render::binding::emit_warnings` (via `RenderWarning::Transform`'s own
    /// `Display` delegation). One sentence, no internal Rust names — mirrors
    /// `RenderWarning`'s own Display contract exactly.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransformWarning::DegenerateFit { transform, group, reason } => match group {
                Some(g) => write!(
                    f,
                    "{transform}: could not fit group {g} ({reason}); it degraded to an \
                     empty/flat output instead of failing the whole chart"
                ),
                None => write!(
                    f,
                    "{transform}: could not fit the data ({reason}); it degraded to an \
                     empty/flat output instead of failing the whole chart"
                ),
            },
            TransformWarning::ParameterClamped { transform, group, param, requested, applied } => {
                match group {
                    Some(g) => write!(
                        f,
                        "{transform}: {param}={requested} could not be honored for group {g}; \
                         {applied} was used instead"
                    ),
                    None => write!(
                        f,
                        "{transform}: {param}={requested} could not be honored; {applied} was \
                         used instead"
                    ),
                }
            }
            TransformWarning::FallbackApplied { transform, group, reason } => match group {
                Some(g) => write!(f, "{transform}: group {g} {reason}"),
                None => write!(f, "{transform}: {reason}"),
            },
        }
    }
}

/// Render one groupby [`crate::transform::group_key::KeyValue`] as the
/// human-readable text `describe_group_row`/`describe_group_keys` embed in a
/// `"col=value"` label — the one place either of them turns a `KeyValue`
/// into a string, so both stay byte-identical for the same value.
fn format_key_value(kv: &crate::transform::group_key::KeyValue) -> String {
    use crate::transform::group_key::KeyValue;
    match kv {
        KeyValue::Null => "null".to_string(),
        KeyValue::Str(s) => s.clone(),
        KeyValue::Float(bits) => format!("{}", f64::from_bits(*bits)),
        KeyValue::Int(i) => i.to_string(),
        KeyValue::UInt(u) => u.to_string(),
        KeyValue::Bool(b) => (*b != 0).to_string(),
    }
}

/// Format a groupby/facet row's key values as a human-readable label for a
/// [`TransformWarning`]'s `group` field, e.g. `"g=A"` or `"g1=A, g2=2"`.
///
/// Reads the row's raw values directly from `batch` rather than requiring
/// the caller to already hold a `group_key::KeyValue` — the shared
/// `group_key::grouped_two_column_fit` helper (`regression_data.rs`,
/// `loess_data.rs`) hands its per-group closure only the row indices, not
/// the parsed key, so a caller reached through it can still label the group
/// it is inside of by reading `batch` at `ixs[0]` (any row in the group
/// carries the same key by construction). Callers that already iterate a
/// parsed `Vec<KeyValue>` directly (`smooth.rs`, `robust.rs`, `kde/mod.rs`)
/// use this too, for one consistent label format across every adopter.
///
/// Returns `None` for an empty `group_cols` (no grouping) or when a column
/// no longer resolves — the caller then reports the degrade with `group:
/// None` rather than losing the whole warning over a labeling failure.
pub(crate) fn describe_group_row(
    batch: &RecordBatch,
    group_cols: &[String],
    row: usize,
) -> Option<String> {
    use crate::transform::group_key::{groupby_key_at, is_groupby_supported_dtype};
    if group_cols.is_empty() {
        return None;
    }
    let schema = batch.schema();
    let mut parts = Vec::with_capacity(group_cols.len());
    for col in group_cols {
        let idx = schema.index_of(col).ok()?;
        let arr = batch.column(idx);
        let dt = arr.data_type();
        if !is_groupby_supported_dtype(dt) {
            return None;
        }
        let kv = groupby_key_at(arr.as_ref(), dt, row)?;
        parts.push(format!("{col}={}", format_key_value(&kv)));
    }
    Some(parts.join(", "))
}

/// Same label format as [`describe_group_row`], for a caller that already
/// holds the parsed group key (`smooth.rs`/`robust.rs`/`kde/mod.rs`'s
/// `apply_grouped` loops, which iterate `group_partition_multi`'s own
/// `Vec<KeyValue>` group order directly) and has no `RecordBatch` row to
/// re-derive it from. `group_cols` supplies the column names in the same
/// order `keys` carries their values.
pub(crate) fn describe_group_keys(
    group_cols: &[String],
    keys: &[crate::transform::group_key::KeyValue],
) -> String {
    group_cols
        .iter()
        .zip(keys.iter())
        .map(|(col, kv)| format!("{col}={}", format_key_value(kv)))
        .collect::<Vec<_>>()
        .join(", ")
}

#[derive(Debug, Clone, Default)]
pub struct TransformContext {
    /// Panel pixel size (width, height). For raster `resolution="screen"`,
    /// this is the grid dimension. For swarm and violin, used to convert
    /// point/band pixels to data-space radius via the value-axis scale.
    ///
    /// **This is a pre-layout ESTIMATE, not the true rendered panel size**
    /// (M2, spec §4.2 of `2026-09-05-batch-d-transform-engine-design.md`).
    /// The production render pipeline populates it, at the single
    /// construction site `render/prepare/mod.rs::run_transforms_with_facet`,
    /// with `layout::estimate_panel_pixel_size`: the viewport minus the
    /// theme's padding, divided per-axis by the facet grid's column/row
    /// counts when the chart facets. It deliberately EXCLUDES axis titles,
    /// tick labels, legends, and facet strip bands — all unknowable before
    /// layout runs, since layout itself measures the transformed output this
    /// context seeds. True panel size is therefore circular pre-layout; this
    /// field is the best available approximation, named as such everywhere
    /// it is read (`raster.rs`, `swarm.rs`, `violin.rs`).
    ///
    /// `None` on a bare `apply()` call with no render context (direct calls,
    /// unit tests, WASM/notebook direct usage) — each of the three readers
    /// above falls back to a fixed default in that case; those fallbacks are
    /// unaffected by this field's production population.
    pub panel_pixel_size: Option<(u32, u32)>,
    /// Named outputs produced by earlier transforms in the same `apply_transforms_named`
    /// run. Used by Reorder(from='<name>') to access a sibling order column.
    pub named_outputs: HashMap<String, RecordBatch>,
    /// True when `batch` (the argument to `apply`/`apply_with_context`) is one
    /// partition of a conceptually larger frame, rather than the whole frame
    /// itself — e.g. one facet panel after
    /// `render::prepare::run_transforms_with_facet` partitions the input by
    /// the facet field(s) BEFORE running transforms (the sole production
    /// construction site that sets this `true` today, batch-D T9 round 3).
    ///
    /// `false` (the derived `Default`) means "genuinely the whole frame":
    /// the non-faceted production path, a bare `apply()`/`apply_with_context`
    /// call with no context (direct calls, unit tests, WASM/notebook direct
    /// usage), and every `TransformContext` built before this field existed.
    ///
    /// Consumers use this to distinguish "the caller asked for something the
    /// data cannot support" (refuse) from "this data subset happens to be
    /// too sparse for the fit" (degrade), matching how a `groupby` group
    /// already degrades rather than refusing — see `smooth::apply_one_group`'s
    /// degenerate-input policy doc for the one consumer today.
    ///
    /// Audited (batch-D T9 round 4) for every other place a frame might
    /// reach a transform as a fragment, per the spec review's ⚠️: `compare=`
    /// small multiples (`ferrum.plots._helpers._compose_compare`) and
    /// composite/repeat leaves (`render::composite_render`,
    /// `render::binding::build_composite_leaves`) never partition a batch
    /// inside Rust. Each composite leaf's `data` index selects a whole
    /// batch Python already uploaded as that leaf's own complete dataset —
    /// one dataframe per `compare=` model, sliced in Python before crossing
    /// the PyO3 boundary — and `prepare_render_inputs` is called exactly
    /// once per leaf with no further slicing (`grep` over the crate finds
    /// no other `TransformContext` construction or `apply_transforms_named`
    /// call outside `run_transforms_with_facet` and the `apply()`
    /// fallbacks). `false` for a composite leaf is therefore the correct
    /// whole-frame-for-that-leaf answer, not an unaudited gap:
    /// `run_transforms_with_facet` remains the ONLY place Rust itself
    /// partitions a batch before running transforms, so it is exhaustively
    /// the only construction site this field needs. A leaf whose OWN
    /// `spec.facet` is set still gets `is_partition = true` correctly,
    /// since `prepare_render_inputs` → `run_transforms_with_facet` runs
    /// per leaf, independent of composite nesting.
    pub is_partition: bool,
    /// The degrade-warning sink (GH #162). `Rc<RefCell<_>>`, not a bare
    /// `RefCell<Vec<_>>`, because `apply_transforms_named` (`core.rs`) does
    /// `let mut step_ctx = ctx.clone();` once per transform in the pipeline —
    /// a bare `RefCell` field would clone into a fresh, independent, and
    /// ultimately discarded sink for every step, silently dropping every
    /// warning a later step pushed. `Rc::clone` shares the same underlying
    /// `Vec` across every clone of a `TransformContext`, so pushes from any
    /// step (or any per-group call within a step) land in the one sink the
    /// ORIGINAL context — the one `render::prepare::run_transforms_with_facet`
    /// constructed and keeps — reads back after the pipeline runs.
    ///
    /// A bare `apply()`/`apply_with_context(..., &TransformContext::default())`
    /// call (direct calls, unit tests, WASM/notebook direct usage) gets an
    /// ephemeral sink of its own: warnings pushed during that call are
    /// dropped along with the throwaway context, exactly like
    /// `panel_pixel_size`'s and `is_partition`'s defaults are inert off the
    /// production path. There is no live PyO3 boundary today that calls a
    /// transform's `apply()` directly (every Python-reachable render goes
    /// through `run_transforms_with_facet`), so this default-context case is
    /// currently theoretical — documented here so a future direct-call
    /// boundary inherits the same "read `ctx.warnings` back after calling"
    /// contract rather than needing to invent one.
    pub warnings: Rc<RefCell<Vec<TransformWarning>>>,
}

impl TransformContext {
    /// Record a degrade at the current call site. Cheap and infallible —
    /// pushes onto the shared sink (see [`TransformContext::warnings`]'s doc
    /// for why `Rc<RefCell<_>>` is load-bearing here).
    pub(crate) fn warn(&self, warning: TransformWarning) {
        self.warnings.borrow_mut().push(warning);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_variants() -> Vec<TransformWarning> {
        vec![
            TransformWarning::DegenerateFit {
                transform: "Smooth".into(),
                group: Some("g=a".into()),
                reason: "fewer than 2 usable points".into(),
            },
            TransformWarning::DegenerateFit {
                transform: "Smooth".into(),
                group: None,
                reason: "zero x variance".into(),
            },
            TransformWarning::ParameterClamped {
                transform: "Smooth".into(),
                group: None,
                param: "k".into(),
                requested: "200".into(),
                applied: "6".into(),
            },
            TransformWarning::ParameterClamped {
                transform: "LetterValue".into(),
                group: Some("g=a".into()),
                param: "k_depth".into(),
                requested: "14".into(),
                applied: "12".into(),
            },
            TransformWarning::FallbackApplied {
                transform: "Impute".into(),
                group: Some("g=b".into()),
                reason: "group has no non-null values; the whole column's mean was used instead"
                    .into(),
            },
            TransformWarning::FallbackApplied {
                transform: "Impute".into(),
                group: None,
                reason: "row's groupby key is null; the whole column's mean was used instead"
                    .into(),
            },
        ]
    }

    /// Round-trip test registration (spec §6): every variant serializes and
    /// deserializes back to itself, mirroring `RenderWarning`'s own
    /// `render_warning_round_trip_each_variant` pattern.
    #[test]
    fn transform_warning_round_trip_each_variant() {
        for w in all_variants() {
            let json = serde_json::to_string(&w).unwrap();
            let parsed: TransformWarning = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, w, "round-trip mismatch for {json}");
        }
    }

    /// Display contract: one user-facing sentence, no internal Rust variant
    /// name leaking into the text — mirrors `RenderWarning`'s own assertion.
    #[test]
    fn transform_warning_display_never_leaks_variant_names() {
        let leak_tokens = ["DegenerateFit", "ParameterClamped", "FallbackApplied"];
        for w in all_variants() {
            let text = w.to_string();
            for token in leak_tokens {
                assert!(
                    !text.contains(token),
                    "Display text {text:?} leaked internal variant name {token:?}"
                );
            }
        }
    }

    #[test]
    fn display_names_the_group_when_present_and_omits_it_otherwise() {
        let with_group = TransformWarning::DegenerateFit {
            transform: "Smooth".into(),
            group: Some("g=a".into()),
            reason: "fewer than 2 usable points".into(),
        };
        assert!(with_group.to_string().contains("g=a"));

        let without_group = TransformWarning::DegenerateFit {
            transform: "Smooth".into(),
            group: None,
            reason: "fewer than 2 usable points".into(),
        };
        assert!(!without_group.to_string().contains("group"));
    }

    /// The keystone gotcha the design doc calls out: `apply_transforms_named`
    /// (`core.rs`) does `let mut step_ctx = ctx.clone();` once per transform.
    /// A push through the CLONE must still be visible through the ORIGINAL —
    /// proof that `Rc<RefCell<_>>` (not a bare `RefCell`) is load-bearing.
    #[test]
    fn warnings_pushed_through_a_clone_are_visible_on_the_original() {
        let ctx = TransformContext::default();
        let step_ctx = ctx.clone();
        step_ctx.warn(TransformWarning::DegenerateFit {
            transform: "Smooth".into(),
            group: None,
            reason: "zero x variance".into(),
        });
        assert_eq!(
            ctx.warnings.borrow().len(),
            1,
            "a warning pushed through a cloned TransformContext must be visible on the original"
        );
    }

    /// Two independent `TransformContext::default()` calls (e.g. two
    /// unrelated bare `apply()` calls) must NOT share a sink — each gets its
    /// own fresh, empty `Vec` behind its own `Rc`.
    #[test]
    fn independent_default_contexts_do_not_share_a_sink() {
        let a = TransformContext::default();
        let b = TransformContext::default();
        a.warn(TransformWarning::DegenerateFit {
            transform: "Smooth".into(),
            group: None,
            reason: "zero x variance".into(),
        });
        assert_eq!(a.warnings.borrow().len(), 1);
        assert_eq!(b.warnings.borrow().len(), 0, "independent contexts must not share a sink");
    }

    fn make_batch_with_group_cols() -> (RecordBatch, Vec<String>) {
        use arrow::array::{Int64Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;
        let schema = Arc::new(Schema::new(vec![
            Field::new("g1", DataType::Utf8, true),
            Field::new("g2", DataType::Int64, true),
            Field::new("v", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "b"])),
                Arc::new(Int64Array::from(vec![1_i64, 2])),
                Arc::new(arrow::array::Float64Array::from(vec![1.0, 2.0])),
            ],
        )
        .unwrap();
        (batch, vec!["g1".to_string(), "g2".to_string()])
    }

    #[test]
    fn describe_group_row_formats_multi_column_key() {
        let (batch, group_cols) = make_batch_with_group_cols();
        let label = describe_group_row(&batch, &group_cols, 0).expect("row 0 must resolve");
        assert_eq!(label, "g1=a, g2=1");
        let label1 = describe_group_row(&batch, &group_cols, 1).expect("row 1 must resolve");
        assert_eq!(label1, "g1=b, g2=2");
    }

    #[test]
    fn describe_group_row_none_for_empty_group_cols() {
        let (batch, _) = make_batch_with_group_cols();
        assert_eq!(describe_group_row(&batch, &[], 0), None);
    }

    #[test]
    fn describe_group_keys_matches_describe_group_row() {
        use crate::transform::group_key::KeyValue;
        let (batch, group_cols) = make_batch_with_group_cols();
        let from_row = describe_group_row(&batch, &group_cols, 0).unwrap();
        let keys = vec![KeyValue::Str("a".into()), KeyValue::Int(1)];
        let from_keys = describe_group_keys(&group_cols, &keys);
        assert_eq!(from_row, from_keys);
    }
}
