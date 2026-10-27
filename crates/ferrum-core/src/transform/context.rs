//! TransformContext: render-time info passed to transforms that need viewport
//! or cross-transform output access.
//!
//! Used by:
//! - Raster (resolution="screen") and Violin (band_pixels) — both read
//!   `panel_pixel_size`, populated by
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
    /// `"fewer than 2 usable points"`, `"zero x variance"`). `panel` names
    /// the facet panel this fit ran in ([`facet_partition_label`]'s text),
    /// `None` for an unfaceted chart (byte-identical text) — see
    /// `run_transforms_with_facet`'s per-partition stamping.
    ///
    /// [`facet_partition_label`]: crate::render::prepare::facet_partition_label
    DegenerateFit {
        transform: String,
        group: Option<String>,
        reason: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        panel: Option<String>,
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
        /// The facet panel this fit ran in, `None` for an unfaceted chart
        /// (byte-identical text). See [`Self::DegenerateFit`]'s `panel` doc.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        panel: Option<String>,
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
    /// Every row's `groupby` key was null, so the request to split the batch
    /// into groups could not be honored (GH #197). This is the degraded arm of
    /// the adjudicated all-null policy: the transform still produces output,
    /// but not the grouped output the caller asked for. Pushed from exactly one
    /// place — `group_key::group_partition_multi` — so all eleven-plus grouped
    /// transforms report the same condition in the same words.
    ///
    /// `columns` lists the groupby column names as the user wrote them;
    /// `disposition` is
    /// [`crate::transform::group_key::NullKeyPolicy::all_null_disposition`]'s
    /// text for the policy that call site chose, so what the message says and
    /// what the transform did cannot drift apart.
    ///
    /// `rows` is the identity of the rows this call read
    /// ([`TransformContext::rows`]), not serialized into the Display text —
    /// it is read only by [`merge_null_key_reports`], together with
    /// `lowering`, so a composite's own multiple lowered steps over the
    /// SAME rows (`BoxStats` and `Outliers`, `Violin` and `BoxStats`) report
    /// once. `lowering` is the id of the statistics marker that opened the
    /// lowering this call's step belongs to
    /// ([`TransformContext::lowering`]), `None` for a user-written step or
    /// any step of a chain or branch with no marker — those never collapse
    /// against another step, whatever rows they share. `panels` names the
    /// facet panels a
    /// [`crate::transform::group_key::NullKeyPolicy::SkipWithWholeBatchFallback`]
    /// or [`crate::transform::group_key::NullKeyPolicy::FillFromWholeBatch`]
    /// step pooled, when that is fewer than every panel; empty (the serde
    /// default) means the whole frame, or every panel, and keeps the
    /// original whole-frame text.
    GroupKeyAllNull {
        transform: String,
        columns: String,
        disposition: String,
        #[serde(default)]
        rows: RowsOrigin,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        lowering: Option<crate::transform::run::StepId>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        panels: Vec<String>,
    },
    /// `rows` rows have a null value in the key `field` and belong to no
    /// panel or group, so the partition `scope` names dropped them.
    ///
    /// For [`NullKeyScope::Facet`], pushed once per facet field by
    /// `render::prepare::run_transforms_with_facet`, the one place rows are
    /// assigned to panels; rows of a composition operand whose frame lacks
    /// the field are not counted, since they repeat in every panel instead
    /// (`layout::FacetRepeat`). For [`NullKeyScope::Groupby`], pushed once
    /// per `groupby` field by
    /// [`crate::transform::group_key::warn_null_groupby_rows_dropped`], for
    /// every grouped transform whose null-key policy drops a row (`Skip`,
    /// `SkipWithWholeBatchFallback`, `SkipRefusingEmpty`), plus `LetterValue`'s
    /// own call.
    NullKeyRowsDropped {
        scope: NullKeyScope,
        field: String,
        rows: usize,
    },
    /// A confidence band could not be computed even though its fit LINE
    /// drew fine: zero or negative residual degrees of freedom, a singular
    /// design, or a bootstrap/sandwich variance that came back non-finite.
    /// The fit still draws; only the band is missing.
    /// `group` is `None` for a genuine whole-frame band, `Some`
    /// for one `groupby`/grouped-two-column-fit partition's. `reason` is
    /// the user-facing clause already naming the cause, and the point
    /// count when the band is undefined at only some grid points (e.g.
    /// "2 points leave no residual degrees of freedom for a 2-parameter
    /// fit; the band needs at least 3", "undefined at 3 of 50 grid
    /// points").
    ///
    /// Distinct from [`Self::DegenerateFit`]: that variant reports the
    /// whole fit (line and band both) lost; this one reports the line
    /// alone surviving a band the data cannot support.
    IntervalUndefined {
        transform: String,
        group: Option<String>,
        ci: f64,
        reason: String,
        /// The facet panel this fit ran in, `None` for an unfaceted chart
        /// (byte-identical text). See [`Self::DegenerateFit`]'s `panel` doc.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        panel: Option<String>,
    },
}

/// Why a confidence band is undefined because `n` points leave zero or
/// negative residual degrees of freedom for a `p`-parameter fit: the band
/// needs at least one residual degree of freedom, i.e. `p + 1` points.
/// Shared by every polynomial-family fit (`lm`, `log`, `sqrt`, `poly`) whose
/// band comes from a t-distribution over `n - p` residual degrees of
/// freedom, and by `Robust`'s own `n <= 2` case (worded identically to
/// `lm`'s, since both lose the same residual degree of freedom), so the one
/// reason text is worded identically everywhere it applies.
pub(crate) fn zero_residual_dof_reason(n: usize, p: usize) -> String {
    let point = if n == 1 { "point" } else { "points" };
    format!(
        "{n} {point} leave no residual degrees of freedom for a {p}-parameter fit; the band \
         needs at least {}",
        p + 1
    )
}

/// A band undefined at only some of a fit's grid points: `cause` is the
/// user-facing clause naming why (already stating the remedy, where one
/// exists), `undefined` of `n_grid` points. Shared by every fit whose band
/// fails per grid point rather than for the whole fit (`Logistic`'s Wald
/// variance, `Robust`'s sandwich variance), so the "at K of N grid points"/
/// "at every grid point" wording is one vocabulary.
pub(crate) fn partial_band_reason(cause: &str, undefined: usize, n_grid: usize) -> String {
    if undefined >= n_grid {
        format!("{cause} at every grid point")
    } else {
        format!("{cause} at {undefined} of {n_grid} grid points")
    }
}

/// Fold every `groupby`-scope null-key report into the one voice a facet or
/// refusal drain reports:
///
/// 1. Sum the [`TransformWarning::NullKeyRowsDropped`] counts one transform
///    STEP pushed once per facet panel into one warning per (step, field),
///    at the first one's position, so the count the user reads is the whole
///    frame's, as the facet scope's own count is. Two steps of the same kind
///    keep their own counts here, as they do on an unfaceted chart.
/// 2. Collapse the reports of several STEPS of ONE LOWERING that read the
///    SAME rows under the SAME key — one lowering's own multiple grouped
///    steps, e.g. `mark_boxplot`'s `BoxStats` and `Outliers`, or
///    `mark_violin`'s `Violin` and `BoxStats` — keeping the first and
///    dropping the later steps' identical reports in order: the first
///    [`TransformWarning::NullKeyRowsDropped`] per
///    `(lowering, rows, key, field)`, the first
///    [`TransformWarning::GroupKeyAllNull`] per
///    `(lowering, rows, columns, disposition, panels)`. `lowering` is
///    [`TransformContext::lowering`], the id of the statistics marker that
///    opened the step's lowering; a report whose `lowering` is `None` —
///    every user-written step, and any step of a chain or branch with no
///    marker — always passes through, never collapsing against another
///    report, however its `rows` and `key` compare: two `+` operands' own
///    statistics, two `Chart.layer` statistics sharing one branch, and two
///    hand-written named steps sharing a `groupby` over one unadvanced tail
///    are different lowerings (or none), not one lowering reporting twice.
///    `rows` is [`TransformContext::rows`], the identity of the rows each
///    call read — two steps of the SAME lowering producing the SAME
///    dropped-row count or the SAME all-null condition over DIFFERENT rows
///    are never collapsed either.
///
/// Every other warning passes through unchanged and in order. Called once
/// per drain, after every all-null settlement for that drain has replaced or
/// appended its own warnings, so stage 2 sees the final set.
pub(crate) fn merge_null_key_reports(warnings: Vec<TransformWarning>) -> Vec<TransformWarning> {
    let mut merged: Vec<TransformWarning> = Vec::with_capacity(warnings.len());
    let mut at: HashMap<(StepInstance, String), usize> = HashMap::new();
    for warning in warnings {
        if let TransformWarning::NullKeyRowsDropped {
            scope: NullKeyScope::Groupby { step, .. },
            field,
            rows,
        } = &warning
        {
            let key = (step.clone(), field.clone());
            if let Some(&i) = at.get(&key) {
                if let Some(TransformWarning::NullKeyRowsDropped { rows: total, .. }) =
                    merged.get_mut(i)
                {
                    *total += rows;
                }
                continue;
            }
            at.insert(key, merged.len());
        }
        merged.push(warning);
    }
    let mut seen_dropped: std::collections::HashSet<(crate::transform::run::StepId, RowsOrigin, Vec<String>, String)> =
        std::collections::HashSet::new();
    let mut seen_all_null: std::collections::HashSet<(crate::transform::run::StepId, RowsOrigin, String, String, Vec<String>)> =
        std::collections::HashSet::new();
    merged.retain(|warning| match warning {
        TransformWarning::NullKeyRowsDropped {
            scope: NullKeyScope::Groupby { lowering: None, .. },
            ..
        } => true,
        TransformWarning::NullKeyRowsDropped {
            scope: NullKeyScope::Groupby { rows, key, lowering: Some(lowering), .. },
            field,
            ..
        } => seen_dropped.insert((lowering.clone(), rows.clone(), key.clone(), field.clone())),
        TransformWarning::GroupKeyAllNull { lowering: None, .. } => true,
        TransformWarning::GroupKeyAllNull {
            columns, disposition, rows, panels, lowering: Some(lowering), ..
        } => seen_all_null.insert((lowering.clone(), rows.clone(), columns.clone(), disposition.clone(), panels.clone())),
        _ => true,
    });
    merged
}

/// Which partition dropped [`TransformWarning::NullKeyRowsDropped`]'s rows.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "partition", rename_all = "snake_case")]
pub(crate) enum NullKeyScope {
    /// The facet partition: a null facet value belongs to no panel.
    Facet,
    /// A transform's `groupby` partition: a null key belongs to no group.
    /// `transform` is the transform's Python-facing name; `step` is the
    /// pipeline step that dropped the rows, read by
    /// [`merge_null_key_reports`]'s per-step sum. `rows` is the identity of
    /// the rows the step read ([`TransformContext::rows`]), `lowering` is
    /// the id of the marker that opened the step's lowering
    /// ([`TransformContext::lowering`], `None` outside one), and `key` is
    /// the full `groupby` field list the step reported against — together
    /// (`lowering`, `rows`, `key`) the cross-step dedup key: a composite's
    /// own several grouped steps of ONE lowering, over the same rows and
    /// key, collapse to the first's report; a step with no `lowering` never
    /// collapses.
    Groupby {
        transform: String,
        step: StepInstance,
        #[serde(default)]
        rows: RowsOrigin,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        lowering: Option<crate::transform::run::StepId>,
        #[serde(default)]
        key: Vec<String>,
    },
}

/// Which written step of a pipeline run a transform call belongs to
/// ([`TransformContext::step`]), so what one step reports in every facet
/// panel is told apart from what another step of the same kind reports.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", content = "at", rename_all = "snake_case")]
pub(crate) enum StepInstance {
    /// A call outside a pipeline run.
    #[default]
    Unplaced,
    /// The step at this position of its chain; a branch's steps nest under
    /// the branch's own position.
    Chain(crate::transform::run::StepId),
    /// Every copy of one composition step, named by the step's identity
    /// ([`crate::transform::composed::ComposedSpec::group`]).
    Composed(String),
}

/// The identity of a set of rows one call reads: the step that produced
/// them, and which of that step's outputs — `None` for the step's own tail
/// or a framed branch's own rows, `Some(name)` for a named step's published
/// output, a secondary output's own key, or a named composition copy's
/// published output. Two calls sharing one `RowsOrigin` read the identical
/// rows; two calls over two DIFFERENT outputs of the SAME step (a named
/// step's tail and its own secondary output, say) do not, because `output`
/// tells them apart.
///
/// [`crate::transform::run::Lineage::origin`] is the only writer; every
/// reader (`TransformContext::rows`, [`NullKeyScope::Groupby::rows`],
/// [`TransformWarning::GroupKeyAllNull::rows`]) just carries the value
/// forward. `output` is omitted from the wire when absent, so this stays
/// invisible on every warning variant's own serialization footprint.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub(crate) struct RowsOrigin {
    pub(crate) step: StepInstance,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub(crate) output: Option<String>,
}

/// A facet partition in which a step's every `groupby` key was null, so its
/// null-key drop left no group: the step, and what the same drop gives the
/// whole frame ([`AllNullOutcome`]). Recorded by the shared partitioner
/// (`group_key::group_partition_multi`) and by
/// [`TransformContext::refuse_unless_partition`]; read back by
/// `render::prepare::run_transforms_with_facet`.
#[derive(Debug)]
pub(crate) struct AllNullPartition {
    pub(crate) step: StepInstance,
    pub(crate) outcome: AllNullOutcome,
}

/// Why one fit's rows could not be fit: the reason a degraded group's
/// warning gives, and the refusal the same rows raise as the whole frame
/// ([`TransformContext::degrade_group_or_refuse`]).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Unfittable {
    pub(crate) reason: String,
    pub(crate) refusal: String,
}

impl Unfittable {
    /// A failure whose refusal is `"{transform}: {reason}"`.
    pub(crate) fn new(transform: &str, reason: impl Into<String>) -> Self {
        let reason = reason.into();
        Self { refusal: format!("{transform}: {reason}"), reason }
    }
}

impl From<Unfittable> for pyo3::PyErr {
    fn from(failure: Unfittable) -> Self {
        pyo3::exceptions::PyValueError::new_err(failure.refusal)
    }
}

/// What a step whose every `groupby` key is null gives the whole frame.
#[derive(Debug)]
pub(crate) enum AllNullOutcome {
    /// The transform refuses (QQ, LetterValue): the panel draws empty, and
    /// the facet run raises the refusal when every panel was emptied.
    Refused(pyo3::PyErr),
    /// The grouped output is empty: the panel draws empty, and the facet
    /// run reports the warning once when every panel was emptied.
    Emptied(TransformWarning),
    /// The step fit the panel's whole batch instead (the whole-batch
    /// fallback): the panel draws that fit, and the facet run reports the
    /// warning once.
    Pooled(TransformWarning),
}

impl AllNullOutcome {
    /// Whether the partition draws nothing for the step.
    pub(crate) fn empties(&self) -> bool {
        !matches!(self, Self::Pooled(_))
    }
}

impl std::fmt::Display for TransformWarning {
    /// User-facing warning text forwarded to Python's ``warnings.warn`` by
    /// `render::binding::emit_warnings` (via `RenderWarning::Transform`'s own
    /// `Display` delegation). One sentence, no internal Rust names — mirrors
    /// `RenderWarning`'s own Display contract exactly.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransformWarning::DegenerateFit { transform, group, reason, panel } => {
                let in_panel = panel_clause(panel.as_deref());
                match group {
                    Some(g) => write!(
                        f,
                        "{transform}: could not fit group {g}{in_panel} ({reason}); it degraded \
                         to an empty/flat output instead of failing the whole chart"
                    ),
                    None => write!(
                        f,
                        "{transform}: could not fit the data{in_panel} ({reason}); it degraded \
                         to an empty/flat output instead of failing the whole chart"
                    ),
                }
            }
            TransformWarning::ParameterClamped { transform, group, param, requested, applied, panel } => {
                let in_panel = panel_clause(panel.as_deref());
                match group {
                    Some(g) => write!(
                        f,
                        "{transform}: {param}={requested} could not be honored for group \
                         {g}{in_panel}; {applied} was used instead"
                    ),
                    None => write!(
                        f,
                        "{transform}: {param}={requested} could not be honored{in_panel}; \
                         {applied} was used instead"
                    ),
                }
            }
            TransformWarning::FallbackApplied { transform, group, reason } => match group {
                Some(g) => write!(f, "{transform}: group {g} {reason}"),
                None => write!(f, "{transform}: {reason}"),
            },
            TransformWarning::GroupKeyAllNull { transform, columns, disposition, panels, .. } => {
                match panels.len() {
                    0 => write!(
                        f,
                        "{transform}: every row has a null in its groupby key ({columns}), so \
                         the data could not be split into groups; {disposition}"
                    ),
                    1 => write!(
                        f,
                        "{transform}: every row in facet panel {} has a null in its groupby key \
                         ({columns}), so that panel's rows could not be split into groups; in \
                         that panel, {disposition}",
                        panels[0]
                    ),
                    _ => write!(
                        f,
                        "{transform}: every row in facet panels {} has a null in its groupby key \
                         ({columns}), so those panels' rows could not be split into groups; in \
                         those panels, {disposition}",
                        panels.join("; ")
                    ),
                }
            }
            TransformWarning::NullKeyRowsDropped { scope, field, rows } => {
                let (noun, verb, belong, pronoun) =
                    if *rows == 1 { ("row", "has", "belongs", "it was") } else { ("rows", "have", "belong", "they were") };
                let (speaker, home) = match scope {
                    NullKeyScope::Facet => (format!("facet '{field}'"), "panel"),
                    NullKeyScope::Groupby { transform, .. } => (transform.clone(), "group"),
                };
                write!(
                    f,
                    "{speaker}: {rows} {noun} {verb} a null '{field}' and {belong} to no {home}, so \
                     {pronoun} dropped; fill or filter the nulls in '{field}'"
                )
            }
            TransformWarning::IntervalUndefined { transform, group, ci, reason, panel } => {
                let in_panel = panel_clause(panel.as_deref());
                match group {
                    Some(g) => write!(
                        f,
                        "{transform}: the ci={ci} band of group {g}{in_panel} is undefined \
                         ({reason}); the fit line is drawn without it"
                    ),
                    None => write!(
                        f,
                        "{transform}: the ci={ci} band{in_panel} is undefined ({reason}); the \
                         fit line is drawn without it"
                    ),
                }
            }
        }
    }
}

/// The " in facet panel f=F2" clause [`TransformWarning::DegenerateFit`],
/// [`TransformWarning::ParameterClamped`] and [`TransformWarning::IntervalUndefined`]
/// each splice into their Display text after the group clause; empty for an
/// unfaceted fit (`panel: None`), keeping that text byte-identical.
fn panel_clause(panel: Option<&str>) -> String {
    match panel {
        Some(p) => format!(" in facet panel {p}"),
        None => String::new(),
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
    /// this is the grid dimension. For violin, used to convert band pixels to
    /// data space via the value-axis scale.
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
    /// it is read (`raster.rs`, `violin.rs`).
    ///
    /// `None` on a bare `apply()` call with no render context (direct calls,
    /// unit tests, WASM/notebook direct usage) — each of the two readers
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
    /// The facet field names a partition run splits by (the wrap/column
    /// field, then the row field in grid mode); empty outside one. Set only
    /// by `render::prepare::run_transforms_with_facet`, beside
    /// `is_partition`. `Project` keeps these columns on every row whatever
    /// its list says, because the facet value is the panel's and belongs to
    /// every row in it (`transform::project`'s doc).
    pub facet_fields: Vec<String>,
    /// The degrade-warning sink (GH #162). `Rc<RefCell<_>>`, not a bare
    /// `RefCell<Vec<_>>`, because `apply_transforms_named` (`run.rs`) does
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
    /// The binned-color grouping of this pipeline run
    /// ([`crate::transform::color_bin_group`]): which steps group which
    /// columns by bin, and the edges. Set only by
    /// `render::prepare::run_transforms_with_facet`; `None` (the default)
    /// runs every step as written.
    pub(crate) color_bin_plan: Option<Rc<crate::transform::color_bin_group::ColorBinPlan>>,
    /// The binned columns the step this context is handed to groups by bin,
    /// set per step by the pipeline runner from [`Self::color_bin_plan`] and
    /// read by the shared partitioner
    /// ([`crate::transform::group_key::GroupKeyReader`]). `None` keys every
    /// groupby column by its raw values.
    pub(crate) color_bins: Option<Rc<crate::transform::color_bin_group::StepBins>>,
    /// The pipeline step this context is handed to, set per step by the
    /// pipeline runner; [`StepInstance::Unplaced`] outside a run.
    pub(crate) step: StepInstance,
    /// The identity of the ROWS this call reads, set beside [`Self::step`]
    /// by the pipeline runner from the input's own lineage (`run::Lineage::origin`).
    /// Two calls sharing one `rows` value read the identical rows: a
    /// composite's several lowered steps over one chain tail (`BoxStats`
    /// and `Outliers`, `Violin` and `BoxStats`) share it, so
    /// [`merge_null_key_reports`] can tell "one lowering reported twice"
    /// from "two lowerings over different rows happened to report the same
    /// count" — together with [`Self::lowering`], since rows identity alone
    /// cannot tell two DIFFERENT lowerings apart when neither advances the
    /// chain (two `+` operands over one frame, two `Chart.layer`
    /// statistics, two hand-written named steps). Defaults to the
    /// [`RowsOrigin`] of [`StepInstance::Unplaced`] outside a run.
    pub(crate) rows: RowsOrigin,
    /// The id of the statistics marker that opened the LOWERING this call's
    /// step belongs to, set beside [`Self::rows`] by the pipeline runner:
    /// `Run::chain` records the id of the marker it last met, and
    /// `Run::step` carries that id into this field only for a step the
    /// marker actually lowers (the same `lowered` flag `Self::step` already
    /// gates on). `None` for a user-written step, any step of a chain or
    /// branch with no marker, and a composition copy
    /// ([`crate::transform::run::Run::run_copy`]) — these never collapse
    /// against another step in [`merge_null_key_reports`], whatever rows
    /// and key they share.
    pub(crate) lowering: Option<crate::transform::run::StepId>,
    /// The facet partitions in which a step's every `groupby` key was null,
    /// shared across every clone of a run's context as [`Self::warnings`]
    /// is, and read back by `render::prepare::run_transforms_with_facet`.
    pub(crate) all_null_partitions: Rc<RefCell<Vec<AllNullPartition>>>,
}

impl TransformContext {
    /// Record a degrade at the current call site. Cheap and infallible —
    /// pushes onto the shared sink (see [`TransformContext::warnings`]'s doc
    /// for why `Rc<RefCell<_>>` is load-bearing here).
    pub(crate) fn warn(&self, warning: TransformWarning) {
        self.warnings.borrow_mut().push(warning);
    }

    /// Raise `refusal`, the refusal a `groupby` null-key drop that left
    /// nothing to draw gives the whole frame. Inside a facet partition
    /// ([`Self::is_partition`]) the call instead records that the drop
    /// emptied this panel and returns `Ok`, so the step returns its empty
    /// output: the panel draws empty, and the facet run raises `refusal`
    /// itself when the step's drop emptied every panel.
    pub(crate) fn refuse_unless_partition(&self, refusal: pyo3::PyErr) -> pyo3::PyResult<()> {
        if !self.is_partition {
            return Err(refusal);
        }
        self.record_all_null(AllNullOutcome::Refused(refusal));
        Ok(())
    }

    /// One fit's rows cannot be fit ([`Unfittable`]). The whole frame
    /// (`group` `None`) refuses with the failure's refusal, as an ungrouped
    /// fit always has; a `groupby` group degrades as `Smooth`'s and
    /// `Robust`'s groups do: a [`TransformWarning::DegenerateFit`] names the
    /// group and the reason, and the caller emits that group's empty
    /// (all-NaN) fit, so the other groups still draw.
    pub(crate) fn degrade_group_or_refuse(
        &self,
        transform: &str,
        group: Option<&str>,
        failure: Unfittable,
    ) -> pyo3::PyResult<()> {
        let Some(group) = group else { return Err(failure.into()) };
        self.warn(TransformWarning::DegenerateFit {
            transform: transform.to_string(),
            group: Some(group.to_string()),
            reason: failure.reason,
            panel: None,
        });
        Ok(())
    }

    /// Record that this facet partition's every `groupby` key was null for
    /// the current step, and what the step gives in that case.
    pub(crate) fn record_all_null(&self, outcome: AllNullOutcome) {
        self.all_null_partitions.borrow_mut().push(AllNullPartition { step: self.step.clone(), outcome });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_variants() -> Vec<TransformWarning> {
        vec![
            TransformWarning::NullKeyRowsDropped {
                scope: NullKeyScope::Facet,
                field: "c".into(),
                rows: 1,
            },
            TransformWarning::NullKeyRowsDropped {
                scope: NullKeyScope::Groupby {
                    transform: "QQ".into(),
                    step: StepInstance::Chain(vec![1]),
                    rows: RowsOrigin { step: StepInstance::Chain(vec![0]), output: None },
                    lowering: Some(vec![0]),
                    key: vec!["h".into()],
                },
                field: "h".into(),
                rows: 3,
            },
            TransformWarning::DegenerateFit {
                transform: "Smooth".into(),
                group: Some("g=a".into()),
                reason: "fewer than 2 usable points".into(),
                panel: None,
            },
            TransformWarning::DegenerateFit {
                transform: "Smooth".into(),
                group: None,
                reason: "zero x variance".into(),
                panel: Some("f=F2".into()),
            },
            TransformWarning::ParameterClamped {
                transform: "Smooth".into(),
                group: None,
                param: "k".into(),
                requested: "200".into(),
                applied: "6".into(),
                panel: None,
            },
            TransformWarning::ParameterClamped {
                transform: "LetterValue".into(),
                group: Some("g=a".into()),
                param: "k_depth".into(),
                requested: "14".into(),
                applied: "12".into(),
                panel: Some("f=F2".into()),
            },
            TransformWarning::FallbackApplied {
                transform: "Impute".into(),
                group: Some("g=b".into()),
                // Not "group has no ..." — `Display`'s `Some(g)` arm already
                // writes "group {g} " ahead of this text.
                reason: "has no non-null values; the whole column's mean was used instead"
                    .into(),
            },
            TransformWarning::FallbackApplied {
                transform: "Impute".into(),
                group: None,
                reason: "row's groupby key is null; the whole column's mean was used instead"
                    .into(),
            },
            TransformWarning::GroupKeyAllNull {
                transform: "Aggregate".into(),
                columns: "g".into(),
                disposition: "the grouped output is empty".into(),
                rows: RowsOrigin::default(),
                lowering: None,
                panels: Vec::new(),
            },
            TransformWarning::GroupKeyAllNull {
                transform: "Smooth".into(),
                columns: "h".into(),
                disposition: "the whole batch was used as one ungrouped computation instead".into(),
                rows: RowsOrigin { step: StepInstance::Chain(vec![0]), output: None },
                lowering: Some(vec![0]),
                panels: vec!["f=F2".into()],
            },
            TransformWarning::GroupKeyAllNull {
                transform: "Smooth".into(),
                columns: "h".into(),
                disposition: "the whole batch was used as one ungrouped computation instead".into(),
                rows: RowsOrigin { step: StepInstance::Chain(vec![0]), output: Some("fitted".into()) },
                lowering: Some(vec![0]),
                panels: vec!["f=F2".into(), "f=F3".into()],
            },
            TransformWarning::IntervalUndefined {
                transform: "Smooth".into(),
                group: Some("g=a".into()),
                ci: 0.95,
                reason: "2 points leave no residual degrees of freedom for a 2-parameter fit; \
                         the band needs at least 3"
                    .into(),
                panel: None,
            },
            TransformWarning::IntervalUndefined {
                transform: "Smooth".into(),
                group: None,
                ci: 0.95,
                reason: "undefined at 3 of 50 grid points".into(),
                panel: Some("f=F2, r=R1".into()),
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
        let leak_tokens =
            ["DegenerateFit", "ParameterClamped", "FallbackApplied", "GroupKeyAllNull", "NullKeyRowsDropped", "NullKeyScope", "IntervalUndefined"];
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
            panel: None,
        };
        assert!(with_group.to_string().contains("g=a"));

        let without_group = TransformWarning::DegenerateFit {
            transform: "Smooth".into(),
            group: None,
            reason: "fewer than 2 usable points".into(),
            panel: None,
        };
        assert!(!without_group.to_string().contains("group"));
    }

    /// Wave-1 close remediation (S1 finding): `FallbackApplied`'s `Some(g)`
    /// Display arm writes `"group {g} {reason}"` — a `reason` clause that
    /// ALSO starts with "the group" renders "group g=b the group has no ...",
    /// doubling the word. The only literal `FallbackApplied` caller
    /// (`impute.rs`) must phrase its group-scoped reasons to start with the
    /// condition itself ("has no ...", not "the group has no ..."), and this
    /// test pins the general contract: "group" must appear exactly once.
    #[test]
    fn fallback_applied_display_does_not_double_the_word_group() {
        let w = TransformWarning::FallbackApplied {
            transform: "Impute".into(),
            group: Some("g=b".into()),
            reason: "has no non-null/non-NaN values; the whole column's statistic was used \
                     instead"
                .into(),
        };
        let text = w.to_string();
        assert_eq!(
            text.matches("group").count(),
            1,
            "\"group\" must appear exactly once in: {text:?}"
        );
    }

    /// The keystone gotcha the design doc calls out: `apply_transforms_named`
    /// (`run.rs`) does `let mut step_ctx = ctx.clone();` once per transform.
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
            panel: None,
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
            panel: None,
        });
        assert_eq!(a.warnings.borrow().len(), 1);
        assert_eq!(b.warnings.borrow().len(), 0, "independent contexts must not share a sink");
    }

    /// One step's per-panel null-key counts sum into one warning; a second
    /// step of the same kind, over rows of its own, keeps its own count,
    /// beside the first. The rejected alternative keys the sum by the
    /// transform's name and folds both steps into one warning.
    #[test]
    fn null_key_counts_merge_per_step_never_per_transform_name() {
        let dropped = |step: usize, rows: usize| TransformWarning::NullKeyRowsDropped {
            scope: NullKeyScope::Groupby {
                transform: "QQ".into(),
                step: StepInstance::Chain(vec![step]),
                rows: RowsOrigin { step: StepInstance::Chain(vec![step]), output: None },
                lowering: None,
                key: vec!["h".into()],
            },
            field: "h".into(),
            rows,
        };
        let merged = merge_null_key_reports(vec![dropped(1, 2), dropped(2, 3), dropped(1, 4), dropped(2, 5)]);
        assert_eq!(merged, [dropped(1, 6), dropped(2, 8)]);
    }

    /// Two steps of ONE lowering (the same marker's id) that read the SAME
    /// rows under the SAME key — one lowering's own several grouped steps,
    /// e.g. `BoxStats` and `Outliers` — report once, keeping the first
    /// (even though its own `transform` name and per-panel step id differ
    /// from the second's).
    #[test]
    fn merge_collapses_two_steps_reporting_the_same_rows_and_key() {
        let rows = RowsOrigin { step: StepInstance::Chain(vec![0]), output: None };
        let box_stats = TransformWarning::NullKeyRowsDropped {
            scope: NullKeyScope::Groupby {
                transform: "BoxStats".into(),
                step: StepInstance::Chain(vec![1]),
                rows: rows.clone(),
                lowering: Some(vec![0]),
                key: vec!["h".into()],
            },
            field: "h".into(),
            rows: 2,
        };
        let outliers = TransformWarning::NullKeyRowsDropped {
            scope: NullKeyScope::Groupby {
                transform: "Outliers".into(),
                step: StepInstance::Chain(vec![2]),
                rows,
                lowering: Some(vec![0]),
                key: vec!["h".into()],
            },
            field: "h".into(),
            rows: 2,
        };
        let merged = merge_null_key_reports(vec![box_stats.clone(), outliers]);
        assert_eq!(merged, [box_stats], "the first step's report wins, not a sum of both");
    }

    /// Two steps reporting the identical count and field over DIFFERENT
    /// rows are two separate facts, not a duplicate, even sharing one
    /// lowering: dedup reads `rows` too, not the lowering or the count or
    /// the field alone.
    #[test]
    fn merge_keeps_two_steps_reporting_different_rows_apart() {
        let dropped = |step: usize| TransformWarning::NullKeyRowsDropped {
            scope: NullKeyScope::Groupby {
                transform: "QQ".into(),
                step: StepInstance::Chain(vec![step]),
                rows: RowsOrigin { step: StepInstance::Chain(vec![step]), output: None },
                lowering: Some(vec![0]),
                key: vec!["h".into()],
            },
            field: "h".into(),
            rows: 2,
        };
        let merged = merge_null_key_reports(vec![dropped(1), dropped(2)]);
        assert_eq!(merged, [dropped(1), dropped(2)]);
    }

    /// The same cross-step collapse applies to [`TransformWarning::GroupKeyAllNull`]:
    /// two steps of one lowering, over the same rows, that both pooled the
    /// same columns under the same disposition and the same (empty, i.e.
    /// whole-frame) panel list report once.
    #[test]
    fn merge_collapses_two_all_null_reports_over_the_same_rows() {
        let all_null = |transform: &str| TransformWarning::GroupKeyAllNull {
            transform: transform.into(),
            columns: "h".into(),
            disposition: "the whole batch was used as one ungrouped computation instead".into(),
            rows: RowsOrigin { step: StepInstance::Chain(vec![0]), output: None },
            lowering: Some(vec![0]),
            panels: Vec::new(),
        };
        let merged = merge_null_key_reports(vec![all_null("Smooth"), all_null("Robust")]);
        assert_eq!(merged, [all_null("Smooth")]);
    }

    /// The dedup key is `(lowering, rows, key, field)`, not `(rows, key,
    /// field)` alone: two reports of the SAME lowering over the SAME rows
    /// collapse; two DIFFERENT lowerings over the SAME rows — the shape a
    /// `+` operand's own branch or a `Chart.layer` statistic beside another
    /// takes, since neither advances the chain — never do; and a report
    /// with NO lowering (a user-written step, or any step of a chain or
    /// branch with no marker) never collapses either, whatever rows and key
    /// it shares with another report.
    #[test]
    fn merge_needs_lowering_and_rows_together() {
        let rows = RowsOrigin { step: StepInstance::Chain(vec![0]), output: None };
        let dropped = |transform: &str, step: usize, lowering: Option<crate::transform::run::StepId>| {
            TransformWarning::NullKeyRowsDropped {
                scope: NullKeyScope::Groupby {
                    transform: transform.into(),
                    step: StepInstance::Chain(vec![step]),
                    rows: rows.clone(),
                    lowering,
                    key: vec!["h".into()],
                },
                field: "h".into(),
                rows: 2,
            }
        };

        let a = dropped("BoxStats", 1, Some(vec![0]));
        let b = dropped("Outliers", 2, Some(vec![0]));
        assert_eq!(merge_null_key_reports(vec![a.clone(), b]), [a], "same lowering, same rows: one");

        let c = dropped("Bin", 1, Some(vec![0]));
        let d = dropped("Kde", 2, Some(vec![1]));
        assert_eq!(
            merge_null_key_reports(vec![c.clone(), d.clone()]),
            [c, d],
            "two lowerings over the same rows: two"
        );

        let e = dropped("QQ", 1, None);
        let f = dropped("QQ", 2, None);
        assert_eq!(merge_null_key_reports(vec![e.clone(), f.clone()]), [e, f], "no lowering: never collapsed");
    }

    /// A whole-frame call raises the refusal; a facet partition records the
    /// step whose drop emptied it and carries on.
    #[test]
    fn refuse_unless_partition_refuses_the_whole_frame_and_records_a_partition() {
        pyo3::Python::initialize();
        let refusal = || pyo3::exceptions::PyValueError::new_err("stat_qq: empty input after filter");
        let whole = TransformContext::default();
        assert!(whole.refuse_unless_partition(refusal()).is_err());
        assert!(whole.all_null_partitions.borrow().is_empty());

        let panel = TransformContext { is_partition: true, step: StepInstance::Chain(vec![1]), ..TransformContext::default() };
        assert!(panel.refuse_unless_partition(refusal()).is_ok());
        let recorded = panel.all_null_partitions.borrow();
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].step, StepInstance::Chain(vec![1]));
        match &recorded[0].outcome {
            AllNullOutcome::Refused(refusal) => {
                assert_eq!(refusal.to_string(), "ValueError: stat_qq: empty input after filter");
            }
            other => panic!("expected the refusal, got {other:?}"),
        }
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
