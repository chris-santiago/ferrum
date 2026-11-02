//! Fit-family transform: `Smooth` — LOESS, polynomial/linear (OLS), log-x,
//! sqrt-x, GLM (Gaussian family, Identity link), and GAM (penalized cubic
//! B-spline) regression lines with optional confidence bands, evaluated
//! over an evenly-spaced x grid. See [`PySmooth`] for the full parameter
//! contract.
//!
//! The heavier fitting cores live beside this file rather than inside it:
//! [`crate::transform::glm`] owns IRLS and the Wald band,
//! [`crate::transform::gam`] owns the penalized-spline basis, penalty, and
//! GCV λ selection. This module owns method routing, the evaluation grid,
//! the CI-band assembly, and the grouped/partitioned dispositions below.
//!
//! ## Degenerate-input policy (batch-D T9 round 3)
//!
//! Two different kinds of "this fit cannot be computed" are handled
//! differently on purpose, and the distinction is deliberate rather than an
//! oversight:
//!
//! - **Parameter absurdity, on a genuine whole-frame call.** The caller
//!   asked for a polynomial `degree` the input cannot determine —
//!   `degree >=` the number of usable (non-null, post-`x_bins`-aggregation) points,
//!   i.e. the Vandermonde design is rank-deficient (`poly_ols`'s `n < p`
//!   case). This refuses with a typed `PyValueError` naming the point count
//!   needed — the same refusal class as `degree=0` — instead of silently
//!   emitting an all-NaN grid (closes the spec review's finding 2:
//!   `degree=20` on a 20-row frame used to render zero polylines with no
//!   diagnostic at all). It fires only when the batch is genuinely the
//!   whole frame: `groupby` is empty AND the render pipeline has not
//!   already partitioned this batch out of a larger one.
//! - **Data degeneracy, or a slice of a larger frame, degrades instead.**
//!   Both of the following degrade to an all-NaN row range rather than
//!   failing the whole render:
//!   - A per-group rank-deficient design under `groupby` (T6's contract —
//!     zero-variance x, non-convergent IRLS, too few points after that
//!     group's own null/NaN drop). A grouped batch keeps succeeding even
//!     when one group can't be fit, rather than one bad group failing the
//!     whole chart.
//!   - An UNGROUPED batch that is itself one partition of a larger frame —
//!     e.g. one facet panel. Round 2 treated `groupby.is_empty()` as proof
//!     of "whole frame", but `render::prepare::run_transforms_with_facet`
//!     partitions the input by the facet field(s) BEFORE running
//!     transforms, so every facet panel is an ungrouped `apply()` call on a
//!     data subset: `fm.lmplot(order=3, col='sp')` on a sparse level raised,
//!     while the identical data and degree under `hue='sp'` rendered (same
//!     "not enough data for this fit" condition, two opposite dispositions
//!     depending only on which channel partitioned it). Round 3 closes this
//!     by having the production pipeline mark a batch as a partition via
//!     [`TransformContext::is_partition`](crate::transform::context::TransformContext::is_partition)
//!     — set only at `render::prepare::run_transforms_with_facet`'s facet
//!     construction site — and gating the refusal on `!ctx.is_partition` in
//!     addition to `groupby.is_empty()`.
//!
//! **`gam` contributes no third case (batch-D T10).** The whole-frame
//! refusal above exists for exactly one shape: the CALLER's `degree`
//! parameter is what makes the design undeterminable, so a silent empty
//! render would hide the caller's own mistake. `gam` exposes no such
//! parameter — its basis dimension is derived from the data
//! (`clamp(distinct_x, 4, 10)`) and its smoothness from a GCV-selected
//! penalty — so every way a `gam` fit can fail is data degeneracy, and
//! data degeneracy degrades here. A whole-frame `gam` on fewer than 4
//! distinct x values therefore emits an all-NaN grid exactly like `lm` on
//! a zero-variance x, rather than raising. Adding a refusal would mean
//! refusing charts that `method="loess"` renders (blank) on the same data,
//! for no caller-actionable reason.
//!
//!   `apply_one_group`'s `only_indices` parameter is `None` for every
//!   ungrouped call, whether or not that call is itself a facet panel;
//!   `is_partition` is the separate signal that disambiguates the two.
//!   `apply()` (no context — direct calls, unit tests, WASM/notebook direct
//!   usage) always treats its input as the whole frame, matching this
//!   module's behavior before this round. `apply_grouped` passes
//!   `Some(ixs)` to each per-group call, but its own empty-partition
//!   fallback (every group key is null) passes `None` on the full batch, so
//!   that fallback threads `is_partition` through too rather than being
//!   unconditionally treated as whole-frame.
//!
//! ## Output schema is spec-determined, not data-determined (round 4)
//!
//! Every degenerate branch above (a rank-deficient design, zero-variance x,
//! non-convergent IRLS, or fewer than 2 usable points) emits its all-NaN
//! output via [`build_degenerate_smooth_batch`], which appends the
//! `_metrics_text`/`_metrics_y` columns — all-null — whenever
//! `spec.inject_metrics` is set, exactly like a well-supported sibling
//! fit's real (non-null-anchor) output. Before this round, every
//! degenerate branch passed `metrics: None` unconditionally, so a
//! `Smooth`'s output schema silently depended on whether THIS PARTICULAR
//! slice's fit happened to degenerate: one `groupby` group or facet panel
//! could carry two more columns than a degenerate sibling. That divergence
//! is what `apply_grouped`'s by-name reconciliation (below) and
//! `render::prepare::concat_batches_by_name` exist to paper over across
//! groups and facet panels respectively; with the schema fixed by the
//! spec instead of the data, both of those are now a belt-and-braces
//! safety net for any *other* named-output divergence, not the primary
//! defense against this one.

// Promoted from a single `transform/smooth.rs` past the CLAUDE.md
// test-vs-source split trigger (the test module exceeded the source
// module: 2361 test lines against ~1806 source lines) into this directory,
// on the `mod.rs` + topic files + one shared `tests.rs` shape established
// by `render/scale_resolve/` and `transform/stats/`. `fits.rs` holds the
// numeric fit routines and the batch-assembly helpers genuinely shared
// among them; `binding.rs` the self-contained PyO3 unit (`PySmooth`, its
// `#[pymethods] impl`, `METHODS`); `tests.rs` the relocated test suite.
// `apply_one_group`/`apply_grouped`/`pre_aggregate_xy` stay here — they are
// the orchestration layer both siblings are called from — alongside
// `extract_xy`/`all_nan_output` (input extraction and the too-sparse-to-fit
// early return): neither is a fit function nor shared by one, and
// `apply_one_group` is their sole caller crate-wide, so they belong beside
// their caller rather than in `fits.rs`.
mod binding;
mod fits;
#[cfg(test)]
mod tests;

pub(crate) use binding::{PySmooth, METHODS};

use arrow::array::{Array, Float64Array, RecordBatch};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};

use crate::transform::color_bin_group::SlotSpace;
use crate::transform::context::{describe_group_keys, TransformContext, TransformWarning};
use crate::transform::core::{SpaceSlot, TransformSpec};
use crate::transform::gam;
use crate::transform::group_key::{
    deserialize_groupby_vec, group_partition_multi, pooled_group_key, stack_grouped_batches,
    NullKeyPolicy,
    KeyValue,
};
use crate::transform::log_space::run_fit_in_log_space;
use crate::transform::numeric_util::coerce_to_float64;
use crate::transform::residuals;
use fits::{
    build_degenerate_smooth_batch, build_predictor, build_residuals_batch, gam_fit,
    gam_residuals_fit, glm_fit, lm_fit, loess_fit, log_x_fit, poly_fit, report_gam_outcome,
    residuals_fit, sqrt_x_fit,
};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SmoothMethod {
    /// Global ordinary least squares (degree-1 polynomial).
    Lm,
    /// Locally-weighted polynomial regression.
    Loess,
    /// Global polynomial regression of the specified degree (2 = quadratic, 3 = cubic).
    Poly { degree: u8 },
    /// Log-x linear fit: y ~ a + b·ln(x).  Requires x > 0.
    LogX,
    /// Sqrt-x linear fit: y ~ a + b·√x.  Requires x ≥ 0.
    SqrtX,
    /// Generalized Linear Model (Gaussian family, Identity link — the
    /// standalone `Glm` transform's own default family/link) fit via IRLS.
    /// Batch-D T9 (spec §4.6): delegates to `glm::fit_irls` so
    /// `mark_smooth(method="glm")` shares one fitting implementation with
    /// the `Glm` transform, rather than re-deriving IRLS inline or aliasing
    /// to `Lm` — Gaussian/Identity IRLS converges to the same point
    /// estimate as OLS, so an alias would look identical here but is the
    /// wrong implementation choice (it silently stops being correct the
    /// moment family/link is ever exposed, and its CI band would not match
    /// the real GLM's Wald band, which `glm_fit` reproduces exactly).
    Glm,
    /// Generalized Additive Model: a penalized cubic B-spline smoother
    /// (P-spline) with the smoothing parameter chosen by GCV. Batch-D T10
    /// (spec §4.6): the numerical core lives in
    /// [`crate::transform::gam`], which documents the basis, knot
    /// placement, penalty, and λ-selection defaults. Deterministic — it
    /// uses no randomness, so unlike `Loess` its band does not depend on
    /// `seed`.
    Gam,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum SmoothOutput {
    Fitted,
    Residuals,
}

pub(crate) fn default_smooth_output() -> SmoothOutput { SmoothOutput::Fitted }

/// LOESS span used when the caller does not ask for one. Non-loess methods
/// store this same value as an inert placeholder (nothing reads it), which is
/// why an EXPLICIT `bandwidth=` equal to it is tolerated on those methods
/// while any other explicit value refuses — see `PySmooth::new`.
pub(crate) const DEFAULT_LOESS_BANDWIDTH: f64 = 0.75;

fn default_gam_k() -> usize { gam::DEFAULT_BASIS_DIM }

fn is_default_gam_k(k: &usize) -> bool { *k == gam::DEFAULT_BASIS_DIM }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct SmoothSpec {
    pub x: String,
    pub y: String,
    pub method: SmoothMethod,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub ci: Option<f64>,
    pub bandwidth: f64,
    pub degree: u8,
    /// Requested spline basis dimension for `method="gam"` — the parameter
    /// mgcv calls `k`. Defaults to [`gam::DEFAULT_BASIS_DIM`]
    /// and is omitted from the wire at that default, so every spec written
    /// before this field existed round-trips byte-identically and every
    /// non-gam method serializes exactly as it did. `gam::fit` clamps it into
    /// `4..=distinct_x`; `PySmooth::new` refuses a value below 4 and refuses
    /// the parameter entirely on a method that has no basis to size.
    #[serde(default = "default_gam_k", skip_serializing_if = "is_default_gam_k")]
    pub k: usize,
    pub n: usize,
    #[serde(default)]
    pub seed: u64,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub x_bins: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub x_estimator: Option<crate::transform::aggregate::AggFn>,
    #[serde(default = "default_smooth_output")]
    pub output: SmoothOutput,
    /// Schwabish SB-followup (2026-05-12): when ``output == Residuals`` and
    /// this is ``true``, append a nullable ``_ref_zero`` Float64 column to
    /// the residuals batch. One non-null entry (value ``0.0``) on the first
    /// row, ``null`` on the rest — so a downstream ``mark_rule(y="_ref_zero")``
    /// renders exactly one horizontal reference line at y=0 without needing
    /// any extra Python-side data manipulation. No-op for ``Fitted`` output.
    #[serde(default)]
    pub inject_zero_ref: bool,
    /// Schwabish SB-followup (2026-05-12): when ``true``, append nullable
    /// ``_metrics_text`` (Utf8) and ``_metrics_y`` (Float64) columns with
    /// a single non-null row holding ``"R² {r2}\nRMSE {rmse}\nMAE {mae}"``
    /// and the anchor y position. Allowed on BOTH outputs (Fitted and
    /// Residuals); the VISUAL intent is identical — anchor the text at
    /// the rightmost finite x of the output and at the highest finite y
    /// of the respective output column, so the text sits in the
    /// top-right corner of the rendered geometry. The y differs by
    /// path only because the y axis itself differs between the two
    /// output modes:
    /// - ``Residuals``: anchor at the rightmost finite ``x`` input row;
    ///   ``_metrics_y`` at the max finite residual (top of the residual
    ///   scatter, near y=0).
    /// - ``Fitted``: anchor at the rightmost finite grid x; ``_metrics_y``
    ///   at the max finite fitted y (top of the fit curve). Metrics are
    ///   computed against the RAW (pre-aggregation) input regardless of
    ///   ``x_bins``.
    /// Designed for a same-data ``mark_text`` overlay reading both columns.
    #[serde(default)]
    pub inject_metrics: bool,
    /// When non-empty, partition input by these columns and emit per-(x, group)
    /// rows. Output schema gains one column per groupby field after the core output.
    /// Wire accepts: `null` → `[]`, `"col"` → `["col"]`, `["a","b"]` → `["a","b"]`.
    #[serde(
        skip_serializing_if = "Vec::is_empty",
        default,
        deserialize_with = "deserialize_groupby_vec"
    )]
    pub groupby: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
    /// WI-7 (2026-05-15): when set, evaluate the fit line over this x range
    /// instead of ``[xs.min(), xs.max()]``. Enables ``lmplot(truncate=False)``
    /// to extend the fit line to the axis domain boundary. Clipped at this
    /// boundary — never extrapolates past it.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub x_range: Option<[f64; 2]>,
    /// A symmetric proportional margin (`(hi - lo) * x_range_pad` on both
    /// ends) applied to the evaluation range `apply_one_group` resolves
    /// ([`crate::transform::numeric_util::evaluation_range`]): `x_range`
    /// when the caller wrote one, else the whole `batch`'s own `x` extent
    /// — "extend past the data's own range" — either way padded in
    /// whichever units the fit actually runs in (data units on a linear
    /// x, log units once a demanded log x has mapped `x_range` and the
    /// batch there). `None` pads nothing. Unlike `x_range`, `x_range_pad`
    /// alone (no `x_range`) is not a no-op: it is the whole request
    /// `truncate=False` makes when the caller gave no explicit range.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub x_range_pad: Option<f64>,
}

pub(crate) fn apply(spec: &SmoothSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    apply_with_context(spec, batch, &TransformContext::default())
}

/// Batch-D T9 round 3: the context-aware entry point, wired into
/// `TransformSpec::apply_with_context`'s dispatch table (`transform/core.rs`)
/// so the production render pipeline's `TransformContext.is_partition` (see
/// its doc) actually reaches the rank-deficiency refusal below. `apply()`
/// forwards here with a default context (`is_partition: false`), preserving
/// its pre-round-3 whole-frame behavior for direct calls, unit tests, and
/// WASM/notebook usage.
///
/// When a drawn log axis demands this fit on its `X` slot, its `Y` slot, or
/// both independently (an independent-y layer can resolve a different, or
/// no, `LogScale` from the primary's), the fit runs in that space instead
/// ([`apply_in_log_space`]); otherwise it runs as written
/// ([`apply_in_data_space`]).
pub(crate) fn apply_with_context(
    spec: &SmoothSpec,
    batch: &RecordBatch,
    ctx: &TransformContext,
) -> PyResult<RecordBatch> {
    let x_slot = ctx.log_space.as_deref().and_then(|s| s.of(SpaceSlot::X));
    let y_slot = ctx.log_space.as_deref().and_then(|s| s.of(SpaceSlot::Y));
    if x_slot.is_none() && y_slot.is_none() {
        return apply_in_data_space(spec, batch, ctx);
    }
    apply_in_log_space(spec, batch, ctx, x_slot, y_slot)
}

/// [`apply_with_context`]'s ordinary (data-space) body: every `Smooth`,
/// demanded or not, runs through this once its `x`/`y` columns (and
/// `x_range`, when the caller wrote one) are already in its own fit
/// space — data units when not demanded, the mapped batch's log units
/// under [`apply_in_log_space`]. `apply_one_group` resolves the actual
/// evaluation range (`spec.x_range`, or the whole batch's own `x` extent,
/// padded by `x_range_pad`) via
/// [`crate::transform::numeric_util::evaluation_range`], so the pad is
/// applied once, in whichever space this body is already running in,
/// identically on both routes.
fn apply_in_data_space(spec: &SmoothSpec, batch: &RecordBatch, ctx: &TransformContext) -> PyResult<RecordBatch> {
    if !spec.groupby.is_empty() {
        return apply_grouped(spec, batch, &spec.groupby, ctx.is_partition, ctx);
    }
    apply_one_group(spec, batch, None, ctx.is_partition, ctx, None)
}

/// `method='sqrt'` on a demanded log x, voiced by both the primary refusal
/// ([`crate::render::prepare::refuse_sqrt_on_log_x`], through
/// `RenderError::SmoothSqrtOnLogX`'s `Display`) and this transform's own
/// backstop ([`apply_in_log_space`]), so the two can never drift apart.
pub(crate) const SQRT_ON_LOG_X_REASON: &str =
    "method='sqrt' fits y on sqrt(x), while the log x axis fits in log(x); use method='lm' or 'loess', or drop the log scale";

/// `apply_with_context` under a drawn log axis's demand on the `x` slot,
/// the `y` slot, or both independently: a caller-written `x_range` is
/// mapped into the demanded `x` axis's units (unpadded —
/// [`apply_in_data_space`]'s `apply_one_group` pads it, in whichever
/// space this body runs in) before the ordinary body runs over the mapped
/// columns, so the grid, `x_bins`, and every method's fit and CI compute
/// in log units and map back ([`run_fit_in_log_space`]). `method="log"` is
/// a predictor transform that is linear in log x (`ln`/`log10` only
/// rescale the coefficient), so under a demanded x it runs the `Lm` body
/// directly on the already-mapped x rather than taking `ln` of log units a
/// second time. `method="sqrt"` composes with no scale transform and is
/// refused outright: a caller reaching this function with it demanded is a
/// bug in the prepare-time gate (`render::prepare::refuse_sqrt_on_log_x`),
/// which is the one place this refusal is voiced in the layer entry's own
/// text — this is the backstop, not the primary refusal.
fn apply_in_log_space(
    spec: &SmoothSpec,
    batch: &RecordBatch,
    ctx: &TransformContext,
    x_slot: Option<&SlotSpace>,
    y_slot: Option<&SlotSpace>,
) -> PyResult<RecordBatch> {
    if x_slot.is_some() && matches!(spec.method, SmoothMethod::SqrtX) {
        return Err(PyValueError::new_err(format!("stat_smooth: {SQRT_ON_LOG_X_REASON}")));
    }
    let method = if x_slot.is_some() && matches!(spec.method, SmoothMethod::LogX) { SmoothMethod::Lm } else { spec.method };
    // `x_slot.map_or(Ok(spec.x_range), ...)`: no demanded X slot passes
    // `spec.x_range` through unchanged; a demanded slot maps it into that
    // slot's units through `extent_in_space`, Bin's own one-line Option
    // pass-through (`bin.rs`'s `extent` arm) — `extent_in_space` never
    // returns `Ok(None)` for a `Some` input (the binding already refused a
    // non-finite endpoint), so this arm needs no fallback branch for that
    // case.
    let x_range = x_slot.map_or(Ok(spec.x_range), |slot| {
        slot.space
            .extent_in_space("stat_smooth", "x_range", spec.x_range.map(|[a, b]| (a, b)))
            .map(|r| r.map(|(a, b)| [a, b]))
    })?;
    let mapped_spec = SmoothSpec { method, x_range, ..spec.clone() };
    run_fit_in_log_space(
        ctx,
        "stat_smooth",
        batch,
        &TransformSpec::fit_space_columns(&spec.x, &spec.y, spec.output),
        (&spec.x, x_slot),
        (&spec.y, y_slot),
        |mapped_batch, x, y, ctx| {
            apply_in_data_space(&SmoothSpec { x: x.to_string(), y: y.to_string(), ..mapped_spec.clone() }, mapped_batch, ctx)
        },
    )
}

/// User-facing label for a `SmoothMethod`, for `TransformWarning` text —
/// matches `PySmooth::new`'s own vocabulary (`METHODS`) rather than the Rust
/// enum's variant names, except `Poly`, which names its degree since "poly"
/// alone is ambiguous between `quadratic`/`cubic`/an explicit `degree=`.
fn method_label(method: &SmoothMethod) -> String {
    match method {
        SmoothMethod::Lm => "lm".to_string(),
        SmoothMethod::Loess => "loess".to_string(),
        SmoothMethod::Poly { degree } => format!("poly(degree={degree})"),
        SmoothMethod::LogX => "log".to_string(),
        SmoothMethod::SqrtX => "sqrt".to_string(),
        SmoothMethod::Glm => "glm".to_string(),
        SmoothMethod::Gam => "gam".to_string(),
    }
}

/// `true` when every value in `batch`'s column `idx` is NaN (and the column
/// is non-empty) — the shape every degenerate branch in this file emits
/// (`build_degenerate_smooth_batch`'s `y`/`residual` column, or an all-NaN
/// `build_residuals_batch`). Used post-dispatch so the six per-method fit
/// paths (`lm`/`poly`/`log`/`sqrt`/`glm`/`gam`) can report a degenerate fit
/// through ONE check instead of threading a warning push through each of
/// their own degenerate-return sites individually — every one of them
/// already routes through `build_degenerate_smooth_batch`/an all-NaN
/// residual vec specifically because a HEALTHY fit on `xs.len() >= 2` with
/// non-zero x variance never produces an all-NaN value column (the grid is
/// built from finite `x_min`/`x_max`, and each method's predictor is finite
/// wherever `build_predictor`'s own degenerate check does not already fire).
fn value_column_is_all_nan(batch: &RecordBatch, idx: usize) -> bool {
    let Some(col) = batch.column(idx).as_any().downcast_ref::<Float64Array>() else {
        return false;
    };
    !col.is_empty() && (0..col.len()).all(|i| col.is_null(i) || col.value(i).is_nan())
}

fn apply_one_group(
    spec: &SmoothSpec,
    batch: &RecordBatch,
    only_indices: Option<&[usize]>,
    is_partition: bool,
    ctx: &TransformContext,
    group: Option<&str>,
) -> PyResult<RecordBatch> {
    let (xs_raw, ys_raw) = extract_xy(spec, batch, only_indices)?;

    // Pre-aggregate xs/ys into n equal-width bins if x_bins is set.
    let (xs, ys) = if let Some(n_bins) = spec.x_bins {
        let estimator = spec.x_estimator.unwrap_or(crate::transform::aggregate::AggFn::Mean);
        pre_aggregate_xy(&xs_raw, &ys_raw, n_bins, estimator)
    } else {
        (xs_raw.clone(), ys_raw.clone())
    };

    if xs.len() < 2 {
        ctx.warn(TransformWarning::DegenerateFit {
            transform: "Smooth".into(),
            group: group.map(str::to_string),
            reason: "fewer than 2 usable (non-null/non-NaN) x/y points".into(),
            panel: None,
        });
        return match spec.output {
            SmoothOutput::Fitted => all_nan_output(spec),
            SmoothOutput::Residuals => build_residuals_batch(
                Vec::new(), Vec::new(),
                spec.inject_zero_ref,
                // Batch-D T9 round 4: zero rows either way, but when
                // `inject_metrics` is requested the columns must still be
                // PRESENT (empty, like every other column here) rather than
                // dropped — the same schema-uniformity fix as the Fitted
                // path's `build_degenerate_smooth_batch`. `residuals::compute`
                // on empty slices is the same "no anchor row" shape
                // `append_metrics_columns` produces for a zero-length input,
                // so this needs no separate null-columns helper.
                spec.inject_metrics.then(|| residuals::compute(&[], &[])),
            ),
        };
    }

    // See the module doc's "Degenerate-input policy" section: a polynomial
    // `degree` the data cannot determine (rank-deficient Vandermonde design)
    // is parameter absurdity, not data degeneracy, so it refuses on a
    // genuine whole-frame call rather than falling through to `poly_fit`'s
    // (or `build_predictor`'s) all-NaN branch. `only_indices.is_none()`
    // alone is NOT sufficient proof of "whole frame" — `render::prepare::
    // run_transforms_with_facet` partitions the input by facet BEFORE
    // running transforms, so an ungrouped call can still be one panel of a
    // larger frame. `!is_partition` is the second half of the gate: a
    // per-group rank-deficient design under `groupby` (`only_indices` is
    // `Some`) and an ungrouped-but-partitioned facet panel (`is_partition`
    // is `true`) both degrade to that slice's own NaN range instead.
    if only_indices.is_none() && !is_partition {
        if let SmoothMethod::Poly { degree } = spec.method {
            let min_points = degree as usize + 1;
            if xs.len() < min_points {
                return Err(PyValueError::new_err(format!(
                    "Smooth: degree={degree} needs at least {min_points} data points to \
                     fit a degree-{degree} polynomial; only {} remain after dropping \
                     null/NaN x/y rows (and any x_bins aggregation). A degree this high \
                     relative to the data is a rank-deficient design, not a valid fit — \
                     lower degree or provide more data",
                    xs.len(),
                )));
            }
        }
    }

    if matches!(spec.output, SmoothOutput::Residuals) {
        // `gam` is handled separately here (rather than falling into
        // `residuals_fit`'s shared `build_predictor` dispatch below) because
        // its clamp/degenerate outcome needs to reach a warning — the same
        // report the Fitted path gets from `gam_fit`'s `GamFitOutcome`
        // further down this function. Reviewer finding (post-#162 close):
        // `build_predictor`'s `Gam` arm called `gam::fit` and threw away
        // whether it clamped, so `Smooth(method="gam", output="residuals",
        // k=<over-large>)` silently clamped with no warning at all.
        if matches!(spec.method, SmoothMethod::Gam) {
            let (out, outcome) = gam_residuals_fit(spec, &xs, &ys, &xs_raw, &ys_raw)?;
            report_gam_outcome(ctx, group, outcome);
            return Ok(out);
        }
        // Fit at original (un-aggregated) input xs; emit (x, y - fit(x)) rows.
        let out = residuals_fit(spec, &xs, &ys, &xs_raw, &ys_raw)?;
        if value_column_is_all_nan(&out, 1) {
            ctx.warn(TransformWarning::DegenerateFit {
                transform: "Smooth".into(),
                group: group.map(str::to_string),
                reason: format!("the {} fit could not be computed for this data", method_label(&spec.method)),
                panel: None,
            });
        }
        return Ok(out);
    }

    let (data_x_min, data_x_max) = xs.iter().fold((f64::INFINITY, f64::NEG_INFINITY),
        |(a, b), &v| (a.min(v), b.max(v)));

    // The evaluation range is `spec.x_range` when the caller gave one, else
    // the whole (pre-group) batch's own `x` extent -- every group spans
    // the SAME range -- either way padded by `x_range_pad` in whichever
    // units `batch`'s `x` column is already
    // stated in (data units on a linear x, this slot's log units once a
    // demanded log x has mapped both the batch and `spec.x` there before
    // this call). `None` (no `x_range` and no `x_range_pad`) keeps the
    // observed per-group data extent.
    let (x_min, x_max) = match crate::transform::numeric_util::evaluation_range(spec.x_range, spec.x_range_pad, batch, &spec.x) {
        Some([lo, hi]) if lo.is_finite() && hi.is_finite() && hi > lo => (lo, hi),
        _ => (data_x_min, data_x_max),
    };

    let grid: Vec<f64> = (0..spec.n)
        .map(|i| if spec.n <= 1 { x_min } else {
            x_min + (x_max - x_min) * (i as f64) / ((spec.n - 1) as f64)
        })
        .collect();

    // Schwabish SB-followup (2026-05-12): when ``inject_metrics`` is set
    // on the Fitted path, compute R²/RMSE/MAE against the RAW input
    // (pre-aggregation) so the metrics reflect the spread of the actual
    // data points, not the binned summary. Mirrors the residuals_fit
    // convention. The predictor is rebuilt here (rather than threaded
    // through ``lm_fit`` / ``loess_fit``) so those fit functions stay
    // focused on "fit a curve over a grid" — see SB-followup C4 rework
    // (2026-05-12).
    let metrics = if spec.inject_metrics {
        let predictor = build_predictor(spec, &xs, &ys);
        let resids: Vec<f64> = xs_raw.iter().zip(ys_raw.iter())
            .map(|(&xi, &yi)| {
                let yhat = predictor(xi);
                if yhat.is_nan() { f64::NAN } else { yi - yhat }
            })
            .collect();
        Some(residuals::compute(&ys_raw, &resids))
    } else {
        None
    };

    // `gam` reports its own (more specific) clamp/degenerate outcome via its
    // return value rather than the generic post-dispatch check below, so it
    // is handled first and returns early — see `GamFitOutcome`'s doc for why
    // `gam_fit` doesn't just take `ctx`/`group` directly. `report_gam_outcome`
    // is shared with the `SmoothOutput::Residuals` branch above, which needs
    // the identical clamp/degenerate reporting for its own gam call.
    if matches!(spec.method, SmoothMethod::Gam) {
        let (out, outcome) = gam_fit(&xs, &ys, &grid, spec.ci, spec.n, spec.k, metrics)?;
        report_gam_outcome(ctx, group, outcome);
        return Ok(out);
    }

    let (out, band_undefined) = match spec.method {
        SmoothMethod::Lm => lm_fit(&xs, &ys, &grid, spec.ci, spec.n, metrics),
        SmoothMethod::Loess => loess_fit(
            &xs, &ys, &grid, spec.bandwidth, spec.degree, spec.ci, spec.n, spec.seed,
            metrics,
        ),
        SmoothMethod::Poly { degree } => poly_fit(&xs, &ys, &grid, spec.ci, spec.n, degree, metrics),
        SmoothMethod::LogX => log_x_fit(&xs, &ys, &grid, spec.ci, spec.n, metrics),
        SmoothMethod::SqrtX => sqrt_x_fit(&xs, &ys, &grid, spec.ci, spec.n, metrics),
        SmoothMethod::Glm => glm_fit(&xs, &ys, &grid, spec.ci, spec.n, metrics),
        SmoothMethod::Gam => unreachable!("Gam handled above"),
    }?;
    // Post-dispatch degenerate check (see `value_column_is_all_nan`'s doc):
    // one check catches every one of `lm`/`poly`/`log`/`sqrt`/`glm`'s
    // degenerate-return sites, all of which route through
    // `build_degenerate_smooth_batch`.
    if value_column_is_all_nan(&out, 1) {
        ctx.warn(TransformWarning::DegenerateFit {
            transform: "Smooth".into(),
            group: group.map(str::to_string),
            reason: format!("the {} fit could not be computed for this data", method_label(&spec.method)),
            panel: None,
        });
    } else if let (Some(level), Some(reason)) = (spec.ci, band_undefined) {
        // The fit line survived; only the band is undefined. Reported once
        // the line's own degeneracy is ruled out, so a whole-fit loss
        // stays DegenerateFit alone rather than also warning here.
        ctx.warn(TransformWarning::IntervalUndefined {
            transform: "Smooth".into(),
            group: group.map(str::to_string),
            ci: level,
            reason,
            panel: None,
        });
    }
    Ok(out)
}

/// Partition input batch by the groupby columns, call apply_one_group per
/// partition, then stack the results into a single batch with the group
/// columns appended (with their original dtypes, XFORM-03) as additional fields.
fn apply_grouped(
    spec: &SmoothSpec,
    batch: &RecordBatch,
    group_cols: &[String],
    is_partition: bool,
    ctx: &TransformContext,
) -> PyResult<RecordBatch> {
    // Multi-column partition path (first-appearance order, null-key skip,
    // dtype-preserve). `SkipWithWholeBatchFallback` is #197's stated exemption
    // for this transform: the all-null-key arm below re-fits the whole batch
    // rather than emitting nothing, and the warning says so.
    let partition = group_partition_multi(
        batch,
        group_cols,
        "stat_smooth",
        NullKeyPolicy::SkipWithWholeBatchFallback,
        ctx,
    )?;

    // Collect per-group outputs and stack them.
    let mut per_group_batches: Vec<(Vec<KeyValue>, RecordBatch)> = Vec::new();
    for (g, ixs) in &partition.groups {
        // `only_indices` is `Some` here, so the rank-deficiency refusal in
        // `apply_one_group` never gates on `is_partition` for this call —
        // T6's per-group degrade-to-NaN contract already applies.
        let label = describe_group_keys(group_cols, g);
        let out = apply_one_group(spec, batch, Some(ixs), is_partition, ctx, Some(&label))?;
        per_group_batches.push((g.clone(), out));
    }

    if per_group_batches.is_empty() {
        // Fallback: every group key was null, so there is no per-group
        // output to stack. This call passes `None` for `only_indices` on
        // the FULL batch (module doc's finding D, batch-D T9 round 3): it
        // is not the `Some(ixs)` shape every other `apply_grouped` call
        // produces, so it forwards `is_partition` too — a rank-deficient
        // `Poly` here degrades like a group when this batch is itself a
        // facet panel, rather than always refusing. The pooled fit IS the
        // null-keyed group: stack it through the same template every real
        // group uses. The key nulls only the `groupby` column(s) that are
        // themselves all-null (`pooled_group_key`'s doc) — a facet field
        // folded into `group_cols` is never null and keeps its real value,
        // so a faceted panel still gets a group it can assign.
        let pooled = apply_one_group(spec, batch, None, is_partition, ctx, None)?;
        let key = pooled_group_key(batch, group_cols)?;
        return stack_grouped_batches("stat_smooth", &[(key, pooled)], group_cols, &partition.dtypes);
    }

    // Reconcile the per-group fit batches and append the group columns via
    // the ONE shared stacking template (`group_key::stack_grouped_batches`,
    // batch-F T2h hoist), which both `Smooth` and `Robust` now use — see its
    // doc for the by-name reconciliation and the groupby/output-column
    // collision guard.
    //
    // Why schema divergence across groups is possible at all: a degenerate
    // fit (rank-deficient/non-convergent design, zero-variance x, too few
    // points) for one group's own slice used to emit the CORE 4-column
    // `[x, y, ci_lower, ci_upper]` batch even when `inject_metrics` was set.
    // Batch-D T9 round 4 closed that at the source by routing every
    // degenerate branch through `build_degenerate_smooth_batch`, which emits
    // null metrics columns when requested — so this is a belt-and-braces
    // net, not a live path. The net still matters because the failure mode
    // it replaced was a live `PanicException` through `Chart.to_svg()`: the
    // original code pinned the reference schema to `per_group_batches[0]`
    // and indexed every other group POSITIONALLY against it, so a
    // degenerate first group silently dropped later groups' real metrics
    // columns and a degenerate later group indexed past its own narrower
    // schema.
    stack_grouped_batches("stat_smooth", &per_group_batches, group_cols, &partition.dtypes)
}

fn pre_aggregate_xy(
    xs: &[f64],
    ys: &[f64],
    n_bins: usize,
    estimator: crate::transform::aggregate::AggFn,
) -> (Vec<f64>, Vec<f64>) {
    use crate::transform::aggregate::aggregate as agg_fn;
    if xs.is_empty() || n_bins == 0 {
        return (Vec::new(), Vec::new());
    }
    let (x_min, x_max) = xs.iter().fold((f64::INFINITY, f64::NEG_INFINITY),
        |(a, b), &v| (a.min(v), b.max(v)));
    if x_min == x_max {
        let avg = agg_fn(ys, estimator, ys.len());
        return (vec![x_min], vec![avg]);
    }
    let mut buckets_x: Vec<Vec<f64>> = vec![Vec::new(); n_bins];
    let mut buckets_y: Vec<Vec<f64>> = vec![Vec::new(); n_bins];
    let width = x_max - x_min;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let mut idx = ((x - x_min) / width * n_bins as f64).floor() as isize;
        if idx >= n_bins as isize { idx = n_bins as isize - 1; }
        if idx < 0 { idx = 0; }
        let u = idx as usize;
        buckets_x[u].push(x);
        buckets_y[u].push(y);
    }
    let mut out_x = Vec::new();
    let mut out_y = Vec::new();
    for (xs_in, ys_in) in buckets_x.iter().zip(buckets_y.iter()) {
        if ys_in.is_empty() { continue; }
        // x always uses mean within the bin; y uses the chosen estimator.
        let mean_x = xs_in.iter().sum::<f64>() / xs_in.len() as f64;
        let agg = agg_fn(ys_in, estimator, ys_in.len());
        out_x.push(mean_x);
        out_y.push(agg);
    }
    (out_x, out_y)
}

fn extract_xy(spec: &SmoothSpec, batch: &RecordBatch, only_indices: Option<&[usize]>) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let schema = batch.schema();
    let xi = schema.index_of(&spec.x)
        .map_err(|_| PyValueError::new_err(format!("stat_smooth: column '{}' not found", spec.x)))?;
    let yi = schema.index_of(&spec.y)
        .map_err(|_| PyValueError::new_err(format!("stat_smooth: column '{}' not found", spec.y)))?;
    let xa = coerce_to_float64(batch.column(xi), "stat_smooth", &spec.x)?;
    let ya = coerce_to_float64(batch.column(yi), "stat_smooth", &spec.y)?;
    // Collect row indices that are valid in BOTH arrays (non-null, non-NaN).
    // Uses the same null+NaN predicate as clean_float64_values in numeric_util,
    // but applied to paired columns so only jointly-clean rows are kept.
    use crate::transform::numeric_util::clean_float64_values;
    let candidate_iter: Box<dyn Iterator<Item = usize>> = match only_indices {
        Some(ixs) => Box::new(ixs.iter().copied()),
        None => Box::new(0..xa.len()),
    };
    let valid_indices: Vec<usize> = candidate_iter
        .filter(|&i| {
            !xa.is_null(i) && !ya.is_null(i)
                && !xa.value(i).is_nan()
                && !ya.value(i).is_nan()
        })
        .collect();

    // Extract each column at the valid indices via the shared cleaner.
    // This reuses the same null+NaN drop logic across all transform callers.
    let xs = clean_float64_values(&xa, Some(&valid_indices));
    let ys = clean_float64_values(&ya, Some(&valid_indices));
    Ok((xs, ys))
}

fn all_nan_output(spec: &SmoothSpec) -> PyResult<RecordBatch> {
    let n = spec.n;
    build_degenerate_smooth_batch(&vec![f64::NAN; n], n, spec.inject_metrics)
}
