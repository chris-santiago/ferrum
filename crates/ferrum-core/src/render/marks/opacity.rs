//! Shared opacity-channel resolution for the `area`/`bar`/`line`/`point`/`rect`
//! marks (FA-11, GH #5).
//!
//! Before this module each mark hand-rolled the same three-step pattern for the
//! `opacity` / `fill_opacity` / `stroke_opacity` encoding columns: load the
//! column via `col_as_f64`, sample the relevant row, finite-check, then clamp to
//! `[0, 1]` with a per-mark default. `OpacityResolver` centralizes that pattern
//! so the five marks share one byte-identical implementation.
//!
//! Option A (GH #5, user decision 2026-06-20): this is a *deduplication plus the
//! line `fill_opacity` fix* — it does NOT change opacity-composition semantics.
//! Each mark keeps its current behavior:
//! - `bar` alone falls `fill_opacity` back to the `opacity` column when the
//!   `fill_opacity` column is absent (`general_fallback = true`); every other
//!   mark uses `general_fallback = false`.
//! - Scale transforms stay at the call sites. `point`/`rect` map the `opacity`
//!   channel through a scale themselves; the resolver only returns the
//!   raw-resolved (finite-checked, clamped, defaulted) encoding values.
//!
//! Sampling mode matches each mark's current behavior: `point`/`bar`/`rect` are
//! per-row (`at_row`), while `line`/`area` sample the group's first valid row
//! (`at_group_first`). Both share the same per-value resolution helper, so the
//! defaults/finite-check/clamp logic is identical regardless of sampling mode.

use crate::render::draw::{col_as_f64, DrawCtx};

/// Resolved opacity-channel columns for a single mark build, plus the fallback
/// flag and the per-mark defaults. Construct once via [`OpacityResolver::load`],
/// then sample with [`OpacityResolver::at_row`] or
/// [`OpacityResolver::at_group_first`].
pub(crate) struct OpacityResolver {
    /// The `opacity` encoding column (`col_as_f64`), if present.
    opacity: Option<Vec<Option<f64>>>,
    /// The `fill_opacity` encoding column (`col_as_f64`), if present.
    fill_opacity: Option<Vec<Option<f64>>>,
    /// The `stroke_opacity` encoding column (`col_as_f64`), if present.
    stroke_opacity: Option<Vec<Option<f64>>>,
    /// When `true`, an absent `fill_opacity` value falls back to the `opacity`
    /// column (bar's historical behavior); otherwise it uses the fill default.
    general_fallback: bool,
    /// `(opacity, fill_opacity, stroke_opacity)` defaults used when a channel is
    /// absent / non-finite for a given index.
    defaults: (f64, f64, f64),
}

/// Finite-check, then clamp to `[0, 1]`, falling back to `default` when the
/// value is absent or non-finite. Mirrors every mark's prior inline logic.
#[inline]
fn resolve(col: &Option<Vec<Option<f64>>>, idx: usize, default: f64) -> f64 {
    col.as_ref()
        .and_then(|v| v.get(idx).copied().flatten())
        .filter(|v| v.is_finite())
        .map(|v| v.clamp(0.0, 1.0))
        .unwrap_or(default)
}

impl OpacityResolver {
    /// Load the `opacity` / `fill_opacity` / `stroke_opacity` encoding columns.
    ///
    /// `general_fallback` is `true` only for `bar` (preserves its
    /// `fill_opacity ← opacity` fallback); `defaults` is
    /// `(opacity, fill_opacity, stroke_opacity)`.
    pub(crate) fn load(
        ctx: &DrawCtx,
        general_fallback: bool,
        defaults: (f64, f64, f64),
    ) -> Self {
        let enc = &ctx.spec.encoding;
        OpacityResolver {
            opacity: enc
                .opacity
                .as_ref()
                .and_then(|e| col_as_f64(ctx.batch, &e.field).ok()),
            fill_opacity: enc
                .fill_opacity
                .as_ref()
                .and_then(|e| col_as_f64(ctx.batch, &e.field).ok()),
            stroke_opacity: enc
                .stroke_opacity
                .as_ref()
                .and_then(|e| col_as_f64(ctx.batch, &e.field).ok()),
            general_fallback,
            defaults,
        }
    }

    /// Resolve `(opacity, fill_opacity, stroke_opacity)` at index `idx`.
    ///
    /// Shared by [`at_row`](Self::at_row) and
    /// [`at_group_first`](Self::at_group_first) — the sampling index is the only
    /// difference between per-row and per-group-first marks.
    fn sample(&self, idx: usize) -> (f64, f64, f64) {
        let (def_op, def_fill, def_stroke) = self.defaults;
        let opacity = resolve(&self.opacity, idx, def_op);
        // bar's quirk: when the fill_opacity column is absent at this index, fall
        // back to the opacity column (clamped/finite-checked) before the default.
        let fill_default = if self.general_fallback {
            resolve(&self.opacity, idx, def_fill)
        } else {
            def_fill
        };
        let fill_opacity = resolve(&self.fill_opacity, idx, fill_default);
        let stroke_opacity = resolve(&self.stroke_opacity, idx, def_stroke);
        (opacity, fill_opacity, stroke_opacity)
    }

    /// Sample at row `i` (per-row marks: `point`, `bar`, `rect`).
    pub(crate) fn at_row(&self, i: usize) -> (f64, f64, f64) {
        self.sample(i)
    }

    /// Sample at the group's first valid row `first` (per-group marks: `line`,
    /// `area`).
    pub(crate) fn at_group_first(&self, first: usize) -> (f64, f64, f64) {
        self.sample(first)
    }
}
