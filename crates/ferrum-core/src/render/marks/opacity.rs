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
//!   `fill_opacity` column is absent ([`OpacityFallback::BarLike`]); every other
//!   mark uses [`OpacityFallback::Standard`].
//! - Scale transforms stay at the call sites. `point`/`rect` map the `opacity`
//!   channel through a scale themselves; the resolver only returns the
//!   raw-resolved (finite-checked, clamped, defaulted) encoding values.
//!
//! Sampling mode matches each mark's current behavior: `point`/`bar`/`rect` are
//! per-row (`at_row`), while `line`/`area` sample the group's first valid row
//! (`at_group_first`). Both share the same per-value resolution helper, so the
//! defaults/finite-check/clamp logic is identical regardless of sampling mode.

use crate::render::draw::{col_as_f64, DrawCtx};
use crate::render::scale_resolve::OpacityScale;

/// Map a single row's `opacity` channel value through the opacity *scale*,
/// falling back to `default` when no scale/column applies or the mapping fails.
///
/// This is the scale-mapped sibling of [`OpacityResolver`] (which only does the
/// raw finite-check/clamp/default of the `opacity`/`fill_opacity`/`stroke_opacity`
/// encoding columns). The `opacity` channel additionally passes through
/// `ctx.scales.opacity` — a per-row block that was previously copy-pasted across
/// `arc`/`point`/`rect`/`polygon` (FA-11, MOD-06). Resolution is byte-identical
/// to that inline `if let (Some(values), Some(scale)) = (&opacity_values,
/// &ctx.scales.opacity) { … } else { default }` block.
///
/// `opacity_values` is the raw `col_as_f64` column for the `opacity` field (or
/// `None` when unbound); `idx` is the per-row index (per-row marks) or the
/// group's representative row (group marks like `polygon`).
#[inline]
pub(crate) fn resolve_scaled_opacity(
    opacity_values: &Option<Vec<Option<f64>>>,
    scale: &Option<OpacityScale>,
    idx: usize,
    default: f64,
) -> f64 {
    match (opacity_values, scale) {
        (Some(values), Some(scale)) => values
            .get(idx)
            .copied()
            .flatten()
            .and_then(|v| scale.inner.to_pixel_f64(v))
            .unwrap_or(default),
        _ => default,
    }
}

/// How an absent `fill_opacity` value is defaulted.
///
/// Replaces the prior `general_fallback: bool` flag with a named two-state type
/// so call sites read intent rather than `true`/`false` (C11). Resolution is
/// byte-identical to the bool: [`BarLike`](Self::BarLike) is the old `true`,
/// [`Standard`](Self::Standard) the old `false`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum OpacityFallback {
    /// Absent `fill_opacity` uses the fill default (every mark except `bar`).
    Standard,
    /// Absent `fill_opacity` falls back to the `opacity` column first, then the
    /// fill default — `bar`'s historical quirk.
    BarLike,
}

/// Resolved opacity-channel columns for a single mark build, plus the fallback
/// mode and the per-mark defaults. Construct once via [`OpacityResolver::load`],
/// then sample with [`OpacityResolver::at_row`] or
/// [`OpacityResolver::at_group_first`].
pub(crate) struct OpacityResolver {
    /// The `opacity` encoding column (`col_as_f64`), if present.
    opacity: Option<Vec<Option<f64>>>,
    /// The `fill_opacity` encoding column (`col_as_f64`), if present.
    fill_opacity: Option<Vec<Option<f64>>>,
    /// The `stroke_opacity` encoding column (`col_as_f64`), if present.
    stroke_opacity: Option<Vec<Option<f64>>>,
    /// Controls how an absent `fill_opacity` value is defaulted: `BarLike` falls
    /// back to the `opacity` column (bar's historical behavior), `Standard` uses
    /// the fill default.
    fallback: OpacityFallback,
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
    /// `fallback` is [`OpacityFallback::BarLike`] only for `bar` (preserves its
    /// `fill_opacity ← opacity` fallback); `defaults` is
    /// `(opacity, fill_opacity, stroke_opacity)`.
    pub(crate) fn load(
        ctx: &DrawCtx,
        fallback: OpacityFallback,
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
            fallback,
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
        let fill_default = match self.fallback {
            OpacityFallback::BarLike => resolve(&self.opacity, idx, def_fill),
            OpacityFallback::Standard => def_fill,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a resolver directly (bypassing `load`) so the resolution semantics
    /// can be exercised without an Arrow batch / DrawCtx.
    fn make(
        opacity: Option<Vec<Option<f64>>>,
        fill_opacity: Option<Vec<Option<f64>>>,
        stroke_opacity: Option<Vec<Option<f64>>>,
        fallback: OpacityFallback,
        defaults: (f64, f64, f64),
    ) -> OpacityResolver {
        OpacityResolver { opacity, fill_opacity, stroke_opacity, fallback, defaults }
    }

    /// C11 guard: `Standard` and `BarLike` are the only difference and only for
    /// an *absent* `fill_opacity` column. When `fill_opacity` is present, the two
    /// fallback modes resolve identically (the bool branch was never reached).
    #[test]
    fn fallback_modes_agree_when_fill_opacity_present() {
        let op = Some(vec![Some(0.3)]);
        let fo = Some(vec![Some(0.7)]);
        let std = make(op.clone(), fo.clone(), None, OpacityFallback::Standard, (1.0, 1.0, 1.0));
        let bar = make(op, fo, None, OpacityFallback::BarLike, (1.0, 1.0, 1.0));
        assert_eq!(std.at_row(0), bar.at_row(0));
    }

    /// C11 guard: with `fill_opacity` absent, `BarLike` falls back to the
    /// `opacity` column while `Standard` uses the fill default — the exact
    /// distinction the bool encoded (`true` ⇒ BarLike, `false` ⇒ Standard).
    #[test]
    fn barlike_falls_fill_back_to_opacity_column() {
        let op = Some(vec![Some(0.4)]);
        let std = make(op.clone(), None, None, OpacityFallback::Standard, (1.0, 1.0, 1.0));
        let bar = make(op, None, None, OpacityFallback::BarLike, (1.0, 1.0, 1.0));
        // Standard: fill = fill default (1.0). BarLike: fill = opacity column (0.4).
        assert_eq!(std.at_row(0).1, 1.0);
        assert_eq!(bar.at_row(0).1, 0.4);
    }

    /// Resolution contract: finite-check, then clamp to [0,1], else default.
    /// Shared by every adopter (point/bar/rect/line/area + C7 tick/segment/rule).
    #[test]
    fn resolve_clamps_and_finite_checks() {
        let r = make(
            Some(vec![Some(1.5), Some(f64::NAN), Some(0.6), None]),
            None,
            None,
            OpacityFallback::Standard,
            (0.25, 1.0, 1.0),
        );
        assert_eq!(r.at_row(0).0, 1.0, ">1 clamps to 1.0");
        assert_eq!(r.at_row(1).0, 0.25, "NaN falls to default");
        assert_eq!(r.at_row(2).0, 0.6, "in-range passes through");
        assert_eq!(r.at_row(3).0, 0.25, "explicit null falls to default");
    }

    /// `at_row` and `at_group_first` are the same resolution at the given index —
    /// only the sampling index differs (per-row vs per-group-first marks).
    #[test]
    fn at_row_and_at_group_first_share_resolution() {
        let r = make(
            Some(vec![Some(0.2), Some(0.8)]),
            None,
            None,
            OpacityFallback::Standard,
            (1.0, 1.0, 1.0),
        );
        assert_eq!(r.at_row(1), r.at_group_first(1));
    }

    // ── resolve_scaled_opacity (FA-11 / MOD-06) ──────────────────────────────

    use crate::render::scale_resolve::ScaleKind;
    use crate::scale::linear::LinearScale;

    /// An opacity scale mapping data domain `[0, 10]` to the opacity band
    /// `[0.2, 1.0]` (the pixel range encodes the opacity endpoints).
    fn opacity_scale() -> OpacityScale {
        OpacityScale {
            inner: ScaleKind::Linear(LinearScale::new_internal(
                vec![0.0, 10.0],
                vec![0.2, 1.0],
                false,
                false,
            )),
        }
    }

    /// MOD-06 guard: a bound `opacity` column maps each value through the scale.
    /// `v=0` → band lower (0.2), `v=10` → band upper (1.0), `v=5` → midpoint 0.6.
    #[test]
    fn scaled_opacity_maps_through_scale() {
        let values = Some(vec![Some(0.0), Some(5.0), Some(10.0)]);
        let scale = Some(opacity_scale());
        assert!((resolve_scaled_opacity(&values, &scale, 0, 0.5) - 0.2).abs() < 1e-9);
        assert!((resolve_scaled_opacity(&values, &scale, 1, 0.5) - 0.6).abs() < 1e-9);
        assert!((resolve_scaled_opacity(&values, &scale, 2, 0.5) - 1.0).abs() < 1e-9);
    }

    /// MOD-06 guard: an absent column or absent scale falls back to `default`,
    /// and a null cell falls back too — byte-identical to the prior inline
    /// `else { ctx.mark_style.opacity }` arms.
    #[test]
    fn scaled_opacity_falls_back_to_default() {
        let scale = Some(opacity_scale());
        // No column.
        assert_eq!(resolve_scaled_opacity(&None, &scale, 0, 0.42), 0.42);
        // No scale.
        let values = Some(vec![Some(5.0)]);
        assert_eq!(resolve_scaled_opacity(&values, &None, 0, 0.42), 0.42);
        // Null cell.
        let with_null = Some(vec![None]);
        assert_eq!(resolve_scaled_opacity(&with_null, &scale, 0, 0.42), 0.42);
        // Out-of-range index.
        assert_eq!(resolve_scaled_opacity(&values, &scale, 9, 0.42), 0.42);
    }
}
