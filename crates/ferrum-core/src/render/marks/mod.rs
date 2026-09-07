//! Per-mark draw functions. Each module exports a free `draw(ctx, out)` fn dispatched
//! from `render::draw::dispatch_mark`. Internal-only helpers (`axis`, `legend`,
//! `strip_title`) are not surfaced as primitive marks.

pub(crate) mod point;
pub(crate) mod line;
pub(crate) mod area;
pub(crate) mod bar;
pub(crate) mod rect;
pub(crate) mod rule;
pub(crate) mod text;
pub(crate) mod tick;
pub(crate) mod polygon;
pub(crate) mod image;
pub(crate) mod ribbon;
pub(crate) mod segment;
pub(crate) mod arc;
pub(crate) mod label;
pub(crate) mod geoshape;
pub(crate) mod axis;
pub(crate) mod legend;
pub(crate) mod strip_title;
pub(crate) mod opacity;
pub(crate) mod channels;
pub(crate) mod curve;

/// Test-only: the cross-mark color-dtype parity table (NF-A3). It has no home
/// inside any single mark because the invariant — and the defect it guards —
/// spans the family; see the module's own doc.
#[cfg(test)]
mod color_dtype_parity;

/// Test-only: the cross-mark band-width family invariant (F-L04-03, spec §4A)
/// — every ordinal mark's width is the resolved scale's drawn band times its
/// own `band_size` factor. Same reason as above: nine formulas across `bar`,
/// `rect` and `tick` implement one sentence.
#[cfg(test)]
mod band_width;

/// Test-only: `clip=False` means one thing across the whole line family
/// (`line`, `area`, `ribbon`). Same reason again — the kwarg is accepted by all
/// three, so a per-mark pin is exactly what let two of them keep only half of
/// it.
#[cfg(test)]
mod clip_escape;

/// Test-only: the cross-mark `CurveAxis` agreement matrix. Three independent
/// derivations (`series_curve_axis`, `BottomEdge::axis`, `RIBBON_CURVE_AXIS`)
/// answer one question, getting it wrong is silent, and it has already gone
/// wrong twice — so the check has to compare each mark's chosen axis against
/// the shape its point loop actually builds.
#[cfg(test)]
mod curve_axis;
