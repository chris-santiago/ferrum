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
pub(crate) mod axis;
pub(crate) mod legend;
pub(crate) mod strip_title;
