//! Phase 7 — static renderer. Pure functions: ChartSpec + RecordBatch + ThemeInputs +
//! Viewport -> deterministic SVG/PNG. See docs/superpowers/specs/2026-05-09-static-renderer-design.md.

pub(crate) mod config;
pub(crate) mod color;
pub(crate) mod palette;
pub(crate) mod font;
pub(crate) mod format;
pub(crate) mod svg;
pub(crate) mod embed_font;
pub(crate) mod scale_resolve;
pub(crate) mod prepare;
pub(crate) mod draw;
pub(crate) mod png;
pub(crate) mod binding;
pub(crate) mod marks;

// Constants (spec §6.1).
pub const FLOAT_PRECISION: usize = 3;
pub const DEFAULT_GRID_ENABLED: bool = true;
pub const CLIP_ID_PREFIX: &str = "ferrum-clip-";
pub const INTER_FONT_FAMILY: &str = "Inter";
