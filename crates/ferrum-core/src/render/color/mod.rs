//! Color subsystem: categorical (8a) + continuous (8b).
pub(crate) mod categorical;
pub(crate) mod continuous;

// Re-export the public-ish API that 8a callers use.
// Glob re-export preserves all 8a public items (Color, ColorParseError,
// from_rgb, from_rgba, from_hex_str, with_opacity, fmt_svg) without
// requiring enumeration.
pub use categorical::*;
pub use continuous::{ContinuousScheme, NamedContinuous};
