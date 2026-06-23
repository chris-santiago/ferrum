//! Color subsystem.
//!
//! - [`primitive`] — the universal `Color` primitive (RGBA), hex/CSS-name
//!   parsing, opacity application, and SVG formatting.
//! - [`continuous`] — sequential/diverging colormaps for raster/density marks.
//! - [`palette`] — categorical palette registry + scheme classification, plus
//!   the PyO3 palette-introspection functions.
pub(crate) mod continuous;
pub(crate) mod palette;
pub(crate) mod primitive;

// Explicit re-export of the color primitive's public API. Replaces an earlier
// glob (`pub use categorical::*`); enumerating the surface keeps the
// crate-visible color API auditable and prevents accidental leakage of new
// private helpers. These are the symbols actually consumed via
// `render::color::*` across the crate. `ColorParseError` (the error returned by
// `parse_color`/`from_hex_str`) is handled by-value at call sites and never
// named through this path, so it stays reachable as `color::primitive::ColorParseError`
// rather than being re-exported here (an explicit re-export would warn unused).
pub use primitive::{
    fmt_svg, from_hex_str, from_rgb, from_rgba, parse_color, with_opacity, Color,
};

pub use continuous::{ContinuousScheme, NamedContinuous};
