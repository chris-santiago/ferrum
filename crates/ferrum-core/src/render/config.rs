//! RenderConfig — Phase 7 honored fields only. See spec §5.3.

use palette::Srgba;

/// Render-time configuration. Phase 7 honors a small subset of `ferrum-spec.md §3.16`.
/// Future phases will widen this struct as their corresponding features ship
/// (engine, backend, raster_*, tile_parallel, font_path).
#[derive(Debug, Clone, PartialEq)]
pub struct RenderConfig {
    /// PNG density multiplier. Default 2.0 (retina).
    pub scale: f64,
    /// Whether to inline @font-face base64 in SVG output.
    /// Phase 7 always treats this as true (locked decision §11 row 15);
    /// the field is preserved for future phases.
    pub embed_fonts: bool,
    /// Override chart background color for export.
    pub background: Option<Srgba<u8>>,
    /// Override viewport.width if Some.
    pub width: Option<f64>,
    /// Override viewport.height if Some.
    pub height: Option<f64>,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self { scale: 2.0, embed_fonts: true, background: None, width: None, height: None }
    }
}
