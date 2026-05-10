//! TransformContext: render-time info passed to transforms that need viewport.
//! Used by Raster (resolution="screen") and Swarm (point radius unit conversion).

#[derive(Debug, Clone, Copy)]
pub struct TransformContext {
    /// Panel pixel size (width, height). For raster `resolution="screen"`,
    /// this is the grid dimension. For swarm, used to convert point pixels
    /// to data-space radius via the value-axis scale.
    pub panel_pixel_size: Option<(u32, u32)>,
}

impl Default for TransformContext {
    fn default() -> Self {
        Self { panel_pixel_size: None }
    }
}
