//! TransformContext: render-time info passed to transforms that need viewport
//! or cross-transform output access.
//!
//! Used by:
//! - Raster (resolution="screen") and Swarm (point radius unit conversion) — both
//!   read `panel_pixel_size`.
//! - Reorder with `from=<named_output>` (Phase 9 finalize) — reads `named_outputs`
//!   to look up the index column from a sibling named transform output.

use std::collections::HashMap;
use arrow::record_batch::RecordBatch;

#[derive(Debug, Clone, Default)]
pub struct TransformContext {
    /// Panel pixel size (width, height). For raster `resolution="screen"`,
    /// this is the grid dimension. For swarm, used to convert point pixels
    /// to data-space radius via the value-axis scale.
    pub panel_pixel_size: Option<(u32, u32)>,
    /// Named outputs produced by earlier transforms in the same `apply_transforms_named`
    /// run. Used by Reorder(from='<name>') to access a sibling order column.
    pub named_outputs: HashMap<String, RecordBatch>,
}
