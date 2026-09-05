//! TransformContext: render-time info passed to transforms that need viewport
//! or cross-transform output access.
//!
//! Used by:
//! - Raster (resolution="screen"), Swarm (point radius unit conversion), and
//!   Violin (band_pixels) — all three read `panel_pixel_size`, populated by
//!   the production pipeline as a pre-layout estimate (see the field's own
//!   doc for the estimate's exact semantics and limits).
//! - Reorder with `from=<named_output>` (Phase 9 finalize) — reads `named_outputs`
//!   to look up the index column from a sibling named transform output.

use std::collections::HashMap;
use arrow::record_batch::RecordBatch;

#[derive(Debug, Clone, Default)]
pub struct TransformContext {
    /// Panel pixel size (width, height). For raster `resolution="screen"`,
    /// this is the grid dimension. For swarm and violin, used to convert
    /// point/band pixels to data-space radius via the value-axis scale.
    ///
    /// **This is a pre-layout ESTIMATE, not the true rendered panel size**
    /// (M2, spec §4.2 of `2026-09-05-batch-d-transform-engine-design.md`).
    /// The production render pipeline populates it, at the single
    /// construction site `render/prepare/mod.rs::run_transforms_with_facet`,
    /// with `layout::estimate_panel_pixel_size`: the viewport minus the
    /// theme's padding, divided per-axis by the facet grid's column/row
    /// counts when the chart facets. It deliberately EXCLUDES axis titles,
    /// tick labels, legends, and facet strip bands — all unknowable before
    /// layout runs, since layout itself measures the transformed output this
    /// context seeds. True panel size is therefore circular pre-layout; this
    /// field is the best available approximation, named as such everywhere
    /// it is read (`raster.rs`, `swarm.rs`, `violin.rs`).
    ///
    /// `None` on a bare `apply()` call with no render context (direct calls,
    /// unit tests, WASM/notebook direct usage) — each of the three readers
    /// above falls back to a fixed default in that case; those fallbacks are
    /// unaffected by this field's production population.
    pub panel_pixel_size: Option<(u32, u32)>,
    /// Named outputs produced by earlier transforms in the same `apply_transforms_named`
    /// run. Used by Reorder(from='<name>') to access a sibling order column.
    pub named_outputs: HashMap<String, RecordBatch>,
}
