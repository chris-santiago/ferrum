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
    /// True when `batch` (the argument to `apply`/`apply_with_context`) is one
    /// partition of a conceptually larger frame, rather than the whole frame
    /// itself — e.g. one facet panel after
    /// `render::prepare::run_transforms_with_facet` partitions the input by
    /// the facet field(s) BEFORE running transforms (the sole production
    /// construction site that sets this `true` today, batch-D T9 round 3).
    ///
    /// `false` (the derived `Default`) means "genuinely the whole frame":
    /// the non-faceted production path, a bare `apply()`/`apply_with_context`
    /// call with no context (direct calls, unit tests, WASM/notebook direct
    /// usage), and every `TransformContext` built before this field existed.
    ///
    /// Consumers use this to distinguish "the caller asked for something the
    /// data cannot support" (refuse) from "this data subset happens to be
    /// too sparse for the fit" (degrade), matching how a `groupby` group
    /// already degrades rather than refusing — see `smooth::apply_one_group`'s
    /// degenerate-input policy doc for the one consumer today.
    ///
    /// Audited (batch-D T9 round 4) for every other place a frame might
    /// reach a transform as a fragment, per the spec review's ⚠️: `compare=`
    /// small multiples (`ferrum.plots._helpers._compose_compare`) and
    /// composite/repeat leaves (`render::composite_render`,
    /// `render::binding::build_composite_leaves`) never partition a batch
    /// inside Rust. Each composite leaf's `data` index selects a whole
    /// batch Python already uploaded as that leaf's own complete dataset —
    /// one dataframe per `compare=` model, sliced in Python before crossing
    /// the PyO3 boundary — and `prepare_render_inputs` is called exactly
    /// once per leaf with no further slicing (`grep` over the crate finds
    /// no other `TransformContext` construction or `apply_transforms_named`
    /// call outside `run_transforms_with_facet` and the `apply()`
    /// fallbacks). `false` for a composite leaf is therefore the correct
    /// whole-frame-for-that-leaf answer, not an unaudited gap:
    /// `run_transforms_with_facet` remains the ONLY place Rust itself
    /// partitions a batch before running transforms, so it is exhaustively
    /// the only construction site this field needs. A leaf whose OWN
    /// `spec.facet` is set still gets `is_partition = true` correctly,
    /// since `prepare_render_inputs` → `run_transforms_with_facet` runs
    /// per leaf, independent of composite nesting.
    pub is_partition: bool,
}
