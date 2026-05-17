//! Phase 5 — stat engine. Mirrors the layout of `crate::scale`:
//! `core.rs` holds the sealed `TransformSpec` enum; per-variant files
//! own their `apply` math; `linalg.rs` is a small shared utility.

pub(crate) mod core;
pub(crate) mod bin;
pub(crate) mod bin_2d;
pub(crate) mod context;
pub(crate) mod kde;
pub(crate) mod kde_2d;
pub(crate) mod smooth;
pub(crate) mod aggregate;
pub(crate) mod summary;
pub(crate) mod outliers;
pub(crate) mod error_extent;
pub(crate) mod box_stats;
pub(crate) mod violin;
pub(crate) mod contour;
pub(crate) mod qq;
pub(crate) mod raster;
pub(crate) mod hex;
pub(crate) mod swarm;
pub(crate) mod unpivot;
pub(crate) mod reorder;
pub(crate) mod reference_line;
pub(crate) mod linkage;
pub(crate) mod letter_value;
pub(crate) mod logistic;
pub(crate) mod glm;
pub(crate) mod robust;
pub(crate) mod identity;
pub(crate) mod residuals;
pub(crate) mod linalg;
pub(crate) mod stats;
pub(crate) mod expr;

// Phase 12 data transforms
pub(crate) mod filter;
pub(crate) mod calculate;
pub(crate) mod fold;
pub(crate) mod pivot;
pub(crate) mod join_aggregate;
pub(crate) mod data_window;
pub(crate) mod density_data;
pub(crate) mod regression_data;
pub(crate) mod loess_data;
pub(crate) mod impute;
pub(crate) mod flatten;
pub(crate) mod sample;
pub(crate) mod top_k;
pub(crate) mod data_stack;
pub(crate) mod timeunit;
pub(crate) mod data_bin;
pub(crate) mod data_aggregate;
