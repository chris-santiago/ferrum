//! Phase 5 — stat engine. Mirrors the layout of `crate::scale`:
//! `core.rs` holds the sealed `TransformSpec` enum; per-variant files
//! own their `apply` math; `linalg.rs` is a small shared utility.

pub(crate) mod core;
pub(crate) mod bin;
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
pub(crate) mod linalg;
