//! Serde-serializable spec types mirroring the Python declaration API.
//!
//! # Serde tag convention
//!
//! All tagged spec enums in this module discriminate on the field `"kind"`
//! (e.g. [`CoordKind`], [`data_ref::DataRef`]). The **single exception** is
//! [`encoding::ScaleSpec`], which uses `tag = "type"` for Vega-Lite wire-format
//! interop. That choice is intentional and isolated to one enum; see the comment
//! on `ScaleSpec` in `encoding.rs` for the rationale. New spec enums should tag
//! on `"kind"` unless they have an equivalent wire-format constraint.

pub(crate) mod data_ref;
pub(crate) mod mark;
pub(crate) mod encoding;
pub(crate) mod chart;
pub(crate) mod composite;
pub(crate) mod parameter;
pub mod layer;
pub use layer::Layer;
pub mod coord;
pub use coord::CoordKind;
pub mod mark_style;
pub use mark_style::MarkKwargsSpec;
pub mod position;
pub use position::PositionAdjust;
pub mod title;
pub use title::TitleSpec;
