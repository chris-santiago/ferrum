//! User-facing PyO3 scale classes and shared scale utilities.
//!
//! This module contains the `*Scale` structs exposed to Python (BandScale,
//! PointScale, LinearScale, LogScale, etc.). These are **builder / compute objects**
//! for direct Python use — e.g., querying bandwidth, tick positions, or scale
//! mappings outside of a chart render.
//!
//! **They are not used by the render path.** The renderer works exclusively from
//! `crate::spec::encoding::ScaleSpec`, which it resolves into `ScaleKind` instances
//! via `crate::render::scale_resolve::positional::build_from_scale_spec`. Specifically:
//!
//! - `BandScale` and `PointScale` constructed here never participate in rendering.
//!   A `ScaleSpec::Band` / `ScaleSpec::Point` resolves to `ScaleKind::Ordinal` via
//!   `OrdinalScale::new_internal`; the `BandScale`/`PointScale` layout math
//!   (`.bandwidth()`, `.scale()`) is user-query-only.
//! - `SequentialScale`, `DivergingScale`, `BinOrdinalScale`, and the `Quantize`
//!   wire type all have `ScaleSpec` variants but no dedicated `ScaleKind`; the
//!   positional resolver degrades `ScaleSpec::Sequential`, `ScaleSpec::Diverging`,
//!   and `ScaleSpec::Quantize` to `ScaleKind::Linear`, and `ScaleSpec::BinOrdinal`
//!   likewise to `ScaleKind::Linear`. These are primarily color scales; in a
//!   positional channel they fall back to Linear.
//! - `QuantileScale` and `ThresholdScale` are PyO3-only classes with **no
//!   `ScaleSpec` counterpart at all** (the wire vocabulary has `Quantize`, not
//!   `Quantile` or `Threshold`). They never participate in rendering through the
//!   positional resolver. This name/representation mismatch is itself a concrete
//!   instance of the dual-representation drift this note warns about.
//!
//! This means a band declared via `fr.BandScale(...)` in Python and one declared
//! via `scale={"type": "band"}` in chart JSON take **different code paths**: the
//! former's `layout()` / `scale_str()` math never runs during rendering. The field
//! sets, defaults, and validation logic of the PyO3 classes and `ScaleSpec` variants
//! can therefore drift independently without any compile-time or runtime check.
//!
//! The full reconciliation (a `to_scale_spec` bridge, or demoting the compute
//! facades to crate-internal helpers) is a tracked follow-up.

pub(crate) mod core;
pub(crate) mod ticks;
pub(crate) mod linear;
pub(crate) mod log;
pub(crate) mod symlog;
pub(crate) mod time;
pub(crate) mod ordinal;
pub(crate) mod threshold;
pub(crate) mod quantile;
pub(crate) mod pow;
pub(crate) mod band;
pub(crate) mod point;
pub(crate) mod sequential;
pub(crate) mod diverging;
pub(crate) mod quantize;
pub(crate) mod bin_ordinal;
