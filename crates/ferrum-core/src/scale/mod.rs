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
//! - `SequentialScale`, `DivergingScale`, `BinOrdinalScale`, and the `Quantize`,
//!   `Quantile`, and `Threshold` wire types all have `ScaleSpec` variants but no
//!   dedicated `ScaleKind`; the positional resolver degrades
//!   `ScaleSpec::Sequential`, `ScaleSpec::Diverging`, `ScaleSpec::Quantize`,
//!   `ScaleSpec::Quantile`, and `ScaleSpec::Threshold` to `ScaleKind::Linear`,
//!   and `ScaleSpec::BinOrdinal` likewise to `ScaleKind::Linear`. These are
//!   primarily color / discrete-binning scales; in a positional channel they fall
//!   back to Linear.
//!
//! This means a band declared via `fr.BandScale(...)` in Python and one declared
//! via `scale={"type": "band"}` in chart JSON still take **different render code
//! paths**: the former's `layout()` / `scale_str()` math never runs during
//! rendering (the renderer resolves the emitted `ScaleSpec::Band`, not the
//! pyclass).
//!
//! **The dual-representation link is now single-sourced (SPEC-04).** Each `*Scale`
//! pyclass exposes an inherent `to_scale_spec(&self) -> ScaleSpec` (with a
//! `#[pymethods]` `_to_scale_spec_dict` wrapper that serializes it via
//! `crate::spec::encoding::encode_serde_value_for_py`). The Python bridge
//! `ferrum.encoding._scale._scale_to_dict` delegates to that wrapper instead of
//! hand-copying fields, so the wire form is emitted from one place next to
//! `ScaleSpec`. Extending a `ScaleSpec` variant now breaks its `to_scale_spec`
//! builder until updated (the compile-time drift guard), and a parity test
//! enumerates every pyclass → variant mapping (the test-time guard). The structs
//! stay thin *builders* — they emit `ScaleSpec` rather than storing one, because
//! their compute facades (`.bandwidth()`, `.scale()`, `.ticks()`) need resolved
//! numeric domain/range that `ScaleSpec` does not carry.

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
