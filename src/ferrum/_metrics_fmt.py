"""Shared corner-metrics formatter used by Python and Rust paths.

The Rust ``Smooth`` / ``Robust`` transforms emit a ``_metrics_text`` column
formatted via ``format!("R² {r2:.3}\\nRMSE {rmse:.3}\\nMAE {mae:.3}")`` (see
``crates/ferrum-core/src/transform/residuals.rs``). The Python-side
``_inject_metrics_corner`` (in ``ferrum._diagnostics.charts``) augments a
DataFrame without going through a Rust transform — but it must produce a
byte-identical string so chart renderings are interchangeable between the
two paths.

This module owns that one formatting decision on the Python side. The
Rust path is a deliberate near-duplicate because the format spec lives
across the FFI boundary; a small regression test in
``tests/test_metrics_fmt_parity.py`` asserts byte-equality of the two
emissions on a known fixture so any future format change is caught.
"""

from __future__ import annotations


def format_corner_metrics(r2: float, rmse: float, mae: float) -> str:
    """Return the canonical ``"R² {r2:.3f}\\nRMSE {rmse:.3f}\\nMAE {mae:.3f}"`` string.

    Matches Rust's ``residuals::append_metrics_columns`` format byte-for-byte
    on the same numeric inputs. Used by ``_inject_metrics_corner`` (Python
    residuals path) and any future Python-side corner-metrics emission.
    """
    return f"R² {r2:.3f}\nRMSE {rmse:.3f}\nMAE {mae:.3f}"
