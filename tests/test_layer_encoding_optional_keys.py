"""Regression tests for two optional per-layer encoding keys that quality
review caught drifting in ``SpecBuildMixin._build_layers_list`` (batch-F
task 8 / F-L09-08, cycle 2).

``_build_layers_list`` builds each non-primary ``_Layer``'s wire dict from a
hand-curated allowlist of optional ``EncodingSpec`` keys -- separate from the
chart-level path (``_build_encoding_specs``), which passes a channel's full
``to_encoding_spec_dict()`` output straight into the PyO3 ``_EncodingSpec``
constructor. Two keys had drifted between the two paths:

- ``format_type``: Rust's ``EncodingSpec.format_type`` carries
  ``#[serde(rename = "formatType")]`` plus ``deny_unknown_fields``
  (``crates/ferrum-core/src/spec/encoding.rs:792``). The chart-level path
  never notices the rename (the PyO3 constructor takes the snake_case
  Python name directly and handles the wire rename internally), but the
  layer path builds a raw JSON dict by hand and must spell the wire name
  out itself -- forwarding the Python-side key verbatim made any layered
  channel with a format that derives a type (e.g. ``X(field,
  format="percent")`` -- ``_emit_format`` derives ``format_type="number"``
  from the preset) raise ``unknown field 'format_type'``.
- ``condition``: a public honored kwarg on every appearance channel
  (``encoding/_honored.py``'s ``APPEARANCE_CONDITION``, ``_emit_condition``
  in ``encoding/base.py``). The chart-level path forwards it; the layer
  path's allowlist simply never listed it, so it was silently dropped --
  the same "wire key vanishes" bug class F-L09-08 fixed for ``title=""``,
  just a different key.

Both tests use ``resolve={"y": "independent"}`` so the assertion exercises
the real public ``.to_svg()``/``.to_json()`` path: default (shared-y)
``LayerChart`` renders statically through the Rust composite overlay tree,
not the merged flat chart, so it would not exercise
``_build_layers_list`` at all (see ``LayerChart``'s module docs).
"""

from __future__ import annotations

import json

import polars as pl

import ferrum as fm


def _df():
    return pl.DataFrame(
        {
            "x": [1, 2, 3],
            "revenue": [100.0, 200.0, 300.0],
            "margin": [0.1, 0.2, 0.3],
            "g": ["a", "b", "a"],
        }
    )


def test_layer_format_percent_forwards_camel_case_format_type():
    """A non-primary layer with ``format="percent"`` must render, not raise
    ``unknown field 'format_type'`` (finding 1)."""
    df = _df()
    bars = fm.Chart(df).mark_bar().encode(x="x", y="revenue")
    points = fm.Chart(df).mark_point().encode(x="x", y=fm.Y("margin", format="percent"))
    merged = fm.LayerChart(bars, points, resolve={"y": "independent"})._build_merged()

    kw = json.loads(merged.to_json())
    y_enc = kw["layers"][1]["encoding"]["y"]
    assert y_enc["formatType"] == "number"
    assert "format_type" not in y_enc

    # Must actually render end to end (pre-fix this raised ValueError before
    # ever reaching SVG output), not just parse successfully.
    svg = merged.to_svg()
    assert "<svg" in svg


def test_layer_condition_reaches_the_wire():
    """A non-primary layer's ``condition=`` kwarg must reach that layer's
    own wire encoding dict, matching the chart-level path (finding 2)."""
    cond = {
        "selection_name": "sel_g",
        "channel": "color",
        "if_selected": {"value": "#ff0000"},
        "if_not": {"value": "#cccccc"},
    }
    df = _df()
    bars = fm.Chart(df).mark_bar().encode(x="x", y="revenue")
    points = (
        fm.Chart(df).mark_point().encode(x="x", y="margin", color=fm.Color("g", condition=cond))
    )
    merged = fm.LayerChart(bars, points, resolve={"y": "independent"})._build_merged()

    kw = json.loads(merged.to_json())
    color_enc = kw["layers"][1]["encoding"]["color"]
    assert color_enc["condition"] == cond

    svg = merged.to_svg()
    assert "<svg" in svg
