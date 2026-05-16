"""Matrix test: verify encoding channels affect SVG output for composite/statistical marks.

Composite marks desugar into multiple primitives internally (e.g. mark_boxplot produces
boxes, whiskers, medians, outlier points). This test verifies that visual encoding channels
actually propagate through the desugaring and produce different SVG output — catching
silently-dropped channels that are accepted at the composite level but never forwarded
to the underlying primitives.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import ferrum as fm


@pytest.fixture()
def df():
    rng = np.random.default_rng(42)
    return pl.DataFrame(
        {
            "cat": ["a"] * 30 + ["b"] * 30,
            "val": rng.normal(0, 1, 60).tolist(),
            "x": rng.normal(0, 1, 60).tolist(),
            "y": rng.normal(0, 1, 60).tolist(),
            "op": rng.uniform(0.2, 1.0, 60).tolist(),
            "sz": rng.uniform(5, 50, 60).tolist(),
        }
    )


CHANNEL_ENCODINGS = {
    "color": {"color": "cat:N"},
    "opacity": {"opacity": "op:Q"},
    "fill_opacity": {"fill_opacity": "op:Q"},
    "stroke_width": {"stroke_width": "sz:Q"},
    "size": {"size": "sz:Q"},
}

COMPOSITE_MARK_CONFIGS = {
    # boxplot/violin: only the groupby column (cat) survives the BoxStats/Violin
    # transform, so only color="cat:N" is a valid data-driven channel.
    "mark_boxplot": {
        "base_encoding": {"x": "cat:N", "y": "val:Q"},
        "channels": ["color"],
    },
    "mark_violin": {
        "base_encoding": {"x": "cat:N", "y": "val:Q"},
        "channels": ["color"],
    },
    # histogram: the Bin transform produces (bin_start, bin_end, count, density);
    # no original columns survive, so no data-driven appearance channels work.
    "mark_histogram": {
        "base_encoding": {"x": "val:Q"},
        "channels": [],
    },
    # density: auto-routes color field to groupby (group column survives in KDE
    # output), but opacity/fill_opacity reference columns that are dropped.
    "mark_density": {
        "base_encoding": {"x": "val:Q"},
        "channels": ["color"],
    },
    # smooth: the Smooth transform produces derived columns; no auto-groupby
    # from color, so no original columns survive for appearance channels.
    "mark_smooth": {
        "base_encoding": {"x": "x:Q", "y": "y:Q"},
        "channels": [],
    },
    # errorbar/errorband: ErrorExtent produces (x_field, lower, upper); only the
    # groupby field survives. color="cat:N" works via the x groupby field
    # forwarding, but opacity/stroke_width/fill_opacity reference dropped columns.
    "mark_errorbar": {
        "base_encoding": {"x": "cat:N", "y": "val:Q"},
        "channels": ["color"],
    },
    "mark_errorband": {
        "base_encoding": {"x": "cat:N", "y": "val:Q"},
        "channels": ["color"],
    },
    "mark_qq": {
        "base_encoding": {"x": "val:Q"},
        "channels": ["color"],
    },
    "mark_hex": {
        "base_encoding": {"x": "x:Q", "y": "y:Q"},
        "channels": ["color"],
    },
    "mark_swarm": {
        "base_encoding": {"x": "cat:N", "y": "val:Q"},
        "channels": ["color", "size", "opacity"],
    },
}


def _make_cases():
    cases = []
    for mark, cfg in COMPOSITE_MARK_CONFIGS.items():
        for channel in cfg["channels"]:
            cases.append((mark, channel))
    return cases


def _make_id(val):
    return val if isinstance(val, str) else None


@pytest.mark.parametrize("mark_method,channel", _make_cases(), ids=_make_id)
def test_channel_affects_output(df, mark_method, channel):
    """Channel encoding produces different SVG than the baseline for composite marks."""
    cfg = COMPOSITE_MARK_CONFIGS[mark_method]
    base_enc = cfg["base_encoding"]

    base_chart = getattr(fm.Chart(df), mark_method)().encode(**base_enc)
    base_svg = base_chart.show_svg()

    extra_encoding = {**base_enc, **CHANNEL_ENCODINGS[channel]}
    channel_chart = getattr(fm.Chart(df), mark_method)().encode(**extra_encoding)
    channel_svg = channel_chart.show_svg()

    assert base_svg != channel_svg, (
        f"{mark_method} + {channel}: SVG unchanged — channel may be silently dropped"
    )


@pytest.mark.parametrize("mark_method", list(COMPOSITE_MARK_CONFIGS.keys()))
def test_all_channels_no_crash(df, mark_method):
    """Encoding every applicable channel at once doesn't crash a composite mark."""
    cfg = COMPOSITE_MARK_CONFIGS[mark_method]
    all_encodings = dict(cfg["base_encoding"])
    for ch in cfg["channels"]:
        all_encodings.update(CHANNEL_ENCODINGS[ch])

    chart = getattr(fm.Chart(df), mark_method)().encode(**all_encodings)
    svg = chart.show_svg()
    assert "<svg" in svg
