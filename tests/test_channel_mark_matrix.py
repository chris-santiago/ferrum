"""Matrix test: verify every renderer-honored channel affects SVG output for applicable marks.

For each (mark, channel) pair where the channel should visually affect that mark,
render twice — once with just positional encodings, once with the channel added — and
assert the SVGs differ. This catches silently-dropped channels that are accepted but
never rendered.
"""

from __future__ import annotations

import pytest
import polars as pl

import ferrum as fm


@pytest.fixture()
def df():
    return pl.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [2.0, 4.0, 1.0, 5.0, 3.0],
        "x2": [2.0, 3.0, 4.0, 5.0, 6.0],
        "y2": [3.0, 5.0, 2.0, 6.0, 4.0],
        "cat": ["a", "b", "a", "b", "a"],
        "cat2": ["x", "y", "x", "y", "x"],
        "label": ["one", "two", "three", "four", "five"],
        "sz": [10.0, 20.0, 30.0, 40.0, 50.0],
        "ang": [0.0, 45.0, 90.0, 135.0, 180.0],
        "op": [0.2, 0.4, 0.6, 0.8, 1.0],
        "url": [
            "http://a.com",
            "http://b.com",
            "http://c.com",
            "http://d.com",
            "http://e.com",
        ],
    })


CHANNEL_ENCODINGS = {
    "color": {"color": "cat:N"},
    "size": {"size": "sz:Q"},
    "shape": {"shape": "cat:N"},
    "opacity": {"opacity": "op:Q"},
    "fill_opacity": {"fill_opacity": "op:Q"},
    "stroke": {"stroke": "cat:N"},
    "stroke_width": {"stroke_width": "sz:Q"},
    "angle": {"angle": "ang:Q"},
    "text": {"text": "label:N"},
    "tooltip": {"tooltip": "label:N"},
    "href": {"href": "url:N"},
}

MARK_CONFIGS = {
    "mark_point": {
        "base_encoding": {"x": "x:Q", "y": "y:Q"},
        "channels": ["color", "size", "shape", "opacity", "fill_opacity", "stroke", "stroke_width", "angle", "tooltip", "href"],
    },
    "mark_line": {
        "base_encoding": {"x": "x:Q", "y": "y:Q"},
        "channels": ["color", "opacity", "stroke_width", "tooltip"],
    },
    "mark_bar": {
        "base_encoding": {"x": "cat:N", "y": "y:Q"},
        "channels": ["color", "opacity", "fill_opacity", "stroke", "stroke_width", "tooltip", "href"],
    },
    "mark_area": {
        "base_encoding": {"x": "x:Q", "y": "y:Q"},
        "channels": ["color", "opacity", "fill_opacity", "tooltip"],
    },
    "mark_rule": {
        "base_encoding": {"x": "x:Q", "y": "y:Q"},
        "channels": ["color", "opacity", "stroke_width", "tooltip"],
    },
    "mark_text": {
        "base_encoding": {"x": "x:Q", "y": "y:Q"},
        "channels": ["color", "opacity", "size", "angle", "text", "tooltip"],
    },
    "mark_tick": {
        "base_encoding": {"x": "x:Q", "y": "y:Q"},
        "channels": ["color", "opacity", "stroke_width", "tooltip"],
    },
    "mark_rect": {
        "base_encoding": {"x": "cat:N", "y": "cat2:N"},
        "channels": ["color", "opacity", "fill_opacity", "stroke", "stroke_width", "tooltip", "href"],
    },
    "mark_segment": {
        "base_encoding": {"x": "x:Q", "y": "y:Q", "x2": "x2:Q", "y2": "y2:Q"},
        "channels": ["color", "opacity", "stroke_width", "tooltip"],
    },
}


def _make_cases():
    cases = []
    for mark, cfg in MARK_CONFIGS.items():
        for channel in cfg["channels"]:
            cases.append((mark, channel))
    return cases


def _make_id(val):
    return val if isinstance(val, str) else None


@pytest.mark.parametrize("mark_method,channel", _make_cases(), ids=_make_id)
def test_channel_affects_output(df, mark_method, channel):
    """Channel encoding produces different SVG than the baseline."""
    cfg = MARK_CONFIGS[mark_method]
    base_enc = cfg["base_encoding"]

    base_chart = getattr(fm.Chart(df), mark_method)().encode(**base_enc)
    base_svg = base_chart.show_svg()

    extra_encoding = {**base_enc, **CHANNEL_ENCODINGS[channel]}
    channel_chart = getattr(fm.Chart(df), mark_method)().encode(**extra_encoding)
    channel_svg = channel_chart.show_svg()

    assert base_svg != channel_svg, (
        f"{mark_method} + {channel}: SVG unchanged — channel may be silently dropped"
    )


@pytest.mark.parametrize("mark_method", list(MARK_CONFIGS.keys()))
def test_all_channels_no_crash(df, mark_method):
    """Encoding every applicable channel at once doesn't crash."""
    cfg = MARK_CONFIGS[mark_method]
    all_encodings = dict(cfg["base_encoding"])
    for ch in cfg["channels"]:
        all_encodings.update(CHANNEL_ENCODINGS[ch])

    chart = getattr(fm.Chart(df), mark_method)().encode(**all_encodings)
    svg = chart.show_svg()
    assert "<svg" in svg
