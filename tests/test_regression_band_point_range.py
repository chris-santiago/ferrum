"""Regression tests for issue #39: BandScale/PointScale `range=` silently dropped.

Before the fix, ``ScaleSpec::Band``/``ScaleSpec::Point`` had no ``range`` field on
the Rust side, so a user-supplied ``fr.BandScale(domain=[...], range=[lo, hi])`` (or
``fr.PointScale(...)``) round-tripped through ``_to_scale_spec_dict()`` /
``chart.to_json()`` with the range silently stripped, and the positional resolver
fell back to the full panel extent regardless of what the caller asked for.

The fix adds the field, emits it from ``to_scale_spec``, and honors it in the
positional resolver (falling back to the panel extent only when ``range`` is
absent). These tests check three things:

1. **Wire-level** — ``chart.to_json()`` carries the requested ``range`` under the
   band/point scale on the x encoding.
2. **Render-level** — with an explicit sub-panel ``range``, all rendered mark
   positions (bar x/width for BandScale, circle cx for PointScale) fall inside
   that sub-range rather than spanning the full panel.
3. **Fallback** — a band scale built *without* ``range`` has no ``"range"`` key
   in its wire dict, so the panel-extent fallback path is untouched by the fix.
"""

from __future__ import annotations

import json
import re

import polars as pl

import ferrum as fr

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _data_bar_rects(svg: str) -> list[dict[str, str]]:
    """Return attr dicts for <rect> elements that are data bars.

    Background/panel rects either carry the theme background fill or omit
    ``fill`` entirely (inheriting from a parent group); data bars always carry
    an explicit non-background fill.
    """
    rects = []
    for m in re.finditer(r"<rect([^/]+)/>", svg):
        attrs = dict(re.findall(r'([\w-]+)="([^"]+)"', m.group(1)))
        fill = attrs.get("fill")
        if fill and fill != "#faf7f2":
            rects.append(attrs)
    return rects


def _circle_cxs(svg: str) -> list[float]:
    """Return sorted cx values from all <circle> elements in the SVG."""
    return sorted(float(v) for v in re.findall(r'<circle[^>]*cx="([^"]+)"', svg))


# ---------------------------------------------------------------------------
# 1. Wire-level: range survives to_json()
# ---------------------------------------------------------------------------


def test_band_scale_range_present_in_wire_json():
    """chart.to_json() carries BandScale(range=[...]) under the x scale.

    Regression: issue #39 — ScaleSpec::Band had no range field, so this key
    was silently absent from the wire dict.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c", "d"], "val": [10.0, 20.0, 30.0, 40.0]})
    chart = (
        fr.Chart(df)
        .mark_bar()
        .encode(
            x=fr.X("cat", scale=fr.BandScale(domain=["a", "b", "c", "d"], range=[40.0, 260.0])),
            y="val",
        )
    )
    spec = json.loads(chart.to_json())
    x_scale = spec["encoding"]["x"]["scale"]
    assert x_scale.get("type") == "band"
    assert x_scale.get("range") == [40.0, 260.0], (
        f"BandScale range was not emitted in to_json(); got scale dict {x_scale!r}"
    )


def test_point_scale_range_present_in_wire_json():
    """chart.to_json() carries PointScale(range=[...]) under the x scale.

    Regression: issue #39 — ScaleSpec::Point had no range field, so this key
    was silently absent from the wire dict.
    """
    df = pl.DataFrame({"cat": ["x", "y", "z"], "val": [5.0, 10.0, 15.0]})
    chart = (
        fr.Chart(df)
        .mark_point()
        .encode(
            x=fr.X("cat", scale=fr.PointScale(domain=["x", "y", "z"], range=[40.0, 260.0])),
            y="val",
        )
    )
    spec = json.loads(chart.to_json())
    x_scale = spec["encoding"]["x"]["scale"]
    assert x_scale.get("type") == "point"
    assert x_scale.get("range") == [40.0, 260.0], (
        f"PointScale range was not emitted in to_json(); got scale dict {x_scale!r}"
    )


# ---------------------------------------------------------------------------
# 2. Render-level: mark positions honor the requested sub-panel range
# ---------------------------------------------------------------------------


def test_band_scale_range_constrains_bar_positions():
    """Bars render within the explicit BandScale range, not the full panel.

    Regression: issue #39 — the dropped range field meant the positional
    resolver fell back to the full panel extent (~0-570px on a 600px-wide
    chart), spilling far outside the requested [40, 260] sub-range.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c", "d"], "val": [10.0, 20.0, 30.0, 40.0]})
    chart = (
        fr.Chart(df)
        .mark_bar()
        .encode(
            x=fr.X("cat", scale=fr.BandScale(domain=["a", "b", "c", "d"], range=[40.0, 260.0])),
            y="val",
        )
        .properties(width=600, height=400)
    )
    svg = chart.to_svg()
    bars = _data_bar_rects(svg)
    assert len(bars) == 4, f"Expected 4 data bars, got {len(bars)}: {bars}"

    tol = 0.5
    for attrs in bars:
        x0 = float(attrs["x"])
        x1 = x0 + float(attrs["width"])
        assert x0 >= 40.0 - tol and x1 <= 260.0 + tol, (
            f"Bar rect [{x0}, {x1}] falls outside BandScale range [40.0, 260.0]: {attrs!r}"
        )


def test_point_scale_range_constrains_point_positions():
    """Points render within the explicit PointScale range, not the full panel.

    Regression: issue #39 — same drop as BandScale but for PointScale, so
    mark_point circle centers spilled outside the requested sub-range.
    """
    df = pl.DataFrame({"cat": ["x", "y", "z"], "val": [5.0, 10.0, 15.0]})
    chart = (
        fr.Chart(df)
        .mark_point()
        .encode(
            x=fr.X("cat", scale=fr.PointScale(domain=["x", "y", "z"], range=[40.0, 260.0])),
            y="val",
        )
        .properties(width=600, height=400)
    )
    svg = chart.to_svg()
    cxs = _circle_cxs(svg)
    assert len(cxs) == 3, f"Expected 3 points, got {len(cxs)}: {cxs}"

    tol = 0.5
    for cx in cxs:
        assert 40.0 - tol <= cx <= 260.0 + tol, (
            f"Circle cx={cx} falls outside PointScale range [40.0, 260.0]: {cxs!r}"
        )


# ---------------------------------------------------------------------------
# 3. Fallback: no range specified -> no "range" key, panel-extent path unchanged
# ---------------------------------------------------------------------------


def test_band_scale_without_range_omits_range_key():
    """A BandScale built without range has no 'range' key in its wire dict.

    Guards the fallback-to-panel-extent path: this must remain untouched by
    the #39 fix. tests/test_scale_spec_parity.py::TestByteIdentity already
    freezes the full no-range band-chart to_json() output; this test only
    checks the narrower invariant that "range" is absent.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c", "d"], "val": [10.0, 20.0, 30.0, 40.0]})
    chart = (
        fr.Chart(df)
        .mark_bar()
        .encode(
            x=fr.X("cat", scale=fr.BandScale(domain=["a", "b", "c", "d"], padding=0.1)),
            y="val",
        )
    )
    spec = json.loads(chart.to_json())
    x_scale = spec["encoding"]["x"]["scale"]
    assert x_scale.get("type") == "band"
    assert "range" not in x_scale, (
        f"BandScale without range= should not emit a 'range' key; got {x_scale!r}"
    )
