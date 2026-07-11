"""Golden SVG tests for the secondary-y-axis / dual-axis feature (GH #52).

Covers acceptance criteria §9.1 and §9.3 of the secondary-y-axis design spec:

- A 2-layer ``LayerChart(bars, line, resolve={"y": "independent"})`` dual-axis
  render (left + right axes, both reserved margin bands, each layer on its
  own scale).
- A 3-layer independent-y chart stacking two right axes outward.
- Temporal y on one layer + numeric y on the other, exercising per-axis tick
  formatting.

These are byte-comparison goldens (mirroring ``tests/test_golden_configure.py``
conventions), not the behavioral SVG-parsing assertions in
``tests/test_secondary_y_axis.py``. Every chart here uses visually disjoint
y-domains and explicit axis titles so a broken dual-axis render (missing
right axis, marks on the wrong scale) is obvious in the rasterized PNG.

Run with ``FERRUM_UPDATE_GOLDENS=1`` to regenerate the on-disk goldens; the
default behavior is byte-identical comparison.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import polars as pl
import pytest

import ferrum as fm
from ferrum.composition import LayerChart

GOLDENS_DIR = Path(__file__).parent / "goldens" / "secondary_y_axis"
UPDATE = os.environ.get("FERRUM_UPDATE_GOLDENS") == "1"


def _check_or_update(name: str, svg: str) -> None:
    """Compare ``svg`` to the on-disk golden ``name``.

    With ``FERRUM_UPDATE_GOLDENS=1`` the on-disk golden is (re)written.
    Otherwise the function asserts byte-for-byte equality against the
    committed file. A missing golden is treated as an explicit test failure
    rather than a silent skip — goldens must be regenerated deliberately.
    """
    path = GOLDENS_DIR / name
    if UPDATE:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(svg)
        return
    if not path.exists():
        pytest.fail(
            f"golden {name!r} does not exist; rerun with FERRUM_UPDATE_GOLDENS=1 to generate it"
        )
    expected = path.read_text()
    from tests._snapshots import assert_svg_eq

    assert_svg_eq(
        svg,
        expected,
        name=name,
        regen_hint="rerun with FERRUM_UPDATE_GOLDENS=1 to refresh",
    )


class TestSecondaryYAxisGoldens:
    def test_two_layer_dual_axis(self):
        """Bars (revenue, $0-500) + line (margin, 18-42%) on disjoint scales."""
        df = pl.DataFrame(
            {
                "month": [1, 2, 3, 4, 5, 6],
                "revenue": [120.0, 180.0, 150.0, 260.0, 310.0, 480.0],
                "margin": [42.0, 38.0, 35.0, 30.0, 22.0, 18.0],
            }
        )
        bars = (
            fm.Chart(df)
            .mark_bar()
            .encode(
                x=fm.X("month:O", title="Month"),
                y=fm.Y("revenue", title="Revenue ($k)"),
            )
        )
        line = (
            fm.Chart(df)
            .mark_line(stroke="#e45756", stroke_width=3)
            .encode(
                x=fm.X("month:O"),
                y=fm.Y("margin", title="Margin (%)"),
            )
        )
        chart = LayerChart(
            bars, line, resolve={"y": "independent"}, title="Revenue vs. Margin (dual axis)"
        )
        _check_or_update("two_layer_dual_axis.svg", chart.to_svg())

    def test_three_layer_stacked_right_axes(self):
        """Temperature (left) + humidity + wind-chill delta stack two right axes."""
        df = pl.DataFrame(
            {
                "day": [1, 2, 3, 4, 5, 6, 7],
                "temperature": [12.0, 14.0, 15.0, 18.0, 20.0, 19.0, 17.0],
                "humidity": [70.0, 65.0, 60.0, 55.0, 50.0, 58.0, 62.0],
                "wind_speed": [-2.0, -5.0, -1.0, -8.0, -3.0, -6.0, -4.0],
            }
        )
        temp = (
            fm.Chart(df)
            .mark_bar()
            .encode(
                x=fm.X("day:O", title="Day"),
                y=fm.Y("temperature", title="Temperature (°C)"),
            )
        )
        humidity = (
            fm.Chart(df)
            .mark_line(stroke="#54a24b", stroke_width=3)
            .encode(
                x=fm.X("day:O"),
                y=fm.Y("humidity", title="Humidity (%)"),
            )
        )
        wind = (
            fm.Chart(df)
            .mark_point()
            .encode(
                x=fm.X("day:O"),
                y=fm.Y("wind_speed", title="Wind chill delta"),
            )
        )
        chart = LayerChart(
            temp,
            humidity,
            wind,
            resolve={"y": "independent"},
            title="Temperature / Humidity / Wind (3-axis)",
        )
        _check_or_update("three_layer_stacked.svg", chart.to_svg())

    def test_temporal_and_numeric_dual_axis(self):
        """Numeric score (left) + temporal milestone date (right) -- per-axis
        tick formatting: plain numbers on the left, month/year dates on the right."""
        df = pl.DataFrame(
            {
                "week": [1, 2, 3, 4, 5],
                "score": [55.0, 62.0, 58.0, 71.0, 80.0],
                "event_date": [
                    dt.date(2024, 1, 1),
                    dt.date(2024, 3, 15),
                    dt.date(2024, 5, 1),
                    dt.date(2024, 8, 20),
                    dt.date(2024, 11, 10),
                ],
            }
        )
        score = (
            fm.Chart(df)
            .mark_bar()
            .encode(
                x=fm.X("week:O", title="Week"),
                y=fm.Y("score", title="Score"),
            )
        )
        events = (
            fm.Chart(df)
            .mark_line(stroke="#f58518", stroke_width=3)
            .encode(
                x=fm.X("week:O"),
                y=fm.Y("event_date", title="Milestone date"),
            )
        )
        chart = LayerChart(
            score,
            events,
            resolve={"y": "independent"},
            title="Score vs. Milestone Date (dual axis, mixed tick formats)",
        )
        _check_or_update("temporal_numeric_dual_axis.svg", chart.to_svg())
