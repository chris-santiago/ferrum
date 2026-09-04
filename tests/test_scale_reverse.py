"""Feature tests for ``reverse=`` on the six continuous scale classes (F-L04-07).

Batch-C task 2's Python half. The Rust half (task 2's rust-coder) added
``reverse: bool = False`` to ``LinearScale``/``LogScale``/``PowScale``/
``SqrtScale``/``SymlogScale``/``TimeScale`` and threaded it into
``to_scale_spec()``; the resolver-side domain swap
(``apply_domain_reverse`` in ``render::scale_resolve::positional``) and its
Rust-side unit + parity tests already prove the wire contract. This module
closes the batch-C T1 spec reviewer's note that a prior Rust y-channel test
*simulated* the resolved range instead of proving it through a real render:
every test here goes through ``Chart(...).to_svg()`` and inspects the
rendered SVG (mark ``<circle>`` coordinates and ``<text>`` tick labels)
rather than asserting on any intermediate Rust struct.

Covers, per §4C/§9 of the batch-C design spec:
  1. For each of the six classes: ``reverse=True`` flips both the rendered
     mark coordinates and the axis tick-label order relative to
     ``reverse=False`` (exact-order assertions, not a mere "differs" check).
  2. Domain-swap equivalence: an explicit-domain ``reverse=True`` scale
     renders byte-identical SVG to the hand-written swapped-domain
     equivalent (``domain=[hi, lo]``), the scoped equivalence per §4C.
  3. A ``y``-channel case (LinearScale, representative of the family since
     the swap logic lives once at the resolver's shared chokepoint) proving
     marks and y tick labels flip together through the real render path.
  4. A raw-dict case (``scale={"type": "linear", "reverse": True}``)
     matching the class spelling byte-for-byte.

RED-proof note (discriminating by construction, not a toggled runtime
check): before this batch's Rust half landed, none of the six classes
accepted a ``reverse`` keyword at all — ``LinearScale(reverse=True)`` (and
its siblings) raised ``TypeError: unexpected keyword argument 'reverse'``
at construction, before any render happened. Every test below constructs
with ``reverse=True`` as its very first step, so the whole module is
non-vacuously RED against any pre-fix build by construction; the
proximate assertions additionally pin the *exact* flipped tick-label and
mark-coordinate orders, so a hypothetical future regression that silently
drops the flag (rather than refusing construction) is still caught.
"""

from __future__ import annotations

import re

import polars as pl
import pytest

import ferrum as fm
import ferrum._core as fc
from tests._svg_extents import axis_tick_labels as _axis_tick_labels

# ---------------------------------------------------------------------------
# SVG parsing helpers
#
# The generic ``<text>``/``<circle>`` parsing (tick-label grouping in screen
# order, raw string) lives in ``tests/_svg_extents.axis_tick_labels`` — see
# that module's docstring for why it's the raw-string sibling of
# ``numeric_text_entries``/``x_axis_extents``/``y_axis_extents`` rather than a
# reuse of those (they discard both non-numeric text and screen order, which
# ``TimeScale``'s "HH:MM:SS" labels and this module's ordering assertions both
# need). Only the feature-specific helpers below (mark-point flip detection,
# tick-value parsing) are local to this module.
# ---------------------------------------------------------------------------


def _circle_points(svg: str) -> list[tuple[float, float]]:
    """Return ``(cx, cy)`` for every ``<circle>`` element, in document order."""
    return [
        (float(cx), float(cy))
        for cx, cy in re.findall(r'<circle[^>]*cx="([^"]+)"[^>]*cy="([^"]+)"', svg)
    ]


def _parse_tick_value(text: str) -> float:
    """Parse a rendered tick label into a comparable float.

    Numeric ticks (Linear/Log/Pow/Sqrt/Symlog) parse directly.
    ``TimeScale`` renders "HH:MM:SS"-formatted labels for a sub-hour
    epoch-float domain; those convert to seconds-since-midnight so one
    ordering assertion works across all six classes.
    """
    try:
        return float(text)
    except ValueError:
        pass
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError(f"tick label {text!r} is neither a float nor an 'HH:MM:SS' time string")
    try:
        hours, minutes, seconds = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"tick label {text!r} could not be parsed as 'HH:MM:SS': {exc}") from exc
    return hours * 3600 + minutes * 60 + seconds


def _lo_hi_cx_by_row(svg: str) -> tuple[float, float]:
    """Return ``(cx of the domain-lo row, cx of the domain-hi row)`` for an x-channel fixture.

    Row identity comes from ``cy``: the x-channel fixtures below always pair
    the domain-lo x-value with ``y=0.0`` and the domain-hi x-value with
    ``y=100.0`` on an un-reversed y-scale, so the row rendered lower on
    screen (the larger ``cy``) is always the domain-lo row.
    """
    points = _circle_points(svg)
    assert len(points) == 2, f"expected exactly 2 marks, got {points}"
    points.sort(key=lambda p: p[1])  # ascending cy: hi-row (smaller cy) first
    (hi_cx, _hi_cy), (lo_cx, _lo_cy) = points
    return lo_cx, hi_cx


def _lo_hi_cy_by_row(svg: str) -> tuple[float, float]:
    """Return ``(cy of the domain-lo row, cy of the domain-hi row)`` for a y-channel fixture.

    Row identity comes from ``cx``: the y-channel fixture below always pairs
    the domain-lo y-value with ``x=0.0`` and the domain-hi y-value with
    ``x=10.0`` on an un-reversed x-scale, so the row rendered further left
    (the smaller ``cx``) is always the domain-lo row.
    """
    points = _circle_points(svg)
    assert len(points) == 2, f"expected exactly 2 marks, got {points}"
    points.sort(key=lambda p: p[0])  # ascending cx: lo-row (smaller cx) first
    (lo_cx, lo_cy), (hi_cx, hi_cy) = points
    return lo_cy, hi_cy


# ---------------------------------------------------------------------------
# Chart-building helpers
# ---------------------------------------------------------------------------


def _svg_x_channel(
    scale_cls: type,
    scale_domain: tuple[float, float],
    *,
    reverse: bool,
    data_x: tuple[float, float] | None = None,
) -> str:
    """Render a 2-point chart with ``scale_cls`` on the ``x`` channel.

    Data defaults to ``x=scale_domain`` (one mark at each domain endpoint) so
    every scale type sees in-domain values, and ``y=[0.0, 100.0]`` always —
    the row identity ``_lo_hi_cx_by_row`` relies on. ``data_x`` lets the
    domain-swap equivalence test hold the data fixed while the scale's own
    ``domain=``/``reverse=`` vary.
    """
    x_values = data_x if data_x is not None else scale_domain
    df = pl.DataFrame({"x": list(x_values), "y": [0.0, 100.0]})
    scale = scale_cls(domain=list(scale_domain), nice=False, reverse=reverse)
    chart = fm.Chart(df).mark_point().encode(x=fm.X("x", scale=scale), y="y")
    return chart.to_svg()


def _svg_y_channel(scale_domain: tuple[float, float], *, reverse: bool) -> str:
    """Render a 2-point chart with ``LinearScale`` on the ``y`` channel.

    Data is always ``x=[0.0, 10.0], y=[0.0, 100.0]``, the mirror of
    ``_svg_x_channel`` with the reversed channel swapped.
    """
    df = pl.DataFrame({"x": [0.0, 10.0], "y": [0.0, 100.0]})
    scale = fc.LinearScale(domain=list(scale_domain), nice=False, reverse=reverse)
    chart = fm.Chart(df).mark_point().encode(x="x", y=fm.Y("y", scale=scale))
    return chart.to_svg()


# ---------------------------------------------------------------------------
# Per-class domains. Chosen so each scale type accepts the domain natively:
# Log requires strictly positive bounds, Symlog exercises a sign-crossing
# domain, the rest are plain positive ranges. TimeScale uses an explicit
# epoch-float domain (datetime-domain acceptance is Task 3, not this one).
# ---------------------------------------------------------------------------

_CONTINUOUS_REVERSE_CASES: list[tuple[str, type, tuple[float, float]]] = [
    ("LinearScale", fc.LinearScale, (0.0, 10.0)),
    ("LogScale", fc.LogScale, (1.0, 100.0)),
    ("PowScale", fc.PowScale, (1.0, 10.0)),
    ("SqrtScale", fc.SqrtScale, (1.0, 10.0)),
    ("SymlogScale", fc.SymlogScale, (-10.0, 10.0)),
    ("TimeScale", fc.TimeScale, (0.0, 1_000_000.0)),
]
_CASE_IDS = [name for name, _cls, _domain in _CONTINUOUS_REVERSE_CASES]


# ---------------------------------------------------------------------------
# 1. Per-class: reverse=True flips marks AND axis tick-label order
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name, scale_cls, domain", _CONTINUOUS_REVERSE_CASES, ids=_CASE_IDS)
def test_axis_tick_labels_flip_order(
    name: str, scale_cls: type, domain: tuple[float, float]
) -> None:
    """``reverse=True``'s x-axis tick labels are the descending mirror of ``reverse=False``'s."""
    svg_forward = _svg_x_channel(scale_cls, domain, reverse=False)
    svg_reversed = _svg_x_channel(scale_cls, domain, reverse=True)

    labels_forward = [_parse_tick_value(t) for t in _axis_tick_labels(svg_forward, axis="x")]
    labels_reversed = [_parse_tick_value(t) for t in _axis_tick_labels(svg_reversed, axis="x")]

    assert len(labels_forward) >= 3, f"{name}: too few forward tick labels to be discriminating"
    assert len(labels_reversed) >= 3, f"{name}: too few reversed tick labels to be discriminating"
    assert labels_forward == sorted(labels_forward), (
        f"{name}: forward ticks must ascend left-to-right"
    )
    assert labels_reversed == sorted(labels_reversed, reverse=True), (
        f"{name}: reversed ticks must descend left-to-right"
    )
    assert labels_forward != labels_reversed, (
        f"{name}: reverse=True must actually change tick order"
    )


@pytest.mark.parametrize("name, scale_cls, domain", _CONTINUOUS_REVERSE_CASES, ids=_CASE_IDS)
def test_mark_coordinates_flip(name: str, scale_cls: type, domain: tuple[float, float]) -> None:
    """``reverse=True`` swaps which side of the plot the domain-lo mark renders on."""
    svg_forward = _svg_x_channel(scale_cls, domain, reverse=False)
    svg_reversed = _svg_x_channel(scale_cls, domain, reverse=True)

    lo_cx_forward, hi_cx_forward = _lo_hi_cx_by_row(svg_forward)
    lo_cx_reversed, hi_cx_reversed = _lo_hi_cx_by_row(svg_reversed)

    assert lo_cx_forward < hi_cx_forward, (
        f"{name}: forward domain-lo mark must render left of domain-hi"
    )
    assert lo_cx_reversed > hi_cx_reversed, (
        f"{name}: reversed domain-lo mark must render right of domain-hi"
    )


# ---------------------------------------------------------------------------
# 2. Domain-swap equivalence: reverse=True + explicit domain == hand-swapped domain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name, scale_cls, domain", _CONTINUOUS_REVERSE_CASES, ids=_CASE_IDS)
def test_reverse_equals_manual_domain_swap(
    name: str, scale_cls: type, domain: tuple[float, float]
) -> None:
    """``domain=[lo, hi], reverse=True`` renders byte-identical SVG to ``domain=[hi, lo]``.

    Scoped equivalence per §4C: both domain endpoints are explicit (zero
    unset), so the swap is a pure reordering with no auto-inference or
    padding to complicate the comparison.
    """
    lo, hi = domain
    svg_reverse = _svg_x_channel(scale_cls, (lo, hi), reverse=True, data_x=(lo, hi))
    svg_manual_swap = _svg_x_channel(scale_cls, (hi, lo), reverse=False, data_x=(lo, hi))

    assert svg_reverse == svg_manual_swap, (
        f"{name}: reverse=True must byte-match the hand-swapped domain"
    )


# ---------------------------------------------------------------------------
# 3. y-channel case: marks and y tick labels flip together
# ---------------------------------------------------------------------------


class TestYChannelReverseFlipsMarksAndAxisLabels:
    """Proves the domain swap through the real render path on the ``y`` channel.

    The x-channel sweep above proves per-class wiring into
    ``to_scale_spec()``; the swap itself resolves once, for every channel and
    every continuous class, at the resolver's shared chokepoint
    (``apply_domain_reverse``). ``LinearScale`` stands in for the family here
    — this is the test that closes the T1 spec reviewer's note that a prior
    Rust y-test simulated the resolved range instead of rendering real SVG.
    """

    def test_axis_tick_labels_flip_order(self) -> None:
        svg_forward = _svg_y_channel((0.0, 100.0), reverse=False)
        svg_reversed = _svg_y_channel((0.0, 100.0), reverse=True)

        labels_forward = [_parse_tick_value(t) for t in _axis_tick_labels(svg_forward, axis="y")]
        labels_reversed = [_parse_tick_value(t) for t in _axis_tick_labels(svg_reversed, axis="y")]

        assert len(labels_forward) >= 3
        assert len(labels_reversed) >= 3
        # Un-reversed y renders descending top-to-bottom (largest value at top);
        # reverse=True flips that to ascending top-to-bottom.
        assert labels_forward == sorted(labels_forward, reverse=True)
        assert labels_reversed == sorted(labels_reversed)
        assert labels_forward != labels_reversed

    def test_mark_coordinates_flip(self) -> None:
        svg_forward = _svg_y_channel((0.0, 100.0), reverse=False)
        svg_reversed = _svg_y_channel((0.0, 100.0), reverse=True)

        lo_cy_forward, hi_cy_forward = _lo_hi_cy_by_row(svg_forward)
        lo_cy_reversed, hi_cy_reversed = _lo_hi_cy_by_row(svg_reversed)

        assert lo_cy_forward > hi_cy_forward, "forward: y=0 mark must render below the y=100 mark"
        assert lo_cy_reversed < hi_cy_reversed, "reverse: y=0 mark must render above the y=100 mark"


# ---------------------------------------------------------------------------
# 4. Raw-dict spelling matches the class spelling
# ---------------------------------------------------------------------------


def test_raw_dict_reverse_matches_class_spelling() -> None:
    """``scale={"type": "linear", "reverse": True}`` renders byte-identical to the class spelling.

    Byte-equality alone is not discriminating on its own: a mutation that
    silently dropped ``reverse`` on *both* the class path and the dict path
    would keep the two SVGs equal and pass this assertion vacuously (it only
    catches dict-path-*specific* drops, e.g. a serde rename or a gate strip).
    So this test also pins the dict-path SVG's own tick-label order directly
    — the same discriminating assertion the per-class sweep above uses — which
    fails under a both-paths drop (the ticks would ascend, not descend).
    """
    df = pl.DataFrame({"x": [0.0, 10.0], "y": [0.0, 100.0]})

    chart_class = (
        fm.Chart(df)
        .mark_point()
        .encode(
            x=fm.X("x", scale=fc.LinearScale(domain=[0.0, 10.0], nice=False, reverse=True)),
            y="y",
        )
    )
    chart_dict = (
        fm.Chart(df)
        .mark_point()
        .encode(
            x=fm.X(
                "x",
                scale={"type": "linear", "domain": [0.0, 10.0], "nice": False, "reverse": True},
            ),
            y="y",
        )
    )

    svg_class = chart_class.to_svg()
    svg_dict = chart_dict.to_svg()

    # Path parity: the raw-dict spelling renders exactly what the class does.
    assert svg_class == svg_dict

    # Standalone discrimination: the dict-path SVG itself must show the
    # descending tick-label order reverse=True produces, independent of the
    # class-path comparison above.
    dict_labels = [_parse_tick_value(t) for t in _axis_tick_labels(svg_dict, axis="x")]
    assert len(dict_labels) >= 3
    assert dict_labels == sorted(dict_labels, reverse=True)
