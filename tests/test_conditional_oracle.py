"""Negative-control self-tests for the conditional pixel harness's pure oracle logic.

Plain tests: no ``wasm_pixels`` marker, no browser, no fixture — these run
in every default suite. Several of the harness's pixel-classification
checks (dash alternation, dash solidity, opacity blend, size radius) can be
made non-discriminating by a threshold loose enough to also pass a broken
renderer. Each helper's positive case must pass here, and each helper's
negative (synthetic, hand-built) case — the exact shape a broken renderer
would produce — must fail, proving the check actually distinguishes real
behavior from that broken shape before this harness is trusted to gate
anything.
"""

from __future__ import annotations

import math

import pytest

from tests._conditional_oracle import (
    assert_radius_close,
    assert_rgb_close,
    predicted_fill_edge,
    ring_is_alternating,
    ring_is_solid,
    ring_sample_count,
    synthetic_dash_classes,
)

# Mirrors tests/test_conditional_pixels.py's BASE_MARK_SIZE / DASH_STROKE_W —
# duplicated as plain numbers (not imported) so this module stays free of any
# dependency on the wasm_pixels-marked module or a browser.
_POINT_MARK_SIZE = 452.0
_DASH_STROKE_W = 6.0


def test_ring_alternating_rejects_a_ring_edge_artifact() -> None:
    """A single contiguous arc (a band boundary crossing the sample radius) is not a dash.

    Sampling a circular dash whose phase depends on radial distance (not
    perimeter arc length) produces one contiguous arc of "stroke" samples
    at the sample radius, not an alternating pattern — a `min_frac`-only
    check (counting how many samples are "stroke" with no transition
    count) would accept this as "mostly stroke" and call it solid or
    alternating without ever checking the samples' spatial arrangement.
    """
    ring_edge = ["."] * 8 + ["S"] * 8
    assert not ring_is_alternating(ring_edge)


def test_ring_alternating_rejects_a_ring_edge_with_an_antialiasing_blip() -> None:
    """A single contiguous arc with one AA-noise pixel flipped at its own edge.

    The exact shape a 32-sample capture of this bug produced on the
    committed bundle: one contiguous run of stroke samples, with an
    isolated single-sample gap right at the run's own edge. A transition
    counter with no denoising sees 4 transitions here and would wrongly
    call it alternating.
    """
    ring_edge_with_blip = list("..SS.SSSSS......")
    assert not ring_is_alternating(ring_edge_with_blip)


def test_ring_alternating_rejects_a_wide_noisy_fringe_around_one_big_arc() -> None:
    """A real 128-sample capture of a radial-phase ring-edge artifact, denser than 16.

    Captured directly on a bundle with a radial (rather than perimeter)
    dash phase (`stroke_dash` `palette` on a point mark, click driver): the
    harness's fixed sampling radius happens to sit close to the shader's
    own phase-wrap boundary, so each of the arc's two true edges produces a
    *multi*-sample noisy fringe, not `denoise_ring`'s single-sample blip.
    6 transitions survive single-sample denoising alone — past
    `min_transitions=4` — while the ring is still fundamentally one big
    arc (its longest run is ~48% of the ring). Pins the exact captured
    shape so a future change to `ring_is_alternating` cannot silently
    regress into accepting this again.
    """
    captured = list(
        ".......S..S...SSSS.S.S.SS.SSSSSS.SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS."
        "SSSSSS.......S........................................"
    )
    assert not ring_is_alternating(captured)


def test_ring_alternating_accepts_a_genuine_multi_dash_pattern() -> None:
    # Each on/off run is 2 samples wide, not 1 — denoising a single-sample
    # anti-aliasing blip should never collapse a real (>=2-sample) dash run.
    dashed = ["S", "S", ".", "."] * 4
    assert ring_is_alternating(dashed)


def test_ring_alternating_accepts_the_real_dash_geometry_at_128_samples() -> None:
    """Positive control built from the real `[6, 3]` pattern and real ring geometry.

    Proof that `ring_is_alternating` can see a genuine short-gap dash at
    the harness's actual sampling density (128, this mark's own floor —
    see `ring_sample_count`) and ring radius, given a perimeter-phased
    dash to sample. If this ever regresses to `False`, the 128-sample
    floor (or the ring radius it is computed against) is too coarse for
    the shortest real segment on this mark.
    """
    r = math.sqrt(_POINT_MARK_SIZE / math.pi)
    ring_r = r - _DASH_STROKE_W / 2
    circumference = 2 * math.pi * ring_r
    classes = synthetic_dash_classes(circumference, [6, 3], n=128)
    assert ring_is_alternating(classes)


def test_ring_solid_rejects_an_unpainted_ring() -> None:
    """The stroke never painted at all is neither alternating nor solid.

    `not ring_is_alternating(...)` alone would accept this (an
    all-background ring is trivially "not alternating"), which is not the
    same claim as "solidly stroked" — an unpainted ring must be positively
    rejected by `ring_is_solid` too, not just fail to match "alternating".
    """
    unpainted = ["."] * 16
    assert not ring_is_solid(unpainted)
    assert not ring_is_alternating(unpainted)


def test_ring_solid_accepts_a_fully_painted_ring() -> None:
    solid = ["S"] * 16
    assert ring_is_solid(solid)


def test_opacity_tolerance_rejects_a_blend_off_by_three_units() -> None:
    with pytest.raises(AssertionError):
        assert_rgb_close((200, 200, 200), (197, 197, 197), tol=2, label="test")


def test_opacity_tolerance_accepts_a_blend_within_tolerance() -> None:
    assert_rgb_close((198, 198, 198), (197, 197, 197), tol=2, label="test")


def test_size_tolerance_rejects_a_radius_off_by_more_than_tolerance() -> None:
    with pytest.raises(AssertionError):
        assert_radius_close(10.0, predicted_fill_edge("point", 900.0), tol=3.0, label="test")


def test_size_tolerance_accepts_a_radius_within_tolerance() -> None:
    predicted = predicted_fill_edge("point", 900.0)
    assert_radius_close(predicted - 1.0, predicted, tol=3.0, label="test")


def test_predicted_fill_edge_formula_per_mark_kind() -> None:
    # Circle: an area-to-radius conversion, then the stroke inset inward.
    assert predicted_fill_edge("point", 900.0, stroke_width=3.0) == pytest.approx(
        math.sqrt(900.0 / math.pi) - 3.0
    )
    # Rect: the raw pixel value itself, inset by the stroke — never value/2
    # (the rect grows right/down from its own fixed top-left corner).
    assert predicted_fill_edge("rect", 900.0, stroke_width=3.0) == pytest.approx(897.0)


# The exact geometry of the row a fixed 128-sample ring under-resolved:
# `stroke_dash-palette-rect-unpacked` (perimeter 959.608 SVG px, `[6, 3]`
# pattern, shortest segment 3).
_UNDER_SAMPLED_PERIMETER = 959.608
_UNDER_SAMPLED_MIN_SEGMENT = 3.0
_OLD_FIXED_RING_SAMPLES = 128

# A real 128-sample capture of that exact row's stroke ring (`classify_ring`
# against the committed bundle's rendered frame, click driver) — not a
# hand-built shape. The `[6, 3]` pattern's 3px gap falls under the fixed
# 128-sample stride, so consecutive samples alias into a beat pattern with
# a few wide runs rather than many short, even ones.
_UNDER_SAMPLED_CAPTURE_128 = (
    "SS.SS.SSS.SS.SS.SS.SS.SS.SSS.SSS.SS.SS.SS.S.SS.SS.SS.SS.SS.SS.S.S.SS.SS.SSS.SS.SS.SS.SS.SS"
    "..SS..SS.SS.SS.SS.SS.SSSS.SS.SS.SS.SS."
)

# The same rendered frame, resampled at `ring_sample_count`'s own answer for
# this row's (perimeter, shortest segment) — 1280 samples, four per 3px gap.
_UNDER_SAMPLED_CAPTURE_AT_COMPUTED_N = (
    "...SSSSSSSSSSSSSSSSSSS...........SSSSSSSSSSSSSSSSSSSSS..........SSSSSSSSSSSSSSSSSSSSS......."
    "....SSSSSSSSSSSSSSSSSSSSS...........SSSSSSSSSSSSSSSSSSSSS..........SSSSSSSSSSSSSSSSSSSSSS..."
    ".......SSSSSSSSSSSSSSSSSSSSS...........SSSSSSSSSSSSSSSSSSSSS...........SSSSSSSSSSSSSSSSSSSSS"
    "..........SSSSSSSSSSSSSSSSSSSSS..SSSSSSSSSSS..SS..SSSSS...SSSSS..SSSSS...SSSSS..SSSSS..SSSSS"
    "...SSSSS..SSSSS...SSSSS..SSSSS..SSSSS...SSSSS..SSSSS...SSSSS..SSSSS..SSSSS...SSSSS..SSSSS..."
    "SSSSS..SSSSS..SSSSS...SSSSS..SSSSS...SSSSS..SSSSS..SSSSS...SSSSS..SSSSS...SSSSS..SSSSS...SSS"
    "S...SSSSS..SSSSS...SSSSS..SSSSS...SSSS...SSSSS..SSSSS...SSSSS..SSSSS...SSSS...SSSSS..SS.SSSS"
    "SSSSSSS..........SSSSSSSSSSSSSSSSSSSSSS..........SSSSSSSSSSSSSSSSSSSSS...........SSSSSSSSSSS"
    "SSSSSSSSSS...........SSSSSSSSSSSSSSSSSSSSS..........SSSSSSSSSSSSSSSSSSSSS...........SSSSSSSS"
    "SSSSSSSSSSSSS...........SSSSSSSSSSSSSSSSSSSSS..........SSSSSSSSSSSSSSSSSSSSSS..........SSSSS"
    "SSSSSSSSSSSSSSSS...........SSS...........SS..SSSSS..SSSSS...SSSS...SSSSS..SSSSS...SSSSS..SSS"
    "SS...SSSS...SSSSS..SSSSS...SSSSS..SSSSS...SSSS...SSSSS..SSSSS...SSSSS..SSSSS...SSSSS..SSSSS."
    ".SSSSS...SSSSS..SSSSS...SSSSS..SSSSS..SSSSS...SSSSS..SSSSS...SSSSS..SSSSS..SSSSS...SSSSS..SS"
    "SSS...SSSSS..SSSSS..SSSSS...SSSSS..SSSSS...SSSSS..SSSSS..SSSSS...SSSSS..SSSSS...SSSS"
)


def test_ring_sample_count_matches_the_real_geometry() -> None:
    """`ring_sample_count`'s answer for this row is the density the two captures below were taken at."""
    n = ring_sample_count(_UNDER_SAMPLED_PERIMETER, _UNDER_SAMPLED_MIN_SEGMENT)
    assert n == len(_UNDER_SAMPLED_CAPTURE_AT_COMPUTED_N)


def test_ring_is_alternating_rejects_the_real_under_sampled_capture() -> None:
    """The self-test that would have caught the under-sampling before it shipped.

    A real 128-sample capture of `stroke_dash-palette-rect-unpacked`'s
    stroke ring, at the exact density the old fixed floor used, reads as
    not alternating — the aliasing `ring_sample_count` now fixes.
    """
    assert len(_UNDER_SAMPLED_CAPTURE_128) == _OLD_FIXED_RING_SAMPLES
    assert not ring_is_alternating(list(_UNDER_SAMPLED_CAPTURE_128))


def test_ring_is_alternating_accepts_the_capture_at_the_computed_n() -> None:
    """The same rendered frame, resampled at `ring_sample_count`'s own density, reads correctly."""
    assert ring_is_alternating(list(_UNDER_SAMPLED_CAPTURE_AT_COMPUTED_N))


def test_ring_sample_count_floors_a_solid_pattern() -> None:
    assert ring_sample_count(_UNDER_SAMPLED_PERIMETER, None) == _OLD_FIXED_RING_SAMPLES


def test_ring_sample_count_raises_rather_than_silently_undersampling() -> None:
    """A density need past the cap fails loudly, naming the perimeter and the count it would take.

    A silent clamp here could under-sample a genuinely alternating pattern
    into misreading as solid; a big enough perimeter against a short enough
    segment must refuse instead.
    """
    with pytest.raises(ValueError, match=r"perimeter 100000\.0.*1\.0px.*400000 samples"):
        ring_sample_count(100_000.0, 1.0)
