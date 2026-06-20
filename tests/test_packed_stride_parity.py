"""Packed-stride parity: Python mirror constants match Rust producer output.

``composition._PACKED_INSTANCE_SIZES = {0: 64, 1: 72}`` is a hand-synced copy
of ``ferrum-core/src/render/pack_instances.rs`` constants ``CIRCLE_STRIDE = 64``
and ``RECT_STRIDE = 72``.  No test previously verified these matched the bytes
the Rust producer actually emits.  This file is the regression gate.

## How stride is measured

We render a >1000-mark chart with ``_auto_tooltips=False`` so the packed buffer
has no tooltip section.  The Rust producer still emits data indices for a plain
chart (flags = 0x2, HAS_DATA_INDICES), so the total packed-buffer length is:

    len(packed) == 20 + count * stride + count * 4

where 20 is the fixed header size (5 × u32) and ``count * 4`` is the data-index
section.  Solving for stride:

    actual_stride = (len(packed) - 20 - count * 4) // count

This is computed entirely from the buffer and the header count field, with no
reference to ``_PACKED_INSTANCE_SIZES``.  The test then asserts:

    actual_stride == _PACKED_INSTANCE_SIZES[kind]

## Why this is a real discriminator

A deliberately-wrong constant (e.g. 60 for circles instead of 64) causes the
assertion to fail, because ``len(packed) - 20 == count * 64`` (as produced by
Rust) but ``_PACKED_INSTANCE_SIZES[0] == 60`` would claim 60.  The test
``test_wrong_stride_would_fail`` exercises this explicitly.

## X@byte0, Y@byte4 sanity check

Both ``CircleInstance`` and ``RectInstance`` lay out X as f32[0] (byte 0) and
Y as f32[1] (byte 4).  We parse these fields for every instance using the
measured stride and assert they fall within the chart's declared data range.
An off-by-one in the field offsets would produce out-of-range float garbage.
"""

from __future__ import annotations

import struct

import polars as pl
import pytest

import ferrum as fm
from ferrum._core import render_interactive
from ferrum.composition import _PACKED_INSTANCE_SIZES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HEADER_SIZE = 20  # 5 × u32 LE
_X_BYTE_OFFSET = 0  # f32[0] = byte 0
_Y_BYTE_OFFSET = 4  # f32[1] = byte 4

# Sentinel: no trailing data when flags == 0.
_FLAG_HAS_DATA_INDICES = 0x2
_FLAG_HAS_TOOLTIPS = 0x1


def _packed_no_tooltips(chart: fm.Chart) -> bytes:
    """Return packed bytes for *chart* with tooltips disabled (flags == 0).

    Uses ``_auto_tooltips=False`` so the Rust producer emits no trailing
    sections.  With a plain scatter/bar chart (no selections → no
    data_indices), flags == 0 and the packed buffer is exactly:
    ``20 + count * stride`` bytes per batch.
    """
    spec, data, viewport, theme_dict, chart_config_dict = chart._render_inputs(_auto_tooltips=False)
    _scene_json, packed = render_interactive(
        spec,
        data,
        viewport=viewport,
        theme=theme_dict,
        chart_config=chart_config_dict or None,
    )
    return packed


def _parse_single_batch_header(packed: bytes) -> tuple[int, int, int, int, int]:
    """Parse the 20-byte header of the first (and expected only) batch.

    Returns (panel_idx, batch_idx, kind, count, flags).
    Raises ``AssertionError`` if there are fewer than 20 bytes.
    """
    assert len(packed) >= _HEADER_SIZE, f"packed buffer too short for a header: {len(packed)} bytes"
    return struct.unpack_from("<5I", packed, 0)


def _measure_stride(packed: bytes) -> tuple[int, int, int]:
    """Derive stride from the packed buffer without referencing ``_PACKED_INSTANCE_SIZES``.

    Assumes exactly one batch and no tooltip section (flags & 0x1 == 0).  The
    batch may have a data-index section (flags & 0x2); its size is
    ``count * 4`` bytes.  The total packed-buffer length is therefore:

        len(packed) == 20 + count * stride + (count * 4 if HAS_DATA_INDICES)

    Solving for stride:

        instance_bytes = len(packed) - 20 - trailing_bytes
        stride = instance_bytes // count

    Returns (kind, count, measured_stride).
    """
    _panel_idx, _batch_idx, kind, count, flags = _parse_single_batch_header(packed)
    assert not (flags & _FLAG_HAS_TOOLTIPS), (
        f"tooltip section present (flags={flags:#x}); cannot measure stride without "
        "knowing tooltip section length.  Use _packed_no_tooltips()."
    )
    # Data-index section: count × u32 (4 bytes each), present when HAS_DATA_INDICES.
    data_index_bytes = count * 4 if (flags & _FLAG_HAS_DATA_INDICES) else 0
    instance_bytes = len(packed) - _HEADER_SIZE - data_index_bytes
    assert instance_bytes > 0, (
        f"no instance bytes left after accounting for header ({_HEADER_SIZE}) "
        f"and data-index section ({data_index_bytes}): total={len(packed)}"
    )
    assert instance_bytes % count == 0, (
        f"instance section ({instance_bytes} bytes) is not divisible by count ({count})"
    )
    return kind, count, instance_bytes // count


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def big_circle_chart() -> fm.Chart:
    """A >1000-mark point chart that triggers circle batch packing."""
    n = 1500
    df = pl.DataFrame({"x": [float(i) for i in range(n)], "y": [float(i % 50) for i in range(n)]})
    return fm.Chart(df).mark_point().encode(x="x", y="y")


@pytest.fixture
def big_rect_chart() -> fm.Chart:
    """A >1000-mark bar chart that triggers rect batch packing."""
    n = 1500
    df = pl.DataFrame(
        {"x": [float(i) for i in range(n)], "y": [float(i % 50 + 1) for i in range(n)]}
    )
    return fm.Chart(df).mark_bar().encode(x="x", y="y")


# ---------------------------------------------------------------------------
# Primary stride-parity tests
# ---------------------------------------------------------------------------


def test_circle_stride_matches_constant(big_circle_chart):
    """Kind-0 (circle) actual stride equals ``_PACKED_INSTANCE_SIZES[0]``."""
    packed = _packed_no_tooltips(big_circle_chart)
    assert packed, "a >1000-mark point chart must produce non-empty packed bytes"

    kind, count, measured = _measure_stride(packed)
    assert kind == 0, f"expected kind=0 (circle); got {kind}"
    assert count >= 1500, f"expected at least 1500 instances; got {count}"
    assert measured == _PACKED_INSTANCE_SIZES[0], (
        f"circle stride mismatch: Rust emits {measured} bytes/instance, "
        f"but _PACKED_INSTANCE_SIZES[0] == {_PACKED_INSTANCE_SIZES[0]}"
    )


def test_rect_stride_matches_constant(big_rect_chart):
    """Kind-1 (rect) actual stride equals ``_PACKED_INSTANCE_SIZES[1]``."""
    packed = _packed_no_tooltips(big_rect_chart)
    assert packed, "a >1000-mark bar chart must produce non-empty packed bytes"

    kind, count, measured = _measure_stride(packed)
    assert kind == 1, f"expected kind=1 (rect); got {kind}"
    assert count >= 1500, f"expected at least 1500 instances; got {count}"
    assert measured == _PACKED_INSTANCE_SIZES[1], (
        f"rect stride mismatch: Rust emits {measured} bytes/instance, "
        f"but _PACKED_INSTANCE_SIZES[1] == {_PACKED_INSTANCE_SIZES[1]}"
    )


# ---------------------------------------------------------------------------
# X@byte0, Y@byte4 field-offset sanity checks
# ---------------------------------------------------------------------------


def test_circle_x_at_byte0_y_at_byte4(big_circle_chart):
    """Circle X is f32[0] (byte 0) and Y is f32[1] (byte 4).

    Parse every instance using the correct stride and assert X and Y fall
    within the chart's data domain.  A wrong field offset would produce
    garbage floats outside this range.
    """
    packed = _packed_no_tooltips(big_circle_chart)
    _kind, count, stride = _measure_stride(packed)

    x_values: list[float] = []
    y_values: list[float] = []
    for i in range(count):
        base = _HEADER_SIZE + i * stride
        (x,) = struct.unpack_from("<f", packed, base + _X_BYTE_OFFSET)
        (y,) = struct.unpack_from("<f", packed, base + _Y_BYTE_OFFSET)
        x_values.append(x)
        y_values.append(y)

    # Data: x in [0, 1499], y in [0, 49] (i % 50).
    # Use a generous margin for plot-area scaling.
    assert min(x_values) >= -10.0, f"circle X@byte0 has implausible minimum: {min(x_values)}"
    assert max(x_values) <= 800.0, f"circle X@byte0 has implausible maximum: {max(x_values)}"
    assert min(y_values) >= -10.0, f"circle Y@byte4 has implausible minimum: {min(y_values)}"
    assert max(y_values) <= 600.0, f"circle Y@byte4 has implausible maximum: {max(y_values)}"


def test_rect_x_at_byte0_y_at_byte4(big_rect_chart):
    """Rect X is f32[0] (byte 0) and Y is f32[1] (byte 4).

    Parse every instance using the correct stride and assert X and Y fall
    within a plausible pixel range for a 600×400 viewport.
    """
    packed = _packed_no_tooltips(big_rect_chart)
    _kind, count, stride = _measure_stride(packed)

    x_values: list[float] = []
    y_values: list[float] = []
    for i in range(count):
        base = _HEADER_SIZE + i * stride
        (x,) = struct.unpack_from("<f", packed, base + _X_BYTE_OFFSET)
        (y,) = struct.unpack_from("<f", packed, base + _Y_BYTE_OFFSET)
        x_values.append(x)
        y_values.append(y)

    # Rect X/Y are top-left corner of the bar.  Y grows downward in SVG
    # coordinates; bars of height ≥ 1 have Y in a sensible pixel range.
    assert min(x_values) >= -10.0, f"rect X@byte0 has implausible minimum: {min(x_values)}"
    assert max(x_values) <= 800.0, f"rect X@byte0 has implausible maximum: {max(x_values)}"
    assert min(y_values) >= -10.0, f"rect Y@byte4 has implausible minimum: {min(y_values)}"
    assert max(y_values) <= 600.0, f"rect Y@byte4 has implausible maximum: {max(y_values)}"


# ---------------------------------------------------------------------------
# Real-discriminator verification
# ---------------------------------------------------------------------------


def test_wrong_stride_would_fail(big_circle_chart):
    """Confirm the stride assertion is a real discriminator.

    A wrong stride constant (e.g. 60 or 68 instead of 64) must not satisfy
    ``measured == wrong_stride``, proving the test catches drift.

    The correct stride (64) is measured from the buffer; all wrong values
    differ from it.  This test documents the discriminator property rather
    than temporarily patching ``_PACKED_INSTANCE_SIZES``.
    """
    packed = _packed_no_tooltips(big_circle_chart)
    _kind, _count, measured_circle_stride = _measure_stride(packed)

    # Measured stride is the ground truth from Rust output.
    assert measured_circle_stride == 64, (
        f"Rust circle stride is not 64: {measured_circle_stride}. "
        "This test's discriminator reasoning is invalid."
    )

    # A wrong constant does NOT equal the measured value.
    wrong_strides = [60, 62, 68, 72, 32]
    for wrong in wrong_strides:
        assert wrong != measured_circle_stride, (
            f"Wrong stride {wrong} accidentally equals the measured stride {measured_circle_stride}; "
            "the discriminator is broken."
        )

    # To make it explicit: if _PACKED_INSTANCE_SIZES[0] were wrong, this is
    # what would fail in test_circle_stride_matches_constant:
    #
    #   assert measured_circle_stride == _PACKED_INSTANCE_SIZES[0]
    #   → assert 64 == 60   # FAILS if constant is wrong
    #
    # The assertion that follows verifies the current constant is correct:
    assert measured_circle_stride == _PACKED_INSTANCE_SIZES[0], (
        f"_PACKED_INSTANCE_SIZES[0] drifted from Rust: "
        f"Rust emits {measured_circle_stride}, constant says {_PACKED_INSTANCE_SIZES[0]}"
    )


def test_wrong_rect_stride_would_fail(big_rect_chart):
    """Confirm the rect stride assertion is a real discriminator.

    A wrong stride constant (e.g. 64 or 76 instead of 72) must not satisfy
    ``measured == wrong_stride``.
    """
    packed = _packed_no_tooltips(big_rect_chart)
    _kind, _count, measured_rect_stride = _measure_stride(packed)

    assert measured_rect_stride == 72, (
        f"Rust rect stride is not 72: {measured_rect_stride}. "
        "This test's discriminator reasoning is invalid."
    )

    wrong_strides = [64, 68, 76, 80, 32]
    for wrong in wrong_strides:
        assert wrong != measured_rect_stride, (
            f"Wrong stride {wrong} accidentally equals the measured stride {measured_rect_stride}; "
            "the discriminator is broken."
        )

    # If _PACKED_INSTANCE_SIZES[1] were wrong, test_rect_stride_matches_constant would fail:
    #
    #   assert measured_rect_stride == _PACKED_INSTANCE_SIZES[1]
    #   → assert 72 == 64   # FAILS if constant is wrong
    assert measured_rect_stride == _PACKED_INSTANCE_SIZES[1], (
        f"_PACKED_INSTANCE_SIZES[1] drifted from Rust: "
        f"Rust emits {measured_rect_stride}, constant says {_PACKED_INSTANCE_SIZES[1]}"
    )
