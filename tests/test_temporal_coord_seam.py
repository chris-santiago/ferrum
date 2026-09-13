"""The one temporal→epoch-ms seam, and the three consumers that speak it.

``ferrum.annotation.coords.temporal_coord_to_epoch_ms`` is the canonical
Python-side conversion from a temporal value to epoch-milliseconds UTC. Three
consumers call it, and a design review (wave-2 break-axis, F1) found them
speaking three different vocabularies: ``BreakAxis.gap`` had grown its own
private ``numpy.datetime64`` branch (epoch table, overflow proof, NaT refusal)
at the leaf, so a ``datetime64`` bound constructed a break while the *scale
domain of the same axis* refused it, and ``pd.NaT`` returned ``nan`` from the
seam itself and only died two layers down as a leaked JSON diagnostic.

This module is the seam's own test home. It exists as a separate file from
``test_structural.py`` (which pins ``BreakAxis`` construction) and
``test_annotations.py`` (which pins annotation coercion) for the reason
``tests/test_boxen_palette.py``'s docstring states as the convention: net-new
coverage of a shared mechanism gets its own module rather than accreting into
whichever consumer's file the work happened to start in. What is pinned here
is the mechanism and the *agreement* of its consumers:

* the seam accepts ``numpy.datetime64`` and agrees with the ``pd.Timestamp`` /
  ISO-string spellings of the same instant;
* the seam refuses ``pd.NaT`` / ``np.datetime64('NaT')`` by name, and each
  consumer surfaces that refusal in its own vocabulary;
* all three consumers (``BreakAxis.gap``, the encoding layer's TimeScale
  domain coercion, the annotation coordinate serializer) return the same
  epoch-ms for the same ``datetime64`` value;
* the ``datetime64`` overflow boundary constant is pinned at its exact
  threshold on both signs (mutation-tester advisory: the branch was covered
  by values many orders of magnitude past the boundary, so a mutation that
  merely *widened* the constant survived).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ferrum.annotation.coords import temporal_coord_to_epoch_ms

_EPOCH_MS_2020_03_01 = 1583020800000.0


# ---------------------------------------------------------------------------
# the seam accepts numpy.datetime64
# ---------------------------------------------------------------------------


def test_seam_accepts_datetime64_day():
    assert temporal_coord_to_epoch_ms(np.datetime64("2020-03-01")) == _EPOCH_MS_2020_03_01


@pytest.mark.parametrize("unit", ["D", "h", "m", "s", "ms", "us", "ns"])
def test_seam_agrees_across_datetime64_units(unit):
    """The same instant spelled at any unit coerces to the same epoch-ms."""
    value = np.datetime64("2020-03-01T00:00:00").astype(f"datetime64[{unit}]")
    assert temporal_coord_to_epoch_ms(value) == _EPOCH_MS_2020_03_01


def test_seam_datetime64_agrees_with_timestamp_and_iso_spellings():
    """A sub-millisecond-precision instant coerces identically however it is
    spelled — ``datetime64[ns]``, ``pd.Timestamp``, ``datetime.datetime``."""
    ts = pd.Timestamp("2020-06-15T12:34:56.789")
    from_dt64 = temporal_coord_to_epoch_ms(np.datetime64("2020-06-15T12:34:56.789"))
    assert from_dt64 == temporal_coord_to_epoch_ms(ts)
    assert from_dt64 == temporal_coord_to_epoch_ms(ts.to_pydatetime())


# ---------------------------------------------------------------------------
# the seam refuses NaT by name (REFUSE MALFORMED), and so do its consumers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value", [pd.NaT, np.datetime64("NaT")], ids=["pandas_NaT", "datetime64_NaT"]
)
def test_seam_refuses_nat_by_name(value):
    """Before F1 this returned ``nan``: ``pd.NaT`` is a ``datetime.datetime``
    subclass whose epoch subtraction yields NaN rather than raising, and
    ``np.datetime64('NaT')`` was not accepted by the seam at all. A NaN
    epoch-ms is not a coordinate; it is a malformed input that must be named
    where it enters."""
    with pytest.raises(ValueError, match="not-a-time"):
        temporal_coord_to_epoch_ms(value)


@pytest.mark.parametrize(
    "value", [pd.NaT, np.datetime64("NaT")], ids=["pandas_NaT", "datetime64_NaT"]
)
def test_break_axis_gap_refuses_nat_in_its_own_vocabulary(value):
    import ferrum as fm

    with pytest.raises(ValueError, match=r"BreakAxis\.gap\[0\]: .*not-a-time"):
        fm.BreakAxis(axis="x", gap=(value, np.datetime64("2020-01-01")))


@pytest.mark.parametrize(
    "value", [pd.NaT, np.datetime64("NaT")], ids=["pandas_NaT", "datetime64_NaT"]
)
def test_timescale_domain_refuses_nat(value):
    """Pre-F1, ``pd.NaT`` in a TimeScale domain converted to ``nan`` and
    sailed to the wire; ``np.datetime64('NaT')`` was refused only as an
    unrecognised *type*, never as not-a-time."""
    from ferrum.encoding._scale import _convert_temporal_domain_elements

    with pytest.raises(ValueError, match="not-a-time"):
        _convert_temporal_domain_elements([value, "2021-01-01"])


@pytest.mark.parametrize(
    "value", [pd.NaT, np.datetime64("NaT")], ids=["pandas_NaT", "datetime64_NaT"]
)
def test_annotation_coordinate_refuses_nat(value):
    """Pre-F1 this reached the Rust wire and died with a leaked JSON-parse
    diagnostic (``chart_config: expected value at line 1 column 40``) or with
    ``Object of type datetime64 is not JSON serializable``."""
    import ferrum.annotation as ann

    with pytest.raises(ValueError, match="not-a-time"):
        ann.text(value, 1.5, "hi").to_dict()


# ---------------------------------------------------------------------------
# the three consumers agree
# ---------------------------------------------------------------------------


def test_all_three_consumers_coerce_datetime64_to_the_same_epoch_ms():
    """F1's actual finding: ``BreakAxis.gap`` accepted a temporal type the
    seam's other two consumers refused. One vocabulary means one number."""
    import ferrum as fm
    import ferrum.annotation as ann
    from ferrum.encoding._scale import _convert_temporal_domain_elements

    value = np.datetime64("2020-03-01")
    expected = _EPOCH_MS_2020_03_01

    assert fm.BreakAxis(axis="x", gap=(value, np.datetime64("2021-01-01"))).gap[0] == expected
    assert _convert_temporal_domain_elements([value]) == [expected]
    assert ann.text(value, 1.5, "hi").to_dict()["x"] == expected


def test_timescale_dict_domain_accepts_datetime64_end_to_end():
    """The raw-``scale={...}``-dict route is the one both wire routes share;
    a ``datetime64`` domain element now survives it as epoch-ms."""
    from ferrum.encoding._scale import _scale_to_dict

    out = _scale_to_dict(
        {"type": "time", "domain": [np.datetime64("2020-03-01"), np.datetime64("2021-01-01")]}
    )
    assert out["domain"] == [_EPOCH_MS_2020_03_01, 1609459200000.0]


def test_datetime_date_and_str_consumers_are_unchanged():
    """The widening must not move the pre-existing spellings."""
    assert temporal_coord_to_epoch_ms("2020-03-01") == _EPOCH_MS_2020_03_01
    assert temporal_coord_to_epoch_ms(dt.date(2020, 3, 1)) == _EPOCH_MS_2020_03_01
    assert temporal_coord_to_epoch_ms(dt.datetime(2020, 3, 1)) == _EPOCH_MS_2020_03_01


# ---------------------------------------------------------------------------
# the datetime64 overflow boundary constant (mutation-tester advisory)
# ---------------------------------------------------------------------------
#
# `_epoch_us_for_datetime64` refuses when `abs(raw) * count * numerator >
# _DATETIME64_MAX_ABS_US * denominator`. Every pre-existing test exercised
# values many orders of magnitude past that threshold, so a mutation that only
# WIDENED the constant (e.g. doubling it) kept the whole suite green. These
# pins sit ON the boundary: the largest accepted tick count and the smallest
# refused one, for a numerator-only unit (`s`) and a denominator-bearing one
# (`datetime64[1000000ns]`), on both signs. The expected threshold is derived
# from int64's own range HERE, not read back out of the implementation's
# constant — a pin that imports the number it is checking cannot fail.

#: The magnitude an int64 microsecond epoch offset can hold.
_INT64_MAX_US = 2**63 - 1

#: Largest `[s]` tick count whose microsecond value still fits (one tick is
#: exactly 1e6 us, so the unit's ratio is `numerator=10**6, denominator=1`).
_MAX_ACCEPTED_SECONDS = _INT64_MAX_US // 10**6

#: Largest `[1000000ns]` tick count that fits: the `ns` ratio is
#: `numerator=1, denominator=1000` and the dtype's `count` is 1e6, so one tick
#: is 1e3 us.
_MAX_ACCEPTED_MEGA_NS = _INT64_MAX_US // 1000


def _mega_ns(ticks: int) -> np.datetime64:
    return np.array([ticks], dtype="datetime64[1000000ns]")[0]


@pytest.mark.parametrize("sign", [1, -1], ids=["positive", "negative"])
def test_largest_in_range_second_tick_is_accepted(sign):
    from ferrum.annotation.coords import _epoch_us_for_datetime64

    ticks = sign * _MAX_ACCEPTED_SECONDS
    assert _epoch_us_for_datetime64(np.datetime64(ticks, "s")) == ticks * 10**6


@pytest.mark.parametrize("sign", [1, -1], ids=["positive", "negative"])
def test_smallest_out_of_range_second_tick_is_refused(sign):
    from ferrum.annotation.coords import _epoch_us_for_datetime64

    ticks = sign * (_MAX_ACCEPTED_SECONDS + 1)
    with pytest.raises(ValueError, match="out of range.*microsecond-precision"):
        _epoch_us_for_datetime64(np.datetime64(ticks, "s"))


@pytest.mark.parametrize("sign", [1, -1], ids=["positive", "negative"])
def test_largest_in_range_denominator_unit_tick_is_accepted(sign):
    from ferrum.annotation.coords import _epoch_us_for_datetime64

    ticks = sign * _MAX_ACCEPTED_MEGA_NS
    assert _epoch_us_for_datetime64(_mega_ns(ticks)) == ticks * 1000


@pytest.mark.parametrize("sign", [1, -1], ids=["positive", "negative"])
def test_smallest_out_of_range_denominator_unit_tick_is_refused(sign):
    from ferrum.annotation.coords import _epoch_us_for_datetime64

    ticks = sign * (_MAX_ACCEPTED_MEGA_NS + 1)
    with pytest.raises(ValueError, match="out of range.*microsecond-precision"):
        _epoch_us_for_datetime64(_mega_ns(ticks))


def test_boundary_constant_is_the_int64_microsecond_limit():
    """The constant itself, not only the branch it gates: a 'simplification'
    that shifts it is what the boundary pins above exist to catch, and this
    states what the value means."""
    from ferrum.annotation.coords import _DATETIME64_MAX_ABS_US

    assert _DATETIME64_MAX_ABS_US == _INT64_MAX_US
