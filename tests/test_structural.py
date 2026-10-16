"""Tests for ferrum.structural — SecondaryY, BreakAxis, and Inset."""

from __future__ import annotations

import datetime as dt
import decimal
import json
import re
import warnings

import numpy as np
import pandas as pd
import pytest
import polars as pl

import ferrum as fm
from ferrum.annotation.coords import norm, px
from ferrum.structural import BreakAxis, Inset, SecondaryY


def _nested_svg_rects(svg: str) -> list[dict[str, str]]:
    """Return the attribute dict of every ``<svg ...>`` tag in *svg*, root first.

    An inset embeds its rendered sub-chart as a nested ``<svg x=... y=...
    width=... height=...>`` element positioned against the parent plot rect
    (``ferrum-core``'s ``build_inset_nodes``). Unlike ``tests/_svg_rects.py``'s
    ``svg_elements`` (which only matches self-closed ``<tag .../>`` elements),
    ``<svg>`` opens a block, so this helper matches the opening tag directly.
    """
    return [
        dict(re.findall(r'([\w-]+)="([^"]+)"', m.group(1)))
        for m in re.finditer(r"<svg([^>]*)>", svg)
    ]


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def dual_field_df():
    return pl.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [10, 20, 30, 40, 50],
            "revenue": [100, 200, 150, 300, 250],
        }
    )


@pytest.fixture()
def base_chart(dual_field_df):
    return fm.Chart(dual_field_df).mark_point().encode(x="x", y="y")


# ---------------------------------------------------------------------------
# SecondaryY
# ---------------------------------------------------------------------------


class TestSecondaryY:
    def test_default_construction(self):
        s = SecondaryY("revenue")
        assert s.field == "revenue"
        assert s.mark == "line"
        assert s.axis is None
        assert s.color is None
        assert s.opacity is None
        assert s.scale is None

    def test_frozen(self):
        s = SecondaryY("revenue")
        with pytest.raises((TypeError, AttributeError)):
            s.mark = "bar"  # type: ignore[misc]

    def test_field_stored(self):
        s = SecondaryY("profit")
        assert s.field == "profit"

    def test_mark_override(self):
        s = SecondaryY("revenue", mark="bar")
        assert s.mark == "bar"

    def test_color_set(self):
        s = SecondaryY("revenue", color="#e45756")
        assert s.color == "#e45756"

    def test_opacity_set(self):
        s = SecondaryY("revenue", opacity=0.7)
        assert s.opacity == 0.7

    def test_axis_set(self):
        # axis accepts any value (Axis instance or None)
        s = SecondaryY("revenue", axis=object())
        assert s.axis is not None

    def test_scale_set(self):
        # scale accepts any value (Scale instance or None)
        s = SecondaryY("revenue", scale=object())
        assert s.scale is not None

    def test_all_fields(self):
        sentinel_axis = object()
        sentinel_scale = object()
        s = SecondaryY(
            "revenue",
            mark="bar",
            axis=sentinel_axis,
            color="steelblue",
            opacity=0.5,
            scale=sentinel_scale,
        )
        assert s.field == "revenue"
        assert s.mark == "bar"
        assert s.axis is sentinel_axis
        assert s.color == "steelblue"
        assert s.opacity == 0.5
        assert s.scale is sentinel_scale


# ---------------------------------------------------------------------------
# BreakAxis
# ---------------------------------------------------------------------------


class TestBreakAxis:
    def test_valid_construction_y(self):
        b = BreakAxis(axis="y", gap=(50, 200))
        assert b.axis == "y"
        assert b.gap == (50, 200)
        assert b.break_size == 12
        assert b.break_style == "slash"

    def test_valid_construction_x(self):
        b = BreakAxis(axis="x", gap=(0, 1))
        assert b.axis == "x"

    def test_frozen(self):
        b = BreakAxis(axis="y", gap=(0, 1))
        with pytest.raises((TypeError, AttributeError)):
            b.axis = "x"  # type: ignore[misc]

    def test_invalid_axis_raises(self):
        with pytest.raises(ValueError, match=r"BreakAxis.axis: axis must be one of"):
            BreakAxis(axis="z", gap=(0, 1))

    def test_invalid_break_style_raises(self):
        with pytest.raises(ValueError, match="break_style"):
            BreakAxis(axis="x", gap=(0, 1), break_style="invalid")

    @pytest.mark.parametrize("style", ["slash", "zigzag", "wave", "gap"])
    def test_valid_break_styles(self, style):
        b = BreakAxis(axis="y", gap=(10, 90), break_style=style)
        assert b.break_style == style

    def test_custom_break_size(self):
        b = BreakAxis(axis="x", gap=(5, 50), break_size=20)
        assert b.break_size == 20

    def test_multiple_gaps_as_list(self):
        b = BreakAxis(axis="y", gap=[(10, 20), (50, 60)])
        assert b.gap == [(10, 20), (50, 60)]


# ---------------------------------------------------------------------------
# BreakAxis.gap construction-time refusal matrix (#92 python half, batch
# wave-1 hardening): NaN, ±inf, reversed, and malformed-shape gaps all raise
# ``ValueError`` at construction instead of sailing through to a blank or
# nonsense break render.
# ---------------------------------------------------------------------------


class TestBreakAxisGapValidation:
    @pytest.mark.parametrize("bad_end", [float("nan"), float("inf"), float("-inf")])
    def test_single_gap_non_finite_end_raises(self, bad_end):
        with pytest.raises(ValueError, match=r"BreakAxis\.gap\[0\]: end must be a finite number"):
            BreakAxis(axis="y", gap=(10, bad_end))

    @pytest.mark.parametrize("bad_start", [float("nan"), float("inf"), float("-inf")])
    def test_single_gap_non_finite_start_raises(self, bad_start):
        with pytest.raises(ValueError, match=r"BreakAxis\.gap\[0\]: start must be a finite number"):
            BreakAxis(axis="y", gap=(bad_start, 90))

    def test_single_gap_reversed_raises(self):
        """The reversed-pair check runs on the coerced (always-``float``)
        bounds, so a plain-int input reports its ``float``-formatted value
        (``90.0``), not the ``int`` the caller wrote."""
        with pytest.raises(ValueError, match=r"BreakAxis\.gap\[0\]: start \(90\.0\) must be less"):
            BreakAxis(axis="y", gap=(90, 10))

    def test_single_gap_equal_bounds_raises(self):
        """``start == end`` is a zero-width gap -- refused too, not just ``>``."""
        with pytest.raises(ValueError, match=r"BreakAxis\.gap\[0\]"):
            BreakAxis(axis="y", gap=(50, 50))

    def test_multiple_gaps_one_bad_entry_names_its_index(self):
        with pytest.raises(ValueError, match=r"BreakAxis\.gap\[1\]: end must be a finite number"):
            BreakAxis(axis="y", gap=[(10, 20), (50, float("nan"))])

    def test_multiple_gaps_reversed_entry_raises(self):
        with pytest.raises(ValueError, match=r"BreakAxis\.gap\[1\]"):
            BreakAxis(axis="y", gap=[(10, 20), (60, 50)])

    def test_malformed_pair_wrong_length_raises(self):
        with pytest.raises(ValueError, match=r"pair of length 2"):
            BreakAxis(axis="y", gap=(10, 20, 30))

    def test_malformed_scalar_entry_in_list_raises(self):
        with pytest.raises(ValueError, match=r"pair of length 2"):
            BreakAxis(axis="y", gap=[10, 20])

    def test_empty_gap_list_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            BreakAxis(axis="y", gap=[])

    def test_non_numeric_gap_raises(self):
        """A bare string bound is now temporal-shaped (#230-6: BreakAxis.gap
        accepts ISO-8601 date/datetime strings), so ``"a"`` is diagnosed as
        an unparseable date rather than "not numeric" -- the pre-#230-6
        message was a symptom of strings being unsupported at all, which is
        no longer true. See ``TestBreakAxisGapTemporalBounds`` below for the
        full temporal-bounds matrix.
        """
        with pytest.raises(
            ValueError, match=r"BreakAxis\.gap\[0\]: cannot parse 'a' as an ISO-8601"
        ):
            BreakAxis(axis="y", gap=("a", 10))

    def test_non_numeric_non_temporal_gap_raises(self):
        """A bound that is neither numeric nor temporal-shaped (a dict, here)
        still raises the original numeric-type message unchanged."""
        with pytest.raises(ValueError, match=r"BreakAxis\.gap\[0\]: start must be a real number"):
            BreakAxis(axis="y", gap=({"not": "numeric"}, 10))

    def test_non_iterable_gap_raises(self):
        with pytest.raises(ValueError, match="must be a .* tuple or a list"):
            BreakAxis(axis="y", gap=5)  # type: ignore[arg-type]

    def test_valid_single_and_multi_gap_construct_without_error(self):
        BreakAxis(axis="y", gap=(10, 90))
        BreakAxis(axis="y", gap=[(10, 20), (50, 60)])

    def test_list_of_list_gap_normalizes_to_list_of_tuples(self):
        """Cycle-1 carryover: the coerce-and-write-back path normalizes a
        list-of-list gap into a list of tuples, which also flips
        ``BreakAxis`` equality between the list and tuple spellings of the
        same gap to ``True`` -- pinned here so that stored-shape contract
        does not silently drift."""
        b = BreakAxis(axis="y", gap=[[10, 20]])
        assert b.gap == [(10, 20)]

    def test_tuple_of_pairs_keeps_its_container_and_stays_hashable(self):
        """The coerce-and-write-back path must not change ``gap``'s container
        type. It did (design review F3): a tuple of pairs came back as a
        LIST of pairs, which made a frozen ``BreakAxis`` unhashable —
        ``hash(BreakAxis(axis="y", gap=((1, 5), (6, 9))))`` worked before the
        temporal-bounds wave and raised ``TypeError: unhashable type: 'list'``
        after it — so hashability depended on which of two equivalent
        spellings the caller chose."""
        b = BreakAxis(axis="y", gap=((1, 5), (6, 9)))
        assert isinstance(b.gap, tuple)
        assert b.gap == ((1, 5), (6, 9))
        assert isinstance(hash(b), int)

    def test_list_of_pairs_stays_a_list(self):
        """The other half of F3: a list-spelled gap is not silently promoted
        to a tuple either. It was never hashable, and is not made so."""
        b = BreakAxis(axis="y", gap=[(1, 5), (6, 9)])
        assert isinstance(b.gap, list)
        assert b.gap == [(1, 5), (6, 9)]
        with pytest.raises(TypeError):
            hash(b)

    def test_bare_pair_stays_a_bare_tuple(self):
        b = BreakAxis(axis="y", gap=(10, 90))
        assert b.gap == (10, 90)
        assert isinstance(hash(b), int)

    @pytest.mark.parametrize(
        "start, end",
        [
            (np.int64(10), np.int64(90)),
            (np.int32(10), np.int32(90)),
            (np.float32(10.0), np.float32(90.0)),
            (np.float64(10.0), np.float64(90.0)),
        ],
        ids=["int64", "int32", "float32", "float64"],
    )
    def test_numpy_numeric_bounds_are_accepted(self, start, end):
        """A numpy scalar is what a dataframe hands back (``df["v"].max()``),
        and the encoding layer's domain coercion has always accepted one via
        its duck-typed ``float(value)``. ``BreakAxis`` refused every numpy
        integer and narrow float while accepting ``np.datetime64`` — an
        asymmetry with no contract behind it (design review F10). ``np.float64``
        reaches the plain ``isinstance(value, (int, float))`` branch by
        subclassing ``float``; that branch now coerces with ``float(value)``
        too, so every numpy spelling stores a plain Python ``float`` rather
        than ``np.float64`` being the one that keeps its numpy type."""
        b = BreakAxis(axis="y", gap=(start, end))
        assert b.gap == (10.0, 90.0)
        assert all(type(bound) is float for bound in b.gap)

    def test_numpy_bool_bound_is_still_refused(self):
        """``bool`` is excluded from the numeric kinds (a boolean is not a
        break region), and widening to numpy numerics must not smuggle
        ``np.bool_`` in: it is neither ``np.integer`` nor ``np.floating``."""
        with pytest.raises(ValueError, match=r"BreakAxis\.gap\[0\]: start must be a real number"):
            BreakAxis(axis="y", gap=(np.bool_(True), 90))


# ---------------------------------------------------------------------------
# BreakAxis.gap temporal bounds (#230-6): str/date/datetime/np.datetime64/
# pd.Timestamp bounds coerce to epoch-ms UTC via the same conversion the
# encoding layer applies to temporal scale domains and annotation
# coordinates (ferrum.annotation.coords.temporal_coord_to_epoch_ms); a pair
# mixing a numeric bound with a temporal bound refuses, an unparseable
# string names the value, and ordering/finite validation runs on the
# coerced (numeric) values -- the #92 matrix above keeps passing unchanged.
# ---------------------------------------------------------------------------

# 2020-03-01T00:00:00Z and 2021-01-01T00:00:00Z in epoch-ms, cross-checked
# against ferrum.annotation.coords.temporal_coord_to_epoch_ms's own doctest
# (``temporal_coord_to_epoch_ms("2020-06-01") == 1590969600000``).
_EPOCH_MS_2020_03_01 = 1583020800000.0
_EPOCH_MS_2021_01_01 = 1609459200000.0


class TestBreakAxisGapTemporalBounds:
    def test_iso_date_strings_coerce_to_epoch_ms(self):
        b = BreakAxis(axis="x", gap=("2020-03-01", "2021-01-01"))
        assert b.gap == (_EPOCH_MS_2020_03_01, _EPOCH_MS_2021_01_01)

    def test_datetime_date_bounds_coerce(self):
        b = BreakAxis(axis="x", gap=(dt.date(2020, 3, 1), dt.date(2021, 1, 1)))
        assert b.gap == (_EPOCH_MS_2020_03_01, _EPOCH_MS_2021_01_01)

    def test_datetime_datetime_bounds_coerce(self):
        b = BreakAxis(
            axis="x",
            gap=(dt.datetime(2020, 3, 1, 12, 0, 0), dt.datetime(2021, 1, 1, 0, 0, 0)),
        )
        start, end = b.gap
        assert start == _EPOCH_MS_2020_03_01 + 12 * 3600 * 1000
        assert end == _EPOCH_MS_2021_01_01

    def test_np_datetime64_bounds_coerce(self):
        b = BreakAxis(
            axis="x",
            gap=(np.datetime64("2020-03-01"), np.datetime64("2021-01-01")),
        )
        assert b.gap == (_EPOCH_MS_2020_03_01, _EPOCH_MS_2021_01_01)

    def test_pandas_timestamp_bounds_coerce(self):
        """``pd.Timestamp`` is a ``datetime.datetime`` subclass, so it rides
        the same coercion branch as plain ``datetime.datetime`` -- no
        pandas-specific code path exists or is needed."""
        b = BreakAxis(
            axis="x",
            gap=(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-01-01")),
        )
        assert b.gap == (_EPOCH_MS_2020_03_01, _EPOCH_MS_2021_01_01)

    def test_multiple_temporal_gaps_as_list(self):
        b = BreakAxis(
            axis="x",
            gap=[("2020-01-01", "2020-02-01"), ("2020-06-01", "2020-07-01")],
        )
        assert len(b.gap) == 2
        assert all(isinstance(bound, float) for pair in b.gap for bound in pair)
        assert all(start < end for start, end in b.gap)

    @pytest.mark.parametrize(
        "gap",
        [(10, "2020-01-01"), ("2020-01-01", 10)],
        ids=["numeric_then_temporal", "temporal_then_numeric"],
    )
    def test_mixed_numeric_and_temporal_pair_raises(self, gap):
        with pytest.raises(
            ValueError,
            match=r"BreakAxis\.gap\[0\]: mixes a numeric bound .* with a temporal bound",
        ):
            BreakAxis(axis="x", gap=gap)

    def test_unparseable_string_bound_names_the_value(self):
        with pytest.raises(ValueError, match=r"BreakAxis\.gap\[0\]: cannot parse 'not-a-date'"):
            BreakAxis(axis="x", gap=("not-a-date", "2020-01-01"))

    def test_reversed_temporal_pair_raises_after_coercion(self):
        """Ordering validation runs on the COERCED (numeric) values -- the
        #92 finite/ordered contract is unchanged, just fed post-coercion
        floats instead of the original strings."""
        with pytest.raises(ValueError, match=r"start \(.*\) must be less than end"):
            BreakAxis(axis="x", gap=("2021-01-01", "2020-01-01"))

    def test_nat_datetime64_bound_raises(self):
        with pytest.raises(ValueError, match="not-a-time"):
            BreakAxis(axis="x", gap=(np.datetime64("NaT"), np.datetime64("2020-01-01")))

    def test_pandas_nat_bound_raises(self):
        """``pd.NaT`` is a ``datetime.datetime`` subclass whose epoch
        subtraction yields NaN rather than raising (unlike an ordinary
        ``datetime``), so left unchecked it would sail past
        ``temporal_coord_to_epoch_ms`` and only get refused two layers
        downstream with a generic "not finite" message. It must get the same
        explicit not-a-time refusal as ``np.datetime64('NaT')``."""
        with pytest.raises(ValueError, match="not-a-time"):
            BreakAxis(axis="x", gap=(pd.NaT, pd.Timestamp("2020-01-01")))

    def test_aware_datetime_bound_converts_to_utc(self):
        """A timezone-aware bound is converted to UTC before its epoch
        offset is computed; the naive spelling of the same wall-clock time
        is instead TREATED AS UTC (module docstring). A +05:00-aware 12:00
        is 07:00 UTC -- five hours earlier than the naive 12:00 spelling --
        so the two must not coerce to the same epoch-ms value."""
        naive = dt.datetime(2020, 3, 1, 12, 0, 0)
        aware = dt.datetime(2020, 3, 1, 12, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=5)))
        naive_start, _ = BreakAxis(axis="x", gap=(naive, dt.datetime(2021, 1, 1))).gap
        aware_start, _ = BreakAxis(axis="x", gap=(aware, dt.datetime(2021, 1, 1))).gap
        assert aware_start == naive_start - 5 * 3600 * 1000

    def test_datetime64_out_of_range_bound_raises(self):
        """An extreme np.datetime64 bound (year ~3e9) silently overflows
        int64 when cast to microsecond resolution instead of raising
        (probed: wraps to a bogus finite epoch-ms value that the ordering/
        finiteness check cannot catch). REFUSE MALFORMED binds every new
        edge this task introduces, so this must raise rather than accept a
        wrapped, meaningless bound."""
        with pytest.raises(ValueError, match="out of range"):
            BreakAxis(
                axis="x",
                gap=(np.datetime64(3_000_000_000, "Y"), np.datetime64("2020-01-01")),
            )

    @pytest.mark.parametrize(
        "years",
        [2**61, -(2**61), 2**62, -(2**62), 2**61 + 1, 2**62 + 1, 3 * 2**61],
    )
    def test_datetime64_wrapped_reference_cast_bound_raises(self, years):
        """Cycle-3's wrap guard compared the microsecond cast against a
        second numpy cast (to ``datetime64[s]``) as a 'trustworthy'
        reference. That reference is not trustworthy: it is itself a
        fixed-width numpy cast, and for a ``"Y"``-unit value with a tick
        count around 2**61-2**62 it overflows int64 too. When both casts
        wrap, they can land in coincidental agreement and the guard
        silently accepts a meaningless bound -- probed end to end,
        ``fm.BreakAxis(gap=(np.datetime64(2**62, "Y"), np.datetime64(2**62 +
        1, "Y")))`` used to return ``(24192000.0, 31560192000.0)`` (1970-01
        -06T43 and 1971-01-01) instead of raising. The guard now reasons in
        arbitrary-precision Python integers, which cannot wrap, so no
        second cast can be fooled this way -- every one of these probed
        power-of-two tick counts must still raise, naming the microsecond
        precision the guard actually tests (not "millisecond", cycle-3's
        wording bug)."""
        with pytest.raises(ValueError, match="out of range.*microsecond-precision"):
            BreakAxis(
                axis="x",
                gap=(np.datetime64(years, "Y"), np.datetime64(years + 1, "Y")),
            )

    @pytest.mark.parametrize(
        "value",
        [
            np.array([2**60], dtype="datetime64[1000000ns]")[0],
            np.array([18448321910509551], dtype="datetime64[1000us]")[0],
        ],
        ids=["ns_count_1e6_at_2pow60", "us_count_1000_year_586562_aliasing"],
    )
    def test_datetime64_multiple_count_fine_unit_bound_raises(self, value):
        """Cycle-4 closed the coarse-unit (``[Y]``) slip-throughs but left
        every fine unit (``us``/``ns``/``ps``/``fs``/``as``) exempt from the
        overflow proof whenever the dtype's ``count`` > 1 -- an invariant
        asserted in a comment ('finer than millisecond ... cannot grow past
        int64 range') that is false the moment ``count`` multiplies a fine
        tick back up past microsecond resolution. Probed end to end through
        this same constructor: ``datetime64[1000000ns]`` at tick ``2**60``
        used to be ACCEPTED as ``-9223372036854776.0`` (a wrapped,
        NaT-adjacent sentinel), and ``datetime64[1000us]`` at tick
        ``18448321910509551`` (a bound roughly year 586,562) used to alias
        to ``1577836799999.384`` -- New Year's Eve 2019. Every unit must
        reach the microsecond cast only after the same arbitrary-precision
        proof, with no unit-based skip branch."""
        with pytest.raises(ValueError, match="out of range.*microsecond-precision"):
            BreakAxis(axis="x", gap=(value, np.datetime64("2020-01-01")))

    def test_datetime64_multiple_count_fine_unit_bound_is_accepted(self):
        """A count>1 fine-unit dtype (``datetime64[1000us]``, i.e.
        millisecond granularity spelled as 1000-microsecond ticks) is a
        legal, ordinary numpy dtype for an in-range instant -- REFUSE
        MALFORMED does not cover a well-formed bound just because its
        dtype's count is not 1. It must be ACCEPTED and agree with the
        ``pd.Timestamp`` coercion of the identical instant, the same
        cross-check ``test_datetime64_ns_and_pandas_timestamp_bound_agree``
        pins for a plain ``[ns]`` dtype."""
        ts = pd.Timestamp("2020-06-15T12:34:56.789")
        ticks = int(ts.value // 1_000_000)  # ts.value is ns since epoch; 1000us == 1ms/tick
        bound = np.array([ticks], dtype="datetime64[1000us]")[0]
        start, _ = BreakAxis(axis="x", gap=(bound, np.datetime64("2021-01-01"))).gap
        pandas_start, _ = BreakAxis(axis="x", gap=(ts, pd.Timestamp("2021-01-01"))).gap
        assert start == pandas_start

    @pytest.mark.parametrize("dtype", ["datetime64[0us]", "datetime64[0D]", "datetime64[0Y]"])
    def test_datetime64_zero_count_unit_bound_raises(self, dtype):
        """A zero-count datetime64 dtype (``datetime64[0us]`` etc.) is a
        legal, constructible numpy dtype whose ticks denote NO duration.
        Cycle-5's multiplicative overflow proof degenerated to ``0 > bound``
        on it and silently accepted a fabricated 1970 epoch instant
        (probed: tick ``1592224496789123`` under ``[0us]`` was accepted as
        ``0.0``), while numpy's own casts raise ``SystemError`` on the same
        dtype -- so the ``[0Y]``/``[0M]`` calendar branch leaked
        ``SystemError`` out of ``__post_init__``. Refused by the explicit
        ``count < 1`` gate before any arithmetic (#230-6 cycle-6); negative
        counts are rejected by numpy at dtype-parse time, so this closes
        the last free metadata field."""
        bound = np.array([1_592_224_496_789_123], dtype=dtype)[0]
        with pytest.raises(ValueError, match="zero-count datetime64 unit"):
            BreakAxis(axis="x", gap=(bound, np.datetime64("2030-01-01")))

    def test_datetime64_huge_but_representable_year_bound_is_accepted(self):
        """A np.datetime64 bound many tens of thousands of years out is
        exotic but still exactly representable at microsecond resolution --
        REFUSE MALFORMED does not cover a merely large, well-formed value.
        This pins the accept side of the wrap guard's boundary at a
        magnitude far larger than any of the sub-ms/ns tests above, so a
        regression that makes the guard over-eager (refusing large-but-valid
        bounds, cycle-2's failure mode) fails loudly here too."""
        start_bound, stop_bound = np.datetime64(200_000, "Y"), np.datetime64(200_001, "Y")
        b = BreakAxis(axis="x", gap=(start_bound, stop_bound))
        start, stop = b.gap
        assert start == int(start_bound.astype("datetime64[us]").astype("int64")) / 1000.0
        assert stop == int(stop_bound.astype("datetime64[us]").astype("int64")) / 1000.0

    def test_sub_millisecond_datetime64_bound_is_accepted(self):
        """A np.datetime64 bound finer than millisecond precision is
        sub-millisecond precision, not malformed -- REFUSE MALFORMED does
        not cover ordinary precision. Cycle-2's overflow guard
        (`ms_value.astype(value.dtype) != value`, an EXACT round-trip check)
        could not tell int64 wrap apart from this and refused it with a
        factually wrong "out of range" message; this pins the accept side of
        that boundary so a regression back to the exact-round-trip check
        fails loudly. A pandas datetime column yields datetime64[ns], so
        this is the ordinary way a user's bound arrives, not an edge case."""
        b = BreakAxis(
            axis="x",
            gap=(
                np.datetime64("2020-03-01T00:00:00.000123456"),
                np.datetime64("2021-01-01"),
            ),
        )
        start, _ = b.gap
        assert start == _EPOCH_MS_2020_03_01 + 0.123

    def test_datetime64_ns_and_pandas_timestamp_bound_agree(self):
        """The recurring defect (cycle-2 -> cycle-3): for the identical
        instant, a np.datetime64 bound and a pd.Timestamp bound must coerce
        to the SAME epoch-ms value. Cycle-2 accepted the pd.Timestamp
        spelling (fractional ms preserved) while refusing the np.datetime64
        spelling outright -- exception drift between two names for one
        well-formed value. Both branches cast at microsecond resolution
        (the precision ceiling datetime.datetime/pd.Timestamp share), so
        for THIS instant (2020-03-01T00:00:00.000123) they produce
        numerically identical results -- not a general bit-for-bit claim
        across all instants, since the sibling (datetime/Timestamp) branch
        routes through a float-seconds intermediate that can differ from
        this branch's integer microsecond path by a few hundred
        nanoseconds at other instants."""
        instant = "2020-03-01T00:00:00.000123"
        pandas_ts_start, _ = BreakAxis(
            axis="x", gap=(pd.Timestamp(instant), pd.Timestamp("2021-01-01"))
        ).gap
        datetime64_start, _ = BreakAxis(
            axis="x", gap=(np.datetime64(instant), np.datetime64("2021-01-01"))
        ).gap
        assert datetime64_start == pandas_ts_start == _EPOCH_MS_2020_03_01 + 0.123

    def test_datetime64_ns_from_pandas_column_is_accepted(self):
        """The exact value shape a pandas datetime column yields
        (`datetime64[ns]` via `.to_numpy()`) must be accepted -- this is the
        ordinary path a real caller hits, not a synthetic edge case."""
        ns_value = pd.to_datetime(["2020-03-01 00:00:00.000123"]).to_numpy()[0]
        b = BreakAxis(axis="x", gap=(ns_value, np.datetime64("2021-01-01")))
        start, _ = b.gap
        assert start == _EPOCH_MS_2020_03_01 + 0.123


# ---------------------------------------------------------------------------
# BreakAxis temporal e2e rendering (#230-6): the docs example on a real
# temporal x axis actually breaks the axis; a coerced date gap on a
# non-temporal axis is out-of-domain and warns instead of refusing.
# ---------------------------------------------------------------------------


class TestBreakAxisTemporalRendering:
    def test_docs_example_temporal_gap_breaks_axis(self):
        """docs/site/guide/concepts/break-axes.md's 'Time series with a data
        gap' recipe (~L211-219) ships this exact form:

            + fm.BreakAxis(
                axis="x",
                gap=("2020-03-01", "2021-01-01"),
                break_style="wave",
            )

        Before #230-6 this raised ``ValueError`` at construction (BreakAxis
        only accepted numeric bounds) — the published docs shipped a form
        that crashed. Wired onto a real temporal x axis whose data straddles
        the gap, it must now actually break the axis rather than merely not
        raising: the ``wave`` break style paints indicator ``<path>``
        geometry a plain line chart never emits (mark_line renders
        ``<polyline>``, not ``<path>``), so a strict increase in ``<path>``
        count against the unbroken baseline is discriminating evidence the
        break took effect, not just that construction succeeded.
        """
        df = pl.DataFrame(
            {
                "date": [
                    dt.date(2019, 6, 1),
                    dt.date(2019, 12, 1),
                    dt.date(2020, 1, 15),
                    dt.date(2021, 3, 1),
                    dt.date(2021, 9, 1),
                    dt.date(2022, 3, 1),
                ],
                "value": [10.0, 12.0, 11.0, 15.0, 18.0, 20.0],
            }
        )
        unbroken = fm.Chart(df).mark_line().encode(x="date:T", y="value:Q")
        broken = unbroken + fm.BreakAxis(
            axis="x",
            gap=("2020-03-01", "2021-01-01"),
            break_style="wave",
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            broken_svg = broken.to_svg()
        assert not [w for w in caught if "had no effect" in str(w.message)], (
            "the docs' date-string gap intersects the temporal x domain "
            f"(2019..2022) and must take effect: {[str(w.message) for w in caught]}"
        )
        assert broken_svg.startswith("<svg")

        unbroken_svg = unbroken.to_svg()
        broken_paths = len(re.findall(r"<path\b", broken_svg))
        unbroken_paths = len(re.findall(r"<path\b", unbroken_svg))
        assert broken_paths > unbroken_paths, (
            "the wave break indicator must paint new <path> geometry that the "
            f"unbroken chart lacks; broken={broken_paths} unbroken={unbroken_paths}"
        )

    def test_temporal_gap_on_numeric_axis_warns_ineffective(self):
        """A coerced date gap on a non-temporal (numeric) axis is
        out-of-domain: it renders (does not raise) and warns
        ``BreakAxisIneffective`` rather than silently no-op'ing."""
        df = pl.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})
        chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q") + fm.BreakAxis(
            axis="x", gap=("2020-03-01", "2021-01-01")
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svg = chart.to_svg()
        assert svg.startswith("<svg")
        assert [w for w in caught if "had no effect" in str(w.message)], (
            "a coerced date gap on a numeric x axis (domain 1..5) is wildly "
            "out-of-domain and must warn BreakAxisIneffective, not silently "
            f"no-op: {[str(w.message) for w in caught]}"
        )


# ---------------------------------------------------------------------------
# Inset
# ---------------------------------------------------------------------------


class TestInset:
    def test_default_construction(self):
        sentinel_chart = object()
        inset = Inset(chart=sentinel_chart, bounds=(0.1, 0.1, 0.9, 0.9))
        assert inset.chart is sentinel_chart
        assert inset.bounds == (0.1, 0.1, 0.9, 0.9)
        assert inset.border is True
        assert inset.border_color == "#999"
        assert inset.border_dash is None
        assert inset.background == "#fff"
        assert inset.shadow is False
        assert inset.connect_to is None
        assert inset.connect_style == "lines"

    def test_frozen(self):
        inset = Inset(chart=object(), bounds=(0, 0, 1, 1))
        with pytest.raises((TypeError, AttributeError)):
            inset.border = False  # type: ignore[misc]

    def test_invalid_connect_style_raises(self):
        with pytest.raises(ValueError, match="connect_style"):
            Inset(chart=object(), bounds=(0, 0, 1, 1), connect_style="arrow")

    @pytest.mark.parametrize("style", ["bracket", "lines", "none"])
    def test_valid_connect_styles(self, style):
        inset = Inset(chart=object(), bounds=(0, 0, 1, 1), connect_style=style)
        assert inset.connect_style == style

    def test_border_false(self):
        inset = Inset(chart=object(), bounds=(0, 0, 1, 1), border=False)
        assert inset.border is False

    def test_shadow_enabled(self):
        inset = Inset(chart=object(), bounds=(0, 0, 1, 1), shadow=True)
        assert inset.shadow is True

    def test_background_none(self):
        inset = Inset(chart=object(), bounds=(0, 0, 1, 1), background=None)
        assert inset.background is None

    def test_border_dash_set(self):
        inset = Inset(chart=object(), bounds=(0, 0, 1, 1), border_dash=[4, 4])
        assert inset.border_dash == [4, 4]

    def test_connect_to_set(self):
        inset = Inset(chart=object(), bounds=(0, 0, 1, 1), connect_to=(0.5, 0.5))
        assert inset.connect_to == (0.5, 0.5)

    def test_px_bounds_accepted_at_construction(self):
        """PixelCoord no longer raises; it is accepted at construction time
        (the old TypeError happened downstream, at serialization)."""
        inset = Inset(chart=object(), bounds=(px(10), px(10), px(220), px(160)))
        assert inset.bounds == (px(10), px(10), px(220), px(160))

    def test_norm_and_px_mixed_bounds_accepted(self):
        inset = Inset(chart=object(), bounds=(px(10), norm(0.1), px(220), norm(0.9)))
        assert inset.bounds == (px(10), norm(0.1), px(220), norm(0.9))


# ---------------------------------------------------------------------------
# Inset bounds construction-time refusal matrix (F-L08-08, batch H T6)
# ---------------------------------------------------------------------------


class TestInsetBoundsValidation:
    """Construction-time typed refusals replace the old silent off-canvas
    render (spec §4 H6): out-of-[0, 1] normalized entries, negative px
    entries, and an empty box (left >= right or top >= bottom, compared only
    within like kinds) all raise ``ValueError`` naming the bad entry.
    """

    @pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan")])
    def test_bare_float_out_of_range_raises(self, bad):
        with pytest.raises(ValueError, match=r"Inset\.bounds\[0\] \(left\)"):
            Inset(chart=object(), bounds=(bad, 0.1, 0.9, 0.9))

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_normcoord_out_of_range_raises(self, bad):
        with pytest.raises(ValueError, match=r"Inset\.bounds\[2\] \(right\)"):
            Inset(chart=object(), bounds=(0.1, 0.1, norm(bad), 0.9))

    def test_negative_px_raises(self):
        with pytest.raises(ValueError, match=r"Inset\.bounds\[1\] \(top\)"):
            Inset(chart=object(), bounds=(px(10), px(-5), px(220), px(160)))

    def test_negative_px_message_names_the_field(self):
        with pytest.raises(ValueError, match=r"Inset\.bounds\[3\] \(bottom\)"):
            Inset(chart=object(), bounds=(px(10), px(10), px(220), px(-1)))

    def test_empty_box_left_right_same_kind_norm_raises(self):
        with pytest.raises(ValueError, match="empty box"):
            Inset(chart=object(), bounds=(0.9, 0.1, 0.1, 0.9))

    def test_empty_box_top_bottom_same_kind_norm_raises(self):
        with pytest.raises(ValueError, match="empty box"):
            Inset(chart=object(), bounds=(0.1, 0.9, 0.9, 0.1))

    def test_empty_box_left_right_same_kind_px_raises(self):
        with pytest.raises(ValueError, match="empty box"):
            Inset(chart=object(), bounds=(px(220), px(10), px(10), px(160)))

    def test_empty_box_equal_bounds_raises(self):
        """``left == right`` is refused too (a zero-width box), not just ``>``."""
        with pytest.raises(ValueError, match="empty box"):
            Inset(chart=object(), bounds=(0.5, 0.1, 0.5, 0.9))

    def test_mixed_kind_pair_not_compared_for_emptiness(self):
        """A norm/px pair on the same side is not directly comparable, so it
        is never refused as an empty box -- only same-kind pairs are checked
        (spec §4 H6: "compared within like kinds")."""
        # left is px(220), right is norm(0.1) -- numerically "left > right" if
        # compared naively, but different kinds means no emptiness check fires.
        inset = Inset(chart=object(), bounds=(px(220), 0.1, norm(0.1), 0.9))
        assert inset.bounds == (px(220), 0.1, norm(0.1), 0.9)

    def test_wrong_length_bounds_raises(self):
        with pytest.raises(ValueError, match="exactly 4 entries"):
            Inset(chart=object(), bounds=(0.1, 0.1, 0.9))

    def test_valid_bounds_construct_without_error(self):
        Inset(chart=object(), bounds=(0.1, 0.1, 0.9, 0.9))
        Inset(chart=object(), bounds=(px(0), px(0), px(200), px(150)))
        Inset(chart=object(), bounds=(px(10), norm(0.1), px(220), norm(0.9)))


# ---------------------------------------------------------------------------
# One Inset classifier: 26-shape probe (#239)
# ---------------------------------------------------------------------------


class _FloatDuck:
    """A duck-typed ``__float__`` object, mirroring the research probe's case.

    Cycle 2 accepted this (coerced via its ``__float__``); cycle 3 closes
    that duck-typed tail entirely (`#239`, decision doc §2 deviation
    disclosed in the task report), so this is now a *refused* case --
    ``test_float_duck_no_longer_accepted`` below pins that a custom
    ``__float__`` object never reopens the coerce-before-validate hole
    ``np.str_``/``np.bytes_``/``np.complex128`` slipped through via the
    same mechanism (numpy scalars inherit ``__float__`` from
    ``np.generic``).
    """

    def __float__(self) -> float:
        return 0.25


def _real_inset_wire(base_chart, dual_field_df, entry):
    """Build a real ``Chart + Inset`` and return its serialized ``bounds[0]``
    through the actual production path (``Chart._resolve_chart_config`` ->
    ``ferrum._render.Chart._serialize_structural``), not a test-local mirror
    of the adapter -- a regression in the real adapter at ``_render.py``
    would not fail a test pinned against a hand-copied reimplementation of
    it.

    ``right`` is a large ``px()`` bound rather than a norm fraction: a
    norm-kind *entry* then never shares a kind with ``right``, so the
    empty-box check (same-kind pairs only) never fires regardless of
    *entry*'s value (some accepted shapes are ``1.0``/``1``, which would
    trip an empty-box refusal against a fixed norm ``right``); a px-kind
    *entry* shares ``right``'s kind but is always well under it.
    """
    from ferrum.annotation.coords import px

    inset_chart = fm.Chart(dual_field_df).mark_point().encode(x="x", y="revenue")
    chart = base_chart + Inset(chart=inset_chart, bounds=(entry, 0.1, px(1_000_000), 0.9))
    return chart._resolve_chart_config()["structural"][0]["bounds"][0]


def _assert_wire_type_and_value(actual, expected) -> None:
    """Assert one serialized ``bounds`` entry matches *expected* by type
    AND value. Plain ``==`` is not discriminating here: ``1 == 1.0``,
    ``np.float32(0.25) == 0.25``, and ``np.float64(0.5) == 0.5`` all hold,
    so a classifier regression that changed ``int`` -> ``float`` or
    ``numpy`` -> native (each of which changes the emitted JSON bytes, e.g.
    ``"1"`` vs. ``"1.0"``) would still pass an ``==`` comparison."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == {"px"}, (
            f"expected a {{'px': ...}} wire entry, got {actual!r}"
        )
        _assert_wire_type_and_value(actual["px"], expected["px"])
        return
    assert type(actual) is type(expected), (
        f"wire type drift: {type(actual).__name__} != {type(expected).__name__} "
        f"({actual!r} vs {expected!r})"
    )
    assert actual == expected


class TestInsetBoundClassifierShapes:
    """``#239``'s 26-shape probe (research ``python-dedup.md`` §1, ``p239.py``):
    the one classifier (``ferrum.structural._inset_bound_kind_and_value``)
    must serialize every accepted shape byte-identically to the pre-fix
    baseline captured on ``fix/hygiene-wave6`` at ``a34f82c0`` (decision doc
    §2's correctness table), and every malformed shape must raise a labeled
    ``ValueError`` naming the entry -- never an unlabeled ``ValueError`` or a
    different exception class. Accepted shapes are pinned through the real
    serialization path with a type-plus-value comparison (cycle 2 fix: the
    cycle 1 pins went through a test-local classifier mirror and compared
    with plain ``==``, so neither a real-adapter regression nor an
    int-vs-float/numpy-vs-native type drift would have failed them).
    """

    # -- Accepted shapes: byte-identical wire, both before and after the fix. --

    @pytest.mark.parametrize(
        ("case", "entry", "expected_wire"),
        [
            ("bare float", 0.5, 0.5),
            ("bare int", 1, 1.0),
            ("bare int 0", 0, 0.0),
            ("norm()", norm(0.5), 0.5),
            # Wrapped values reach the wire as floats too (the one
            # real-number predicate coerces every accepted number).
            ("px()", px(10), {"px": 10.0}),
            ("NormCoord int", norm(1), 1.0),
            ("PixelCoord int", px(10), {"px": 10.0}),
            ("NormCoord np", norm(np.float64(0.5)), 0.5),
            ("np.float32", np.float32(0.25), 0.25),
            ("np.int64", np.int64(0), 0.0),
            ("Decimal", decimal.Decimal("0.5"), 0.5),
        ],
    )
    def test_accepted_shapes_wire_is_byte_identical(
        self, base_chart, dual_field_df, case, entry, expected_wire
    ):
        actual = _real_inset_wire(base_chart, dual_field_df, entry)
        _assert_wire_type_and_value(actual, expected_wire)

    def test_float_duck_no_longer_accepted(self):
        """Deliberate deviation from the decision doc's correctness table
        (`#239` cycle 3, disclosed in the task report): a duck-typed
        ``__float__`` object was accepted through cycle 2's open-ended
        ``hasattr(bound, "__float__")`` tail -- the same tail every numpy
        scalar satisfies via inheritance from ``np.generic``, which is how
        ``np.str_``/``np.bytes_``/``np.complex128`` laundered through
        despite being refused-by-name types. Closing that tail for numpy
        necessarily closes it for a plain custom object too; REFUSE
        MALFORMED and user ruling Q6 outrank the byte-stability row this
        case used to satisfy."""
        with pytest.raises(ValueError, match=r"Inset\.bounds\[0\] \(left\) must be a number"):
            Inset(chart=object(), bounds=(_FloatDuck(), 0.1, 0.9, 0.9))

    # -- The five Q6-named malformed shapes: exact labeled ValueError text. --

    def test_non_numeric_string_raises_labeled_value_error(self):
        with pytest.raises(
            ValueError,
            match=r"^Inset\.bounds\[0\] \(left\) must be a number between 0 and 1, "
            r"not str; received 'a'$",
        ):
            Inset(chart=object(), bounds=("a", 0.1, 0.9, 0.9))

    def test_none_raises_labeled_value_error_not_type_error(self):
        """LB-239-a: ``None`` previously raised ``TypeError``, an exception
        class an ``except ValueError`` around ``Inset(...)`` would miss."""
        with pytest.raises(
            ValueError,
            match=r"^Inset\.bounds\[0\] \(left\) must be a number between 0 and 1, "
            r"not NoneType; received None$",
        ):
            Inset(chart=object(), bounds=(None, 0.1, 0.9, 0.9))

    def test_true_refused_by_name(self):
        """LB-239-b: ``True`` previously laundered to ``1.0`` past the
        package's own bool exclusion."""
        with pytest.raises(
            ValueError,
            match=r"^Inset\.bounds\[0\] \(left\) must be a number between 0 and 1, "
            r"not bool; received True$",
        ):
            Inset(chart=object(), bounds=(True, 0.1, 0.9, 0.9))

    def test_false_refused_by_name(self):
        with pytest.raises(
            ValueError,
            match=r"^Inset\.bounds\[0\] \(left\) must be a number between 0 and 1, "
            r"not bool; received False$",
        ):
            Inset(chart=object(), bounds=(False, 0.1, 0.9, 0.9))

    def test_numeric_string_refused_by_name(self):
        """LB-239-b: a numeric string previously laundered to a float past
        the package's own str exclusion (user ruling Q6)."""
        with pytest.raises(
            ValueError,
            match=r"^Inset\.bounds\[0\] \(left\) must be a number between 0 and 1, "
            r"not str; received '0\.5'$",
        ):
            Inset(chart=object(), bounds=("0.5", 0.1, 0.9, 0.9))

    # -- cycle 2: a denylist coercion (exclude bool/str/None, float() the rest)
    # launders anything else float() the builtin happens to accept through a
    # non-__float__ code path -- bytes/bytearray parse as numeric strings.
    # These pin the allowlist fix under both bound kinds: norm (bare) and px
    # (wrapped in PixelCoord, since PixelCoord.value bypasses the bare-branch
    # coercion entirely and reaches coerce_pixel_value raw either way).
    # The numpy-specific rows that used to live here (np.bool_, np.int64) are
    # cycle 3's "hand list of refused numpy shapes" the coordinator asked to
    # replace -- superseded by TestInsetBoundNumpyScalarMatrix below, which
    # derives the population from numpy's own scalar-type registry instead.

    @pytest.mark.parametrize(("case", "bad"), [("bytes", b"0.5"), ("bytearray", bytearray(b"0.5"))])
    def test_norm_kind_refuses_laundering_shapes_by_name(self, case, bad):
        with pytest.raises(ValueError, match=r"Inset\.bounds\[0\] \(left\) must be a number"):
            Inset(chart=object(), bounds=(bad, 0.1, 0.9, 0.9))

    @pytest.mark.parametrize(
        ("case", "bad"), [("bytes", b"10"), ("bytearray", bytearray(b"10")), ("None", None)]
    )
    def test_px_kind_refuses_laundering_shapes_by_name(self, case, bad):
        """``px(None)`` in particular must be refused with the *pixel*
        contract's wording, not the norm-kind message a kind-blind ``None``
        guard produced in cycle 1."""
        with pytest.raises(
            ValueError, match=r"Inset\.bounds\[0\] \(left\) must be a numeric pixel value"
        ):
            Inset(chart=object(), bounds=(px(bad), px(0), px(100), px(100)))

    def test_px_none_refused_with_pixel_wording_exact_text(self):
        with pytest.raises(
            ValueError,
            match=r"^Inset\.bounds\[0\] \(left\) must be a numeric pixel value "
            r"\(int or float\), not NoneType; received None$",
        ):
            Inset(chart=object(), bounds=(px(None), px(0), px(100), px(100)))

    # -- The remaining probe shapes: still refused, labeled (message may change). --

    @pytest.mark.parametrize(
        ("case", "bad"),
        [
            ("px str", "10px"),
            ("out-of-range norm", 1.5),
            ("negative", -0.5),
            ("nan", float("nan")),
            ("inf", float("inf")),
            ("list", [0.5]),
            ("complex", 1 + 0j),
        ],
    )
    def test_remaining_probe_shapes_refuse_with_a_labeled_error(self, case, bad):
        with pytest.raises(ValueError, match=r"Inset\.bounds\[0\] \(left\)"):
            Inset(chart=object(), bounds=(bad, 0.1, 0.9, 0.9))

    def test_negative_px_probe_shape_refuses(self):
        with pytest.raises(ValueError, match=r"Inset\.bounds\[0\] \(left\)"):
            Inset(chart=object(), bounds=(px(-5), px(0), px(10), px(10)))

    # test_pixelcoord_np_int64_probe_shape_refuses (px-kind np.int64) is
    # superseded by TestInsetBoundNumpyScalarMatrix below, which covers
    # every numpy scalar type under px, not just int64.


def _numpy_scalar_samples() -> dict:
    """One instance of every numpy concrete scalar type, keyed by type name.

    Derived from ``np.sctypeDict`` (deduplicated by type) rather than a
    hand-typed list of "the shapes a reviewer happened to probe" -- the
    exact gap that let ``np.str_``/``np.bytes_``/``np.complex128``/a
    1-element ``ndarray`` through cycle 2's duck-typed tail, and let
    ``np.timedelta64`` raise a raw ``TypeError`` (`#239` cycle 3).

    ``np.object_`` is skipped: its constructor never yields an actual
    ``np.object_`` instance for a scalar argument (``np.object_(5)`` is a
    plain ``int``, ``np.object_("x")`` is a plain ``str``) -- there is no
    "one sample" of it to construct. Each sample is built from ``0``
    (falling back to ``b"0"``/``"0"`` for the types that need it), a value
    that is finite and in ``[0, 1]`` for the numeric kinds, so an accepted
    sample also clears the fraction-range check and only the type
    acceptance/refusal boundary is exercised.

    ``np.timedelta64`` is special-cased to a *unit-qualified* sample
    (``np.timedelta64(0, "s")``, matching the reviewers' exact
    reproduction) rather than the generic trial args above: a bare,
    unit-less ``np.timedelta64(0)`` happens to support ``float()``
    (returns ``0.0``) and so would not exercise the defect this test
    exists to pin -- every real ``np.timedelta64`` a pandas/polars/numpy
    duration column actually produces carries a unit.
    """
    import numpy as np

    samples: dict = {}
    for scalar_type in sorted(set(np.sctypeDict.values()), key=lambda t: t.__name__):
        if scalar_type is np.object_:
            continue
        if scalar_type is np.timedelta64:
            samples[scalar_type.__name__] = np.timedelta64(0, "s")
            continue
        for arg in (0, 0.0, b"0", "0"):
            try:
                value = scalar_type(arg)
            except Exception:  # noqa: BLE001 - probing constructibility, not a real error path
                continue
            if type(value) is scalar_type:
                samples[scalar_type.__name__] = value
                break
        else:
            raise RuntimeError(f"no working sample constructor found for {scalar_type!r}")
    return samples


# The fourteen numpy scalar kinds `ferrum._validate.is_real_number` accepts: np.integer/np.floating concrete leaves,
# excluding np.bool_ (not a bool subclass but explicitly excluded) and
# np.timedelta64 (an np.signedinteger subclass whose float() raises).
_ACCEPTED_NUMPY_NORM_KINDS = frozenset(
    {
        "int8",
        "int16",
        "int32",
        "int64",
        "longlong",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "ulonglong",
        "float16",
        "float32",
        "float64",
        "longdouble",
    }
)

# A PixelCoord-wrapped (px-kind) bound accepts the same numpy kinds: both
# kinds ask the one real-number predicate (`ferrum._validate.is_real_number`).
_ACCEPTED_NUMPY_PX_KINDS = _ACCEPTED_NUMPY_NORM_KINDS


class TestInsetBoundNumpyScalarMatrix:
    """`#239` cycle 3: the closed allowlist's accept/refuse boundary,
    derived from numpy's own scalar-type registry (``_numpy_scalar_samples``)
    plus the non-numpy shapes the reviewers named alongside it, under both
    ``norm`` (bare) and ``px`` (``PixelCoord``-wrapped) bound kinds.
    """

    @pytest.fixture(scope="class")
    def numpy_samples(self):
        return _numpy_scalar_samples()

    def test_population_is_not_vacuous(self, numpy_samples):
        """Vacuity control: the derived population must contain both an
        accepted and a refused numpy kind, or the matrix below would pass
        trivially no matter what the allowlist does."""
        assert set(numpy_samples) & _ACCEPTED_NUMPY_NORM_KINDS
        assert set(numpy_samples) - _ACCEPTED_NUMPY_NORM_KINDS

    def test_norm_kind_accepts_only_the_named_numpy_integer_and_floating_kinds(
        self, base_chart, dual_field_df, numpy_samples
    ):
        for kind, value in numpy_samples.items():
            if kind not in _ACCEPTED_NUMPY_NORM_KINDS:
                continue
            actual = _real_inset_wire(base_chart, dual_field_df, value)
            _assert_wire_type_and_value(actual, float(value))

    def test_norm_kind_refuses_every_other_numpy_kind_by_name(self, numpy_samples):
        for kind, value in numpy_samples.items():
            if kind in _ACCEPTED_NUMPY_NORM_KINDS:
                continue
            with pytest.raises(
                ValueError, match=r"Inset\.bounds\[0\] \(left\) must be a number"
            ) as excinfo:
                Inset(chart=object(), bounds=(value, 0.5, 0.9, 0.9))
            # No raw, unlabeled TypeError/ValueError may escape (the
            # np.timedelta64 finding): pytest.raises above already pins the
            # exception class to ValueError; this additionally confirms the
            # message names the actual type, not a bare float()/isfinite
            # failure with no type name in it.
            assert "not " in str(excinfo.value)

    def test_px_kind_accepts_the_numpy_integer_and_floating_kinds(self, numpy_samples):
        for kind, value in numpy_samples.items():
            if kind not in _ACCEPTED_NUMPY_PX_KINDS:
                continue
            Inset(chart=object(), bounds=(px(value), px(0), px(100), px(100)))

    def test_px_kind_refuses_every_other_numpy_kind_by_name(self, numpy_samples):
        for kind, value in numpy_samples.items():
            if kind in _ACCEPTED_NUMPY_PX_KINDS:
                continue
            with pytest.raises(
                ValueError, match=r"Inset\.bounds\[0\] \(left\) must be a numeric pixel value"
            ):
                Inset(chart=object(), bounds=(px(value), px(0), px(100), px(100)))

    # -- Non-numpy additions the reviewers named alongside the numpy population. --

    @pytest.mark.parametrize(
        ("case", "bad"),
        [
            ("complex", 1 + 2j),
            ("str", "0.5"),
            ("bytes", b"0.5"),
            ("bytearray", bytearray(b"0.5")),
            ("bool", True),
            ("None", None),
            ("1-element ndarray", np.array([0.5])),
        ],
    )
    def test_norm_kind_refuses_non_numpy_additions_by_name(self, case, bad):
        with pytest.raises(ValueError, match=r"Inset\.bounds\[0\] \(left\) must be a number"):
            Inset(chart=object(), bounds=(bad, 0.1, 0.9, 0.9))

    @pytest.mark.parametrize(
        ("case", "bad"),
        [
            ("complex", 1 + 2j),
            ("str", "10"),
            ("bytes", b"10"),
            ("bytearray", bytearray(b"10")),
            ("bool", True),
            ("None", None),
            ("1-element ndarray", np.array([10.0])),
        ],
    )
    def test_px_kind_refuses_non_numpy_additions_by_name(self, case, bad):
        with pytest.raises(
            ValueError, match=r"Inset\.bounds\[0\] \(left\) must be a numeric pixel value"
        ):
            Inset(chart=object(), bounds=(px(bad), px(0), px(100), px(100)))


# ---------------------------------------------------------------------------
# Serialization tests (_resolve_chart_config)
# ---------------------------------------------------------------------------


class TestStructuralSerialization:
    def test_secondary_y_desugars_to_independent_layer(self, base_chart):
        """GH #52: SecondaryY no longer accumulates in ``_structural`` -- it
        desugars to an appended layer flagged ``independent_y=True`` (see
        ``_desugar_secondary_y`` in ``ferrum.chart``)."""
        chart = base_chart + SecondaryY("revenue")
        cfg = chart._resolve_chart_config()
        assert "structural" not in cfg
        assert chart._layers is not None
        assert len(chart._layers) == 2
        secondary_layer = chart._layers[-1]
        assert secondary_layer.independent_y is True
        assert secondary_layer.mark == "line"
        assert secondary_layer.encoding["y"].field == "revenue"
        # x is the base chart's written x, taken when the spec is built and
        # listed as inherited on the wire.
        assert "x" not in secondary_layer.encoding
        wire_layer = json.loads(chart.to_spec().to_json())["layers"][-1]
        assert wire_layer["encoding"]["x"]["field"] == "x"
        assert wire_layer["inherited_positional"] == ["x"]
        # Optional fields omitted when None.
        assert secondary_layer.mark_kwargs is None

    def test_secondary_y_with_optional_fields(self, base_chart):
        chart = base_chart + SecondaryY("revenue", mark="bar", color="#e45756", opacity=0.7)
        secondary_layer = chart._layers[-1]
        assert secondary_layer.mark == "bar"
        assert secondary_layer.independent_y is True
        # "bar" is a fill-primary mark, so color= resolves to "fill" (MarkBase
        # convention -- see marks/base.py _STROKE_PRIMARY_MARKS).
        assert secondary_layer.mark_kwargs == {"fill": "#e45756", "opacity": 0.7}

    def test_break_axis_single_gap_serialization(self, base_chart):
        chart = base_chart + BreakAxis(axis="y", gap=(50, 200))
        cfg = chart._resolve_chart_config()
        assert "structural" in cfg
        s = cfg["structural"][0]
        assert s["type"] == "break_axis"
        assert s["axis"] == "y"
        # Single tuple normalized to nested list
        assert s["gaps"] == [[50, 200]]
        assert s["break_size"] == 12
        assert s["break_style"] == "slash"

    def test_break_axis_multiple_gaps(self, base_chart):
        chart = base_chart + BreakAxis(axis="y", gap=[(10, 20), (50, 60)])
        cfg = chart._resolve_chart_config()
        s = cfg["structural"][0]
        assert s["gaps"] == [[10, 20], [50, 60]]

    def test_break_axis_custom_style(self, base_chart):
        chart = base_chart + BreakAxis(axis="x", gap=(0, 5), break_size=20, break_style="zigzag")
        cfg = chart._resolve_chart_config()
        s = cfg["structural"][0]
        assert s["axis"] == "x"
        assert s["break_size"] == 20
        assert s["break_style"] == "zigzag"

    def test_inset_serialization(self, base_chart, dual_field_df):
        inset_chart = fm.Chart(dual_field_df).mark_point().encode(x="x", y="revenue")
        chart = base_chart + Inset(chart=inset_chart, bounds=(0.6, 0.1, 0.95, 0.45))
        cfg = chart._resolve_chart_config()
        assert "structural" in cfg
        s = cfg["structural"][0]
        assert s["type"] == "inset"
        assert isinstance(s["svg"], str)
        assert s["svg"].startswith("<svg")
        assert s["bounds"] == [0.6, 0.1, 0.95, 0.45]
        assert s["border"] is True
        assert s["border_color"] == "#999"
        assert s["background"] == "#fff"
        assert s["shadow"] is False
        assert s["connect_style"] == "lines"
        # Optional fields omitted when None
        assert "border_dash" not in s
        assert "connect_to" not in s

    def test_inset_px_bounds_serialize_to_tagged_wire(self, base_chart, dual_field_df):
        """F-L08-08: PixelCoord bounds emit ``{"px": f}`` on the wire (Task 5's
        tagged ``InsetCoord`` contract), not the old ``TypeError`` stopgap."""
        inset_chart = fm.Chart(dual_field_df).mark_point().encode(x="x", y="revenue")
        chart = base_chart + Inset(chart=inset_chart, bounds=(px(10), px(15), px(220), px(165)))
        cfg = chart._resolve_chart_config()
        s = cfg["structural"][0]
        assert s["bounds"] == [
            {"px": 10.0},
            {"px": 15.0},
            {"px": 220.0},
            {"px": 165.0},
        ]

    def test_inset_bare_float_and_normcoord_serialize_identically(self, base_chart, dual_field_df):
        """Byte-stability: a bare float and the equivalent ``fm.norm(f)`` must
        produce the identical wire entry (both are the pre-2026-09-11 back-compat
        bare-number shape, never data-space — spec §4 H6)."""
        inset_chart = fm.Chart(dual_field_df).mark_point().encode(x="x", y="revenue")
        bare_chart = base_chart + Inset(chart=inset_chart, bounds=(0.6, 0.1, 0.95, 0.45))
        norm_chart = base_chart + Inset(
            chart=inset_chart,
            bounds=(norm(0.6), norm(0.1), norm(0.95), norm(0.45)),
        )
        bare_bounds = bare_chart._resolve_chart_config()["structural"][0]["bounds"]
        norm_bounds = norm_chart._resolve_chart_config()["structural"][0]["bounds"]
        assert bare_bounds == norm_bounds == [0.6, 0.1, 0.95, 0.45]

    def test_inset_mixed_px_and_norm_bounds_serialize_independently(
        self, base_chart, dual_field_df
    ):
        """Mixed kinds within one bounds tuple resolve independently (spec §4 H6)."""
        inset_chart = fm.Chart(dual_field_df).mark_point().encode(x="x", y="revenue")
        chart = base_chart + Inset(
            chart=inset_chart, bounds=(px(10), norm(0.1), px(220), norm(0.9))
        )
        s = chart._resolve_chart_config()["structural"][0]
        assert s["bounds"] == [{"px": 10.0}, 0.1, {"px": 220.0}, 0.9]

    def test_inset_optional_fields(self, base_chart, dual_field_df):
        inset_chart = fm.Chart(dual_field_df).mark_point().encode(x="x", y="revenue")
        chart = base_chart + Inset(
            chart=inset_chart,
            bounds=(0.5, 0.5, 1.0, 1.0),
            border_dash=[4, 4],
            connect_to=(2.0, 20.0),
            shadow=True,
            background=None,
        )
        cfg = chart._resolve_chart_config()
        s = cfg["structural"][0]
        assert s["border_dash"] == [4, 4]
        assert s["connect_to"] == [2.0, 20.0]
        assert s["shadow"] is True
        assert s["background"] is None

    def test_multiple_structural_features(self, base_chart):
        """SecondaryY desugars to a layer; only BreakAxis remains structural."""
        chart = base_chart + SecondaryY("revenue") + BreakAxis(axis="y", gap=(10, 90))
        cfg = chart._resolve_chart_config()
        assert "structural" in cfg
        assert len(cfg["structural"]) == 1
        assert cfg["structural"][0]["type"] == "break_axis"
        assert chart._layers is not None
        assert chart._layers[-1].independent_y is True

    def test_no_structural_key_when_empty(self, base_chart):
        cfg = base_chart._resolve_chart_config()
        assert "structural" not in cfg

    def test_structural_with_configure(self, base_chart):
        from ferrum.configure import Configure, AxisConfig

        chart = base_chart + Configure(axis=AxisConfig(grid=True)) + SecondaryY("revenue")
        cfg = chart._resolve_chart_config()
        # SecondaryY desugars to a layer, not a structural entry.
        assert "structural" not in cfg
        assert chart._layers[-1].independent_y is True
        # Configure entry is still present.
        assert "axis" in cfg


# ---------------------------------------------------------------------------
# Rendering integration tests (to_svg round-trip)
# ---------------------------------------------------------------------------


class TestStructuralRendering:
    def test_secondary_y_renders_svg(self, base_chart):
        chart = base_chart + SecondaryY("revenue")
        svg = chart.to_svg()
        assert svg.startswith("<svg")

    def test_break_axis_renders_svg(self, base_chart):
        """The smoke test must actually smoke: the gap has to fall INSIDE the
        chart's y domain (10..50) or no break is applied at all.

        The fixture used ``gap=(50, 200)`` -- entirely above the domain -- so
        the render exercised the no-break path and proved nothing. The Rust
        ``BreakAxisIneffective`` warning added in the batch-H close
        remediation is what surfaced it; asserting the warning does NOT fire
        makes the fixture self-checking against the same drift.
        """
        chart = base_chart + BreakAxis(axis="y", gap=(20, 40))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svg = chart.to_svg()
        assert svg.startswith("<svg")
        assert not [w for w in caught if "had no effect" in str(w.message)], (
            "gap=(20, 40) is inside the y domain, so the break must take effect: "
            f"{[str(w.message) for w in caught]}"
        )

    def test_inset_renders_svg(self, base_chart, dual_field_df):
        inset_chart = fm.Chart(dual_field_df).mark_point().encode(x="x", y="revenue")
        chart = base_chart + Inset(chart=inset_chart, bounds=(0.6, 0.1, 0.95, 0.45))
        svg = chart.to_svg()
        assert svg.startswith("<svg")

    def test_structural_with_configure_renders(self, base_chart):
        from ferrum.configure import Configure, AxisConfig

        chart = base_chart + Configure(axis=AxisConfig(grid=True)) + SecondaryY("revenue")
        svg = chart.to_svg()
        assert svg.startswith("<svg")

    def test_inset_px_bounds_render_at_expected_rect(self, base_chart, dual_field_df):
        """F-L08-08: a px() inset places its nested ``<svg>`` at exactly the plot
        rect's origin plus the given pixel offsets — no scaling, unlike norm.

        Derives the parent's own plot-rect origin/span from a full-plot-area
        norm(0)/norm(1) inset (rather than hardcoding chrome margins), then
        asserts a px-bounded inset's nested ``<svg>`` rect equals
        ``origin + px`` for each side — the exact contract Task 5 wired in
        ``build_inset_nodes`` (``InsetCoord::Px.resolve``).
        """
        inset_chart = fm.Chart(dual_field_df).mark_point().encode(x="x", y="revenue")

        full_plot_chart = base_chart + Inset(
            chart=inset_chart, bounds=(norm(0.0), norm(0.0), norm(1.0), norm(1.0))
        )
        origin_rect = _nested_svg_rects(full_plot_chart.to_svg())[1]
        origin_x = float(origin_rect["x"])
        origin_y = float(origin_rect["y"])

        left_px, top_px, right_px, bottom_px = 10.0, 15.0, 220.0, 165.0
        px_chart = base_chart + Inset(
            chart=inset_chart,
            bounds=(px(left_px), px(top_px), px(right_px), px(bottom_px)),
        )
        inset_rect = _nested_svg_rects(px_chart.to_svg())[1]

        assert float(inset_rect["x"]) == pytest.approx(origin_x + left_px)
        assert float(inset_rect["y"]) == pytest.approx(origin_y + top_px)
        assert float(inset_rect["width"]) == pytest.approx(right_px - left_px)
        assert float(inset_rect["height"]) == pytest.approx(bottom_px - top_px)

    def test_inset_bare_float_and_normcoord_render_byte_identically(
        self, base_chart, dual_field_df
    ):
        """Byte-stability (spec §7): in-range bare-float and NormCoord bounds
        render byte-identically."""
        inset_chart = fm.Chart(dual_field_df).mark_point().encode(x="x", y="revenue")
        bare_svg = (base_chart + Inset(chart=inset_chart, bounds=(0.6, 0.1, 0.95, 0.45))).to_svg()
        norm_svg = (
            base_chart
            + Inset(chart=inset_chart, bounds=(norm(0.6), norm(0.1), norm(0.95), norm(0.45)))
        ).to_svg()
        assert bare_svg == norm_svg

    def test_inset_mixed_px_and_norm_bounds_renders(self, base_chart, dual_field_df):
        """Mixed px/norm bounds in one tuple render without error (spec §4 H6)."""
        inset_chart = fm.Chart(dual_field_df).mark_point().encode(x="x", y="revenue")
        chart = base_chart + Inset(
            chart=inset_chart, bounds=(px(10), norm(0.1), px(220), norm(0.6))
        )
        svg = chart.to_svg()
        assert svg.startswith("<svg")
        assert len(_nested_svg_rects(svg)) == 2


# ---------------------------------------------------------------------------
# SecondaryY desugar (GH #52) -- appends an independent-y layer
# ---------------------------------------------------------------------------


class TestSecondaryYDesugar:
    """Pins the GH #52 re-basing: SecondaryY appends a layer flagged
    ``independent_y=True`` rather than accumulating a structural spec entry.
    """

    def test_layered_base_plus_secondary_y_keeps_base_sharing(self, dual_field_df):
        """A two-layer base built via ``a + b`` keeps its own layers sharing
        the left axis; only the appended SecondaryY layer becomes
        independent-y (the base is unrestricted by the desugar)."""
        a = fm.Chart(dual_field_df).mark_point().encode(x="x", y="y")
        b = fm.Chart(dual_field_df).mark_line().encode(x="x", y="y")
        layered = a + b
        chart = layered + SecondaryY("revenue", mark="bar")

        assert len(chart._layers) == 3
        base_layer_0, base_layer_1, secondary_layer = chart._layers
        assert base_layer_0.independent_y is False
        assert base_layer_1.independent_y is False
        assert secondary_layer.independent_y is True
        assert secondary_layer.mark == "bar"
        assert secondary_layer.encoding["y"].field == "revenue"
        # x is inherited from the base chart when the spec is built, not
        # re-specified per SecondaryY.
        assert "x" not in secondary_layer.encoding
        wire_layer = json.loads(chart.to_spec().to_json())["layers"][-1]
        assert wire_layer["encoding"]["x"]["field"] == "x"
        assert wire_layer["inherited_positional"] == ["x"]

        svg = chart.to_svg()
        assert svg.startswith("<svg")

    def test_multiple_secondary_y_stack_right_axes(self, dual_field_df):
        """Multiple ``+ SecondaryY(...)`` additions append multiple
        independent-y layers, in order (slot contract: layer 0 is always
        primary; each subsequent independent layer stacks another right
        axis)."""
        base = fm.Chart(dual_field_df).mark_point().encode(x="x", y="y")
        chart = base + SecondaryY("revenue") + SecondaryY("revenue", mark="point")

        assert len(chart._layers) == 3
        assert chart._layers[0].independent_y is False
        assert chart._layers[1].independent_y is True
        assert chart._layers[2].independent_y is True

        svg = chart.to_svg()
        assert svg.startswith("<svg")

    def test_secondary_y_reserves_right_margin_band(self, dual_field_df):
        """The dual-axis render narrows the plot area vs. the base chart
        alone -- the upgrade over the legacy overlay-only renderer, which
        drew the secondary axis over the plot area without reserving space
        for it."""
        import re

        base = fm.Chart(dual_field_df).mark_point().encode(x="x", y="y")
        svg_without = base.to_svg()
        svg_with = (base + SecondaryY("revenue")).to_svg()

        def rightmost_x_tick(svg: str) -> float:
            m = re.search(
                r'<text x="([0-9.]+)" y="[^"]+"[^>]*text-anchor="middle"[^>]*>5</text>',
                svg,
            )
            assert m, "could not find x-axis tick label '5'"
            return float(m.group(1))

        assert rightmost_x_tick(svg_with) < rightmost_x_tick(svg_without)

    def test_secondary_y_requires_base_x_encoding(self):
        """SecondaryY's x is inherited from the base chart; a base chart with
        no x encoding cannot be desugared."""
        df = pl.DataFrame({"y": [1, 2, 3], "revenue": [10, 20, 30]})
        base = fm.Chart(df).mark_point().encode(y="y")
        with pytest.raises(ValueError, match="x encoding"):
            base + SecondaryY("revenue")

    def test_secondary_y_unknown_field_raises_at_add_time(self):
        """SecondaryY reads its field from the base chart's own table (the
        desugar performs no data merge); a field that is not a column raises
        a boundary ValueError at ``+`` time instead of a downstream Rust
        column error."""
        df = pl.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
        base = fm.Chart(df).mark_point().encode(x="x", y="y")
        with pytest.raises(ValueError, match=r"'reevnue'.*not a column"):
            base + SecondaryY("reevnue")

    def test_secondary_y_pyarrow_table_valid_field_accepted(self):
        """`Chart._data` is stored raw, and pyarrow's ``.columns`` yields
        column OBJECTS, not names — the field guard must be type-aware, so a
        VALID field on a pyarrow ``Table`` base must not be rejected."""
        import pyarrow as pa

        table = pa.table({"x": [1, 2, 3], "y": [1, 2, 3], "rev": [10, 20, 30]})
        base = fm.Chart(table).mark_bar().encode(x="x", y="y")
        layered = base + SecondaryY("rev")  # must not raise
        assert layered._layers[-1].independent_y

    def test_secondary_y_pyarrow_table_unknown_field_raises(self):
        import pyarrow as pa

        table = pa.table({"x": [1, 2, 3], "y": [1, 2, 3]})
        base = fm.Chart(table).mark_bar().encode(x="x", y="y")
        with pytest.raises(ValueError, match=r"'rev'.*not a column"):
            base + SecondaryY("rev")
