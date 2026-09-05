"""Window transform: tie-aware ranks, peer-aware frames, dtype ingestion.

Covers batch-D M4 (F-L05-08 tie ranks, F-L05-10 ``ignore_peers``) and the
window half of the M1 dtype policy (F-L05-02: a ``sort=`` over an integer
column used to be a silent no-op).

Every behavioral test runs through the production entry point
(``Chart(...).transform(...).to_svg()``) because the Rust window internals are
``pub(crate)``: a ``mark_text`` layer whose ``text`` channel *is* the computed
window column prints the values into the SVG, so the assertions below read the
transform's real output rather than a proxy for it. The hand-computed
arithmetic behind each expectation is spelled out in the docstrings, and is
pinned a second time at the Rust unit level in ``data_window.rs``.
"""

from __future__ import annotations

import datetime
import re

import polars as pl
import pytest

import ferrum as fm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mark_labels(svg: str, x_title: str) -> list[float]:
    """The text-mark labels, in data row order.

    Ferrum emits y-axis ticks, the y title, x-axis ticks, the x title, then the
    marks — so everything after the x title is a mark label.
    """
    texts = re.findall(r"<text[^>]*>([^<]*)</text>", svg)
    assert x_title in texts, f"x-axis title {x_title!r} not in {texts!r}"
    return [float(t) for t in texts[texts.index(x_title) + 1 :]]


def _window_values(df: pl.DataFrame, transform: dict, column: str) -> list[float]:
    """Render ``df`` through ``transform`` and read back ``column``'s values."""
    chart = (
        fm.Chart(df)
        .mark_text()
        .encode(x="idx:Q", y=f"{column}:Q", text=f"{column}:Q")
        .transform(transform)
    )
    return _mark_labels(chart.to_svg(), "idx")


def _scores_frame() -> pl.DataFrame:
    """The finding's fixture: tied scores that must draw tied ranks."""
    return pl.DataFrame(
        {
            "idx": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "score": [5.0, 1.0, 3.0, 1.0, 5.0, 3.0],
        }
    )


def _tied_frame() -> pl.DataFrame:
    """Sort key ``t`` with two peer runs of two around a singleton."""
    return pl.DataFrame(
        {
            "idx": [0.0, 1.0, 2.0, 3.0, 4.0],
            "t": [1.0, 1.0, 2.0, 3.0, 3.0],
            "x": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )


# ---------------------------------------------------------------------------
# M4a — rank / dense_rank tie on the sort key (F-L05-08)
# ---------------------------------------------------------------------------


def test_field_less_rank_draws_competition_ranks_from_the_sort_key():
    """The docs' own ``{"op": "rank"}`` over tied scores.

    Sorted, the scores read 1, 1, 3, 3, 5, 5 → competition ranks
    1, 1, 3, 3, 5, 5, which in input order is [5, 1, 3, 1, 5, 3].
    Before M4 a field-less rank degraded to ``row_number`` (1..6).
    """
    t = fm.transform_window({"op": "rank", "as": "r"}, sort=["score"])
    assert _window_values(_scores_frame(), t, "r") == [5.0, 1.0, 3.0, 1.0, 5.0, 3.0]


def test_field_less_dense_rank_draws_dense_ranks_from_the_sort_key():
    """Same fixture, dense semantics: 1, 1, 2, 2, 3, 3 in sorted order."""
    t = fm.transform_window({"op": "dense_rank", "as": "d"}, sort=["score"])
    assert _window_values(_scores_frame(), t, "d") == [3.0, 1.0, 2.0, 1.0, 3.0, 2.0]


def test_rank_without_sort_stays_row_number():
    """With no sort key there is nothing to tie on, so ranks are unchanged."""
    t = fm.transform_window({"op": "rank", "as": "r"})
    assert _window_values(_scores_frame(), t, "r") == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_explicit_field_rank_keys_on_that_field_not_the_sort():
    """``field=`` keeps its documented field-keyed behavior.

    Sorted by ``idx``, ``value`` reads 7, 7, 9, 9 → competition ranks
    1, 1, 3, 3 — the sort column ``idx`` (all distinct) does not drive them.
    """
    df = pl.DataFrame(
        {
            "idx": [0.0, 1.0, 2.0, 3.0],
            "value": [7.0, 7.0, 9.0, 9.0],
        }
    )
    t = fm.transform_window({"op": "rank", "field": "value", "as": "r"}, sort=["idx"])
    assert _window_values(df, t, "r") == [1.0, 1.0, 3.0, 3.0]


def test_ranks_restart_per_group():
    """Ties are scoped to the partition, as they were before M4."""
    df = pl.DataFrame(
        {
            "idx": [0.0, 1.0, 2.0, 3.0],
            "g": ["a", "a", "b", "b"],
            "score": [2.0, 2.0, 5.0, 7.0],
        }
    )
    t = fm.transform_window({"op": "rank", "as": "r"}, sort=["score"], groupby=["g"])
    assert _window_values(df, t, "r") == [1.0, 1.0, 1.0, 2.0]


# ---------------------------------------------------------------------------
# M4b — ignore_peers: peer-aware vs row-based frames (F-L05-10)
# ---------------------------------------------------------------------------


def test_peer_aware_frames_are_the_default_and_tied_rows_share_a_value():
    """Rolling ``sum`` over ``frame=(-1, 0)`` with sort key [1,1,2,3,3].

    Each peer run gets ONE frame, anchored on the run's own bounds and grown
    to whole runs, so tied rows share a value:

    * run [0, 1] (t=1): rows [0-1 ..= 1+0] → [0..1] → 10 + 20 = 30
    * run [2]    (t=2): rows [2-1 ..= 2]  → grown to [0..2] → 10 + 20 + 30 = 60
    * run [3, 4] (t=3): rows [3-1 ..= 4]  → [2..4] → 30 + 40 + 50 = 120
    """
    t = fm.transform_window({"op": "sum", "field": "x", "as": "s"}, sort=["t"], frame=(-1, 0))
    assert _window_values(_tied_frame(), t, "s") == [30.0, 30.0, 60.0, 120.0, 120.0]


def test_ignore_peers_true_reproduces_row_based_frames():
    """The escape hatch: pure row offsets, i.e. the pre-M4 output.

    pos0 [0..0] = 10, pos1 [0..1] = 30, pos2 [1..2] = 50,
    pos3 [2..3] = 70, pos4 [3..4] = 90.
    """
    t = fm.transform_window(
        {"op": "sum", "field": "x", "as": "s"},
        sort=["t"],
        frame=(-1, 0),
        ignore_peers=True,
    )
    assert _window_values(_tied_frame(), t, "s") == [10.0, 30.0, 50.0, 70.0, 90.0]


def test_untied_sort_key_renders_identically_under_both_modes():
    """The disclosed default change reaches only frames over a tied sort key."""
    df = pl.DataFrame(
        {
            "idx": [0.0, 1.0, 2.0, 3.0, 4.0],
            "t": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    op = {"op": "sum", "field": "x", "as": "s"}
    peer_aware = fm.transform_window(op, sort=["t"], frame=(-1, 0))
    row_based = fm.transform_window(op, sort=["t"], frame=(-1, 0), ignore_peers=True)
    assert _window_values(df, peer_aware, "s") == [10.0, 30.0, 50.0, 70.0, 90.0]
    assert _window_values(df, peer_aware, "s") == _window_values(df, row_based, "s")


def test_ignore_peers_defaults_off_and_only_serializes_when_set():
    """Wire stability: specs written before this parameter are unchanged."""
    op = {"op": "sum", "field": "x", "as": "s"}
    assert "ignore_peers" not in fm.transform_window(op, sort=["t"])
    assert "ignore_peers" not in fm.transform_window(op, sort=["t"], ignore_peers=False)
    assert fm.transform_window(op, ignore_peers=True)["ignore_peers"] is True


def test_ignore_peers_true_leaves_rank_and_dense_rank_untouched():
    """``ignore_peers`` is documented as scoping only rolling aggregates with a
    ``frame``. Passing it on ``rank``/``dense_rank`` must not flip them to
    row-based ranks — the shared ``PeerGroups`` machinery makes this a
    plausible one-line future mistake if a branch ever started reading
    ``spec.ignore_peers`` for ranks too.
    """
    rank = fm.transform_window({"op": "rank", "as": "r"}, sort=["score"], ignore_peers=True)
    dense = fm.transform_window({"op": "dense_rank", "as": "d"}, sort=["score"], ignore_peers=True)
    assert _window_values(_scores_frame(), rank, "r") == [5.0, 1.0, 3.0, 1.0, 5.0, 3.0]
    assert _window_values(_scores_frame(), dense, "d") == [3.0, 1.0, 2.0, 1.0, 3.0, 2.0]


def test_frame_entirely_before_anchor_is_empty():
    """A frame that ends before the current row selects no rows, rather than
    wrapping around to the whole partition (Rust-level pin:
    ``frame_entirely_before_the_anchor_is_empty``). Positions 0 and 1 both
    land on an empty window and are dropped as non-finite; only position 2's
    finite mean survives to render.
    """
    df = pl.DataFrame({"idx": [0.0, 1.0, 2.0], "x": [1.0, 2.0, 3.0]})
    t = fm.transform_window({"op": "mean", "field": "x", "as": "m"}, frame=(-3, -2))
    assert _window_values(df, t, "m") == [1.0]


# ---------------------------------------------------------------------------
# M1 — window dtype ingestion (the window half of F-L05-02)
# ---------------------------------------------------------------------------


def test_null_sort_value_orders_last():
    """Nulls in a sort column order last and tie with each other, rather than
    comparing equal to every value (Rust-level pin:
    ``null_sort_values_order_last_and_tie_together``).
    """
    df = pl.DataFrame(
        {
            "idx": [0.0, 1.0, 2.0, 3.0],
            "t": [None, 2.0, None, 1.0],
        }
    )
    t = fm.transform_window({"op": "rank", "as": "r"}, sort=["t"])
    assert _window_values(df, t, "r") == [3.0, 2.0, 3.0, 1.0]


def test_int_sort_column_orders_a_cumulative_sum():
    """The guide's cumsum-over-``day`` example, with the days shuffled.

    ``day`` is an integer column, which used to compare ``Ordering::Equal``
    and leave the window unsorted (so the "cumulative" sum accumulated in row
    order). In day order the running total is 10, 30, 60.
    """
    df = pl.DataFrame(
        {
            "idx": [0.0, 1.0, 2.0],
            "day": pl.Series([3, 1, 2], dtype=pl.Int64),
            "sales": [30.0, 10.0, 20.0],
        }
    )
    t = fm.transform_window(
        {"op": "sum", "field": "sales", "as": "cumulative"},
        sort=["day"],
        frame=(None, 0),
    )
    assert _window_values(df, t, "cumulative") == [60.0, 10.0, 30.0]


@pytest.mark.parametrize("date_dtype", [pl.Date, pl.Datetime])
def test_temporal_sort_column_orders_a_cumulative_sum(date_dtype):
    """``Timestamp`` acceptance is a Python-boundary claim: ferrum pre-casts
    ``Date32``/``Date64`` to ``Timestamp(ms)``, so a date-sorted window is the
    one place this dtype decision is actually exercised end to end. Same
    shuffled-day cumsum shape as the integer case above, built from a
    ``pl.Date`` (and ``pl.Datetime``) sort column instead of an int.
    """
    df = pl.DataFrame(
        {
            "idx": [0.0, 1.0, 2.0],
            "day": pl.Series(
                [
                    datetime.date(2024, 1, 3),
                    datetime.date(2024, 1, 1),
                    datetime.date(2024, 1, 2),
                ],
                dtype=date_dtype,
            ),
            "sales": [30.0, 10.0, 20.0],
        }
    )
    t = fm.transform_window(
        {"op": "sum", "field": "sales", "as": "cumulative"},
        sort=["day"],
        frame=(None, 0),
    )
    assert _window_values(df, t, "cumulative") == [60.0, 10.0, 30.0]


@pytest.mark.parametrize("dtype", [pl.Int64, pl.Int32, pl.UInt8, pl.Float32])
def test_integer_and_float_value_columns_feed_lag_and_rolling(dtype):
    """A non-Float64 value column no longer reads back as all-NaN."""
    df = pl.DataFrame(
        {
            "idx": [0.0, 1.0, 2.0],
            "x": pl.Series([1, 2, 3], dtype=dtype),
        }
    )
    rolling = fm.transform_window({"op": "sum", "field": "x", "as": "s"}, frame=(-1, 0))
    assert _window_values(df, rolling, "s") == [1.0, 3.0, 5.0]

    lagged = fm.transform_window({"op": "lag", "field": "x", "as": "lag_x"})
    # The first row has no predecessor, so its label is dropped as non-finite;
    # the remaining two carry the preceding values.
    assert _window_values(df, lagged, "lag_x") == [1.0, 2.0]


def test_unsupported_sort_dtype_raises_naming_column_and_dtype():
    df = pl.DataFrame(
        {
            "idx": [0.0, 1.0],
            "flag": pl.Series([True, False], dtype=pl.Boolean),
        }
    )
    t = fm.transform_window({"op": "rank", "as": "r"}, sort=["flag"])
    chart = fm.Chart(df).mark_point().encode(x="idx:Q", y="r:Q").transform(t)
    with pytest.raises(ValueError, match=r"data_window.*'flag'.*Boolean"):
        chart.to_svg()


def test_text_value_column_refuses_to_aggregate():
    """A string field raises instead of silently aggregating to NaN."""
    df = pl.DataFrame({"idx": [0.0, 1.0], "label": ["a", "b"]})
    t = fm.transform_window({"op": "mean", "field": "label", "as": "m"})
    chart = fm.Chart(df).mark_point().encode(x="idx:Q", y="m:Q").transform(t)
    with pytest.raises(ValueError, match=r"data_window.*numeric.*'label'"):
        chart.to_svg()


def test_missing_sort_column_still_names_the_sort_role():
    df = pl.DataFrame({"idx": [0.0, 1.0], "x": [1.0, 2.0]})
    t = fm.transform_window({"op": "rank", "as": "r"}, sort=["ghost"])
    chart = fm.Chart(df).mark_point().encode(x="idx:Q", y="r:Q").transform(t)
    with pytest.raises(ValueError, match=r"data_window: sort column 'ghost' not found"):
        chart.to_svg()
