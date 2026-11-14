"""``desugar_function`` owns the units of the parent column it infers a domain from.

``Chart.mark_function`` hands over the parent x column as the chart holds it; the desugar reads a
floating, integer or decimal column as float64 and a timestamp column in epoch milliseconds, and
refuses a column of text or booleans naming the column.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pyarrow as pa
import pytest

from ferrum.marks.heavy_stat import desugar_function


def _domain(column: pa.ChunkedArray) -> tuple[float, float]:
    xs = desugar_function(np.sin, column, n=5).data["x"].to_numpy()
    return float(xs[0]), float(xs[-1])


@pytest.mark.parametrize(
    "column",
    [
        pytest.param(pa.chunked_array([[1.5, 4.0, 2.0]]), id="float"),
        pytest.param(pa.chunked_array([pa.array([2, 4, 1], pa.int32())]), id="int"),
    ],
)
def test_a_numeric_parent_column_gives_its_own_extent(column: pa.ChunkedArray):
    low, high = _domain(column)
    assert (low, high) == (min(column.to_pylist()), max(column.to_pylist()))


def test_a_timestamp_parent_column_is_read_in_epoch_milliseconds():
    first, last = dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 3)
    column = pa.chunked_array([pa.array([first, last], pa.timestamp("us"))])
    low, high = _domain(column)
    epoch = dt.datetime(1970, 1, 1)
    assert (low, high) == (
        (first - epoch).total_seconds() * 1000,
        (last - epoch).total_seconds() * 1000,
    )


def test_a_text_parent_column_is_refused_by_name():
    column = pa.chunked_array([["a", "b"]])
    with pytest.raises(ValueError, match="x column 'when' holds neither numbers nor times"):
        desugar_function(np.sin, column, parent_chart_x_name="when")


def test_an_explicit_domain_reads_no_parent_column():
    column = pa.chunked_array([["a", "b"]])
    xs = desugar_function(np.sin, column, domain=(0.0, 1.0), n=3).data["x"].to_pylist()
    assert xs == [0.0, 0.5, 1.0]


def test_no_parent_column_and_no_domain_is_refused():
    with pytest.raises(ValueError, match="requires explicit domain"):
        desugar_function(np.sin)
