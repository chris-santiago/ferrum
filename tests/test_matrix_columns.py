"""``heatmap`` and ``clustermap`` split their input by one rule, ``_matrix_columns``.

The value columns are the numeric ones; the id column is the first column that
is not numeric, or none when every column is. A table with no numeric column
refuses, voiced as the figure function the caller invoked.
"""

from __future__ import annotations

import re

import polars as pl
import pytest

import ferrum as fm
from ferrum._coerce import to_arrow_table
from ferrum.plots.matrix import _matrix_columns

_ENTRIES = ("heatmap", "clustermap")


@pytest.mark.parametrize("entry", _ENTRIES)
def test_a_table_with_no_numeric_column_refuses_in_the_entry_voice(entry: str) -> None:
    df = pl.DataFrame({"a": ["p", "q"], "b": ["r", "s"]})
    with pytest.raises(ValueError, match=re.escape(f"{entry}: no numeric columns found in data")):
        getattr(fm, entry)(df)


def test_the_id_column_is_the_first_non_numeric_one() -> None:
    df = pl.DataFrame({"v1": [1.0, 2.0], "gene": ["g1", "g2"], "v2": [3.0, 4.0], "tag": ["t", "u"]})
    assert _matrix_columns(to_arrow_table(df), "heatmap") == (["v1", "v2"], "gene")


def test_an_all_numeric_table_has_no_id_column() -> None:
    df = pl.DataFrame({"v1": [1.0, 2.0], "v2": [3.0, 4.0]})
    assert _matrix_columns(to_arrow_table(df), "clustermap") == (["v1", "v2"], None)
