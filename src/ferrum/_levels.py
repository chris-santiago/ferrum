"""The levels a categorical column draws.

A null value in a category, grouping or facet column belongs to no band, group or panel: the
engine drops its rows. Anything that counts or lists a column's levels to size or place
something (a facet grid, a canvas height, an annotation's band centre, a heatmap's aspect) reads
them through :func:`distinct_non_null_levels` so it agrees with what is drawn.
"""

from __future__ import annotations

from typing import Any

import polars as pl


def distinct_non_null_levels(column: pl.Series | Any) -> list:
    """Return the distinct non-null values of *column*, in order of first appearance.

    *column* is a polars ``Series`` or a pyarrow ``Array``/``ChunkedArray``. A float NaN is not
    null, so it is a level of its own.
    """
    if isinstance(column, pl.Series):
        return column.drop_nulls().unique(maintain_order=True).to_list()
    return [value for value in column.unique().to_pylist() if value is not None]
