"""The usable values of a column.

A value is usable when it is present and, on a float column, finite: a null, a NaN, ``+inf`` and
``-inf`` are not. It is the rule the engine's statistics read through ``numeric_util::usable_at``,
and every Python fold over a caller's column (a domain, a mean and deviation, an extent, a
curve's area) reads its values through these two functions so it agrees with what the engine
computes.
"""

from __future__ import annotations

import polars as pl


def usable(column: str, dtype: pl.DataType) -> pl.Expr:
    """Return the expression that is true for each usable value of *column*.

    Parameters
    ----------
    column : str
        The column's name.
    dtype : polars.DataType
        The column's dtype. A float column's usable values are non-null and finite; any other
        dtype holds no NaN or infinity, so its usable values are its non-null ones.

    Returns
    -------
    polars.Expr
        A boolean expression, never null.
    """
    present = pl.col(column).is_not_null()
    return present & pl.col(column).is_finite() if dtype.is_float() else present


def usable_values(series: pl.Series) -> pl.Series:
    """Return the usable values of *series*, in order.

    Parameters
    ----------
    series : polars.Series
        The values to read.

    Returns
    -------
    polars.Series
        *series* without its nulls and, on a float series, without its NaN and infinite values.
    """
    return series.filter(series.is_finite()) if series.dtype.is_float() else series.drop_nulls()
