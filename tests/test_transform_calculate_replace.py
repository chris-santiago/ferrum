"""Pins ``fm.transform_calculate(as_=, expr=)``'s column-replace and column-append contract.

With an ``as_`` naming a column the batch already holds, ``Calculate``
overwrites that column at its schema position with the expression's values
and dtype, so a batch never holds two columns of one name. With an ``as_``
the batch doesn't already hold, it appends a new column and every existing
column is unchanged.

Two independent render paths exercise two independent Rust consumers for
the replace case, so agreement between them (and with the expected
post-calculate values) is only possible when there is exactly one live
column under that name, at whatever schema position it already held:

- ``mark_text``'s ``text=`` channel formats the field's cell value
  directly, including a dtype change (``Float64`` -> ``Utf8``) that only
  the replaced column's dtype can produce.
- A positional channel's own scale resolves its axis extent from the same
  field; the extent only reflects the calculated values when the position
  encoder reads the one replaced column too.

The append case is checked by folding the original column together with
the new one (``transform_fold``): ``z`` exists under its own name, ``x``
is unchanged, and ``z`` holds the expression's values.
"""

from __future__ import annotations

import re

import polars as pl

import ferrum as fm


def _mark_text_values(svg: str) -> list[str]:
    """Rendered ``mark_text`` cell content, in document order.

    Ferrum's default theme draws mark text in the foreground color
    (``fill="#1f2937"``) with no ``font-weight``; axis titles share the
    foreground color but carry ``font-weight="600"``, and tick labels use
    the muted tick color (``#6b7280``), so this pattern reaches only the
    mark's own rendered values.
    """
    return re.findall(
        r'<text x="[^"]+" y="[^"]+" fill="#1f2937" font-family="Inter" font-size="11" '
        r'text-anchor="middle">([^<]+)</text>',
        svg,
    )


def _y_axis_numeric_ticks(svg: str) -> list[str]:
    """The y-axis's rendered numeric tick labels, in document order."""
    return re.findall(r'<text[^>]*text-anchor="end"[^>]*>([0-9.]+)</text>', svg)


def _y_axis_category_ticks(svg: str) -> list[str]:
    """The y-axis's rendered nominal/ordinal tick labels, in document order."""
    return re.findall(r'<text[^>]*fill="#6b7280"[^>]*text-anchor="end"[^>]*>([^<]+)</text>', svg)


class TestExistingColumnReplaced:
    """``Calculate(as_="x", ...)`` overwrites the existing ``x`` column in place."""

    def test_text_formatting_reads_the_replaced_value_and_dtype(self):
        df = pl.DataFrame({"id": ["a", "b", "c"], "x": [1.0, 2.0, 3.0]})
        chart = (
            fm.Chart(df)
            .mark_text()
            .encode(x=fm.X("id", type_="N"), y=fm.Y("id", type_="N"), text="x")
            .transform(fm.transform_calculate(as_="x", expr='datum.x > 1.5 ? "big" : "small"'))
        )
        svg = chart.to_svg()
        assert _mark_text_values(svg) == ["small", "big", "big"]

    def test_positional_scale_extent_reads_the_replaced_value(self):
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        chart = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x", y="x")
            .transform(fm.transform_calculate(as_="x", expr="datum.x * 100"))
        )
        svg = chart.to_svg()
        ticks = [float(t) for t in _y_axis_numeric_ticks(svg)]
        assert max(ticks) >= 300.0
        assert min(ticks) <= 100.0


class TestNewColumnAppends:
    """``Calculate(as_="z", ...)`` with a name the batch doesn't hold appends a column."""

    def test_appends_one_column_and_leaves_the_existing_one_unchanged(self):
        df = pl.DataFrame({"id": ["r0", "r1", "r2"], "x": [1.0, 2.0, 3.0]})
        chart = (
            fm.Chart(df)
            .mark_text()
            .encode(x=fm.X("id", type_="N"), y=fm.Y("key", type_="N"), text="value")
            .transform(fm.transform_calculate(as_="z", expr="datum.x * 2"))
            .transform(fm.transform_fold(["x", "z"], as_=("key", "value")))
        )
        svg = chart.to_svg()
        # Folding "x" together with "z" shows "z" exists under its own
        # name alongside "x".
        assert _y_axis_category_ticks(svg) == ["x", "z"]
        values = _mark_text_values(svg)
        assert values[:3] == ["1", "2", "3"]  # the existing column, unchanged
        assert values[3:] == ["2", "4", "6"]  # the appended column's calculated values
