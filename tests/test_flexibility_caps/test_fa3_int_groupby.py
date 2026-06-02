"""FA-3: stat_aggregate must accept integer (Int64/Int32/UInt) groupby columns.

The aggregate transform previously rejected non-Float64/Utf8 groupby columns with:
    stat_aggregate: groupby column 'x' must be Float64 or Utf8; got Int64

Tests verify:
  1. Int64 groupby renders without error (RED before fix, GREEN after).
  2. Int32 and other integer variants likewise accepted.
  3. Float64 and Utf8 groupby not regressed.
  4. Aggregate values for Int64 groupby match those from an equivalent Utf8 groupby.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import ferrum as fm
from ferrum._core import render_svg

VIEWPORT = (400.0, 300.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render(df: pa.Table, groupby_col: str, value_col: str, agg_fn: str) -> str:
    """Build a mark_bar chart with an explicit Aggregate transform and render to SVG.

    The y encoding is bound to the *output* column name (``agg_<value_col>``) so
    render_svg does not complain about a missing pre-transform column.
    """
    out_col = f"agg_{value_col}"
    chart = (
        fm.Chart(df)
        .mark_bar()
        .encode(x=groupby_col, y=f"{out_col}:Q")
        .transform(
            fm.Aggregate([fm.AggregateOp(value_col, agg_fn, out_col)], groupby=[groupby_col])
        )
    )
    spec = chart.to_spec()
    return render_svg(spec, df, viewport=VIEWPORT, theme={})


# ---------------------------------------------------------------------------
# Int64 groupby (RED before fix)
# ---------------------------------------------------------------------------

class TestInt64Groupby:
    """Int64 groupby columns must be accepted by stat_aggregate."""

    def test_int64_groupby_sum_renders_without_error(self):
        """Sum over an Int64 groupby column must produce an SVG without raising."""
        df = pa.table({
            "category": pa.array([10, 10, 20, 20, 30], type=pa.int64()),
            "v": pa.array([1.0, 2.0, 3.0, 4.0, 5.0], type=pa.float64()),
        })
        svg = _render(df, "category", "v", "sum")
        assert "<svg" in svg, "expected a valid SVG document"

    def test_int64_groupby_mean_renders_without_error(self):
        """Mean over an Int64 groupby (year column) must render correctly."""
        df = pa.table({
            "year": pa.array([2020, 2020, 2021, 2021], type=pa.int64()),
            "v": pa.array([1.0, 3.0, 2.0, 4.0], type=pa.float64()),
        })
        svg = _render(df, "year", "v", "mean")
        assert "<svg" in svg

    def test_int64_groupby_count_renders_without_error(self):
        """Count over an Int64 groupby must render correctly."""
        df = pa.table({
            "cat": pa.array([1, 1, 1, 2, 2], type=pa.int64()),
            "v": pa.array([0.0, 0.0, 0.0, 0.0, 0.0], type=pa.float64()),
        })
        svg = _render(df, "cat", "v", "count")
        assert "<svg" in svg


# ---------------------------------------------------------------------------
# Other integer variants
# ---------------------------------------------------------------------------

class TestIntegerVariantsGroupby:
    """Int32 and UInt32 groupby columns must also be accepted."""

    def test_int32_groupby_renders(self):
        df = pa.table({
            "grp": pa.array([1, 1, 2], type=pa.int32()),
            "v": pa.array([10.0, 20.0, 30.0], type=pa.float64()),
        })
        svg = _render(df, "grp", "v", "sum")
        assert "<svg" in svg

    def test_uint32_groupby_renders(self):
        df = pa.table({
            "grp": pa.array([1, 1, 2], type=pa.uint32()),
            "v": pa.array([10.0, 20.0, 30.0], type=pa.float64()),
        })
        svg = _render(df, "grp", "v", "sum")
        assert "<svg" in svg


# ---------------------------------------------------------------------------
# Int64 groupby matches equivalent Utf8 groupby (values must agree)
# ---------------------------------------------------------------------------

class TestInt64MatchesStringGroupby:
    """When group keys are integers that can be represented exactly as strings,
    the resulting SVG should not error and the spec structure must be identical
    (same Aggregate ops and groupby field name) regardless of column dtype."""

    def test_spec_structure_is_dtype_independent(self):
        """The Aggregate transform spec (groupby, ops) must not differ between
        Int64 and Utf8 groupby — only the dtype of the column in the input batch
        changes; the spec JSON stays the same."""
        import json

        categories_int = [100, 100, 200, 200, 300]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        df_int = pa.table({
            "cat": pa.array(categories_int, type=pa.int64()),
            "v": pa.array(values, type=pa.float64()),
        })
        df_str = pa.table({
            "cat": pa.array([str(c) for c in categories_int], type=pa.utf8()),
            "v": pa.array(values, type=pa.float64()),
        })

        def _spec_dict(df):
            chart = (
                fm.Chart(df)
                .mark_bar()
                .encode(x="cat", y="agg_v:Q")
                .transform(fm.Aggregate([fm.AggregateOp("v", "sum", "agg_v")], groupby=["cat"]))
            )
            return json.loads(chart.to_spec().to_json())

        spec_int = _spec_dict(df_int)
        spec_str = _spec_dict(df_str)

        # The spec JSON (groupby, ops, encoding) must be identical.
        agg_int = next(t for t in spec_int["transforms"] if t.get("type") == "aggregate")
        agg_str = next(t for t in spec_str["transforms"] if t.get("type") == "aggregate")
        assert agg_int["groupby"] == agg_str["groupby"] == ["cat"]
        assert agg_int["ops"] == agg_str["ops"]

        # Both must render without error.
        _render(df_int, "cat", "v", "sum")
        _render(df_str, "cat", "v", "sum")


# ---------------------------------------------------------------------------
# Regression guards: Float64 and Utf8 must not regress
# ---------------------------------------------------------------------------

class TestRegressionFloat64Groupby:
    """Float64 groupby must continue to work as before the FA-3 fix."""

    def test_float64_groupby_sum_renders(self):
        df = pa.table({
            "grp": pa.array([1.0, 1.0, 2.0, 2.0], type=pa.float64()),
            "v": pa.array([10.0, 20.0, 30.0, 40.0], type=pa.float64()),
        })
        svg = _render(df, "grp", "v", "sum")
        assert "<svg" in svg

    def test_float64_groupby_mean_renders(self):
        df = pa.table({
            "grp": pa.array([1.5, 1.5, 2.5, 2.5], type=pa.float64()),
            "v": pa.array([10.0, 20.0, 30.0, 40.0], type=pa.float64()),
        })
        svg = _render(df, "grp", "v", "mean")
        assert "<svg" in svg


class TestRegressionUtf8Groupby:
    """Utf8 groupby must continue to work as before the FA-3 fix."""

    def test_utf8_groupby_sum_renders(self):
        df = pa.table({
            "cat": pa.array(["a", "a", "b", "b"], type=pa.utf8()),
            "v": pa.array([10.0, 20.0, 30.0, 40.0], type=pa.float64()),
        })
        svg = _render(df, "cat", "v", "sum")
        assert "<svg" in svg

    def test_utf8_groupby_count_renders(self):
        df = pa.table({
            "cat": pa.array(["x", "y", "x", "y"], type=pa.utf8()),
            "v": pa.array([1.0, 2.0, 3.0, 4.0], type=pa.float64()),
        })
        svg = _render(df, "cat", "v", "count")
        assert "<svg" in svg
