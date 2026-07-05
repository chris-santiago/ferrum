"""Tests for ``ferrum._render_prepare``'s per-channel field introspection
helpers (``_column_minmax``, ``_classify_field``) -- shared by
``Chart._resolve_chart_config``/``_render_inputs`` and by
``ferrum.composition``'s cross-chart union-domain computation
(``compute_union_domain``/``inject_scale``, the raw-column seam that
``LayerChart._build_merged`` uses for its one-panel interactive render).
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum
from ferrum._render_prepare import _classify_field, _column_minmax


class TestColumnMinmaxTemporal:
    """``_column_minmax`` must handle ``Date``/``Datetime``/``Time`` columns
    without raising. Previously it called Python's ``float()`` directly on
    the ``datetime``/``date``/``time`` scalar ``col.min()``/``col.max()``
    return for a temporal column, which raises ``TypeError`` -- reachable
    via a ``LayerChart`` interactive render sharing a temporal x/y (see
    ``TestLayerChartInteractiveSharedTemporalDomain`` below).
    """

    def test_datetime_column_returns_epoch_ms_floats(self):
        df = pl.DataFrame({"t": ["2020-01-01", "2020-06-01", "2021-01-01"]}).with_columns(
            pl.col("t").str.to_datetime()
        )
        assert _classify_field(df, "t") == "time"
        lo, hi = _column_minmax(df, "t")
        assert isinstance(lo, float)
        assert isinstance(hi, float)
        # Matches polars' own epoch-ms cast -- the same normalization
        # ferrum._coerce.to_arrow_table applies before the data reaches
        # the Rust renderer (Date and non-ms Datetime both cast to
        # Datetime("ms")).
        expected_lo = float(df["t"].cast(pl.Datetime("ms")).cast(pl.Int64).min())
        expected_hi = float(df["t"].cast(pl.Datetime("ms")).cast(pl.Int64).max())
        assert lo == pytest.approx(expected_lo)
        assert hi == pytest.approx(expected_hi)

    def test_datetime_us_and_ns_units_normalize_to_ms(self):
        """Datetime(us)/Datetime(ns) columns (not yet coerce-normalized --
        the union seam runs on raw chart._data, before ferrum._coerce's
        Datetime("ms") normalization happens at render time) must produce
        the SAME epoch-ms values as an already-ms column."""
        df_ms = pl.DataFrame({"t": ["2020-01-01T00:00:00.123"]}).with_columns(
            pl.col("t").str.to_datetime(time_unit="ms")
        )
        df_us = df_ms.with_columns(pl.col("t").cast(pl.Datetime("us")))
        df_ns = df_ms.with_columns(pl.col("t").cast(pl.Datetime("ns")))
        assert _column_minmax(df_ms, "t") == _column_minmax(df_us, "t")
        assert _column_minmax(df_ms, "t") == _column_minmax(df_ns, "t")

    def test_date_column_returns_epoch_ms_floats(self):
        df = pl.DataFrame({"d": ["2020-01-01", "2020-06-01"]}).with_columns(
            pl.col("d").str.to_date()
        )
        assert _classify_field(df, "d") == "time"
        lo, hi = _column_minmax(df, "d")
        assert isinstance(lo, float)
        assert isinstance(hi, float)
        assert lo < hi

    def test_time_column_returns_ms_of_day_floats(self):
        df = pl.DataFrame({"tod": ["01:00:00", "23:00:00"]}).with_columns(
            pl.col("tod").str.to_time()
        )
        assert _classify_field(df, "tod") == "time"
        lo, hi = _column_minmax(df, "tod")
        assert isinstance(lo, float)
        assert isinstance(hi, float)
        assert lo == pytest.approx(1 * 3_600_000.0)
        assert hi == pytest.approx(23 * 3_600_000.0)

    def test_linear_column_unaffected(self):
        """Non-temporal columns keep the original plain-float() path."""
        df = pl.DataFrame({"x": [1, 5, 3]})
        assert _column_minmax(df, "x") == (1.0, 5.0)


class TestLayerChartInteractiveSharedTemporalDomain:
    """Regression test for the reachable path: ``LayerChart``'s interactive
    ``_build_merged`` seam (``compute_union_domain``/``inject_scale``, the
    raw-column union used only by the one-panel interactive render -- see
    ``composition.py::LayerChart._build_merged``) sharing a temporal x/y.
    """

    def _temporal_charts(self):
        df1 = pl.DataFrame({"t": ["2020-01-01", "2020-06-01"], "v": [1.0, 2.0]}).with_columns(
            pl.col("t").str.to_datetime()
        )
        df2 = pl.DataFrame({"t": ["2019-01-01", "2021-01-01"], "v": [3.0, 4.0]}).with_columns(
            pl.col("t").str.to_datetime()
        )
        c1 = ferrum.Chart(df1).mark_line().encode(x="t", y="v")
        c2 = ferrum.Chart(df2).mark_point().encode(x="t", y="v")
        return df1, df2, c1, c2

    def test_render_interactive_succeeds_on_shared_temporal_x(self):
        """RED-proof: previously raised TypeError from float(datetime)
        inside compute_union_domain -> _column_minmax."""
        _df1, _df2, c1, c2 = self._temporal_charts()
        layer = ferrum.LayerChart(c1, c2, resolve={"x": "shared"})
        scene_json, packed = layer._render_interactive()
        assert scene_json
        assert isinstance(packed, (bytes, bytearray))

    def test_shared_x_domain_unions_both_layers_time_ranges(self):
        from ferrum.composition import compute_union_domain

        df1, df2, c1, c2 = self._temporal_charts()
        sd = compute_union_domain([c1, c2], "x")
        assert sd is not None
        assert sd["type"] == "time"

        combined = pl.concat([df1["t"], df2["t"]]).cast(pl.Datetime("ms")).cast(pl.Int64)
        expected_lo, expected_hi = float(combined.min()), float(combined.max())
        assert sd["domain"][0] == pytest.approx(expected_lo)
        assert sd["domain"][1] == pytest.approx(expected_hi)

        # df2's range (2019..2021) strictly contains df1's (2020 H1) --
        # the union must equal df2's own extent, discriminating against a
        # bug that only unions the FIRST binding rather than every layer.
        df2_lo = float(df2["t"].cast(pl.Datetime("ms")).cast(pl.Int64).min())
        df2_hi = float(df2["t"].cast(pl.Datetime("ms")).cast(pl.Int64).max())
        assert sd["domain"][0] == pytest.approx(df2_lo)
        assert sd["domain"][1] == pytest.approx(df2_hi)


def test_introspection_helpers_tolerate_missing_columns():
    """A genuinely missing column returns the documented None/[] fallbacks.

    Regression: polars raises ``ColumnNotFoundError`` (not a ``KeyError``
    subclass) on ``df["missing"]``, so the introspection helpers' documented
    missing-column fallback silently never applied — a missing field raised
    instead of returning ``None``/``[]`` (72h-review follow-on finding).
    """
    import polars as pl

    from ferrum._render_prepare import _classify_field, _column_minmax, _column_unique

    df = pl.DataFrame({"present": [1.0, 2.0]})
    assert _column_minmax(df, "absent") is None
    assert _column_unique(df, "absent") == []
    assert _classify_field(df, "absent") is None
