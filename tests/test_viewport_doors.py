"""Every Python door that sets a chart's viewport refuses a malformed width/height at the call.

A viewport ``width``/``height`` is ``None`` (unset), ``"container"``, or a
finite, positive real number (``bool`` refused), stored as a ``float``.
``Chart(...)`` and ``Chart.properties`` refuse in their own voice, a
composite's ``properties`` names its own class, a figure function names
itself, and ``Chart.override(width=/height=)`` refuses as
``FerrumOverrideError`` in the ``Chart.override`` voice.
"""

from __future__ import annotations

import math
import re

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from ferrum import ClusterMapChart, JointChart, LayerChart, RepeatChart
from ferrum._entry import voiced_as
from ferrum._override_apply import build_payload
from ferrum._validate import coerce_pixel_value, coerce_positive_pixel_value
from ferrum.composition import ConcatChart, HConcatChart, VConcatChart
from ferrum.exceptions import FerrumOverrideError

_DF = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})

# (value, the refusal text after the "<door>(width=...) " prefix)
_MALFORMED = [
    pytest.param(math.nan, "must be a finite numeric pixel value; received nan", id="nan"),
    pytest.param(math.inf, "must be a finite numeric pixel value; received inf", id="inf"),
    pytest.param(-5, "must be positive; received -5", id="negative"),
    pytest.param(0, "must be positive; received 0", id="zero"),
    pytest.param(
        "abc",
        "must be a numeric pixel value (int or float), not str; received 'abc'",
        id="str",
    ),
    pytest.param(
        True,
        "must be a numeric pixel value (int or float), not bool; received True",
        id="bool",
    ),
]

_ACCEPTED = [
    pytest.param(300, 300.0, id="int"),
    pytest.param(312.5, 312.5, id="float"),
    pytest.param(np.int64(250), 250.0, id="numpy-int"),
    pytest.param(np.float32(200.0), 200.0, id="numpy-float"),
]


def _chart() -> fm.Chart:
    return fm.Chart(_DF).mark_point().encode(x="x", y="y")


class TestChartDoors:
    @pytest.mark.parametrize("param", ["width", "height"])
    @pytest.mark.parametrize(("value", "reason"), _MALFORMED)
    def test_properties_refuses_at_the_call(self, param, value, reason):
        with pytest.raises(ValueError) as excinfo:
            _chart().properties(**{param: value})
        assert str(excinfo.value) == f"Chart.properties({param}=...) {reason}"

    @pytest.mark.parametrize("param", ["width", "height"])
    @pytest.mark.parametrize(("value", "reason"), _MALFORMED)
    def test_constructor_refuses_at_the_call(self, param, value, reason):
        with pytest.raises(ValueError) as excinfo:
            fm.Chart(_DF, **{param: value})
        assert str(excinfo.value) == f"Chart({param}=...) {reason}"

    @pytest.mark.parametrize(("value", "stored"), _ACCEPTED)
    def test_a_number_is_stored_as_a_float(self, value, stored):
        chart = _chart().properties(width=value, height=value)
        assert (chart._width, chart._height) == (stored, stored)
        assert type(chart._width) is float
        constructed = fm.Chart(_DF, width=value, height=value)
        assert (constructed._width, constructed._height) == (stored, stored)
        assert type(constructed._height) is float

    def test_a_stored_float_renders_the_same_bytes_as_the_int(self):
        """The door's ``float`` changes nothing on the wire."""
        assert (
            _chart().properties(width=300, height=200).to_svg()
            == _chart().properties(width=300.0, height=200.0).to_svg()
        )

    @pytest.mark.parametrize("door", ["properties", "constructor"])
    def test_none_and_container_pass_unchanged(self, door):
        if door == "properties":
            chart = _chart().properties(width="container", height=None)
        else:
            chart = fm.Chart(_DF, width="container", height=None)
        assert chart._width == "container"
        assert chart._height is None


class TestOverrideDoor:
    @pytest.mark.parametrize("param", ["width", "height"])
    @pytest.mark.parametrize(
        ("value", "reason"),
        [_MALFORMED[0], _MALFORMED[3], _MALFORMED[4]],
    )
    def test_override_refuses_as_ferrum_override_error(self, param, value, reason):
        with pytest.raises(FerrumOverrideError) as excinfo:
            _chart().override(**{param: value}).to_svg()
        assert str(excinfo.value) == f"Chart.override({param}=...) {reason}"

    def test_the_payload_stores_a_float_and_passes_container(self):
        payload = build_payload({"width": np.int64(320), "height": "container"})
        assert payload.properties == {"width": 320.0, "height": "container"}
        assert type(payload.properties["width"]) is float

    @pytest.mark.parametrize("param", ["width", "height"])
    def test_a_none_override_is_no_override(self, param):
        chart = _chart().properties(width=310, height=210)
        with pytest.warns(DeprecationWarning):
            overridden = chart.override(**{param: None}).to_svg()
        assert overridden == chart.to_svg()
        assert build_payload({param: None}).properties == {}

    def test_a_numpy_override_renders_like_its_float(self):
        with pytest.warns(DeprecationWarning):
            numpy_svg = _chart().override(width=np.int64(320)).to_svg()
        with pytest.warns(DeprecationWarning):
            float_svg = _chart().override(width=320.0).to_svg()
        assert numpy_svg == float_svg

    def test_a_figure_re_voices_the_override_refusal(self):
        chart = _chart().override(width=math.nan)
        with pytest.raises(FerrumOverrideError) as excinfo:
            with voiced_as("roc_chart"):
                chart.to_spec()
        assert str(excinfo.value) == (
            "roc_chart(width=...) must be a finite numeric pixel value; received nan"
        )


def _hconcat():
    return fm.hconcat(_chart(), _chart())


def _vconcat():
    return fm.vconcat(_chart(), _chart())


def _concat():
    return fm.concat(_chart(), _chart(), columns=2)


def _joint():
    return JointChart(_chart(), top=_chart())


def _repeat():
    template = fm.Chart(_DF).encode(x=fm.Repeat.column, y="y").mark_point()
    return RepeatChart(template, column=["x", "y"])


def _cluster_map():
    return ClusterMapChart(_chart(), dendrogram_ratio=0.3)


def _layer():
    return fm.layer(_chart(), _chart())


_COMPOSITES = [
    pytest.param(_hconcat, HConcatChart, id="hconcat"),
    pytest.param(_vconcat, VConcatChart, id="vconcat"),
    pytest.param(_concat, ConcatChart, id="concat"),
    pytest.param(_joint, JointChart, id="joint"),
    pytest.param(_repeat, RepeatChart, id="repeat"),
    pytest.param(_cluster_map, ClusterMapChart, id="cluster-map"),
    pytest.param(_layer, LayerChart, id="layer"),
]


class TestCompositeDoors:
    @pytest.mark.parametrize(("build", "cls"), _COMPOSITES)
    def test_properties_names_the_composite(self, build, cls):
        composite = build()
        assert type(composite) is cls
        with pytest.raises(ValueError) as excinfo:
            composite.properties(width=math.nan)
        assert str(excinfo.value) == (
            f"{cls.__name__}.properties(width=...) must be a finite numeric pixel value; "
            "received nan"
        )

    def test_a_nested_composite_names_the_outermost_call(self):
        with pytest.raises(ValueError) as excinfo:
            fm.hconcat(_vconcat(), _chart()).properties(height=0)
        assert (
            str(excinfo.value) == "HConcatChart.properties(height=...) must be positive; received 0"
        )

    def test_a_valid_width_still_reaches_every_child_as_a_float(self):
        composite = _hconcat().properties(width=np.int64(240))
        assert [child._width for child in composite.charts] == [240.0, 240.0]
        assert all(type(child._width) is float for child in composite.charts)


_FRAME = pl.DataFrame(
    {"row": ["r0", "r1", "r2", "r3"], "A": [0.1, 0.4, 0.2, 0.9], "B": [1.0, 0.3, 0.6, 0.2]}
)
_CATEGORICAL = pl.DataFrame(
    {"c": ["a", "b", "a", "b"], "v": [1.0, 2.0, 3.0, 4.0], "g": ["p", "p", "q", "q"]}
)
_EMPTY = pl.DataFrame(
    {
        "c": pl.Series([], dtype=pl.Utf8),
        "v": pl.Series([], dtype=pl.Float64),
        "g": pl.Series([], dtype=pl.Utf8),
    }
)

# Each figure that takes height=/aspect=, called with a well-formed chart.
_HEIGHT_ASPECT_FIGURES = [
    pytest.param("displot", lambda **kw: fm.displot(_CATEGORICAL, x="v", **kw), id="displot"),
    pytest.param(
        "catplot", lambda **kw: fm.catplot(_CATEGORICAL, x="c", y="v", **kw), id="catplot"
    ),
    pytest.param(
        "relplot", lambda **kw: fm.relplot(_CATEGORICAL, x="v", y="v", **kw), id="relplot"
    ),
    pytest.param("pairplot", lambda **kw: fm.pairplot(_FRAME, **kw), id="pairplot"),
]


class TestFigureDoors:
    """A figure refuses the size argument the caller passed, by its own name."""

    def test_jointplot_names_its_height(self):
        with pytest.raises(ValueError) as excinfo:
            fm.jointplot(_DF, x="x", y="y", height=math.nan)
        assert str(excinfo.value) == (
            "jointplot(height=...) must be a finite numeric pixel value; received nan"
        )

    @pytest.mark.parametrize(("name", "build"), _HEIGHT_ASPECT_FIGURES)
    @pytest.mark.parametrize(("value", "reason"), _MALFORMED)
    def test_height_is_refused_by_its_own_name(self, name, build, value, reason):
        with pytest.raises(ValueError) as excinfo:
            build(height=value)
        assert str(excinfo.value) == f"{name}(height=...) {reason}"

    @pytest.mark.parametrize(("name", "build"), _HEIGHT_ASPECT_FIGURES)
    @pytest.mark.parametrize(
        ("value", "reason"),
        [
            pytest.param(0, "must be positive; received 0", id="zero"),
            pytest.param(-2, "must be positive; received -2", id="negative"),
            pytest.param(
                math.nan, "must be a finite number a float can hold; received nan", id="nan"
            ),
            pytest.param(
                True,
                "must be a real number (int, float, numpy scalar, or Decimal), "
                "not bool; received True",
                id="bool",
            ),
        ],
    )
    def test_aspect_is_refused_by_its_own_name(self, name, build, value, reason):
        with pytest.raises(ValueError) as excinfo:
            build(aspect=value)
        assert str(excinfo.value) == f"{name}(aspect=...) {reason}"

    @pytest.mark.parametrize(("name", "build"), _HEIGHT_ASPECT_FIGURES)
    def test_a_well_formed_height_and_aspect_still_size_the_chart(self, name, build):
        assert build(height=200, aspect=1.5).to_svg().startswith("<svg")

    @pytest.mark.parametrize(
        ("figsize", "reason"),
        [
            pytest.param(
                (math.nan, 300), "must be a finite numeric pixel value; received nan", id="nan"
            ),
            pytest.param((300, 0), "must be positive; received 0", id="zero"),
            pytest.param(
                (300, None),
                "must be a numeric pixel value (int or float), not NoneType; received None",
                id="none-side",
            ),
            pytest.param(
                (300, 200, 100),
                "must be a (width, height) pair; received (300, 200, 100)",
                id="triple",
            ),
            pytest.param(300, "must be a (width, height) pair; received 300", id="scalar"),
        ],
    )
    def test_clustermap_names_its_figsize(self, figsize, reason):
        with pytest.raises(ValueError) as excinfo:
            fm.clustermap(_FRAME, figsize=figsize)
        assert str(excinfo.value) == f"clustermap(figsize=...) {reason}"

    def test_clustermap_internal_sizing_still_renders(self):
        """`ClusterMapChart` sizes its dendrograms through `.properties` at
        render; every computed size is a positive float, so it passes."""
        svg = fm.clustermap(_FRAME, figsize=(320, 240)).to_svg()
        assert svg.startswith("<svg")


class TestEmptyDataSizing:
    """A size the figure derives from empty data is left unset, so the chart
    renders as the empty dataset it is instead of refusing a size the caller
    never passed."""

    def test_displot_row_facet_on_empty_data_renders_empty(self):
        assert fm.displot(_EMPTY, x="v", row="g").to_svg() == (
            '<svg xmlns="http://www.w3.org/2000/svg" width="640.0" height="480.0">'
            "<!-- empty dataset --></svg>"
        )

    def test_catplot_row_facet_on_empty_data_renders_empty(self):
        assert fm.catplot(_EMPTY, x="c", y="v", row="g").to_svg() == (
            '<svg xmlns="http://www.w3.org/2000/svg" width="640.0" height="480.0">'
            "<!-- empty dataset --></svg>"
        )

    def test_catplot_row_facet_on_empty_data_keeps_the_aspect_width(self):
        """The width comes from the caller's aspect, not the empty data, so it stays."""
        assert fm.catplot(_EMPTY, x="c", y="v", row="g", aspect=2).to_svg() == (
            '<svg xmlns="http://www.w3.org/2000/svg" width="300.0" height="480.0">'
            "<!-- empty dataset --></svg>"
        )

    def test_heatmap_square_on_an_empty_table_renders_empty(self):
        """One value column keeps the square side at 30px even with no rows."""
        svg = fm.heatmap(pl.DataFrame({"a": pl.Series([], dtype=pl.Float64)}), square=True).to_svg()
        assert "<!-- empty dataset -->" in svg
        assert re.search(r'width="30(\.0)?" height="30(\.0)?"', svg)


class TestPixelValidatorFamily:
    def test_coerce_pixel_value_keeps_its_contract(self):
        """The non-negative sibling still accepts 0 and refuses a negative as non-negative."""
        assert coerce_pixel_value("padding.top", 0) == 0.0
        assert coerce_pixel_value("padding.top", None) is None
        with pytest.raises(ValueError) as excinfo:
            coerce_pixel_value("padding.top", -1)
        assert str(excinfo.value) == "padding.top must be non-negative; received -1"
        with pytest.raises(ValueError) as excinfo:
            coerce_pixel_value("padding.top", math.nan)
        assert (
            str(excinfo.value) == "padding.top must be a finite numeric pixel value; received nan"
        )
        with pytest.raises(ValueError) as excinfo:
            coerce_pixel_value("padding.top", "x")
        assert str(excinfo.value) == (
            "padding.top must be a numeric pixel value (int or float), not str; received 'x'"
        )

    def test_coerce_positive_pixel_value_refuses_zero(self):
        assert coerce_positive_pixel_value("w", 0.5) == 0.5
        assert coerce_positive_pixel_value("w", None) is None
        with pytest.raises(ValueError, match=r"^w must be positive; received 0$"):
            coerce_positive_pixel_value("w", 0)
        with pytest.raises(ValueError, match=r"not NoneType"):
            coerce_positive_pixel_value("w", None, required=True)
