"""Regression tests for finding P3: LayerChart lacks ``|`` / ``&``.

``LayerChart`` predated the composition operators — ``__or__``/``__and__``
were only defined on ``_CompositeBase``, which ``LayerChart`` does not
extend (it extends ``_ChartLike`` directly). Every downstream path already
worked (``hconcat(L, c)``, ``vconcat(L, c)``, ``c | L``, composite lowering
of LayerChart children), so ``L | c`` / ``L & c`` raising ``TypeError`` was
a pure asymmetry, not a capability gap.

The fix moves ``__or__``/``__and__`` from ``_CompositeBase`` up to
``_ChartLike``, so every ``_ChartLike`` subclass — including ``LayerChart``
— inherits them automatically.
"""

import polars as pl
import pytest

import ferrum as fm
from ferrum.composition import HConcatChart, LayerChart, VConcatChart


@pytest.fixture
def df():
    return pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})


def _layer(df):
    c1 = fm.Chart(df).mark_point().encode(x="a", y="b")
    c2 = fm.Chart(df).mark_line().encode(x="a", y="b")
    return fm.layer(c1, c2)


def test_layer_or_chart_constructs_hconcat(df):
    layered = _layer(df)
    other = fm.Chart(df).mark_bar().encode(x="a", y="b")

    result = layered | other

    assert isinstance(result, HConcatChart)
    assert len(result.charts) == 2
    assert isinstance(result.charts[0], LayerChart)
    assert result.charts[1] is other


def test_layer_and_chart_constructs_vconcat(df):
    layered = _layer(df)
    other = fm.Chart(df).mark_bar().encode(x="a", y="b")

    result = layered & other

    assert isinstance(result, VConcatChart)
    assert len(result.charts) == 2
    assert isinstance(result.charts[0], LayerChart)
    assert result.charts[1] is other


def test_layer_or_chart_byte_equal_to_hconcat(df):
    layered = _layer(df)
    other = fm.Chart(df).mark_bar().encode(x="a", y="b")

    via_operator = (layered | other).to_svg()
    via_function = fm.hconcat(layered, other).to_svg()

    assert via_operator == via_function


def test_layer_and_chart_byte_equal_to_vconcat(df):
    layered = _layer(df)
    other = fm.Chart(df).mark_bar().encode(x="a", y="b")

    via_operator = (layered & other).to_svg()
    via_function = fm.vconcat(layered, other).to_svg()

    assert via_operator == via_function


def test_chart_or_layer_unchanged(df):
    """The already-working reflected direction (``c | L``) is unaffected."""
    layered = _layer(df)
    other = fm.Chart(df).mark_bar().encode(x="a", y="b")

    result = other | layered

    assert isinstance(result, HConcatChart)
    assert len(result.charts) == 2
    assert result.charts[0] is other
    assert isinstance(result.charts[1], LayerChart)
    # Renders without error.
    assert "<svg" in result.to_svg()
