"""Phase 3 — ChartSpec Python integration tests.

These tests exercise the Python boundary: the #[pyclass] constructors,
the string-shorthand sugar for x/y, the round-trip through JSON, and
the canonical wire format. The Rust side already verifies serde
mechanics; these tests verify Python semantics.
"""

import pytest

from ferrum._core import ChartSpec, EncodingSpec


# -- Construction ---------------------------------------------------------

def test_construct_minimal():
    spec = ChartSpec(mark="point", x="price", y="weight")
    assert spec.mark == "point"
    assert spec.x is not None and spec.x.field == "price"
    assert spec.y is not None and spec.y.field == "weight"
    assert spec.data == "default"


def test_x_y_string_shorthand():
    spec = ChartSpec(mark="point", x="price", y="weight")
    assert isinstance(spec.x, EncodingSpec)
    assert spec.x.field == "price"
    assert spec.x.type_ is None


def test_x_y_encoding_spec_explicit():
    e = EncodingSpec(field="price", type_="Q")
    spec = ChartSpec(mark="point", x=e, y="weight")
    assert spec.x is not None
    assert spec.x.field == "price"
    assert spec.x.type_ == "quantitative"


def test_data_default_when_omitted():
    spec = ChartSpec(mark="point", x="a", y="b")
    assert spec.data == "default"


def test_data_named():
    spec = ChartSpec(mark="point", x="a", y="b", data="my_table")
    assert spec.data == "my_table"


def test_data_type_short_and_long_forms_equivalent():
    s_short = ChartSpec(mark="point", x=EncodingSpec(field="p", type_="Q"), y="w")
    s_long = ChartSpec(
        mark="point",
        x=EncodingSpec(field="p", type_="quantitative"),
        y="w",
    )
    assert s_short == s_long
    assert s_short.to_json() == s_long.to_json()


# -- Errors ---------------------------------------------------------------

def test_unknown_mark_raises():
    with pytest.raises(ValueError) as exc_info:
        ChartSpec(mark="spaghetti", x="a", y="b")
    msg = str(exc_info.value)
    assert "spaghetti" in msg
    assert "point" in msg  # variant list present


def test_unknown_data_type_raises():
    with pytest.raises(ValueError) as exc_info:
        EncodingSpec(field="x", type_="Z")
    msg = str(exc_info.value)
    assert "'Z'" in msg
    assert "quantitative" in msg


# -- Round-trip -----------------------------------------------------------

def test_python_to_json_round_trip():
    spec = ChartSpec(mark="point", x="price", y=EncodingSpec(field="weight", type_="Q"))
    json = spec.to_json()
    spec2 = ChartSpec.from_json(json)
    assert spec == spec2


def test_python_to_json_idempotent():
    spec = ChartSpec(mark="point", x="price", y=EncodingSpec(field="weight", type_="Q"))
    json1 = spec.to_json()
    spec2 = ChartSpec.from_json(json1)
    json2 = spec2.to_json()
    assert json1 == json2


def test_canonical_json_shape():
    spec = ChartSpec(
        mark="point",
        x="price",
        y=EncodingSpec(field="weight", type_="Q"),
    )
    expected = (
        '{"data":{"kind":"named","name":"default"},'
        '"mark":"point",'
        '"encoding":{"x":{"field":"price"},'
        '"y":{"field":"weight","type":"quantitative"}}}'
    )
    assert spec.to_json() == expected


# -- Repr -----------------------------------------------------------------

def test_repr_contains_fields():
    spec = ChartSpec(mark="point", x="price", y="weight")
    r = repr(spec)
    assert "mark='point'" in r
    assert "price" in r
    assert "weight" in r


def test_repr_preserves_encoding_type():
    spec = ChartSpec(
        mark="point",
        x=EncodingSpec(field="price", type_="Q"),
        y="weight",
    )
    r = repr(spec)
    assert "type_='quantitative'" in r, f"type_ missing from repr: {r}"
