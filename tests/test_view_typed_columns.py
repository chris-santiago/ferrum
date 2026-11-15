"""A view-typed Arrow column is rewritten to its plain twin at ingest.

``pl.from_arrow`` segfaults on a ``string_view`` or ``binary_view`` array made by a
cast, so every crash case runs in its own subprocess: a regression fails the test,
not the pytest worker.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
import warnings

import pyarrow as pa
import pytest

from ferrum import _coerce as coerce
from ferrum._coerce import to_arrow_table

# Each builder yields ``(view_table, plain_table)``: the same data once with the
# view type and once with its plain twin. Source text so the subprocess can run it.
_BUILDERS = {
    "short_cast": """
        c = pa.array(["x", "y", "z"])
        view = pa.table({"a": [1, 2, 3], "c": c.cast(pa.string_view())})
        plain = pa.table({"a": [1, 2, 3], "c": c.cast(pa.large_string())})
    """,
    "long_cast": """
        c = pa.array(["a-value-longer-than-twelve-bytes-%d" % i for i in range(3)])
        view = pa.table({"a": [1, 2, 3], "c": c.cast(pa.string_view())})
        plain = pa.table({"a": [1, 2, 3], "c": c.cast(pa.large_string())})
    """,
    "native": """
        c = pa.array(["x", "y", "z"], type=pa.string_view())
        view = pa.table({"a": [1, 2, 3], "c": c})
        plain = pa.table({"a": [1, 2, 3], "c": pa.array(["x", "y", "z"], pa.large_string())})
    """,
    "chunked": """
        chunks = [
            pa.array(["x", "y"]).cast(pa.string_view()),
            pa.array(["z"]).cast(pa.string_view()),
        ]
        view = pa.table({"a": [1, 2, 3], "c": pa.chunked_array(chunks)})
        plain = pa.table({"a": [1, 2, 3], "c": pa.array(["x", "y", "z"], pa.large_string())})
    """,
    "dictionary": """
        d = pa.array(["x", "y", "x"]).dictionary_encode()
        dv = pa.DictionaryArray.from_arrays(
            d.indices, d.dictionary.cast(pa.string_view())
        )
        view = pa.table({"a": [1, 2, 3], "c": dv})
        plain = pa.table({"a": [1, 2, 3], "c": pa.array(["x", "y", "x"], pa.large_string())})
    """,
    "list_of_view": """
        c = pa.array([["x"], ["y"], ["z"]]).cast(pa.list_(pa.string_view()))
        view = pa.table({"a": [1, 2, 3], "c": c})
        plain = pa.table({"a": [1, 2, 3], "c": pa.array([["x"], ["y"], ["z"]], pa.list_(pa.large_string()))})
    """,
    "struct_of_view": """
        s = pa.array([{"k": "x"}, {"k": "y"}, {"k": "z"}])
        c = s.cast(pa.struct([pa.field("k", pa.string_view())]))
        view = pa.table({"a": [1, 2, 3], "c": c})
        plain = pa.table({"a": [1, 2, 3], "c": s.cast(pa.struct([pa.field("k", pa.large_string())]))})
    """,
    "map_of_view": """
        m = pa.array([[("x", 1)], [("y", 2)], [("z", 3)]], pa.map_(pa.string(), pa.int64()))
        c = m.cast(pa.map_(pa.string_view(), pa.int64()))
        view = pa.table({"a": [1, 2, 3], "c": c})
        plain = pa.table({"a": [1, 2, 3], "c": m.cast(pa.map_(pa.large_string(), pa.int64()))})
    """,
    "list_view": """
        c = pa.array([[1], [2], [3]], pa.list_view(pa.int64()))
        view = pa.table({"a": [1, 2, 3], "c": c})
        plain = pa.table({"a": [1, 2, 3], "c": pa.array([[1], [2], [3]], pa.large_list(pa.int64()))})
    """,
    "sparse_union": """
        types = pa.array([0, 1, 0], pa.int8())
        sv = pa.array(["x", "y", "z"]).cast(pa.string_view())
        lg = pa.array(["x", "y", "z"]).cast(pa.large_string())
        ints = pa.array([1, 2, 3])
        view = pa.table({"a": [1, 2, 3], "c": pa.UnionArray.from_sparse(types, [sv, ints])})
        plain = pa.table({"a": [1, 2, 3], "c": pa.UnionArray.from_sparse(types, [lg, ints])})
    """,
    "dense_union": """
        types = pa.array([0, 1, 0], pa.int8())
        offsets = pa.array([0, 0, 1], pa.int32())
        sv = pa.array(["x", "z"]).cast(pa.string_view())
        lg = pa.array(["x", "z"]).cast(pa.large_string())
        ints = pa.array([2])
        view = pa.table({"a": [1, 2, 3], "c": pa.UnionArray.from_dense(types, offsets, [sv, ints])})
        plain = pa.table({"a": [1, 2, 3], "c": pa.UnionArray.from_dense(types, offsets, [lg, ints])})
    """,
    "run_end_encoded": """
        ends = pa.array([2, 3], pa.int32())
        sv = pa.array(["x", "z"]).cast(pa.string_view())
        lg = pa.array(["x", "z"]).cast(pa.large_string())
        view = pa.table({"a": [1, 2, 3], "c": pa.RunEndEncodedArray.from_arrays(ends, sv)})
        plain = pa.table({"a": [1, 2, 3], "c": pa.RunEndEncodedArray.from_arrays(ends, lg)})
    """,
    "extension": """
        class SvType(pa.ExtensionType):
            def __init__(self):
                super().__init__(pa.string_view(), "ferrum.test.sv")
            def __arrow_ext_serialize__(self):
                return b""
            @classmethod
            def __arrow_ext_deserialize__(cls, storage_type, serialized):
                return cls()
        sv = pa.array(["x", "y", "z"]).cast(pa.string_view())
        ext = pa.ExtensionArray.from_storage(SvType(), sv)
        view = pa.table({"a": [1, 2, 3], "c": ext})
        plain = pa.table({"a": [1, 2, 3], "c": pa.array(["x", "y", "z"], pa.large_string())})
    """,
    "binary_view": """
        c = pa.array([b"x", b"y", b"z"])
        view = pa.table({"a": [1, 2, 3], "c": c.cast(pa.binary_view())})
        plain = pa.table({"a": [1, 2, 3], "c": c.cast(pa.large_binary())})
    """,
}

# How a case is turned into the object handed to Chart.
_WRAPPERS = {
    "table": "data = view",
    "record_batch": "data = view.combine_chunks().to_batches()[0]",
    "dict_of_arrays": "data = {n: view.column(n).combine_chunks() for n in view.column_names}",
}

_RUNNER = """
import hashlib, sys
import pyarrow as pa
import ferrum as fm

{builder}
{wrapper}
twin = plain
encode, route = sys.argv[1], sys.argv[2]

def svg(source):
    chart = fm.Chart(source).mark_point()
    if encode == "color":
        chart = chart.encode(x="a", y="a", color="c")
    else:
        chart = chart.encode(x="a", y="a")
    if route == "interactive":
        return chart.interactive().to_html()
    return chart.to_svg()

def outcome(fn):
    try:
        return "svg " + hashlib.sha256(fn().encode()).hexdigest()
    except Exception as e:
        return type(e).__name__ + ": " + str(e)

print(outcome(lambda: svg(data)))
print(outcome(lambda: svg(twin)))
"""


def _run(
    builder: str, *, wrapper: str = "table", encode: str = "color", route: str = "static"
) -> tuple[str, str]:
    script = _RUNNER.format(
        builder=textwrap.dedent(_BUILDERS[builder]).strip(),
        wrapper=_WRAPPERS[wrapper],
    )
    proc = subprocess.run(
        [sys.executable, "-c", script, encode, route],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"subprocess died rc={proc.returncode}\n{proc.stderr[-800:]}"
    lines = proc.stdout.strip().splitlines()
    assert len(lines) == 2, proc.stdout
    return lines[0], lines[1]


@pytest.mark.parametrize("route", ["static", "interactive"])
@pytest.mark.parametrize("encode", ["color", "unused"])
@pytest.mark.parametrize("builder", ["short_cast", "long_cast", "native", "chunked"])
def test_string_view_column_renders_as_its_large_string_twin(builder, encode, route):
    got, twin = _run(builder, encode=encode, route=route)
    assert got.startswith("svg "), got
    assert got == twin


@pytest.mark.parametrize("route", ["static", "interactive"])
@pytest.mark.parametrize("wrapper", ["record_batch", "dict_of_arrays"])
def test_other_arrow_inputs_with_view_column_render_as_twin(wrapper, route):
    got, twin = _run("short_cast", wrapper=wrapper, route=route)
    assert got.startswith("svg "), got
    assert got == twin


@pytest.mark.parametrize("route", ["static", "interactive"])
def test_binary_view_column_is_refused_like_a_native_binary_column(route):
    got, twin = _run("binary_view", encode="color", route=route)
    assert got == twin
    assert got.startswith("ValueError: ")
    assert "unsupported dtype: LargeBinary" in got


@pytest.mark.parametrize("route", ["static", "interactive"])
def test_binary_view_column_unused_by_any_encoding_renders(route):
    got, twin = _run("binary_view", encode="unused", route=route)
    assert got.startswith("svg "), got
    assert got == twin


@pytest.mark.parametrize("route", ["static", "interactive"])
@pytest.mark.parametrize(
    "builder",
    [
        "dictionary",
        "list_of_view",
        "struct_of_view",
        "map_of_view",
        "list_view",
        "sparse_union",
        "dense_union",
        "run_end_encoded",
        "extension",
    ],
)
def test_nested_view_column_does_not_crash_and_matches_its_twin(builder, route):
    got, twin = _run(builder, encode="unused", route=route)
    assert got == twin
    got, twin = _run(builder, encode="color", route=route)
    assert got == twin


def _ingest(builder: str):
    ns: dict = {"pa": pa}
    exec(textwrap.dedent(_BUILDERS[builder]), ns)  # noqa: S102 (test-local fixture source)
    return ns["view"], ns["plain"]


@pytest.mark.parametrize(
    ("builder", "expected"),
    [
        ("short_cast", pa.large_string()),
        ("native", pa.large_string()),
        ("chunked", pa.large_string()),
        ("binary_view", pa.large_binary()),
        ("list_of_view", pa.list_(pa.large_string())),
        ("list_view", pa.large_list(pa.int64())),
        ("struct_of_view", pa.struct([pa.field("k", pa.large_string())])),
        ("map_of_view", pa.map_(pa.large_string(), pa.int64())),
        ("dictionary", pa.dictionary(pa.int32(), pa.large_string())),
    ],
)
def test_ingest_rewrites_the_column_type_to_its_plain_twin(builder, expected):
    view, plain = _ingest(builder)
    out = to_arrow_table(view)
    assert out.schema.field("c").type == expected
    assert out.column("c").to_pylist() == plain.column("c").to_pylist()
    assert out.column("a").to_pylist() == [1, 2, 3]


def test_ingest_keeps_field_names_nullability_and_dictionary_order():
    arr = pa.array([["x"], None, ["z"]]).cast(pa.list_(pa.string_view()))
    schema = pa.schema(
        [
            pa.field("keep", pa.int64(), nullable=False),
            pa.field("c", arr.type, metadata={b"k": b"v"}),
        ]
    )
    tbl = pa.table([pa.array([1, 2, 3]), arr], schema=schema)
    out = to_arrow_table(tbl)
    assert out.column_names == ["keep", "c"]
    assert out.schema.field("keep").nullable is False
    assert out.schema.field("c").type == pa.list_(pa.large_string())
    assert out.column("c").to_pylist() == [["x"], None, ["z"]]

    d = pa.DictionaryArray.from_arrays(
        pa.array([0, 1], pa.int8()), pa.array(["p", "q"]).cast(pa.string_view()), ordered=True
    )
    out = to_arrow_table(pa.table({"c": d}))
    assert out.schema.field("c").type == pa.dictionary(pa.int8(), pa.large_string(), ordered=True)


def test_dict_input_with_view_arrays_passes_through_the_normalization():
    view = pa.array(["x", "y"]).cast(pa.string_view())
    out = to_arrow_table({"c": view})
    assert out.schema.field("c").type == pa.large_string()
    assert out.column("c").to_pylist() == ["x", "y"]


def test_list_of_dicts_holding_view_scalars_passes_through_the_normalization():
    rows = [{"c": pa.scalar(v, pa.string_view())} for v in ("x", "y")]
    out = to_arrow_table(rows)
    assert out.schema.field("c").type == pa.large_string()
    assert out.column("c").to_pylist() == ["x", "y"]
    assert to_arrow_table([{"c": "x"}]).schema.field("c").type == pa.string()


def test_uncastable_column_is_refused_with_spec_text(monkeypatch):
    view = pa.table({"a": [1, 2], "c": pa.array(["x", "y"], pa.string_view())})

    def refuse(arr, target):
        raise coerce._CannotCast("no cast")

    monkeypatch.setattr(coerce, "_cast_array", refuse)
    with pytest.raises(TypeError) as err:
        to_arrow_table(view)
    assert str(err.value) == (
        "column 'c' has Arrow type string_view, which ferrum cannot read; "
        "cast it to a plain type (e.g. pa.large_string()) before charting"
    )


def test_inputs_without_view_types_are_never_cast(monkeypatch):
    import numpy as np
    import pandas as pd
    import polars as pl

    def forbid(column, pa):
        raise AssertionError("a column without view types must not be cast")

    plain = pa.table({"a": [1, 2], "c": pa.array(["x", "y"], pa.large_string())})
    nested = pa.table({"c": pa.array([[1], [2]], pa.large_list(pa.int64()))})
    monkeypatch.setattr(coerce, "_convert_column", forbid)
    assert to_arrow_table(plain) is plain
    assert to_arrow_table(nested) is nested
    to_arrow_table(pl.DataFrame({"a": [1], "c": ["x"]}))
    to_arrow_table(pl.LazyFrame({"a": [1], "c": ["x"]}))
    to_arrow_table(pd.DataFrame({"a": [1], "c": ["x"]}))
    to_arrow_table(np.arange(6).reshape(3, 2))
    to_arrow_table({"a": [1, 2]})
    to_arrow_table([{"a": 1}])


_LIST_VIEW_ROWS = [[1, 2], None, [], [3], [4, 5, 6], [7]]


def _nested_view_arrays() -> dict[str, tuple[pa.Array, list]]:
    """Arrays holding a list view inside another container, with the rows they must read as."""
    lv = pa.array(_LIST_VIEW_ROWS, pa.list_view(pa.int64()))
    llv = pa.array(_LIST_VIEW_ROWS, pa.large_list_view(pa.int64()))
    return {
        "list_view": (lv, _LIST_VIEW_ROWS),
        "large_list_view": (llv, _LIST_VIEW_ROWS),
        "list_view_sliced": (lv.slice(1, 4), _LIST_VIEW_ROWS[1:5]),
        "struct_of_list_view": (
            pa.StructArray.from_arrays([lv], ["k"]),
            [{"k": r} for r in _LIST_VIEW_ROWS],
        ),
        "null_struct_of_view": (
            pa.StructArray.from_arrays(
                [pa.array(["x", "y", "z"], pa.string_view())],
                ["k"],
                mask=pa.array([False, True, False]),
            ),
            [{"k": "x"}, None, {"k": "z"}],
        ),
        "list_of_list_view": (
            pa.ListArray.from_arrays(pa.array([0, 2, 3], pa.int32()), lv.slice(0, 3)),
            [[[1, 2], None], [[]]],
        ),
        "fixed_size_list_of_view": (
            pa.FixedSizeListArray.from_arrays(pa.array(["x", "y", "z", "w"], pa.string_view()), 2),
            [["x", "y"], ["z", "w"]],
        ),
    }


@pytest.mark.parametrize("name", list(_nested_view_arrays()))
def test_ingest_preserves_every_value_and_null_of_nested_views(name):
    arr, rows = _nested_view_arrays()[name]
    out = to_arrow_table(pa.table({"c": arr}))
    assert out.column("c").to_pylist() == rows


def test_ingest_preserves_values_of_a_chunked_list_view_column():
    chunks = [
        pa.array([[1], [2, 3]], pa.list_view(pa.int64())),
        pa.array([], pa.list_view(pa.int64())),
        pa.array([None, [4]], pa.list_view(pa.int64())),
    ]
    out = to_arrow_table(pa.table({"c": pa.chunked_array(chunks)}))
    assert out.schema.field("c").type == pa.large_list(pa.int64())
    assert out.column("c").to_pylist() == [[1], [2, 3], None, [4]]


def test_ingest_preserves_a_zero_row_view_column():
    out = to_arrow_table(pa.table({"c": pa.array([], pa.string_view())}))
    assert out.schema.field("c").type == pa.large_string()
    assert out.num_rows == 0


def _sv(values):
    return pa.array(values, pa.string_view())


def _containers_with_nulls() -> dict[str, tuple[pa.Array, list, pa.DataType]]:
    """Four-row columns holding a view type and a null row, with their rows and plain type."""
    lists = pa.array([["a"], None, ["b", "c"], ["d"]]).cast(pa.list_(pa.string_view()))
    large = pa.array([["a"], None, ["b", "c"], ["d"]], pa.large_list(pa.string())).cast(
        pa.large_list(pa.string_view())
    )
    maps = pa.array(
        [[("a", 1)], None, [("b", 2), ("c", 3)], [("d", 4)]], pa.map_(pa.string(), pa.int64())
    ).cast(pa.map_(pa.string_view(), pa.int64()))
    fixed = pa.FixedSizeListArray.from_arrays(
        _sv(["a", "b", "c", "d", "e", "f", "g", "h"]),
        2,
        mask=pa.array([False, True, False, False]),
    )
    return {
        "list": (lists, [["a"], None, ["b", "c"], ["d"]], pa.list_(pa.large_string())),
        "large_list": (large, [["a"], None, ["b", "c"], ["d"]], pa.large_list(pa.large_string())),
        "map": (
            maps,
            [[("a", 1)], None, [("b", 2), ("c", 3)], [("d", 4)]],
            pa.map_(pa.large_string(), pa.int64()),
        ),
        "fixed_size_list": (
            fixed,
            [["a", "b"], None, ["e", "f"], ["g", "h"]],
            pa.list_(pa.large_string(), 2),
        ),
    }


@pytest.mark.parametrize("start", [0, 1, 2, 3])
@pytest.mark.parametrize("name", list(_containers_with_nulls()))
def test_a_container_of_views_with_null_rows_loads_at_every_slice_offset(name, start):
    arr, rows, plain = _containers_with_nulls()[name]
    tbl = pa.table({"c": arr}).slice(start)
    out = to_arrow_table(tbl)
    assert out.schema.field("c").type == plain
    assert out.column("c").to_pylist() == rows[start:]


@pytest.mark.parametrize("name", list(_containers_with_nulls()))
def test_a_container_of_views_with_null_rows_loads_in_every_chunk(name):
    arr, rows, plain = _containers_with_nulls()[name]
    chunked = pa.chunked_array([arr.slice(0, 1), arr.slice(1, 2), arr.slice(3)])
    out = to_arrow_table(pa.table({"c": chunked}))
    assert out.schema.field("c").type == plain
    assert out.column("c").to_pylist() == rows


def _list_view_containers_with_nulls() -> dict[str, tuple[pa.Array, list]]:
    lv = pa.array([[1], None, [2, 3], [4]], pa.list_view(pa.int64()))
    return {
        "list_of_list_view": (
            pa.ListArray.from_arrays(
                pa.array([0, 1, 1, 3, 4], pa.int32()),
                lv,
                mask=pa.array([False, True, False, False]),
            ),
            [[[1]], None, [None, [2, 3]], [[4]]],
        ),
        "map_of_list_view": (
            pa.MapArray.from_arrays(
                pa.array([0, 1, 1, 3, 4], pa.int32()),
                _sv(["a", "b", "c", "d"]),
                lv,
                mask=pa.array([False, True, False, False]),
            ),
            [[("a", [1])], None, [("b", None), ("c", [2, 3])], [("d", [4])]],
        ),
        "fixed_size_list_of_list_view": (
            pa.FixedSizeListArray.from_arrays(lv, 2, mask=pa.array([False, True])),
            [[[1], None], None],
        ),
        "struct_of_list_view": (
            pa.StructArray.from_arrays([lv], ["k"], mask=pa.array([False, True, False, False])),
            [{"k": [1]}, None, {"k": [2, 3]}, {"k": [4]}],
        ),
    }


@pytest.mark.parametrize("start", [0, 1, 2])
@pytest.mark.parametrize("name", list(_list_view_containers_with_nulls()))
def test_a_container_of_list_views_with_null_rows_loads_at_every_slice_offset(name, start):
    arr, rows = _list_view_containers_with_nulls()[name]
    out = to_arrow_table(pa.table({"c": arr}).slice(start))
    assert out.column("c").to_pylist() == rows[start:]
    assert "view" not in str(out.schema.field("c").type)


@pytest.mark.parametrize("name", ["sparse_union", "dense_union", "run_end_encoded", "extension"])
def test_unions_run_end_encoded_and_extension_columns_load_as_plain_twins(name):
    view, plain = _ingest(name)
    out = to_arrow_table(view)
    assert "view" not in str(out.schema.field("c").type)
    assert not isinstance(out.schema.field("c").type, pa.BaseExtensionType)
    assert out.column("c").to_pylist() == plain.column("c").to_pylist()
    sliced = to_arrow_table(view.slice(1))
    assert sliced.column("c").to_pylist() == plain.column("c").to_pylist()[1:]


def test_a_cast_pyarrow_has_no_kernel_for_is_a_refusal_not_a_raw_error():
    with pytest.raises(coerce._CannotCast):
        coerce._cast_array(pa.array([1, 2]), pa.struct([pa.field("k", pa.int64())]))


def test_a_failure_of_the_rebuild_is_not_reported_as_an_unreadable_column(monkeypatch):
    view = pa.table({"c": pa.array([[1]], pa.list_view(pa.int64()))})

    def broken(col, pa_module):
        raise pa.ArrowInvalid("rebuild broke")

    monkeypatch.setattr(coerce, "_convert_column", broken)
    with pytest.raises(TypeError) as err:
        to_arrow_table(view)
    assert str(err.value) == (
        "column 'c' has Arrow type list_view<item: int64>, which ferrum could not rewrite "
        "to a plain type (rebuild broke); cast it to a plain type "
        "(e.g. pa.large_string()) before charting"
    )


def _extension_storage_view_type():
    class ViewStorageType(pa.ExtensionType):
        def __init__(self):
            super().__init__(pa.string_view(), "ferrum.test.view_storage")

        def __arrow_ext_serialize__(self):
            return b""

        @classmethod
        def __arrow_ext_deserialize__(cls, storage_type, serialized):
            return cls()

    return ViewStorageType()


def _plain_type_matrix() -> pa.Table:
    """One column per kind of type pyarrow has, none holding a view type."""
    types = pa.array([0, 1, 0], pa.int8())
    columns = {
        "int": pa.array([1, 2, 3]),
        "float": pa.array([1.0, 2.0, 3.0], pa.float32()),
        "bool": pa.array([True, False, None]),
        "string": pa.array(["a", "b", "c"]),
        "large_string": pa.array(["a", "b", "c"], pa.large_string()),
        "binary": pa.array([b"a", b"b", b"c"]),
        "decimal": pa.array([1, 2, 3], pa.decimal128(5, 1)),
        "null": pa.nulls(3, pa.null()),
        "timestamp": pa.array([1, 2, 3], pa.timestamp("ms")),
        "duration": pa.array([1, 2, 3], pa.duration("ns")),
        "list": pa.array([[1], None, [2, 3]]),
        "large_list": pa.array([[1], None, [2, 3]], pa.large_list(pa.int64())),
        "fixed_size_list": pa.array([[1, 2], None, [3, 4]], pa.list_(pa.int64(), 2)),
        "map": pa.array([[("a", 1)], None, []], pa.map_(pa.string(), pa.int64())),
        "struct": pa.array([{"k": 1}, None, {"k": 3}]),
        "dictionary": pa.array(["x", "y", "x"]).dictionary_encode(),
        "sparse_union": pa.UnionArray.from_sparse(
            types, [pa.array(["x", "y", "z"]), pa.array([1, 2, 3])]
        ),
        "dense_union": pa.UnionArray.from_dense(
            types, pa.array([0, 0, 1], pa.int32()), [pa.array(["x", "z"]), pa.array([2])]
        ),
        "run_end_encoded": pa.RunEndEncodedArray.from_arrays(
            pa.array([2, 3], pa.int32()), pa.array(["x", "z"])
        ),
        "extension_bool8": pa.ExtensionArray.from_storage(
            pa.bool8(), pa.array([1, 0, 1], pa.int8())
        ),
        "extension_json": pa.ExtensionArray.from_storage(
            pa.json_(), pa.array(['{"a": 1}', "[]", "1"])
        ),
        "extension_uuid": pa.ExtensionArray.from_storage(
            pa.uuid(), pa.array([bytes(16)] * 3, pa.binary(16))
        ),
        "chunked": pa.chunked_array([pa.array([1, 2]), pa.array([3])]),
    }
    return pa.table(columns)


def test_a_table_with_no_view_type_is_returned_untouched():
    tbl = _plain_type_matrix()
    assert to_arrow_table(tbl) is tbl


@pytest.mark.parametrize("name", _plain_type_matrix().column_names)
def test_a_plain_column_keeps_its_type_and_values_beside_a_view_column(name):
    plain = _plain_type_matrix()
    mixed = plain.append_column("v", _sv(["x", "y", "z"]))
    out = to_arrow_table(mixed)
    assert out.schema.field(name).type == plain.schema.field(name).type
    assert out.column(name).equals(plain.column(name))
    assert out.schema.field("v").type == pa.large_string()


def test_a_plain_column_inside_a_container_stays_an_extension_beside_a_view_sibling():
    ext = pa.ExtensionArray.from_storage(pa.bool8(), pa.array([1, 0, 1], pa.int8()))
    nested = pa.StructArray.from_arrays([ext, _sv(["x", "y", "z"])], ["flag", "text"])
    out = to_arrow_table(pa.table({"c": nested}))
    assert out.schema.field("c").type == pa.struct(
        [pa.field("flag", pa.bool8()), pa.field("text", pa.large_string())]
    )


def test_pandas_extension_columns_arrive_as_narwhals_built_them():
    import narwhals as nw
    import pandas as pd

    frame = pd.DataFrame(
        {
            "period": pd.period_range("2020-01", periods=3, freq="M"),
            "interval": pd.interval_range(0, 3),
        }
    )
    built = nw.from_native(frame, eager_only=True).to_arrow()
    assert to_arrow_table(frame).schema == built.schema


def test_an_extension_whose_storage_holds_a_view_type_loads_as_its_plain_storage():
    ext = pa.ExtensionArray.from_storage(_extension_storage_view_type(), _sv(["x", "y", "z"]))
    out = to_arrow_table(pa.table({"c": ext}))
    assert out.schema.field("c").type == pa.large_string()
    assert out.column("c").to_pylist() == ["x", "y", "z"]


def test_a_fixed_size_list_of_list_views_loads_every_row_past_a_slice_offset():
    lv = pa.array([[1], [2], [3], [4], [5], [6]], pa.list_view(pa.int64()))
    arr = pa.FixedSizeListArray.from_arrays(lv, 2)
    for start in range(3):
        out = to_arrow_table(pa.table({"c": arr}).slice(start))
        assert (
            out.column("c").to_pylist()
            == [
                [[1], [2]],
                [[3], [4]],
                [[5], [6]],
            ][start:]
        )


def _empty_union_columns() -> dict[str, pa.ChunkedArray]:
    def sparse(strings):
        n = len(strings)
        return pa.UnionArray.from_sparse(
            pa.array([0] * n, pa.int8()), [_sv(strings), pa.array([0] * n, pa.int64())]
        )

    def dense(strings):
        n = len(strings)
        return pa.UnionArray.from_dense(
            pa.array([0] * n, pa.int8()),
            pa.array(range(n), pa.int32()),
            [_sv(strings), pa.array([], pa.int64())],
        )

    return {
        "sparse_zero_rows": pa.chunked_array([sparse([])]),
        "dense_zero_rows": pa.chunked_array([dense([])]),
        "dense_sliced_to_zero": pa.chunked_array([dense(["x", "y"]).slice(2)]),
        "sparse_empty_chunk": pa.chunked_array([sparse(["x"]), sparse([]), sparse(["y"])]),
        "dense_empty_chunk": pa.chunked_array([dense(["x"]), dense([]), dense(["y"])]),
    }


@pytest.mark.parametrize("name", list(_empty_union_columns()))
def test_a_union_of_views_loads_at_every_length(name):
    col = _empty_union_columns()[name]
    out = to_arrow_table(pa.table({"c": col}))
    assert "view" not in str(out.schema.field("c").type)
    assert out.column("c").to_pylist() == col.to_pylist()


def _union_child(kind: str, strings: list[str]) -> pa.Array:
    """A child array holding a view type, ``len(strings)`` long, of the named kind."""
    n = len(strings)
    sv = pa.array(strings, pa.string_view())
    ints = pa.array(range(n), pa.int64())
    codes = pa.array([0] * n, pa.int8())
    if kind == "sparse_union":
        return pa.UnionArray.from_sparse(codes, [sv, ints])
    if kind == "dense_union":
        return pa.UnionArray.from_dense(
            codes, pa.array(range(n), pa.int32()), [sv, pa.array([], pa.int64())]
        )
    if kind == "list_view":
        return pa.array([[i] for i in range(n)], pa.list_view(pa.int64()))
    if kind == "list_of_union":
        union = pa.UnionArray.from_sparse(pa.array([0] * n, pa.int8()), [sv, ints])
        return pa.ListArray.from_arrays(pa.array(range(n + 1), pa.int32()), union)
    return sv


def _wrap(container: str, child: pa.Array) -> pa.Array:
    n = len(child)
    if container == "struct":
        return pa.StructArray.from_arrays([child], ["k"])
    if container == "list":
        return pa.ListArray.from_arrays(pa.array(range(n + 1), pa.int32()), child)
    if container == "large_list":
        return pa.LargeListArray.from_arrays(pa.array(range(n + 1), pa.int64()), child)
    if container == "fixed_size_list":
        return pa.FixedSizeListArray.from_arrays(child, 1)
    keys = pa.array([str(i) for i in range(n)], pa.string())
    return pa.MapArray.from_arrays(pa.array(range(n + 1), pa.int32()), keys, child)


_CONTAINERS = ["struct", "list", "large_list", "fixed_size_list", "map"]
_CHILDREN = ["string_view", "list_view", "sparse_union", "dense_union", "list_of_union"]


@pytest.mark.parametrize("shape", ["zero_rows", "empty_chunk", "sliced_to_zero"])
@pytest.mark.parametrize("child", _CHILDREN)
@pytest.mark.parametrize("container", _CONTAINERS)
def test_a_container_holding_a_view_child_loads_at_every_length(container, child, shape):
    def make(strings):
        return _wrap(container, _union_child(child, strings))

    if shape == "zero_rows":
        col = pa.chunked_array([make([])])
    elif shape == "empty_chunk":
        col = pa.chunked_array([make(["x"]), make([]), make(["y"])])
    else:
        col = pa.chunked_array([make(["x", "y"]).slice(2)])
    out = to_arrow_table(pa.table({"c": col}))
    assert "view" not in str(out.schema.field("c").type)
    assert out.column("c").to_pylist() == col.to_pylist()


_EXTENSION_BUILDERS = {
    "view_storage": """
        class T(pa.ExtensionType):
            def __init__(self):
                super().__init__(pa.string_view(), "ferrum.test.view_storage")
            def __arrow_ext_serialize__(self):
                return b""
            @classmethod
            def __arrow_ext_deserialize__(cls, storage_type, serialized):
                return cls()
        storage = pa.array(["x", "y", "z"], pa.string_view())
    """,
    "plain_storage": """
        class T(pa.ExtensionType):
            def __init__(self):
                super().__init__(pa.string(), "ferrum.test.plain_storage")
            def __arrow_ext_serialize__(self):
                return b""
            @classmethod
            def __arrow_ext_deserialize__(cls, storage_type, serialized):
                return cls()
        storage = pa.array(["x", "y", "z"])
    """,
}

_WARNING_RUNNER = """
import warnings
import pyarrow as pa
import ferrum as fm

{builder}
table = pa.table({{"a": [1, 2, 3], "c": pa.ExtensionArray.from_storage(T(), storage)}})
chart = fm.Chart(table).mark_point().encode(x="a", y="a", color="c")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    if "{route}" == "interactive":
        chart.interactive().to_html()
    else:
        chart.to_svg()
print(repr([(w.category.__name__, str(w.message)) for w in caught]))
"""

_POLARS_UNREGISTERED = (
    "Extension type 'ferrum.test.plain_storage' is not registered; loading as its storage "
    "type.\n\nTo avoid this warning, register the extension type or set environment variable "
    "'POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR' to 'load_as_storage' or 'load_as_extension'."
    "\n\nIn Polars 2.0, the default behavior will change to 'load_as_extension'."
)
_VIEW_STORAGE_DROPPED = (
    "Extension type 'ferrum.test.view_storage' holds a view type; loading as its plain "
    "storage type."
)


def _warnings_of(builder: str, route: str) -> list[tuple[str, str]]:
    script = _WARNING_RUNNER.format(
        builder=textwrap.dedent(_EXTENSION_BUILDERS[builder]).strip(), route=route
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    return ast.literal_eval(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("route", ["static", "interactive"])
def test_the_view_storage_warning_names_the_users_frame(route):
    """The warning is attributed to the caller's file, not to a frame inside ferrum."""
    script = _WARNING_RUNNER.format(
        builder=textwrap.dedent(_EXTENSION_BUILDERS["view_storage"]).strip(), route=route
    ).replace("(w.category.__name__, str(w.message))", "(w.category.__name__, w.filename)")
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    caught = ast.literal_eval(proc.stdout.strip().splitlines()[-1])
    assert caught and {filename for _category, filename in caught} == {"<string>"}


@pytest.mark.parametrize(
    ("route", "renders"),
    [("static", 1), ("interactive", 2)],
)
def test_an_extension_with_view_storage_names_the_wrapper_it_drops(route, renders):
    # The interactive route reads the chart's data in two passes, so the ingest of one
    # column runs twice there; the default warning filter shows the repeat once.
    assert _warnings_of("view_storage", route) == [("UserWarning", _VIEW_STORAGE_DROPPED)] * renders


@pytest.mark.parametrize("route", ["static", "interactive"])
def test_an_extension_with_plain_storage_is_named_by_polars(route):
    assert _warnings_of("plain_storage", route) == [("UserWarning", _POLARS_UNREGISTERED)]


def test_the_dropped_extension_warning_is_raised_once_per_column_not_per_chunk():
    ext = _extension_storage_view_type()
    chunks = [pa.ExtensionArray.from_storage(ext, _sv(["x"])) for _ in range(3)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        to_arrow_table(pa.table({"c": pa.chunked_array(chunks)}))
    assert [(w.category.__name__, str(w.message)) for w in caught] == [
        ("UserWarning", _VIEW_STORAGE_DROPPED)
    ]


# Every door that reads the caller's table beside ``to_arrow_table`` (facet level counting,
# the empty-slice schema probe) or builds a composition must read a cast ``string_view``
# column exactly as its ``large_string`` twin. Each builder takes the table as ``src``.
_DOORS = {
    "facet_by_view": "fm.Chart(src).mark_point().encode(x='a', y='b').facet('c')",
    "facet_by_view_wrap": "fm.Chart(src).mark_point().encode(x='a', y='b').facet('c', ncols=1)",
    "facet_col_by_view": "fm.Chart(src).mark_point().encode(x='a', y='b').facet(col='c')",
    "facet_row_by_view": "fm.Chart(src).mark_point().encode(x='a', y='b').facet(row='c')",
    "facet_col_view_encoded": (
        "fm.Chart(src).mark_point().encode(x='a', y='b', color='c').facet(col='g')"
    ),
    "facet_row_view_encoded": (
        "fm.Chart(src).mark_point().encode(x='a', y='b', color='c').facet(row='g')"
    ),
    "facet_wrap_view_encoded": (
        "fm.Chart(src).mark_point().encode(x='a', y='b', color='c').facet('g', ncols=1)"
    ),
    "plus": (
        "fm.Chart(src).mark_point().encode(x='a', y='b', color='c')"
        " + fm.Chart(src).mark_line().encode(x='a', y='b', color='c')"
    ),
    "chart_layer": (
        "fm.Chart(src).mark_point().encode(x='a', y='b', color='c')"
        ".layer(fm.Layer(src, 'line', encoding={'x': 'a', 'y': 'b', 'color': 'c'}))"
    ),
    "layer_chart": (
        "fm.LayerChart(fm.Chart(src).mark_point().encode(x='a', y='b', color='c'),"
        " fm.Chart(src).mark_line().encode(x='a', y='b', color='c'))"
    ),
    "hconcat": (
        "fm.hconcat(fm.Chart(src).mark_point().encode(x='a', y='b', color='c'),"
        " fm.Chart(src).mark_bar().encode(x='c', y='b'))"
    ),
    "catplot_category": "fm.catplot(src, x='c', y='b', kind='bar')",
    "displot_category": "fm.displot(src, x='b', hue='c')",
}

_DOOR_RUNNER = """
import hashlib, sys
import pyarrow as pa
import ferrum as fm

strings = pa.array(["x", "y", "x", "y", "x", "y"])
base = {{"a": [1, 2, 3, 4, 5, 6], "b": [2, 3, 1, 5, 4, 2], "g": ["p", "p", "q", "q", "p", "q"]}}
view = pa.table({{**base, "c": strings.cast(pa.string_view())}})
twin = pa.table({{**base, "c": strings.cast(pa.large_string())}})
route = sys.argv[1]

def render(src):
    chart = {door}
    if route == "interactive":
        return chart.interactive().to_html()
    return chart.to_svg()

def outcome(src):
    try:
        return "svg " + hashlib.sha256(render(src).encode()).hexdigest()
    except Exception as e:
        return type(e).__name__ + ": " + str(e)

print(outcome(view))
print(outcome(twin))
"""


@pytest.mark.parametrize("route", ["static", "interactive"])
@pytest.mark.parametrize("door", list(_DOORS))
def test_a_view_column_renders_as_its_twin_through_every_door(door, route):
    script = _DOOR_RUNNER.format(door=_DOORS[door])
    proc = subprocess.run(
        [sys.executable, "-c", script, route], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, f"rc={proc.returncode}\n{proc.stderr[-1200:]}"
    got, twin = proc.stdout.strip().splitlines()[-2:]
    assert twin.startswith("svg "), f"the twin itself failed: {twin}"
    assert got == twin
