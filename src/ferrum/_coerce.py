"""Data ingestion: normalize any supported input to a pyarrow.Table.

Supports (per spec §3.18):
- polars.DataFrame, polars.LazyFrame
- pyarrow.Table, pyarrow.RecordBatch
- pandas + modin + cuDF + dask + ibis (via narwhals)
- dict[str, list], list[dict]
- numpy.ndarray (2D, auto-named "col_0", "col_1", ...)

Raises TypeError for unsupported types or numpy 1D without column names.

Boundary normalization (``normalize_for_rust``)
------------------------------------------------
``to_arrow_table`` converts any input to an Arrow table.  Before the table
crosses the Rust CDI boundary, callers must run it through
``normalize_for_rust``, which owns the single canonical list of Arrow dtypes
the Rust renderer accepts and their required forms:

- ``date32`` / ``date64`` → ``timestamp[ms]``
- ``timestamp[us]`` / ``timestamp[ns]`` / ``timestamp[s]`` → ``timestamp[ms]``
- ``Duration(any unit)`` → ``timestamp[ms]``  (same temporal scaling as Date)
- ``Dictionary``-encoded → decoded to the plain value type
- ``Null`` (all-None, unknown type) → ``float64``

Ingest normalization (``to_arrow_table``)
-----------------------------------------
Every input that reaches polars passes through ``to_arrow_table``, which also
rewrites Arrow view types to their plain twins, because ``pl.from_arrow``
crashes the process on a ``string_view`` or ``binary_view`` array made by a cast:

- ``string_view`` → ``large_string``, ``binary_view`` → ``large_binary``
- ``list_view`` / ``large_list_view`` → ``large_list``
- applied recursively through list, large list, fixed-size list, struct, map and
  dictionary value types; a column without view types is not cast

The SVG path (``_render.py::_render_inputs``) and the interactive path both
call ``normalize_for_rust(to_arrow_table(data))`` so the contract is applied
exactly once, by one function.
"""

from __future__ import annotations

import warnings
from typing import Any


def _normalize_column(col: "pyarrow.Array", pa: object, pc: object) -> "tuple[pyarrow.Array, bool]":
    """Normalize a single decoded (non-dictionary) Arrow column to a Rust-safe dtype.

    Returns ``(normalized_col, changed)`` where ``changed`` is True when the
    column was modified.  Callers must not pass a Dictionary-typed column here;
    dictionary decoding is the caller's responsibility (``normalize_for_rust``
    does it before calling this function).

    Normalizations (applied in priority order):

    - ``date32`` / ``date64``                    → ``timestamp[ms]``
    - ``timestamp[us/ns/s]``                     → ``timestamp[ms]``
    - ``duration(any unit)``                     → ``timestamp[ms]``
    - ``null`` (all-None, unknown type)          → ``float64``
    - everything else                            → unchanged
    """
    field_type = col.type

    if pa.types.is_date32(field_type) or pa.types.is_date64(field_type):
        return col.cast(pa.timestamp("ms")), True

    if pa.types.is_timestamp(field_type) and field_type.unit != "ms":
        # The cast keeps the column's time zone, as a ``timestamp[ms]`` column already does.
        return col.cast(pa.timestamp("ms", tz=field_type.tz)), True

    if pa.types.is_duration(field_type):
        # Arrow does not support a direct duration→timestamp cast, and
        # duration[ns/us]→duration[ms] is a lossy safe-cast (Arrow refuses
        # values not divisible by the scale factor).  Instead we reinterpret
        # the integer representation with unit-aware scaling:
        #   duration[unit] → int64          (raw ticks in source unit)
        #   int64 * scale_to_ms             (convert ticks to ms)
        #   int64 → timestamp[ms]           (attach temporal semantics)
        unit = field_type.unit  # "ns", "us", "ms", or "s"
        as_int = col.cast(pa.int64())
        if unit == "ns":
            ms_ints = pc.divide(as_int, 1_000_000)
        elif unit == "us":
            ms_ints = pc.divide(as_int, 1_000)
        elif unit == "ms":
            ms_ints = as_int
        else:  # "s"
            # NOTE: pc.multiply wraps on overflow; only affects durations
            # exceeding ~292 million years, which is not a practical concern.
            ms_ints = pc.multiply(as_int, 1_000)
        return ms_ints.cast(pa.timestamp("ms")), True

    if pa.types.is_null(field_type):
        return col.cast(pa.float64()), True

    return col, False


def normalize_for_rust(tbl: "pyarrow.Table") -> "pyarrow.Table":
    """Apply CDI-boundary dtype casts required by the Rust renderer.

    This is the pre-Rust boundary normalizer that **owns** the
    ``Duration``→``timestamp[ms]``, ``Dictionary``-decode, and
    ``Null``→``float64`` rules, and **re-asserts** the date/timestamp
    canonicalization that ``to_arrow_table`` also applies at ingest (so a
    caller passing a raw pyarrow table directly is still safe).  Note that
    ``to_arrow_table``'s polars and pyarrow branches already cast
    ``date32``/``date64``→``timestamp[ms]`` and non-ms timestamps at ingest,
    making those casts here idempotent for the common path.  Both the SVG
    render path and the interactive render path must call this function after
    ``to_arrow_table`` and before handing data to any ``ferrum._core``
    binding.

    Normalizations applied (in column order, one pass):

    - ``date32`` / ``date64`` → ``timestamp[ms]``
    - ``timestamp[us]`` / ``timestamp[ns]`` / ``timestamp[s]`` →
      ``timestamp[ms]``
    - ``duration(any unit)`` → ``timestamp[ms]``  (temporal scaling; matches
      how Date is handled so a Duration axis and a Date axis use the same
      epoch-millisecond units)
    - ``Dictionary``-encoded column → decoded to the plain value type, then
      the decoded column passes through the same full normalization above
      (date, timestamp, duration, null are all handled after decode)
    - ``Null`` (all-None column with unknown type) → ``float64``

    Parameters
    ----------
    tbl : pyarrow.Table
        An Arrow table produced by ``to_arrow_table``.

    Returns
    -------
    pyarrow.Table
        The same table with normalized dtypes.  Returned as-is when no
        column requires a cast.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    new_cols: list = []
    needs_rebuild = False
    for i in range(len(tbl.schema)):
        col = tbl.column(i)
        field_type = tbl.schema.field(i).type

        if pa.types.is_dictionary(field_type):
            # Decode first, then apply the full per-column normalization so
            # that a dict-over-duration, dict-over-timestamp[us], or
            # dict-over-null is handled identically to the non-dict path.
            col = pc.dictionary_decode(col)
            col, _ = _normalize_column(col, pa, pc)
            needs_rebuild = True
        else:
            col, changed = _normalize_column(col, pa, pc)
            if changed:
                needs_rebuild = True

        new_cols.append(col)

    if not needs_rebuild:
        return tbl
    return pa.table(new_cols, names=[tbl.schema.field(i).name for i in range(len(tbl.schema))])


def _plain_type(t: "pyarrow.DataType", pa: object) -> "pyarrow.DataType":
    """Return ``t`` with every view type in its type tree replaced by its plain twin.

    Returns a type equal to ``t`` when the tree holds no view type.  An
    extension type whose storage holds a view type is replaced by its plain
    storage type; any other extension type is returned unchanged.
    """
    types = pa.types
    if isinstance(t, pa.BaseExtensionType):
        storage = _plain_type(t.storage_type, pa)
        return t if storage == t.storage_type else storage
    if types.is_string_view(t):
        return pa.large_string()
    if types.is_binary_view(t):
        return pa.large_binary()
    if types.is_list_view(t) or types.is_large_list_view(t):
        return pa.large_list(_plain_field(t.value_field, pa))
    if types.is_large_list(t):
        field = _plain_field(t.value_field, pa)
        return t if field.type == t.value_type else pa.large_list(field)
    if types.is_fixed_size_list(t):
        field = _plain_field(t.value_field, pa)
        return t if field.type == t.value_type else pa.list_(field, t.list_size)
    if types.is_map(t):
        key = _plain_field(t.key_field, pa)
        item = _plain_field(t.item_field, pa)
        if key.type == t.key_type and item.type == t.item_type:
            return t
        return pa.map_(key.type, item, keys_sorted=t.keys_sorted)
    if types.is_list(t):
        field = _plain_field(t.value_field, pa)
        return t if field.type == t.value_type else pa.list_(field)
    if types.is_struct(t):
        fields = [_plain_field(t.field(i), pa) for i in range(t.num_fields)]
        if all(new.type == t.field(i).type for i, new in enumerate(fields)):
            return t
        return pa.struct(fields)
    if types.is_union(t):
        fields = [_plain_field(t.field(i), pa) for i in range(t.num_fields)]
        if all(new.type == t.field(i).type for i, new in enumerate(fields)):
            return t
        return pa.union(fields, t.mode, t.type_codes)
    if types.is_run_end_encoded(t):
        value = _plain_type(t.value_type, pa)
        return t if value == t.value_type else pa.run_end_encoded(t.run_end_type, value)
    if types.is_dictionary(t):
        value = _plain_type(t.value_type, pa)
        if value == t.value_type:
            return t
        return pa.dictionary(t.index_type, value, ordered=t.ordered)
    return t


def _plain_field(field: "pyarrow.Field", pa: object) -> "pyarrow.Field":
    """Return ``field`` with its type rewritten by :func:`_plain_type`."""
    plain = _plain_type(field.type, pa)
    return field if plain == field.type else field.with_type(plain)


class _CannotCast(Exception):
    """pyarrow has no cast from a view type to its plain twin for this array."""


def _cast_array(arr: "pyarrow.Array", target: "pyarrow.DataType") -> "pyarrow.Array":
    """Cast ``arr`` with pyarrow's own cast; a cast pyarrow refuses raises ``_CannotCast``."""
    import pyarrow as pa

    try:
        return arr.cast(target)
    except pa.ArrowNotImplementedError as e:
        raise _CannotCast(str(e)) from e


def _needs_rebuild(t: "pyarrow.DataType", pa: object) -> bool:
    """True when ``t`` holds a type pyarrow cannot cast in place.

    pyarrow's cast of a list view drops the last row's values and aborts the
    process on some chunked inputs, and it has no cast for a union, a
    run-end-encoded array or an extension array holding a view type.
    """
    types = pa.types
    if (
        types.is_list_view(t)
        or types.is_large_list_view(t)
        or types.is_union(t)
        or types.is_run_end_encoded(t)
        or isinstance(t, pa.BaseExtensionType)
    ):
        return True
    if types.is_dictionary(t):
        return _needs_rebuild(t.value_type, pa)
    if types.is_nested(t):
        return any(_needs_rebuild(t.field(i).type, pa) for i in range(t.num_fields))
    return False


def _rebased_offsets(arr: "pyarrow.Array", pa: object) -> "tuple[pyarrow.Array, int, int]":
    """Offsets of a list or map array re-based to zero, with the start and length of its values.

    ``from_arrays`` rejects an offsets array that carries a slice offset beside a
    null mask, so the re-based offsets are a fresh array.
    """
    import pyarrow.compute as pc

    offsets = arr.offsets
    start = offsets[0].as_py()
    stop = offsets[len(offsets) - 1].as_py()
    rebased = pc.subtract(offsets, pa.scalar(start, offsets.type))
    return rebased, start, stop - start


def _empty_array(t: "pyarrow.DataType", pa: object) -> "pyarrow.Array":
    """A zero-length array of type ``t``.

    ``pa.array`` cannot build an empty union, or an empty container of one, so
    every container is built around an empty child.
    """
    types = pa.types
    if types.is_union(t):
        children = [_empty_array(t.field(i).type, pa) for i in range(t.num_fields)]
        codes = pa.array([], pa.int8())
        names = [t.field(i).name for i in range(t.num_fields)]
        if t.mode == "sparse":
            return pa.UnionArray.from_sparse(
                codes, children, field_names=names, type_codes=list(t.type_codes)
            )
        return pa.UnionArray.from_dense(
            codes,
            pa.array([], pa.int32()),
            children,
            field_names=names,
            type_codes=list(t.type_codes),
        )
    if types.is_struct(t):
        children = [_empty_array(t.field(i).type, pa) for i in range(t.num_fields)]
        return pa.StructArray.from_arrays(children, fields=list(t))
    if types.is_map(t):
        offsets = pa.array([0], pa.int32())
        keys = _empty_array(t.key_type, pa)
        return pa.MapArray.from_arrays(offsets, keys, _empty_array(t.item_type, pa), type=t)
    if types.is_large_list(t):
        offsets = pa.array([0], pa.int64())
        return pa.LargeListArray.from_arrays(offsets, _empty_array(t.value_type, pa), type=t)
    if types.is_list(t):
        offsets = pa.array([0], pa.int32())
        return pa.ListArray.from_arrays(offsets, _empty_array(t.value_type, pa), type=t)
    if types.is_fixed_size_list(t):
        return pa.FixedSizeListArray.from_arrays(_empty_array(t.value_type, pa), type=t)
    if types.is_dictionary(t):
        indices = pa.array([], t.index_type)
        return pa.DictionaryArray.from_arrays(
            indices, _empty_array(t.value_type, pa), ordered=t.ordered
        )
    return pa.array([], type=t)


def _plain_array(arr: "pyarrow.Array", pa: object) -> "pyarrow.Array":
    """Return ``arr`` with every view type in its type tree replaced by its plain twin.

    An array whose type pyarrow can cast in place is cast.  Otherwise the
    container is rebuilt around its rewritten children: a list view from its
    flattened values and per-row lengths, the others from their offsets,
    fields or run values.
    """
    import pyarrow.compute as pc

    t = arr.type
    target = _plain_type(t, pa)
    if target == t:
        return arr
    if not _needs_rebuild(t, pa):
        return _cast_array(arr, target)
    if len(arr) == 0:
        return _empty_array(target, pa)
    types = pa.types
    mask = pc.invert(arr.is_valid()) if arr.null_count else None
    if isinstance(t, pa.BaseExtensionType):
        return _plain_array(arr.storage, pa)
    if types.is_list_view(t) or types.is_large_list_view(t):
        lengths = pc.fill_null(pc.list_value_length(arr), 0).cast(pa.int64())
        offsets = pa.concat_arrays([pa.array([0], pa.int64()), pc.cumulative_sum(lengths)])
        values = _plain_array(pc.list_flatten(arr), pa)
        return pa.LargeListArray.from_arrays(offsets, values, type=target, mask=mask)
    if types.is_map(t):
        offsets, start, count = _rebased_offsets(arr, pa)
        keys = _plain_array(arr.keys.slice(start, count), pa)
        items = _plain_array(arr.items.slice(start, count), pa)
        return pa.MapArray.from_arrays(offsets, keys, items, type=target, mask=mask)
    if types.is_large_list(t) or types.is_list(t):
        offsets, start, count = _rebased_offsets(arr, pa)
        values = _plain_array(arr.values.slice(start, count), pa)
        array_type = pa.LargeListArray if types.is_large_list(t) else pa.ListArray
        return array_type.from_arrays(offsets, values, type=target, mask=mask)
    if types.is_fixed_size_list(t):
        size = t.list_size
        values = _plain_array(arr.values.slice(arr.offset * size, len(arr) * size), pa)
        return pa.FixedSizeListArray.from_arrays(values, type=target, mask=mask)
    if types.is_struct(t):
        children = [_plain_array(arr.field(i), pa) for i in range(t.num_fields)]
        return pa.StructArray.from_arrays(children, fields=list(target), mask=mask)
    if types.is_union(t):
        names = [target.field(i).name for i in range(target.num_fields)]
        children = [_plain_array(arr.field(i), pa) for i in range(t.num_fields)]
        # ``type_codes`` and ``offsets`` read the unsliced buffers from the
        # start, so a sliced union takes them from its buffers at its offset (and
        # the codes are copied to an offset-free array, as its children are).
        buffers = arr.buffers()
        codes = pc.add(
            pa.Array.from_buffers(pa.int8(), len(arr), [None, buffers[1]], offset=arr.offset),
            pa.scalar(0, pa.int8()),
        )
        if t.mode == "sparse":
            return pa.UnionArray.from_sparse(
                codes, children, field_names=names, type_codes=list(t.type_codes)
            )
        offsets = pc.add(
            pa.Array.from_buffers(pa.int32(), len(arr), [None, buffers[2]], offset=arr.offset),
            pa.scalar(0, pa.int32()),
        )
        return pa.UnionArray.from_dense(
            codes, offsets, children, field_names=names, type_codes=list(t.type_codes)
        )
    if types.is_run_end_encoded(t):
        # pyarrow has no decode kernel for a view-valued array: rewrite the run
        # values, expand them by run, and encode the result again.
        import numpy as np

        run_ends = arr.run_ends.to_numpy(zero_copy_only=False)
        run_lengths = np.diff(run_ends, prepend=0)
        run_of_row = np.repeat(np.arange(len(run_ends)), run_lengths)
        expanded = _plain_array(arr.values, pa).take(
            pa.array(run_of_row[arr.offset : arr.offset + len(arr)])
        )
        return pc.run_end_encode(expanded, run_end_type=t.run_end_type)
    # dictionary: ``_plain_type`` only differs here through the value type.
    dictionary = _plain_array(arr.dictionary, pa)
    return pa.DictionaryArray.from_arrays(arr.indices, dictionary, ordered=t.ordered)


def _convert_column(col: "pyarrow.ChunkedArray", pa: object) -> "pyarrow.ChunkedArray":
    """Rewrite the view types of one column, chunk by chunk."""
    target = _plain_type(col.type, pa)
    return pa.chunked_array([_plain_array(c, pa) for c in col.chunks], type=target)


def _dropped_extension_names(t: "pyarrow.DataType", pa: object) -> list[str]:
    """Names of the extension types in ``t`` that the rewrite unwraps to plain storage.

    An extension type is unwrapped only when its storage holds a view type.
    Each name is listed once, in tree order.
    """
    types = pa.types
    names: list[str] = []
    if isinstance(t, pa.BaseExtensionType):
        if _plain_type(t, pa) != t:
            names.append(t.extension_name)
        children = [t.storage_type]
    elif types.is_dictionary(t) or types.is_run_end_encoded(t):
        children = [t.value_type]
    elif types.is_nested(t):
        children = [t.field(i).type for i in range(t.num_fields)]
    else:
        children = []
    for child in children:
        names.extend(n for n in _dropped_extension_names(child, pa) if n not in names)
    return names


def _rewrite_view_columns(tbl: "pyarrow.Table", pa: object) -> "pyarrow.Table":
    """Rewrite every column whose type tree holds a view type to its plain twin.

    Returns ``tbl`` itself when no column holds a view type.  Raises
    ``TypeError`` naming the column when pyarrow cannot cast it, or when the
    rewrite itself fails.
    """
    for i, field in enumerate(tbl.schema):
        plain = _plain_type(field.type, pa)
        if plain == field.type:
            continue
        try:
            col = _convert_column(tbl.column(i), pa)
            tbl = tbl.set_column(i, field.with_type(plain), col)
            for name in _dropped_extension_names(field.type, pa):
                warnings.warn(
                    f"Extension type {name!r} holds a view type; loading as its plain "
                    f"storage type.",
                    UserWarning,
                    stacklevel=2,
                )
        except _CannotCast as e:
            raise TypeError(
                f"column {field.name!r} has Arrow type {field.type}, which ferrum cannot "
                f"read; cast it to a plain type (e.g. pa.large_string()) before charting"
            ) from e
        except (pa.ArrowException, ValueError) as e:
            raise TypeError(
                f"column {field.name!r} has Arrow type {field.type}, which ferrum could not "
                f"rewrite to a plain type ({e}); cast it to a plain type "
                f"(e.g. pa.large_string()) before charting"
            ) from e
    return tbl


def to_polars(data: Any) -> "polars.DataFrame":
    """Coerce any supported input to a ``polars.DataFrame``.

    Backed by :func:`to_arrow_table`, so every input type accepted by
    ``Chart(data=...)`` is accepted here: polars, pyarrow, narwhals-compatible
    (pandas/modin/cuDF/dask/ibis), dict, list, numpy 2D. Passing a
    ``polars.DataFrame`` returns it unchanged; everything else routes through
    ``to_arrow_table`` and inherits its normalization (datetime unit
    canonicalization, categorical/dictionary decoding, etc.).

    This is the single canonical to-polars entry point; do not reimplement
    this dispatch elsewhere.
    """
    import polars as pl

    if isinstance(data, pl.DataFrame):
        return data
    return pl.from_arrow(to_arrow_table(data))


def to_arrow_table(data: Any) -> "pyarrow.Table":
    """Normalize any supported input to a ``pyarrow.Table``.

    Dispatch order (fastest first):

    1. ``polars.DataFrame`` / ``polars.LazyFrame`` — zero-copy via Arrow CDI.
    2. ``pyarrow.Table`` / ``pyarrow.RecordBatch`` — passthrough or single-batch wrap.
    3. ``dict[str, list]`` — ``pa.Table.from_pydict``.
    4. ``list[dict]`` — ``pa.Table.from_pylist``.
    5. ``numpy.ndarray`` (2D) — columns named ``col_0``, ``col_1``, …
    6. Pandas / modin / cuDF / dask / ibis — via ``narwhals``.

    Parameters
    ----------
    data : Any
        Input data in any of the supported formats listed above.

    Returns
    -------
    pyarrow.Table
        A fully materialized Arrow table with named columns.

    Raises
    ------
    ValueError
        If ``data`` is ``None``, or if a ``list`` input is empty.
    TypeError
        If ``data`` is a 1D numpy array (column name required), a numpy
        array with ``ndim > 2``, a list of non-dict items, or an Arrow column
        holding a view type that pyarrow cannot cast to its plain twin (or
        whose rewrite fails); the message names the column and the remedy.
    ImportError
        If the input type requires ``narwhals`` and it is not installed.

    Warns
    -----
    UserWarning
        Once per column for each extension type whose storage holds a view
        type: the column loads as its plain storage type and the extension
        wrapper is dropped, as polars does for an extension type it has no
        registration for.

    Examples
    --------
    >>> import pyarrow as pa
    >>> tbl = to_arrow_table({"a": [1, 2], "b": [3.0, 4.0]})
    >>> tbl.schema.names
    ['a', 'b']
    """
    import pyarrow as pa

    if data is None:
        raise ValueError(
            "Chart(data=None) requires per-layer data. "
            "Construct layers with individual data= arguments and pass them via Chart.layer(), "
            "or use the + operator to combine independent-data charts."
        )

    # Fast path: polars (zero-copy via Arrow C Data Interface)
    # K5: also accept polars.Series — convert to single-column DataFrame first.
    try:
        import polars as pl

        if isinstance(data, pl.Series):
            data = data.to_frame()
        if isinstance(data, pl.DataFrame):
            casts = []
            for c in data.columns:
                dt = data[c].dtype
                if dt == pl.Date:
                    casts.append(pl.col(c).cast(pl.Datetime("ms")))
                elif isinstance(dt, pl.Datetime) and dt.time_unit != "ms":
                    # Normalize Datetime(us) and Datetime(ns) to Datetime(ms) so the
                    # Rust temporal scale produces correct epoch-millisecond tick values.
                    # Datetime(ms) is the canonical temporal unit for the renderer.
                    casts.append(pl.col(c).cast(pl.Datetime("ms", dt.time_zone)))
                elif dt == pl.Categorical or isinstance(dt, pl.Enum):
                    casts.append(pl.col(c).cast(pl.Utf8))
                # Duration columns: do NOT pre-cast here.  polars.to_arrow()
                # produces Arrow duration[unit], which normalize_for_rust then
                # converts to timestamp[ms] uniformly for both frontends.
            if casts:
                data = data.with_columns(casts)
            return data.to_arrow()
        if isinstance(data, pl.LazyFrame):
            return to_arrow_table(data.collect())
    except ImportError:
        pass

    # Fast path: pyarrow native
    # K5: accept pandas.Series — convert to single-column DataFrame and fall
    #     through to the narwhals path below.
    try:
        import pandas as pd

        if isinstance(data, pd.Series):
            name = data.name if data.name is not None else "value"
            data = data.to_frame(name=str(name))
            # Fall through to the narwhals path at the bottom of this function.
    except ImportError:
        pass

    if isinstance(data, pa.Table):
        data = _rewrite_view_columns(data, pa)
        # K7: cast date32/date64 columns to timestamp[ms] for Rust compat.
        # Also normalize timestamp[us] and timestamp[ns] to timestamp[ms] so the
        # Rust temporal scale produces correct epoch-millisecond tick values.
        new_cols = []
        needs_cast = False
        for i in range(len(data.schema)):
            col = data.column(i)
            field_type = data.schema.field(i).type
            if pa.types.is_date32(field_type) or pa.types.is_date64(field_type):
                col = col.cast(pa.timestamp("ms"))
                needs_cast = True
            elif pa.types.is_timestamp(field_type) and field_type.unit != "ms":
                col = col.cast(pa.timestamp("ms", tz=field_type.tz))
                needs_cast = True
            new_cols.append(col)
        if needs_cast:
            return pa.table(
                new_cols, names=[data.schema.field(i).name for i in range(len(data.schema))]
            )
        return data
    if isinstance(data, pa.RecordBatch):
        return to_arrow_table(pa.Table.from_batches([data]))

    # Direct conversions: dict, list, numpy
    if isinstance(data, dict):
        # GeoJSON FeatureCollection detection: split properties→columns, geometry→column.
        if _is_geojson_feature_collection(data):
            return _geojson_to_arrow(data)
        # Bare Geometry or GeometryCollection: wrap in a synthetic FeatureCollection.
        if _is_geojson_geometry_root(data):
            wrapped = _wrap_geometry_as_feature_collection(data)
            return _geojson_to_arrow(wrapped)
        return _rewrite_view_columns(pa.Table.from_pydict(data), pa)
    if isinstance(data, list):
        if not data:
            raise ValueError("Cannot construct Chart from empty list")
        if not isinstance(data[0], dict):
            raise TypeError(
                f"Chart(list) expects a list of dicts (one per row), got list of "
                f"{type(data[0]).__name__}"
            )
        return _rewrite_view_columns(pa.Table.from_pylist(data), pa)

    # numpy
    try:
        import numpy as np

        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                raise TypeError(
                    "1D numpy arrays need column names — pass `Chart({'value': arr})` "
                    "or `Chart(arr.reshape(-1, 1), columns=['value'])`."
                )
            if data.ndim == 2:
                cols = [f"col_{i}" for i in range(data.shape[1])]
                return pa.table({cols[i]: data[:, i] for i in range(data.shape[1])})
            raise TypeError(f"numpy arrays with ndim={data.ndim} not supported (use 2D)")
    except ImportError:
        pass

    # Everything else: try narwhals
    try:
        import narwhals as nw
    except ImportError as e:
        raise ImportError(
            f"Input type {type(data).__name__} requires narwhals. "
            f"Install with `pip install narwhals` (or use polars/pyarrow directly)."
        ) from e

    try:
        nw_df = nw.from_native(data, eager_only=True)
        # Route the narwhals-produced Arrow table back through to_arrow_table so
        # that timestamp normalization (timestamp[us/ns] → timestamp[ms]) is applied
        # consistently for pandas/modin/cuDF/dask inputs.
        return to_arrow_table(nw_df.to_arrow())
    except (TypeError, NotImplementedError) as e:
        raise TypeError(
            f"Unsupported data type: {type(data).__name__}. "
            f"Supported: polars, pyarrow, pandas, modin, cuDF, dask, ibis, dict, list, numpy 2D. "
            f"Underlying error: {e}"
        ) from e


# ── GeoJSON FeatureCollection support ────────────────────────────────────────


def _is_geojson_feature_collection(data: dict) -> bool:
    """Return True if *data* looks like a GeoJSON FeatureCollection."""
    return data.get("type") == "FeatureCollection" and isinstance(data.get("features"), list)


_GEOJSON_GEOMETRY_TYPES = frozenset(
    {
        "Point",
        "MultiPoint",
        "LineString",
        "MultiLineString",
        "Polygon",
        "MultiPolygon",
        "GeometryCollection",
    }
)


def _is_geojson_geometry_root(data: dict) -> bool:
    """Return True if *data* is a bare GeoJSON Geometry or GeometryCollection."""
    t = data.get("type")
    return isinstance(t, str) and t in _GEOJSON_GEOMETRY_TYPES


def _wrap_geometry_as_feature_collection(geom: dict) -> dict:
    """Wrap a bare GeoJSON Geometry into a synthetic single-feature FeatureCollection."""
    return {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "geometry": geom, "properties": {}}],
    }


def _geojson_to_arrow(data: dict) -> "pyarrow.Table":
    """Convert a GeoJSON FeatureCollection to a ``pyarrow.Table``.

    Feature properties become regular columns.  Each feature's geometry is
    serialized to a JSON string and stored in a ``__geometry__`` column so
    that ``mark_geoshape`` can read and project it without a sidecar field
    on ``ChartSpec``.
    """
    import json
    import pyarrow as pa

    features = data.get("features", [])
    if not features:
        return pa.table({"__geometry__": pa.array([], type=pa.string())})

    # Collect property columns and geometry strings.
    rows: list[dict] = []
    geom_strings: list[str] = []
    for feat in features:
        props = dict(feat.get("properties") or {})
        geom = feat.get("geometry")
        geom_strings.append(json.dumps(geom) if geom is not None else "null")
        rows.append(props)

    geom_col = pa.array(geom_strings, type=pa.string())
    if rows and all(not r for r in rows):
        # All features have empty properties — build a schema-less table with
        # the correct row count so the geometry column can be appended.
        tbl = pa.table({"__geometry__": geom_col})
        return tbl
    tbl = pa.Table.from_pylist(rows)
    return tbl.append_column("__geometry__", geom_col)
