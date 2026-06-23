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

The SVG path (``_render.py::_render_inputs``) and the interactive path both
call ``normalize_for_rust(to_arrow_table(data))`` so the contract is applied
exactly once, by one function.
"""

from __future__ import annotations

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
        return col.cast(pa.timestamp("ms")), True

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
        array with ``ndim > 2``, or a list of non-dict items.
    ImportError
        If the input type requires ``narwhals`` and it is not installed.

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
                    casts.append(pl.col(c).cast(pl.Datetime("ms")))
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
                col = col.cast(pa.timestamp("ms"))
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
        return pa.Table.from_pydict(data)
    if isinstance(data, list):
        if not data:
            raise ValueError("Cannot construct Chart from empty list")
        if not isinstance(data[0], dict):
            raise TypeError(
                f"Chart(list) expects a list of dicts (one per row), got list of "
                f"{type(data[0]).__name__}"
            )
        return pa.Table.from_pylist(data)

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
