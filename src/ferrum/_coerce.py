"""Data ingestion: normalize any supported input to a pyarrow.Table.

Supports (per spec §3.18):
- polars.DataFrame, polars.LazyFrame
- pyarrow.Table, pyarrow.RecordBatch
- pandas + modin + cuDF + dask + ibis (via narwhals)
- dict[str, list], list[dict]
- numpy.ndarray (2D, auto-named "col_0", "col_1", ...)

Raises TypeError for unsupported types or numpy 1D without column names.
"""

from __future__ import annotations

from typing import Any


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
        raise ValueError("Chart(data=None) requires per-layer data — not yet supported in Phase 8a")

    # Fast path: polars (zero-copy via Arrow C Data Interface)
    try:
        import polars as pl

        if isinstance(data, pl.DataFrame):
            return data.to_arrow()
        if isinstance(data, pl.LazyFrame):
            return data.collect().to_arrow()
    except ImportError:
        pass

    # Fast path: pyarrow native
    if isinstance(data, pa.Table):
        return data
    if isinstance(data, pa.RecordBatch):
        return pa.Table.from_batches([data])

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
        return nw_df.to_arrow()
    except (TypeError, NotImplementedError) as e:
        raise TypeError(
            f"Unsupported data type: {type(data).__name__}. "
            f"Supported: polars, pyarrow, pandas, modin, cuDF, dask, ibis, dict, list, numpy 2D. "
            f"Underlying error: {e}"
        ) from e


# ── GeoJSON FeatureCollection support ────────────────────────────────────────

def _is_geojson_feature_collection(data: dict) -> bool:
    """Return True if *data* looks like a GeoJSON FeatureCollection."""
    return (
        data.get("type") == "FeatureCollection"
        and isinstance(data.get("features"), list)
    )


_GEOJSON_GEOMETRY_TYPES = frozenset({
    "Point", "MultiPoint", "LineString", "MultiLineString",
    "Polygon", "MultiPolygon", "GeometryCollection",
})


def _is_geojson_geometry_root(data: dict) -> bool:
    """Return True if *data* is a bare GeoJSON Geometry or GeometryCollection."""
    return data.get("type") in _GEOJSON_GEOMETRY_TYPES


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
