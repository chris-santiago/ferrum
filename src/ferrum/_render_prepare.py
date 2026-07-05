"""Spec-prep transforms run before a chart's data reaches the Rust renderer.

This module holds the self-contained pre-render transforms that the render
mixin invokes but that are not part of render dispatch itself: the
``Axis(label_map=...)`` column-value remap engine and the ordinal-domain /
annotation category-coordinate resolver.  Most of these are pure helpers
driven by ``_RenderMixin._resolve_chart_config`` / ``_render_inputs`` in
``ferrum._render``; the per-channel field/column introspection helpers
(``_chart_bindings``, ``_column_minmax``, ``_column_unique``,
``_classify_field``) are also reused by ``ferrum.composition``'s cross-chart
union-domain computation (``compute_union_domain``/``inject_scale``), since
both are "which field is bound to this channel, and what does its data look
like" questions on the same chart shapes.
"""

from __future__ import annotations

import warnings
from typing import Any, Iterable

import polars as pl

from ferrum.encoding.base import ChannelBase


def _collect_label_maps(chart: Any) -> dict[str, dict[str, str]]:
    """Collect Axis(label_map=...) entries from a chart's encoding.

    Returns a mapping of ``{column_name: {old_value: new_value, ...}}``
    by scanning the chart's ``_encoding`` dict (single-mark path) and
    the ``_layers`` list (layered-mark path).

    Only x and y channels are checked because axis label remapping only
    applies to positional axes.
    """
    from ferrum.axis import Axis as _Axis

    result: dict[str, dict[str, str]] = {}

    def _check_enc(enc_dict: dict) -> None:
        for ch_name in ("x", "y"):
            ch = enc_dict.get(ch_name)
            if not isinstance(ch, ChannelBase):
                continue
            axis_kwarg = ch._kwargs.get("axis")
            if not isinstance(axis_kwarg, _Axis):
                continue
            if axis_kwarg.label_map is None:
                continue
            col = ch.field
            if col is None:
                continue
            if col in result:
                # Merge: later layers override earlier for same column.
                result[col].update(axis_kwarg.label_map)
            else:
                result[col] = dict(axis_kwarg.label_map)

    # Single-mark or top-level encoding
    _check_enc(chart._encoding)

    # Layered chart: each _Layer has its own encoding dict
    if chart._layers:
        for layer in chart._layers:
            _check_enc(layer.encoding)

    return result


def _apply_label_maps(
    data: Any,
    label_maps: dict[str, dict[str, str]],
) -> Any:
    """Apply label remapping to a polars DataFrame.

    For each ``(column_name, mapping)`` pair in *label_maps*, replaces
    string values in that column according to the mapping.  Values not
    present in the mapping are left unchanged.

    If the column does not exist in *data* or the data is not a polars
    DataFrame, the data is returned unchanged.
    """
    if not label_maps:
        return data

    if not isinstance(data, pl.DataFrame):
        # Non-polars data: coerce first, but we can't modify Arrow tables
        # in-place easily — convert to polars, remap, return polars.
        try:
            from ferrum._coerce import to_arrow_table as _to_arrow
            import pyarrow as pa

            arrow = _to_arrow(data)
            df = pl.from_arrow(arrow)
        except (ImportError, TypeError, ValueError):
            import warnings

            warnings.warn(
                "Axis label_map could not be applied (data coercion failed); labels unchanged.",
                stacklevel=2,
            )
            return data
    else:
        df = data

    for col, mapping in label_maps.items():
        if col not in df.columns:
            continue
        series = df[col]
        if series.dtype not in (pl.Utf8, pl.String, pl.Categorical):
            continue
        # replace(mapping) without a default leaves unmatched values unchanged.
        df = df.with_columns(series.replace(mapping).alias(col))

    return df


def _extract_field_name(ch: Any) -> str | None:
    """Return the field name bound to an encoding value, or ``None``.

    Encoding values are either bare strings (``encode(x="hp")``) or
    ``ChannelBase`` instances (``encode(x=X("hp", scale=...))``).
    """
    if isinstance(ch, str):
        return ch
    field = getattr(ch, "field", None)
    return field if isinstance(field, str) else None


def _chart_bindings(chart: Any, channel: str) -> Iterable[str | None]:
    """Yield every field name bound to *channel* across *chart*'s layers.

    Layered charts (``Chart + Chart`` composites) keep per-layer encoding
    dicts on ``_layers``; unlayered charts keep a single ``_encoding`` dict at
    the top level. Shared by :func:`_extract_ordinal_domain` (single-chart
    ordinal-domain lookup) and ``ferrum.composition``'s cross-chart
    union-domain computation.
    """
    layers = getattr(chart, "_layers", None)
    if layers:
        for layer in layers:
            yield _extract_field_name(layer.encoding.get(channel))
        return
    encoding = getattr(chart, "_encoding", {}) or {}
    yield _extract_field_name(encoding.get(channel))


def _column_minmax(data, field: str) -> tuple | None:
    """Return ``(min, max)`` of *field* in *data* as floats, or ``None``.

    Temporal columns (``Date``/``Datetime``/``Time``) are cast to their
    epoch-millisecond numeric representation before taking min/max --
    the same units ``TimeScale(domain=[...])`` expects on the wire and
    the same normalization ``ferrum._coerce.to_arrow_table`` applies to
    every temporal column before it reaches the Rust renderer (``Date``
    and non-ms ``Datetime`` both cast to ``Datetime("ms")``). Casting the
    *column* (not calling Python's ``datetime.timestamp()`` on the
    scalar ``col.min()``/``col.max()`` return value) avoids the stdlib's
    local-timezone assumption for naive datetimes; ``float(a_datetime)``
    also simply raises ``TypeError``, which is the bug this guards
    against (reachable via ``LayerChart``'s interactive
    ``compute_union_domain``/``inject_scale`` seam sharing a temporal
    x/y -- see ``composition.py::LayerChart._build_merged``).
    """
    try:
        col = data[field]
    except (KeyError, AttributeError, pl.exceptions.ColumnNotFoundError):
        return None

    dtype = col.dtype
    if dtype == pl.Date or isinstance(dtype, pl.Datetime):
        col = col.cast(pl.Datetime("ms")).cast(pl.Int64)
    elif dtype == pl.Time:
        # Time has no epoch; nanoseconds-since-midnight -> ms-since-midnight.
        col = col.cast(pl.Int64) / 1_000_000
    lo, hi = col.min(), col.max()
    if lo is None or hi is None:
        return None
    return (float(lo), float(hi))


def _column_unique(data, field: str) -> list:
    """Return the unique values of *field* in *data* as a list, preserving
    appearance order."""
    try:
        col = data[field]
    except (KeyError, AttributeError, pl.exceptions.ColumnNotFoundError):
        return []
    return col.unique(maintain_order=True).to_list()


def _classify_field(data, field: str) -> str | None:
    """Return ``"linear"``, ``"ordinal"``, or ``"time"`` for *field*'s dtype.

    Returns ``None`` for unknown dtypes — the caller skips sharing on that
    channel rather than guessing a scale type.
    """
    try:
        col = data[field]
    except (KeyError, AttributeError, pl.exceptions.ColumnNotFoundError):
        return None
    dtype = col.dtype
    if dtype.is_numeric():
        return "linear"
    if dtype in (pl.Datetime, pl.Date, pl.Time):
        return "time"
    if dtype in (pl.Utf8, pl.Categorical):
        return "ordinal"
    return None


def _build_ordinal_norm_map(domain: list) -> dict[str, float]:
    """Build a category → norm-center mapping for an ordinal domain list.

    Each category occupies an equal band of width ``1/n``.  The band center
    for item at index *i* (0-based) is ``(i + 0.5) / n``.

    Parameters
    ----------
    domain : list
        Ordered list of category values as they appear in the data.

    Returns
    -------
    dict[str, float]
        ``{str(category): norm_center}``

    Examples
    --------
    >>> _build_ordinal_norm_map(["a", "b", "c"])
    {'a': 0.1667, 'b': 0.5, 'c': 0.8333}
    """
    n = len(domain)
    if n == 0:
        return {}
    # NOTE: (i+0.5)/n is a band-center approximation that ignores the Rust band
    # scale's padding_inner/outer (default 0.1), so the annotation lands in the
    # correct band/order but is not pixel-aligned to the bar center.
    return {str(v): (i + 0.5) / n for i, v in enumerate(domain)}


def _extract_ordinal_domain(chart: Any, channel: str) -> list:
    """Return the ordered unique values for *channel* in *chart*, or an empty list.

    Checks the chart's ``_data`` attribute against the field name(s) bound to
    *channel* in ``_encoding`` (single-mark path) and ``_layers`` (layered path).
    Only processes columns whose dtype is ordinal-compatible (string or
    categorical).

    Parameters
    ----------
    chart : Chart
        Chart to inspect.
    channel : str
        Encoding channel, typically ``"x"`` or ``"y"``.

    Returns
    -------
    list
        Unique values in first-appearance order.
    """

    data = getattr(chart, "_data", None)
    if data is None or not isinstance(data, pl.DataFrame):
        return []

    for field in _chart_bindings(chart, channel):
        if field is None:
            continue
        try:
            col = data[field]
        except (KeyError, AttributeError, pl.exceptions.ColumnNotFoundError):
            continue
        if col.dtype not in (pl.Utf8, pl.String, pl.Categorical):
            continue
        # Preserve appearance order using unique(maintain_order=True)
        return col.unique(maintain_order=True).to_list()

    return []


def _resolve_category_coords_in_annotations(
    ann_list: list[dict],
    chart: Any,
) -> list[dict]:
    """Replace ``{"category": value}`` annotation coordinate dicts with ``{"norm": frac}``.

    The Rust annotation renderer's ``CoordValue`` enum only accepts
    ``Data(f64)``, ``Pixel { px }``, and ``Norm { norm }``.  Ordinal category
    coordinates (strings that are not ISO-8601 dates) are serialized by
    ``_coord()`` as ``{"category": value}`` so they can be post-processed here
    using the chart's ordinal domain before the annotation list is sent to Rust.

    Resolution rules:
    - Build the ordinal domain for the x and y axes from the chart's data.
    - For each coordinate dict that is ``{"category": v}``, look up ``v`` in
      the appropriate axis domain and replace the dict with
      ``{"norm": band_center}`` where ``band_center = (index + 0.5) / n``.
    - If the category is not found in the domain (unknown label, or wrong axis),
      fall back to ``{"norm": 0.5}`` (plot center) so the annotation renders
      without crashing.

    Parameters
    ----------
    ann_list : list[dict]
        Annotation spec dicts (as produced by ``to_dict_list()``).
    chart : Chart
        The chart whose ordinal domain is used for resolution.

    Returns
    -------
    list[dict]
        Annotation spec dicts with all ``{"category": ...}`` entries resolved.
    """
    # Lazily build the norm maps only when at least one {"category": ...} entry
    # exists to avoid scanning chart data on every render.
    _x_map: dict[str, float] | None = None
    _y_map: dict[str, float] | None = None

    def _x_norm_map() -> dict[str, float]:
        nonlocal _x_map
        if _x_map is None:
            _x_map = _build_ordinal_norm_map(_extract_ordinal_domain(chart, "x"))
        return _x_map

    def _y_norm_map() -> dict[str, float]:
        nonlocal _y_map
        if _y_map is None:
            _y_map = _build_ordinal_norm_map(_extract_ordinal_domain(chart, "y"))
        return _y_map

    # Coordinate keys that belong to the x-axis vs y-axis.
    _X_KEYS = frozenset({"x", "x1", "x2"})
    _Y_KEYS = frozenset({"y", "y1", "y2"})

    def _resolve_coord(key: str, val: Any) -> Any:
        """Replace a single coordinate value if it is a category dict."""
        if not isinstance(val, dict) or "category" not in val:
            return val
        cat = val["category"]
        if key in _X_KEYS:
            norm_map = _x_norm_map()
            axis_label = "x"
        elif key in _Y_KEYS:
            norm_map = _y_norm_map()
            axis_label = "y"
        else:
            # Unrecognised key — fall back to center.
            norm_map = {}
            axis_label = key
        cat_str = str(cat)
        if cat_str not in norm_map:
            domain_repr = sorted(norm_map.keys()) if norm_map else []
            warnings.warn(
                f"annotation category {cat_str!r} not found in {axis_label}-axis domain "
                f"{domain_repr}; placing at plot center",
                UserWarning,
                stacklevel=2,
            )
        norm_val = norm_map.get(cat_str, 0.5)
        return {"norm": norm_val}

    resolved = []
    for ann in ann_list:
        new_ann = dict(ann)
        for key, val in ann.items():
            new_ann[key] = _resolve_coord(key, val)
        resolved.append(new_ann)
    return resolved
