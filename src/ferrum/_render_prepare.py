"""Spec-prep transforms run before a chart's data reaches the Rust renderer.

This module holds the self-contained pre-render transforms that the render
mixin invokes but that are not part of render dispatch itself: the
``Axis(label_map=...)`` column-value remap engine and ``_classify_field``.
Most of these are pure helpers driven by ``_RenderMixin._resolve_chart_config`` /
``_render_inputs`` in ``ferrum._render``. Annotation category coordinates are not
resolved here: they cross to the renderer as ``{"category": label}`` and are placed
through the axis's own scale.
"""

from __future__ import annotations

from typing import Any, Iterator

import polars as pl

from ferrum.encoding.base import ChannelBase


def _label_map_entries(chart: Any) -> Iterator[tuple[str, str, dict[str, str]]]:
    """Yield ``(channel, column, label_map)`` for every ``Axis(label_map=...)`` on *chart*.

    Scans the chart's ``_encoding`` dict (single-mark path) and the ``_layers`` list
    (layered-mark path), x and y channels only (label remapping applies to positional axes).
    :func:`_collect_label_maps` folds the entries by column and :func:`_axis_label_maps` by
    channel.
    """
    from ferrum.axis import Axis as _Axis

    def _entries(enc_dict: dict) -> Iterator[tuple[str, str, dict[str, str]]]:
        for ch_name in ("x", "y"):
            ch = enc_dict.get(ch_name)
            if not isinstance(ch, ChannelBase):
                continue
            axis_kwarg = ch._kwargs.get("axis")
            if not isinstance(axis_kwarg, _Axis) or axis_kwarg.label_map is None:
                continue
            if ch.field is None:
                continue
            yield ch_name, ch.field, axis_kwarg.label_map

    yield from _entries(chart._encoding)
    if chart._layers:
        for layer in chart._layers:
            yield from _entries(layer.encoding)


def _collect_label_maps(chart: Any) -> dict[str, dict[str, str]]:
    """Collect Axis(label_map=...) entries from a chart's encoding.

    Returns a mapping of ``{column_name: {old_value: new_value, ...}}``; later
    layers override earlier ones for the same column.
    """
    result: dict[str, dict[str, str]] = {}
    for _channel, column, label_map in _label_map_entries(chart):
        result.setdefault(column, {}).update(label_map)
    return result


def _takes_label_map(df: pl.DataFrame, column: str) -> bool:
    """Whether ``label_map`` renames *column*: present, and text or categorical.

    The one gate the data rename reads, so the annotation rename (which follows the
    maps the data rename applied) cannot disagree with it.
    """
    return column in df.columns and df[column].dtype in (pl.Utf8, pl.String, pl.Categorical)


def _axis_label_maps(chart: Any, applied: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    """The label map each positional axis was renamed through: ``{"x": {...}, "y": {...}}``.

    *applied* is what :func:`_rename_through_label_maps` reports it renamed in the data, so a
    category written as a data value is renamed exactly when the rows that hold that value
    were.
    """
    axes: dict[str, dict[str, str]] = {"x": {}, "y": {}}
    for channel, column, label_map in _label_map_entries(chart):
        taken = applied.get(column, {})
        axes[channel].update({k: v for k, v in label_map.items() if k in taken})
    return axes


# The coordinate fields of an annotation spec dict, by the axis each is measured on. A span's
# ``start`` and ``end`` follow its own ``axis``. Mirrors ``coords_with_axes`` in the renderer's
# annotation module.
_X_COORD_FIELDS = frozenset({"x", "x1", "x2", "text_x"})
_Y_COORD_FIELDS = frozenset({"y", "y1", "y2", "text_y"})


def _rename_annotation_categories(
    annotations: list[dict[str, Any]],
    axis_maps: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    """Rename every ``{"category": value}`` through the label map of the axis it sits on.

    A value that is a key of the axis's map crosses as the mapped display label (the label
    the data was renamed to); any other value crosses unchanged. Returns new dicts.
    """
    if not any(axis_maps.values()):
        return annotations
    renamed = []
    for annotation in annotations:
        out = dict(annotation)
        for key, value in annotation.items():
            if not (isinstance(value, dict) and "category" in value):
                continue
            if key in _X_COORD_FIELDS:
                axis = "x"
            elif key in _Y_COORD_FIELDS:
                axis = "y"
            elif key in ("start", "end"):
                axis = annotation.get("axis", "")
            else:
                continue
            label_map = axis_maps.get(axis, {})
            category = value["category"]
            out[key] = {**value, "category": label_map.get(category, category)}
        renamed.append(out)
    return renamed


def _rename_through_label_maps(
    data: Any,
    label_maps: dict[str, dict[str, str]],
) -> tuple[Any, dict[str, dict[str, str]]]:
    """Apply label remapping to a polars DataFrame.

    For each ``(column_name, mapping)`` pair in *label_maps*, replaces
    string values in that column according to the mapping.  Values not
    present in the mapping are left unchanged.

    If the column does not exist in *data* or the data is not a polars
    DataFrame, the data is returned unchanged.

    Returns the data and, per renamed column, the entries of its map whose key
    some row of the column held (the renames that changed the data).
    """
    if not label_maps:
        return data, {}

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
            return data, {}
    else:
        df = data

    applied: dict[str, dict[str, str]] = {}
    for col, mapping in label_maps.items():
        if not _takes_label_map(df, col):
            continue
        held = {str(v) for v in df[col].drop_nulls().unique().to_list()}
        applied[col] = {k: v for k, v in mapping.items() if k in held}
        # replace(mapping) without a default leaves unmatched values unchanged.
        df = df.with_columns(df[col].replace(mapping).alias(col))

    return df, applied


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
