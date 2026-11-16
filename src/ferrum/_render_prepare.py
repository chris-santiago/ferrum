"""Spec-prep transforms run before a chart's data reaches the Rust renderer.

This module holds the self-contained pre-render transforms that the render
mixin invokes but that are not part of render dispatch itself: ``_classify_field``
and the check of the columns a category binding reads (a Duration refused,
colliding date-times warned). Most of these are pure helpers driven by
``_RenderMixin._resolve_chart_config`` / ``_render_inputs`` in ``ferrum._render``.
Annotation category coordinates are not resolved here: they cross to the renderer
as ``{"category": label}`` and are placed through the axis's own scale. An
``Axis(label_map=...)`` also crosses as is: the renderer relabels tick text.
"""

from __future__ import annotations

import json
import warnings
from typing import Any

import polars as pl


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


def _watched_columns(data: pl.DataFrame) -> dict[str, pl.DataType]:
    """The columns whose reading as categories changes what is drawn: a Duration, and a Datetime
    finer than a millisecond (it reaches the engine at millisecond precision)."""
    return {
        name: dtype
        for name, dtype in data.schema.items()
        if isinstance(dtype, pl.Duration)
        or (isinstance(dtype, pl.Datetime) and dtype.time_unit in ("us", "ns"))
    }


def _branch_renames(spec: Any) -> dict[str, dict[str, str]]:
    """The column renames of each operand a ``+`` composition branches the chart's frame into,
    keyed by the branch name a layer's ``data_source`` names."""
    found: dict[str, dict[str, str]] = {}

    def walk(transforms: list[dict[str, Any]]) -> None:
        for transform in transforms:
            if transform.get("type") == "branch":
                found[transform["name"]] = transform.get("frame", {}).get("renames", {})
                walk(transform.get("transforms", []))

    walk(json.loads(spec.to_json()).get("transforms", []))
    return found


def _check_category_columns(spec: Any, data: Any) -> None:
    """Refuse a Duration, and warn of colliding date-times, on the bindings the engine reads as
    categories.

    The engine answers which bindings read their column as categories
    (:func:`ferrum._core.category_read_fields`, from the built *spec* and the schema *data*
    reaches it with); this check never restates that rule. A Duration crosses to the engine
    as a timestamp, so it would be labelled with 1970 epoch dates: it is refused here, before it
    crosses, naming the field the caller wrote. Date-times finer than a millisecond reach the
    engine at millisecond precision, so distinct values within one millisecond draw as one
    category: one warning per column names the counts.

    Parameters
    ----------
    spec :
        The chart's built ``ChartSpec``.
    data :
        The chart's frame before it is converted for the engine (anything else is left alone).

    Raises
    ------
    ValueError
        When a binding that reads categories names a Duration column.
    """
    if not isinstance(data, pl.DataFrame):
        return
    watched = _watched_columns(data)
    if not watched:
        return
    from ferrum._coerce import normalize_for_rust
    from ferrum._core import category_read_fields
    from ferrum._warn import user_stacklevel

    reads = category_read_fields(spec, normalize_for_rust(data.head(0).to_arrow()).schema)
    renames = _branch_renames(spec) if any(source for source, _, _ in reads) else {}
    colliding: dict[str, str] = {}
    for source, channel, field in reads:
        column = renames.get(source, {}).get(field, field)
        dtype = watched.get(column)
        if isinstance(dtype, pl.Duration):
            where = (
                "it is a facet field"
                if channel == "facet"
                else f"the {channel} channel reads categories"
            )
            raise ValueError(
                f"column '{field}' is a Duration, which cannot be read as categories ({where}); "
                f"convert it to text with pl.col('{field}').dt.to_string('polars') for one "
                "category per value"
            )
        if dtype is not None:
            colliding.setdefault(column, field)
    for column, field in colliding.items():
        values = data[column].drop_nulls()
        before = values.n_unique()
        after = values.cast(pl.Datetime("ms", data.schema[column].time_zone)).n_unique()
        if after < before:
            warnings.warn(
                f"column '{field}' holds {before} distinct date-times, which draw as {after} "
                "categories: date-times reach the renderer at millisecond precision, so values "
                "within one millisecond share a category",
                UserWarning,
                stacklevel=user_stacklevel(),
            )
