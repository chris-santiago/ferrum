"""Composite-mark desugar helpers (Phase 8b).

Each desugar_<name> returns a 5-tuple:
    ("__layered__", transforms: list, _ignored: None, _ignored: None, layers: list)

Where `layers` is a list of dicts each of shape:
    {"mark": str, "encoding": {channel: field}, "mark_kwargs": dict (opt), "data_source": str|None (opt)}
"""
from __future__ import annotations
from typing import Any, Optional

from ferrum import BoxStats, ErrorExtent, LetterValue, Outliers


def desugar_boxplot(
    x_field: str | None,
    y_field: str | None,
    *,
    extent: float | str = 1.5,
    outliers: bool = True,
    size: Optional[float] = None,
    color_field: Optional[str] = None,
    horizontal: bool = False,
) -> tuple:
    if x_field is None or y_field is None:
        raise ValueError("mark_boxplot() requires .encode(x=..., y=...)")
    cat = y_field if horizontal else x_field
    val = x_field if horizontal else y_field
    groupby = [cat] + ([color_field] if color_field else [])

    transforms = [
        BoxStats(field=val, groupby=groupby, whisker_extent=_extent_to_box(extent), name="box")
    ]
    if outliers:
        transforms.append(
            Outliers(field=val, groupby=groupby, extent=_extent_to_iqr_k(extent), name="outliers")
        )

    band = size or 0.6

    def enc(y_col, y2_col=None):
        if horizontal:
            d = {"x": y_col, "y": cat}
            if y2_col:
                d["x2"] = y2_col
        else:
            d = {"x": cat, "y": y_col}
            if y2_col:
                d["y2"] = y2_col
        return d

    layers = [
        {"mark": "rule", "encoding": enc("lower_whisker", "upper_whisker"), "data_source": "box"},
        {"mark": "rect", "encoding": enc("q1", "q3"), "mark_kwargs": {"width": band}, "data_source": "box"},
        {"mark": "tick", "encoding": enc("median"), "mark_kwargs": {"band_size": band}, "data_source": "box"},
    ]
    if outliers:
        layers.append({"mark": "point", "encoding": enc(val), "data_source": "outliers"})

    return ("__layered__", transforms, None, None, layers)


def desugar_errorbar(
    x_field: str | None,
    y_field: str | None,
    *,
    extent: str = "ci",
    ticks: bool = True,
) -> tuple:
    if x_field is None or y_field is None:
        raise ValueError("mark_errorbar() requires .encode(x=..., y=...)")
    transforms = [ErrorExtent(field=y_field, groupby=[x_field], method=extent, name="err")]
    layers = [
        {"mark": "rule", "encoding": {"x": x_field, "y": "lower", "y2": "upper"}, "data_source": "err"},
    ]
    if ticks:
        layers.extend([
            {"mark": "tick", "encoding": {"x": x_field, "y": "lower"},
             "mark_kwargs": {"band_size": 6}, "data_source": "err"},
            {"mark": "tick", "encoding": {"x": x_field, "y": "upper"},
             "mark_kwargs": {"band_size": 6}, "data_source": "err"},
        ])
    return ("__layered__", transforms, None, None, layers)


def desugar_errorband(
    x_field: str | None,
    y_field: str | None,
    *,
    extent: str = "ci",
    borders: bool = False,
) -> tuple:
    if x_field is None or y_field is None:
        raise ValueError("mark_errorband() requires .encode(x=..., y=...)")
    transforms = [ErrorExtent(field=y_field, groupby=[x_field], method=extent, name="err")]
    layers = [
        {"mark": "ribbon", "encoding": {"x": x_field, "y": "lower", "y2": "upper"},
         "mark_kwargs": {"opacity": 0.3}, "data_source": "err"},
    ]
    if borders:
        layers.extend([
            {"mark": "line", "encoding": {"x": x_field, "y": "lower"}, "data_source": "err"},
            {"mark": "line", "encoding": {"x": x_field, "y": "upper"}, "data_source": "err"},
        ])
    return ("__layered__", transforms, None, None, layers)


def desugar_ribbon(
    x_field: str | None,
    y_field: str | None,
    *,
    y2_field: str | None = None,
    opacity: float = 0.3,
    interpolate: str = "linear",
) -> tuple:
    """mark_ribbon directly emits a ribbon layer; no transform.

    y2_field is read from the chart's encoding state (passed in by the
    Chart._resolve_pending special case for kind == "ribbon") since
    ribbon is invoked AFTER .encode(x=..., y=..., y2=...).
    """
    if x_field is None or y_field is None:
        raise ValueError("mark_ribbon() requires .encode(x=..., y=...)")
    if y2_field is None:
        raise ValueError("mark_ribbon() requires y2 in encoding (e.g. .encode(x=, y=lower, y2=upper))")
    layers = [
        {"mark": "ribbon", "encoding": {"x": x_field, "y": y_field, "y2": y2_field},
         "mark_kwargs": {"opacity": opacity}, "data_source": None},
    ]
    return ("__layered__", [], None, None, layers)


def desugar_boxen(
    x_field: str | None,
    y_field: str | None,
    *,
    k_depth: str = "proportion",
    k_proportion: float = 0.007,
    outlier_threshold: float = 1.5,
    palette=None,
    horizontal: bool = False,
    color_field: str | None = None,
) -> tuple:
    """Letter-value (boxen) composite mark.

    Desugars into a `LetterValue` transform plus N nested rect bands (one per
    depth, opacity ramping outer→inner), a median rule, and a point layer for
    outliers. Each rect layer reads from a dedicated `lv_depth_K` named output
    (no overlapping rows).

    LetterValue secondary-output schemas:
        ``lv_depth_K``:  ``[group, lower, upper, level]``
        ``lv_outliers``: ``[group, value, is_outlier]``

    Layer encodings therefore use ``"group"`` / ``"lower"`` / ``"upper"`` /
    ``"value"`` rather than the original chart-level column names — but the
    chart-level x/y encoding still references the user's columns, so axis
    scales resolve naturally (LetterValue copies the groupby values verbatim
    into ``group``, and quantile outputs lie within the original value range).
    """
    if x_field is None or y_field is None:
        raise ValueError("mark_boxen() requires .encode(x=..., y=...)")
    cat = y_field if horizontal else x_field
    val = x_field if horizontal else y_field
    group = color_field if color_field else cat

    transforms = [
        LetterValue(
            value=val,
            group=group,
            k_depth=k_depth,
            k_proportion=k_proportion,
            outlier_threshold=outlier_threshold,
            name="lv",
        ),
    ]

    # Per-depth named outputs (lv_depth_1 … lv_depth_6) let each rect layer
    # read its own slice — no overlap. K_MAX = 6 visible bands; for data with
    # fewer effective depths, unused outputs are zero-row batches and render
    # nothing.
    K_MAX = 6
    layers: list[dict] = []
    for k in range(1, K_MAX + 1):
        opacity = 0.85 - (0.55 * (k - 1) / max(K_MAX - 1, 1))
        enc = (
            {"x": "lower", "x2": "upper", "y": "group"}
            if horizontal
            else {"x": "group", "y": "lower", "y2": "upper"}
        )
        layers.append({
            "mark": "rect",
            "encoding": enc,
            "mark_kwargs": {"opacity": opacity},
            "data_source": f"lv_depth_{k}",
        })

    # Median rule: at depth=1, ``lower == upper == median``.
    layers.append({
        "mark": "rule",
        "encoding": (
            {"x": "lower", "y": "group"} if horizontal else {"x": "group", "y": "lower"}
        ),
        "data_source": "lv_depth_1",
    })

    # Outliers: point layer reading from the dedicated outliers output. Schema:
    # [group, value, is_outlier].
    layers.append({
        "mark": "point",
        "encoding": (
            {"x": "value", "y": "group"} if horizontal else {"x": "group", "y": "value"}
        ),
        "data_source": "lv_outliers",
    })

    return ("__layered__", transforms, None, None, layers)


def _extent_to_box(extent):
    return "min-max" if extent == "min-max" else float(extent)


def _extent_to_iqr_k(extent):
    return 1.5 if extent == "min-max" else float(extent)
