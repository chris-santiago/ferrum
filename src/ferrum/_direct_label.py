"""Private helper — emit text labels at the endpoint of each series.

Schwabish SB3 (2026-05-11) direct-label idiom for diagnostic charts that
otherwise rely on a legend (``learning_curve_chart``,
``validation_curve_chart``, and the gallery-autonomous fixer when it
decides to swap a legend for endpoint labels). Not a public mark — a
composite helper that mirrors the augmented-DataFrame pattern used by
:func:`ferrum.annotations._apply_metric_label` so labels share the
base chart's ``_data`` and overlay cleanly via ``+``.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from ferrum.annotations import _resolve_field
from ferrum.chart import Chart


def _direct_label_endpoint(
    chart: Chart,
    label_field: str,
    *,
    x_col: str | None = None,
    y_col: str | None = None,
    position: str = "end",
) -> Chart:
    """Append a text label at the endpoint of each series of ``chart``.

    Returns the chart augmented with one ``mark_text`` overlay row per
    unique value of ``label_field``. The base chart's ``_data`` is
    extended with a single ``_direct_label_text`` column (non-null only
    on the chosen-endpoint row per series, null elsewhere), and a new
    ``Chart`` over the same DataFrame is added via ``+`` so the result
    is a true overlay rather than an HConcat fallback.

    Parameters
    ----------
    chart : Chart
        Base chart whose data carries ``label_field`` plus the x and y
        coordinates of each series.
    label_field : str
        Column whose unique values name the series. Each unique value
        becomes one direct-label entry.
    x_col, y_col : str, optional
        Field names for the x and y coordinates. When omitted, falls
        back to the base chart's ``x`` / ``y`` encodings (resolved via
        :func:`ferrum.annotations._resolve_field`). The helper bails
        out (returns ``chart`` unchanged) when both fallbacks fail.
    position : {"end", "start"}, default "end"
        Endpoint at which to place each series' label.

    Returns
    -------
    Chart
        Augmented chart with the text overlay applied.
    """
    from ferrum._coerce import to_arrow_table

    x_col = x_col or _resolve_field(chart._encoding.get("x"))
    y_col = y_col or _resolve_field(chart._encoding.get("y"))
    tbl = to_arrow_table(chart._data)
    if (
        x_col is None
        or y_col is None
        or label_field not in tbl.column_names
        or x_col not in tbl.column_names
    ):
        return chart  # bail rather than crash — caller can fall back to legend

    series_arr = np.asarray(tbl.column(label_field).to_pylist())
    x_all = np.asarray(tbl.column(x_col).to_pylist(), dtype=float)
    labels_col: list[str | None] = [None] * len(series_arr)
    for series in sorted(set(series_arr.tolist()), key=str):
        mask = series_arr == series
        if not mask.any():
            continue
        masked_x = x_all[mask]
        idx_in_mask = int(
            np.argmax(masked_x) if position == "end" else np.argmin(masked_x)
        )
        global_idx = int(np.where(mask)[0][idx_in_mask])
        labels_col[global_idx] = str(series)

    base_pl = (
        chart._data
        if isinstance(chart._data, pl.DataFrame)
        else pl.from_arrow(tbl)
    )
    augmented = base_pl.with_columns(
        pl.Series("_direct_label_text", labels_col, dtype=pl.Utf8),
    )
    chart_aug = chart._clone()
    chart_aug._data = augmented
    # Anchor labels just inside the endpoint (align="right", dx=-4) so
    # they remain inside the plot extent even without extra horizontal
    # padding. Placing them outside-right would require axis padding the
    # default theme doesn't provide.
    annot_layer = (
        Chart(augmented)
        .mark_text(align="right", dx=-4)
        .encode(x=x_col, y=y_col, text="_direct_label_text")
    )
    return chart_aug + annot_layer
