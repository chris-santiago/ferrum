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

import polars as pl

from typing import TYPE_CHECKING

from ferrum.annotations import _resolve_field

if TYPE_CHECKING:
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

    base_pl = chart._data if isinstance(chart._data, pl.DataFrame) else pl.from_arrow(tbl)
    df = base_pl.with_row_index("_idx")
    has_y = y_col in df.columns
    n = df.height
    labels_col: list[str | None] = [None] * n
    label_y_col: list[float | None] = [None] * n

    series_list = sorted(df[label_field].unique().to_list(), key=str)
    series_endpoints: list[tuple[str, int, float, float]] = []
    descending = position == "end"
    for series in series_list:
        group = df.filter(pl.col(label_field) == series)
        if group.is_empty():
            continue
        row = group.sort(x_col, descending=descending).row(0, named=True)
        global_idx = int(row["_idx"])
        ep_x = float(row[x_col])
        ep_y = float(row[y_col]) if has_y else 0.0
        series_endpoints.append((str(series), global_idx, ep_x, ep_y))

    y_range = (
        float(df[y_col].max() - df[y_col].min()) if (has_y and n > 0) else 1.0
    )
    stagger_step = max(y_range * 0.05, 1e-9)
    stagger_threshold = stagger_step * 0.8

    used_y: list[float] = []
    for series, global_idx, ep_x, ep_y in sorted(
        series_endpoints,
        key=lambda t: -t[3],
    ):
        target_y = ep_y
        while any(abs(target_y - prev_y) < stagger_threshold for prev_y in used_y):
            target_y -= stagger_step
        used_y.append(target_y)
        labels_col[global_idx] = series
        label_y_col[global_idx] = target_y

    augmented = base_pl.with_columns(
        pl.Series("_direct_label_text", labels_col, dtype=pl.Utf8),
        pl.Series("_direct_label_y", label_y_col, dtype=pl.Float64),
    )
    from ferrum._layer import _Layer

    chart_aug = chart._clone()
    chart_aug._data = augmented
    return chart_aug.layer(
        _Layer(
            mark="text",
            encoding={"x": x_col, "y": "_direct_label_y", "text": "_direct_label_text"},
            mark_kwargs={"align": "right", "dx": -4},
        )
    )
