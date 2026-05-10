"""Phase 9e — pairplot, heatmap, clustermap."""
from __future__ import annotations
from typing import Any

from ferrum import Chart, RepeatChart, Repeat


_VALID_PAIR_KINDS = {"scatter", "kde", "hist", "reg"}
_VALID_DIAG_KINDS = {"auto", "hist", "kde", None, "none"}


def pairplot(
    data: Any, *,
    vars: Any = None, x_vars: Any = None, y_vars: Any = None,
    hue: Any = None, kind: str = "scatter",
    diag_kind: str = "auto",
    markers: Any = None, height: float | None = None, aspect: float | None = None,
    corner: bool = False, dropna: bool = False, theme: Any = None,
    **encode_kwargs: Any,
) -> RepeatChart:
    """Pairwise-scatter grid — see ferrum-spec.md §3.14.

    Returns a RepeatChart whose template repeats over the cartesian product of
    ``row`` × ``column`` field lists (resolved from ``vars`` or
    ``x_vars``/``y_vars``).
    """
    if kind not in _VALID_PAIR_KINDS:
        raise ValueError(
            f"pairplot: kind must be one of {sorted(_VALID_PAIR_KINDS)}; got {kind!r}"
        )
    if diag_kind not in _VALID_DIAG_KINDS:
        raise ValueError(
            f"pairplot: diag_kind must be one of {sorted(k for k in _VALID_DIAG_KINDS if k)}|None; "
            f"got {diag_kind!r}"
        )

    # Resolve vars / x_vars / y_vars to (row, column) field lists.
    if vars is not None:
        if x_vars is not None or y_vars is not None:
            raise ValueError(
                "pairplot: cannot pass both vars= and x_vars=/y_vars="
            )
        rows = list(vars)
        cols = list(vars)
    elif x_vars is not None or y_vars is not None:
        if x_vars is None or y_vars is None:
            raise ValueError(
                "pairplot: x_vars and y_vars must be passed together"
            )
        rows = list(y_vars)
        cols = list(x_vars)
    else:
        # No vars specified — auto-detect numeric columns from data.
        from ferrum._coerce import to_arrow_table
        tbl = to_arrow_table(data)
        numeric_cols = []
        for name in tbl.column_names:
            t = tbl[name].type
            ts = str(t).lower()
            if any(s in ts for s in ("int", "float", "double", "decimal")):
                numeric_cols.append(name)
        rows = numeric_cols
        cols = numeric_cols

    # Build the off-diagonal template.
    if kind == "scatter":
        off = Chart(data).mark_point()
    elif kind == "kde":
        off = Chart(data).mark_density()
    elif kind == "hist":
        off = Chart(data).mark_histogram()
    elif kind == "reg":
        off = Chart(data).mark_smooth(method="lm")
    enc: dict = {"x": Repeat.column, "y": Repeat.row}
    if hue is not None:
        enc["color"] = hue
    enc.update(encode_kwargs)
    off = off.encode(**enc)

    # Build the diagonal template (only meaningful for symmetric vars).
    symmetric = (rows == cols)
    diagonal = None
    effective_diag_kind = diag_kind
    if effective_diag_kind == "auto":
        # auto = histogram by default (cheaper).
        effective_diag_kind = "hist"
    if effective_diag_kind not in (None, "none") and symmetric:
        diag_enc: dict = {"x": Repeat.column}
        if hue is not None:
            diag_enc["color"] = hue
        if effective_diag_kind == "hist":
            diagonal = Chart(data).mark_histogram().encode(**diag_enc)
        elif effective_diag_kind == "kde":
            diagonal = Chart(data).mark_density().encode(**diag_enc)

    if theme is not None:
        off = off.theme(theme)
        if diagonal is not None:
            diagonal = diagonal.theme(theme)

    rc = RepeatChart(
        off,
        row=rows,
        column=cols,
        diagonal=diagonal,
        corner=corner,
    )
    return rc


def heatmap(
    data: Any, *,
    annot: bool = True, fmt: str = ".2f",
    cmap: str = "blues",
    linewidths: float = 0.5, linecolor: str = "white",
    vmin: float | None = None, vmax: float | None = None,
    center: float | None = None, robust: bool = False,
    square: bool = False, mask: Any = None, theme: Any = None,
    **encode_kwargs: Any,
) -> Chart:
    """2D heatmap of a wide-format DataFrame — see ferrum-spec.md §3.14.

    Each row of ``data`` becomes a row of the heatmap; numeric columns become
    columns. The id column (first non-numeric column, if any) is used as the
    row label. The DataFrame is unpivoted to long form, then rendered with
    ``mark_rect`` + a continuous color scale.
    """
    from ferrum import Unpivot
    from ferrum._coerce import to_arrow_table

    tbl = to_arrow_table(data)
    # Identify id column (first non-numeric) and value columns (numeric).
    id_col: str | None = None
    value_cols: list[str] = []
    for name in tbl.column_names:
        t = str(tbl[name].type).lower()
        is_numeric = any(s in t for s in ("int", "float", "double", "decimal"))
        if is_numeric:
            value_cols.append(name)
        elif id_col is None:
            id_col = name
    if not value_cols:
        raise ValueError("heatmap: no numeric columns found in data")

    # If no id column found, synthesize a row index.
    import polars as pl
    if id_col is None:
        # Wrap data into a polars frame, add a row id.
        try:
            pdf = data if isinstance(data, pl.DataFrame) else pl.from_arrow(tbl)
        except Exception:
            pdf = pl.from_arrow(tbl)
        pdf = pdf.with_row_index("_row_id").with_columns(pl.col("_row_id").cast(pl.Utf8))
        data = pdf
        id_col = "_row_id"
        tbl = to_arrow_table(data)

    # robust=True: compute vmin/vmax from 2nd/98th percentiles in Python.
    if robust:
        import numpy as np
        all_vals = []
        for c in value_cols:
            all_vals.extend(tbl[c].to_numpy().tolist())
        arr = np.asarray(all_vals, dtype=float)
        arr = arr[~np.isnan(arr)]
        if vmin is None and arr.size:
            vmin = float(np.percentile(arr, 2.0))
        if vmax is None and arr.size:
            vmax = float(np.percentile(arr, 98.0))

    # Build the chart: Unpivot to long format, then mark_rect.
    unpivot = Unpivot(
        id_vars=[id_col], value_vars=value_cols,
        var_name="column", value_name="value",
    )
    rect_kwargs: dict = {}
    if linewidths > 0:
        rect_kwargs["stroke"] = linecolor
        rect_kwargs["stroke_width"] = linewidths

    enc: dict = {"x": "column", "y": id_col, "color": "value"}
    if "color" in encode_kwargs:
        enc["color"] = encode_kwargs.pop("color")
    enc.update(encode_kwargs)

    # Apply color scale (cmap / vmin / vmax / center).
    if vmin is not None or vmax is not None or center is not None or cmap is not None:
        from ferrum.encoding import Color
        scale_kwargs: dict = {"type": "linear"}
        if vmin is not None and vmax is not None:
            scale_kwargs["domain"] = [vmin, vmax]
        if cmap is not None:
            scale_kwargs["scheme"] = cmap
        # `center` is a diverging-scale hint; Rust scale resolution may use it
        # when present.
        if center is not None:
            scale_kwargs["domainMid"] = center
        if len(scale_kwargs) > 1:  # > just "type"
            enc["color"] = Color(enc["color"], scale=scale_kwargs)

    chart = Chart(data).transform(unpivot).mark_rect(**rect_kwargs).encode(**enc)

    # square=True → fix width=height proportions.
    if square:
        n_rows = tbl.num_rows
        n_cols = len(value_cols)
        side = 30 * max(n_rows, n_cols)
        chart = chart.properties(width=side, height=side)

    # annot=True: layer mark_text on top of rects, with the `text` channel
    # bound to the value column. Format spec lives on the text channel.
    if annot:
        from ferrum.encoding import Text
        text_layer = (
            Chart(data)
            .transform(unpivot)
            .mark_text()
            .encode(x="column", y=id_col, text=Text("value", format=fmt))
        )
        from ferrum.figure.regression import _merge_layers
        chart = _merge_layers(chart, text_layer)

    if theme is not None:
        chart = chart.theme(theme)
    return chart


def clustermap(
    data: Any, *,
    method: str = "ward", metric: str = "euclidean",
    cmap: str = "viridis",
    z_score: Any = None, standard_scale: Any = None,
    figsize: Any = None, dendrogram_ratio: float = 0.2,
    theme: Any = None,
    **encode_kwargs: Any,
) -> Any:
    """Clustered heatmap with row + column dendrograms — see ferrum-spec.md §3.14.

    Returns a ``ClusterMapChart`` composing:
      * a heatmap of the (reordered) wide-format DataFrame, and
      * two ``mark_segment`` dendrograms reading the per-linkage ``segments``
        named outputs.
    """
    from ferrum import (
        ClusterMapChart, Linkage, Reorder, Unpivot,
    )
    from ferrum._coerce import to_arrow_table

    tbl = to_arrow_table(data)
    # Identify id column + numeric columns (same approach as heatmap).
    id_col: str | None = None
    value_cols: list[str] = []
    for name in tbl.column_names:
        t = str(tbl[name].type).lower()
        is_numeric = any(s in t for s in ("int", "float", "double", "decimal"))
        if is_numeric:
            value_cols.append(name)
        elif id_col is None:
            id_col = name
    if not value_cols:
        raise ValueError("clustermap: no numeric columns found in data")

    # Linkage transforms (rows + columns) with explicit names so we can route
    # their `segments` named outputs to the dendrogram layers.
    row_link = Linkage(
        method=method, metric=metric, axis="rows",
        z_score=z_score, standard_scale=standard_scale,
        name="row_link",
    )
    col_link = Linkage(
        method=method, metric=metric, axis="columns",
        z_score=z_score, standard_scale=standard_scale,
        name="col_link",
    )

    # Center heatmap: reorder rows + columns then unpivot.
    unpivot = Unpivot(
        id_vars=[id_col] if id_col else [],
        value_vars=value_cols,
        var_name="column", value_name="value",
    )
    center = (
        Chart(data)
        .transform(row_link, col_link,
                   Reorder(by="row_link_order"),
                   Reorder(by="col_link_order"),
                   unpivot)
        .mark_rect()
        .encode(
            x="column",
            y=(id_col if id_col else "_row_id"),
            color="value",
        )
    )

    # Column dendrogram (top): reads col_link_segments and draws diagonal
    # segments. mark_segment requires x, y, x2, y2 in encoding.
    col_dendro_layer = {
        "mark": "segment",
        "encoding": {"x": "x", "y": "y", "x2": "x2", "y2": "y2"},
        "transforms": [],
        "mark_style": {},
        "data_source": "col_link_segments",
    }
    col_dendro = Chart(data).transform(col_link)
    col_dendro._mark = None
    col_dendro._layers = [col_dendro_layer]

    # Row dendrogram (left): reads row_link_segments, rotated via CoordFlip.
    from ferrum import CoordFlip
    row_dendro_layer = {
        "mark": "segment",
        "encoding": {"x": "x", "y": "y", "x2": "x2", "y2": "y2"},
        "transforms": [],
        "mark_style": {},
        "data_source": "row_link_segments",
    }
    row_dendro = Chart(data).transform(row_link)
    row_dendro._mark = None
    row_dendro._layers = [row_dendro_layer]
    row_dendro = row_dendro.coord(CoordFlip())

    if theme is not None:
        center = center.theme(theme)
        col_dendro = col_dendro.theme(theme)
        row_dendro = row_dendro.theme(theme)

    cm = ClusterMapChart(
        center,
        row_dendrogram=row_dendro,
        col_dendrogram=col_dendro,
        dendrogram_ratio=dendrogram_ratio,
    )
    return cm
