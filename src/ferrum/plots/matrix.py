"""Matrix-of-plots convenience functions (pairplot, heatmap, clustermap, jointplot).

Public API
----------
pairplot, heatmap, clustermap, jointplot.

``pairplot``, ``heatmap``, and ``clustermap`` originated in
``ferrum.figure.matrix``; ``jointplot`` originated in
``ferrum.figure.joint``.  Both are merged here as they share the
"matrix / multi-panel" domain concern.

Internal helper
---------------
_ensure_id_column — materialises a synthetic row-identity column when
the input has no non-numeric id column (used by ``heatmap`` and
``clustermap``).
"""

from __future__ import annotations

from typing import Any

from ferrum import Bin2D, Chart, ClusterMapChart, JointChart, RepeatChart, Repeat
from ferrum._overrides import _apply_overrides
from ferrum.plots._helpers import _finalize_chart
from ferrum.plots.regression import _merge_layers


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_PAIR_KINDS = {"scatter", "kde", "hist", "reg"}
_VALID_DIAG_KINDS = {"auto", "hist", "kde", None, "none"}
_VALID_CENTER_KINDS = {"scatter", "kde", "hist", "hex", "reg"}
_VALID_MARGINAL_KINDS = {"hist", "kde", "rug", "box"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_id_column(data: Any, tbl: Any, id_col: str | None) -> tuple[Any, Any, str]:
    """Return ``(data, tbl, id_col)`` with a guaranteed id column.

    If the input has no non-numeric id column, materialise ``_row_id`` via
    ``pl.DataFrame.with_row_index`` (cast to Utf8) and refresh the Arrow table
    so downstream transforms see the synthetic column. When ``id_col`` is
    already set, return inputs unchanged.

    Used by ``heatmap`` and ``clustermap`` — both accept wide-format
    DataFrames where the row identity may be implicit.
    """
    if id_col is not None:
        return data, tbl, id_col

    import polars as pl
    from ferrum._coerce import to_arrow_table

    try:
        pdf = data if isinstance(data, pl.DataFrame) else pl.from_arrow(tbl)
    except Exception:
        pdf = pl.from_arrow(tbl)
    pdf = pdf.with_row_index("_row_id").with_columns(pl.col("_row_id").cast(pl.Utf8))
    return pdf, to_arrow_table(pdf), "_row_id"


# ---------------------------------------------------------------------------
# pairplot
# ---------------------------------------------------------------------------


def pairplot(
    data: Any,
    *,
    vars: Any = None,
    x_vars: Any = None,
    y_vars: Any = None,
    hue: Any = None,
    kind: str = "scatter",
    diag_kind: str = "auto",
    markers: Any = None,
    height: float | None = None,
    aspect: float | None = None,
    corner: bool = False,
    dropna: bool = False,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
    **encode_kwargs: Any,
) -> RepeatChart:
    """Pairwise-scatter grid (scatterplot matrix).

    Returns a ``RepeatChart`` whose template repeats over the cartesian
    product of ``row`` x ``column`` field lists, resolved from ``vars`` or
    ``x_vars``/``y_vars``.  When neither is supplied, all numeric columns
    in ``data`` are used.

    Parameters
    ----------
    data : DataFrame-like
        Input data accepted by ``Chart(data)``.
    vars : sequence of str, optional
        Column names to plot on both axes (symmetric grid).  Cannot be
        combined with ``x_vars``/``y_vars``.
    x_vars : sequence of str, optional
        Column names for the columns of the grid.  Must be paired with
        ``y_vars``.
    y_vars : sequence of str, optional
        Column names for the rows of the grid.  Must be paired with
        ``x_vars``.
    hue : str or encoding, optional
        Column name to map to color across all panels.
    kind : {"scatter", "kde", "hist", "reg"}, default "scatter"
        Mark for the off-diagonal cells.  ``"scatter"`` calls
        ``mark_point``; ``"kde"`` calls ``mark_density``; ``"hist"``
        calls ``mark_histogram``; ``"reg"`` calls
        ``mark_smooth(method="lm")``.
    diag_kind : {"auto", "hist", "kde", None, "none"}, default "auto"
        Mark for the diagonal cells (only when ``vars`` is symmetric).
        ``"auto"`` resolves to ``"kde"`` (KDE is smoother and more informative
        than histograms for overlapping distributions).  Pass ``None`` or
        ``"none"`` to suppress diagonal marks.
    markers : str or list of str, optional
        Point-marker shape(s).  A single string (e.g. ``"square"``) sets the
        shape on every scatter panel.  A list maps each hue level to a shape
        via the ``Shape`` encoding.  Only applies when ``kind="scatter"``.
    height : float or None, optional
        Height of each individual panel in pixels.  When set alongside
        ``aspect``, each panel's width is ``height * aspect``.
    aspect : float or None, optional
        Aspect ratio per panel (width = ``height * aspect``).  Defaults to
        ``1.0`` when ``height`` is set and ``aspect`` is omitted.
    corner : bool, default False
        Render only the lower-triangle panels.
    dropna : bool, default False
        When ``True``, drop rows with any null value in the selected
        variable columns before building the pairplot.
    theme : Theme, optional
        Visual theme applied to all panels.
    **encode_kwargs
        Additional keyword arguments forwarded to ``Chart.encode()`` on
        every panel template.

    Returns
    -------
    RepeatChart
        Grid of panels sharing the same off-diagonal and (optionally)
        diagonal chart templates.

    Raises
    ------
    ValueError
        If ``kind`` or ``diag_kind`` is not one of the supported values.
    ValueError
        If both ``vars`` and ``x_vars``/``y_vars`` are supplied.
    ValueError
        If only one of ``x_vars`` / ``y_vars`` is supplied.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.pairplot(df)

    Specific variables with KDE off-diagonal and per-species color:

    >>> fm.pairplot(
    ...     df, vars=["sepal_length", "sepal_width", "petal_length"],
    ...     kind="kde", hue="species",
    ... )
    """
    if kind not in _VALID_PAIR_KINDS:
        raise ValueError(f"pairplot: kind must be one of {sorted(_VALID_PAIR_KINDS)}; got {kind!r}")
    if diag_kind not in _VALID_DIAG_KINDS:
        raise ValueError(
            f"pairplot: diag_kind must be one of {sorted(k for k in _VALID_DIAG_KINDS if k)}|None; "
            f"got {diag_kind!r}"
        )

    # Resolve vars / x_vars / y_vars to (row, column) field lists.
    if vars is not None:
        if x_vars is not None or y_vars is not None:
            raise ValueError("pairplot: cannot pass both vars= and x_vars=/y_vars=")
        rows = list(vars)
        cols = list(vars)
    elif x_vars is not None or y_vars is not None:
        if x_vars is None or y_vars is None:
            raise ValueError("pairplot: x_vars and y_vars must be passed together")
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

    # dropna: drop rows with any null in the selected variable columns.
    if dropna:
        import polars as pl

        data = pl.DataFrame(data) if not isinstance(data, pl.DataFrame) else data
        all_vars = list(dict.fromkeys(rows + cols))  # deduplicate, preserve order
        data = data.drop_nulls(subset=all_vars)

    # Build the off-diagonal template.
    mark_kwargs: dict = {}
    if markers is not None and kind == "scatter":
        if isinstance(markers, str):
            mark_kwargs["shape"] = markers

    if kind == "scatter":
        off = Chart(data).mark_point(**mark_kwargs)
    elif kind == "kde":
        off = Chart(data).mark_density()
    elif kind == "hist":
        off = Chart(data).mark_histogram()
    elif kind == "reg":
        off = Chart(data).mark_smooth(method="lm")
    enc: dict = {"x": Repeat.column, "y": Repeat.row}
    if hue is not None:
        enc["color"] = hue
    # markers as list: map each hue level to a shape via Shape encoding.
    if markers is not None and isinstance(markers, list) and hue is not None and kind == "scatter":
        from ferrum.encoding import Shape as _Shape

        enc["shape"] = _Shape(hue, scale={"range": markers})
    enc.update(encode_kwargs)
    off = off.encode(**enc)

    # Per-panel sizing via height / aspect.
    if height is not None:
        panel_aspect = aspect if aspect is not None else 1.0
        off = off.properties(height=height, width=height * panel_aspect)

    # Build the diagonal template (only meaningful for symmetric vars).
    symmetric = rows == cols
    diagonal = None
    effective_diag_kind = diag_kind
    if effective_diag_kind == "auto":
        # auto = kde by default (smoother, more informative for overlapping distributions).
        effective_diag_kind = "kde"
    if effective_diag_kind not in (None, "none") and symmetric:
        diag_enc: dict = {"x": Repeat.column}
        if hue is not None:
            diag_enc["color"] = hue
        if effective_diag_kind == "hist":
            # When hue is set, thread groupby so Bin emits per-(bin, hue) rows
            # preserving the hue column for color encoding.
            hist_kwargs: dict = {}
            if hue is not None:
                hist_kwargs["groupby"] = hue
            diagonal = Chart(data).mark_histogram(**hist_kwargs).encode(**diag_enc)
        elif effective_diag_kind == "kde":
            # When hue is set, thread groupby so Kde emits per-group density
            # curves preserving the hue column for color encoding.
            kde_kwargs: dict = {}
            if hue is not None:
                kde_kwargs["groupby"] = hue
            diagonal = Chart(data).mark_density(**kde_kwargs).encode(**diag_enc)

        # Apply same panel sizing to diagonal template.
        if height is not None:
            panel_aspect = aspect if aspect is not None else 1.0
            diagonal = diagonal.properties(height=height, width=height * panel_aspect)

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
    rc = _finalize_chart(rc, mark=mark, encode=encode, properties=properties, layers=layers, theme=None)
    return rc


# ---------------------------------------------------------------------------
# heatmap
# ---------------------------------------------------------------------------


def heatmap(
    data: Any,
    *,
    annot: bool = True,
    fmt: str = ".2f",
    cmap: str | None = None,
    linewidths: float = 0.5,
    linecolor: str = "white",
    vmin: float | None = None,
    vmax: float | None = None,
    center: float | None = None,
    robust: bool = False,
    square: bool = False,
    mask: Any = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
    **encode_kwargs: Any,
) -> Chart:
    """2-D heatmap of a wide-format DataFrame.

    Each row of ``data`` becomes a row of the heatmap; each numeric column
    becomes a column.  The first non-numeric column (if any) is used as the
    row-label axis; rows without a label column get a synthetic integer index.
    The DataFrame is unpivoted to long form via ``Unpivot``, then rendered
    with ``mark_rect`` and an optional annotation layer (``mark_text``).

    Parameters
    ----------
    data : DataFrame-like
        Wide-format input.  Numeric columns become the heatmap cells; the
        first non-numeric column (if present) labels the rows.
    annot : bool, default True
        Overlay cell values as text using ``mark_text``.
    fmt : str, default ".2f"
        Python format specifier applied to cell values in the annotation layer.
    cmap : str or None, optional
        Color scheme name (e.g. ``"blues"``, ``"viridis"``, ``"rdbu"``).
        ``None`` (default) defers to the theme's sequential scheme.
    linewidths : float, default 0.5
        Width of the cell border stroke in pixels.  ``0`` disables borders.
    linecolor : str, default "white"
        Color of the cell border stroke.
    vmin : float or None, optional
        Minimum of the color scale domain.  Overrides ``robust`` when set.
    vmax : float or None, optional
        Maximum of the color scale domain.  Overrides ``robust`` when set.
    center : float or None, optional
        Value to center a diverging color scale on (maps to
        ``scale.domainMid``).
    robust : bool, default False
        When ``True`` and ``vmin``/``vmax`` are unset, clip the color scale
        to the 2nd and 98th percentiles of the data.
    square : bool, default False
        Force equal width and height per cell so the heatmap is square.
    mask : array-like, "upper", "lower", or None, optional
        Cell-masking control.  When ``"upper"``, only the upper triangle
        (including the diagonal) is shown.  When ``"lower"``, only the
        lower triangle (including the diagonal) is shown.  When an
        array-like boolean matrix (same shape as the numeric portion of
        ``data``), cells where the mask is ``True`` are hidden.
    theme : Theme, optional
        Visual theme applied via ``Chart.theme()``.
    **encode_kwargs
        Additional keyword arguments forwarded to ``Chart.encode()``.

    Returns
    -------
    Chart
        Possibly layered chart (``mark_rect`` + ``mark_text`` when
        ``annot=True``).

    Raises
    ------
    ValueError
        If ``data`` has no numeric columns.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.heatmap(corr_df, cmap="rdbu", center=0)

    Suppress annotations and use a custom domain:

    >>> fm.heatmap(wide_df, annot=False, vmin=0, vmax=1, cmap="greens")
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

    # If no id column found, synthesize a `_row_id` column so downstream
    # transforms have a row identity to unpivot against.
    data, tbl, id_col = _ensure_id_column(data, tbl, id_col)

    # robust=True: compute vmin/vmax from 2nd/98th percentiles in Python.
    if robust:
        import numpy as np

        all_vals = []
        for c in value_cols:
            all_vals.extend(tbl[c].to_pylist())
        arr = np.asarray(all_vals, dtype=float)
        arr = arr[~np.isnan(arr)]
        if vmin is None and arr.size:
            vmin = float(np.percentile(arr, 2.0))
        if vmax is None and arr.size:
            vmax = float(np.percentile(arr, 98.0))

    # When mask is set, we melt the data in Python and filter out masked
    # cells before chart construction (no Rust Unpivot needed).
    # Otherwise, use the Rust Unpivot transform for the standard path.
    melted_data = None
    if mask is not None:
        import polars as pl

        pdf = data if isinstance(data, pl.DataFrame) else pl.from_arrow(tbl)
        # Build a long-form DataFrame: (id_col, "column", "value").
        n_rows = pdf.height
        n_cols = len(value_cols)

        if isinstance(mask, str):
            if mask == "upper":
                # Keep upper triangle + diagonal: row_idx <= col_idx.
                keep = [
                    (r, c)
                    for r in range(n_rows)
                    for c in range(n_cols)
                    if r <= c
                ]
            elif mask == "lower":
                # Keep lower triangle + diagonal: row_idx >= col_idx.
                keep = [
                    (r, c)
                    for r in range(n_rows)
                    for c in range(n_cols)
                    if r >= c
                ]
            else:
                raise ValueError(
                    f"heatmap: mask must be 'upper', 'lower', or array-like; got {mask!r}"
                )
        else:
            # Array-like boolean mask: True = hide, False = show.
            import numpy as np

            mask_arr = np.asarray(mask, dtype=bool)
            if mask_arr.shape != (n_rows, n_cols):
                raise ValueError(
                    f"heatmap: mask shape {mask_arr.shape} does not match data "
                    f"shape ({n_rows}, {n_cols})"
                )
            keep = [
                (r, c)
                for r in range(n_rows)
                for c in range(n_cols)
                if not mask_arr[r, c]
            ]

        # Build the filtered long-form DataFrame.
        row_labels = pdf[id_col].to_list()
        rows_long: list[dict] = []
        for r, c in keep:
            rows_long.append({
                id_col: row_labels[r],
                "column": value_cols[c],
                "value": pdf[value_cols[c]][r],
            })
        melted_data = pl.DataFrame(rows_long)

    # Build the chart: Unpivot to long format, then mark_rect.
    unpivot = Unpivot(
        id_vars=[id_col],
        value_vars=value_cols,
        var_name="column",
        value_name="value",
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

    if melted_data is not None:
        # Data already melted and filtered — no Unpivot transform needed.
        chart = Chart(melted_data).mark_rect(**rect_kwargs).encode(**enc)
    else:
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

        if melted_data is not None:
            text_layer = (
                Chart(melted_data)
                .mark_text()
                .encode(x="column", y=id_col, text=Text("value", format=fmt))
            )
        else:
            text_layer = (
                Chart(data)
                .transform(unpivot)
                .mark_text()
                .encode(x="column", y=id_col, text=Text("value", format=fmt))
            )
        chart = _merge_layers(chart, text_layer, scatter_name="cells", fit_name="label")

    return _finalize_chart(chart, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme)


# ---------------------------------------------------------------------------
# clustermap
# ---------------------------------------------------------------------------


def clustermap(
    data: Any,
    *,
    method: str = "ward",
    metric: str = "euclidean",
    cmap: str | None = None,
    z_score: Any = None,
    standard_scale: Any = None,
    figsize: Any = None,
    dendrogram_ratio: float = 0.2,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
    **encode_kwargs: Any,
) -> "ClusterMapChart":
    """Clustered heatmap with row and column dendrograms.

    Returns a ``ClusterMapChart`` composed of:

    * a center heatmap of the hierarchically-reordered wide-format
      DataFrame (``Linkage`` + ``Reorder`` + ``Unpivot`` + ``mark_rect``),
    * a column dendrogram (top, ``mark_segment``, reading
      ``col_link_segments``), and
    * a row dendrogram (left, ``mark_segment`` + ``CoordFlip``, reading
      ``row_link_segments``).

    Parameters
    ----------
    data : DataFrame-like
        Wide-format input.  Same layout requirements as ``heatmap``.
    method : str, default "ward"
        Linkage algorithm forwarded to ``Linkage`` (e.g. ``"ward"``,
        ``"average"``, ``"complete"``, ``"single"``).
    metric : str, default "euclidean"
        Distance metric forwarded to ``Linkage`` (e.g. ``"euclidean"``,
        ``"cosine"``, ``"correlation"``).
    cmap : str or None, optional
        Color scheme name for the center heatmap (e.g. ``"magma"``,
        ``"viridis"``, ``"rdbu"``).  Forwarded to the ``Color`` encoding's
        ``scheme`` scale option.  ``"magma"`` is preferred for dense heatmaps
        due to its perceptual uniformity across a wide luminance range.
    z_score : {0, 1, None}, optional
        Standardise data along rows (``0``) or columns (``1``) before
        clustering; forwarded to ``Linkage``.
    standard_scale : {0, 1, None}, optional
        Normalise to [0, 1] along rows (``0``) or columns (``1``);
        forwarded to ``Linkage``.
    figsize : tuple of (float, float), optional
        Overall figure size as ``(width, height)`` in pixels, applied to
        the center heatmap panel via ``.properties()``.
    dendrogram_ratio : float, default 0.2
        Fraction of the total width/height allocated to each dendrogram
        panel, forwarded to ``ClusterMapChart``.
    theme : Theme, optional
        Visual theme applied to the center and dendrogram charts.
    **encode_kwargs
        Additional keyword arguments forwarded to ``Chart.encode()`` on
        the center heatmap chart.

    Returns
    -------
    ClusterMapChart
        Compound view with ``center``, ``row_dendrogram``, and
        ``col_dendrogram`` sub-charts.

    Raises
    ------
    ValueError
        If ``data`` has no numeric columns.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.clustermap(expr_df, method="average", cmap="rdbu", z_score=1)

    Correlation matrix with euclidean ward clustering:

    >>> fm.clustermap(corr_df, cmap="viridis")
    """
    from ferrum import (
        ClusterMapChart,
        Linkage,
        Reorder,
        Unpivot,
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

    # If no id column found, synthesize a `_row_id` column so the unpivot has
    # a row identity to carry through to the heatmap's y encoding. Mirrors
    # heatmap's handling for all-numeric input.
    data, tbl, id_col = _ensure_id_column(data, tbl, id_col)

    # Linkage transforms (rows + columns) with explicit names so we can route
    # their `segments` named outputs to the dendrogram layers.
    row_link = Linkage(
        method=method,
        metric=metric,
        axis="rows",
        z_score=z_score,
        standard_scale=standard_scale,
        name="row_link",
    )
    col_link = Linkage(
        method=method,
        metric=metric,
        axis="columns",
        z_score=z_score,
        standard_scale=standard_scale,
        name="col_link",
    )

    # Center heatmap: reorder rows + columns then unpivot.
    unpivot = Unpivot(
        id_vars=[id_col],
        value_vars=value_cols,
        var_name="column",
        value_name="value",
    )
    # Reorder reads its index column from a sibling named output via from=:
    # row_link publishes `row_link_order` (n rows of permutation indices);
    # the Reorder transform permutes the chained batch accordingly. The col
    # Reorder consumes col_link_order. (Phase 9 finalize: Reorder.from=)
    # Note: Linkage row/col indexing here assumes the column-Reorder runs
    # AFTER the unpivot — but currently we keep them pre-unpivot, which only
    # makes sense for the row axis. For column reordering with the wide-format
    # data, the canonical approach is to reorder columns by passing through
    # the column linkage order before unpivoting. We do both axes pre-unpivot;
    # the resulting unpivoted long-form preserves the visually-clustered order.
    # `drop_index=False` keeps the data columns intact (we're not dropping a
    # column from the chained batch — the index column lives in `from=` only).
    from ferrum.encoding import Color as _Color, X as _X, Y as _Y

    _color_enc = _Color("value", scheme=cmap)
    # Suppress synthetic axis titles (_row_id, column) that bleed through from
    # the internal unpivot column names — real row-label columns keep their names.
    _x_enc = _X("column", title="")
    _y_enc = _Y(id_col, title="" if id_col in ("_row_id", "column") else id_col)
    center_enc: dict = {"x": _x_enc, "y": _y_enc, "color": _color_enc}
    center_enc.update(encode_kwargs)
    center = (
        Chart(data)
        .transform(
            row_link,
            col_link,
            Reorder(by="original_idx", from_="row_link_order", drop_index=False),
            unpivot,
        )
        .mark_rect()
        .encode(**center_enc)
    )

    # figsize → set width/height on center heatmap panel.
    if figsize is not None:
        center = center.properties(width=figsize[0], height=figsize[1])

    from ferrum._layer import _Layer

    # Column dendrogram (top): reads col_link_segments and draws diagonal
    # segments. mark_segment requires x, y, x2, y2 in encoding.
    col_dendro_layer = _Layer(
        name="col_dendrogram",
        mark="segment",
        encoding={"x": "x", "y": "y", "x2": "x2", "y2": "y2"},
        data_source="col_link_segments",
    )
    col_dendro = Chart(data).transform(col_link)
    col_dendro._mark = None
    col_dendro._layers = [col_dendro_layer]
    col_dendro = col_dendro.axis(show=False)

    # Row dendrogram (left): reads row_link_segments, rotated via CoordFlip.
    from ferrum import CoordFlip

    row_dendro_layer = _Layer(
        name="row_dendrogram",
        mark="segment",
        encoding={"x": "x", "y": "y", "x2": "x2", "y2": "y2"},
        data_source="row_link_segments",
    )
    row_dendro = Chart(data).transform(row_link)
    row_dendro._mark = None
    row_dendro._layers = [row_dendro_layer]
    row_dendro = row_dendro.coord(CoordFlip()).axis(show=False)

    # Dendrogram panels should not show gridlines behind the tree branches.
    # Build a no-grid base theme; if the user supplied a theme, merge on top.
    from ferrum.themes import Theme as _Theme

    _dendro_base = _Theme(grid=False)
    if theme is not None:
        center = center.theme(theme)
        _dendro_theme = _dendro_base.update(**theme._props)
        col_dendro = col_dendro.theme(_dendro_theme)
        row_dendro = row_dendro.theme(_dendro_theme)
    else:
        col_dendro = col_dendro.theme(_dendro_base)
        row_dendro = row_dendro.theme(_dendro_base)

    cm = ClusterMapChart(
        center,
        row_dendrogram=row_dendro,
        col_dendrogram=col_dendro,
        dendrogram_ratio=dendrogram_ratio,
    )
    cm = _finalize_chart(cm, mark=mark, encode=encode, properties=properties, layers=layers, theme=None)
    return cm


# ---------------------------------------------------------------------------
# jointplot
# ---------------------------------------------------------------------------


def jointplot(
    data: Any,
    *,
    x: str,
    y: str,
    hue: Any = None,
    kind: str = "scatter",
    marginal_kind: str = "hist",
    ratio: int = 5,
    space: float = 0.05,
    xlim: Any = None,
    ylim: Any = None,
    joint_kws: Any = None,
    marginal_kws: Any = None,
    height: float | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
    **encode_kwargs: Any,
) -> JointChart:
    """Joint-distribution plot with marginals.

    Builds a ``JointChart`` composed of a central bivariate plot flanked by
    univariate marginals along the ``x`` (top) and ``y`` (right) axes.

    Parameters
    ----------
    data : DataFrame-like
        Input data accepted by ``Chart(data)``.
    x : str
        Column name for the horizontal axis (required).
    y : str
        Column name for the vertical axis (required).
    hue : str or encoding, optional
        Column name to map to color in both the center and marginal charts.
    kind : {"scatter", "kde", "hist", "hex", "reg"}, default "scatter"
        Mark to use for the central panel.  ``"scatter"`` draws
        ``mark_point``; ``"kde"`` draws ``mark_density``; ``"hist"``
        draws a 2-D histogram via ``Bin2D`` + ``mark_rect``; ``"hex"``
        draws ``mark_hex``; ``"reg"`` layers ``mark_point`` + a
        ``mark_smooth(method="lm")`` fit line.
    marginal_kind : {"hist", "kde", "rug", "box"}, default "hist"
        Mark to use for the marginal panels (same kind applied to both
        the top x-marginal and the right y-marginal).
    ratio : int, default 5
        Size ratio of the center panel to the marginal panels.
    space : float, default 0.05
        Gap (in layout units) between the center and marginal panels.
    xlim : tuple, optional
        ``(min, max)`` domain override for the x-axis.  Applied as an
        explicit scale domain on the center and top-marginal x encodings
        via ``X(field, scale={"domain": [min, max]})``.
    ylim : tuple, optional
        ``(min, max)`` domain override for the y-axis.  Applied as an
        explicit scale domain on the center and right-marginal y encodings
        via ``Y(field, scale={"domain": [min, max]})``.
    joint_kws : dict, optional
        Extra keyword arguments forwarded to the center-panel mark call.
    marginal_kws : dict, optional
        Extra keyword arguments forwarded to the marginal mark calls.
    height : float or None, optional
        Height and width of the square central panel in pixels.
    theme : Theme, optional
        Visual theme applied to all three panels via ``Chart.theme()``.
    **encode_kwargs
        Additional keyword arguments forwarded to ``Chart.encode()`` on
        the center chart.

    Returns
    -------
    JointChart
        Compound view with ``center``, ``top``, and ``right`` sub-charts.

    Raises
    ------
    ValueError
        If ``kind`` or ``marginal_kind`` is not one of the supported values.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.jointplot(df, x="sepal_length", y="sepal_width")

    2-D histogram center with KDE marginals, colored by species:

    >>> fm.jointplot(
    ...     df, x="sepal_length", y="petal_length",
    ...     kind="hist", marginal_kind="kde", hue="species",
    ... )
    """
    if kind not in _VALID_CENTER_KINDS:
        raise ValueError(
            f"jointplot: kind must be one of {sorted(_VALID_CENTER_KINDS)}; got {kind!r}"
        )
    if marginal_kind not in _VALID_MARGINAL_KINDS:
        raise ValueError(
            f"jointplot: marginal_kind must be one of {sorted(_VALID_MARGINAL_KINDS)}; "
            f"got {marginal_kind!r}"
        )

    # Build the center chart per `kind`.
    from ferrum.encoding import X as _X, Y as _Y

    jk = dict(joint_kws or {})

    enc_center: dict = {"x": x, "y": y}
    if hue is not None:
        enc_center["color"] = hue
    enc_center.update(encode_kwargs)

    # Apply xlim/ylim as scale domain overrides on center encodings.
    if xlim is not None:
        x_field = enc_center.get("x", x)
        if isinstance(x_field, str):
            enc_center["x"] = _X(x_field, scale={"domain": list(xlim)})
    if ylim is not None:
        y_field = enc_center.get("y", y)
        if isinstance(y_field, str):
            enc_center["y"] = _Y(y_field, scale={"domain": list(ylim)})

    if kind == "scatter":
        center = Chart(data).mark_point(**jk).encode(**enc_center)
    elif kind == "kde":
        center = Chart(data).mark_density(**jk).encode(**enc_center)
    elif kind == "hist":
        # Use Bin2D + mark_rect for 2D histogram.
        bin2d_kwargs: dict = {}
        for k in ("bins_x", "bins_y"):
            if k in jk:
                bin2d_kwargs[k] = jk.pop(k)
        hist_enc: dict = {"x": "bin_x_start", "y": "bin_y_start", "color": "count"}
        if xlim is not None:
            hist_enc["x"] = _X("bin_x_start", scale={"domain": list(xlim)})
        if ylim is not None:
            hist_enc["y"] = _Y("bin_y_start", scale={"domain": list(ylim)})
        center = (
            Chart(data)
            .transform(Bin2D(x=x, y=y, **bin2d_kwargs))
            .mark_rect()
            .encode(**hist_enc)
        )
    elif kind == "hex":
        center = Chart(data).mark_hex(**jk).encode(**enc_center)
    elif kind == "reg":
        # Layered: scatter + smoothed regression line.
        scatter = Chart(data).mark_point(**jk).encode(**enc_center)
        reg_enc: dict = {"x": x, "y": y}
        if xlim is not None:
            reg_enc["x"] = _X(x, scale={"domain": list(xlim)})
        if ylim is not None:
            reg_enc["y"] = _Y(y, scale={"domain": list(ylim)})
        fit = Chart(data).mark_smooth(method="lm", ci=None).encode(**reg_enc)
        center = _merge_layers(scatter, fit, scatter_name="scatter", fit_name="fit")

    # Build top marginal (over x).
    mk = dict(marginal_kws or {})
    enc_top: dict = {"x": x}
    if xlim is not None:
        enc_top["x"] = _X(x, scale={"domain": list(xlim)})
    if hue is not None:
        enc_top["color"] = hue
    if marginal_kind == "hist":
        top = Chart(data).mark_histogram(**mk).encode(**enc_top)
    elif marginal_kind == "kde":
        top = Chart(data).mark_density(**mk).encode(**enc_top)
    elif marginal_kind == "rug":
        top = Chart(data).mark_tick(**mk).encode(**enc_top)
    elif marginal_kind == "box":
        top = Chart(data).mark_boxplot(**mk).encode(**enc_top)

    # Build right marginal — oriented horizontally so bars/density grow along
    # the marginal's x-axis while the binned data dimension stays on the
    # marginal's y-axis (shared with the centre cell via share_y).
    enc_right: dict = {"y": y}
    if ylim is not None:
        enc_right["y"] = _Y(y, scale={"domain": list(ylim)})
    if hue is not None:
        enc_right["color"] = hue
    if marginal_kind == "hist":
        right = Chart(data).mark_histogram(orientation="horizontal", **mk).encode(**enc_right)
    elif marginal_kind == "kde":
        right = Chart(data).mark_density(orientation="horizontal", **mk).encode(**enc_right)
    elif marginal_kind == "rug":
        # Tick mark has no bin/density direction — the y-binding is enough.
        right = Chart(data).mark_tick(**mk).encode(**enc_right)
    elif marginal_kind == "box":
        # Boxplot is intrinsically asymmetric across the categorical axis;
        # JointChart uses the default vertical orientation with the data on x
        # (composite-mark orientation work tracked separately).
        right = Chart(data).mark_boxplot(**mk).encode(x=y)

    if height is not None:
        center = center.properties(width=height, height=height)

    if theme is not None:
        center = center.theme(theme)
        top = top.theme(theme)
        right = right.theme(theme)

    result = JointChart(
        center,
        top=top,
        right=right,
        ratio=ratio,
        spacing=space,
    )
    result = _finalize_chart(result, mark=mark, encode=encode, properties=properties, layers=layers, theme=None)
    return result
