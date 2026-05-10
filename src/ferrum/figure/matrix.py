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


def heatmap(*args, **kwargs):
    raise NotImplementedError("heatmap — implementation lands in Task 33")


def clustermap(*args, **kwargs):
    raise NotImplementedError("clustermap — implementation lands in Task 34")
