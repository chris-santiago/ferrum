"""Data transforms — Python constructors for Phase 12 Rust data transforms.

Each function returns a plain dict matching the Rust ``TransformSpec`` serde
wire format (``#[serde(tag = "type", rename_all = "snake_case")]``). The dict
is passed through the ``transforms_json`` path at render time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from ferrum.parameter import Parameter

__all__ = [
    "transform_filter",
    "transform_calculate",
    "transform_aggregate",
    "transform_bin",
    "transform_fold",
    "transform_pivot",
    "transform_join_aggregate",
    "transform_window",
    "transform_density",
    "transform_regression",
    "transform_loess",
    "transform_impute",
    "transform_flatten",
    "transform_sample",
    "transform_top_k",
    "transform_stack",
    "transform_timeunit",
]


def transform_filter(predicate: "str | dict | Parameter") -> dict:
    """Filter rows by a predicate expression or a reactive parameter.

    Parameters
    ----------
    predicate : str, dict, or Parameter
        Vega-style expression string (e.g. ``"datum.x > 5"``), a dict filter
        specification, or a :class:`~ferrum.parameter.Parameter` (a selection
        or variable parameter).  A ``Parameter`` predicate emits a pass-through
        ``"true"`` predicate plus a ``param`` marker: the static render keeps
        all rows while the WASM runtime crossfilters live against the linked
        parameter.

    Returns
    -------
    dict
        Transform specification for the Rust engine.

    Examples
    --------
    >>> import ferrum as fm
    >>> t = fm.transform_filter("datum.age >= 18")
    >>> t["type"]
    'filter'

    >>> brush = fm.selection_interval(name="brush")
    >>> fm.transform_filter(brush)
    {'type': 'filter', 'predicate': 'true', 'param': 'brush'}
    """
    from ferrum.parameter import Parameter

    if isinstance(predicate, Parameter):
        return {"type": "filter", "predicate": "true", "param": predicate.name}
    if isinstance(predicate, dict):
        # Dict predicates are serialized as the expression string representation.
        # Convert common dict shapes to an expression string.
        parts = []
        for field, constraint in predicate.items():
            if isinstance(constraint, (list, tuple)):
                values_str = ", ".join(repr(v) for v in constraint)
                parts.append(f"indexof([{values_str}], datum.{field}) >= 0")
            elif isinstance(constraint, dict):
                for op, val in constraint.items():
                    parts.append(f"datum.{field} {op} {val!r}")
            else:
                parts.append(f"datum.{field} == {constraint!r}")
        predicate = " && ".join(parts) if parts else "true"
    return {"type": "filter", "predicate": str(predicate)}


def transform_calculate(as_: str, expr: str) -> dict:
    """Add a derived column via an expression.

    Parameters
    ----------
    as_ : str
        Name of the output column.
    expr : str
        Expression string (e.g. ``"datum.x * 2 + datum.y"``).

    Returns
    -------
    dict
        Transform specification for the Rust engine.

    Examples
    --------
    >>> import ferrum as fm
    >>> t = fm.transform_calculate("ratio", "datum.x / datum.y")
    >>> t["type"]
    'calculate'
    >>> t["as_field"]
    'ratio'
    """
    return {"type": "calculate", "as_field": as_, "expr": expr}


def transform_aggregate(
    *aggregates: dict,
    groupby: Sequence[str] | None = None,
) -> dict:
    """Group-by aggregation (collapses rows).

    Parameters
    ----------
    *aggregates : dict
        Aggregation specs, each a dict with keys ``field``, ``fn``, ``as``
        (e.g. ``{"field": "price", "fn": "mean", "as": "avg_price"}``).
    groupby : list of str, optional
        Columns to group by.

    Returns
    -------
    dict
        Transform specification for the Rust engine.

    Examples
    --------
    >>> import ferrum as fm
    >>> t = fm.transform_aggregate(
    ...     {"field": "price", "fn": "mean", "as": "avg_price"},
    ...     groupby=["category"],
    ... )
    >>> t["type"]
    'data_aggregate'
    """
    spec: dict = {
        "type": "data_aggregate",
        "aggregates": list(aggregates),
    }
    if groupby is not None:
        spec["groupby"] = list(groupby)
    return spec


def transform_bin(
    field: str,
    *,
    as_: str | None = None,
    maxbins: int | None = None,
    step: float | None = None,
    nice: bool = True,
) -> dict:
    """Bin a continuous field (adds a bin column without collapsing rows).

    Parameters
    ----------
    field : str
        Column to bin.
    as_ : str, optional
        Output column name. Defaults to ``"{field}_bin"``.
    maxbins : int, optional
        Maximum number of bins.
    step : float, optional
        Explicit bin width (overrides *maxbins*).
    nice : bool, default True
        Whether to "nice" the bin boundaries.

    Returns
    -------
    dict
        Transform specification for the Rust engine.

    Examples
    --------
    >>> import ferrum as fm
    >>> t = fm.transform_bin("horsepower", maxbins=10)
    >>> t["type"]
    'data_bin'
    >>> t["field"]
    'horsepower'
    """
    spec: dict = {"type": "data_bin", "field": field, "nice": nice}
    if as_ is not None:
        spec["as_"] = as_
    if maxbins is not None:
        spec["maxbins"] = maxbins
    if step is not None:
        spec["step"] = step
    return spec


def transform_fold(
    fields: Sequence[str],
    *,
    as_: tuple[str, str] = ("key", "value"),
) -> dict:
    """Fold (melt) columns from wide to long format.

    Parameters
    ----------
    fields : list of str
        Column names to fold.
    as_ : tuple of (str, str), default ("key", "value")
        Output column names for the key and value columns.

    Returns
    -------
    dict
        Transform specification for the Rust engine.

    Examples
    --------
    >>> import ferrum as fm
    >>> t = fm.transform_fold(["col_a", "col_b"])
    >>> t["type"]
    'fold'
    >>> t["as_"]
    ['key', 'value']
    """
    return {
        "type": "fold",
        "fields": list(fields),
        "as_": list(as_),
    }


def transform_pivot(
    field: str,
    value: str,
    *,
    groupby: Sequence[str] | None = None,
    limit: int | None = None,
    op: str = "sum",
) -> dict:
    """Pivot from long to wide format.

    Parameters
    ----------
    field : str
        Column whose unique values become new column headers.
    value : str
        Column whose values fill the pivoted cells.
    groupby : list of str, optional
        Columns to group by.
    limit : int, optional
        Maximum number of pivot columns to create.
    op : str, default "sum"
        Aggregation operation when multiple values map to the same cell.

    Returns
    -------
    dict
        Transform specification for the Rust engine.

    Examples
    --------
    >>> import ferrum as fm
    >>> t = fm.transform_pivot("category", "amount", groupby=["date"])
    >>> t["type"]
    'pivot'
    >>> t["field"]
    'category'
    """
    spec: dict = {"type": "pivot", "field": field, "value": value, "op": op}
    if groupby is not None:
        spec["groupby"] = list(groupby)
    if limit is not None:
        spec["limit"] = limit
    return spec


def transform_join_aggregate(
    *aggregates: dict,
    groupby: Sequence[str] | None = None,
) -> dict:
    """Add aggregate columns without collapsing rows (window-join pattern).

    Parameters
    ----------
    *aggregates : dict
        Aggregation specs, each a dict with keys ``field``, ``fn``, ``as``
        (e.g. ``{"field": "price", "fn": "mean", "as": "avg_price"}``).
    groupby : list of str, optional
        Columns to group by.

    Returns
    -------
    dict
        Transform specification for the Rust engine.

    Examples
    --------
    >>> import ferrum as fm
    >>> t = fm.transform_join_aggregate(
    ...     {"field": "sales", "fn": "sum", "as": "total_sales"},
    ...     groupby=["region"],
    ... )
    >>> t["type"]
    'join_aggregate'
    """
    spec: dict = {
        "type": "join_aggregate",
        "aggregates": list(aggregates),
    }
    if groupby is not None:
        spec["groupby"] = list(groupby)
    return spec


def transform_window(
    *window_transforms: dict,
    sort: Sequence[str] | None = None,
    groupby: Sequence[str] | None = None,
    frame: tuple[int | None, int | None] | None = None,
) -> dict:
    """Window transform (ranking, lag/lead, rolling aggregates).

    Parameters
    ----------
    *window_transforms : dict
        Window operation specs, each a dict with keys ``op``, ``as``
        and optionally ``field`` and ``param``
        (e.g. ``{"op": "row_number", "as": "rank"}``).
    sort : list of str, optional
        Sort fields for the window.
    groupby : list of str, optional
        Partition-by columns.
    frame : tuple of (int or None, int or None), optional
        Window frame bounds: (preceding, following). None means unbounded.

    Returns
    -------
    dict
        Transform specification for the Rust engine.

    Examples
    --------
    >>> import ferrum as fm
    >>> t = fm.transform_window(
    ...     {"op": "row_number", "as": "rank"},
    ...     sort=["score"],
    ... )
    >>> t["type"]
    'data_window'
    """
    spec: dict = {"type": "data_window", "ops": list(window_transforms)}
    if sort is not None:
        spec["sort"] = list(sort)
    if groupby is not None:
        spec["groupby"] = list(groupby)
    if frame is not None:
        spec["frame"] = list(frame)
    return spec


def transform_density(
    field: str,
    *,
    bandwidth: float | None = None,
    groupby: Sequence[str] | None = None,
    extent: tuple[float, float] | None = None,
    steps: int | None = None,
    cumulative: bool = False,
    as_: tuple[str, str] = ("value", "density"),
) -> dict:
    """Kernel density estimation as a data transform.

    Parameters
    ----------
    field : str
        Column to estimate density for.
    bandwidth : float, optional
        KDE bandwidth. If None, estimated automatically.
    groupby : list of str, optional
        Compute separate densities per group.
    extent : tuple of (float, float), optional
        Domain extent for the density grid.
    steps : int, optional
        Number of grid steps.
    cumulative : bool, default False
        If True, compute cumulative density.
    as_ : tuple of (str, str), default ("value", "density")
        Output column names.

    Returns
    -------
    dict
        Transform specification for the Rust engine.

    Examples
    --------
    >>> import ferrum as fm
    >>> t = fm.transform_density("weight", bandwidth=0.5)
    >>> t["type"]
    'density_data'
    >>> t["as_"]
    ['value', 'density']
    """
    spec: dict = {
        "type": "density_data",
        "field": field,
        "cumulative": cumulative,
        "as_": list(as_),
    }
    if bandwidth is not None:
        spec["bandwidth"] = bandwidth
    if groupby is not None:
        spec["groupby"] = list(groupby)
    if extent is not None:
        spec["extent"] = list(extent)
    if steps is not None:
        spec["steps"] = steps
    return spec


def transform_regression(
    x: str,
    y: str,
    *,
    method: str = "linear",
    order: int = 1,
    groupby: Sequence[str] | None = None,
    as_: tuple[str, str] = ("x", "y"),
) -> dict:
    """Regression fit as a data transform.

    Parameters
    ----------
    x : str
        Independent variable column.
    y : str
        Dependent variable column.
    method : str, default "linear"
        Regression method (e.g. "linear", "poly", "exp", "log", "pow").
    order : int, default 1
        Polynomial order (for method="poly").
    groupby : list of str, optional
        Fit separate regressions per group.
    as_ : tuple of (str, str), default ("x", "y")
        Output column names.

    Returns
    -------
    dict
        Transform specification for the Rust engine.

    Examples
    --------
    >>> import ferrum as fm
    >>> t = fm.transform_regression("x", "y", method="poly", order=2)
    >>> t["type"]
    'regression_data'
    >>> t["method"]
    'poly'
    """
    spec: dict = {
        "type": "regression_data",
        "x": x,
        "y": y,
        "method": method,
        "order": order,
        "as_": list(as_),
    }
    if groupby is not None:
        spec["groupby"] = list(groupby)
    return spec


def transform_loess(
    x: str,
    y: str,
    *,
    bandwidth: float = 0.3,
    groupby: Sequence[str] | None = None,
    as_: tuple[str, str] = ("x", "y"),
) -> dict:
    """LOESS/LOWESS smoothing as a data transform.

    Parameters
    ----------
    x : str
        Independent variable column.
    y : str
        Dependent variable column.
    bandwidth : float, default 0.3
        Smoothing bandwidth (fraction of data).
    groupby : list of str, optional
        Fit separate curves per group.
    as_ : tuple of (str, str), default ("x", "y")
        Output column names.

    Returns
    -------
    dict
        Transform specification for the Rust engine.

    Examples
    --------
    >>> import ferrum as fm
    >>> t = fm.transform_loess("x", "y", bandwidth=0.5)
    >>> t["type"]
    'loess_data'
    >>> t["bandwidth"]
    0.5
    """
    spec: dict = {
        "type": "loess_data",
        "x": x,
        "y": y,
        "bandwidth": bandwidth,
        "as_": list(as_),
    }
    if groupby is not None:
        spec["groupby"] = list(groupby)
    return spec


def transform_impute(
    field: str,
    *,
    method: str = "value",
    value: float | None = None,
    groupby: Sequence[str] | None = None,
    key: str | None = None,
) -> dict:
    """Impute missing values in a column.

    Parameters
    ----------
    field : str
        Column to impute.
    method : str, default "value"
        Imputation method: "value", "mean", "median", "min", "max".
    value : float, optional
        Constant value for method="value".
    groupby : list of str, optional
        Impute within groups.
    key : str, optional
        Key column for sequence generation.

    Returns
    -------
    dict
        Transform specification for the Rust engine.

    Examples
    --------
    >>> import ferrum as fm
    >>> t = fm.transform_impute("sales", method="mean")
    >>> t["type"]
    'impute'
    >>> t["method"]
    'mean'
    """
    spec: dict = {"type": "impute", "field": field, "method": method}
    if value is not None:
        spec["value"] = value
    if groupby is not None:
        spec["groupby"] = list(groupby)
    if key is not None:
        spec["key"] = key
    return spec


def transform_flatten(
    fields: Sequence[str],
    *,
    as_: Sequence[str] | None = None,
) -> dict:
    """Flatten list/array columns into separate rows.

    Parameters
    ----------
    fields : list of str
        Column names to flatten.
    as_ : list of str, optional
        Output names for flattened columns.

    Returns
    -------
    dict
        Transform specification for the Rust engine.

    Examples
    --------
    >>> import ferrum as fm
    >>> t = fm.transform_flatten(["tags"])
    >>> t["type"]
    'flatten'
    >>> t["fields"]
    ['tags']
    """
    spec: dict = {"type": "flatten", "fields": list(fields)}
    if as_ is not None:
        spec["as_"] = list(as_)
    return spec


def transform_sample(n: int, *, seed: int = 42) -> dict:
    """Random sample of rows.

    Parameters
    ----------
    n : int
        Number of rows to sample.
    seed : int, default 42
        RNG seed for deterministic sampling.

    Returns
    -------
    dict
        Transform specification for the Rust engine.

    Examples
    --------
    >>> import ferrum as fm
    >>> t = fm.transform_sample(100, seed=7)
    >>> t["type"]
    'sample'
    >>> t["n"]
    100
    """
    return {"type": "sample", "n": n, "seed": seed}


def transform_top_k(
    n: int,
    *,
    field: str,
    op: str = "sum",
    sort: str = "descending",
) -> dict:
    """Keep top-k groups by an aggregate value.

    Parameters
    ----------
    n : int
        Number of top groups to keep.
    field : str
        Field to aggregate for ranking.
    op : str, default "sum"
        Aggregation operation: "sum", "mean", "count", "min", "max".
    sort : str, default "descending"
        Sort direction: "descending" or "ascending".

    Returns
    -------
    dict
        Transform specification for the Rust engine.

    Examples
    --------
    >>> import ferrum as fm
    >>> t = fm.transform_top_k(5, field="revenue", op="sum")
    >>> t["type"]
    'top_k'
    >>> t["n"]
    5
    """
    return {"type": "top_k", "n": n, "field": field, "op": op, "sort": sort}


_VALID_STACK_OFFSETS: frozenset[str] = frozenset(["zero", "normalize", "center"])


def transform_stack(
    field: str,
    *,
    groupby: Sequence[str],
    sort: Sequence[str] | None = None,
    as_: tuple[str, str] = ("y0", "y1"),
    offset: str = "zero",
) -> dict:
    """Compute stacked (cumulative) positions for bar/area charts.

    Parameters
    ----------
    field : str
        Field to stack.
    groupby : list of str
        Columns defining each stack group.
    sort : list of str, optional
        Sort order within each stack.
    as_ : tuple of (str, str), default ("y0", "y1")
        Output column names for stack start and end.
    offset : str, default "zero"
        Offset mode: "zero", "normalize", or "center".

    Returns
    -------
    dict
        Transform specification for the Rust engine.

    Raises
    ------
    ValueError
        If ``offset`` is not one of ``"zero"``, ``"normalize"``, ``"center"``.

    Examples
    --------
    >>> import ferrum as fm
    >>> t = fm.transform_stack("sales", groupby=["region", "quarter"])
    >>> t["type"]
    'data_stack'
    >>> t["as_"]
    ['y0', 'y1']
    """
    if offset not in _VALID_STACK_OFFSETS:
        raise ValueError(
            f"transform_stack: offset={offset!r} is not valid. "
            f"Valid values: {sorted(_VALID_STACK_OFFSETS)}"
        )
    spec: dict = {
        "type": "data_stack",
        "field": field,
        "groupby": list(groupby),
        "as_": list(as_),
        "offset": offset,
    }
    if sort is not None:
        spec["sort"] = list(sort)
    return spec


def transform_timeunit(
    field: str,
    unit: str,
    *,
    utc: bool = False,
    as_: str | None = None,
) -> dict:
    """Extract a temporal unit from a datetime field.

    Parameters
    ----------
    field : str
        Datetime column.
    unit : str
        Unit to extract: "year", "month", "day", "hour", "minute",
        "second", "day_of_week", "week", "quarter".
    utc : bool, default False
        Whether to interpret timestamps as UTC.
    as_ : str, optional
        Output column name. Defaults to ``"{unit}_{field}"``.

    Returns
    -------
    dict
        Transform specification for the Rust engine.

    Examples
    --------
    >>> import ferrum as fm
    >>> t = fm.transform_timeunit("date", "month")
    >>> t["type"]
    'time_unit'
    >>> t["unit"]
    'month'
    """
    spec: dict = {"type": "time_unit", "field": field, "unit": unit, "utc": utc}
    if as_ is not None:
        spec["as_"] = as_
    return spec
