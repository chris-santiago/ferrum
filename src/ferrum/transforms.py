"""Data transforms — Python constructors for Phase 12 Rust data transforms.

Each function returns a plain dict matching the Rust ``TransformSpec`` serde
wire format (``#[serde(tag = "type", rename_all = "snake_case")]``). The dict
is passed through the ``transforms_json`` path at render time.

Spelling note — two ``as`` forms
---------------------------------
There are two intentionally different spellings of the output-column name:

* **Wrapper-level kwarg:** ``as_`` (a valid Python identifier; trailing
  underscore avoids the ``as`` keyword).  Every ``transform_*`` wrapper that
  accepts an output-column name uses ``as_`` as the kwarg.

* **Inner-dict Vega wire form:** ``"as"`` — used inside the user-supplied
  aggregate/window operation dicts that are passed verbatim to the Rust
  engine (e.g. ``{"field": "price", "fn": "mean", "as": "avg_price"}``).
  The Vega wire protocol uses ``"as"`` without the trailing underscore.

The two spellings are intentional and not drift: the wrapper kwarg is a
Python API convenience; the inner-dict key is the Rust/Vega serialization
contract.  Do not change the inner-dict key.
"""

from __future__ import annotations

import math
import numbers
import operator
from typing import TYPE_CHECKING, Any, Sequence

from ferrum.position import _validate_stack_offset

if TYPE_CHECKING:
    from ferrum.parameter import Parameter


# ---------------------------------------------------------------------------
# as_ validation helpers (SEAM-03 parity)
# ---------------------------------------------------------------------------


def _validate_as_str(value: str, *, where: str) -> None:
    """Raise ``ValueError`` if *value* is not a non-empty string.

    Used for single-column ``as_`` parameters (``transform_calculate``,
    ``transform_bin``, ``transform_timeunit``).

    Parameters
    ----------
    value : str
        The output-column name to validate.
    where : str
        Caller name, used in the error message.

    Raises
    ------
    ValueError
        If *value* is not a non-empty string.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where}: as_ must be a non-empty string, got {value!r}")


def _validate_as_pair(value: tuple[str, str], *, where: str) -> list[str]:
    """Validate and materialize a pair ``as_`` argument.

    Used for two-column ``as_`` parameters (``transform_fold``,
    ``transform_density``, ``transform_regression``, ``transform_loess``,
    ``transform_stack``).

    Parameters
    ----------
    value : tuple of (str, str)
        The pair of output-column names to validate.
    where : str
        Caller name, used in the error message.

    Returns
    -------
    list of str
        The materialized two-element list.  Callers MUST use this return
        value — not ``list(as_)`` — so that one-shot iterables (generators,
        map, zip) are consumed exactly once.

    Raises
    ------
    ValueError
        If *value* is not a length-2 sequence of non-empty strings.
        A bare string is rejected even if it has exactly 2 characters,
        because iterating a string yields individual characters, not
        column names.
    """
    if isinstance(value, str):
        raise ValueError(
            f"{where}: as_ must be a pair of non-empty strings (tuple or list),"
            f" got a bare string {value!r}"
        )
    try:
        items = list(value)
    except TypeError:
        raise ValueError(f"{where}: as_ must be a pair of non-empty strings, got {value!r}")
    if len(items) != 2 or not all(isinstance(s, str) and s for s in items):
        raise ValueError(f"{where}: as_ must be a pair of non-empty strings, got {value!r}")
    return items


def _reject_bare_string(value: object, *, where: str, param: str, noun: str) -> None:
    """Raise ``ValueError`` if *value* is a bare string, not a sequence.

    The single refusal mechanism for the class of parameter whose declared
    type is a sequence of strings (``groupby=``, ``fields=``, ``sort=``,
    ``keyvals=``, ``as_=``), where a bare string is not a type error but a
    silent corruption: ``list("cat")`` is three one-character column names
    the wrapper would happily send to Rust.

    Deliberately not enumerated here. Which parameters belong to the class
    is decided by their annotations, and
    ``tests/test_transform_bare_string_sequence_params.py`` derives that set
    by introspecting this module and asserts each member refuses a bare
    string — so a newly added sequence parameter that forgets this guard
    fails there instead of quietly re-opening the bug. An enumeration in
    this docstring went stale twice while the fix was being written.

    Parameters
    ----------
    value : object
        The candidate value.
    where : str
        Caller name, used in the error message.
    param : str
        The parameter name being validated (e.g. ``"groupby"``,
        ``"keyvals"``), used in the error message.
    noun : str
        Plural noun describing what the sequence's elements are (e.g.
        ``"column names"``, ``"key values"``), used in the error message.

    Raises
    ------
    ValueError
        If *value* is a ``str``.
    """
    if isinstance(value, str):
        raise ValueError(
            f"{where}: {param} must be a sequence of {noun} (list or tuple), "
            f"got a bare string {value!r}; iterating a string yields individual characters"
        )


def _reject_bare_string_groupby(value: object, *, where: str) -> None:
    """Raise ``ValueError`` if *value* is a bare string, not a sequence of
    column names.

    A thin, ``groupby``-specific wrapper over :func:`_reject_bare_string`
    (which supplies the shared message) so call sites and tests that pass
    only ``where=`` keep working. Every ``transform_*`` wrapper that accepts
    ``groupby=`` routes through it — a claim proven by the introspection-derived
    coverage in ``tests/test_transform_bare_string_sequence_params.py``, not
    by a list of wrapper names kept in sync by hand here.

    Before this helper existed, ``transform_regression``/``transform_loess``/
    ``transform_impute`` silently accepted a bare string and char-split it
    (``list("cat")`` -> ``['c', 'a', 't']``), sending three bogus
    single-character group columns to Rust instead of raising —
    ``transform_top_k`` was the only wrapper with a hand-rolled version of
    this same guard. This is the wrapper-level typed-refusal half of the
    two-layer groupby shape; the Rust-side pyclasses (``PyBin``, ``PyKde``,
    ``PyKde2d``, ``PySmooth``, ``PyTopK``) separately *normalize* a bare
    string to a single-element list via ``parse_groupby_pyany`` at their own
    constructor boundary — the two layers intentionally differ (refuse vs.
    normalize) because a `transform_*` wrapper builds a plain wire dict with
    no constructor to normalize through.

    Materializing the value into a list remains each call site's own job,
    since the required-vs-optional and default-vs-explicit shapes differ
    per wrapper.

    Parameters
    ----------
    value : object
        The candidate ``groupby`` value.
    where : str
        Caller name, used in the error message.

    Raises
    ------
    ValueError
        If *value* is a ``str``.
    """
    _reject_bare_string(value, where=where, param="groupby", noun="column names")


def _validate_as_seq(value: Sequence[str], *, where: str) -> list[str]:
    """Validate and materialize a variable-length ``as_`` argument.

    Used for variable-length ``as_`` parameters (``transform_flatten``).

    Parameters
    ----------
    value : sequence of str
        The output-column names to validate.
    where : str
        Caller name, used in the error message.

    Returns
    -------
    list of str
        The materialized list.  Callers MUST use this return value — not
        ``list(as_)`` — so that one-shot iterables (generators, map, zip)
        are consumed exactly once.

    Raises
    ------
    ValueError
        If *value* is a bare string, empty, or contains a non-string or
        empty-string element.  A bare string is refused through the same
        :func:`_reject_bare_string` mechanism the sequence parameters use:
        every character of ``as_="tag"`` is a non-empty string, so the
        element checks below would otherwise accept it as three one-letter
        output columns.
    """
    _reject_bare_string(value, where=where, param="as_", noun="output-column names")
    try:
        items = list(value)
    except TypeError:
        raise ValueError(f"{where}: as_ must be a non-empty sequence of strings, got {value!r}")
    if not items or not all(isinstance(s, str) and s for s in items):
        raise ValueError(
            f"{where}: as_ must be a non-empty sequence of non-empty strings, got {value!r}"
        )
    return items


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


def _expr_literal(value: object, *, field: str | None = None) -> str:
    """Serialize a Python scalar as a ferrum expression-grammar literal.

    Used by ``transform_filter``'s dict-shorthand branches to build a
    ``datum.field <op> <literal>`` / ``indexof([<literal>, ...], datum.field)``
    predicate string the Rust expression grammar (``transform/expr.rs``) can
    parse — plain ``repr()`` disagrees with the grammar for bools (``True``
    vs. ``true``) and ``None`` (``None`` vs. ``null``), and does not escape
    embedded quotes the way the grammar's string tokenizer expects.

    ``field`` is the predicate field name the value came from, threaded
    through only to name it in raised error messages; it does not affect
    the emitted literal.

    Boolean and numeric acceptance boundaries
    ------------------------------------------
    Booleans are accepted as Python ``bool`` or numpy ``bool_`` (the routine
    result of a pandas/polars boolean column access), both normalized to
    ``"true"`` or ``"false"``.

    Numeric values are accepted by ABC (``numbers.Integral`` /
    ``numbers.Real``), not by concrete type, so numpy scalars — the routine
    result of a pandas/polars aggregation (``df[col].mean()``,
    ``.max()``, ...) — are accepted and normalized to a plain literal
    rather than forwarding a hostile ``repr()`` (numpy >= 2's
    ``repr(np.float64(2.0))`` is ``"np.float64(2.0)"``, which the grammar's
    tokenizer cannot parse). Integers and floats are normalized separately
    (``operator.index`` / ``float``) rather than collapsed into one path,
    since routing a large integer through ``float()`` would lose precision.
    ``decimal.Decimal`` and ``datetime`` are deliberately *not* accepted:
    neither is registered under ``numbers.Real`` (the stdlib's own
    ``decimal`` module opts out for the same reason — silently coercing a
    ``Decimal`` to ``float`` reintroduces the precision-loss hazard this
    normalization exists to avoid), so both types still raise ``TypeError``.

    Note on large-integer precision
    --------------------------------
    The emitted literal for a large integer represents the input exactly
    (e.g. ``2**60 + 1`` emits the exact digit string). However, the Rust
    expression grammar evaluates all numbers as ``f64``, so a filter that
    should match only ``2**60 + 1`` can match its ``f64``-equivalent neighbor
    ``2**60`` end-to-end. This is a grammar-side limitation, not a
    serialization gap.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    # Accept numpy bool scalars the same way we accept their Python twin.
    # np.bool_ is not a bool subclass, so it must be checked separately.
    try:
        import numpy as np

        if isinstance(value, np.bool_):
            return "true" if bool(value) else "false"
    except (ImportError, AttributeError):
        pass
    if value is None:
        return "null"
    if isinstance(value, str):
        escaped = (
            value.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\t", "\\t")
        )
        return f'"{escaped}"'
    if isinstance(value, numbers.Integral):
        return repr(operator.index(value))
    if isinstance(value, numbers.Real):
        as_float = float(value)
        if not math.isfinite(as_float):
            where = f" for field {field!r}" if field is not None else ""
            accepted = "string, number, bool, or None"
            msg = (
                f"transform_filter: non-finite float {value!r} not allowed{where}; "
                f"accepted: {accepted}"
            )
            raise ValueError(msg)
        return repr(as_float)
    where = f" for field {field!r}" if field is not None else ""
    accepted = "string, number, bool, or None"
    msg = f"transform_filter: {type(value).__name__} {value!r} not allowed{where}; accepted: {accepted}"
    raise TypeError(msg)


def transform_filter(predicate: "str | dict | Parameter") -> dict:
    """Filter rows by a predicate expression or a reactive parameter.

    Parameters
    ----------
    predicate : str, dict, or Parameter
        Vega-style expression string (e.g. ``"datum.x > 5"``), a dict filter
        specification, or a [Parameter][ferrum.Parameter] (a selection
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
                values_str = ", ".join(_expr_literal(v, field=field) for v in constraint)
                parts.append(f"indexof([{values_str}], datum.{field}) >= 0")
            elif isinstance(constraint, dict):
                for op, val in constraint.items():
                    parts.append(f"datum.{field} {op} {_expr_literal(val, field=field)}")
            else:
                parts.append(f"datum.{field} == {_expr_literal(constraint, field=field)}")
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
    _validate_as_str(as_, where="transform_calculate")
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

    Raises
    ------
    ValueError
        If ``groupby`` is a bare string — iterating a string yields
        individual characters, not column names.

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
        _reject_bare_string_groupby(groupby, where="transform_aggregate")
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
    if as_ is not None:
        _validate_as_str(as_, where="transform_bin")
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

    Raises
    ------
    ValueError
        If ``fields`` or ``as_`` is a bare string — iterating a string yields
        individual characters, not column names.

    Examples
    --------
    >>> import ferrum as fm
    >>> t = fm.transform_fold(["col_a", "col_b"])
    >>> t["type"]
    'fold'
    >>> t["as_"]
    ['key', 'value']
    """
    _reject_bare_string(fields, where="transform_fold", param="fields", noun="column names")
    as_list = _validate_as_pair(as_, where="transform_fold")
    return {
        "type": "fold",
        "fields": list(fields),
        "as_": as_list,
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

    Raises
    ------
    ValueError
        If ``groupby`` is a bare string — iterating a string yields
        individual characters, not column names.

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
        _reject_bare_string_groupby(groupby, where="transform_pivot")
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

    Raises
    ------
    ValueError
        If ``groupby`` is a bare string — iterating a string yields
        individual characters, not column names.

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
        _reject_bare_string_groupby(groupby, where="transform_join_aggregate")
        spec["groupby"] = list(groupby)
    return spec


def transform_window(
    *window_transforms: dict,
    sort: Sequence[str] | None = None,
    groupby: Sequence[str] | None = None,
    frame: tuple[int | None, int | None] | None = None,
    ignore_peers: bool = False,
) -> dict:
    """Window transform (ranking, lag/lead, rolling aggregates).

    Parameters
    ----------
    *window_transforms : dict
        Window operation specs, each a dict with keys ``op``, ``as``
        and optionally ``field`` and ``param``
        (e.g. ``{"op": "row_number", "as": "rank"}``).
    sort : list of str, optional
        Sort fields for the window. ``rank`` and ``dense_rank`` written
        without a ``field`` take their ties from these columns: rows equal on
        every sort column draw the same rank (``1, 1, 3`` for ``rank``,
        ``1, 1, 2`` for ``dense_rank``).
    groupby : list of str, optional
        Partition-by columns.
    frame : tuple of (int or None, int or None), optional
        Window frame bounds: (preceding, following). None means unbounded.
    ignore_peers : bool, default False
        Whether a rolling frame ignores *peers* — rows tied on every ``sort``
        column. Left ``False`` (Vega's default), a frame expands to whole peer
        groups, so tied rows share one frame and therefore one aggregate
        value. Set ``True`` for frames defined by row offsets alone. Only
        rolling aggregates with an explicit ``frame`` over a tied sort key are
        affected.

    Returns
    -------
    dict
        Transform specification for the Rust engine.

    Raises
    ------
    ValueError
        If ``sort`` or ``groupby`` is a bare string — iterating a string
        yields individual characters, not column names.

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
        _reject_bare_string(sort, where="transform_window", param="sort", noun="column names")
        spec["sort"] = list(sort)
    if groupby is not None:
        _reject_bare_string_groupby(groupby, where="transform_window")
        spec["groupby"] = list(groupby)
    if frame is not None:
        spec["frame"] = list(frame)
    # Omitted at the default so specs built before this parameter existed
    # serialize byte-identically.
    if ignore_peers:
        spec["ignore_peers"] = True
    return spec


def transform_density(
    field: str,
    *,
    bandwidth: float | None = None,
    kernel: str = "gaussian",
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
    kernel : str, default "gaussian"
        Kernel function: ``"gaussian"``, ``"epanechnikov"`` (alias
        ``"epan"``), ``"tophat"`` (alias ``"uniform"``), ``"cosine"``,
        ``"exponential"``, or ``"linear"``. An unrecognized kernel raises a
        typed error from the Rust engine at render time (not validated here
        -- the transform functions in this module defer vocabulary checks
        to the single Rust-side source of truth rather than duplicating it).
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

    Raises
    ------
    ValueError
        If ``groupby`` is a bare string — iterating a string yields
        individual characters, not column names.

    Examples
    --------
    >>> import ferrum as fm
    >>> t = fm.transform_density("weight", bandwidth=0.5)
    >>> t["type"]
    'density_data'
    >>> t["as_"]
    ['value', 'density']
    >>> t = fm.transform_density("weight", kernel="epanechnikov")
    >>> t["kernel"]
    'epanechnikov'
    """
    as_list = _validate_as_pair(as_, where="transform_density")
    spec: dict = {
        "type": "density_data",
        "field": field,
        "cumulative": cumulative,
        "as_": as_list,
    }
    if bandwidth is not None:
        spec["bandwidth"] = bandwidth
    # Omitted at the default so specs built before this parameter existed
    # serialize byte-identically (mirrors DensityDataSpec's serde default).
    if kernel != "gaussian":
        spec["kernel"] = kernel
    if groupby is not None:
        _reject_bare_string_groupby(groupby, where="transform_density")
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

    Raises
    ------
    ValueError
        If ``groupby`` is a bare string — iterating a string yields
        individual characters, not column names.

    Examples
    --------
    >>> import ferrum as fm
    >>> t = fm.transform_regression("x", "y", method="poly", order=2)
    >>> t["type"]
    'regression_data'
    >>> t["method"]
    'poly'
    """
    as_list = _validate_as_pair(as_, where="transform_regression")
    spec: dict = {
        "type": "regression_data",
        "x": x,
        "y": y,
        "method": method,
        "order": order,
        "as_": as_list,
    }
    if groupby is not None:
        _reject_bare_string_groupby(groupby, where="transform_regression")
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

    Raises
    ------
    ValueError
        If ``groupby`` is a bare string — iterating a string yields
        individual characters, not column names.

    Examples
    --------
    >>> import ferrum as fm
    >>> t = fm.transform_loess("x", "y", bandwidth=0.5)
    >>> t["type"]
    'loess_data'
    >>> t["bandwidth"]
    0.5
    """
    as_list = _validate_as_pair(as_, where="transform_loess")
    spec: dict = {
        "type": "loess_data",
        "x": x,
        "y": y,
        "bandwidth": bandwidth,
        "as_": as_list,
    }
    if groupby is not None:
        _reject_bare_string_groupby(groupby, where="transform_loess")
        spec["groupby"] = list(groupby)
    return spec


def _normalize_keyval(value: object, *, key: str | None) -> bool | int | float | str:
    """Normalize one ``transform_impute(keyvals=...)`` element to a plain,
    JSON-serializable Python scalar.

    Mirrors ``_expr_literal``'s ABC-based (not concrete-type) numeric/bool
    acceptance boundary: a numpy scalar from ``df[col].unique()`` — the
    routine way to build a ``keyvals`` list off the key column itself —
    normalizes the same way its Python-native twin does (``np.int64`` and
    ``np.float64`` both accepted), rather than reaching the JSON encoder as
    an unserializable numpy scalar (``TypeError: Object of type int64 is
    not JSON serializable``, naming neither ``transform_impute`` nor
    ``keyvals``). Unlike ``_expr_literal``, this returns a plain scalar
    rather than a grammar-literal string, since ``keyvals`` rides the wire
    as JSON, not an expression.
    """
    if isinstance(value, bool):
        return value
    # Accept numpy bool scalars the same way we accept their Python twin.
    try:
        import numpy as np

        if isinstance(value, np.bool_):
            return bool(value)
    except (ImportError, AttributeError):
        pass
    if isinstance(value, str):
        return value
    where = f" for key {key!r}" if key is not None else ""
    if isinstance(value, numbers.Integral):
        return operator.index(value)
    if isinstance(value, numbers.Real):
        as_float = float(value)
        if not math.isfinite(as_float):
            raise ValueError(
                f"transform_impute: non-finite float {value!r} not allowed in "
                f"keyvals{where}; accepted: string, number, or bool"
            )
        return as_float
    raise TypeError(
        f"transform_impute: {type(value).__name__} {value!r} not allowed in "
        f"keyvals{where}; accepted: string, number, or bool"
    )


def transform_impute(
    field: str,
    *,
    method: str = "value",
    value: float | None = None,
    groupby: Sequence[str] | None = None,
    key: str | None = None,
    keyvals: Sequence[Any] | None = None,
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
        Impute within groups: the fill value (per ``method``) is computed
        separately for each group, not once across the whole column.
    key : str, optional
        Key column. Within each group (or, with no ``groupby``, the whole
        batch as one implicit group — Vega's "no groupby means one group"
        rule), a key value present in the key universe (see ``keyvals``)
        but missing from that group gets one appended row: the group's own
        columns, the missing key, and that group's own imputed value; every
        other column is null.
    keyvals : list or tuple, optional
        The key universe. **Requires** ``key`` — passing ``keyvals``
        without ``key`` raises ``ValueError`` at render time, since there
        is no key column for the universe to apply to. Must be a sequence
        (list, tuple, or other iterable), never a bare string — a bare
        string iterates into characters, not key values, and raises
        ``ValueError`` at construction time.
        When omitted, the universe is the distinct ``key`` values observed
        across the whole batch. Each value is typed against the key
        column's own dtype at render time: a whole-valued float (e.g.
        ``1.0``) coerces onto an integer key column, but a genuinely
        fractional float against an integer key column raises ``ValueError``
        (that coercion would be lossy); a string against a numeric key
        column raises the same way.

    Returns
    -------
    dict
        Transform specification for the Rust engine.

    Raises
    ------
    ValueError
        If ``groupby`` or ``keyvals`` is a bare string — iterating a
        string yields individual characters, not column/key values.

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
        _reject_bare_string_groupby(groupby, where="transform_impute")
        spec["groupby"] = list(groupby)
    if key is not None:
        spec["key"] = key
    if keyvals is not None:
        _reject_bare_string(keyvals, where="transform_impute", param="keyvals", noun="key values")
        spec["keyvals"] = [_normalize_keyval(v, key=key) for v in keyvals]
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

    Raises
    ------
    ValueError
        If ``fields`` or ``as_`` is a bare string — iterating a string yields
        individual characters, not column names.

    Examples
    --------
    >>> import ferrum as fm
    >>> t = fm.transform_flatten(["tags"])
    >>> t["type"]
    'flatten'
    >>> t["fields"]
    ['tags']
    """
    _reject_bare_string(fields, where="transform_flatten", param="fields", noun="column names")
    as_list = _validate_as_seq(as_, where="transform_flatten") if as_ is not None else None
    spec: dict = {"type": "flatten", "fields": list(fields)}
    if as_list is not None:
        spec["as_"] = as_list
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
    groupby: Sequence[str] | None = None,
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
    groupby : list of str, optional
        Rank groups defined by these columns' distinct value tuples,
        aggregating ``op`` over ``field`` within each group — the
        top-k-per-category use case (e.g. top 3 products by revenue sum).
        Every row belonging to one of the top-``n`` groups is kept; no rows
        or columns are synthesized. When omitted, the legacy behavior is
        preserved byte-identically: ``field``'s own distinct values double
        as the group key, so ranking degenerates to the field's raw value
        rather than a separate category — a documented historical quirk
        (F-L05-01), not the intended use case. Pass ``groupby=`` for that.
        An empty sequence (``groupby=[]``) takes the same legacy ungrouped
        path as omitting the parameter entirely — it is not treated as "no
        groups", consistent with the rest of the ``groupby=``-accepting
        transform family.

    Returns
    -------
    dict
        Transform specification for the Rust engine.

    Raises
    ------
    ValueError
        If ``groupby`` is a bare string — iterating a string yields
        individual characters, not column names.

    Examples
    --------
    >>> import ferrum as fm
    >>> t = fm.transform_top_k(5, field="revenue", op="sum")
    >>> t["type"]
    'top_k'
    >>> t["n"]
    5

    >>> t = fm.transform_top_k(3, field="revenue", op="sum", groupby=["product"])
    >>> t["groupby"]
    ['product']
    """
    spec: dict = {"type": "top_k", "n": n, "field": field, "op": op, "sort": sort}
    if groupby is not None:
        _reject_bare_string_groupby(groupby, where="transform_top_k")
        spec["groupby"] = list(groupby)
    return spec


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
        If ``offset`` is not one of ``"zero"``, ``"normalize"``, ``"center"``,
        or if ``groupby`` or ``sort`` is a bare string — iterating a string
        yields individual characters, not column names.

    Examples
    --------
    >>> import ferrum as fm
    >>> t = fm.transform_stack("sales", groupby=["region", "quarter"])
    >>> t["type"]
    'data_stack'
    >>> t["as_"]
    ['y0', 'y1']
    """
    _validate_stack_offset(offset, func_name="transform_stack", param="offset")
    as_list = _validate_as_pair(as_, where="transform_stack")
    _reject_bare_string_groupby(groupby, where="transform_stack")
    spec: dict = {
        "type": "data_stack",
        "field": field,
        "groupby": list(groupby),
        "as_": as_list,
        "offset": offset,
    }
    if sort is not None:
        _reject_bare_string(sort, where="transform_stack", param="sort", noun="column names")
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
    if as_ is not None:
        _validate_as_str(as_, where="transform_timeunit")
    spec: dict = {"type": "time_unit", "field": field, "unit": unit, "utc": utc}
    if as_ is not None:
        spec["as_"] = as_
    return spec
