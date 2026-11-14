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

import datetime
from typing import TYPE_CHECKING, Any, Sequence, cast

import numpy as np

from ferrum._validate import (
    coerce_finite_real,
    coerce_frame,
    coerce_impute_options,
    is_real_number,
)
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


def _reject_repeated_groupby_names(value: Sequence[str], *, where: str) -> None:
    """Raise ``ValueError`` if *value* names the same groupby column twice.

    A repeat is malformed independently of any data: a grouped transform
    appends one output column per groupby dimension, so a repeat means two
    identically-named columns on the wire — silent schema corruption, not a
    type error. Companion to :func:`_reject_bare_string_groupby` for
    ``transform_pivot``/``transform_aggregate``/``transform_density``: these
    three build a raw wire dict with no Rust pyclass constructor to refuse
    at (unlike ``Aggregate``, whose ``#[new]`` already calls the Rust-side
    twin of this check), so before this guard the duplicate was only caught
    downstream at Rust's apply-time partitioning — reachable, but deferred
    past the construction-time boundary that already has every fact it
    needs (GH #179).

    Message wording matches Rust's ``refuse_repeated_groupby_names``
    (``crates/ferrum-core/src/transform/group_key/mod.rs``) so the same
    mistake reads identically whether caught here or at the Rust boundary.

    Parameters
    ----------
    value : sequence of str
        The already-materialized ``groupby`` column names (call after
        :func:`_reject_bare_string_groupby` has ruled out a bare string).
    where : str
        Caller name, used in the error message.

    Raises
    ------
    ValueError
        If any name in *value* appears more than once.
    """
    seen: set[str] = set()
    for name in value:
        if name in seen:
            raise ValueError(
                f"{where}: groupby column {name!r} is repeated in `groupby`; "
                f"each groupby column must have a distinct name "
                f"(got groupby={list(value)!r})"
            )
        seen.add(name)


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

    Numeric values are accepted by :func:`ferrum._validate.is_real_number`,
    not by concrete type, so numpy scalars — the routine
    result of a pandas/polars aggregation (``df[col].mean()``,
    ``.max()``, ...) — are accepted and normalized to a plain literal
    rather than forwarding a hostile ``repr()`` (numpy >= 2's
    ``repr(np.float64(2.0))`` is ``"np.float64(2.0)"``, which the grammar's
    tokenizer cannot parse). The number is normalized by
    :func:`ferrum._validate.coerce_finite_real`, which keeps an integer an
    ``int`` (routing a large integer through ``float()`` would lose
    precision) and refuses ``nan``/``inf`` and an integer a float cannot hold.
    A ``decimal.Decimal`` is emitted as its ``float`` value: the expression
    grammar evaluates every number as ``f64`` (see below), so no precision a
    ``Decimal`` literal could carry survives evaluation anyway. ``datetime``
    is not a number and raises ``TypeError``.

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
    where = f" for field {field!r}" if field is not None else ""
    if is_real_number(value):
        return repr(coerce_finite_real(f"transform_filter: value{where}", value))
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

        Expression strings support: field access (``datum.field``, bracket
        access ``datum["field with spaces"]``), arithmetic (``+ - * / %``),
        comparison and logical operators (``&& || !``), ternary
        (``cond ? a : b``), string/boolean/null literals (``'x'``, ``true``,
        ``false``, ``null``), scientific notation (``1e10``), array literals
        (``[1, 2, 3]``), and one builtin call, ``indexof(array, value)``
        (returns the index of the first match, or ``-1``) — the full table
        lives in the data-transforms guide's "Expression syntax" section.
        A dict predicate's list/tuple values (``{"field": [1, 2, 3]}``)
        compile to exactly this ``indexof([...], datum.field) >= 0`` form.

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

    >>> t = fm.transform_filter("indexof([1, 3, 5], datum.id) >= 0")
    >>> t["predicate"]
    'indexof([1, 3, 5], datum.id) >= 0'

    >>> brush = fm.selection_interval(name="brush")
    >>> fm.transform_filter(brush)
    {'type': 'filter', 'predicate': 'true', 'param': 'brush'}
    """
    from ferrum.parameter import Parameter, _decode_param_ref

    if isinstance(predicate, Parameter):
        return {"type": "filter", "predicate": "true", "param": predicate.name}
    # A `.ref()` marker (F-L13-01) is the by-name spelling of the same
    # reference -- checked BEFORE the constraint-dict branch below, which
    # would otherwise silently mangle `{"param": "qmax"}` into the garbage
    # predicate `datum.param == "qmax"` (a dict with a "param" key looks,
    # to the constraint-dict branch, like a field named "param" constrained
    # to the string "qmax").
    ref_name = _decode_param_ref(predicate)
    if ref_name is not None:
        return {"type": "filter", "predicate": "true", "param": ref_name}
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
        Expression string (e.g. ``"datum.x * 2 + datum.y"``). Supports the
        same grammar as ``transform_filter``'s expression-string form
        (arithmetic, comparisons, ternary, literals, ``indexof(...)``) — see
        the data-transforms guide's "Expression syntax" section. An
        expression whose *overall* result is an array (rather than a
        number, string, or boolean) raises ``ValueError``; arrays are only
        valid as an ``indexof(...)`` argument, not as a column value.

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
        individual characters, not column names — or names the same column
        more than once.

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
        groupby_list = list(groupby)
        _reject_repeated_groupby_names(groupby_list, where="transform_aggregate")
        spec["groupby"] = groupby_list
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
        individual characters, not column names — or names the same column
        more than once.

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
        groupby_list = list(groupby)
        _reject_repeated_groupby_names(groupby_list, where="transform_pivot")
        spec["groupby"] = groupby_list
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
    frame: Sequence[int | None] | None = None,
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
    frame : pair of (int or None), optional
        Window frame bounds: (preceding, following), a tuple, list, ``range`` or
        1-D numpy array of two ends. None means unbounded.
        Each end is None or an integer a 64-bit row offset holds; anything
        else raises ``ValueError`` at construction, and an offset past the
        batch reads to its edge.
    ignore_peers : bool, default False
        Whether a rolling frame ignores *peers* — rows tied on every ``sort``
        column. Left ``False`` (the same default *setting* Vega chose), a
        frame expands to whole peer groups, so tied rows share one frame and
        therefore one aggregate value. Note the peer-aware frame semantics
        themselves deliberately diverge from Vega's: both expand a frame's
        ends outward to whole peer groups, but ferrum *anchors* the frame on
        the peer run's own bounds where Vega anchors on the current row, so
        ferrum guarantees tied rows share one frame and one value while
        Vega's need not (see the guide's window-functions section for a
        worked example). Set ``True`` for
        frames defined by row offsets alone. Only rolling aggregates with an
        explicit ``frame`` over a tied sort key are affected.

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
        spec["frame"] = coerce_frame("transform_window", frame)
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
        individual characters, not column names — or names the same column
        more than once.

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
        groupby_list = list(groupby)
        _reject_repeated_groupby_names(groupby_list, where="transform_density")
        spec["groupby"] = groupby_list
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
        Regression method; see
        ``transform_registry("regression_data")["methods"]`` for the
        accepted vocabulary.
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


_IMPUTE_KINDS = "string, number, bool, date or datetime"

#: The registry params of ``impute`` that are not options of the impute rule: the
#: transform's own columns and its output name. Every other param is an option
#: both doors accept.
_IMPUTE_NOT_OPTIONS = frozenset({"field", "groupby", "key", "name"})


def _finite_real(label: str, value: object) -> int | float:
    """*value* in its wire form, refused in *label*'s voice unless it is a finite real number."""
    return cast("int | float", coerce_finite_real(label, value, required=True))


def _temporal_epoch_ms(label: str, value: object) -> dict[str, int] | None:
    """*value* as a tagged instant ``{"epoch_ms": n}`` when it is a date, datetime or numpy ``datetime64``, else ``None``.

    The tag tells the engine this number is an instant, which a temporal column
    or key reads as that time and a numeric, text or boolean one refuses, where
    a bare number is just a number. The milliseconds come from
    :func:`ferrum.annotation.coords.temporal_to_whole_epoch_ms`, which is exact
    integer arithmetic. The wire holds a whole millisecond, so a value finer
    than that is refused (not rounded), as is a not-a-time (``NaT``), in
    *label*'s voice.
    """
    if not isinstance(value, (datetime.date, np.datetime64)):
        return None
    from ferrum.annotation.coords import _SubMillisecondError, temporal_to_whole_epoch_ms

    try:
        millis = temporal_to_whole_epoch_ms(value)
    except _SubMillisecondError:
        raise ValueError(
            f"{label}: {value!r} is finer than a millisecond, which an impute value or key "
            "cannot carry; round it to a whole millisecond"
        ) from None
    except ValueError as exc:
        raise ValueError(f"{label}: {exc}") from exc
    return {"epoch_ms": millis}


def _impute_literal(label: str, value: object) -> bool | int | float | str | dict[str, int] | None:
    """Normalize one impute literal (a ``value=`` or a ``keyvals`` element) to its wire form.

    The one dispatch both options read: a bool or ``numpy.bool_`` is a bool, a
    string is kept, a date, datetime or numpy ``datetime64`` is a tagged instant
    (:func:`_temporal_epoch_ms`), and a real number is its finite wire form
    (:func:`ferrum._validate.coerce_finite_real`). Anything else returns
    ``None`` for the caller to refuse in its own words. Which of these a column
    or key can hold is the engine's to check, since only it sees the dtype.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value
    instant = _temporal_epoch_ms(label, value)
    if instant is not None:
        return instant
    if is_real_number(value):
        return _finite_real(label, value)
    return None


def _normalize_keyval(
    value: object, *, key: str | None, door: str = "transform_impute"
) -> bool | int | float | str | dict[str, int]:
    """Normalize one ``keyvals`` element to a plain, JSON-serializable wire literal.

    A numpy scalar from ``df[col].unique()``, the routine way to build a
    ``keyvals`` list off the key column itself, normalizes the same way its
    Python-native twin does, through :func:`_impute_literal`. *door* is the
    public call the refusal speaks as (``"Y(impute=...)"`` on the encoding dict).

    Raises
    ------
    ValueError
        If the element is of no kind a key holds, or is a number that is not
        finite, or an instant finer than a millisecond.
    """
    where = f" for key {key!r}" if key is not None else ""
    literal = _impute_literal(f"{door}: keyvals value{where}", value)
    if literal is None:
        raise ValueError(
            f"{door}: {type(value).__name__} {value!r} not allowed in "
            f"keyvals{where}; accepted: {_IMPUTE_KINDS}"
        )
    return literal


def _normalize_impute_value(label: str, value: object) -> bool | int | float | str | dict[str, int]:
    """Normalize an impute ``value=`` to the literal that rides the wire (see :func:`_impute_literal`).

    *label* is the pre-composed door (``"transform_impute(value=...)"``).
    """
    literal = _impute_literal(label, value)
    if literal is None:
        raise ValueError(
            f"{label} must be a number, a string, True or False, a date or a datetime, "
            f"not {type(value).__name__}; received {value!r}"
        )
    return literal


def _impute_snapshot(options: object) -> object:
    """Copy an impute option dict so the caller cannot change or drain it afterwards.

    A channel keeps the dict its caller wrote, in the caller's spelling, so this
    copy is what it stores: a fresh dict. ``keyvals`` that is a ``dict`` (the
    ``{'start', 'stop', 'step'}`` form) is copied; any other ``keyvals`` that is
    not a string or bytes is read once into a list (a generator, ``map``, numpy
    array or Series would otherwise be drained by the first validation). A
    ``frame`` list is copied and a 1-D numpy array of two integers becomes a tuple; a
    tuple or ``range`` is immutable and kept, and anything else is left as it is
    and never read, for the validation to refuse. Validation, every rebuild and
    the emission then read the same snapshot. A value that is not a dict is
    returned as it is, for the validation to refuse; the caller's own dict is
    never changed.
    """
    if not isinstance(options, dict):
        return options
    snapshot = dict(options)
    keyvals = snapshot.get("keyvals")
    if isinstance(keyvals, dict):
        snapshot["keyvals"] = dict(keyvals)
    elif keyvals is not None and not isinstance(keyvals, (str, bytes)):
        try:
            snapshot["keyvals"] = list(keyvals)
        except TypeError:
            pass
    frame = snapshot.get("frame")
    if isinstance(frame, list):
        snapshot["frame"] = list(frame)
    elif isinstance(frame, np.ndarray) and frame.ndim == 1 and len(frame) == 2:
        ends = frame.tolist()
        if all(end is None or type(end) is int for end in ends):
            snapshot["frame"] = tuple(ends)
    return snapshot


def _impute_options(
    door: str, options: object, *, key: str | None = None, value_label: str
) -> dict[str, Any]:
    """Validate an impute option dict for ``transform_impute`` and the encoding dict.

    Wires :func:`ferrum._validate.coerce_impute_options` to the literal
    normalizers and to the option names the engine's registry publishes for
    ``impute`` (the registry's params that are options of the rule, in its
    order), so a ``Y(impute=...)`` and a ``transform_impute(...)`` refuse the
    same malformed option in the same words. *door* is the public call the
    refusals speak as, *key* the key column's name where the door has one, and
    *value_label* the label ``value`` is refused under.
    """
    from ferrum._core import transform_registry

    accepted = [
        name for name in transform_registry("impute")["params"] if name not in _IMPUTE_NOT_OPTIONS
    ]
    if isinstance(options, dict):
        _reject_bare_string(options.get("keyvals"), where=door, param="keyvals", noun="key values")
    return coerce_impute_options(
        door,
        options,
        accepted=accepted,
        normalize_value=lambda value: _normalize_impute_value(value_label, value),
        normalize_keyval=lambda keyval: _normalize_keyval(keyval, key=key, door=door),
    )


def transform_impute(
    field: str,
    *,
    method: str = "value",
    value: float | str | bool | datetime.date | np.datetime64 | None = None,
    groupby: Sequence[str] | None = None,
    key: str | None = None,
    keyvals: Sequence[Any] | dict[str, Any] | None = None,
    frame: Sequence[int | None] | None = None,
) -> dict:
    """Impute missing values in a column.

    A missing value is a null or ``NaN`` cell of ``field``, and, with ``key``,
    a row a group lacks for a key. Each is filled as ``method`` says. An
    infinite value is a value, not a gap, and is left as it is. The rule is
    Vega's keyed imputation, and the encoding dict ``Y(impute={...})`` takes the
    same options.

    Parameters
    ----------
    field : str
        Column to impute. Any numeric column is read and written as a float; a
        date, text or boolean column keeps its own dtype and takes a ``value``
        of its own kind.
    method : str, default "value"
        Imputation method; see ``transform_registry("impute")["methods"]``
        for the accepted vocabulary: ``"value"`` fills a constant, and
        ``"mean"``, ``"median"``, ``"min"`` and ``"max"`` fill the group's
        statistic over its usable (non-null, finite) values.
    value : float, str, bool, datetime.date or datetime.datetime, optional
        Constant fill for ``method="value"``, of the column's own kind: a
        number for a numeric column (``0`` when omitted), a string for a text
        column, ``True`` or ``False`` for a boolean column, and a
        ``datetime.date``, ``datetime.datetime``, ``numpy.datetime64`` or a
        number of epoch milliseconds for a date column. A date, text or boolean
        column has no default, so ``value`` is required there (omitting it
        raises ``ValueError`` at render time). An instant finer than a
        millisecond raises ``ValueError`` at construction. Given with another
        ``method`` it raises ``ValueError`` at render time.
    groupby : list of str, optional
        Impute within groups: the fill value (per ``method``) is computed
        separately for each group, not once across the whole column. A
        group with no usable (non-null, finite) values of its own to compute
        a statistic from falls back to the whole column's statistic, rather than
        degrading to ``0.0`` or an infinite fill value (unless the column
        as a whole has no usable values either).
    key : str, optional
        Key column. Within each group (or, with no ``groupby``, the whole
        batch as one implicit group — Vega's "no groupby means one group"
        rule), a key value in the key universe (see ``keyvals``) but missing
        from that group gets one imputed row: the group's own columns, the
        missing key and the group's fill; every other column is null. The row
        is inserted right after the group's last row whose key comes earlier in
        the key order, or before the group's first row when none does; imputed
        rows that land at one place keep key order among themselves, and a row
        the caller wrote never moves. The key order is ascending for a number or
        a date (``NaN`` last), ``False`` before ``True``, and for text the order
        each key first appears, ``keyvals`` first.
    keyvals : list, tuple or dict, optional
        Keys every group must hold, **added** to the keys the frame holds (the
        key universe is both). **Requires** ``key`` — passing ``keyvals``
        without ``key`` raises ``ValueError`` at render time, since there
        is no key column for the universe to apply to. Must be a sequence
        (list, tuple, or other iterable), never a bare string — a bare
        string iterates into characters, not key values, and raises
        ``ValueError`` at construction time. A ``datetime.date``,
        ``datetime.datetime`` or ``numpy.datetime64`` names an instant for a
        date key. A ``{"start": ..., "stop": ..., "step": ...}`` dict names the
        numbers from ``start`` (default ``0``) up to, not including, ``stop``
        by ``step`` (default ``1``, or ``-1`` when ``stop`` is below
        ``start``); ``stop`` is required, a zero ``step`` and a sequence on a
        text or boolean key raise ``ValueError`` at render time. Each value is
        typed against the key
        column's own dtype at render time: a whole-valued float (e.g.
        ``1.0``) coerces onto an integer key column, but a genuinely
        fractional float against an integer key column raises ``ValueError``
        (that coercion would be lossy); a string against a numeric key
        column raises the same way.
    frame : pair of (int or None), optional
        A tuple, list, ``range`` or 1-D numpy array of two ends.
        For ``mean``, ``median``, ``min`` and ``max``: the window each fill
        reads, ``(preceding, following)`` row offsets with ``None`` for an open
        end, counted in key order (in row order with no ``key``) with imputed
        rows counted as rows that hold no value. Default ``(None, None)``, the
        whole group. A window with no usable value falls back to the group's
        statistic, then the column's, each with a warning. Each end is ``None``
        or an integer a 64-bit row offset holds; anything else raises
        ``ValueError`` at construction, and an offset past the group reads to
        its edge. Given with ``method="value"`` it raises ``ValueError`` at
        render time.

    Returns
    -------
    dict
        Transform specification for the Rust engine.

    Raises
    ------
    ValueError
        If ``groupby`` or ``keyvals`` is a bare string — iterating a
        string yields individual characters, not column/key values; or if
        ``value``, ``keyvals`` or ``frame`` is malformed, or a literal (a
        ``value`` or a ``keyvals`` element) is of no accepted kind.

    Examples
    --------
    >>> import ferrum as fm
    >>> t = fm.transform_impute("sales", method="mean")
    >>> t["type"]
    'impute'
    >>> t["method"]
    'mean'
    >>> fm.transform_impute("sales", method="mean", key="month", frame=(-1, 1))["frame"]
    [-1, 1]
    """
    given = {"method": method, "value": value, "keyvals": keyvals, "frame": frame}
    options = _impute_options(
        "transform_impute",
        {name: option for name, option in given.items() if option is not None},
        key=key,
        value_label="transform_impute(value=...)",
    )
    spec: dict = {"type": "impute", "field": field, "method": options.get("method", "value")}
    if "value" in options:
        spec["value"] = options["value"]
    if groupby is not None:
        _reject_bare_string_groupby(groupby, where="transform_impute")
        spec["groupby"] = list(groupby)
    if key is not None:
        spec["key"] = key
    for option in ("keyvals", "frame"):
        if option in options:
            spec[option] = options[option]
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
        Aggregation operation; see ``transform_registry("top_k")["methods"]``
        for the accepted vocabulary.
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
