"""The registry-consumer analyzer: no Python consumer hardcodes a registry-declared
output column, checked by static analysis over ``src/ferrum``.

Moved out of ``tests/test_registry_output_column_consumers.py`` (`#164`,
wave 6B): an analyzer that grows every review round belongs in its own
module with its own unit tests, not embedded in the pytest file that
consumes it. This module carries the analyzer itself -- the AST plumbing,
:data:`ACCOUNTING`, and ferrum's own known dispositions and rechecked
populations (:data:`DISPOSITIONED`, :data:`KNOWN_NON_CHANNEL_KEYWORDS`,
:data:`KNOWN_CALL_ARGUMENT_CALLEES`). ``tests/test_registry_consumer_lint_unit.py``
carries the synthetic-module RED proof and one fixture per accounting rule;
``tests/test_registry_output_column_consumers.py`` is now the thin wrapper
plus the behavioral pin, consuming this module's :func:`run` and public
constants without holding any project-specific data of its own.

The population is enumerated by the machine, from the outside in.

**The outer bound.** Every ``ast.Constant`` string in ``src_root/**.py``
whose value equals an output-column name that some registry kind declares --
in its primary ``outputs`` or in any of its ``secondary_outputs`` batches
(:func:`run`). Both halves are derived: the names come from the live
registry, the occurrences from parsing the package. Nothing about ferrum's
shapes, conventions or call sites narrows it. It is the widest set that could
possibly contain an unconverted consumer, because an unconverted consumer *is*
a string literal equal to a declared output name.

**The accounting.** Every member of that outer bound is assigned exactly one
label from the ordered rule table :data:`ACCOUNTING`. Two labels mean "this
is a real reference to a transform's output column":

``converted-argument`` / ``converted-lookup-read``
    The literal is an argument of a ``registry_output_columns(kind, [...])``
    or ``registry_secondary_columns(kind, key, [...])`` call, or the key a
    call site reads back out of one -- the M6a conversion, checked against
    the live registry at call time.

``column-reference``
    The literal sits in a *channel position* -- somewhere a column name is
    what the value means. Every member must either be converted or carry a
    per-site entry in :data:`DISPOSITIONED` explaining why it is not a
    registry-declared output despite colliding with one.

The remaining labels are the non-column positions, each with its
justification recorded in :data:`ACCOUNTING`.

What this proves, and what it does not
---------------------------------------
Proved: no string constant equal to a declared output column name sits in a
syntactic channel position anywhere in the scanned tree without being either
registry-checked or dispositioned with a reason; and every other occurrence
of such a constant is accounted for by a named rule.

Not proved: that no column name reaches a channel through *inter-procedural*
data flow -- a constant returned from a helper, threaded through a caller's
argument, or assembled from pieces. :func:`run`'s internal name resolution
follows local names, ternaries and containers within a function and its
enclosing scopes -- including a chained ``a = b = value`` assignment, a
same-arity tuple/list unpack (``lo, hi = "x_lo", "x_hi"``), and a
``for name in (...)`` loop over a literal container -- but it does not follow
values across a *function* boundary, through a loop over a *module-level*
constant, or through a comprehension-built encoding; such a constant is
still in the outer bound and still accounted, but by its own syntactic
position rather than as a column reference. Establishing that would need a
dataflow analysis Python does not hand us.
"""

from __future__ import annotations

import ast
import functools
import inspect
import pathlib
from collections import Counter
from dataclasses import dataclass
from typing import NamedTuple

from ferrum import _core
import ferrum.encoding as _encoding
from ferrum.encoding.base import ChannelBase

#: The two conversion points, both in ``ferrum._registry``. The primary helper
#: takes ``(kind, names)``; the secondary one takes ``(kind, key, names)`` --
#: the names list is the last positional argument of either.
PRIMARY_LOOKUP = "registry_output_columns"
SECONDARY_LOOKUP = "registry_secondary_columns"
_LOOKUP_HELPERS = frozenset({PRIMARY_LOOKUP, SECONDARY_LOOKUP})

#: Keyword arguments whose value is a *container* of channel bindings rather
#: than a single one. A dict here is keyed by channel name; a call here builds
#: such a dict, so its arguments are channel bindings too.
_ENCODING_CONTAINERS = frozenset({"encoding", "mark_kwargs", "remap"})

#: Keyword arguments that are not themselves an encoding-channel name (so
#: ``_channel_vocabulary()``'s reflection over ``ferrum.encoding`` does not
#: find them) but whose value a consumer treats exactly like one -- it ends
#: up bound to a real channel downstream. ``_Layer.contrast_against``
#: is one: ``_build_layers_list`` assigns its value straight into the
#: layer's own ``color`` encoding when no caller fill reached the layer, so
#: a literal here names a column precisely as a ``color=`` keyword would.
_CHANNEL_EQUIVALENT_KEYWORDS = frozenset({"contrast_against"})


# ---------------------------------------------------------------------------
# Inputs derived from the live registry and the live encoding package
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _registry_kinds() -> tuple[str, ...]:
    """Every kind the Rust registry knows, taken from its own refusal message.

    ``transform_registry`` builds that list from ``all_transform_kinds()``, so
    this tracks the registry rather than a Python copy of it. There is no
    dedicated enumeration binding today; if the message shape ever changes,
    the ``transform_registry(kind)`` round-trip below fails loudly instead of
    yielding a quietly empty list.
    """
    try:
        _core.transform_registry("__not_a_kind__")
    except ValueError as exc:
        _, _, listed = str(exc).partition("known kinds: ")
        kinds = tuple(k.strip() for k in listed.split(",") if k.strip())
    else:  # pragma: no cover - the binding is contractually required to raise
        raise AssertionError("transform_registry accepted an unknown kind")
    assert len(kinds) > 30, f"suspiciously few kinds parsed from the refusal: {kinds}"
    for kind in kinds:
        _core.transform_registry(kind)  # raises if the parse produced garbage
    return kinds


@functools.lru_cache(maxsize=None)
def declared_output_names() -> frozenset[str]:
    """Union of every output column name any registry kind declares.

    Both halves count: a kind's primary ``outputs`` and the columns of every
    named batch in its ``secondary_outputs``.
    """
    names: set[str] = set()
    for kind in _registry_kinds():
        entry = _core.transform_registry(kind)
        names |= {name for name, _dtype in entry["outputs"]}
        for _key, columns in entry["secondary_outputs"]:
            names |= {name for name, _dtype in columns}
    return frozenset(names)


@functools.lru_cache(maxsize=None)
def _channel_vocabulary() -> tuple[frozenset[str], frozenset[str]]:
    """(channel class names, channel names) taken from :mod:`ferrum.encoding`.

    Derived rather than hardcoded so a new channel is swept the day it is
    added: ``X("bin_start")`` and ``{"x": "bin_start"}`` are both channel
    positions, and both halves of that come from the class objects.
    """
    classes: set[str] = set()
    names: set[str] = set()
    for obj in vars(_encoding).values():
        if inspect.isclass(obj) and issubclass(obj, ChannelBase):
            classes.add(obj.__name__)
            channel = getattr(obj, "_channel_name", None)
            if channel:
                names.add(channel)
    assert "x" in names and "X" in classes, "ferrum.encoding exposed no channel classes"
    return frozenset(classes), frozenset(names)


#: ``ch.__class__(col, **kwargs)`` is the rebuild-the-same-channel-type idiom
#: (``_encode_lowering._rebind_channel``); its first positional argument
#: is a column name by construction, exactly as ``X(col)``'s is.
def _channel_callees() -> frozenset[str]:
    classes, _names = _channel_vocabulary()
    return classes | {"__class__"}


# ---------------------------------------------------------------------------
# AST plumbing
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _parse(path: pathlib.Path) -> ast.Module:
    """Parse once: several passes below walk the whole package."""
    return ast.parse(path.read_text())


def _callee_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _local_aliases(tree: ast.AST, names: frozenset[str] | set[str]) -> set[str]:
    """*names* plus every local alias bound to one of them by an import.

    ``from ferrum.encoding import X as _X_ecdf`` must not hide a channel.
    """
    aliases = set(names)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in names and alias.asname:
                    aliases.add(alias.asname)
    return aliases


def _enclosing_functions(tree: ast.AST) -> dict[ast.AST, ast.AST | None]:
    """node -> the innermost function containing it (``None`` at module level).

    Needed because modules reuse local names: three functions in
    ``marks/statistical.py`` each build a local ``encoding_remap`` dict, so a
    module-wide name map would resolve all three to whichever was defined last
    and silently skip the other two.
    """
    enclosing: dict[ast.AST, ast.AST | None] = {tree: None}

    def walk(node: ast.AST, current: ast.AST | None) -> None:
        for child in ast.iter_child_nodes(node):
            enclosing[child] = current
            inner = child if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else current
            walk(child, inner)

    walk(tree, None)
    return enclosing


class _ModuleIndex(NamedTuple):
    tree: ast.Module
    parents: dict[int, ast.AST]
    enclosing: dict[ast.AST, ast.AST | None]
    #: (enclosing function, local name) -> the expression it is bound to.
    #: Populated for a plain ``name = value`` assignment, each target of a
    #: chained ``a = b = value`` assignment, each element of a same-arity
    #: tuple/list unpack (``lo, hi = "x_lo", "x_hi"``), and a ``for name in
    #: (...)`` loop over a literal tuple/list/set -- bound to the whole
    #: container, so a channel-position use resolves to every element the
    #: loop variable could hold across iterations.
    bindings: dict[tuple[ast.AST | None, str], ast.expr]
    #: local names bound to a ``registry_output_columns(...)`` result.
    lookup_names: set[str]
    channel_callees: set[str]


def _index(path: pathlib.Path) -> _ModuleIndex:
    tree = _parse(path)
    parents: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    enclosing = _enclosing_functions(tree)
    bindings: dict[tuple[ast.AST | None, str], ast.expr] = {}
    lookup_names: set[str] = set()

    def bind(scope: ast.AST | None, name: str, value: ast.expr) -> None:
        bindings[(scope, name)] = value
        if isinstance(value, ast.Call) and _callee_name(value) in _LOOKUP_HELPERS:
            lookup_names.add(name)

    for node in ast.walk(tree):
        scope = enclosing.get(node)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    # Plain assignment, or one arm of a chained
                    # ``a = b = value`` -- every target of a multi-target
                    # Assign shares the same value node.
                    bind(scope, target.id, node.value)
                elif (
                    isinstance(target, (ast.Tuple, ast.List))
                    and isinstance(node.value, (ast.Tuple, ast.List))
                    and len(target.elts) == len(node.value.elts)
                    and not any(isinstance(elt, ast.Starred) for elt in target.elts)
                ):
                    # Same-arity tuple/list unpack: bind each name to its
                    # positional counterpart (``lo, hi = "x_lo", "x_hi"``).
                    for elt_target, elt_value in zip(target.elts, node.value.elts):
                        if isinstance(elt_target, ast.Name):
                            bind(scope, elt_target.id, elt_value)
        elif (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and isinstance(node.iter, (ast.Tuple, ast.List, ast.Set))
        ):
            # ``for name in ("q1", "q3"): ...`` -- bind to the whole
            # container; a channel-position use of ``name`` then flattens
            # to every element via the container-resolution case below.
            bind(scope, node.target.id, node.iter)
    return _ModuleIndex(
        tree=tree,
        parents=parents,
        enclosing=enclosing,
        bindings=bindings,
        lookup_names=lookup_names,
        channel_callees=_local_aliases(tree, _channel_callees()),
    )


def _static_str(node: ast.expr, index: _ModuleIndex, owner: ast.AST | None) -> str | None:
    """*node*'s string value if it is statically knowable, else ``None``.

    A string literal, or a name bound to one anywhere in the enclosing scope
    chain (``_QQ_LINE_KEY = "qq_line"`` at module level, ``_segments_key =
    "segments"`` in the enclosing function). Used to read the ``kind`` and
    ``key`` arguments of a lookup call, which call sites bind to a named
    constant when the same string also has to build a ``data_source``.
    """
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.Name):
        scope = owner
        while True:
            bound = index.bindings.get((scope, node.id))
            if bound is not None:
                return _static_str(bound, index, owner) if bound is not node else None
            if scope is None:
                return None
            scope = index.enclosing.get(scope)
    return None


# ---------------------------------------------------------------------------
# The two rules that say "this literal names a transform output column"
# ---------------------------------------------------------------------------


def _converted_nodes(index: _ModuleIndex) -> tuple[set[int], set[int]]:
    """(argument nodes, lookup-read nodes) of the M6a conversion.

    An *argument* node is a name inside a lookup call's names list --
    ``registry_output_columns(kind, [...])`` or
    ``registry_secondary_columns(kind, key, [...])``, the point where the
    registry check happens. The names list is the last positional argument of
    either helper. A *lookup-read* node is the key a call site reads back,
    either out of a local bound to that call (``_bin_cols["bin_start"]``) or
    straight off it
    (``registry_output_columns("bin", ["bin_start"])["bin_start"]``).
    """
    arguments: set[int] = set()
    reads: set[int] = set()
    for node in ast.walk(index.tree):
        if isinstance(node, ast.Call) and _callee_name(node) in _LOOKUP_HELPERS:
            if node.args:
                for elt in getattr(node.args[-1], "elts", []):
                    if isinstance(elt, ast.Constant):
                        arguments.add(id(elt))
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            base = node.value
            from_name = isinstance(base, ast.Name) and base.id in index.lookup_names
            from_call = isinstance(base, ast.Call) and _callee_name(base) in _LOOKUP_HELPERS
            if from_name or from_call:
                reads.add(id(node.slice))
    return arguments, reads


def _column_reference_nodes(index: _ModuleIndex, channel_names: frozenset[str]) -> set[int]:
    """Every constant that sits in a *channel position*.

    A channel position is one of:

    * a value in a dict entry whose key is an encoding channel name --
      ``{"x": "x_lo"}`` -- wherever that dict is written, including a dict
      bound to a local and a dict chosen by a ternary;
    * the first positional argument of an encoding-channel constructor --
      ``X("bin_start")``, ``ch.__class__("bin_start")`` -- under any import
      alias;
    * the value of a keyword argument named after a channel --
      ``.encode(y=Y("sample"))``, ``.encode(x="x_lo")``, ``Smooth(x="x")`` --
      or named in :data:`_CHANNEL_EQUIVALENT_KEYWORDS`, a keyword that is not
      itself a channel name but whose value a consumer binds to one
      downstream (``_Layer(contrast_against="value")``);
    * the value assigned to a channel-keyed subscript -- ``enc["y"] = "count"``;
    * any argument of a call passed as ``encoding=``/``mark_kwargs=``/``remap=``,
      since such a call builds the channel dict (``encoding=enc("q1")``).

    Resolution follows local names, ternary branches and containers within the
    enclosing function chain, and additionally resolves a chained
    ``a = b = value`` assignment, a same-arity tuple/list unpack, and a
    ``for name in (...)`` loop over a literal container.
    """
    marked: set[int] = set()

    def mark(node: ast.expr, owner: ast.AST | None, seen: set[int]) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                marked.add(id(node))
        elif isinstance(node, ast.IfExp):
            mark(node.body, owner, seen)
            mark(node.orelse, owner, seen)
        elif isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            for elt in node.elts:
                mark(elt, owner, seen)
        elif isinstance(node, ast.Name):
            bound = _resolve(node, owner)
            if bound is not None:
                mark(bound, owner, seen)
        elif isinstance(node, ast.Call) and _callee_name(node) in index.channel_callees:
            if node.args:
                mark(node.args[0], owner, seen)

    def mark_container(node: ast.expr, owner: ast.AST | None, seen: set[int]) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        if isinstance(node, ast.Dict):
            mark_dict(node, owner)
        elif isinstance(node, ast.IfExp):
            mark_container(node.body, owner, seen)
            mark_container(node.orelse, owner, seen)
        elif isinstance(node, ast.Name):
            bound = _resolve(node, owner)
            if bound is not None:
                mark_container(bound, owner, seen)
        elif isinstance(node, ast.Call):
            for arg in node.args:
                mark(arg, owner, set())
            for kw in node.keywords:
                mark(kw.value, owner, set())

    def mark_dict(node: ast.Dict, owner: ast.AST | None) -> None:
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and key.value in channel_names:
                mark(value, owner, set())

    def _resolve(node: ast.Name, owner: ast.AST | None) -> ast.expr | None:
        scope = owner
        while True:
            bound = index.bindings.get((scope, node.id))
            if bound is not None:
                return bound
            if scope is None:
                return None
            scope = index.enclosing.get(scope)

    for node in ast.walk(index.tree):
        owner = index.enclosing.get(node)
        if isinstance(node, ast.Dict):
            mark_dict(node, owner)
        elif isinstance(node, ast.Call):
            if _callee_name(node) in index.channel_callees and node.args:
                mark(node.args[0], owner, set())
            for kw in node.keywords:
                if kw.arg in channel_names or kw.arg in _CHANNEL_EQUIVALENT_KEYWORDS:
                    mark(kw.value, owner, set())
                elif kw.arg in _ENCODING_CONTAINERS:
                    mark_container(kw.value, owner, set())
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value in channel_names
                ):
                    mark(node.value, owner, set())
    return marked


# ---------------------------------------------------------------------------
# The accounting: one label per outer-bound occurrence
# ---------------------------------------------------------------------------

#: Ordered ``(label, justification)``. The first two labels mean "this is a
#: real output-column reference"; the rest are the positions in which a string
#: that happens to equal a declared output name cannot be one. Order matters:
#: the column-meaning rules are tried first, so a channel position is never
#: swallowed by a syntactic rule beneath it.
ACCOUNTING: tuple[tuple[str, str], ...] = (
    (
        "converted-argument",
        "A name passed to registry_output_columns(kind, [...]) or to "
        "registry_secondary_columns(kind, key, [...]), which checks it against "
        "transform_registry(kind)['outputs'] -- or against that kind's declared "
        "schema for the named secondary batch -- at call time. This is the M6a "
        "conversion itself.",
    ),
    (
        "converted-lookup-read",
        "The key a call site reads back out of a registry_output_columns result. "
        "Its validity was established by the call that produced the lookup.",
    ),
    (
        "column-reference",
        "The literal sits in a channel position, so it names a data column. Must "
        "be converted or carry a reasoned DISPOSITIONED entry.",
    ),
    (
        "mapping-key",
        "A dict key or a subscript index that is not a registry-checked lookup: "
        "it addresses a Python mapping -- an encoding-channel map, a mark-kwargs "
        "or theme/config property map, a polars rename map. Transform output "
        "batches exist only inside Rust; Python's sole contact with them is "
        "*naming* a column in a channel position, which the rule above covers.",
    ),
    (
        "vocabulary-member",
        "An element of a tuple/list/set display: a validation vocabulary "
        "(_VALID_JITTER_AXES), an iteration list (for axis in ('x', 'y')), an "
        "__all__ entry, a Literal[...] member, or a transform wrapper's as_= "
        "default. The as_= defaults *declare* the column names the transform "
        "shall emit -- Rust honours them verbatim -- rather than read names it "
        "chose.",
    ),
    (
        "predicate-operand",
        "An operand of ==/!=/in: a test against an axis, channel, aggregate or "
        "mask vocabulary, never a column read.",
    ),
    (
        "non-channel-keyword",
        "The value of a keyword argument whose name is neither an encoding "
        "channel nor an encoding container: align='right', axis='x', how='left', "
        "name='median', title='sample', value_name='value'. The parameter fixes "
        "what the string means, and none of these takes a column name out of a "
        "transform batch (value_name= declares an Unpivot output name).",
    ),
    (
        "non-channel-dict-value",
        "A dict value under a non-channel key: a mark-kwargs property "
        "({'align': 'right'}), a wire-spec field ({'type': 'sample'}), a rename "
        "target ({'len': 'count'}).",
    ),
    (
        "call-argument",
        "A positional argument of a call that is not an encoding-channel "
        "constructor: dict.get('x'), pl.col('value'), pl.Series('y', ...), "
        "df.group_by('y'), AggregateOp(field, 'count', 'n'). Python only ever "
        "holds frames it materialised itself, so a column name here belongs to "
        "such a frame, not to a transform's output batch.",
    ),
    (
        "local-binding",
        "The value of a plain assignment, an annotated assignment, or a "
        "parameter default that does not flow into a channel position -- "
        "_channel_name = 'x', aggregate: str = 'count'. Bindings that DO reach a "
        "channel are resolved into column-reference above.",
    ),
    ("message-text", "A piece of an f-string: prose, not a column name."),
)

LABELS = tuple(label for label, _why in ACCOUNTING)


class Occurrence(NamedTuple):
    path: str
    lineno: int
    function: str
    literal: str
    label: str
    #: For ``non-channel-keyword``, the keyword parameter name (``"align"``);
    #: for ``call-argument``, the callee name (``"col"``). Empty otherwise.
    detail: str = ""

    @property
    def site(self) -> tuple[str, str, str]:
        """The DISPOSITIONED key: stable under edits elsewhere in the file."""
        return (self.path, self.function, self.literal)


def _classify_module(
    path: pathlib.Path, rel: str, declared_names: frozenset[str], channel_names: frozenset[str]
) -> list[Occurrence]:
    index = _index(path)
    converted_args, converted_reads = _converted_nodes(index)
    column_refs = _column_reference_nodes(index, channel_names)

    dict_keys: set[int] = set()
    subscript_indices: set[int] = set()
    for node in ast.walk(index.tree):
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant):
                    dict_keys.add(id(key))
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            subscript_indices.add(id(node.slice))

    occurrences: list[Occurrence] = []
    for node in ast.walk(index.tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if node.value not in declared_names:
            continue
        parent = index.parents.get(id(node))
        if id(node) in converted_args:
            label = "converted-argument"
        elif id(node) in converted_reads:
            label = "converted-lookup-read"
        elif id(node) in column_refs:
            label = "column-reference"
        elif id(node) in dict_keys or id(node) in subscript_indices:
            label = "mapping-key"
        elif isinstance(parent, (ast.Tuple, ast.List, ast.Set)):
            label = "vocabulary-member"
        elif isinstance(parent, ast.Compare):
            label = "predicate-operand"
        elif isinstance(parent, ast.keyword):
            label = "non-channel-keyword"
        elif isinstance(parent, ast.Dict):
            label = "non-channel-dict-value"
        elif isinstance(parent, ast.Call):
            label = "call-argument"
        elif isinstance(parent, (ast.Assign, ast.AnnAssign, ast.arguments, ast.IfExp)):
            label = "local-binding"
        elif isinstance(parent, ast.JoinedStr):
            label = "message-text"
        else:
            label = f"UNACCOUNTED({type(parent).__name__})"
        detail = ""
        if label == "non-channel-keyword":
            detail = parent.arg or ""
        elif label == "call-argument":
            detail = _callee_name(parent) or ""
        owner = index.enclosing.get(node)
        occurrences.append(
            Occurrence(
                path=rel,
                lineno=node.lineno,
                function=getattr(owner, "name", "<module>"),
                literal=node.value,
                label=label,
                detail=detail,
            )
        )
    return occurrences


def _converted_columns(
    src_root: pathlib.Path,
) -> tuple[dict[str, frozenset[str]], dict[tuple[str, str], frozenset[str]]]:
    """The two conversion inventories, collected from the source.

    Returns ``(primary, secondary)``: ``kind -> names`` for every
    ``registry_output_columns(kind, [...])`` call site, and
    ``(kind, key) -> names`` for every
    ``registry_secondary_columns(kind, key, [...])`` one.

    A kind or key argument is read through :func:`_static_str`, which sees a
    literal or a name bound to one (``_QQ_LINE_KEY``, ``_segments_key``); a
    call whose kind or key is neither is skipped -- there is one such site,
    ``marks/_fit_layers.build_fit_layers``' ``kind`` parameter, whose four
    possible arguments (the fit family) are pinned by
    ``tests/test_lmplot_valid_methods_registry_pin.py`` instead.
    """
    primary: dict[str, set[str]] = {}
    secondary: dict[tuple[str, str], set[str]] = {}
    for path in sorted(src_root.rglob("*.py")):
        index = _index(path)
        for node in ast.walk(index.tree):
            if not isinstance(node, ast.Call):
                continue
            callee = _callee_name(node)
            if callee not in _LOOKUP_HELPERS:
                continue
            arity = 2 if callee == PRIMARY_LOOKUP else 3
            if len(node.args) != arity:
                continue
            owner = index.enclosing.get(node)
            leading = [_static_str(arg, index, owner) for arg in node.args[:-1]]
            if any(value is None for value in leading):
                continue
            names = {
                elt.value
                for elt in getattr(node.args[-1], "elts", [])
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            }
            if callee == PRIMARY_LOOKUP:
                primary.setdefault(leading[0], set()).update(names)
            else:
                secondary.setdefault((leading[0], leading[1]), set()).update(names)
    return (
        {kind: frozenset(names) for kind, names in primary.items()},
        {site: frozenset(names) for site, names in secondary.items()},
    )


@dataclass(frozen=True)
class Findings:
    """The outer bound, classified, plus the derived conversion inventories.

    ``outer_bound`` is every string constant in the scanned tree equal to a
    declared registry output-column name, each assigned exactly one label.
    ``converted``/``converted_secondary`` are the drift-guard inventories
    used to cross-check the M6a conversion sites against the live registry.
    """

    outer_bound: tuple[Occurrence, ...]
    declared_names: frozenset[str]
    converted: dict[str, frozenset[str]]
    converted_secondary: dict[tuple[str, str], frozenset[str]]

    @property
    def literal_set(self) -> frozenset[str]:
        return frozenset(o.literal for o in self.outer_bound)

    @property
    def path_set(self) -> frozenset[str]:
        return frozenset(o.path for o in self.outer_bound)

    @property
    def label_counts(self) -> Counter[str]:
        return Counter(o.label for o in self.outer_bound)

    @property
    def column_references(self) -> tuple[Occurrence, ...]:
        return tuple(o for o in self.outer_bound if o.label == "column-reference")


def run(src_root: pathlib.Path) -> Findings:
    """The computable outer bound, classified, for every ``*.py`` under *src_root*.

    *src_root* must sit two directories below the tree's own root (e.g.
    ``<root>/src/ferrum``), matching ferrum's own layout: each occurrence's
    ``path`` is recorded relative to ``src_root.parent.parent`` so paths read
    ``src/ferrum/...`` the way the sweep's dispositions key them.
    """
    declared_names = declared_output_names()
    _classes, channel_names = _channel_vocabulary()
    root = src_root.parent.parent
    found: list[Occurrence] = []
    for path in sorted(src_root.rglob("*.py")):
        found.extend(
            _classify_module(path, str(path.relative_to(root)), declared_names, channel_names)
        )
    primary, secondary = _converted_columns(src_root)
    return Findings(
        outer_bound=tuple(sorted(found)),
        declared_names=declared_names,
        converted=primary,
        converted_secondary=secondary,
    )


# ---------------------------------------------------------------------------
# Ferrum's own known dispositions and rechecked populations. These are
# project-specific knowledge about *this* codebase's consumers, not part of
# the reusable analyzer above -- kept here (rather than in the pytest
# wrapper) because they, too, are part of "the analyzer" the decision doc
# names for the move (live inputs, AST plumbing, reference rules,
# ACCOUNTING, DISPOSITIONED, membership rechecks).
# ---------------------------------------------------------------------------

_UNPIVOT_VALUE = (
    'The column is named by the `Unpivot(value_name="value")` built a few lines '
    "above, so Python chose it; `unpivot` declares no outputs at all. It collides "
    "with kde's and hex's declared `value`. Deriving it from the registry would "
    "assert a relationship that does not exist."
)
_RESIDPLOT_X = (
    "residplot's `_resid_x` reads the residuals batch `build_residuals_batch` "
    "emits (`smooth.rs:667`, `robust.rs:365`), which declares its own literal "
    '`x` column uncoupled from `FIT_OUTPUT_COLUMNS` -- the `output="fitted"` '
    "schema `transform_registry('smooth'|'robust')` actually declares "
    "(`registry/mod.rs:165-167` scopes that constant to fitted mode). Round 4 "
    "converted this site against `FIT_OUTPUT_COLUMNS` because both schemas "
    "happen to list `x`; that lookup would stay silently green if the residuals "
    "batch's own `x` were ever renamed independently, and would raise here on an "
    "unrelated `FIT_OUTPUT_COLUMNS` rename that never touched this code path -- "
    "a lookup against the wrong schema is worse than no lookup. Reverted to a "
    "bare literal (round 5). The registry declares no residuals-mode schema at "
    "all, so this class stays invisible to the outer bound above by "
    "construction; declaring one is the batch-close follow-up, not this task."
)

#: The letter_value and linkage secondary batches (`lv_depth_{k}` /
#: `lv_outliers`, `*_link_segments` / `*_link_order`) were dispositioned here
#: until 2026-09-06 on the grounds that the registry declared no schema to
#: check them against. It declares one now (`secondary_outputs`), so those six
#: sites are converted through `registry_secondary_columns` instead of
#: excused -- which is why no entry below names them.
DISPOSITIONED: dict[tuple[str, str, str], str] = {
    # --- a column Python itself materialised, colliding with a declared one -
    ("src/ferrum/marks/heavy_stat.py", "desugar_function", "x"): (
        "desugar_function's single-mark result carries `data=` -- a synthetic "
        "frame materialised in Python from the user's callable, with no transform "
        "in the chain. Its x/y are that frame's own column names, colliding with "
        "smooth's declared x/y."
    ),
    ("src/ferrum/marks/heavy_stat.py", "desugar_function", "y"): (
        "See the desugar_function x entry above."
    ),
    ("src/ferrum/marks/diagnostic/_selection.py", "desugar_learning_curve", "lower"): (
        "The learning-curve mark's documented data contract: train_size / split / "
        "mean_score / lower / upper as emitted by ModelSource.learning_curve(), a "
        "Python-built frame. `lower`/`upper` collide with error_extent's declared "
        "outputs, but no ErrorExtent transform runs here -- the CI columns are "
        "pre-computed (see the mark's docstring)."
    ),
    ("src/ferrum/marks/diagnostic/_selection.py", "desugar_learning_curve", "upper"): (
        "See the desugar_learning_curve lower entry above."
    ),
    ("src/ferrum/marks/diagnostic/_selection.py", "desugar_validation_curve", "lower"): (
        "Same contract as the learning curve: ModelSource.validation_curve() emits "
        "lower/upper itself; no error_extent transform is in the chain."
    ),
    ("src/ferrum/marks/diagnostic/_selection.py", "desugar_validation_curve", "upper"): (
        "See the desugar_validation_curve lower entry above."
    ),
    ("src/ferrum/marks/diagnostic/_clustering.py", "desugar_intercluster_distance", "x"): (
        "The intercluster-distance mark's data contract is a Python-built MDS/PCA "
        "embedding frame with literal x/y/size/cluster columns "
        "(ModelSource.intercluster_distance); they collide with smooth's declared "
        "x/y but no transform runs on this chart."
    ),
    ("src/ferrum/marks/diagnostic/_clustering.py", "desugar_intercluster_distance", "y"): (
        "See the desugar_intercluster_distance x entry above."
    ),
    ("src/ferrum/marks/diagnostic/_clustering.py", "desugar_decision_boundary", "x"): (
        "The decision-boundary mark's documented data contract: a pre-computed "
        "grid with x/x2/y/y2 cell bounds and z, produced by the chart builder in "
        "Python. Collides with smooth's declared x/y, and (since the secondary "
        "batches joined the registry) with linkage's dendrogram segments too."
    ),
    ("src/ferrum/marks/diagnostic/_clustering.py", "desugar_decision_boundary", "y"): (
        "See the desugar_decision_boundary x entry above."
    ),
    ("src/ferrum/marks/diagnostic/_clustering.py", "desugar_decision_boundary", "x2"): (
        "See the desugar_decision_boundary x entry above -- the same Python-built "
        "grid's far cell bound."
    ),
    ("src/ferrum/marks/diagnostic/_clustering.py", "desugar_decision_boundary", "y2"): (
        "See the desugar_decision_boundary x entry above."
    ),
    ("src/ferrum/marks/diagnostic/_ranking.py", "desugar_parallel_coordinates", "value"): (
        "The parallel-coordinates data contract is feature/value/sample_id from a "
        "Python-melted frame; `value` collides with kde's and hex's declared "
        "output of the same name."
    ),
    (
        "src/ferrum/marks/diagnostic/_classification.py",
        "desugar_discrimination_threshold",
        "value",
    ): (
        "Reads the long frame that _discrimination_threshold_chart_from_source "
        'builds with polars `unpivot(value_name="value")`; collides with kde\'s '
        "and hex's declared `value`."
    ),
    ("src/ferrum/marks/diagnostic/_classification.py", "desugar_class_prediction_error", "value"): (
        "Reads ModelSource.confusion_matrix()'s own `value` column (a Python-built "
        "frame), colliding with kde's and hex's declared `value`."
    ),
    ("src/ferrum/plots/classification.py", "_class_prediction_error_chart_from_source", "value"): (
        "The figure-level half of the same confusion-matrix frame as "
        "desugar_class_prediction_error above."
    ),
    (
        "src/ferrum/plots/classification.py",
        "_discrimination_threshold_chart_from_source",
        "value",
    ): (
        'The `value_name="value"` of the polars unpivot three lines above; the '
        "column is Python's own."
    ),
    ("src/ferrum/plots/classification.py", "_classification_report_chart", "value"): (
        "Reads the frame built by the shared "
        "`_curve_frames.classification_report_frame` helper (both the model "
        "and precomputed paths call it), whose rows carry literal "
        "metric/class/value/value_fmt keys."
    ),
    ("src/ferrum/plots/classification.py", "_class_balance_chart_from_dataframe", "count"): (
        "Reads the counts frame built two lines above by "
        '`.group_by("y").agg(...).rename({"len": "count"})`; the name is '
        "Python's, colliding with bin's and bin_2d's declared `count`."
    ),
    ("src/ferrum/plots/classification.py", "_class_balance_chart_from_dataframe", "y"): (
        "The class-label column of that same counts frame, colliding with smooth's declared `y`."
    ),
    ("src/ferrum/plots/clustering.py", "_intercluster_distance_chart_from_source", "x"): (
        "The figure-level half of desugar_intercluster_distance's contract: "
        "`source.intercluster_distance(...)` returns a frame with literal x/y "
        "columns, read here for the axis domain padding as well as the encoding."
    ),
    ("src/ferrum/plots/clustering.py", "_intercluster_distance_chart_from_source", "y"): (
        "See the _intercluster_distance_chart_from_source x entry above."
    ),
    ("src/ferrum/plots/matrix.py", "_heatmap_build", "value"): _UNPIVOT_VALUE,
    ("src/ferrum/plots/matrix.py", "_clustermap_build", "value"): _UNPIVOT_VALUE,
    # --- residuals batch, wrong-schema mispin reverted (round 5) -----------
    ("src/ferrum/plots/regression.py", "residplot", "x"): _RESIDPLOT_X,
}


def undispositioned(findings: Findings) -> list[Occurrence]:
    """Column references that are neither converted nor excused."""
    return [o for o in findings.column_references if o.site not in DISPOSITIONED]


def stale_dispositions(findings: Findings) -> list[tuple[str, str, str]]:
    """DISPOSITIONED entries no longer matched by any live column-reference hit."""
    live = {o.site for o in findings.column_references}
    return sorted(key for key in DISPOSITIONED if key not in live)


#: Population recheck for the two labels whose justification in ACCOUNTING is
#: an enumeration rather than a structural predicate -- both were "justified
#: by enumerating the population they currently match ... but nothing
#: re-checks membership." A keyword argument name or callee that was not
#: already confirmed safe must be re-justified before it can pass silently
#: into the bucket's "-class" hand-wave.
#:
#: `__setattr__` (`object.__setattr__(self, "x"/"y", ...)` in `AxisConfig`'s hand
#: `__init__`, `configure.py:412-413` at `a34f82c0`) left `KNOWN_CALL_ARGUMENT_CALLEES`
#: on `#149`: the generated dataclass constructor (`InitVar` aliases, wave 6B) deleted
#: that hand `__init__` outright, so no call-argument site with that callee exists in
#: the tree any more -- this is `test_call_argument_population_is_rechecked` doing its
#: job, not a false positive.
KNOWN_NON_CHANNEL_KEYWORDS = frozenset(
    {
        # Y(aggregate="count") -- names the encoding statistic's op, not a column.
        "aggregate",
        "align",  # e.g. mark_kwargs align="right" / a rule layer's text align
        "axis",  # e.g. jitter axis="x"
        "how",  # e.g. a join/merge how="left"
        "label_align",  # a text-mark label alignment kwarg, same class as align
        "name",  # e.g. a named transform or layer name="median"
        # _required_field_name(x, param="x") -- names the lmplot PARAMETER the
        # refusal message quotes, not a column: the column name is the value
        # being narrowed, passed positionally beside it.
        "param",
        "title",  # e.g. an axis/legend title="sample"
        "value_name",  # Unpivot(value_name="value") -- declares, does not read
    }
)
KNOWN_CALL_ARGUMENT_CALLEES = frozenset(
    {
        "Series",  # pl.Series("y", ...)
        "_channel_class_for",  # internal channel-name -> class lookup, not a column read
        "_extent_alignment",  # _extent_alignment(f, l, "x"|"y") -- an AXIS token, never a column
        "_extract_ordinal_domain",  # internal helper, positional field name
        "_refuse_statistic_input",  # _refuse_statistic_input(entry, "x"/"y", ch, reason=...) -- the channel SLOT, never a transform's output column read
        "_set_composite_mark",  # internal composite-mark metadata setter
        "_unique_col_name",  # _unique_col_name(df.columns, "count") -- reads the SOURCE frame's own columns, never a transform's output batch
        "append",  # list.append("count") building a column-name list
        "col",  # pl.col("value")
        "get",  # dict.get("x") / dict.get("theoretical")
        "getattr",  # getattr(obj, "count")
        "group_by",  # df.group_by("y")
        "pop",  # dict.pop("count")
        "sort",  # list.sort(key=...) target naming, or df.sort("x")
    }
)


def population_drift(
    observed: set[str], known: frozenset[str]
) -> tuple[frozenset[str], frozenset[str]]:
    """(new members not yet justified, known members no longer observed)."""
    return frozenset(observed - known), frozenset(known - observed)
