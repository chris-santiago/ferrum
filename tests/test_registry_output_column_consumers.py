"""Batch-D T12: no Python consumer hardcodes a registry-declared output column.

Spec §4.6 / decision M6a require that the output-column names a Python
desugar/figure consumer reads out of a transform's output batch derive from
``ferrum._core.transform_registry`` instead of being embedded as bare string
literals. ``ferrum.marks._desugar_helpers.registry_output_columns`` is the
conversion point; this module is the proof that the conversion is complete.

Why the *boundary* is computed, not stated
------------------------------------------
Four review cycles of this task each shipped a completeness claim backed by a
list a human maintained, and each was falsified by a wider look: a
"family-wide" groupby fix that missed ``transform_pivot``; a drift-guard table
that missed ``hex``'s ``value``; a conversion inventory that covered the seven
kinds a reviewer had named; and then a genuinely derived sweep whose *own
boundary sentence* ("the one such consumer in the tree today is converted") was
hand-counted and off by three. The lesson is not "look harder" -- it is that a
claim whose population is enumerated by a person is worth exactly as much as
that person's attention span.

So the population is enumerated by the machine, from the outside in:

**The outer bound.** Every ``ast.Constant`` string in ``src/ferrum/**.py``
whose value equals an output-column name that some registry kind declares
(:func:`_outer_bound`). Both halves are derived: the names come from the live
registry, the occurrences from parsing the package. Nothing about ferrum's
shapes, conventions or call sites narrows it. It is the widest set that could
possibly contain an unconverted consumer, because an unconverted consumer *is*
a string literal equal to a declared output name.

**The accounting.** Every member of that outer bound is assigned exactly one
label by :func:`_classify_module`, from the ordered rule table
:data:`_ACCOUNTING`.
``test_every_outer_bound_occurrence_is_accounted_for`` fails if any occurrence
matches no rule, so the tail cannot be silently dropped -- and
``test_every_accounting_rule_fires`` fails if any rule matches nothing, so a
rule cannot rot into decoration.

Two labels mean "this is a real reference to a transform's output column":

``converted-argument`` / ``converted-lookup-read``
    The literal is an argument of a ``registry_output_columns(kind, [...])``
    call, or the key a call site reads back out of one. This is the M6a
    conversion: the helper checks the name against the live registry, so a
    Rust-side rename raises at desugar time naming the stale column.

``column-reference``
    The literal sits in a *channel position* -- somewhere a column name is what
    the value means (see :func:`_column_reference_nodes`). Every member must
    either be converted or carry a per-site entry in :data:`_DISPOSITIONED`
    explaining why it is not a registry-declared output despite colliding with
    one. ``test_no_column_reference_is_a_bare_literal`` is the gate.

The remaining labels are the non-column positions, each with its justification
recorded in :data:`_ACCOUNTING` rather than in prose here.

What this proves, and what it does not
--------------------------------------
Proved: no string constant equal to a declared output column name sits in a
syntactic channel position anywhere in ``src/ferrum`` without being either
registry-checked or dispositioned with a reason; and every other occurrence of
such a constant is accounted for by a named rule.

Not proved: that no column name reaches a channel through *inter-procedural*
data flow -- a constant returned from a helper, threaded through a caller's
argument, or assembled from pieces. :func:`_column_reference_nodes` resolves
local names, ternaries and containers within a function and its enclosing
scopes -- including a chained ``a = b = value`` assignment, a same-arity
tuple/list unpack (``lo, hi = "x_lo", "x_hi"``), and a ``for name in
(...)`` loop over a literal container, each of which is intra-procedural and
therefore now covered (round 5; a fourth review cycle found ``for col in
("q1", "median", "q3")`` invisible in this exact shape in ``desugar_violin``
until round 4 removed the site, not the gap) -- but it does not follow values
across a *function* boundary, through a loop over a *module-level* constant
(``for name in _MODULE_CONST:``), or through a comprehension-built encoding;
such a constant is still in the outer bound and still accounted, but by its
own syntactic position rather than as a column reference. Establishing that would need a dataflow analysis Python does not
hand us. That is the boundary of this file's claim, and unlike the previous
four it is a property of the mechanism rather than a count somebody made.
"""

from __future__ import annotations

import ast
import functools
import inspect
import pathlib
from typing import NamedTuple

import polars as pl
import pytest

from ferrum import _core
import ferrum.encoding as _encoding
from ferrum.encoding.base import ChannelBase

_SRC = pathlib.Path(__file__).parent.parent / "src" / "ferrum"

_LOOKUP_HELPER = "registry_output_columns"

#: Keyword arguments whose value is a *container* of channel bindings rather
#: than a single one. A dict here is keyed by channel name; a call here builds
#: such a dict, so its arguments are channel bindings too.
_ENCODING_CONTAINERS = frozenset({"encoding", "mark_kwargs", "remap"})


# ---------------------------------------------------------------------------
# Inputs derived from the live registry and the live encoding package
# ---------------------------------------------------------------------------


def _registry_kinds() -> list[str]:
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
        kinds = [k.strip() for k in listed.split(",") if k.strip()]
    else:  # pragma: no cover - the binding is contractually required to raise
        raise AssertionError("transform_registry accepted an unknown kind")
    assert len(kinds) > 30, f"suspiciously few kinds parsed from the refusal: {kinds}"
    for kind in kinds:
        _core.transform_registry(kind)  # raises if the parse produced garbage
    return kinds


def _declared_output_names() -> set[str]:
    """Union of every output column name any registry kind declares."""
    names: set[str] = set()
    for kind in _registry_kinds():
        names |= {name for name, _dtype in _core.transform_registry(kind)["outputs"]}
    return names


def _channel_vocabulary() -> tuple[set[str], set[str]]:
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
    return classes, names


_DECLARED_NAMES = _declared_output_names()
_CHANNEL_CLASSES, _CHANNEL_NAMES = _channel_vocabulary()

#: ``ch.__class__(col, **kwargs)`` is the rebuild-the-same-channel-type idiom
#: (``_layer_transforms._resolve_layer_bins``); its first positional argument
#: is a column name by construction, exactly as ``X(col)``'s is.
_CHANNEL_CALLEES = _CHANNEL_CLASSES | {"__class__"}


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
    #: loop variable could hold across iterations (see :func:`mark` in
    #: :func:`_column_reference_nodes`, which already flattens a container).
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
        if isinstance(value, ast.Call) and _callee_name(value) == _LOOKUP_HELPER:
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
            # to every element via ``mark()``'s existing container case.
            bind(scope, node.target.id, node.iter)
    return _ModuleIndex(
        tree=tree,
        parents=parents,
        enclosing=enclosing,
        bindings=bindings,
        lookup_names=lookup_names,
        channel_callees=_local_aliases(tree, _CHANNEL_CALLEES),
    )


# ---------------------------------------------------------------------------
# The two rules that say "this literal names a transform output column"
# ---------------------------------------------------------------------------


def _converted_nodes(index: _ModuleIndex) -> tuple[set[int], set[int]]:
    """(argument nodes, lookup-read nodes) of the M6a conversion.

    An *argument* node is a name inside ``registry_output_columns(kind, [...])``
    -- the point where the registry check happens. A *lookup-read* node is the
    key a call site reads back, either out of a local bound to that call
    (``_bin_cols["bin_start"]``) or straight off it
    (``registry_output_columns("bin", ["bin_start"])["bin_start"]``).

    Recognising the reads is load-bearing, not decorative: without it every one
    of them would be labelled ``mapping-key`` and ``converted-lookup-read``
    would match nothing, which ``test_every_accounting_rule_fires`` refuses.
    """
    arguments: set[int] = set()
    reads: set[int] = set()
    for node in ast.walk(index.tree):
        if isinstance(node, ast.Call) and _callee_name(node) == _LOOKUP_HELPER:
            if len(node.args) == 2:
                for elt in getattr(node.args[1], "elts", []):
                    if isinstance(elt, ast.Constant):
                        arguments.add(id(elt))
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            base = node.value
            from_name = isinstance(base, ast.Name) and base.id in index.lookup_names
            from_call = isinstance(base, ast.Call) and _callee_name(base) == _LOOKUP_HELPER
            if from_name or from_call:
                reads.add(id(node.slice))
    return arguments, reads


def _column_reference_nodes(index: _ModuleIndex) -> set[int]:
    """Every constant that sits in a *channel position*.

    A channel position is one of:

    * a value in a dict entry whose key is an encoding channel name --
      ``{"x": "x_lo"}`` -- wherever that dict is written, including a dict
      bound to a local and a dict chosen by a ternary;
    * the first positional argument of an encoding-channel constructor --
      ``X("bin_start")``, ``ch.__class__("bin_start")`` -- under any import
      alias;
    * the value of a keyword argument named after a channel --
      ``.encode(y=Y("sample"))``, ``.encode(x="x_lo")``, ``Smooth(x="x")``;
    * the value assigned to a channel-keyed subscript -- ``enc["y"] = "count"``;
    * any argument of a call passed as ``encoding=``/``mark_kwargs=``/``remap=``,
      since such a call builds the channel dict (``encoding=enc("q1")``).

    Resolution follows local names, ternary branches and containers within the
    enclosing function chain, which is what closes the four escape shapes the
    cycle-3 review named: a ``replace()``-built layer whose ``data_source`` is
    a variable (the position no longer depends on ``data_source`` at all), a
    ternary encoding value, a local bound to a channel object, and a
    chart-level ``.encode()``. ``_index`` (see its docstring) additionally
    resolves a chained ``a = b = value`` assignment, a same-arity tuple/list
    unpack, and a ``for name in (...)`` loop over a literal container --
    closing the three intra-procedural gaps a fourth review cycle found
    (round 5).
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
            if isinstance(key, ast.Constant) and key.value in _CHANNEL_NAMES:
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
                if kw.arg in _CHANNEL_NAMES:
                    mark(kw.value, owner, set())
                elif kw.arg in _ENCODING_CONTAINERS:
                    mark_container(kw.value, owner, set())
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value in _CHANNEL_NAMES
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
_ACCOUNTING: tuple[tuple[str, str], ...] = (
    (
        "converted-argument",
        "A name passed to registry_output_columns(kind, [...]), which checks it "
        "against transform_registry(kind)['outputs'] at call time. This is the "
        "M6a conversion itself.",
    ),
    (
        "converted-lookup-read",
        "The key a call site reads back out of a registry_output_columns result. "
        "Its validity was established by the call that produced the lookup.",
    ),
    (
        "column-reference",
        "The literal sits in a channel position, so it names a data column. Must "
        "be converted or carry a reasoned _DISPOSITIONED entry.",
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

_LABELS = tuple(label for label, _why in _ACCOUNTING)


class _Occurrence(NamedTuple):
    path: str
    lineno: int
    function: str
    literal: str
    label: str
    #: For ``non-channel-keyword``, the keyword parameter name (``"align"``);
    #: for ``call-argument``, the callee name (``"col"``). Empty otherwise.
    #: Exists so the two labels' population-enumeration justifications in
    #: ``_ACCOUNTING`` can be rechecked by a test instead of trusted by prose.
    detail: str = ""

    @property
    def site(self) -> tuple[str, str, str]:
        """The _DISPOSITIONED key: stable under edits elsewhere in the file."""
        return (self.path, self.function, self.literal)


def _classify_module(path: pathlib.Path, rel: str) -> list[_Occurrence]:
    index = _index(path)
    converted_args, converted_reads = _converted_nodes(index)
    column_refs = _column_reference_nodes(index)

    dict_keys: set[int] = set()
    subscript_indices: set[int] = set()
    for node in ast.walk(index.tree):
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant):
                    dict_keys.add(id(key))
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            subscript_indices.add(id(node.slice))

    occurrences: list[_Occurrence] = []
    for node in ast.walk(index.tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if node.value not in _DECLARED_NAMES:
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
            _Occurrence(
                path=rel,
                lineno=node.lineno,
                function=getattr(owner, "name", "<module>"),
                literal=node.value,
                label=label,
                detail=detail,
            )
        )
    return occurrences


def _outer_bound() -> list[_Occurrence]:
    """The computable outer bound, classified.

    Every string constant in ``src/ferrum/**.py`` equal to a declared registry
    output-column name. No ferrum-specific shape narrows it.
    """
    found: list[_Occurrence] = []
    for path in sorted(_SRC.rglob("*.py")):
        found.extend(_classify_module(path, str(path.relative_to(_SRC.parent.parent))))
    return sorted(found)


_OUTER_BOUND = _outer_bound()
_COLUMN_REFERENCES = [o for o in _OUTER_BOUND if o.label == "column-reference"]


# ---------------------------------------------------------------------------
# Per-site dispositions: a column reference that is NOT a registry-declared
# output despite colliding with the name of one. Every entry must still be a
# live column-reference hit (see the staleness test), so a disposition cannot
# outlive the code that needed it.
# ---------------------------------------------------------------------------

_BOXEN = (
    "desugar_boxen's layers read the SECONDARY letter_value batches "
    "(`lv_depth_{k}` / `lv_outliers`), which the registry deliberately does not "
    "declare; the primary letter_value schema is group/depth/lower/upper/level. "
    "Checking these against the primary schema would pin a schema this code path "
    "never reads -- `value` is not even a member of it."
)
_DENDROGRAM = (
    "Dendrogram link segments read the `linkage` transform's SECONDARY named "
    "batch (`*_link_segments`), whose x/y/x2/y2 columns the registry does not "
    "declare; `linkage`'s declared outputs are node_id/left/right/distance/n_obs."
)
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

_DISPOSITIONED: dict[tuple[str, str, str], str] = {
    # --- letter_value secondary batches -----------------------------------
    ("src/ferrum/marks/composite.py", "desugar_boxen", "group"): _BOXEN,
    ("src/ferrum/marks/composite.py", "desugar_boxen", "lower"): _BOXEN,
    ("src/ferrum/marks/composite.py", "desugar_boxen", "upper"): _BOXEN,
    ("src/ferrum/marks/composite.py", "desugar_boxen", "value"): _BOXEN,
    # --- linkage secondary batches ----------------------------------------
    ("src/ferrum/plots/matrix.py", "_clustermap_build", "x"): _DENDROGRAM,
    ("src/ferrum/plots/matrix.py", "_clustermap_build", "y"): _DENDROGRAM,
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
        "Python. Collides with smooth's declared x/y."
    ),
    ("src/ferrum/marks/diagnostic/_clustering.py", "desugar_decision_boundary", "y"): (
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
        "Reads the `pl.DataFrame(rows)` built in the same function, whose rows "
        "carry literal metric/class/value/value_fmt keys."
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


def _undispositioned() -> list[_Occurrence]:
    return [o for o in _COLUMN_REFERENCES if o.site not in _DISPOSITIONED]


# ---------------------------------------------------------------------------
# The gates
# ---------------------------------------------------------------------------


def test_the_outer_bound_is_wide_and_real():
    """A vacuous outer bound would make every test below pass for free.

    The floors are deliberately far below the observed values (774 occurrences
    of 40 distinct declared names across 59 files at the time of writing): they
    catch a parse that silently found nothing, not a refactor.
    """
    assert len(_OUTER_BOUND) > 300, len(_OUTER_BOUND)
    assert len({o.literal for o in _OUTER_BOUND}) > 20
    assert len({o.path for o in _OUTER_BOUND}) > 10
    assert len(_DECLARED_NAMES) > 40, sorted(_DECLARED_NAMES)


def test_every_outer_bound_occurrence_is_accounted_for():
    """Zero unaccounted. This is the boundary claim.

    Each occurrence carries exactly one label from _ACCOUNTING; a syntactic
    position no rule describes fails here rather than falling off the end of a
    list somebody wrote by hand.
    """
    unaccounted = [o for o in _OUTER_BOUND if o.label not in _LABELS]
    assert not unaccounted, "\n".join(
        f"{o.path}:{o.lineno} [{o.function}] {o.literal!r} -- {o.label}; "
        "add a rule to _ACCOUNTING (with its justification) or convert the site"
        for o in unaccounted
    )


def test_every_accounting_rule_fires():
    """No rule may be decoration.

    A label that matches nothing is either dead scaffolding or a rule whose
    predicate silently stopped working -- both of which would let the rules
    beneath it swallow occurrences they do not describe.
    """
    seen = {o.label for o in _OUTER_BOUND}
    dead = [label for label in _LABELS if label not in seen]
    assert not dead, f"accounting rules matched by no occurrence: {dead}"


# ---------------------------------------------------------------------------
# Population recheck for the two labels whose justification in _ACCOUNTING
# is an enumeration ("align='right', axis='x', how='left', ...", "dict.get,
# pl.col, pl.Series, df.group_by, AggregateOp") rather than a structural
# predicate. Cycle-4 review: both were "justified by enumerating the
# population they currently match ... but nothing re-checks membership."
# These two sets ARE that recheck: a keyword argument name or callee that
# was not already confirmed safe must be re-justified before it can pass
# silently into the bucket's "-class" hand-wave.
# ---------------------------------------------------------------------------

_KNOWN_NON_CHANNEL_KEYWORDS = frozenset(
    {
        "align",  # e.g. mark_kwargs align="right" / a rule layer's text align
        "axis",  # e.g. jitter axis="x"
        "how",  # e.g. a join/merge how="left"
        "label_align",  # a text-mark label alignment kwarg, same class as align
        "name",  # e.g. a named transform or layer name="median"
        "title",  # e.g. an axis/legend title="sample"
        "value_name",  # Unpivot(value_name="value") -- declares, does not read
    }
)
_KNOWN_CALL_ARGUMENT_CALLEES = frozenset(
    {
        "AggregateOp",  # AggregateOp(field, "count", "n")
        "Series",  # pl.Series("y", ...)
        "__setattr__",  # object.__setattr__(obj, "count", ...) on a frozen dataclass
        "_channel_class_for",  # internal channel-name -> class lookup, not a column read
        "_extract_ordinal_domain",  # internal helper, positional field name
        "_orig_field",  # internal helper, positional field name
        "_set_composite_mark",  # internal composite-mark metadata setter
        "append",  # list.append("count") building a column-name list
        "col",  # pl.col("value")
        "get",  # dict.get("x") / dict.get("theoretical")
        "getattr",  # getattr(obj, "count")
        "group_by",  # df.group_by("y")
        "pop",  # dict.pop("count")
        "sort",  # list.sort(key=...) target naming, or df.sort("x")
    }
)


def test_non_channel_keyword_population_is_rechecked():
    """A new keyword-argument name in this bucket must be re-justified.

    _ACCOUNTING's non-channel-keyword justification is an enumeration
    ("align='right', axis='x', how='left', name='median', title='sample',
    value_name='value'"), not a structural predicate -- a keyword parameter
    that happens to take a column name out of a transform batch would land
    here unnoticed otherwise. New members fail loudly; a member that
    disappears must be removed too, so this set cannot go stale in either
    direction (mirrors ``test_every_disposition_is_still_live``'s discipline
    for _DISPOSITIONED).
    """
    observed = {o.detail for o in _OUTER_BOUND if o.label == "non-channel-keyword"}
    new = observed - _KNOWN_NON_CHANNEL_KEYWORDS
    stale = _KNOWN_NON_CHANNEL_KEYWORDS - observed
    assert not new, (
        f"new non-channel-keyword argument name(s) {sorted(new)}: confirm none of "
        "them takes a column name out of a transform batch (if one does, it belongs "
        "in _column_reference_nodes instead), then add it to "
        "_KNOWN_NON_CHANNEL_KEYWORDS with a one-line reason"
    )
    assert not stale, f"_KNOWN_NON_CHANNEL_KEYWORDS entries no longer observed: {sorted(stale)}"


def test_call_argument_population_is_rechecked():
    """A new callee in this bucket must be re-justified.

    _ACCOUNTING's call-argument justification is illustrative examples
    ("dict.get('x'), pl.col('value'), pl.Series('y', ...), df.group_by('y'),
    AggregateOp(field, 'count', 'n')"), backed by an architectural premise
    ("Python only ever holds frames it materialised itself") that holds
    per-callee, not for the bucket as a whole. New members fail loudly and
    must be checked against that premise before being added; a member that
    disappears must be removed too.
    """
    observed = {o.detail for o in _OUTER_BOUND if o.label == "call-argument"}
    new = observed - _KNOWN_CALL_ARGUMENT_CALLEES
    stale = _KNOWN_CALL_ARGUMENT_CALLEES - observed
    assert not new, (
        f"new call-argument callee(s) {sorted(new)}: confirm the call does not read "
        "a transform's output batch (if it does, the site needs a registry_output_"
        "columns conversion, not this bucket), then add it to "
        "_KNOWN_CALL_ARGUMENT_CALLEES with a one-line reason"
    )
    assert not stale, f"_KNOWN_CALL_ARGUMENT_CALLEES entries no longer observed: {sorted(stale)}"


def test_no_column_reference_is_a_bare_literal():
    """The M6a conversion is complete over every channel position in the tree.

    This is the test the round-1 and round-2 hand inventories could not
    produce, and whose population the round-3 sweep hand-scoped.
    """
    offenders = _undispositioned()
    assert not offenders, "\n".join(
        f"{o.path}:{o.lineno} [{o.function}] {o.literal!r} in a channel position "
        f"-- a registry-declared output column embedded as a bare literal; read it "
        f"through {_LOOKUP_HELPER}(kind, [...]) or add a reasoned entry to "
        f"_DISPOSITIONED keyed {o.site!r}"
        for o in offenders
    )


def test_every_disposition_is_still_live():
    """A disposition that no longer matches a column-reference hit is stale.

    This doubles as the coverage floor for the classifier itself: 29
    dispositions across 10 files must each still be *recognised* as a channel
    position, so a predicate that quietly stopped matching consumer code fails
    here in bulk instead of letting the gate above pass vacuously.
    """
    live = {o.site for o in _COLUMN_REFERENCES}
    stale = sorted(key for key in _DISPOSITIONED if key not in live)
    assert not stale, f"_DISPOSITIONED entries no longer matched by any channel position: {stale}"


def test_the_conversion_is_visible_to_the_accounting():
    """The converted sites are labelled as converted, in both shapes.

    ``converted-lookup-read`` existing at all is the proof that recognising a
    subscript into a lookup binding is load-bearing: drop that recognition and
    every read falls to ``mapping-key``.
    """
    args = [o for o in _OUTER_BOUND if o.label == "converted-argument"]
    reads = [o for o in _OUTER_BOUND if o.label == "converted-lookup-read"]
    assert len(args) >= 40, len(args)
    assert len(reads) >= 30, len(reads)
    assert len({o.path for o in args}) >= 6, sorted({o.path for o in args})


def test_the_classifier_labels_every_shape_it_claims(tmp_path):
    """RED proof, in each shape a real consumer is written in.

    Every function below mirrors a site that exists (or existed pre-conversion)
    in ``src/ferrum``; the comment names it. The controls at the end must NOT be
    labelled ``column-reference``, so the classifier is shown to discriminate
    rather than to flag every string it meets.
    """
    source = '''
from dataclasses import replace
from ferrum._layer import _Layer, MarkDesugarResult
from ferrum.encoding import X, Y
from ferrum.encoding import X as _X_ecdf
from ferrum.marks._desugar_helpers import registry_output_columns

def inline_dict():
    """marks/composite.py desugar_boxplot, pre-conversion."""
    return _Layer(mark="line", encoding={"y": "ci_lower"}, data_source="smooth")

def name_bound_dict():
    """marks/statistical.py desugar_histogram's encoding_remap."""
    encoding_remap = {"x": "bin_start"}
    return MarkDesugarResult(mark="bar", remap=encoding_remap)

def helper_built_encoding():
    """marks/composite.py desugar_boxplot's enc(...) helper."""
    def enc(col):
        return {"y": col}
    return _Layer(mark="rule", encoding=enc("lower_whisker"), data_source="box")

def aliased_channel_class():
    """plots/distribution.py displot(kind='ecdf'), pre-conversion."""
    enc = {}
    enc["x"] = _X_ecdf("bin_start", title="size")
    enc["y"] = "count"
    return enc

def ternary_encoding(horizontal):
    """marks/composite.py desugar_boxen's ternary encoding dict."""
    return _Layer(
        mark="rule",
        encoding={"x": "lower", "y": "q1"} if horizontal else {"x": "q3", "y": "lower"},
        data_source="lv_depth_1",
    )

def name_bound_to_a_channel_object(cat_sort):
    """marks/composite.py desugar_boxen's group_enc_x."""
    group_enc_x = X("group", sort=cat_sort) if cat_sort is not None else "group"
    return _Layer(mark="bar", encoding={"x": group_enc_x}, data_source="lv_outliers")

def replace_built_layer(layer, ch):
    """_layer_transforms.py _resolve_layer_bins."""
    return replace(layer, encoding={"x": ch.__class__("bin_start")}, data_source=ch.name)

def chart_level_encode(chart):
    """plots/matrix.py jointplot(kind='hist') and plots/regression.py's Q-Q panel."""
    return chart.encode(x="x_lo", y=Y("sample", title="Sample quantiles"))

def converted_argument_and_read():
    cols = registry_output_columns("smooth", ["ci_upper"])
    return _Layer(mark="line", encoding={"y": cols["ci_upper"]}, data_source="smooth")

def tuple_unpacked():
    """Round-5: a same-arity tuple unpack feeding a channel position."""
    lo, hi = "x_lo", "x_hi"
    return _Layer(mark="rect", encoding={"x": lo, "x2": hi}, data_source="bin_2d")

def chained_assign():
    """Round-5: a chained ``a = b = value`` assignment feeding a channel."""
    a = b = "bin_start"
    return _Layer(mark="bar", encoding={"x": a, "y": b}, data_source="bin")

def loop_target():
    """Round-5: desugar_violin's pre-round-4 shape -- a for-loop variable
    bound to a literal container, used in a channel position on each pass."""
    out = {}
    for name in ("q1", "median", "q3"):
        out["x"] = name
    return _Layer(mark="rule", encoding=out, data_source="box_stats")

def controls(other, axis):
    """None of these names a transform output column."""
    picked = other["ci_lower"]
    for name in ("q1", "median"):
        picked = picked or name
    if axis == "density":
        picked = other.get("theoretical")
    return _Layer(mark="text", mark_kwargs={"align": "right"}, encoding={"text": picked})
'''
    module = tmp_path / "synthetic_consumers.py"
    module.write_text(source)
    found = _classify_module(module, "synthetic_consumers.py")
    by_label: dict[str, set[str]] = {}
    for occurrence in found:
        by_label.setdefault(occurrence.label, set()).add(
            f"{occurrence.function}:{occurrence.literal}"
        )

    assert by_label["column-reference"] == {
        "inline_dict:ci_lower",
        "name_bound_dict:bin_start",
        "helper_built_encoding:lower_whisker",
        "aliased_channel_class:bin_start",
        "aliased_channel_class:count",
        "ternary_encoding:lower",
        "ternary_encoding:q1",
        "ternary_encoding:q3",
        "name_bound_to_a_channel_object:group",
        "replace_built_layer:bin_start",
        "chart_level_encode:x_lo",
        "chart_level_encode:sample",
        "tuple_unpacked:x_lo",
        "tuple_unpacked:x_hi",
        "chained_assign:bin_start",
        "loop_target:q1",
        "loop_target:median",
        "loop_target:q3",
    }
    assert by_label["converted-argument"] == {"converted_argument_and_read:ci_upper"}
    assert by_label["converted-lookup-read"] == {"converted_argument_and_read:ci_upper"}
    # The controls land in the non-column bucket that describes them. (These
    # are subset assertions because the module's own channel keys -- the "x"
    # and "y" of every encoding dict above -- are themselves declared output
    # names and so are mapping-key occurrences too.)
    assert "controls:ci_lower" in by_label["mapping-key"]
    assert {"controls:q1", "controls:median"} <= by_label["vocabulary-member"]
    assert "controls:density" in by_label["predicate-operand"]
    assert "controls:theoretical" in by_label["call-argument"]
    assert "controls:right" in by_label["non-channel-dict-value"]
    assert not [o for o in found if o.label not in _LABELS]


# ---------------------------------------------------------------------------
# The consumed-column drift guard, derived from the conversion call sites
# ---------------------------------------------------------------------------


def _converted_columns() -> dict[str, set[str]]:
    """kind -> every column name a ``registry_output_columns`` call site in
    ``src/ferrum`` asks for, collected from the source rather than mirrored in
    a table here."""
    consumed: dict[str, set[str]] = {}
    for path in sorted(_SRC.rglob("*.py")):
        tree = _parse(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _callee_name(node) != _LOOKUP_HELPER:
                continue
            if len(node.args) != 2 or not isinstance(node.args[0], ast.Constant):
                continue
            names = {
                elt.value
                for elt in getattr(node.args[1], "elts", [])
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            }
            consumed.setdefault(node.args[0].value, set()).update(names)
    return consumed


_CONVERTED = _converted_columns()


def test_conversion_sites_were_found():
    """The drift guard below is derived from call sites; an empty derivation
    would make it pass vacuously."""
    assert len(_CONVERTED) >= 10, _CONVERTED


@pytest.mark.parametrize("kind", sorted(_CONVERTED))
def test_every_consumed_column_is_a_real_registry_output(kind):
    """Python's consumption side mirrored against Rust's declaration side: the
    registry-checked lookups pass their runtime check on every code path,
    including the branches a given test run never executes."""
    declared = {name for name, _dtype in _core.transform_registry(kind)["outputs"]}
    missing = sorted(_CONVERTED[kind] - declared)
    assert not missing, (
        f"{kind}: Python reads column(s) {missing} that "
        f"transform_registry({kind!r})['outputs'] does not declare ({sorted(declared)})"
    )


def test_residuals_qq_panel_binds_its_y_channel_to_the_declared_qq_column():
    """Behavioural pin on the built panel, not on the source text.

    The residuals Q-Q panel's y channel must be the `qq` transform's declared
    `sample` output, and its x channel must NOT be a qq output (it names the
    panel's own input column) -- so a reformat cannot break this and a
    re-binding cannot slip past it.
    """
    from ferrum.plots import regression

    declared = {name for name, _dtype in _core.transform_registry("qq")["outputs"]}
    frame = pl.DataFrame(
        {
            "y_pred": [1.0, 2.0, 3.0, 4.0],
            "residual": [0.1, -0.2, 0.3, -0.4],
            "studentized_residual": [0.4, -0.8, 1.2, -1.6],
        }
    )
    panel = regression._residuals_panel(frame, "qq", kind="studentized")

    y_field = panel._encoding["y"].field
    assert y_field == "sample"
    assert y_field in declared
    assert panel._encoding["x"].field not in declared
