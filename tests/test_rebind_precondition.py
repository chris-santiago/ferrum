"""A new re-point site or statistic mark cannot drop a caller's bin=/aggregate= silently.

``_rebind_channel``/``_rebind_positional``/``_apply_remap`` rebuild a
channel at a statistic's output column, dropping its now-resolved
``bin=``/``aggregate=`` kwargs. That is correct only because every caller
of these three functions runs them after the channel's ``bin=``/
``aggregate=`` was either consumed by the statistic the caller's door
builds, or refused by that door first (``_rebind_channel``'s own
docstring states the precondition). Two classes of regression would break
that precondition silently:

* a new call to one of the three functions, added somewhere with no guard
  running first;
* a new statistic mark (or a layered branch of an existing one) that never
  calls any of the three at all, but also never refuses, so a channel's
  ``bin=``/``aggregate=`` is just never read anywhere.

Part 1 below is a call-site allowlist, in the shape
``tests/test_color_binding_completeness.py``'s ``ALLOWED`` uses: every
call to the three functions, keyed by ``(path, enclosing function)``, must
be a named, rationale-carrying entry. It also checks the narrower claim
that is actually decidable by source inspection: every module-level
``desugar_*`` in ``ferrum.marks`` that can return a positional ``remap``
(the thing ``_apply_remap`` re-points) is reached by a ``_PENDING_PREP``
kind carrying its own ``_prep_refuse_*`` step, or is a named direct-refusal
entry (``mark_function`` never goes through the dispatcher at all).

Part 2 is the behavioral backstop part 1's call-site lint cannot see: a
layered statistic (one whose desugar returns ``layers=``, never reaching
``_apply_remap`` at all) that drops a channel's ``bin=``/``aggregate=``
without ever calling any of the three functions. It sweeps every
composite mark in ``tests.test_figure_regroup_population``'s
``_composite_mark_population()``, every one of its ``_MARK_BUILD_VARIANTS``
build variants, crossed with x/y and a swept ``bin=``/``aggregate=``, and
requires each to either refuse (naming the kwarg, in its own voice) or
render differently from the same binding with the kwarg left off. A
channel the mark's own default build does not establish a literal column
for is probed on a real data column anyway -- the same ``_axes_for``
fallback the fixture module uses for its own point-chart partner -- not
skipped, because a caller can bind any column there and some marks (the
26 diagnostic marks) drop it whole. Those 26, and the six categorical
composites and ``mark_ribbon`` already pinned as silently ignored or
honored by ``tests/test_statistic_bin_aggregate_refusal.py``, are
explicit, rationale-carrying, stale-checked entries in
``_EXCLUDED_FROM_SWEEP`` rather than a derived or skipped set.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Callable, NamedTuple

import pytest

import ferrum as fm
from ferrum._desugar import _KindPrep, _PENDING_PREP
from ferrum.encoding import X, Y
from tests.test_desugar_private_params import _desugar_functions
from tests.test_figure_regroup_population import (
    _MARK_BUILD_VARIANTS,
    _axes_for,
    _composite_mark_population,
    _established_field,
    _missing_build_cases,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
_SCANNED_ROOT = REPO_ROOT / "src" / "ferrum"

# ---------------------------------------------------------------------------
# Part 1a: every call to the three re-pointing functions is allowlisted,
# keyed by (path, enclosing function), with the guard that runs before it.
# ---------------------------------------------------------------------------

#: The re-pointing primitives this guard watches. A new call to any of
#: these drops a channel's bin=/aggregate= the moment it runs, so every
#: call site needs the precondition their own docstrings state: the
#: channel's bin=/aggregate= was already consumed or refused.
_TARGET_CALLS = frozenset({"_rebind_positional", "_apply_remap", "_rebind_channel"})


class _CallSite(NamedTuple):
    path: str
    enclosing: str
    lineno: int

    @property
    def key(self) -> tuple[str, str]:
        """Allowlist key: file plus enclosing function, not the line number,
        so an unrelated edit that moves code does not produce a spurious
        failure -- while still failing the moment a NEW enclosing function
        starts calling one of these.
        """
        return (self.path, self.enclosing)


class _EnclosingFunctionVisitor(ast.NodeVisitor):
    """Collects every call to a name in :data:`_TARGET_CALLS`, tagged with
    the name of its nearest enclosing function (``"<module>"`` for a call
    at module scope).

    Structural ceiling, like ``test_color_binding_completeness.py``'s own
    AST scan: this sees a bare-name call (``_rebind_channel(...)``) only,
    which is the only shape any call site in this codebase actually uses
    (verified by grep); an aliased import or an attribute-qualified call
    would not be seen.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._stack: list[str] = []
        self.sites: list[_CallSite] = []

    def _descend(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._descend(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._descend(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in _TARGET_CALLS:
            enclosing = self._stack[-1] if self._stack else "<module>"
            self.sites.append(_CallSite(self._path, enclosing, node.lineno))
        self.generic_visit(node)


def _call_sites_in(tree: ast.AST, path: str) -> list[_CallSite]:
    visitor = _EnclosingFunctionVisitor(path)
    visitor.visit(tree)
    return visitor.sites


def _scan_call_sites() -> list[_CallSite]:
    found: list[_CallSite] = []
    for path in sorted(_SCANNED_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found.extend(_call_sites_in(tree, str(path.relative_to(REPO_ROOT))))
    return found


def _undeclared(
    sites: Iterable[_CallSite], allowed: dict[tuple[str, str], str]
) -> list[tuple[str, str]]:
    """Sorted keys of *sites* absent from *allowed* -- the one comparison
    both the real check below and its negative control run, so a broken
    comparison cannot leave the control green.
    """
    return sorted({site.key for site in sites} - set(allowed))


#: Every live call site, adjudicated: ``(path, enclosing function) ->
#: rationale`` naming the guard that runs before it, or why none is needed.
ALLOWED: dict[tuple[str, str], str] = {
    (
        "src/ferrum/_encode_lowering.py",
        "_rebind_positional",
    ): "The bare-name-or-channel face of _rebind_channel: for a ChannelBase "
    "operand it delegates straight to _rebind_channel, carrying the identical "
    "precondition that function's own docstring states (the channel's bin=/"
    "aggregate= was already consumed or refused by the caller's door before "
    "this runs). This is the implementation of the re-pointing primitive "
    "itself, not a usage site that needs its own guard.",
    (
        "src/ferrum/_encode_lowering.py",
        "_lower_bin_encoding",
    ): "The one function that lowers a PLAIN channel's own bin=/aggregate= "
    "into a Bin/Aggregate statistic plan for an ordinary mark -- not a "
    "composite statistic's dispatch at all. Each rebind writes into a local "
    "new_encoding that only leaves this function inside the _StatisticPlan it "
    "returns, carrying the exact Bin or per-bin aggregate column the rebound "
    "channel's own bin=/aggregate= became. Every path that cannot fold a "
    "channel's kwarg into that plan -- a second bin=, bin= and aggregate= on "
    "one channel, a declared statistic no channel draws, a pre-built Bin "
    "asked to absorb an aggregate= -- raises ValueError before any return, "
    "so a rebind here runs only once its own kwarg has already become the "
    "plan's output column.",
    (
        "src/ferrum/_desugar.py",
        "_apply_remap",
    ): "Delegates every x/y rebind to _rebind_positional, carrying the same "
    "precondition its own docstring states: the caller runs this only after "
    "the kind's own refusal prep, or a figure's call-time pre-check, already "
    "consumed or refused the channel's bin=/aggregate=. Implementation of the "
    "re-pointing primitive, not a usage site.",
    (
        "src/ferrum/_desugar.py",
        "_resolve_pending_as",
    ): "The one dispatcher every _PENDING_PREP kind resolves through: before "
    "calling the kind's desugar and then _apply_remap, it runs that kind's "
    "own pre steps in order, so the guard here is whichever _prep_refuse_* "
    "step the kind registers. Verified per kind by the remap-returning-desugar "
    "check below, rather than read out of this function's own body.",
    (
        "src/ferrum/marks/_chart_methods_statistical.py",
        "mark_function",
    ): "Two mutually exclusive branches of one call, split on has_prior_layer. "
    "When a prior layer already draws the caller's x/y, that layer's own "
    "bin=/aggregate= lowering already honors or refuses the binding, so this "
    "branch's remap -- which only re-points mark_function's own synthetic "
    "grid -- is deliberately unguarded here. When there is no prior layer, "
    "_refuse_function_alone_statistics runs earlier in this same call, before "
    "either branch is reached, and refuses a bin=/aggregate= that branch "
    "would otherwise drop.",
    (
        "src/ferrum/plots/regression.py",
        "residplot",
    ): "_refuse_statistic_input runs over both x and y earlier in this same "
    "function, before either rebind: the residual fit reads the raw "
    "predictor and response, so a bin=/aggregate= on either channel is "
    "refused there first.",
}


def test_every_rebind_call_site_is_allowlisted_with_its_guard() -> None:
    """Fails when a call to one of the three re-pointing functions has no
    allowlisted guard. The remedy is almost always to run a refusal (or
    confirm one already runs) before the new call, then add an
    :data:`ALLOWED` entry naming it.
    """
    undeclared = _undeclared(_scan_call_sites(), ALLOWED)
    assert not undeclared, "re-pointing call(s) with no allowlisted guard:\n" + "\n".join(
        f"  {path}: {enclosing}" for path, enclosing in undeclared
    )


def test_allowlist_has_no_stale_entries() -> None:
    """An entry matching no live call site has outlived the code it excused."""
    live = {site.key for site in _scan_call_sites()}
    stale = sorted(key for key in ALLOWED if key not in live)
    assert not stale, "ALLOWED entries matching no call site -- delete them:\n" + "\n".join(
        f"  {path}: {enclosing}" for path, enclosing in stale
    )


@pytest.mark.parametrize("key", sorted(ALLOWED), ids=lambda k: f"{Path(k[0]).name}::{k[1]}")
def test_every_allowlist_entry_carries_a_rationale(key: tuple[str, str]) -> None:
    rationale = ALLOWED[key]
    assert len(rationale.split()) >= 8, (
        f"allowlist entry {key} needs a real rationale, got {rationale!r}"
    )


def test_an_unlisted_call_is_flagged_by_the_scanner() -> None:
    """Negative control: a synthetic snippet calling ``_rebind_channel`` from
    a function the allowlist has never seen is found, through the same
    ``_undeclared`` comparison the real check runs.
    """
    snippet = "def a_future_re_point(ch, column):\n    return _rebind_channel(ch, column)\n"
    sites = _call_sites_in(ast.parse(snippet), "src/ferrum/synthetic.py")
    assert _undeclared(sites, ALLOWED) == [("src/ferrum/synthetic.py", "a_future_re_point")]


#: (kwarg, value, value_id) triples both part 1b's mark_function test and
#: part 2's sweep cross with x/y. bin=None, bin=False and aggregate=None
#: are not swept: they ask for nothing, which
#: test_bin_or_aggregate_unset_values_render_byte_identically_to_leaving_them_out
#: in tests/test_statistic_bin_aggregate_refusal.py already pins as
#: byte-identical to leaving the kwarg out entirely.
_SWEPT_CHANNEL_KWARGS: tuple[tuple[str, Any, str], ...] = (
    ("bin", True, "true"),
    ("bin", {"bin_count": 3}, "spec"),
    ("aggregate", "mean", "mean"),
)


# ---------------------------------------------------------------------------
# Part 1b: every module-level desugar_* in ferrum.marks that can return a
# positional remap is reached by a guarded _PENDING_PREP kind, or is a named
# direct-refusal entry.
# ---------------------------------------------------------------------------


def _unwrap(fn: Callable) -> Callable:
    """The real desugar a resolve-adapter wraps.

    ``statistical.py``'s ``_resolve_*`` functions, and ``mark_qq``'s own
    local closure, stamp ``__wrapped__`` with the underlying ``desugar_*``
    so ``inspect.signature`` (and this check) sees the real transform
    parameters instead of a thin ``**kwargs`` passthrough.
    """
    return getattr(fn, "__wrapped__", fn)


def _returns_positional_remap(fn: Callable) -> bool:
    """True when *fn* (unwrapped) can return ``MarkDesugarResult(remap=...)``
    with a non-``None`` value -- the single-mark route ``_apply_remap``
    re-points. A kind whose desugar returns ``layers=`` instead never
    reaches ``_apply_remap`` at all, so it is not in this rule's population
    (its own guard, if any, is a direct refusal with no rebind downstream).
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(_unwrap(fn))))
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "MarkDesugarResult"
        ):
            continue
        for keyword in node.keywords:
            if keyword.arg == "remap" and not (
                isinstance(keyword.value, ast.Constant) and keyword.value.value is None
            ):
                return True
    return False


def _desugar_fn_for_kind(kind: str) -> Callable:
    """The ``desugar_fn`` a ``_PENDING_PREP`` *kind* actually resolves
    through, read off a real pending mark rather than a hand-written
    kind -> function table, so a new kind joins this check with no edit
    here. ``"ecdf"`` has no public ``mark_ecdf`` door (``displot`` is its
    only caller), so it is read directly off the module-level resolver
    ``statistical.py`` stamps ``__wrapped__`` onto.
    """
    if kind == "ecdf":
        from ferrum.marks.statistical import _resolve_ecdf

        return _resolve_ecdf
    mark_name = f"mark_{kind}"
    chart = _MARK_BUILD_VARIANTS[mark_name]["default"]()
    pending = chart._pending_stat_mark
    assert pending is not None, f"{mark_name}'s own default build already resolved its pending mark"
    return pending.desugar_fn


def _qualified_desugar_name(kind: str) -> str:
    """The qualified ``module.function`` name of the desugar a
    ``_PENDING_PREP`` *kind* resolves through, in the same spelling
    :func:`~tests.test_desugar_private_params._desugar_functions` keys its
    population with.
    """
    fn = _unwrap(_desugar_fn_for_kind(kind))
    return f"{fn.__module__}.{fn.__name__}"


def _unguarded_remap_desugars(
    functions: dict[str, Callable], prep_table: dict[str, _KindPrep], direct: dict[str, str]
) -> list[str]:
    """Every remap-returning desugar that can drop a channel's bin=/
    aggregate= with no guard: a *prep_table* kind whose own desugar
    returns a positional remap but carries no ``_prep_refuse_*`` step
    (named by kind -- checked on the kind's own unwrapped desugar
    directly, not only through *functions*'s walk, so a kind reaching a
    desugar outside that walk is still judged), or a desugar in
    *functions* that no *prep_table* kind reaches and *direct* does not
    name (named by its qualified function name). The one completeness
    check both the real test below and its negative controls run.
    """
    remap_returning = {name for name, fn in functions.items() if _returns_positional_remap(fn)}
    reached: set[str] = set()
    missing_kinds: list[str] = []
    for kind, prep in prep_table.items():
        reached.add(_qualified_desugar_name(kind))
        if _returns_positional_remap(_desugar_fn_for_kind(kind)) and not any(
            step.__name__.startswith("_prep_refuse_") for step in prep.pre
        ):
            missing_kinds.append(kind)
    orphaned = (name for name in remap_returning if name not in reached and name not in direct)
    return sorted({*missing_kinds, *orphaned})


#: desugar_function is reached by no _PENDING_PREP kind at all -- mark_function
#: calls it directly, bypassing _resolve_pending_as entirely -- so its guard
#: has to be named here rather than discovered through the prep table.
_DIRECT_REFUSAL: dict[str, str] = {
    "ferrum.marks.heavy_stat.desugar_function": (
        "mark_function calls desugar_function directly, never through "
        "_resolve_pending_as, so no _PENDING_PREP kind ever reaches it. Its "
        "guard is _refuse_function_alone_statistics, which mark_function "
        "runs itself before calling desugar_function whenever there is no "
        "prior layer to keep."
    ),
}


def test_every_remap_returning_desugar_is_guarded() -> None:
    assert _unguarded_remap_desugars(_desugar_functions(), _PENDING_PREP, _DIRECT_REFUSAL) == []


def test_the_remap_returning_population_is_exactly_the_known_five() -> None:
    """Pins the exact desugars this rule governs, so a change in the
    detector itself -- not just a missing guard -- is visible by name.
    """
    functions = _desugar_functions()
    remapping = sorted(name for name, fn in functions.items() if _returns_positional_remap(fn))
    assert remapping == [
        "ferrum.marks.heavy_stat.desugar_function",
        "ferrum.marks.statistical.desugar_density",
        "ferrum.marks.statistical.desugar_ecdf",
        "ferrum.marks.statistical.desugar_histogram",
        "ferrum.marks.statistical.desugar_smooth",
    ]


def test_a_kind_with_its_refusal_prep_removed_is_caught() -> None:
    """Negative control: a scratch copy of ``_PENDING_PREP`` with
    histogram's own refusal step removed fails the same check, naming it.
    """
    stripped = _KindPrep(
        pre=tuple(
            step
            for step in _PENDING_PREP["histogram"].pre
            if not step.__name__.startswith("_prep_refuse_")
        )
    )
    scratch = {**_PENDING_PREP, "histogram": stripped}
    assert _unguarded_remap_desugars(_desugar_functions(), scratch, _DIRECT_REFUSAL) == ["histogram"]


def test_a_remap_returning_desugar_no_kind_reaches_is_caught() -> None:
    """Negative control: a synthetic desugar that no ``_PENDING_PREP`` kind
    reaches, and no ``_DIRECT_REFUSAL`` entry names, is caught by its own
    qualified name.
    """
    from ferrum._layer import MarkDesugarResult

    def desugar_future(x_field, y_field, **kwargs):
        return MarkDesugarResult(mark="line", remap={"x": "x_out"})

    functions = {**_desugar_functions(), "ferrum.marks.synthetic.desugar_future": desugar_future}
    missing = _unguarded_remap_desugars(functions, _PENDING_PREP, _DIRECT_REFUSAL)
    assert missing == ["ferrum.marks.synthetic.desugar_future"]


def test_the_direct_refusal_allowlist_has_no_stale_entries() -> None:
    """An entry is stale when its key names no live ``desugar_*``, when
    that desugar stops returning a positional remap, or when some
    ``_PENDING_PREP`` kind with its own ``_prep_refuse_*`` step reaches
    it -- any of which means the entry needs no naming here at all.
    """
    functions = _desugar_functions()
    guarded_names = {
        _qualified_desugar_name(kind)
        for kind, prep in _PENDING_PREP.items()
        if any(step.__name__.startswith("_prep_refuse_") for step in prep.pre)
    }
    stale = []
    for name in _DIRECT_REFUSAL:
        if name not in functions:
            stale.append(f"{name}: not a live desugar_*")
        elif not _returns_positional_remap(functions[name]):
            stale.append(f"{name}: does not return a positional remap")
        elif name in guarded_names:
            stale.append(f"{name}: reached by a prep-bearing kind")
    assert not stale, "stale direct-refusal entries:\n" + "\n".join(stale)


#: Every (channel, kwarg, value, case_id) this sweep crosses with
#: mark_function's own x/y, from _SWEPT_CHANNEL_KWARGS above.
_FUNCTION_ALONE_CASES: list[tuple[str, str, Any, str]] = [
    (channel, kwarg, value, f"{channel}-{kwarg}-{value_id}")
    for channel in ("x", "y")
    for kwarg, value, value_id in _SWEPT_CHANNEL_KWARGS
]


@pytest.mark.parametrize(
    "channel,kwarg,value",
    [(channel, kwarg, value) for channel, kwarg, value, _case_id in _FUNCTION_ALONE_CASES],
    ids=[case_id for *_rest, case_id in _FUNCTION_ALONE_CASES],
)
def test_desugar_function_alone_is_covered_by_its_own_refusal(
    channel: str, kwarg: str, value: Any
) -> None:
    """``mark_function`` has no ``_PENDING_PREP`` entry at all -- it builds
    via ``self._clone()``/``self + fn_chart``, bypassing
    ``_resolve_pending_as`` entirely -- so the desugar-level check above
    cannot see its guard. Pinned directly, on every channel and swept
    kwarg: with no prior layer to keep, ``_refuse_function_alone_statistics``
    runs before either of ``mark_function``'s own ``_apply_remap`` calls is
    reached, and names the kwarg it refuses.
    """
    cls = X if channel == "x" else Y
    with pytest.raises(ValueError) as excinfo:
        fm.Chart(None).encode(**{channel: cls("v", **{kwarg: value})}).mark_function(
            lambda xs: xs, domain=[0.0, 10.0]
        ).to_svg()
    message = str(excinfo.value)
    assert message.startswith("mark_function(") and f"{kwarg}=" in message


# ---------------------------------------------------------------------------
# Part 2: the behavioral sweep. Every composite mark, every build variant,
# crossed with x/y and a concrete bin=/aggregate=, either refuses (naming
# the kwarg, in its own voice) or renders differently from the plain chart --
# the backstop for a layered statistic that drops a channel without ever
# calling any of the three functions above, which part 1 cannot see.
# ---------------------------------------------------------------------------

#: One narrow, named, stale-checked exemption from the sweep below, keyed
#: by the exact (mark, channel, kwarg, value) the production code honors
#: as a harmless restatement rather than drops: mark_histogram's own
#: binned x already bins by its own rule, so a channel bin=True asks for
#: exactly what the histogram already does by default
#: (_refuse_histogram_bin_aggregate's own `bin_value is not True` carve-out,
#: src/ferrum/_encode_lowering.py:1218). Every other (mark, channel, kwarg,
#: value) pair, hex's bin=True included, is refused and stays swept.
_HONORED_REDUNDANCY: dict[tuple[str, str, str, Any], str] = {
    ("mark_histogram", "x", "bin", True): (
        "mark_histogram's own binned x axis already bins by its own rule, "
        "so a channel bin=True restates exactly what the histogram already "
        "does by default and is honored rather than dropped -- the one "
        "documented exception in _refuse_histogram_bin_aggregate "
        "(bin_value is not True)."
    ),
}


def _is_honored_redundancy(name: str, channel: str, kwarg: str, value: Any) -> bool:
    """Membership test against :data:`_HONORED_REDUNDANCY`'s keys by
    equality, never by hashing *value* -- a bin spec value is an unhashable
    dict.
    """
    return any(key == (name, channel, kwarg, value) for key in _HONORED_REDUNDANCY)


#: mark_name -> (verdict, rationale). "ignored": every probe below renders
#: byte-identically to its field-only baseline, so a channel's bin=/
#: aggregate= is silently dropped. "honors": every probe renders
#: differently, through a mechanism unrelated to any statistic refusal.
#: Both verdicts are behaviorally stale-checked, never assumed.
_CATEGORICAL_IGNORED_SIBLINGS: dict[str, str] = {
    "mark_boxplot": "boxen/errorbar/violin/errorband",
    "mark_boxen": "boxplot/errorbar/violin/errorband",
    "mark_errorbar": "boxplot/boxen/violin/errorband",
    "mark_errorband": "boxplot/boxen/errorbar/violin",
    "mark_violin": "boxplot/boxen/errorbar/errorband",
}

#: swarm's own pre is only _prep_cat_axis_sorts/_prep_cat_axis_types/
#: _prep_axis_titles (src/ferrum/_desugar.py's "swarm" entry) -- it lacks
#: the _prep_color_field/_prep_grouping steps boxplot/boxen/errorbar/
#: errorband/violin share, so it is not a sibling of their categorical
#: prep chain, only of their refusal-free outcome.
_SWARM_IGNORED_RATIONALE = (
    "Shares the sort/type preps with the other categorical composites, and "
    "carries no bin=/aggregate= refusal either; silently ignored for the "
    "same reason, pinned directly by "
    "test_categorical_composites_bin_and_aggregate_stay_silently_ignored "
    "in tests/test_statistic_bin_aggregate_refusal.py."
)

#: Shared by every one of the 26 diagnostic composite marks below: none of
#: them read the caller's chart-level x/y field at all (each builds its
#: layers from fixed columns of its own diagnostic frame -- ModelSource's
#: roc_curve(), pr_curve(), and the rest), so a chart-level x/y binding,
#: bin=/aggregate= included, is dropped whole rather than a statistic
#: declining to fold a kwarg in. Refusing it is a rule for every
#: fixed-schema composite mark, not a statistic-specific refusal, and
#: writing that rule reaches production code outside this test module.
_DIAGNOSTIC_MARK_IGNORED_RATIONALE = (
    "This diagnostic mark draws its layers from fixed columns of its own "
    "diagnostic frame and never reads the caller's x/y field at all, so a "
    "chart-level x/y binding -- bin=/aggregate= included -- is dropped "
    "whole. That is a rule for every fixed-schema composite mark, not a "
    "statistic-specific refusal, and writing it reaches production code "
    "outside what a test module changes."
)

_DIAGNOSTIC_MARKS: tuple[str, ...] = (
    "mark_alpha_selection",
    "mark_calibration",
    "mark_class_prediction_error",
    "mark_confusion",
    "mark_cv_scores",
    "mark_decision_boundary",
    "mark_discrimination_threshold",
    "mark_gain",
    "mark_importance",
    "mark_intercluster_distance",
    "mark_learning_curve",
    "mark_lift",
    "mark_parallel_coordinates",
    "mark_pca_scree",
    "mark_pdp",
    "mark_pr",
    "mark_prediction_error",
    "mark_rank1d",
    "mark_rank2d",
    "mark_residuals",
    "mark_roc",
    "mark_shap_bar",
    "mark_shap_beeswarm",
    "mark_shap_waterfall",
    "mark_silhouette",
    "mark_validation_curve",
)

_EXCLUDED_FROM_SWEEP: dict[str, tuple[str, str]] = {
    **{
        name: (
            "ignored",
            f"Shares the categorical prep chain with {siblings}: none of them "
            "carry a bin=/aggregate= refusal, so a channel's bin=/aggregate= is "
            "silently ignored, pinned directly by "
            "test_categorical_composites_bin_and_aggregate_stay_silently_ignored "
            "in tests/test_statistic_bin_aggregate_refusal.py.",
        )
        for name, siblings in _CATEGORICAL_IGNORED_SIBLINGS.items()
    },
    "mark_swarm": ("ignored", _SWARM_IGNORED_RATIONALE),
    "mark_ribbon": (
        "honors",
        "Computes no statistic of its own: its layer carries the caller's x/y/"
        "y2 channel objects verbatim, so a channel's bin=/aggregate= is honored "
        "through the ordinary channel-to-transform lowering every plain mark "
        "gets, never through a statistic refusal, pinned directly by "
        "test_mark_ribbon_honors_bin_and_aggregate_never_refused in "
        "tests/test_statistic_bin_aggregate_refusal.py.",
    ),
    **{name: ("ignored", _DIAGNOSTIC_MARK_IGNORED_RATIONALE) for name in _DIAGNOSTIC_MARKS},
}


def _malformed_exclusions(entries: dict[str, tuple[str, str]], population: set[str]) -> list[str]:
    """Entries naming a mark outside *population*, or carrying a verdict
    outside ``{"ignored", "honors"}`` -- the two ways an entry can stop
    meaning anything, shared by the real check and its control.
    """
    return sorted(
        name
        for name, (verdict, _) in entries.items()
        if name not in population or verdict not in ("ignored", "honors")
    )


def test_every_exclusion_entry_names_a_live_mark_with_a_known_verdict() -> None:
    assert _malformed_exclusions(_EXCLUDED_FROM_SWEEP, _composite_mark_population()) == []


def test_the_exclusion_check_catches_a_dead_name_and_an_unknown_verdict() -> None:
    synthetic = {
        "mark_not_a_real_mark": ("ignored", "a rationale written long enough to pass the word check"),
        "mark_roc": ("ignore", "a rationale written long enough to pass the word check too"),
    }
    assert _malformed_exclusions(synthetic, _composite_mark_population()) == [
        "mark_not_a_real_mark",
        "mark_roc",
    ]


@pytest.mark.parametrize("name", sorted(_EXCLUDED_FROM_SWEEP), ids=lambda n: n)
def test_every_exclusion_entry_carries_a_rationale(name: str) -> None:
    _, rationale = _EXCLUDED_FROM_SWEEP[name]
    assert len(rationale.split()) >= 8, f"{name}'s exclusion rationale is too short: {rationale!r}"


class _Outcome(NamedTuple):
    refused: bool
    message: str | None
    rendered_same: bool | None


def _probed_field(base: Any, channel: str) -> str:
    """The real data column this sweep binds *channel* to on chart *base*:
    the mark's own established field, else the fixture's own
    point-partner fallback (``_axes_for``, ``tests/test_figure_regroup_population.py``),
    which reads a real numeric column off the chart's data rather than
    leaving the channel unbound.
    """
    field = _established_field(base, channel)
    if field is not None:
        return field
    x, y = _axes_for(base)
    return x if channel == "x" else y


def _probe(
    build: Callable[[], Any], name: str, channel: str, field: str, kwarg: str, value: Any
) -> _Outcome:
    """Rebuild *build*(), bind *channel* to *field* carrying *kwarg*=*value*,
    and report whether the result refuses or renders differently from the
    SAME binding without the kwarg.

    The baseline is ``build().encode(channel=cls(field))`` rendered without
    *kwarg* -- not the plain ``build()`` -- because the field binding alone
    can change the render on a channel the mark's own build never
    established (a fresh column reaching a layer that had none at all).
    When that field-only binding itself refuses, in *name*'s own voice,
    this channel takes no field on this mark at all (mark_qq's y, which
    refuses any field-bearing y in its own right): the kwarg is then probed
    fieldless, against the plain, unbound *build*() as the baseline. A
    field-only refusal in any other voice is a malformed probe and raises
    rather than passing silently.
    """
    cls = X if channel == "x" else Y
    try:
        baseline_svg = build().encode(**{channel: cls(field)}).to_svg()
        channel_obj = cls(field, **{kwarg: value})
    except ValueError as exc:
        message = str(exc)
        if not message.startswith(f"{name}("):
            raise AssertionError(
                f"{name}'s field-only {channel}={field!r} binding raised in an "
                f"unexpected voice: {message!r} -- malformed probe"
            ) from exc
        baseline_svg = build().to_svg()
        channel_obj = cls(**{kwarg: value})
    probed = build().encode(**{channel: channel_obj})
    try:
        svg = probed.to_svg()
    except ValueError as exc:
        return _Outcome(refused=True, message=str(exc), rendered_same=None)
    return _Outcome(refused=False, message=None, rendered_same=svg == baseline_svg)


def _refused_naming_the_kwarg(outcome: _Outcome, name: str, kwarg: str) -> bool:
    """True when *outcome* is a refusal in *name*'s own voice that names
    *kwarg*= -- every ``_refuse_bin_or_aggregate`` text already does
    ("... so bin= would be ignored"), so this also catches a real refusal
    about something else entirely (mark_qq's field-y refusal, whose text
    never mentions bin=/aggregate=) instead of letting it count as a pass.
    """
    return bool(
        outcome.refused
        and outcome.message is not None
        and outcome.message.startswith(f"{name}(")
        and f"{kwarg}=" in outcome.message
    )


def _sweep_targets(names: Iterable[str]) -> Iterator[tuple[str, str, str, str, Any, str]]:
    """``(mark, variant, channel, kwarg, value, value_id)`` for every mark
    in *names*, crossed with every one of its ``_MARK_BUILD_VARIANTS``
    entries, x/y, and :data:`_SWEPT_CHANNEL_KWARGS`, skipping exactly the
    keys named in :data:`_HONORED_REDUNDANCY`. Shared by the main sweep's
    case list and the exclusion stale check below. A name with no
    ``_MARK_BUILD_VARIANTS`` entry yields nothing here -- it fails by name
    in :func:`test_every_swept_mark_has_a_build_case`, not as a collection
    error on every test in this module.
    """
    for name in names:
        for variant in _MARK_BUILD_VARIANTS.get(name, {}):
            for channel in ("x", "y"):
                for kwarg, value, value_id in _SWEPT_CHANNEL_KWARGS:
                    if _is_honored_redundancy(name, channel, kwarg, value):
                        continue
                    yield name, variant, channel, kwarg, value, value_id


_SWEEP_NAMES = sorted(_composite_mark_population() - set(_EXCLUDED_FROM_SWEEP))
_SWEEP_TARGETS = list(_sweep_targets(_SWEEP_NAMES))
_SWEEP_CASES = [
    (name, variant, channel, kwarg, value)
    for name, variant, channel, kwarg, value, _value_id in _SWEEP_TARGETS
]
_SWEEP_IDS = [
    f"{name}-{variant}-{channel}-{kwarg}-{value_id}"
    for name, variant, channel, kwarg, _value, value_id in _SWEEP_TARGETS
]


def test_every_swept_mark_has_a_build_case() -> None:
    missing = _missing_build_cases(set(_SWEEP_NAMES), set(_MARK_BUILD_VARIANTS))
    assert not missing, f"no build case for mark(s): {missing}"


def test_the_population_check_catches_a_swept_mark_without_a_build_case() -> None:
    """Control: the real swept population, plus one name with no build
    case, fails by that name.
    """
    population = set(_SWEEP_NAMES) | {"mark_not_a_real_mark"}
    missing = _missing_build_cases(population, set(_MARK_BUILD_VARIANTS))
    assert missing == ["mark_not_a_real_mark"]


@pytest.mark.parametrize("name,variant,channel,kwarg,value", _SWEEP_CASES, ids=_SWEEP_IDS)
def test_a_composite_mark_refuses_or_differs_on_a_bin_or_aggregate_it_cannot_honor(
    name: str, variant: str, channel: str, kwarg: str, value: Any
) -> None:
    build = _MARK_BUILD_VARIANTS[name][variant]
    field = _probed_field(build(), channel)
    outcome = _probe(build, name, channel, field, kwarg, value)
    if outcome.refused:
        assert _refused_naming_the_kwarg(outcome, name, kwarg), (
            f"{name}'s refusal did not name {kwarg}= in its own voice: {outcome.message!r}"
        )
        return
    assert not outcome.rendered_same, (
        f"{name}'s {channel}={kwarg}={value!r} neither refused naming {kwarg}= nor rendered "
        "differently from its field-only baseline -- it silently dropped the caller's binding."
    )


def test_the_exclusion_allowlist_has_no_stale_entries() -> None:
    """Every excluded mark's verdict is a claim about its actual rendered
    behavior, probed on every variant and channel through the same
    :func:`_sweep_targets`/:func:`_probe` rule the main sweep uses.
    "ignored" claims every probe is accepted (not refused) and
    byte-identical to its baseline; "honors" claims every probe is
    accepted and differs. A claim this check cannot reproduce is stale,
    and the mark belongs back in the sweep instead of being shielded from
    it. An entry with nothing to probe at all is stale too.
    """
    stale: list[str] = []
    for name, (verdict, _) in _EXCLUDED_FROM_SWEEP.items():
        probed_any = False
        for _name, variant, channel, kwarg, value, _value_id in _sweep_targets([name]):
            probed_any = True
            build = _MARK_BUILD_VARIANTS[name][variant]
            field = _probed_field(build(), channel)
            outcome = _probe(build, name, channel, field, kwarg, value)
            matches_ignored = not outcome.refused and outcome.rendered_same
            matches_honors = not outcome.refused and not outcome.rendered_same
            if verdict == "ignored" and not matches_ignored:
                stale.append(f"{name} {variant} {channel} {kwarg}")
            if verdict == "honors" and not matches_honors:
                stale.append(f"{name} {variant} {channel} {kwarg}")
        if not probed_any:
            stale.append(f"{name}: no probe ran")
    assert not stale, "exclusion entries that do not match the mark's rendered behavior:\n" + "\n".join(
        stale
    )


def test_the_honored_redundancy_allowance_is_live_and_still_honored() -> None:
    """Each allowance names a real (mark, channel) pair in the population
    with a real rationale, and is still the harmless restatement it
    claims: every variant of the mark, probed on that exact (channel,
    kwarg, value), is accepted and byte-identical to its field-only
    baseline. An allowance that is ever refused or rendered differently
    has outgrown its exemption and belongs back in the sweep.
    """
    population = _composite_mark_population()
    broken: list[str] = []
    for (name, channel, kwarg, value), rationale in _HONORED_REDUNDANCY.items():
        if name not in population or channel not in ("x", "y"):
            broken.append(f"{name} {channel}: not a live (mark, channel) pair")
            continue
        if name in _EXCLUDED_FROM_SWEEP:
            broken.append(f"{name} {channel}: excluded from the sweep, so it exempts nothing there")
            continue
        if len(rationale.split()) < 8:
            broken.append(f"{name} {channel} {kwarg}: rationale too short")
            continue
        for variant, build in _MARK_BUILD_VARIANTS[name].items():
            field = _probed_field(build(), channel)
            outcome = _probe(build, name, channel, field, kwarg, value)
            if outcome.refused or not outcome.rendered_same:
                broken.append(f"{name} {variant} {channel} {kwarg}: refused or rendered differently")
    assert not broken, "stale honored-redundancy allowance(s):\n" + "\n".join(broken)
