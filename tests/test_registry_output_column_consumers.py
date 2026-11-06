"""Batch-D T12: no Python consumer hardcodes a registry-declared output column.

Spec §4.6 / decision M6a require that the output-column names a Python
desugar/figure consumer reads out of a transform's output batch derive from
``ferrum._core.transform_registry`` instead of being embedded as bare string
literals. ``ferrum._registry`` holds the two conversion points --
``registry_output_columns`` for a kind's primary batch and
``registry_secondary_columns`` for one of its named secondary batches -- and
this module is the proof that the conversion is complete.

Wave 6B `#164`: the analyzer itself (AST plumbing, the accounting rules,
ferrum's own known dispositions and rechecked populations) moved to
``tests/_registry_consumer_lint.py`` -- an analyzer that grows every review
round belongs in its own module with its own unit tests
(``tests/test_registry_consumer_lint_unit.py``), not embedded in the pytest
file that consumes it. This module is now the thin wrapper: it runs the
lint module against ``src/ferrum`` once and asserts the gates below, plus
the behavioral pin at the bottom that does not depend on source text at all.
See ``tests/_registry_consumer_lint.py``'s module docstring for what the
sweep proves and does not prove.
"""

from __future__ import annotations

import pathlib

import polars as pl
import pytest

from ferrum import _core

from tests import _registry_consumer_lint as lint

_SRC = pathlib.Path(__file__).parent.parent / "src" / "ferrum"

_FINDINGS = lint.run(_SRC)
_OUTER_BOUND = _FINDINGS.outer_bound
_DECLARED_NAMES = _FINDINGS.declared_names
_COLUMN_REFERENCES = _FINDINGS.column_references
_CONVERTED = _FINDINGS.converted
_CONVERTED_SECONDARY = _FINDINGS.converted_secondary


# ---------------------------------------------------------------------------
# The gates
# ---------------------------------------------------------------------------


def test_the_outer_bound_is_wide_and_real():
    """A vacuous outer bound would make every test below pass for free.

    The floors are deliberately far below the observed values: they catch a
    parse that silently found nothing, not a refactor.
    """
    assert len(_OUTER_BOUND) > 300, len(_OUTER_BOUND)
    assert len({o.literal for o in _OUTER_BOUND}) > 20
    assert len({o.path for o in _OUTER_BOUND}) > 10
    assert len(_DECLARED_NAMES) > 40, sorted(_DECLARED_NAMES)


def test_every_outer_bound_occurrence_is_accounted_for():
    """Zero unaccounted. This is the boundary claim.

    Each occurrence carries exactly one label from lint.ACCOUNTING; a
    syntactic position no rule describes fails here rather than falling off
    the end of a list somebody wrote by hand.
    """
    unaccounted = [o for o in _OUTER_BOUND if o.label not in lint.LABELS]
    assert not unaccounted, "\n".join(
        f"{o.path}:{o.lineno} [{o.function}] {o.literal!r} -- {o.label}; "
        "add a rule to lint.ACCOUNTING (with its justification) or convert the site"
        for o in unaccounted
    )


def test_every_accounting_rule_fires():
    """No rule may be decoration.

    A label that matches nothing is either dead scaffolding or a rule whose
    predicate silently stopped working -- both of which would let the rules
    beneath it swallow occurrences they do not describe.
    """
    seen = {o.label for o in _OUTER_BOUND}
    dead = [label for label in lint.LABELS if label not in seen]
    assert not dead, f"accounting rules matched by no occurrence: {dead}"


def test_non_channel_keyword_population_is_rechecked():
    """A new keyword-argument name in this bucket must be re-justified.

    lint.ACCOUNTING's non-channel-keyword justification is an enumeration,
    not a structural predicate -- a keyword parameter that happens to take a
    column name out of a transform batch would land here unnoticed
    otherwise. New members fail loudly; a member that disappears must be
    removed too, so this set cannot go stale in either direction (mirrors
    ``test_every_disposition_is_still_live``'s discipline for
    lint.DISPOSITIONED).
    """
    observed = {o.detail for o in _OUTER_BOUND if o.label == "non-channel-keyword"}
    new, stale = lint.population_drift(observed, lint.KNOWN_NON_CHANNEL_KEYWORDS)
    assert not new, (
        f"new non-channel-keyword argument name(s) {sorted(new)}: confirm none of "
        "them takes a column name out of a transform batch (if one does, it belongs "
        "in the column-reference rule instead), then add it to "
        "lint.KNOWN_NON_CHANNEL_KEYWORDS with a one-line reason"
    )
    assert not stale, f"lint.KNOWN_NON_CHANNEL_KEYWORDS entries no longer observed: {sorted(stale)}"


def test_call_argument_population_is_rechecked():
    """A new callee in this bucket must be re-justified.

    lint.ACCOUNTING's call-argument justification is illustrative examples,
    backed by an architectural premise ("Python only ever holds frames it
    materialised itself") that holds per-callee, not for the bucket as a
    whole. New members fail loudly and must be checked against that premise
    before being added; a member that disappears must be removed too.
    """
    observed = {o.detail for o in _OUTER_BOUND if o.label == "call-argument"}
    new, stale = lint.population_drift(observed, lint.KNOWN_CALL_ARGUMENT_CALLEES)
    assert not new, (
        f"new call-argument callee(s) {sorted(new)}: confirm the call does not read "
        "a transform's output batch (if it does, the site needs a registry_output_"
        "columns conversion, not this bucket), then add it to "
        "lint.KNOWN_CALL_ARGUMENT_CALLEES with a one-line reason"
    )
    assert not stale, (
        f"lint.KNOWN_CALL_ARGUMENT_CALLEES entries no longer observed: {sorted(stale)}"
    )


def test_no_column_reference_is_a_bare_literal():
    """The M6a conversion is complete over every channel position in the tree."""
    offenders = lint.undispositioned(_FINDINGS)
    assert not offenders, "\n".join(
        f"{o.path}:{o.lineno} [{o.function}] {o.literal!r} in a channel position "
        f"-- a registry-declared output column embedded as a bare literal; read it "
        f"through {lint.PRIMARY_LOOKUP}(kind, [...]) (or {lint.SECONDARY_LOOKUP}(kind, key, "
        f"[...]) for a named secondary batch), or add a reasoned entry to "
        f"lint.DISPOSITIONED keyed {o.site!r}"
        for o in offenders
    )


def test_every_disposition_is_still_live():
    """A disposition that no longer matches a column-reference hit is stale.

    This doubles as the coverage floor for the classifier itself: every
    lint.DISPOSITIONED entry must still be *recognised* as a channel
    position, so a predicate that quietly stopped matching consumer code
    fails here in bulk instead of letting the gate above pass vacuously.
    """
    stale = lint.stale_dispositions(_FINDINGS)
    assert not stale, (
        f"lint.DISPOSITIONED entries no longer matched by any channel position: {stale}"
    )


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


def test_conversion_sites_were_found():
    """The drift guards below are derived from call sites; an empty derivation
    would make them pass vacuously."""
    assert len(_CONVERTED) >= 10, _CONVERTED
    assert len(_CONVERTED_SECONDARY) >= 3, _CONVERTED_SECONDARY


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


@pytest.mark.parametrize("site", sorted(_CONVERTED_SECONDARY))
def test_every_consumed_secondary_column_is_a_real_registry_output(site):
    """The same mirror for the secondary batches: every ``(kind, key)`` a
    consumer names is declared, and every column it reads out of that batch
    is in that batch's schema -- on every code path, not just the executed
    ones."""
    kind, key = site
    batches = dict(_core.transform_registry(kind)["secondary_outputs"])
    assert key in batches, (
        f"{kind}: Python reads a secondary batch {key!r} that "
        f"transform_registry({kind!r})['secondary_outputs'] does not declare "
        f"({sorted(batches)})"
    )
    declared = {name for name, _dtype in batches[key]}
    missing = sorted(_CONVERTED_SECONDARY[site] - declared)
    assert not missing, (
        f"{kind}/{key}: Python reads column(s) {missing} that the declared "
        f"schema does not carry ({sorted(declared)})"
    )


def test_residuals_qq_panel_titles_its_y_axis_without_binding_a_column():
    """Behavioural pin on the built panel, not on the source text.

    `mark_qq` draws its own sample quantiles on y and refuses a
    field-bearing y= outright, so the residuals Q-Q panel titles that axis
    with a fieldless `Y(title=...)` rather than binding the `qq`
    transform's declared `sample` output by name. Its x channel is not a
    qq output either (it names the panel's own input column) -- so a
    reformat cannot break this and a re-binding cannot slip past it.
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

    y_ch = panel._encoding["y"]
    assert y_ch.field is None
    assert y_ch.option("title") == "Sample quantiles"
    assert panel._encoding["x"].field not in declared
    assert panel.to_svg()
