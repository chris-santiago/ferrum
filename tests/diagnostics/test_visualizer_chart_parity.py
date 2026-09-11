"""Mechanical visualizer <-> chart-builder signature parity (batch G T9).

F-L11-06 + F-L12-09 + F-L10-07: every ``FerrumVisualizer`` subclass must
accept, as explicit keyword-only ``__init__`` parameters, every kwarg the
chart builder its ``_build_chart`` invokes documents -- including the
uniform override quartet ``mark=``/``encode=``/``properties=``/``layers=``
-- and forward them. This module is the drift-proof mechanical check the
batch G decision doc (`.claude/output/decisions/2026-09-10-batch-g-diagnostics-correctness-decision.md`,
G5) calls for, patterned on
``test_core_stub_parity.py::test_programmatic_signature_parity``: a live
signature diff with an explicit, rationale-carrying skip-map for the three
adjudicated shape drifts, rather than a hand-maintained "did I remember"
checklist.

For each visualizer the "target" is the actual function ``_build_chart``
(or the equivalent no-model ``fit`` override) invokes -- the private
``_*_from_source``/``_*_from_dataframe`` builder in the common case. Only
that function's **keyword-only** parameters are checked: every builder
here takes its data (``source``/``df``/``scores``/...) as a positional-or-
keyword parameter before the ``*``, so restricting the diff to keyword-only
names excludes the data parameter with no per-entry exclude list to
maintain. ``CalibrationVisualizer`` is the one exception -- its target is
the *public* ``calibration_chart`` function, specifically because the
decision doc's skip-map entry for it (``*models`` <-> ``compare=``) only
exists on the public signature (the private builder never had ``compare=``
to begin with, since compare-resolution happens before the builder runs).
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, NamedTuple

import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

import ferrum
from ferrum.diagnostics.visualizers._base import FerrumVisualizer
from ferrum.diagnostics.visualizers._classification import (
    CalibrationVisualizer,
    ClassBalanceVisualizer,
    ClassificationReportVisualizer,
    ClassPredictionErrorVisualizer,
    ConfusionMatrixVisualizer,
    DiscriminationThresholdVisualizer,
    PRVisualizer,
    ROCVisualizer,
)
from ferrum.diagnostics.visualizers._clustering import (
    ElbowVisualizer,
    InterclusterDistanceVisualizer,
    ManifoldVisualizer,
    PCAVarianceVisualizer,
    SilhouetteVisualizer,
)
from ferrum.diagnostics.visualizers._explanation import (
    FeatureImportancesVisualizer,
    SHAPBarVisualizer,
    SHAPBeeswarmVisualizer,
    SHAPVisualizer,
    SHAPWaterfallVisualizer,
)
from ferrum.diagnostics.visualizers._ranking import (
    ParallelCoordinatesVisualizer,
    Rank1DVisualizer,
    Rank2DVisualizer,
)
from ferrum.diagnostics.visualizers._regression import (
    CooksDistanceVisualizer,
    PredictionErrorVisualizer,
    ResidualsVisualizer,
)
from ferrum.diagnostics.visualizers._selection import (
    AlphaSelectionVisualizer,
    CVScoresVisualizer,
    LearningCurveVisualizer,
    ValidationCurveVisualizer,
)
from ferrum.plots.classification import (
    _calibration_chart_from_source,
    _class_balance_chart_from_dataframe,
    _class_prediction_error_chart_from_source,
    _classification_report_chart,
    _confusion_chart_from_source,
    _discrimination_threshold_chart_from_source,
    _pr_chart_from_source,
    _roc_chart_from_source,
    calibration_chart,
)
from ferrum.plots.clustering import (
    _elbow_chart_from_source,
    _intercluster_distance_chart_from_source,
    _manifold_chart_from_source,
    _pca_scree_chart_from_source,
    _silhouette_chart_from_source,
)
from ferrum.plots.explanation import (
    _importance_chart_from_source,
    _shap_bar_chart_from_source,
    _shap_beeswarm_chart_from_source,
    _shap_waterfall_chart_from_source,
)
from ferrum.plots.model_selection import (
    _alpha_selection_chart_from_source,
    _cv_scores_chart_from_source,
    _learning_curve_chart_from_source,
    _validation_curve_chart_from_source,
)
from ferrum.plots.ranking import (
    _parallel_coords_chart_from_dataframe,
    _rank1d_chart_from_dataframe,
    _rank2d_chart_from_dataframe,
)
from ferrum.plots.regression import (
    _prediction_error_chart_from_source,
    _residuals_chart_from_source,
)
from tests.fixtures import load_dataset, load_fixture

# ---------------------------------------------------------------------------
# Registry: visualizer -> the function its _build_chart (or fit override)
# actually invokes. Every `FerrumVisualizer` subclass must appear either
# here or in `_REGISTRY_EXCLUSIONS` below, with a rationale -- enforced by
# `test_parity_registry_is_complete`, so a visualizer can never silently
# escape this module's drift-proof check by omission.
# ---------------------------------------------------------------------------


class ParityCase(NamedTuple):
    visualizer: type[FerrumVisualizer]
    target: Callable[..., Any]
    # Structural params on `target` that are never expected on the
    # visualizer, beyond what the keyword-only filter already excludes.
    # Only needed for CalibrationVisualizer, whose target is the *public*
    # `calibration_chart` (chosen so the *models<->compare= skip-map entry
    # checks something real) -- that function's precomputed-path y_true=/
    # y_pred= keyword-only params have no equivalent on CalibrationVisualizer,
    # which only ever wraps fitted models via its variadic *models.
    exclude: frozenset[str] = frozenset()
    # An explicit override for the function whose *defaults* the visualizer
    # must match. Normally nothing sets this: `_defaults_target` below
    # resolves the PUBLIC figure function automatically, because a default
    # is a user-facing promise made by the public wrapper. `residuals_chart`
    # re-defaults `panels=` on its way into `_residuals_chart_from_source`,
    # so diffing against the private builder would compare the visualizer
    # to a number no user ever reads (the blind spot the close review
    # named). Auto-resolution rather than opt-in (cycle 3) means a public
    # wrapper that starts re-defaulting tomorrow is caught by construction,
    # not when someone remembers to add a row here.
    default_target: Callable[..., Any] | None = None


_PARITY_REGISTRY: list[ParityCase] = [
    # -- classification -----------------------------------------------------
    ParityCase(ROCVisualizer, _roc_chart_from_source),
    ParityCase(PRVisualizer, _pr_chart_from_source),
    ParityCase(ConfusionMatrixVisualizer, _confusion_chart_from_source),
    ParityCase(ClassPredictionErrorVisualizer, _class_prediction_error_chart_from_source),
    ParityCase(DiscriminationThresholdVisualizer, _discrimination_threshold_chart_from_source),
    # Calibration targets the PUBLIC function (not the private builder) so
    # the *models<->compare= skip-map entry below is checking something
    # real: the private builder never had compare= (compare-resolution
    # happens before the builder runs), only the public wrapper does.
    ParityCase(
        CalibrationVisualizer,
        calibration_chart,
        exclude=frozenset({"y_true", "y_pred"}),
    ),
    ParityCase(ClassificationReportVisualizer, _classification_report_chart),
    ParityCase(ClassBalanceVisualizer, _class_balance_chart_from_dataframe),
    # -- regression -----------------------------------------------------
    ParityCase(ResidualsVisualizer, _residuals_chart_from_source),
    ParityCase(PredictionErrorVisualizer, _prediction_error_chart_from_source),
    ParityCase(CooksDistanceVisualizer, _residuals_chart_from_source),
    # -- explanation -----------------------------------------------------
    ParityCase(FeatureImportancesVisualizer, _importance_chart_from_source),
    ParityCase(SHAPBeeswarmVisualizer, _shap_beeswarm_chart_from_source),
    ParityCase(SHAPBarVisualizer, _shap_bar_chart_from_source),
    ParityCase(SHAPWaterfallVisualizer, _shap_waterfall_chart_from_source),
    # SHAPVisualizer (deprecated kind=-dispatch wrapper) is checked against
    # all three SHAP builders it can route to -- `registered` (below) is a
    # set comprehension, so three rows keyed on the same visualizer just
    # union their coverage rather than conflicting. Every kwarg any of the
    # three builders documents must be reachable, regardless of which
    # `kind=` branch a caller picks.
    ParityCase(SHAPVisualizer, _shap_beeswarm_chart_from_source),
    ParityCase(SHAPVisualizer, _shap_bar_chart_from_source),
    ParityCase(SHAPVisualizer, _shap_waterfall_chart_from_source),
    # -- clustering -----------------------------------------------------
    ParityCase(SilhouetteVisualizer, _silhouette_chart_from_source),
    ParityCase(ElbowVisualizer, _elbow_chart_from_source),
    ParityCase(ManifoldVisualizer, _manifold_chart_from_source),
    ParityCase(InterclusterDistanceVisualizer, _intercluster_distance_chart_from_source),
    ParityCase(PCAVarianceVisualizer, _pca_scree_chart_from_source),
    # -- ranking (no-model) -----------------------------------------------------
    ParityCase(Rank1DVisualizer, _rank1d_chart_from_dataframe),
    ParityCase(Rank2DVisualizer, _rank2d_chart_from_dataframe),
    ParityCase(ParallelCoordinatesVisualizer, _parallel_coords_chart_from_dataframe),
    # -- model selection -----------------------------------------------------
    ParityCase(LearningCurveVisualizer, _learning_curve_chart_from_source),
    ParityCase(ValidationCurveVisualizer, _validation_curve_chart_from_source),
    ParityCase(CVScoresVisualizer, _cv_scores_chart_from_source),
    ParityCase(AlphaSelectionVisualizer, _alpha_selection_chart_from_source),
]

# Visualizers that gained an explicit `compare=` constructor param routing
# through `_resolve_source` (a real addition, not a skip-mapped drift) --
# checked separately below since `compare=` is not part of the *builder*
# signature the main mechanical diff targets (compare-resolution happens
# before the builder runs).
_MUST_HAVE_COMPARE: tuple[type[FerrumVisualizer], ...] = (
    ROCVisualizer,
    PRVisualizer,
    DiscriminationThresholdVisualizer,
)

# Visualizers deliberately absent from `_PARITY_REGISTRY`, with the
# rationale a stale entry here must keep being true (checked by
# `test_parity_registry_is_complete`) -- the exclusion-list counterpart of
# `_SKIP_ENTRIES`, so a visualizer can't silently escape either list. Empty
# today: `SHAPVisualizer` (the cycle-1 exclusion) is covered by three
# `ParityCase` rows above instead -- every `FerrumVisualizer` subclass now
# gets real mechanical coverage.
_REGISTRY_EXCLUSIONS: dict[type[FerrumVisualizer], str] = {}


def _all_visualizer_subclasses() -> set[type[FerrumVisualizer]]:
    """Every `FerrumVisualizer` subclass loaded so far, found by walking
    `__subclasses__` recursively rather than hand-listing them -- so a
    visualizer added next month is caught by construction, not by whether
    someone remembered to add it here too. Relies on every visualizer
    module already being imported above (which loads every subclass onto
    `FerrumVisualizer.__subclasses__()` as an import side effect); a
    visualizer this test file doesn't import at all would not be counted,
    but that's true of the registry and skip-map too, and none of the
    three lists can silently diverge from each other under that shared
    assumption.
    """
    seen: set[type[FerrumVisualizer]] = set()
    stack = [FerrumVisualizer]
    while stack:
        cls = stack.pop()
        for sub in cls.__subclasses__():
            if sub not in seen:
                seen.add(sub)
                stack.append(sub)
    return seen


def test_parity_registry_is_complete() -> None:
    """Every `FerrumVisualizer` subclass is covered by the mechanical
    parity diff (`_PARITY_REGISTRY`) or by a rationale-carrying entry in
    `_REGISTRY_EXCLUSIONS` -- never silently absent from both. Closes the
    exact gap a future visualizer could otherwise use to escape the
    drift-proof check this module exists to install.
    """
    registered = {case.visualizer for case in _PARITY_REGISTRY}
    excluded = set(_REGISTRY_EXCLUSIONS)

    overlap = registered & excluded
    assert not overlap, (
        f"visualizer(s) in both _PARITY_REGISTRY and _REGISTRY_EXCLUSIONS: "
        f"{sorted(c.__name__ for c in overlap)}"
    )

    all_subclasses = _all_visualizer_subclasses()
    uncovered = all_subclasses - registered - excluded
    assert not uncovered, (
        "FerrumVisualizer subclass(es) missing from both _PARITY_REGISTRY "
        f"and _REGISTRY_EXCLUSIONS: {sorted(c.__name__ for c in uncovered)}"
    )

    empty_rationale = [
        cls.__name__ for cls, rationale in _REGISTRY_EXCLUSIONS.items() if not rationale.strip()
    ]
    assert not empty_rationale, f"exclusion entries with an empty rationale: {empty_rationale}"

    orphaned = [cls.__name__ for cls in _REGISTRY_EXCLUSIONS if cls not in all_subclasses]
    assert not orphaned, (
        f"exclusion entries for classes that are no longer a FerrumVisualizer "
        f"subclass (stale -- remove the entry): {orphaned}"
    )


class SkipEntry(NamedTuple):
    visualizer: type[FerrumVisualizer]
    param: str
    reference: Callable[..., Any]
    rationale: str


# The three adjudicated shape drifts from the decision doc (G5) -- existing
# spellings kept for back-compat, unification logged as a follow-up. Each
# entry's `reference` is the function that actually carries `param`, so
# `test_skip_map_entries_are_not_stale` can independently verify the drift
# is still real.
_SKIP_ENTRIES: list[SkipEntry] = [
    SkipEntry(
        ROCVisualizer,
        "average",
        _roc_chart_from_source,
        "ROCVisualizer keeps its pre-existing micro=/macro= booleans "
        "(translated internally to average= before calling "
        "_roc_chart_from_source) for back-compat; adjudicated shape "
        "drift, not unified with average= (batch G decision doc, G5).",
    ),
    SkipEntry(
        CooksDistanceVisualizer,
        "cook_threshold",
        _residuals_chart_from_source,
        "CooksDistanceVisualizer keeps its pre-existing threshold= "
        "spelling for back-compat; adjudicated shape drift, not unified "
        "with cook_threshold= (batch G decision doc, G5).",
    ),
    SkipEntry(
        CalibrationVisualizer,
        "compare",
        calibration_chart,
        "CalibrationVisualizer already supports multi-model overlay via "
        "its variadic *models constructor; not unified with compare= "
        "(batch G decision doc, G5).",
    ),
]


def _keyword_only_param_names(fn: Callable[..., Any]) -> set[str]:
    """Names of `fn`'s KEYWORD_ONLY parameters -- never POSITIONAL_OR_KEYWORD
    ones, even though `self`/`model`/`X`/`y` (the only positional-or-keyword
    params anywhere in this family) would otherwise slip through a looser
    filter. The spec's own words are "explicit **keyword-only** parameters"
    -- a param a caller could pass positionally doesn't satisfy that, so
    this one helper enforces both sides of the diff (a visualizer's own
    params via `_own_param_names`, and a target builder's params) through
    the same rule, rather than two different filters that could drift.
    """
    sig = inspect.signature(fn)
    return {
        name
        for name, param in sig.parameters.items()
        if param.kind is inspect.Parameter.KEYWORD_ONLY
    }


def _own_param_names(cls: type) -> set[str]:
    """A visualizer's own explicit KEYWORD_ONLY `__init__` params (excludes
    `self` and the leading `model`/`model_class`/`alphas`/... positional-or-
    keyword params every visualizer's `__init__` starts with -- see
    `_keyword_only_param_names`).
    """
    return _keyword_only_param_names(cls.__init__)


_SKIP_MAP: dict[tuple[type, str], SkipEntry] = {
    (entry.visualizer, entry.param): entry for entry in _SKIP_ENTRIES
}


def test_visualizer_chart_parity() -> None:
    """Every builder kwarg a visualizer's `_build_chart` invokes must be an
    explicit, forwarded `__init__` param -- or a documented skip-map entry.
    """
    failures: list[str] = []
    for case in _PARITY_REGISTRY:
        own_params = _own_param_names(case.visualizer)
        target_params = _keyword_only_param_names(case.target) - case.exclude
        for name in sorted(target_params):
            if name in own_params:
                continue
            if (case.visualizer, name) in _SKIP_MAP:
                continue
            failures.append(
                f"{case.visualizer.__name__} is missing {name!r} "
                f"(present on {case.target.__qualname__})"
            )
    assert not failures, "Visualizer <-> chart-builder parameter drift detected:\n" + "\n".join(
        f"  {f}" for f in failures
    )


# ---------------------------------------------------------------------------
# Defaults parity (batch G close, design F12)
# ---------------------------------------------------------------------------
#
# Names alone are not the contract: a visualizer that accepts `alpha=` and a
# chart function that accepts `alpha=` still render differently if their
# defaults disagree, and nothing in the name diff notices. The close review
# found one live divergence that way (`ParallelCoordinatesVisualizer.alpha`
# 0.5 vs the builder's dead 0.3, since fixed at the builder) and named a
# second that the name diff structurally cannot see (`ResidualsVisualizer`
# vs the *public* `residuals_chart`'s `panels=`). This diff carries the same
# rationale-bearing, staleness-checked skip-map discipline as the name diff.


#: Private builders whose public wrapper is not simply the builder's name
#: with the decoration stripped. Two in the whole family; without them the
#: auto-resolution below would silently fall back to the private builder
#: for exactly these two, which is the opt-in failure mode all over again.
_PUBLIC_NAME_ALIASES = {
    "confusion_chart": "confusion_matrix_chart",
    "parallel_coords_chart": "parallel_coordinates_chart",
}


def _public_counterpart(builder: Callable[..., Any]) -> Callable[..., Any] | None:
    """The public `ferrum.*` figure function a private builder backs, if any.

    `_residuals_chart_from_source` -> `ferrum.residuals_chart`. Returns
    `None` for a builder with no public wrapper of that name, in which case
    the defaults diff falls back to the builder itself -- there is no
    user-facing default to compare against.
    """
    stem = builder.__name__.lstrip("_")
    for suffix in ("_from_source", "_from_dataframe"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    stem = _PUBLIC_NAME_ALIASES.get(stem, stem)
    public = getattr(ferrum, stem, None)
    return public if callable(public) else None


def _defaults_target(case: ParityCase) -> Callable[..., Any]:
    """The function whose defaults `case.visualizer` must match.

    An explicit `default_target=` wins; otherwise the public figure
    function when one exists; otherwise the private builder.
    """
    return case.default_target or _public_counterpart(case.target) or case.target


class DefaultSkipEntry(NamedTuple):
    visualizer: type[FerrumVisualizer]
    param: str
    reference: Callable[..., Any]
    rationale: str


_DEFAULT_SKIP_ENTRIES: list[DefaultSkipEntry] = [
    DefaultSkipEntry(
        ResidualsVisualizer,
        "panels",
        ferrum.residuals_chart,
        "residuals_chart defaults to the 4-panel diagnostic grid; the "
        "visualizer defaults to a single panel, which is what it has always "
        "rendered. Changing either default would break existing callers' "
        "output, so the divergence is deliberate and documented (batch G "
        "T9 / F-L10-07 -- the finding's symptom was that the 4-panel layout "
        "was *unreachable* from the visualizer, which panels='auto' fixed).",
    ),
    DefaultSkipEntry(
        CooksDistanceVisualizer,
        "panels",
        ferrum.residuals_chart,
        "CooksDistanceVisualizer exists to render the leverage panel, so it "
        "defaults to exactly that panel rather than to residuals_chart's "
        "4-panel grid -- the class's whole purpose, and its unchanged "
        "pre-existing behavior (pinned by "
        "test_cooks_distance_visualizer_default_panels_unchanged).",
    ),
]

_DEFAULT_SKIP_ENTRIES += [
    DefaultSkipEntry(
        Rank1DVisualizer,
        "algorithm",
        ferrum.rank1d_chart,
        "Two spellings of one default, not two defaults: rank1d_chart "
        'documents algorithm=None as "defaults to shapiro" and resolves it '
        "before calling the builder, while the visualizer names shapiro "
        "outright. Both render the identical chart when left unset (pinned "
        "by test_rank_visualizer_default_algorithm_matches_the_chart).",
    ),
    DefaultSkipEntry(
        Rank2DVisualizer,
        "algorithm",
        ferrum.rank2d_chart,
        "Same sentinel spelling as Rank1DVisualizer, with pearson as the resolved default.",
    ),
]

_DEFAULT_SKIP_MAP: dict[tuple[type, str], DefaultSkipEntry] = {
    (entry.visualizer, entry.param): entry for entry in _DEFAULT_SKIP_ENTRIES
}


def _keyword_only_defaults(fn: Callable[..., Any]) -> dict[str, Any]:
    """KEYWORD_ONLY parameter -> default, skipping params with no default.

    A required keyword-only parameter on the target (``sample_idx`` on the
    SHAP waterfall builder) has no default to diverge from; the visualizer
    supplying one is the adapter doing its job, not drift.
    """
    sig = inspect.signature(fn)
    return {
        name: param.default
        for name, param in sig.parameters.items()
        if param.kind is inspect.Parameter.KEYWORD_ONLY
        and param.default is not inspect.Parameter.empty
    }


def test_visualizer_chart_default_parity() -> None:
    """Every parameter a visualizer shares with its chart builder must
    default to the same value -- or carry a documented skip-map entry.
    """
    failures: list[str] = []
    for case in _PARITY_REGISTRY:
        reference = _defaults_target(case)
        own_defaults = _keyword_only_defaults(case.visualizer.__init__)
        target_defaults = _keyword_only_defaults(reference)
        for name in sorted(set(own_defaults) & (set(target_defaults) - case.exclude)):
            if own_defaults[name] == target_defaults[name]:
                continue
            if (case.visualizer, name) in _DEFAULT_SKIP_MAP:
                continue
            failures.append(
                f"{case.visualizer.__name__}.{name} defaults to "
                f"{own_defaults[name]!r} but {reference.__qualname__}.{name} "
                f"defaults to {target_defaults[name]!r}"
            )
    assert not failures, "Visualizer <-> chart-builder default drift detected:\n" + "\n".join(
        f"  {f}" for f in failures
    )


def test_default_skip_map_entries_are_not_stale() -> None:
    """Every defaults skip-map entry must carry a rationale and still
    describe a real divergence -- once the two sides agree, the entry is
    stale and must be removed, not left to rot.
    """
    stale: list[str] = []
    for entry in _DEFAULT_SKIP_ENTRIES:
        assert entry.rationale.strip(), (
            f"{entry.visualizer.__name__}.{entry.param} defaults skip-map "
            "entry has an empty rationale."
        )
        own_defaults = _keyword_only_defaults(entry.visualizer.__init__)
        reference_defaults = _keyword_only_defaults(entry.reference)
        if entry.param not in own_defaults or entry.param not in reference_defaults:
            stale.append(
                f"{entry.visualizer.__name__}.{entry.param} is stale -- one "
                f"side no longer has a defaulted keyword-only {entry.param!r}."
            )
        elif own_defaults[entry.param] == reference_defaults[entry.param]:
            stale.append(
                f"{entry.visualizer.__name__}.{entry.param} is stale -- both "
                f"sides now default to {own_defaults[entry.param]!r}."
            )
    assert not stale, "Stale defaults skip-map entries:\n" + "\n".join(f"  {f}" for f in stale)


def test_parallel_coordinates_alpha_default_is_shared() -> None:
    """The close review's live defaults divergence: the private builder's
    ``alpha=0.3`` was dead (both callers always pass ``alpha``), so a third
    caller would silently have gotten a different look than either public
    surface documents. All three now agree on 0.5.
    """
    builder_alpha = _keyword_only_defaults(_parallel_coords_chart_from_dataframe)["alpha"]
    visualizer_alpha = _keyword_only_defaults(ParallelCoordinatesVisualizer.__init__)["alpha"]
    public_alpha = _keyword_only_defaults(ferrum.parallel_coordinates_chart)["alpha"]
    assert builder_alpha == visualizer_alpha == public_alpha == 0.5


def test_skip_map_entries_are_not_stale() -> None:
    """Every skip-map entry must carry a rationale and still describe a
    real difference -- a param the visualizer still lacks, that its
    reference function still documents. Once either side changes, the
    entry is stale and must be removed or updated, not left to rot.
    """
    stale: list[str] = []
    for entry in _SKIP_ENTRIES:
        assert entry.rationale.strip(), (
            f"{entry.visualizer.__name__}.{entry.param} skip-map entry has an empty rationale."
        )
        own_params = _own_param_names(entry.visualizer)
        reference_params = _keyword_only_param_names(entry.reference)
        if entry.param in own_params:
            stale.append(
                f"{entry.visualizer.__name__}.{entry.param} is stale -- the "
                f"visualizer now defines {entry.param!r} itself."
            )
        if entry.param not in reference_params:
            stale.append(
                f"{entry.visualizer.__name__}.{entry.param} is stale -- "
                f"{entry.reference.__qualname__} no longer has a keyword-only "
                f"{entry.param!r} parameter."
            )
    assert not stale, "Stale skip-map entries:\n" + "\n".join(f"  {f}" for f in stale)


def test_roc_pr_discrimination_threshold_accept_compare() -> None:
    """checkpoint decision 3 (full parity incl. the override quartet) plus
    the G5 drift map name `compare=` explicitly for exactly these three
    curve-family visualizers -- not the builder-level mechanical diff
    (compare-resolution happens before the builder runs), so it's asserted
    directly here.
    """
    for visualizer in _MUST_HAVE_COMPARE:
        assert "compare" in _own_param_names(visualizer), (
            f"{visualizer.__name__} is missing compare= (batch G decision doc, G5)."
        )


def test_unknown_kwarg_raises_type_error() -> None:
    """FerrumVisualizer's base __init__ no longer swallows unknown kwargs
    via **_extra -- a typo now raises TypeError naming the kwarg, instead
    of silently doing nothing (F-L11-06).

    The base class itself is the direct proof: every subclass declares its
    own explicit keyword-only params with no **kwargs catch-all of its own,
    so an unknown kwarg on a subclass call was *already* rejected by the
    subclass's own signature before this fix -- only a bare
    ``FerrumVisualizer(...)`` construction actually exercised the base's
    former ``**_extra`` swallow.
    """
    with pytest.raises(TypeError):
        FerrumVisualizer(model=None, aggregate="micro")
    # Subclass-level regression coverage: every visualizer's own signature
    # rejects an unrecognized kwarg too (not specific to the base-class
    # fix, but a real, useful invariant for the family).
    with pytest.raises(TypeError):
        ferrum.ResidualsVisualizer(load_fixture("regression_ridge"), pnaels="auto")
    with pytest.raises(TypeError):
        ferrum.ROCVisualizer(load_fixture("binary_logistic"), aggregate="micro")


# ---------------------------------------------------------------------------
# Behavioral spot-checks (spec Sec 9.6) -- RED-proven: each assertion fails
# against the pre-sweep signatures (either TypeError on the new kwarg, or
# the old behavior the new kwarg is supposed to change).
# ---------------------------------------------------------------------------


def test_residuals_visualizer_panels_auto_renders_four_panels() -> None:
    """ResidualsVisualizer(panels="auto") renders the same 4-panel grid as
    residuals_chart(panels="auto") -- the trap this batch closes: a naive
    forward of "auto" to the builder falls back to single-panel, since the
    builder's own vocabulary is None/"single"/a list, not "auto".
    """
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    y = df["y"]

    figure_svg = ferrum.residuals_chart(model, X, y, panels="auto").to_svg()
    visualizer_svg = ferrum.ResidualsVisualizer(model, panels="auto").fit(X, y).show().to_svg()

    panel_titles = ("Residuals vs Fitted", "Normal Q", "Scale", "Residuals vs Leverage")
    for title in panel_titles:
        assert title in figure_svg, f"residuals_chart(panels='auto') is missing {title!r}"
        assert title in visualizer_svg, (
            f"ResidualsVisualizer(panels='auto') is missing {title!r} -- "
            "it rendered fewer than 4 panels."
        )

    # The visualizer's own default (panels=None) stays single-panel --
    # back-compat, not a behavior change for existing callers.
    default_svg = ferrum.ResidualsVisualizer(model).fit(X, y).show().to_svg()
    assert "Normal Q" not in default_svg
    assert "Scale" not in default_svg


def test_cooks_distance_visualizer_default_panels_unchanged() -> None:
    """CooksDistanceVisualizer's new `panels=` kwarg defaults to exactly
    the leverage panel it always rendered -- adding the kwarg must not
    change the unset-default behavior.
    """
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    y = df["y"]

    svg = ferrum.CooksDistanceVisualizer(model).fit(X, y).show().to_svg()
    assert "Residuals vs Leverage" in svg
    assert "Normal Q" not in svg
    assert "Scale" not in svg

    auto_svg = ferrum.CooksDistanceVisualizer(model, panels="auto").fit(X, y).show().to_svg()
    assert "Normal Q" in auto_svg
    assert "Scale" in auto_svg


def test_shap_beeswarm_visualizer_zero_line_false_drops_rule() -> None:
    """SHAPBeeswarmVisualizer(zero_line=False) drops the `_ref_zero` rule
    -- forwarding proof for the newly-added `zero_line=` kwarg.
    """
    model = load_fixture("regression_rf")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    y = df["y"]

    with_rule = ferrum.SHAPBeeswarmVisualizer(model, random_state=0).fit(X, y).show()
    without_rule = (
        ferrum.SHAPBeeswarmVisualizer(model, random_state=0, zero_line=False).fit(X, y).show()
    )

    with_cols = with_rule._resolve_pending()._data.columns
    without_cols = without_rule._resolve_pending()._data.columns
    assert "_ref_zero" in with_cols
    assert "_ref_zero" not in without_cols


def test_shap_visualizer_zero_line_and_per_class_forwarded() -> None:
    """The deprecated SHAPVisualizer(kind="beeswarm") reaches the exact
    same `zero_line=` rule its dedicated sibling (SHAPBeeswarmVisualizer)
    does -- letter-vs-intent gap from the spec review: a builder kwarg
    unreachable from a public visualizer is exactly the finding this whole
    task closes, deprecated or not. `per_class` is checked structurally
    (an explicit, forwarded param) since a per-class facet needs a
    multi-class fixture this module doesn't otherwise carry.
    """
    import warnings

    model = load_fixture("regression_rf")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    y = df["y"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with_rule = ferrum.SHAPVisualizer(model, kind="beeswarm", random_state=0).fit(X, y).show()
        without_rule = (
            ferrum.SHAPVisualizer(model, kind="beeswarm", random_state=0, zero_line=False)
            .fit(X, y)
            .show()
        )

    with_cols = with_rule._resolve_pending()._data.columns
    without_cols = without_rule._resolve_pending()._data.columns
    assert "_ref_zero" in with_cols
    assert "_ref_zero" not in without_cols

    assert "per_class" in _own_param_names(SHAPVisualizer)


def test_roc_visualizer_annotate_auc_false_drops_label() -> None:
    """ROCVisualizer(annotate_auc=False) drops the per-curve AUC text
    label -- forwarding proof for the newly-added `annotate_auc=` kwarg.
    """
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    y = df["y"]

    with_label = ferrum.ROCVisualizer(model).fit(X, y).show().to_svg()
    without_label = ferrum.ROCVisualizer(model, annotate_auc=False).fit(X, y).show().to_svg()

    assert "AUC = 0." in with_label
    assert "AUC = 0." not in without_label


def test_roc_visualizer_compare_overlays_second_model() -> None:
    """ROCVisualizer(compare={...}) actually reaches a ComparedModelSource
    -- not just an accepted-but-inert kwarg.
    """
    model = load_fixture("binary_logistic")
    rival = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    y = df["y"]

    viz = ferrum.ROCVisualizer(model, compare={"rival": rival}).fit(X, y)
    assert viz._source.model_names == ["base", "rival"]


def _weak_and_strong_binary_models():
    """A deliberately weak base classifier and a strong rival, fit on the
    same data -- the quality review's weak-base/strong-rival repro shape,
    for pinning that a visualizer's headline metric under `compare=`
    describes the base model, not a pooled aggregate or the rival's
    optimum. `DummyClassifier` predicts from the class prior alone (no
    signal from X), so its ROC/PR/threshold curves differ sharply from a
    real classifier's -- pooling the two would move every headline number.
    """
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    y = df["y"]
    weak = DummyClassifier(strategy="stratified", random_state=0).fit(X, y)
    strong = LogisticRegression(random_state=0).fit(X, y)
    return weak, strong, X, y


def test_roc_visualizer_compare_metrics_describe_base_model() -> None:
    """ROCVisualizer(weak, compare={"strong": strong})._metrics['auc_mean']
    equals the weak model's own solo run (and .score()), not a pooled mean
    across every compared model's rows -- RED-proven regression for the
    quality review's S3 finding (repr reported the rival's number while
    .score() stayed base-only).
    """
    weak, strong, X, y = _weak_and_strong_binary_models()

    solo = ferrum.ROCVisualizer(weak, random_state=0).fit(X, y)
    compared = ferrum.ROCVisualizer(weak, compare={"strong": strong}, random_state=0).fit(X, y)

    # ROCVisualizer.score() computes AUC directly from self.model (not from
    # _materialize's frame), so it's independent of compare= by
    # construction; the real proof is _metrics agreeing with it and with
    # the uncompared solo run -- the exact two quantities the review found
    # disagreeing (repr's pooled/rival number vs. score()'s base-only one).
    assert compared._metrics["auc_mean"] == pytest.approx(solo._metrics["auc_mean"])
    assert compared._metrics["auc_mean"] == pytest.approx(compared.score(X, y))
    assert compared.score(X, y) == pytest.approx(solo.score(X, y))


def test_pr_visualizer_compare_metrics_describe_base_model() -> None:
    """PRVisualizer(weak, compare={"strong": strong})._metrics['ap_mean']
    equals the weak model's own solo run, not a pooled mean across every
    compared model's rows -- same S3 finding, PR's softer variant.
    """
    weak, strong, X, y = _weak_and_strong_binary_models()

    solo = ferrum.PRVisualizer(weak, random_state=0).fit(X, y)
    compared = ferrum.PRVisualizer(weak, compare={"strong": strong}, random_state=0).fit(X, y)

    assert compared._metrics["ap_mean"] == pytest.approx(solo._metrics["ap_mean"])
    assert compared.score(X, y) == pytest.approx(weak.score(X, y))


def test_discrimination_threshold_visualizer_compare_metrics_describe_base_model() -> None:
    """DiscriminationThresholdVisualizer(weak, compare={"strong": strong})'s
    best_threshold/best_f1 equal the weak model's own solo run, not the
    rival's pooled-argmax optimum -- the exact S3 finding the quality
    review demonstrated live (repr showed the rival's threshold/F1 while
    .score() stayed base-only).
    """
    weak, strong, X, y = _weak_and_strong_binary_models()

    solo = ferrum.DiscriminationThresholdVisualizer(weak, random_state=0).fit(X, y)
    compared = ferrum.DiscriminationThresholdVisualizer(
        weak, compare={"strong": strong}, random_state=0
    ).fit(X, y)

    assert compared._metrics["best_threshold"] == pytest.approx(solo._metrics["best_threshold"])
    assert compared._metrics["best_f1"] == pytest.approx(solo._metrics["best_f1"])
    assert compared.score(X, y) == pytest.approx(weak.score(X, y))


def test_quartet_reaches_finalize_chart() -> None:
    """properties={} on a visualizer reaches _finalize_chart and changes
    the rendered SVG -- proof the mark=/encode=/properties=/layers=
    quartet is forwarded, not merely accepted (spec Sec 9.6).
    """
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    y = df["y"]

    svg = ferrum.ResidualsVisualizer(model, properties={"width": 611}).fit(X, y).show().to_svg()
    assert 'width="611"' in svg


def test_quartet_reaches_finalize_chart_classification() -> None:
    """Same proof on the classification side of the family (ROCVisualizer),
    since ROC/PR/DiscriminationThreshold route through `_build_source`
    rather than the base `fit`.
    """
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    y = df["y"]

    svg = ferrum.ROCVisualizer(model, properties={"width": 622}).fit(X, y).show().to_svg()
    assert 'width="622"' in svg


# ---------------------------------------------------------------------------
# Documented surface (batch G close, design F7)
# ---------------------------------------------------------------------------


def test_visualizer_family_docstring_describes_the_real_surface() -> None:
    """The family module docstring is the first thing a reader meets, so it
    must name the methods that exist and claim only the composability they
    support. It advertised `.predict` / `get_params` / `set_params` and
    `GridSearchCV` composability, none of which the base class implements --
    and without `get_params`, `sklearn.base.clone` (and therefore every
    sklearn meta-estimator) cannot work at all.
    """
    from ferrum.diagnostics import visualizers

    doc = visualizers.__doc__ or ""
    surface = {name for name in dir(FerrumVisualizer) if not name.startswith("_")}
    assert surface == {"fit", "has_score", "score", "show"}
    for name in sorted(surface):
        assert f"``{name}" in doc, f"the family docstring never mentions {name}()"

    # While the family lacks `get_params`, it may not advertise the
    # clone()-dependent sklearn composability that needs it. If someone
    # implements `get_params` later, this guard steps aside on its own.
    if not hasattr(FerrumVisualizer, "get_params"):
        assert "composable inside sklearn" not in doc


def test_compare_rejects_the_reserved_base_key_on_every_visualizer() -> None:
    """`compare={"base": ...}` would silently overwrite the primary `model=`
    (GH #76). The refusal exists in two places -- `plots._helpers
    ._resolve_source` for the figure functions and `visualizers
    ._classification._build_compare_source` for the visualizers (a
    DUPLICATION-PENDING pair; a diagnostics module may not import the
    plots-private helper) -- and only the figure-function copy was pinned,
    so the visualizer copy's guard could be deleted with the suite green.
    """
    model = load_fixture("binary_logistic")
    rival = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    y = df["y"]

    for visualizer in _MUST_HAVE_COMPARE:
        with pytest.raises(ValueError, match=r"reserved for the primary"):
            visualizer(model, compare={"base": rival}).fit(X, y)
        # A non-reserved key on the same call shape still works, so the
        # refusal is keyed on "base" rather than on compare= at large.
        assert visualizer(model, compare={"rival": rival}).fit(X, y)._source.model_names == [
            "base",
            "rival",
        ]


def test_compare_rejects_a_non_dict_on_every_visualizer() -> None:
    """The sibling guard in the same duplicated block: `compare=` must be a
    dict, and the refusal names the type it got.
    """
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])

    for visualizer in _MUST_HAVE_COMPARE:
        with pytest.raises(TypeError, match=r"compare= must be dict\[str, model\] or None"):
            visualizer(model, compare=[model]).fit(X, df["y"])


@pytest.mark.parametrize(
    "visualizer, chart_fn",
    [
        (Rank1DVisualizer, ferrum.rank1d_chart),
        (Rank2DVisualizer, ferrum.rank2d_chart),
    ],
)
def test_rank_visualizer_default_algorithm_matches_the_chart(visualizer, chart_fn) -> None:
    """The two defaults skip-map entries claim `algorithm=None` and the
    visualizer's named default are the same default in two spellings. That
    claim is only worth making if it is checked: left unset, both surfaces
    must render byte-identically.
    """
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])

    assert visualizer().fit(X).show().to_svg() == chart_fn(X).to_svg()


def test_defaults_diff_targets_the_public_figure_function() -> None:
    """The defaults diff must resolve the PUBLIC wrapper automatically.

    A default is a promise the public figure function makes, and several
    wrappers re-default on their way into the private builder
    (`residuals_chart`'s `panels=`, `rank1d_chart`'s `algorithm=`). Diffing
    against the private builder compares the visualizer to a number no user
    reads, and hides exactly those divergences -- which is why the
    resolution is automatic rather than opt-in. Asserted directly, because
    the diff itself gets *quieter*, not louder, if this regresses.
    """
    by_visualizer = {case.visualizer: case for case in _PARITY_REGISTRY}

    assert _defaults_target(by_visualizer[ResidualsVisualizer]) is ferrum.residuals_chart
    assert _defaults_target(by_visualizer[Rank1DVisualizer]) is ferrum.rank1d_chart
    # Both name-aliased builders resolve too -- without the alias map these
    # two would silently fall back to their private builders.
    assert _defaults_target(by_visualizer[ConfusionMatrixVisualizer]) is (
        ferrum.confusion_matrix_chart
    )
    assert _defaults_target(by_visualizer[ParallelCoordinatesVisualizer]) is (
        ferrum.parallel_coordinates_chart
    )

    # Every registry row must resolve to a public `ferrum.*` function; a
    # silent fallback to the private builder is the failure mode, so this
    # counts them rather than trusting the four spot checks above.
    # (`CalibrationVisualizer`'s registry target is already the public
    # `calibration_chart` -- see its ParityCase comment -- so it satisfies
    # this directly rather than through the name derivation.)
    public_functions = {
        obj for obj in (getattr(ferrum, name, None) for name in dir(ferrum)) if callable(obj)
    }
    private_fallbacks = [
        case.visualizer.__name__
        for case in _PARITY_REGISTRY
        if _defaults_target(case) not in public_functions
    ]
    assert not private_fallbacks, (
        "these visualizers' defaults are diffed against a private builder, "
        f"so a public re-default would be invisible: {sorted(set(private_fallbacks))}"
    )
