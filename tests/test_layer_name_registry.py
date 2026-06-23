"""Drift guard for the layer-name registry (MARK-05, T6.1b cohesion campaign).

Every desugar builds ``_Layer(name=...)`` instances and a separate module-scope
``register_layer_names(kind, frozenset({...}))`` declares the *full* possible
set of sub-layer names for that kind. ``_apply_mark_overrides`` validates user
overrides against ``LAYER_NAME_CATALOG[kind]``, so a layer the desugar *builds*
but the registry does not declare makes a *valid* override raise "Unknown
sub-layer". That is the harmful drift direction.

This test drives each layered composite/diagnostic desugar at **layer-maximizing**
kwargs (every conditional layer turned on, verified by reading each desugar body)
and asserts::

    {layer.name for layer in result.layers} <= LAYER_NAME_CATALOG[kind]

so any new or renamed layer that is not registered fails CI. The reverse drift
(a registered name no build produces) is benign: it is identical to the
conditional-absent case (e.g. boxplot ``outlier`` when ``outliers=False``) and an
override targeting it silently no-ops, which is the documented contract of the
catalog. We therefore guard only ``built <= registered``.
"""

from __future__ import annotations

import pytest

from ferrum._overrides import LAYER_NAME_CATALOG
from ferrum.marks.composite import (
    _BOXEN_K_MAX,
    desugar_boxen,
    desugar_boxplot,
    desugar_errorband,
    desugar_errorbar,
    desugar_ribbon,
)
from ferrum.marks.diagnostic._classification import (
    desugar_calibration,
    desugar_class_prediction_error,
    desugar_confusion,
    desugar_discrimination_threshold,
    desugar_gain,
    desugar_lift,
    desugar_pr,
    desugar_roc,
)
from ferrum.marks.diagnostic._clustering import (
    desugar_decision_boundary,
    desugar_intercluster_distance,
    desugar_pca_scree,
    desugar_pca_scree_with_threshold,
    desugar_silhouette,
)
from ferrum.marks.diagnostic._explanation import (
    desugar_importance,
    desugar_pdp,
    desugar_shap_bar,
    desugar_shap_beeswarm,
    desugar_shap_waterfall,
)
from ferrum.marks.diagnostic._ranking import (
    desugar_parallel_coordinates,
    desugar_rank1d,
    desugar_rank2d,
)
from ferrum.marks.diagnostic._regression import (
    desugar_prediction_error,
    desugar_residuals,
)
from ferrum.marks.diagnostic._selection import (
    desugar_alpha_selection,
    desugar_cv_scores,
    desugar_learning_curve,
    desugar_validation_curve,
)
from ferrum.marks.heavy_stat import (
    desugar_contour,
    desugar_hex,
    desugar_qq,
    desugar_raster,
    desugar_swarm,
    desugar_violin,
)
from ferrum.marks.statistical import desugar_smooth

# ---------------------------------------------------------------------------
# Layer-maximizing cases: (case_id, kind, desugar_fn, args, maximizing_kwargs).
#
# The ``kind`` is the registry key passed to ``register_layer_names``. The
# ``maximizing_kwargs`` were each verified by reading the desugar body to turn
# ON every conditional layer that kind can build. Where a single call cannot
# build every registered name (e.g. contour polygon vs segment, violin inner
# variants, pdp average vs both), multiple cases cover the variants; each is
# independently asserted ``built <= registered``.
# ---------------------------------------------------------------------------

MAXIMIZING_CASES: list[tuple] = [
    # --- composite ----------------------------------------------------------
    # boxplot: outliers=True (default) adds the `outlier` layer.
    ("boxplot", "boxplot", desugar_boxplot, ("species", "sepal_length"), {"outliers": True}),
    # errorbar/errorband/ribbon: fixed-shape, no conditional layers.
    ("errorbar", "errorbar", desugar_errorbar, ("day", "tip"), {}),
    ("errorband", "errorband", desugar_errorband, ("x", "y"), {}),
    ("ribbon", "ribbon", desugar_ribbon, ("x", "lower"), {"y2_field": "upper"}),
    # boxen: builds depth_1.._BOXEN_K_MAX + median + outlier unconditionally.
    ("boxen", "boxen", desugar_boxen, ("species", "sepal_length"), {}),
    # --- statistical --------------------------------------------------------
    # smooth: ci + show_metrics builds {ribbon, line, metrics} (layered path).
    ("smooth", "smooth", desugar_smooth, ("x", "y"), {"ci": 0.95, "show_metrics": True}),
    # --- heavy_stat ---------------------------------------------------------
    # contour: fill=True (default) builds `polygon`; fill=False builds `segment`.
    ("contour_polygon", "contour", desugar_contour, ("x", "y"), {"fill": True}),
    ("contour_segment", "contour", desugar_contour, ("x", "y"), {"fill": False}),
    # violin inner variants (no single inner builds every name):
    #   box      -> {body, whisker, lower_cap, upper_cap, box, median}
    #   quartile -> {body, q1, median, q3}
    #   point    -> {body, point}
    ("violin_box", "violin", desugar_violin, ("species", "petal_length"), {"inner": "box"}),
    (
        "violin_quartile",
        "violin",
        desugar_violin,
        ("species", "petal_length"),
        {"inner": "quartile"},
    ),
    ("violin_point", "violin", desugar_violin, ("species", "petal_length"), {"inner": "point"}),
    # qq: line=True (default) adds the `reference` layer.
    ("qq", "qq", desugar_qq, ("residuals",), {"line": True}),
    ("raster", "raster", desugar_raster, ("x", "y"), {}),
    ("hex", "hex", desugar_hex, ("x", "y"), {}),
    ("swarm", "swarm", desugar_swarm, ("species", "petal_length"), {}),
    # --- diagnostic: classification -----------------------------------------
    # roc: annotate_auc=True adds `auc_label`; reference_line=True (default) adds `reference`.
    ("roc", "roc", desugar_roc, (None, None), {"annotate_auc": True}),
    # pr: iso_lines=True adds {iso_line, iso_label}; annotate_ap=True adds `ap_label`.
    ("pr", "pr", desugar_pr, (None, None), {"iso_lines": True, "annotate_ap": True}),
    # calibration: reference_line=True (default) adds `reference`. (`point` is
    # registered by the figure-function variant in plots/classification.py.)
    ("calibration", "calibration", desugar_calibration, (None, None), {}),
    ("gain", "gain", desugar_gain, (None, None), {}),
    ("lift", "lift", desugar_lift, (None, None), {}),
    # discrimination_threshold: threshold_line=True adds `threshold`;
    # optimum_label=True (default) adds `optimum_label`.
    (
        "discrimination_threshold",
        "discrimination_threshold",
        desugar_discrimination_threshold,
        (None, None),
        {"threshold_line": True, "optimum_label": True},
    ),
    # confusion: annotate=True (default) adds `label`.
    ("confusion", "confusion", desugar_confusion, (None, None), {}),
    # class_prediction_error: show_counts=True (default) adds `label`.
    (
        "class_prediction_error",
        "class_prediction_error",
        desugar_class_prediction_error,
        (None, None),
        {},
    ),
    # --- diagnostic: regression ---------------------------------------------
    # residuals: reference_line=True (default) adds `reference`; cook_threshold
    # set adds `outlier`.
    ("residuals", "residuals", desugar_residuals, (None, None), {"cook_threshold": 3.0}),
    # prediction_error: reference_line=True (default) adds `identity`; ci /
    # reference_band adds `band`.
    (
        "prediction_error",
        "prediction_error",
        desugar_prediction_error,
        (None, None),
        {"ci": 0.95, "reference_band": True},
    ),
    # --- diagnostic: selection ----------------------------------------------
    # learning/validation curve: ci_style="band" (default) adds `band`.
    ("learning_curve", "learning_curve", desugar_learning_curve, (None, None), {}),
    ("validation_curve", "validation_curve", desugar_validation_curve, (None, None), {}),
    # cv_scores: bar -> {bar, mean}; strip -> {point}; box delegates to
    # desugar_boxplot and borrows its six layer names. The registry declares the
    # union of all three kinds, so each variant asserts built <= registered.
    ("cv_scores_bar", "cv_scores", desugar_cv_scores, (None, None), {"kind": "bar"}),
    ("cv_scores_strip", "cv_scores", desugar_cv_scores, (None, None), {"kind": "strip"}),
    ("cv_scores_box", "cv_scores", desugar_cv_scores, (None, None), {"kind": "box"}),
    # alpha_selection: highlight_best=True (default) adds `best`.
    ("alpha_selection", "alpha_selection", desugar_alpha_selection, (None, None), {}),
    # --- diagnostic: clustering ---------------------------------------------
    # silhouette: zero_line=True (default) adds `reference`.
    ("silhouette", "silhouette", desugar_silhouette, (None, None), {}),
    # pca_scree: cumulative_line=True (default) adds `cumulative`.
    ("pca_scree", "pca_scree", desugar_pca_scree, (None, None), {}),
    # pca_scree_with_threshold: always appends `threshold` on top of pca_scree.
    (
        "pca_scree_with_threshold",
        "pca_scree_with_threshold",
        desugar_pca_scree_with_threshold,
        (None, None),
        {},
    ),
    # intercluster_distance: label_clusters=True (default) adds `label`.
    (
        "intercluster_distance",
        "intercluster_distance",
        desugar_intercluster_distance,
        (None, None),
        {},
    ),
    ("decision_boundary", "decision_boundary", desugar_decision_boundary, (None, None), {}),
    # --- diagnostic: explanation -------------------------------------------
    # importance: error_bars=True (default) adds `errorbar`.
    ("importance", "importance", desugar_importance, (None, None), {}),
    # shap_beeswarm: zero_line=True (default) adds `reference`.
    ("shap_beeswarm", "shap_beeswarm", desugar_shap_beeswarm, (None, None), {}),
    ("shap_bar", "shap_bar", desugar_shap_bar, (None, None), {}),
    (
        "shap_waterfall",
        "shap_waterfall",
        desugar_shap_waterfall,
        (None, None),
        {"sample_idx": 0},
    ),
    # pdp: kind="both" builds {ice, average}; kind="average" builds {line}.
    ("pdp_both", "pdp", desugar_pdp, (None, None), {"kind": "both"}),
    ("pdp_average", "pdp", desugar_pdp, (None, None), {"kind": "average"}),
    # --- diagnostic: ranking ------------------------------------------------
    ("rank1d", "rank1d", desugar_rank1d, (None, None), {}),
    # rank2d: annot=True (default) adds `label`.
    ("rank2d", "rank2d", desugar_rank2d, (None, None), {}),
    (
        "parallel_coordinates",
        "parallel_coordinates",
        desugar_parallel_coordinates,
        (None, None),
        {},
    ),
]


@pytest.mark.parametrize(
    "kind, fn, args, kwargs",
    [(kind, fn, args, kw) for _, kind, fn, args, kw in MAXIMIZING_CASES],
    ids=[case_id for case_id, *_ in MAXIMIZING_CASES],
)
def test_built_layers_are_registered(kind: str, fn, args: tuple, kwargs: dict) -> None:
    """Every layer a desugar builds must be a registered sub-layer name.

    Regression: MARK-05 (T6.1b). Guards the harmful drift direction
    ``built <= registered`` so a new or renamed layer that is not declared in
    ``register_layer_names`` fails CI rather than silently rejecting valid
    user overrides at runtime.
    """
    registered = LAYER_NAME_CATALOG.get(kind)
    assert registered, f"kind {kind!r} is not registered in LAYER_NAME_CATALOG"

    result = fn(*args, **kwargs)
    built = {layer.name for layer in (result.layers or []) if layer.name is not None}
    unregistered = built - registered
    assert not unregistered, (
        f"{kind}: desugar builds layer(s) {sorted(unregistered)} not registered in "
        f"LAYER_NAME_CATALOG[{kind!r}]={sorted(registered)} — a valid sub-layer "
        f"override would raise 'Unknown sub-layer'. Add them to register_layer_names()."
    )


def test_every_registered_kind_is_non_empty() -> None:
    """Each registered kind must declare at least one sub-layer name."""
    empty = sorted(k for k, names in LAYER_NAME_CATALOG.items() if not names)
    assert not empty, f"kinds registered with an empty name set: {empty}"


def test_boxen_depth_count_derives_from_constant() -> None:
    """Pin the 2a derivation: boxen builds and registers exactly _BOXEN_K_MAX bands.

    The depth-band loop bound and the registered ``depth_*`` name set both come
    from ``_BOXEN_K_MAX``, so this asserts they stay in lockstep.
    """
    result = desugar_boxen("species", "sepal_length")
    built_depths = sorted(
        layer.name
        for layer in (result.layers or [])
        if layer.name and layer.name.startswith("depth_")
    )
    assert len(built_depths) == _BOXEN_K_MAX

    registered_depths = sorted(
        name for name in LAYER_NAME_CATALOG["boxen"] if name.startswith("depth_")
    )
    assert len(registered_depths) == _BOXEN_K_MAX
    assert built_depths == registered_depths
