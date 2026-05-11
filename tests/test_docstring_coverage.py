"""Ratcheting docstring coverage test.

Symbols in _DOC_ALLOWLIST must have a non-empty __doc__. The allowlist grew
commit-by-commit during the docstring sweep (see
docs/superpowers/specs/2026-05-11-docstrings-design.md §7). The sweep is now
complete: the allowlist covers all of ferrum.__all__ except the three
namespace re-exports (themes, encoding, figure).
"""
from __future__ import annotations

import ferrum

# Symbols required to have a non-empty __doc__. Grows commit-by-commit.
# DO NOT add symbols here unless their docstrings have actually landed.
_DOC_ALLOWLIST: set[str] = {
    # Task 2 — top-level Chart class
    "Chart",
    # Task 3 — figure-level functions
    "displot", "catplot", "lmplot", "residplot",
    "pairplot", "heatmap", "clustermap", "jointplot",
    # Task 4 — encoding channels (31)
    "X", "Y", "X2", "Y2", "XError", "YError", "XError2", "YError2",
    "Theta", "Radius",
    "Color", "Fill", "Stroke", "Opacity", "FillOpacity", "StrokeOpacity",
    "StrokeWidth", "StrokeDash", "Size", "Shape", "Angle",
    "Text", "Detail", "Tooltip", "TooltipField", "Href", "Description", "Key",
    "Facet", "FacetRow", "FacetCol",
    # Task 5 — composition / layer / repeat
    "HConcatChart", "VConcatChart", "JointChart", "RepeatChart", "ClusterMapChart",
    "Layer", "Repeat",
    # Task 6 — themes / position / coord / annotations / schemes
    "Theme", "set_default_theme", "get_default_theme", "theme_context",
    "Identity", "Dodge", "Jitter", "Stack",
    "CoordFlip", "CoordCartesian", "CoordPolar", "CoordGeo", "CoordFixed",
    "annotate_hline", "annotate_vline", "annotate_rect", "annotate_text",
    "continuous_palette",
    # Task 7 — spec types (Rust)
    "ChartSpec", "EncodingSpec",
    # Task 8 — transforms (Rust)
    "Aggregate", "AggregateOp", "Bin", "Bin2D", "BoxStats", "Contour",
    "ErrorExtent", "Hex", "Kde", "Kde2D", "Glm", "LetterValue", "Linkage",
    "Logistic", "Outliers", "QQ", "Raster", "Reorder", "Robust", "Smooth",
    "Summary", "Swarm", "Unpivot", "Violin",
    # Task 9 — scales + schemes (Rust)
    "LinearScale", "LogScale", "TimeScale", "SymlogScale", "OrdinalScale",
    "QuantileScale", "ThresholdScale",
    "ContinuousScheme", "Gradient",
    # Task 10 — free functions (Rust)
    "process_batch", "compute_layout", "render_svg", "render_png",
    "compose_svg_horizontal", "compose_svg_vertical", "compose_svg_grid",
    # Phase 10 — model diagnostics layer (added 2026-05-11)
    "ModelSource", "ComparedModelSource",
    "FerrumVisualizer",
    "ResidualsVisualizer", "PredictionErrorVisualizer", "CooksDistanceVisualizer",
    "ROCVisualizer", "PRVisualizer", "CalibrationVisualizer",
    "DiscriminationThresholdVisualizer",
    "ConfusionMatrixVisualizer", "ClassificationReportVisualizer",
    "ClassPredictionErrorVisualizer", "ClassBalanceVisualizer",
    "FeatureImportancesVisualizer", "SHAPVisualizer",
    "LearningCurveVisualizer", "ValidationCurveVisualizer",
    "CVScoresVisualizer", "AlphaSelectionVisualizer",
    "SilhouetteVisualizer", "ElbowVisualizer", "ManifoldVisualizer",
    "InterclusterDistanceVisualizer", "PCAVarianceVisualizer",
    "Rank1DVisualizer", "Rank2DVisualizer", "ParallelCoordinatesVisualizer",
    "residuals_chart",
    "roc_chart", "pr_chart", "calibration_chart",
    "gain_chart", "lift_chart", "discrimination_threshold_chart",
    "confusion_matrix_chart", "class_prediction_error_chart",
    "importance_chart", "shap_chart", "pdp_chart",
    "learning_curve_chart", "validation_curve_chart",
    "cv_scores_chart", "alpha_selection_chart",
    "pca_scree_chart", "cluster_diagnostics",
    "intercluster_distance_chart", "decision_boundary_chart",
    "rank_chart", "parallel_coordinates_chart",
}

# Namespace re-exports — exempt forever (they're submodules, not symbols).
_NAMESPACE_EXEMPT: set[str] = {"themes", "encoding", "figure"}


def test_allowlist_symbols_have_docstrings() -> None:
    """Every symbol in _DOC_ALLOWLIST must have a non-empty __doc__."""
    missing: list[str] = []
    for name in sorted(_DOC_ALLOWLIST):
        obj = getattr(ferrum, name, None)
        assert obj is not None, f"ferrum.{name} not found"
        doc = getattr(obj, "__doc__", None)
        if not doc or not doc.strip():
            missing.append(name)
    assert not missing, f"Missing docstrings on: {missing}"


def test_allowlist_covers_all_public_api_after_sweep() -> None:
    """The allowlist must guard every entry in ferrum.__all__ except namespaces."""
    expected = set(ferrum.__all__) - _NAMESPACE_EXEMPT
    missing = expected - _DOC_ALLOWLIST
    assert not missing, (
        f"Undocumented public API symbols: {sorted(missing)}\n\n"
        "TO FIX: Run the ferrum-docstrings skill for each missing symbol, then add it\n"
        "to _DOC_ALLOWLIST in this file.\n\n"
        "  Skill path: .claude/skills/ferrum-docstrings/SKILL.md\n"
        "  Invoke via: /ferrum-docstrings\n\n"
        "The skill enforces NumPy-style conventions, stub-param honesty, and accurate\n"
        "output column names for Rust transforms. Do not add symbols to _DOC_ALLOWLIST\n"
        "unless their docstrings have actually landed."
    )
