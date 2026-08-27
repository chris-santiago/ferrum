"""Regression tests for finding P5 (2026-08-27 batch): inverted dependency
directions across ``plots/_helpers.py``.

Two inverted edges were identified:

1. **marks -> plots**: ``ferrum.marks._chart_mixins._{classification,regression}``
   lazily imported ``_sort_by`` from ``ferrum.plots._helpers`` (a leaf-package
   private module) at 3 call sites. ``_sort_by`` had no consumers inside
   ``plots`` itself, so it was relocated to ``ferrum.marks._desugar_helpers``
   (the established marks-side shared-helper home) and the 3 lazy imports
   became direct module-level imports.
2. **diagnostics -> plots (constant)**: ``ferrum.diagnostics.visualizers._clustering``
   imported ``_ELBOW_METRICS`` from ``ferrum.plots.clustering`` and
   hand-duplicated its validation error message instead of calling
   ``ferrum._validate.validate_choice``. ``_ELBOW_METRICS`` was relocated to
   ``ferrum.diagnostics.sources._clustering`` (the vocabulary's single home;
   the computation stays in ``plots.clustering``), and both ``plots.clustering`` and
   ``diagnostics.visualizers._clustering`` now import the constant from
   there and validate through ``validate_choice`` -- one message shape.

Note: ``diagnostics.visualizers.*`` modules legitimately import *chart-building*
functions from ``plots.*`` elsewhere (e.g. ``_silhouette_chart_from_source``,
``_elbow_chart_from_source``, and the sibling regression/classification/
explanation/ranking/selection visualizer modules) -- that is the established,
deliberate "visualizer wraps a plots figure function" architecture, not an
inverted edge, and is out of scope for this finding. These tests assert the
two *specific* edges above are closed, not a blanket ban on any
``diagnostics -> plots`` import.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import ferrum
from ferrum.diagnostics.sources._clustering import _ELBOW_METRICS as _sources_elbow_metrics
from ferrum.marks._desugar_helpers import _sort_by
from ferrum.plots.clustering import _ELBOW_METRICS as _clustering_elbow_metrics
from ferrum.plots.clustering import _elbow_scores

_SRC_ROOT = Path(ferrum.__file__).resolve().parent
_MARKS_ROOT = _SRC_ROOT / "marks"
_DIAGNOSTICS_ROOT = _SRC_ROOT / "diagnostics"


def _module_dotted_name(path: Path) -> str:
    """Return the absolute dotted module name for a file under ``src/ferrum``
    (e.g. ``ferrum.diagnostics.sources._clustering``); ``__init__.py`` files
    resolve to their containing package's own dotted name.
    """
    rel = path.resolve().relative_to(_SRC_ROOT.parent)
    parts = list(rel.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_relative_module(path: Path, node: ast.ImportFrom) -> str:
    """Resolve an ``ast.ImportFrom`` node to its absolute dotted module name.

    Mirrors ``importlib._bootstrap._resolve_name``: for a relative import
    (``node.level > 0``), ``node.module`` is only the *tail* after the dots
    -- e.g. ``from ...plots._helpers import _sort_by`` written inside
    ``ferrum/marks/_chart_mixins/_classification.py`` has
    ``node.module == 'plots._helpers'``, not ``'ferrum.plots._helpers'``.
    Resolution walks up ``node.level`` packages from the *importing file's*
    own package (its own dotted name for an ``__init__.py``, otherwise its
    parent package) and appends ``node.module``. Absolute imports
    (``node.level == 0``) pass through unchanged.
    """
    if node.level == 0:
        return node.module or ""
    dotted = _module_dotted_name(path)
    package = dotted if path.name == "__init__.py" else dotted.rsplit(".", 1)[0]
    base = package.rsplit(".", node.level - 1)[0]
    return f"{base}.{node.module}" if node.module else base


def _imported_modules(path: Path) -> list[str]:
    """Return every module a file imports, as absolute dotted names.

    Covers both ``import`` statements (already absolute) and
    ``from ... import`` statements, including relative forms -- which are
    resolved to their absolute name via :func:`_resolve_relative_module`
    rather than matched on the raw (relative-tail-only) ``node.module``.
    """
    tree = ast.parse(path.read_text())
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolve_relative_module(path, node)
            if resolved:
                modules.append(resolved)
    return modules


def _importfrom_nodes(path: Path) -> list[tuple[ast.ImportFrom, str]]:
    """Return every ``ImportFrom`` node in a file paired with its absolute
    dotted module name (relative imports resolved, see
    :func:`_resolve_relative_module`).
    """
    tree = ast.parse(path.read_text())
    return [
        (node, _resolve_relative_module(path, node))
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    ]


def test_marks_never_imports_from_plots():
    """Grep-proof: no module under ``src/ferrum/marks/`` imports ``ferrum.plots``.

    Structural proof that the ``_sort_by`` inverted edge (marks -> plots) is
    fully closed -- ``_sort_by`` was the only cross-package import and it now
    lives in ``marks._desugar_helpers``.
    """
    offenders = []
    for path in _MARKS_ROOT.rglob("*.py"):
        for module in _imported_modules(path):
            if module == "ferrum.plots" or module.startswith("ferrum.plots."):
                offenders.append(f"{path}: imports {module!r}")
    assert not offenders, f"found plots import(s) under marks/: {offenders}"


def test_sort_by_relocated_to_desugar_helpers():
    """``_sort_by`` is defined once, in ``marks._desugar_helpers``, not in
    ``plots._helpers``.
    """
    import ferrum.plots._helpers as helpers_mod

    assert not hasattr(helpers_mod, "_sort_by"), (
        "_sort_by must not survive in plots._helpers after the P5 relocation"
    )
    assert callable(_sort_by)


def test_diagnostics_visualizers_do_not_import_elbow_metrics_from_plots():
    """Grep-proof: no module under ``src/ferrum/diagnostics/`` imports
    ``_ELBOW_METRICS`` from any ``ferrum.plots`` module.

    Other ``diagnostics -> plots`` imports (chart-building functions such as
    ``_elbow_chart_from_source``, ``_silhouette_chart_from_source``, and the
    regression/classification/explanation/ranking/selection visualizer
    imports) are the deliberate visualizer-wraps-figure-function
    architecture and are untouched by this check.
    """
    offenders = []
    for path in _DIAGNOSTICS_ROOT.rglob("*.py"):
        for node, module in _importfrom_nodes(path):
            if module == "ferrum.plots" or module.startswith("ferrum.plots."):
                names = [alias.name for alias in node.names]
                if "_ELBOW_METRICS" in names:
                    offenders.append(f"{path}:{node.lineno} imports _ELBOW_METRICS from {module!r}")
    assert not offenders, f"found inverted _ELBOW_METRICS import(s): {offenders}"


def test_diagnostics_never_imports_plots_private_modules():
    """Grep-proof: no module under ``src/ferrum/diagnostics/`` imports a
    plots-*private* module (e.g. ``ferrum.plots._helpers``).

    This is the diagnostics-side counterpart of the marks-side blanket ban
    -- ``plots._helpers`` (and any other ``ferrum.plots._*`` module) is
    plots-internal, so nothing outside ``plots`` may import it, regardless
    of which name is pulled from it. Public plots submodules (e.g.
    ``ferrum.plots.clustering``) are unaffected -- those are the deliberate
    visualizer-wraps-figure-function imports covered by the test above.
    """
    offenders = []
    for path in _DIAGNOSTICS_ROOT.rglob("*.py"):
        # _imported_modules covers both `from ... import` (relative forms
        # resolved) and plain `import ferrum.plots._helpers`, so neither
        # spelling can evade the ban.
        for module in _imported_modules(path):
            if not module.startswith("ferrum.plots."):
                continue
            submodule = module[len("ferrum.plots.") :].split(".")[0]
            if submodule.startswith("_"):
                offenders.append(f"{path} imports private module {module!r}")
    assert not offenders, f"found plots-private-module import(s) under diagnostics/: {offenders}"


def test_elbow_metrics_has_one_definition():
    """Grep-proof: ``_ELBOW_METRICS`` is assigned exactly once across the
    package, in ``diagnostics.sources._clustering`` (the vocabulary's single
    home; the elbow computation itself lives in ``plots.clustering``, which
    imports the constant from here).
    """
    definitions = []
    for path in _SRC_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "_ELBOW_METRICS" for t in node.targets
            ):
                definitions.append(str(path))
            elif isinstance(node, ast.AnnAssign) and (
                isinstance(node.target, ast.Name) and node.target.id == "_ELBOW_METRICS"
            ):
                definitions.append(str(path))
    assert definitions == [str(_DIAGNOSTICS_ROOT / "sources" / "_clustering.py")], (
        f"expected exactly one _ELBOW_METRICS definition, found: {definitions}"
    )


def test_elbow_metrics_constant_shared_by_reference():
    """``plots.clustering`` and ``diagnostics.sources._clustering`` reference
    the same tuple object -- proof the constant was moved, not copied.
    """
    assert _clustering_elbow_metrics is _sources_elbow_metrics


def test_elbow_metric_invalid_error_message_single_authority():
    """Behavioral: the ``ElbowVisualizer`` construction-time error and the
    ``elbow_chart``/``_elbow_scores`` computation-time error for the same
    invalid ``metric`` are byte-identical -- one validation authority via
    ``ferrum._validate.validate_choice``, not two hand-rolled messages.
    """
    with pytest.raises(ValueError) as visualizer_exc:
        ferrum.ElbowVisualizer(object, ks=range(2, 5), metric="bogus")

    with pytest.raises(ValueError) as plots_exc:
        _elbow_scores(object, [[1.0, 2.0], [3.0, 4.0]], ks=[2, 3], metric="bogus")

    assert str(visualizer_exc.value) == str(plots_exc.value)
    assert str(visualizer_exc.value) == (
        "ElbowVisualizer: metric must be one of "
        "['calinski_harabasz', 'distortion', 'silhouette']; got 'bogus'"
    )
