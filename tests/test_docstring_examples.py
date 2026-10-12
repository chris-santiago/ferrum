"""Docstring examples run as doctests, so a stale example fails the suite.

Every ``ferrum`` module that carries docstring examples is in exactly one of
two groups:

- ``MUST_PASS``: every example runs and matches its shown output. A module
  joins this group once its examples are self-contained.
- ``ILLUSTRATIVE_FAILURES``: modules whose examples include illustrative
  snippets that reference names the docstring never defines (``df``,
  ``model``, ``X_train``, a fitted ``viz``), so those snippets fail as
  doctests with ``NameError``. Each module's failure count is pinned as a
  ceiling, and every failure must be a ``NameError``: a new rotting example
  either raises the count past the ceiling or fails some other way (a
  ``TypeError``, an output mismatch), and either fails here. A module whose
  examples all come to pass belongs in ``MUST_PASS`` instead.

``test_every_module_with_examples_is_classified`` keeps the two groups
complete, so a new module with examples cannot go unchecked. The type stubs
in ``ferrum/_core.pyi`` are never imported, so their docstrings are collected
from the source with ``ast`` and must all pass.

Examples run in a temporary working directory, because some of them save
charts to relative paths, and in a copy of the current ``contextvars``
context, because some of them set process defaults (``ferrum.config.set``,
``ferrum.set_default_theme``) that would otherwise leak into every later
test on the same worker.
"""

from __future__ import annotations

import ast
import contextvars
import doctest
import importlib
import pkgutil
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import ferrum

MUST_PASS = [
    "ferrum._coerce",
    "ferrum._layer",
    "ferrum._render",
    "ferrum._render_prepare",
    "ferrum._shorthand",
    "ferrum._warn",
    "ferrum.annotation.container",
    "ferrum.annotation.coords",
    "ferrum.axis",
    "ferrum.color",
    "ferrum.config",
    "ferrum.diagnostics.visualizers._ranking",
    "ferrum.display",
    "ferrum.exceptions",
    "ferrum.format_presets",
    "ferrum.layer",
    "ferrum.legend",
    "ferrum.marks._chart_methods_statistical",
    "ferrum.marks._informational_kwargs",
    "ferrum.marks._mark_kwargs",
    "ferrum.marks._multiple",
    "ferrum.marks.base",
    "ferrum.marks.composite",
    "ferrum.marks.heavy_stat",
    "ferrum.marks.statistical",
    "ferrum.parameter",
    "ferrum.themes._defaults",
    "ferrum.themes._theme",
    "ferrum.transforms",
]

# Ceiling on failing examples per module. Every failure counted here is an
# illustrative snippet that uses a name its docstring never defines, which
# ``test_illustrative_failures_are_only_undefined_names`` enforces.
# ``ferrum._core`` is the compiled extension: its runtime docstrings come
# from the Rust doc comments, not from ``_core.pyi``.
ILLUSTRATIVE_FAILURES = {
    "ferrum": 11,
    "ferrum._configure_mixin": 1,
    "ferrum._core": 34,
    "ferrum._metric_labels": 8,
    "ferrum.annotations": 4,
    "ferrum.chart": 9,
    "ferrum.composition": 22,
    "ferrum.coord": 4,
    "ferrum.diagnostics.source": 7,
    "ferrum.diagnostics.sources._compared": 4,
    "ferrum.diagnostics.visualizers._base": 3,
    "ferrum.diagnostics.visualizers._classification": 22,
    "ferrum.diagnostics.visualizers._clustering": 19,
    "ferrum.diagnostics.visualizers._explanation": 12,
    "ferrum.diagnostics.visualizers._regression": 8,
    "ferrum.diagnostics.visualizers._selection": 8,
    "ferrum.encoding.appearance": 19,
    "ferrum.encoding.facet": 5,
    "ferrum.encoding.positional": 18,
    "ferrum.encoding.text": 10,
    "ferrum.grid": 1,
    "ferrum.marks._chart_mixins._classification": 16,
    "ferrum.marks._chart_mixins._clustering": 8,
    "ferrum.marks._chart_mixins._explanation": 10,
    "ferrum.marks._chart_mixins._ranking": 4,
    "ferrum.marks._chart_mixins._regression": 4,
    "ferrum.marks._chart_mixins._selection": 8,
    "ferrum.plots.classification": 19,
    "ferrum.plots.clustering": 6,
    "ferrum.plots.distribution": 10,
    "ferrum.plots.explanation": 6,
    "ferrum.plots.matrix": 8,
    "ferrum.plots.model_selection": 4,
    "ferrum.plots.ranking": 6,
    "ferrum.plots.regression": 13,
    "ferrum.position": 4,
    "ferrum.repeat": 2,
    "ferrum.schemes": 1,
    "ferrum.selection": 5,
    "ferrum.title": 1,
}

CORE_STUB = Path(ferrum.__file__).parent / "_core.pyi"


def _ferrum_module_names() -> list[str]:
    """Return every importable module under the ``ferrum`` package."""
    names = [ferrum.__name__]
    names += [info.name for info in pkgutil.walk_packages(ferrum.__path__, "ferrum.")]
    return sorted(names)


def _example_count(module_name: str) -> int:
    """Count the docstring examples doctest would collect from a module."""
    module = importlib.import_module(module_name)
    return sum(len(test.examples) for test in doctest.DocTestFinder().find(module))


@dataclass
class _DoctestOutcome:
    """Examples attempted, and one ``(where, kind)`` pair per failing example.

    ``kind`` is the exception type name for an example that raised, or
    ``"output mismatch"`` for one whose output differed from the shown text.
    """

    attempted: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        return "; ".join(f"{where}: {kind}" for where, kind in self.failures)


class _RecordingRunner(doctest.DocTestRunner):
    """A doctest runner that records each failure's location and kind."""

    def __init__(self) -> None:
        super().__init__(verbose=False)
        self.outcome = _DoctestOutcome()

    def report_failure(self, out, test, example, got):
        self.outcome.failures.append((f"{test.name}:{example.lineno}", "output mismatch"))

    def report_unexpected_exception(self, out, test, example, exc_info):
        where = f"{test.name}:{example.lineno}"
        self.outcome.failures.append((where, exc_info[0].__name__))


def _run_doctests(tests: list[doctest.DocTest]) -> _DoctestOutcome:
    """Run *tests* in one copy of the current ``contextvars`` context."""
    runner = _RecordingRunner()
    context = contextvars.copy_context()
    for test in tests:
        context.run(runner.run, test, out=lambda _text: None)
    runner.outcome.attempted = runner.tries
    return runner.outcome


def _run_module_doctests(module_name: str) -> _DoctestOutcome:
    module = importlib.import_module(module_name)
    return _run_doctests(doctest.DocTestFinder().find(module))


def _core_stub_doctests() -> list[doctest.DocTest]:
    """Collect one doctest per class or function docstring in ``_core.pyi``."""
    tree = ast.parse(CORE_STUB.read_text())
    parser = doctest.DocTestParser()
    tests: list[doctest.DocTest] = []

    def collect(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef)):
                name = prefix + child.name
                docstring = ast.get_docstring(child)
                if docstring:
                    tests.append(
                        parser.get_doctest(docstring, {}, name, str(CORE_STUB), child.lineno)
                    )
                collect(child, name + ".")

    collect(tree, "")
    return [test for test in tests if test.examples]


@pytest.fixture(autouse=True)
def _run_in_tmp_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_every_module_with_examples_is_classified():
    """A module with docstring examples must be pinned in one of the groups,
    so its examples are checked instead of silently skipped."""
    classified = set(MUST_PASS) | set(ILLUSTRATIVE_FAILURES)
    assert not set(MUST_PASS) & set(ILLUSTRATIVE_FAILURES)
    unclassified = [
        name
        for name in _ferrum_module_names()
        if name not in classified and _example_count(name) > 0
    ]
    assert not unclassified, (
        f"modules with docstring examples in neither MUST_PASS nor "
        f"ILLUSTRATIVE_FAILURES: {unclassified}"
    )


@pytest.mark.parametrize("module_name", MUST_PASS)
def test_module_doctests_pass(module_name):
    """Every example in a pinned module runs and matches its shown output."""
    result = _run_module_doctests(module_name)
    assert result.attempted > 0, f"{module_name} has no doctests to run"
    assert not result.failures, (
        f"{module_name}: {len(result.failures)}/{result.attempted} doctests failed: "
        f"{result.summary()}"
    )


@pytest.mark.parametrize("module_name", sorted(ILLUSTRATIVE_FAILURES))
def test_illustrative_failure_count_does_not_grow(module_name):
    """The illustrative failures stay at or below the pinned ceiling, so a
    newly broken example fails here instead of hiding among them."""
    ceiling = ILLUSTRATIVE_FAILURES[module_name]
    result = _run_module_doctests(module_name)
    failed = len(result.failures)
    assert result.attempted > 0, f"{module_name} has no doctests to run"
    assert failed <= ceiling, (
        f"{module_name}: {failed}/{result.attempted} doctests failed, "
        f"past the ceiling of {ceiling}; a docstring example rotted: {result.summary()}"
    )
    assert failed > 0, f"{module_name}: every example passes now; move it to MUST_PASS"


@pytest.mark.parametrize("module_name", sorted(ILLUSTRATIVE_FAILURES))
def test_illustrative_failures_are_only_undefined_names(module_name):
    """Every ratcheted failure is a ``NameError`` from an undefined
    illustrative name, so a real rot (a ``TypeError``, an output mismatch)
    cannot hide under the ceiling behind an illustrative snippet that was
    fixed in the same change."""
    result = _run_module_doctests(module_name)
    rot = [(where, kind) for where, kind in result.failures if kind != "NameError"]
    assert not rot, f"{module_name}: examples fail with more than an undefined name: {rot}"


def test_doctests_do_not_leak_process_defaults():
    """Examples that set process defaults (``ferrum.config.set``,
    ``ferrum.set_default_theme``) run in a copied context, so the defaults
    every later test on this worker renders with are unchanged.

    The check runs in its own copied context with sentinel defaults the
    examples never set, so a leak shows even when an earlier test on this
    worker already ran the same examples, and the sentinels do not leak.
    """

    def check() -> None:
        ferrum.config.set(width=123)
        ferrum.set_default_theme(ferrum.themes.paper_ink)
        for module_name in ("ferrum.config", "ferrum.themes._defaults"):
            assert _run_module_doctests(module_name).attempted > 0
        assert ferrum.config.get("width") == 123
        assert ferrum.get_default_theme() is ferrum.themes.paper_ink

    contextvars.copy_context().run(check)


def test_core_stub_doctests_pass():
    """Every example in the ``_core.pyi`` stub docstrings runs and matches."""
    tests = _core_stub_doctests()
    assert tests, "_core.pyi has no docstring examples"
    result = _run_doctests(tests)
    assert not result.failures, (
        f"_core.pyi: {len(result.failures)}/{result.attempted} doctests failed: {result.summary()}"
    )
