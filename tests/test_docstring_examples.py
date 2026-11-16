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
import re
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import ferrum

MUST_PASS = [
    "ferrum._coerce",
    "ferrum._layer",
    "ferrum._render",
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
    "ferrum._core": 15,
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


REGRESSION_FAMILY = ("Glm", "Logistic", "Robust", "Smooth")
STATISTIC_TRANSFORMS = (
    "AggregateOp",
    "Aggregate",
    "Bin",
    "Bin2D",
    "BoxStats",
    "ErrorExtent",
    "Kde",
    "Kde2D",
    "LetterValue",
    "Outliers",
    "QQ",
    "Summary",
    "Violin",
)
RUNNING_CLASSES = REGRESSION_FAMILY + STATISTIC_TRANSFORMS

# Per class, the alternative sets of tokens one rendered chart example must carry:
# the class itself, or the mark that builds it. A class whose examples go only
# through the mark says so in its docstring (the lead-in below).
STATISTIC_DOORS = {
    "AggregateOp": [("fm.AggregateOp(", "fm.Aggregate(")],
    "Aggregate": [("fm.Aggregate(",)],
    "Bin": [("fm.Bin(",)],
    "Bin2D": [("fm.Bin2D(",)],
    "BoxStats": [("fm.BoxStats(",)],
    "ErrorExtent": [("fm.ErrorExtent(",)],
    "Kde": [("fm.Kde(",)],
    "Kde2D": [("fm.Kde2D(",), (".mark_contour(",)],
    "LetterValue": [("fm.LetterValue(",), (".mark_boxen(",)],
    "Outliers": [("fm.Outliers(",)],
    "QQ": [("fm.QQ(",)],
    "Summary": [("fm.Summary(",)],
    "Violin": [("fm.Violin(",), (".mark_violin(",)],
}
MARK_LEAD_IN = "mark builds the same transform:"


def _core_class_doctests(class_name: str) -> list[doctest.DocTest]:
    module = importlib.import_module("ferrum._core")
    return [
        test
        for test in doctest.DocTestFinder().find(module)
        if test.name == f"ferrum._core.{class_name}" and test.examples
    ]


def _example_sources(class_name: str) -> list[str]:
    return [
        example.source.strip()
        for test in _core_class_doctests(class_name)
        for example in test.examples
    ]


_CHART_EXAMPLE = re.compile(r"(\w+) = fm\.")


def _is_chart_example(source: str) -> bool:
    return _CHART_EXAMPLE.match(source) is not None


def _unrendered_charts(sources: list[str]) -> list[str]:
    """Return each example that builds a chart (``<name> = fm...``) and is not
    directly followed by the example that renders that name, so a chart that
    is built but never drawn (and so never refused at render time) cannot
    pass."""
    unrendered = []
    for source, following in zip(sources, [*sources[1:], ""], strict=True):
        match = _CHART_EXAMPLE.match(source)
        if match and not re.fullmatch(rf"\w+ = {match.group(1)}\.to_svg\(\)", following):
            unrendered.append(source)
    return unrendered


@pytest.mark.parametrize("class_name", RUNNING_CLASSES)
def test_class_examples_run_and_render(class_name):
    """Each regression-family and statistic-transform class docstring carries
    examples that all run as written (no ``NameError``, no refusal), and every chart one builds is
    rendered by the example that follows it."""
    result = _run_doctests(_core_class_doctests(class_name))
    assert result.attempted > 0, f"{class_name} has no docstring examples"
    assert not result.failures, f"{class_name}: {result.summary()}"
    sources = _example_sources(class_name)
    assert any(_is_chart_example(source) for source in sources)
    assert not _unrendered_charts(sources), f"{class_name}: built but never rendered"


def _absent_doors(source: str) -> list[str]:
    """Return the doors that do not exist an example reaches for: a heatmap
    mark, a ``fn_=`` keyword on ``Summary``, and a ``transform=`` keyword
    anywhere inside an ``encode(...)`` call (continuation lines included)."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and node.attr == "mark_heatmap":
            found.append("mark_heatmap")
        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        name = callee.attr if isinstance(callee, ast.Attribute) else getattr(callee, "id", "")
        keywords = {keyword.arg for keyword in node.keywords}
        if name == "Summary" and "fn_" in keywords:
            found.append("Summary(fn_=)")
        if name == "encode" and "transform" in keywords:
            found.append("encode(transform=)")
    return found


def test_absent_door_check_detects_each_door_and_spares_a_real_keyword():
    """The check fires on every nonexistent door, wherever the keyword sits,
    and leaves ``AggregateOp``'s real ``fn_=`` keyword alone."""
    assert _absent_doors("fm.Chart(df).mark_heatmap()") == ["mark_heatmap"]
    assert _absent_doors('fm.Summary("p", fn_="mean")') == ["Summary(fn_=)"]
    assert _absent_doors('chart.encode(x="a", transform=t)') == ["encode(transform=)"]
    assert _absent_doors('chart.encode(\n    x="a",\n    y="b",\n    transform=t,\n)') == [
        "encode(transform=)"
    ]
    assert not _absent_doors('fm.AggregateOp(field="p", fn_="mean", as_="m")')
    assert not _absent_doors('fm.Chart(df).transform(t).encode(x="a", y="b")')


@pytest.mark.parametrize("class_name", STATISTIC_TRANSFORMS)
def test_statistic_transform_examples_name_only_doors_that_exist(class_name):
    for source in _example_sources(class_name):
        assert not _absent_doors(source), f"{class_name}: {source}"


def _door_gaps(class_name: str, sources: list[str], docstring: str) -> list[str]:
    """Return what keeps a class's examples from using the class it documents
    or the mark that builds it: no rendered chart example carrying one of its
    door alternatives, and, when no example names the class itself, a docstring
    that does not say the mark builds the same transform."""
    gaps = []
    rendered = [
        source
        for i, source in enumerate(sources)
        if _is_chart_example(source) and not _unrendered_charts(sources[i : i + 2])
    ]
    for tokens in STATISTIC_DOORS[class_name]:
        if any(all(token in source for token in tokens) for source in rendered):
            break
    else:
        gaps.append("no rendered chart example uses the class or its mark")
    if not any(f"fm.{class_name}(" in source for source in rendered) and (
        MARK_LEAD_IN not in docstring
    ):
        gaps.append(f"examples go through a mark without saying it '{MARK_LEAD_IN}'")
    return gaps


def _class_docstring(class_name: str) -> str:
    return "\n".join(test.docstring for test in _core_class_doctests(class_name))


@pytest.mark.parametrize("class_name", STATISTIC_TRANSFORMS)
def test_statistic_transform_examples_use_the_class_or_the_mark_that_builds_it(class_name):
    sources = _example_sources(class_name)
    assert _door_gaps(class_name, sources, _class_docstring(class_name)) == []


def test_door_check_detects_an_example_that_stops_using_its_class():
    """Violin's mark swapped for the boxplot mark, and Kde's transform example
    replaced by a plain chart, each leave the no-door gap; a docstring without
    the lead-in adds the second gap."""
    no_door = "no rendered chart example uses the class or its mark"
    no_lead_in = f"examples go through a mark without saying it '{MARK_LEAD_IN}'"
    violin = _example_sources("Violin")
    swapped = [source.replace(".mark_violin()", ".mark_boxplot()") for source in violin]
    assert swapped != violin
    assert _door_gaps("Violin", swapped, MARK_LEAD_IN) == [no_door]
    assert _door_gaps("Violin", swapped, "") == [no_door, no_lead_in]
    kde = _example_sources("Kde")
    plain = [
        source.replace('.transform(fm.Kde("val", bandwidth=0.8)).mark_area()', ".mark_point()")
        for source in kde
    ]
    assert plain != kde
    assert _door_gaps("Kde", plain, MARK_LEAD_IN) == [no_door]
    assert _door_gaps("Kde", kde, "") == []


@pytest.mark.parametrize(
    ("class_name", "doors"),
    [
        ("Logistic", ('method="logistic"', "fm.Logistic(")),
        ("Robust", ('method="robust"', "fm.Robust(")),
    ],
)
def test_logistic_and_robust_examples_show_the_doors_that_exist(class_name, doors):
    """Logistic and Robust are reached through ``lmplot(method=...)`` and
    ``.transform(...)``, never ``mark_smooth(method=...)``, which refuses them.
    Each door is its own chart example, rendered by the example after it."""
    sources = _example_sources(class_name)
    assert not any("mark_smooth" in source for source in sources)
    for door in doors:
        built = [
            i for i, source in enumerate(sources) if _is_chart_example(source) and door in source
        ]
        assert len(built) == 1, f"{class_name}: expected one chart example using {door}"
        assert not _unrendered_charts(sources[built[0] : built[0] + 2])


def test_unrendered_chart_is_detected():
    """The render check fails when one example's render line is removed."""
    sources = _example_sources("Logistic")
    assert not _unrendered_charts(sources)
    first_render = sources.index("svg = chart.to_svg()")
    assert _unrendered_charts(sources[:first_render] + sources[first_render + 1 :])
    last_render = len(sources) - 1 - sources[::-1].index("svg = chart.to_svg()")
    assert _unrendered_charts(sources[:last_render])
    renamed = [source.replace("chart", "fig") for source in sources]
    assert not _unrendered_charts(renamed)
    renamed[first_render] = "svg = chart.to_svg()"
    assert _unrendered_charts(renamed)


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
