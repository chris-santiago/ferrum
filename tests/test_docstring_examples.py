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
    "ferrum._core",
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
# ``ferrum._core`` is not here: the compiled extension's runtime docstrings
# come from the Rust doc comments, and every example in them must pass.
ILLUSTRATIVE_FAILURES = {
    "ferrum": 11,
    "ferrum._configure_mixin": 1,
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
# The remaining transform and ordering classes whose examples draw a chart.
DRAWING_CLASSES = (
    "Contour",
    "Hex",
    "Linkage",
    "Raster",
    "ReferenceLine",
    "Reorder",
    "Swarm",
    "Unpivot",
)
# Chart-drawing classes outside the transform families: the scale a chart uses.
SCALE_CLASSES = ("ContinuousScheme",)
RUNNING_CLASSES = REGRESSION_FAMILY + STATISTIC_TRANSFORMS + DRAWING_CLASSES + SCALE_CLASSES
# Classes whose examples build a value, not a chart, and render it through a function.
VALUE_CLASSES = ("ChartSpec", "EncodingSpec")
CORE_FUNCTIONS = ("compute_layout", "render_svg", "render_png", "process_batch")

# Per class, the alternative sets of tokens one rendered chart example must carry:
# the class itself, or the mark that builds it. A class whose examples go only
# through the mark says so in its docstring (the lead-in below). Linkage and
# ReferenceLine have no mark that builds them, so their one door is the class.
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
DRAWING_DOORS = {
    "Contour": [("fm.Contour(",), (".mark_contour(",)],
    "Hex": [("fm.Hex(",), (".mark_hex(",)],
    "Linkage": [("fm.Linkage(",)],
    "Raster": [("fm.Raster(",), (".mark_raster(",)],
    "ReferenceLine": [("fm.ReferenceLine(",)],
    "Reorder": [("fm.Reorder(",)],
    "Swarm": [("fm.Swarm(",), (".mark_swarm(",)],
    "Unpivot": [("fm.Unpivot(",)],
}
DOORS = {**STATISTIC_DOORS, **DRAWING_DOORS}
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


# A binding of an extension class or function (a scheme, a spec, a transform, a
# rendered string, a layout) is a value an example builds on the way to a chart;
# every other ``name = fm.<call>`` builds a chart.
_CORE_CALLABLES = "|".join(
    sorted(
        name
        for name, obj in vars(importlib.import_module("ferrum._core")).items()
        if callable(obj) and not name.startswith("_")
    )
)
_CHART_EXAMPLE = re.compile(rf"(\w+) = fm\.(?!(?:{_CORE_CALLABLES})\()")
_CORE_CLASS_NAMES = "|".join(
    sorted(
        name
        for name, obj in vars(importlib.import_module("ferrum._core")).items()
        if isinstance(obj, type)
    )
)
_CORE_VALUE_BINDING = re.compile(rf"(\w+) = fm\.(?:{_CORE_CLASS_NAMES})\(")


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
    """Each chart-drawing ``_core`` class docstring carries examples that all
    run as written (no ``NameError``, no refusal), and every chart one builds is
    rendered by the example that follows it."""
    result = _run_doctests(_core_class_doctests(class_name))
    assert result.attempted > 0, f"{class_name} has no docstring examples"
    assert not result.failures, f"{class_name}: {result.summary()}"
    sources = _example_sources(class_name)
    assert any(_is_chart_example(source) for source in sources)
    assert not _unrendered_charts(sources), f"{class_name}: built but never rendered"


def _absent_doors(source: str) -> list[str]:
    """Return the doors that do not exist an example reaches for: a
    ``mark_*`` method ``Chart`` does not have, a ``fn_=`` keyword on
    ``Summary``, and a ``transform=`` keyword anywhere inside an ``encode(...)``
    call (continuation lines included)."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Attribute)
            and node.attr.startswith("mark_")
            and not hasattr(ferrum.Chart, node.attr)
        ):
            found.append(node.attr)
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
    assert _absent_doors("fm.Chart(df).mark_dendrogram()") == ["mark_dendrogram"]
    assert not _absent_doors("fm.Chart(df).mark_raster()")
    assert _absent_doors('fm.Summary("p", fn_="mean")') == ["Summary(fn_=)"]
    assert _absent_doors('chart.encode(x="a", transform=t)') == ["encode(transform=)"]
    assert _absent_doors('chart.encode(\n    x="a",\n    y="b",\n    transform=t,\n)') == [
        "encode(transform=)"
    ]
    assert not _absent_doors('fm.AggregateOp(field="p", fn_="mean", as_="m")')
    assert not _absent_doors('fm.Chart(df).transform(t).encode(x="a", y="b")')


@pytest.mark.parametrize("class_name", RUNNING_CLASSES + VALUE_CLASSES + CORE_FUNCTIONS)
def test_core_examples_name_only_doors_that_exist(class_name):
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
    for tokens in DOORS[class_name]:
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


@pytest.mark.parametrize("class_name", sorted(DOORS))
def test_transform_examples_use_the_class_or_the_mark_that_builds_it(class_name):
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


def test_every_core_object_with_examples_is_checked():
    """Every ``_core`` class or function that carries examples is in one of the
    run-and-render, value-class or function groups, so a new one cannot fall
    outside the render and door checks."""
    with_examples = {
        test.name.removeprefix("ferrum._core.")
        for test in doctest.DocTestFinder().find(importlib.import_module("ferrum._core"))
        if test.examples
    }
    checked = set(RUNNING_CLASSES) | set(VALUE_CLASSES) | set(CORE_FUNCTIONS)
    assert with_examples == checked, (
        f"unchecked: {sorted(with_examples - checked)}; no examples: {sorted(checked - with_examples)}"
    )


NO_CHART_DOOR_NOTE = "No chart draws this class directly:"


def _unused_values(sources: list[str], docstring: str) -> list[str]:
    """Return each ``_core`` value an example binds (``name = fm.<Class>(...)``)
    that no later example draws: one that uses the name inside a rendered chart
    (a chart example followed by its ``to_svg()``) or inside a ``render_svg`` or
    ``render_png`` call. A docstring that says the class has no chart door
    (``NO_CHART_DOOR_NOTE``) may leave its value unused."""
    if NO_CHART_DOOR_NOTE in docstring:
        return []
    unused = []
    for i, source in enumerate(sources):
        match = _CORE_VALUE_BINDING.match(source)
        if not match:
            continue
        name = re.compile(rf"\b{match.group(1)}\b")
        drawn = any(
            name.search(later)
            and (
                re.search(r"fm\.render_(?:svg|png)\(", later)
                or (_is_chart_example(later) and not _unrendered_charts(sources[j : j + 2]))
            )
            for j, later in enumerate(sources[i + 1 :], start=i + 1)
        )
        if not drawn:
            unused.append(source)
    return unused


@pytest.mark.parametrize("class_name", RUNNING_CLASSES + VALUE_CLASSES)
def test_a_value_an_example_builds_is_drawn(class_name):
    """A class value an example binds is used by a rendered chart or a render
    call, or the docstring says the class has no chart door."""
    assert not _unused_values(_example_sources(class_name), _class_docstring(class_name))


def test_unused_value_is_detected():
    """A bound class value no chart uses fails the check; using it in the
    rendered chart, or saying the class has no chart door, passes."""
    dead = [
        'raster = fm.Raster("x", "y")',
        "chart = fm.Chart(df).mark_raster()",
        "svg = chart.to_svg()",
    ]
    assert _unused_values(dead, "") == [dead[0]]
    used = [
        'raster = fm.Raster("x", "y")',
        "chart = fm.Chart(df).transform(raster).mark_image()",
        "svg = chart.to_svg()",
    ]
    assert not _unused_values(used, "")
    unrendered = used[:2]
    assert _unused_values(unrendered, "") == [used[0]]
    assert not _unused_values(dead, NO_CHART_DOOR_NOTE)
    spec = ['spec = fm.ChartSpec(mark="point")', "fm.render_svg(spec, df).startswith('<svg')"]
    assert not _unused_values(spec, "")
    assert _unused_values(spec[:1], "") == spec[:1]
    # A rendered string or a layout is a function result, not a chart or a class value.
    for binding in ("svg = fm.render_svg(spec, df)", "layout = fm.compute_layout(spec)"):
        assert not _is_chart_example(binding)
        assert not _unused_values([binding], "")
    assert _is_chart_example("chart = fm.lmplot(df, x='x', y='y')")


def test_transform_examples_draw_columns_the_transform_emits():
    """An example that encodes a column its transform does not emit fails when
    run: the renderer refuses it. This covers column names only, not whether the
    mark suits the rows."""
    prelude = ">>> import ferrum as fm\n>>> import polars as pl\n>>> df = pl.DataFrame({'x': [0.1, 0.5, 0.9], 'y': [1.0, 2.0, 3.0]})\n"
    source = 'fm.Chart(df).transform(fm.Hex("x", "y", bin_size=1.0)).mark_point().encode(x="hex_x", y="count").to_svg()'
    test = doctest.DocTestParser().get_doctest(f"{prelude}>>> {source}\n", {}, "bad", "bad", 0)
    assert [kind for _, kind in _run_doctests([test]).failures] == ["ValueError"]


def test_contour_transform_example_draws_segments():
    """The default isoline output is one row per segment with an end point, so
    the transform-door example draws segments from both ends, not start dots."""
    sources = [s for s in _example_sources("Contour") if "fm.Contour(" in s and "encode(" in s]
    assert sources, "no Contour transform-door chart example"
    for source in sources:
        assert ".mark_segment(" in source, source
        for column in ("contour_x", "contour_y", "contour_x2", "contour_y2"):
            assert f'"{column}"' in source, f"{column}: {source}"


def test_hex_transform_example_does_not_draw_vertex_rows_as_points():
    """Hex emits six vertex rows per cell, so drawing its rows with
    ``mark_point`` shows corner dots, not cells."""
    sources = [s for s in _example_sources("Hex") if "fm.Hex(" in s and "encode(" in s]
    assert sources, "no Hex transform-door chart example"
    for source in sources:
        assert ".mark_point(" not in source, source


def test_compute_layout_returns_section_names_keys_the_result_has():
    """Each key the Returns section names is a key of an actual layout."""
    docstring = _class_docstring("compute_layout")
    returns = docstring.split("Returns\n-------")[1].split("Raises\n------")[0]
    named = re.findall(r"``(\w+)``", returns)
    assert named, "the Returns section names no key"
    namespace: dict = {}
    for test in _core_class_doctests("compute_layout"):
        runner = doctest.DocTestRunner(verbose=False)
        runner.run(test, out=lambda _text: None, clear_globs=False)
        namespace = test.globs
    layout = namespace["layout"]
    assert [key for key in named if key not in layout and key != "LayoutResult"] == []


def test_wrong_door_example_fails_the_gate():
    """An example that reaches for a door that does not exist is flagged by the
    door check and also fails when run, so a ``ferrum._core`` docstring that
    carries one cannot pass the must-pass group."""
    prelude = (
        ">>> import ferrum as fm\n>>> import polars as pl\n>>> df = pl.DataFrame({'x': [1.0]})\n"
    )
    wrong_doors = {
        "fm.Chart(df).mark_heatmap().encode(x='x', y='x')": "AttributeError",
        "fm.Chart(df).mark_point().encode(x='x', y='x', transform=fm.Unpivot(value_vars=['x']))": "ValueError",
        "fm.Summary('x', fn_='mean')": "TypeError",
    }
    for source, kind in wrong_doors.items():
        assert _absent_doors(source), source
        docstring = f"{prelude}>>> {source}\n"
        test = doctest.DocTestParser().get_doctest(docstring, {}, "wrong_door", "wrong_door", 0)
        outcome = _run_doctests([test])
        assert [failed_kind for _, failed_kind in outcome.failures] == [kind], source


def test_core_examples_are_must_pass_with_no_allowance():
    """The compiled extension's examples sit in the must-pass group: no ceiling
    lets an undefined name, a wrong door or a refusal hide among them."""
    assert "ferrum._core" in MUST_PASS
    assert "ferrum._core" not in ILLUSTRATIVE_FAILURES


@pytest.mark.parametrize("class_name", VALUE_CLASSES)
def test_value_class_examples_run_and_stay_self_contained(class_name):
    """``ChartSpec`` and ``EncodingSpec`` build a value, not a chart: their
    examples run as written, and any chart one builds is rendered."""
    result = _run_doctests(_core_class_doctests(class_name))
    assert result.attempted > 0, f"{class_name} has no docstring examples"
    assert not result.failures, f"{class_name}: {result.summary()}"
    assert not _unrendered_charts(_example_sources(class_name))


def _renders_and_shows_svg(examples: list[tuple[str, str]]) -> bool:
    """Whether one ``(source, want)`` example calls ``fm.render_svg`` and shows,
    as its own expected output, that the drawing is an SVG."""
    return any(
        "fm.render_svg(" in source
        and source.endswith('.startswith("<svg")')
        and want.strip() == "True"
        for source, want in examples
    )


def _source_and_want(class_name: str) -> list[tuple[str, str]]:
    return [
        (example.source.strip(), example.want)
        for test in _core_class_doctests(class_name)
        for example in test.examples
    ]


def test_chart_spec_example_renders_its_spec():
    """The spec the ``ChartSpec`` example builds is drawn by ``render_svg``,
    and the example itself shows the drawing is an SVG."""
    examples = _source_and_want("ChartSpec")
    assert _renders_and_shows_svg(examples)
    render = next(i for i, (source, _) in enumerate(examples) if "fm.render_svg(" in source)
    unshown = "svg = fm.render_svg(spec, df, viewport=(400, 300))"
    hidden = [*examples[:render], (unshown, ""), *examples[render + 1 :]]
    assert not _renders_and_shows_svg(hidden)
    discarded = [
        *examples[:render],
        ("_ = (fm.render_svg(spec, df, viewport=(400, 300)), None)[1]", ""),
        *examples[render + 1 :],
    ]
    assert not _renders_and_shows_svg(discarded)
    wrong_want = [*examples[:render], (examples[render][0], "False\n"), *examples[render + 1 :]]
    assert not _renders_and_shows_svg(wrong_want)


def test_continuous_scheme_example_draws_with_the_scheme_it_builds():
    """The scheme the example builds is the scale of the chart it renders."""
    sources = _example_sources("ContinuousScheme")
    names = [
        match.group(1)
        for source in sources
        if (match := re.match(r"(\w+) = fm\.ContinuousScheme\(", source))
    ]
    assert names, "ContinuousScheme builds no scheme in its examples"
    rendered = [
        source
        for i, source in enumerate(sources)
        if _is_chart_example(source) and not _unrendered_charts(sources[i : i + 2])
    ]
    assert any(f"scale={name}" in source for name in names for source in rendered)


@pytest.mark.parametrize("function_name", CORE_FUNCTIONS)
def test_core_function_examples_run_and_show_their_result(function_name):
    """Each ``_core`` function's examples run as written, the data they use is
    defined in the example, and the last one shows the result it expects."""
    result = _run_doctests(_core_class_doctests(function_name))
    assert result.attempted > 0, f"{function_name} has no docstring examples"
    assert not result.failures, f"{function_name}: {result.summary()}"
    examples = [
        example for test in _core_class_doctests(function_name) for example in test.examples
    ]
    assert examples[-1].want.strip(), f"{function_name}: the last example shows no output"


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
