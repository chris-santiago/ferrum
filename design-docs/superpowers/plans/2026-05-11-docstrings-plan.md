# Docstring sweep — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add complete NumPy-style docstrings to every public symbol in `ferrum.__all__` and stand up a ratcheting lint + coverage-test guardrail that keeps them honest.

**Architecture:** Twelve commits along the module-grouped sweep from the spec. Pure-Python docstrings as standard `"""..."""`. Rust-backed docstrings as PyO3 `///` doc-comments on `#[pyclass]` items with mandatory `#[pyo3(signature = (...))]`. Ratcheting per-file-ignores in ruff and ratcheting allowlist in a new coverage test keep every commit green.

**Tech Stack:** Python 3.10+, ruff 0.6+ (D-rules, NumPy convention), PyO3 0.28, maturin 1.7+, pytest 8+, uv.

**Reference spec:** `docs/superpowers/specs/2026-05-11-docstrings-design.md` — read §4 (template), §5 (PyO3 mechanics), §6 (lint), §7 (commit table). Treat the spec as the contract; this plan is the execution sequence.

**Branch:** Working in `.claude/worktrees/chore+docs` on `worktree-chore+docs`. PR/merge after Task 11.

---

## Conventions used in every task

**Build/test commands** (memorize these — they appear in every task):

```bash
# Run Python tests
uv run pytest

# Run lint
uv run --no-sync ruff check src/ tests/

# Rebuild Rust extension after editing any Rust source
unset CONDA_PREFIX && uv run --no-sync maturin develop

# Verify Rust extension is importable
uv run --no-sync python -c "import ferrum; print(ferrum.__version__)"
```

**Commit message style:** `<type>: <imperative subject>` where `<type>` is `chore` or `docs`. No Claude co-author trailer (per project CLAUDE.md).

**NumPy docstring template** (apply to every public symbol, see spec §4):

```python
"""One-line summary, period at end.

Extended description if needed.

Parameters
----------
name : type, default value
    Description.

Returns
-------
ReturnType
    Description.

Examples
--------
>>> import ferrum as fm
>>> fm.Chart(df).point()
"""
```

**Per-file-ignore ratchet rule:** Every Python-docstring task removes the file(s) it covers from `[tool.ruff.lint.per-file-ignores]` in `pyproject.toml` AS PART OF THE SAME COMMIT. If lint fails, the docstrings aren't complete yet.

---

## Task 1: Enable ruff D-rules + add docstring coverage test

**Files:**
- Modify: `pyproject.toml` (add `[tool.ruff]` block, add `ruff>=0.6` dev dep)
- Create: `tests/test_docstring_coverage.py`
- Modify: `uv.lock` (regenerated automatically)

### Steps

- [ ] **Step 1.1: Add ruff dev dep + lint config to `pyproject.toml`**

Append the dev-dep change and the entire `[tool.ruff]` config block. Final state of the relevant pyproject.toml sections:

```toml
[dependency-groups]
dev = ["maturin>=1.7,<2.0", "pytest>=8", "zensical>=0.0.40", "ruff>=0.6"]

[tool.ruff]
line-length = 100
target-version = "py310"
src = ["src", "tests"]

[tool.ruff.lint]
select = ["D"]
ignore = [
    "D203",  # one-blank-line-before-class (conflicts with D211)
    "D213",  # multi-line-summary-second-line (conflicts with D212)
]

[tool.ruff.lint.pydocstyle]
convention = "numpy"

[tool.ruff.lint.per-file-ignores]
# Permanent exemptions:
"src/ferrum/_*.py" = ["D"]
"tests/**" = ["D"]
"src/ferrum/_core.pyi" = ["D"]

# Ratcheting exemptions — each row is removed by the commit that completes that scope.
"src/ferrum/chart.py" = ["D"]
"src/ferrum/figure/**" = ["D"]
"src/ferrum/encoding/**" = ["D"]
"src/ferrum/composition.py" = ["D"]
"src/ferrum/layer.py" = ["D"]
"src/ferrum/repeat.py" = ["D"]
"src/ferrum/themes/**" = ["D"]
"src/ferrum/position.py" = ["D"]
"src/ferrum/coord.py" = ["D"]
"src/ferrum/annotations.py" = ["D"]
"src/ferrum/schemes.py" = ["D"]
"src/ferrum/__init__.py" = ["D"]
```

- [ ] **Step 1.2: Create `tests/test_docstring_coverage.py`**

```python
"""Ratcheting docstring coverage test.

Symbols in _DOC_ALLOWLIST must have a non-empty __doc__. The allowlist grows
commit-by-commit during the docstring sweep (see
docs/superpowers/specs/2026-05-11-docstrings-design.md §7). After commit 11
the allowlist covers all of ferrum.__all__ except the three namespace
re-exports (themes, encoding, figure).
"""
from __future__ import annotations

import ferrum

# Symbols required to have a non-empty __doc__. Grows commit-by-commit.
# DO NOT add symbols here unless their docstrings have actually landed.
_DOC_ALLOWLIST: set[str] = set()

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
    """Final assertion: after commit 11 the allowlist guards all of __all__.

    Skipped while the allowlist is incomplete. Once commit 11 lands, this
    test starts running and stays as a permanent guardrail.
    """
    expected = set(ferrum.__all__) - _NAMESPACE_EXEMPT
    if not _DOC_ALLOWLIST >= expected:
        import pytest
        missing = expected - _DOC_ALLOWLIST
        pytest.skip(f"Allowlist incomplete (sweep in progress); missing: {sorted(missing)}")
    assert _DOC_ALLOWLIST >= expected
```

- [ ] **Step 1.3: Check for any test that asserts specific `__doc__` content**

Run:

```bash
grep -rn "__doc__" tests/
```

Expected: zero matches outside `tests/test_docstring_coverage.py`. If any prior test asserts on docstring content, flag it before proceeding (the sweep will change that content).

- [ ] **Step 1.4: Install + verify**

```bash
uv sync
```

Expected: `ruff` installed (+ resolved transitive deps).

- [ ] **Step 1.5: Run lint to confirm baseline is green**

```bash
uv run --no-sync ruff check src/ tests/
```

Expected: `All checks passed!` (every ferrum source file is in `per-file-ignores`).

- [ ] **Step 1.6: Run tests to confirm baseline is green**

```bash
uv run pytest
```

Expected: existing tests pass, plus 2 new tests from `test_docstring_coverage.py` (one passes trivially, one is skipped with the "sweep in progress" message).

- [ ] **Step 1.7: Commit**

```bash
git add pyproject.toml uv.lock tests/test_docstring_coverage.py
git commit -m "chore: enable ruff D-rules + add docstring coverage test

Adds ruff>=0.6 dev dep, enables D-rules with NumPy convention, scopes
enforcement via ratcheting per-file-ignores covering src/ferrum/.
Adds tests/test_docstring_coverage.py with a _DOC_ALLOWLIST that grows
commit-by-commit through the sweep."
```

---

## Task 2: Rewrite `Chart` docstrings in NumPy format

**Files:**
- Modify: `src/ferrum/chart.py` (every public method + class docstring)
- Modify: `pyproject.toml` (remove `"src/ferrum/chart.py" = ["D"]` line)

### Steps

- [ ] **Step 2.1: Read the current state**

```bash
wc -l src/ferrum/chart.py
grep -n '    def \|^class ' src/ferrum/chart.py
```

Expected output: a list of all public methods (those NOT starting with `_`). Note: `Chart` class docstring exists; many method docstrings exist in free-form prose.

- [ ] **Step 2.2: Rewrite the class docstring**

Locate the `class Chart:` definition (around line 49 per current state) and replace its docstring with the NumPy template. Apply spec §4 placement rule (class doc owns constructor `Parameters`):

```python
class Chart:
    """Top-level chart value class.

    Immutable — every method returns a new `Chart`. Pass data once at
    construction; declare encodings, marks, and transforms with chained
    methods; render with ``render_svg`` / ``render_png`` or display in a
    Jupyter cell.

    Parameters
    ----------
    data : polars.DataFrame, pyarrow.Table, pandas.DataFrame, or any object \
            implementing ``__arrow_c_stream__``
        Input data. Polars and pyarrow flow through Arrow C Data Interface
        with zero copies; other DataFrame types use ``narwhals``.
    width : int, default 400
        Chart width in pixels.
    height : int, default 300
        Chart height in pixels.

    Examples
    --------
    >>> import ferrum as fm
    >>> chart = fm.Chart(df).encode(x="hp", y="mpg").point()
    >>> svg = chart.render_svg()
    """
```

- [ ] **Step 2.3: Rewrite every method docstring**

For each public method (every `def` not prefixed with `_`), rewrite the docstring to the NumPy template (spec §4). Example for `Chart.smooth`:

```python
def smooth(
    self,
    method: str = "loess",
    ci: float = 0.95,
    span: float = 0.75,
    seed: int = 0,
) -> "Chart":
    """Add a smoothed regression line layer.

    Layered atop the current encoding; requires both ``x`` and ``y``.

    Parameters
    ----------
    method : {"loess", "lm", "glm", "logistic"}, default "loess"
        Smoothing method. ``"loess"`` uses local regression;
        ``"lm"`` linear; ``"glm"`` generalized linear; ``"logistic"``
        binary classification.
    ci : float, default 0.95
        Confidence interval level for the band (0 disables).
    span : float, default 0.75
        Smoothing span for LOESS. Ignored for other methods.
    seed : int, default 0
        RNG seed for bootstrap CI. See spec §"byte-deterministic randomness".

    Returns
    -------
    Chart
        New chart with the smoothed-regression layer appended.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="hp", y="mpg").smooth(method="lm")
    """
```

Public methods to document (from current `chart.py`):
- `__init__` is documented via the **class docstring** (NumPy convention).
- Mark methods: `point`, `line`, `bar`, `area`, `rect`, `rule`, `text`, `tick`, `polygon`, `image`, `ribbon_mark`.
- Statistical marks: `density`, `histogram`, `smooth`, `boxplot`, `boxen`, `errorbar`, `errorband`, `contour`, `violin`, `qq`, `swarm`.
- Composite marks: `heatmap`, `clustermap`.
- Declarations: `encode`, `transform`, `facet`, `theme`, `width`, `height`, `layer`.
- Output: `to_json`, `from_json`, `render_svg`, `render_png`, `_repr_html_` (if public).

For each: write summary, Parameters, Returns, Examples. Internal/composite-desugar helpers (those prefixed `_`) are exempt.

- [ ] **Step 2.4: Add `Chart` to `_DOC_ALLOWLIST`**

In `tests/test_docstring_coverage.py`, change:

```python
_DOC_ALLOWLIST: set[str] = set()
```

to:

```python
_DOC_ALLOWLIST: set[str] = {
    # Task 2 — top-level Chart class
    "Chart",
}
```

- [ ] **Step 2.5: Remove `chart.py` from ratcheting ignores**

In `pyproject.toml`, delete this line from `[tool.ruff.lint.per-file-ignores]`:

```toml
"src/ferrum/chart.py" = ["D"]
```

- [ ] **Step 2.6: Run lint**

```bash
uv run --no-sync ruff check src/ tests/
```

Expected: `All checks passed!`. If D-violations surface, fix them in `chart.py` before continuing.

- [ ] **Step 2.7: Run tests**

```bash
uv run pytest
```

Expected: green (including `test_allowlist_symbols_have_docstrings` now asserting `Chart` has a non-empty `__doc__`).

- [ ] **Step 2.8: Spot-check `help()` output**

```bash
uv run --no-sync python -c "import ferrum; help(ferrum.Chart)" | head -40
uv run --no-sync python -c "import ferrum; help(ferrum.Chart.smooth)" | head -30
```

Expected: full NumPy-formatted output for class + at least one method.

- [ ] **Step 2.9: Commit**

```bash
git add src/ferrum/chart.py pyproject.toml tests/test_docstring_coverage.py
git commit -m "docs: rewrite Chart docstrings in NumPy format

Class docstring owns constructor Parameters per NumPy convention.
Every public method gets summary, Parameters, Returns, and a
contextual Examples block. Removes chart.py from ruff ratchet
and seeds the docstring coverage allowlist with Chart."
```

---

## Task 3: Figure-level convenience functions

**Files:**
- Modify: `src/ferrum/figure/__init__.py`, `categorical.py`, `distribution.py`, `joint.py`, `matrix.py`, `regression.py`
- Modify: `pyproject.toml` (remove `"src/ferrum/figure/**" = ["D"]` line)

### Steps

- [ ] **Step 3.1: Enumerate functions**

```bash
grep -rn "^def " src/ferrum/figure/ | grep -v "^.*: def _"
```

Expected: `displot` (`distribution.py`), `catplot` (`categorical.py`), `lmplot`, `residplot` (`regression.py`), `pairplot`, `heatmap`, `clustermap` (`matrix.py`), `jointplot` (`joint.py`).

- [ ] **Step 3.2: Rewrite each function's docstring to NumPy format**

Example for `displot`:

```python
def displot(
    data,
    x: str | None = None,
    hue: str | None = None,
    kind: str = "hist",
    **kwargs,
) -> "Chart":
    """Univariate distribution plot.

    Convenience wrapper that dispatches to ``Chart.histogram`` or
    ``Chart.density`` based on ``kind``.

    Parameters
    ----------
    data : DataFrame-like
        Input data.
    x : str, optional
        Column name for the distribution variable.
    hue : str, optional
        Column name to map to color (one distribution per level).
    kind : {"hist", "kde"}, default "hist"
        ``"hist"`` builds a histogram; ``"kde"`` a smoothed density.
    **kwargs
        Forwarded to the underlying mark method.

    Returns
    -------
    Chart
        Configured chart.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.displot(df, x="hp", kind="kde", hue="cyl")
    """
```

Apply the same shape to all eight figure-level functions. Match the actual function signatures (`grep -n "^def " src/ferrum/figure/<file>.py` first).

- [ ] **Step 3.3: Add module-level docstring to each `figure/` file**

Top of each figure module file (`__init__.py`, `categorical.py`, etc.):

```python
"""Figure-level convenience functions for <topic>."""
```

Where `<topic>` is e.g. "categorical plots" (`categorical.py`), "distributions" (`distribution.py`), "joint distributions" (`joint.py`), "matrix plots" (`matrix.py`), "regression plots" (`regression.py`).

- [ ] **Step 3.4: Add to allowlist**

In `tests/test_docstring_coverage.py`, replace `_DOC_ALLOWLIST: set[str] = set()` with:

```python
_DOC_ALLOWLIST: set[str] = {
    # Task 2 — top-level Chart class
    "Chart",
    # Task 3 — figure-level functions
    "displot", "catplot", "lmplot", "residplot",
    "pairplot", "heatmap", "clustermap", "jointplot",
}
```

- [ ] **Step 3.5: Remove `figure/**` from ratcheting ignores**

Delete this line from `[tool.ruff.lint.per-file-ignores]`:

```toml
"src/ferrum/figure/**" = ["D"]
```

- [ ] **Step 3.6: Lint + test**

```bash
uv run --no-sync ruff check src/ tests/
uv run pytest
```

Both green.

- [ ] **Step 3.7: Commit**

```bash
git add src/ferrum/figure/ pyproject.toml tests/test_docstring_coverage.py
git commit -m "docs: figure-level convenience functions

NumPy docstrings for displot, catplot, lmplot, residplot, pairplot,
heatmap, clustermap, jointplot. Adds module-level docstrings to each
figure/ file. Removes figure/** from ruff ratchet and adds the 8
functions to the docstring coverage allowlist."
```

---

## Task 4: Encoding channels (31 classes)

**Files:**
- Modify: `src/ferrum/encoding/__init__.py`, `base.py`, `positional.py`, `appearance.py`, `text.py`, `facet.py`
- Modify: `pyproject.toml` (remove `"src/ferrum/encoding/**" = ["D"]` line)
- Modify: `tests/test_docstring_coverage.py` (extend allowlist)

### Steps

- [ ] **Step 4.1: Enumerate channel classes**

```bash
grep -rn "^class " src/ferrum/encoding/
```

Expected: 31 classes split across `positional.py` (X, Y, X2, Y2, XError, YError, XError2, YError2, Theta, Radius), `appearance.py` (Color, Fill, Stroke, Opacity, FillOpacity, StrokeOpacity, StrokeWidth, StrokeDash, Size, Shape, Angle), `text.py` (Text, Detail, Tooltip, TooltipField, Href, Description, Key), `facet.py` (Facet, FacetRow, FacetCol), and a `base.py` with shared parent classes (those are internal — exempt).

- [ ] **Step 4.2: Apply the contextual example template (spec §4.4) to every channel**

For each of the 31 channels, write a docstring in this shape:

```python
class X:
    """Positional X channel — maps a field to the horizontal axis.

    Parameters
    ----------
    field : str
        Column name in the input DataFrame.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type: quantitative, nominal, ordinal, temporal. Inferred when
        omitted from the column dtype.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x=fm.X("hp", type_="Q"))
    """
```

Vary the summary line per channel (e.g. `Color` → "Color encoding channel — maps a field to mark color.", `Size` → "Size encoding channel — maps a field to mark area or stroke width.", etc.). Same Parameters block; same Examples shape with the channel in `Chart.encode(...)`.

**Special cases:**
- `TooltipField` is a helper used inside `Tooltip(*fields)`. Its example shows `Tooltip(TooltipField("hp", title="Horsepower"))`.
- `Facet`, `FacetRow`, `FacetCol` show usage via `Chart.facet(...)` not `Chart.encode(...)`.

- [ ] **Step 4.3: Add module-level docstrings**

Each `encoding/*.py` file gets a top-of-file `"""..."""` describing what channels it groups (positional, appearance, text, facet). The `encoding/__init__.py` gets a one-line summary of the namespace.

- [ ] **Step 4.4: Extend the allowlist**

```python
_DOC_ALLOWLIST: set[str] = {
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
}
```

- [ ] **Step 4.5: Remove `encoding/**` from ratcheting ignores**

- [ ] **Step 4.6: Lint + test**

```bash
uv run --no-sync ruff check src/ tests/
uv run pytest
```

Both green.

- [ ] **Step 4.7: Spot-check**

```bash
uv run --no-sync python -c "import ferrum; help(ferrum.X)"
uv run --no-sync python -c "import ferrum; help(ferrum.Tooltip)"
```

- [ ] **Step 4.8: Commit**

```bash
git add src/ferrum/encoding/ pyproject.toml tests/test_docstring_coverage.py
git commit -m "docs: encoding channels

NumPy docstrings for all 31 channel classes with contextual examples
showing channel usage inside Chart.encode(...) (or Chart.facet(...)
for the three facet channels). Adds module docstrings to encoding/
files. Removes encoding/** from ruff ratchet and extends the
coverage allowlist."
```

---

## Task 5: Composition, Layer, Repeat

**Files:**
- Modify: `src/ferrum/composition.py` (HConcatChart, VConcatChart, JointChart, RepeatChart, ClusterMapChart)
- Modify: `src/ferrum/layer.py` (Layer)
- Modify: `src/ferrum/repeat.py` (Repeat)
- Modify: `pyproject.toml` (remove three rows from per-file-ignores)
- Modify: `tests/test_docstring_coverage.py`

### Steps

- [ ] **Step 5.1: Enumerate**

```bash
grep -n "^class " src/ferrum/composition.py src/ferrum/layer.py src/ferrum/repeat.py
```

- [ ] **Step 5.2: Rewrite class + method docstrings**

Example for `HConcatChart`:

```python
class HConcatChart:
    """Horizontal concatenation of two or more charts.

    Each sub-chart keeps its own scales, axes, and legend. Use ``|``
    operator on two charts to construct.

    Parameters
    ----------
    charts : list of Chart
        Sub-charts to concatenate left-to-right.
    spacing : int, default 8
        Pixels of horizontal gap between adjacent charts.

    Returns
    -------
    HConcatChart
        New concatenated chart.

    Examples
    --------
    >>> import ferrum as fm
    >>> combined = fm.Chart(df).point() | fm.Chart(df).histogram()
    """
```

Repeat for `VConcatChart` (vertical), `JointChart` (center + optional margins), `RepeatChart` (template repeated over grid), `ClusterMapChart` (clustered heatmap with dendrograms), `Layer` (internal-ish but in `__all__`), `Repeat` (namespace for typed sentinels).

Public methods on these classes (`__or__`, `__and__`, `render_svg`, etc.) get their own docstrings.

- [ ] **Step 5.3: Add module-level docstrings**

```python
# composition.py
"""Multi-chart composition: HConcat, VConcat, Joint, Repeat, ClusterMap."""

# layer.py
"""Internal layer wrapper used by composite marks."""

# repeat.py
"""Typed sentinel namespace for RepeatChart templates."""
```

- [ ] **Step 5.4: Extend allowlist**

```python
# Task 5 — composition / layer / repeat
"HConcatChart", "VConcatChart", "JointChart", "RepeatChart", "ClusterMapChart",
"Layer", "Repeat",
```

- [ ] **Step 5.5: Remove three rows from ratcheting ignores**

```toml
# Remove:
"src/ferrum/composition.py" = ["D"]
"src/ferrum/layer.py" = ["D"]
"src/ferrum/repeat.py" = ["D"]
```

- [ ] **Step 5.6: Lint + test + commit**

```bash
uv run --no-sync ruff check src/ tests/
uv run pytest
git add src/ferrum/composition.py src/ferrum/layer.py src/ferrum/repeat.py pyproject.toml tests/test_docstring_coverage.py
git commit -m "docs: composition, layer, repeat

NumPy docstrings for HConcat/VConcat/Joint/Repeat/ClusterMap charts,
Layer, and the Repeat sentinel namespace. Adds module docstrings.
Removes three rows from ruff ratchet and extends the coverage
allowlist."
```

---

## Task 6: Themes, position, coord, annotations, schemes

**Files:**
- Modify: `src/ferrum/themes/__init__.py`, `_defaults.py`, `builtins.py`
- Modify: `src/ferrum/position.py`
- Modify: `src/ferrum/coord.py`
- Modify: `src/ferrum/annotations.py`
- Modify: `src/ferrum/schemes.py`
- Modify: `pyproject.toml` (remove five rows from per-file-ignores)
- Modify: `tests/test_docstring_coverage.py`

### Steps

- [ ] **Step 6.1: Enumerate**

```bash
grep -n "^def \|^class " src/ferrum/themes/*.py src/ferrum/position.py src/ferrum/coord.py src/ferrum/annotations.py src/ferrum/schemes.py
```

Expected symbols: `Theme`, `set_default_theme`, `get_default_theme`, `theme_context` (themes); `Identity`, `Dodge`, `Jitter`, `Stack` (position); `CoordFlip`, `CoordCartesian`, `CoordPolar`, `CoordGeo`, `CoordFixed` (coord); `annotate_hline`, `annotate_vline`, `annotate_rect`, `annotate_text` (annotations); `continuous_palette` (schemes — `Gradient` and `ContinuousScheme` are Rust-backed, covered in Task 9).

- [ ] **Step 6.2: Rewrite docstrings**

Example for `set_default_theme`:

```python
def set_default_theme(theme: "Theme") -> "AbstractContextManager[None]":
    """Set the process-wide default theme.

    The default is stored in a per-thread ``contextvars.ContextVar`` and is
    the only sanctioned process-scoped theme state in ferrum. Per-chart
    ``Chart.theme(t)`` always wins at render time. Use as a context manager
    to automatically revert.

    Parameters
    ----------
    theme : Theme
        The theme to install as the new default.

    Returns
    -------
    contextlib.AbstractContextManager[None]
        Context manager that reverts the default on exit.

    Examples
    --------
    >>> import ferrum as fm
    >>> with fm.set_default_theme(fm.themes.dark):
    ...     chart = fm.Chart(df).point()
    """
```

Example for `Jitter`:

```python
class Jitter:
    """Random per-row noise on x and/or y.

    Deterministic given the seed (uses ChaCha8 RNG — see CLAUDE.md
    "byte-deterministic randomness").

    Parameters
    ----------
    x : float, default 0.4
        Maximum absolute jitter applied to x (in scaled units).
    y : float, default 0.0
        Maximum absolute jitter applied to y.
    seed : int, default 0
        RNG seed.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x="grp", y="value").point(position=fm.Jitter())
    """
```

Apply NumPy template to every symbol in the enumeration above.

- [ ] **Step 6.3: Module-level docstrings**

```python
# themes/__init__.py
"""Theme namespace — Theme class, default-theme controls, and built-ins."""

# themes/_defaults.py (already partly documented — confirm top docstring)
"""Built-in default theme values."""

# themes/builtins.py
"""Named built-in themes (dark, light, minimal, ...)."""

# position.py — already has a docstring, confirm it's NumPy-clean

# coord.py
"""Coordinate-system declarations (Cartesian, Polar, Geo, ...)."""

# annotations.py
"""Reference-line and shape annotation helpers."""

# schemes.py — already has a docstring, confirm it's NumPy-clean
```

- [ ] **Step 6.4: Extend allowlist**

```python
# Task 6 — themes / position / coord / annotations / schemes
"Theme", "set_default_theme", "get_default_theme", "theme_context",
"Identity", "Dodge", "Jitter", "Stack",
"CoordFlip", "CoordCartesian", "CoordPolar", "CoordGeo", "CoordFixed",
"annotate_hline", "annotate_vline", "annotate_rect", "annotate_text",
"continuous_palette",
```

- [ ] **Step 6.5: Remove five rows from ratcheting ignores**

```toml
# Remove:
"src/ferrum/themes/**" = ["D"]
"src/ferrum/position.py" = ["D"]
"src/ferrum/coord.py" = ["D"]
"src/ferrum/annotations.py" = ["D"]
"src/ferrum/schemes.py" = ["D"]
```

- [ ] **Step 6.6: Lint + test + commit**

```bash
uv run --no-sync ruff check src/ tests/
uv run pytest
git add src/ferrum/themes/ src/ferrum/position.py src/ferrum/coord.py src/ferrum/annotations.py src/ferrum/schemes.py pyproject.toml tests/test_docstring_coverage.py
git commit -m "docs: themes, position, coord, annotations, schemes

NumPy docstrings for Theme, theme controls, position adjustments,
coord declarations, annotation helpers, and continuous_palette.
Removes five rows from ruff ratchet and extends the coverage
allowlist."
```

---

## Task 7: ChartSpec, EncodingSpec (Rust)

**Files:**
- Modify: `crates/ferrum-core/src/spec/chart.rs`
- Modify: `crates/ferrum-core/src/spec/encoding.rs`
- Modify: `tests/test_docstring_coverage.py`

### Steps

- [ ] **Step 7.1: Read current structs and methods**

```bash
grep -n "pub struct\|pub fn\|#\[pyclass\]\|#\[pymethods\]\|#\[new\]\|fn " crates/ferrum-core/src/spec/chart.rs
grep -n "pub struct\|pub fn\|#\[pyclass\]\|#\[pymethods\]\|#\[new\]\|fn " crates/ferrum-core/src/spec/encoding.rs
```

Note all public methods exposed via `#[pymethods]` blocks (`to_json`, `from_json`, `__repr__`, `__eq__`, getters, etc.).

- [ ] **Step 7.2: Add `///` doc-comment on `#[pyclass]` for `ChartSpec`**

Above the `#[pyclass] pub struct ChartSpec` block:

```rust
/// Intermediate Representation for a chart.
///
/// A tree-structured configuration: top-level mark, encoding channels,
/// optional transforms, optional layers, coordinate system, theme styling.
/// Serializes to JSON via ``to_json`` and round-trips via ``from_json``.
///
/// Parameters
/// ----------
/// mark : {"point", "line", "bar", "area", "rule", "text", "tick", \
///         "rect", "polygon", "image", "ribbon"}
///     Mark kind.
/// x, y, color, size, shape, opacity, x2, y2 : EncodingSpec or str, optional
///     Encoding channels. Strings auto-wrap as ``EncodingSpec(field)``.
/// data : str, default ""
///     Dataset name (referenced by Layer.data_source for multi-batch specs).
/// transforms : list of transform objects, optional
///     Stat transforms applied before rendering.
/// facet : dict, optional
///     Faceting configuration.
/// layers : list of Layer, optional
///     When set, ``mark`` + top-level encodings are ignored; renderer
///     iterates the layers.
/// coord : str, optional
///     Coordinate system name (``"cartesian"``, ``"polar"``, ...).
/// mark_style : dict, optional
///     Theme/style overrides.
///
/// Notes
/// -----
/// ``ChartSpec`` is the contract between Python and Rust. Python's
/// ``Chart`` class builds a ``ChartSpec`` lazily and renders via
/// ``ferrum._core.render_svg``.
#[pyclass]
pub struct ChartSpec { ... }
```

Same shape for `EncodingSpec` (simpler — `field` + optional `type_`).

- [ ] **Step 7.3: Add `#[pyo3(signature = (...))]` to every `#[new]` and `#[pymethods]` block**

Verify present on `ChartSpec::new`; add if missing. Same for `EncodingSpec::new`. Method-level `signature = (...)` is needed when a method takes optional args. Example:

```rust
#[pymethods]
impl ChartSpec {
    #[new]
    #[pyo3(signature = (
        *, mark, x=None, y=None, color=None, size=None, shape=None,
        opacity=None, x2=None, y2=None, data="", transforms=None,
        facet=None, layers=None, coord=None, mark_style=None,
    ))]
    fn new(...) -> PyResult<Self> { ... }

    /// Serialize this spec to its canonical JSON form.
    ///
    /// Returns
    /// -------
    /// str
    ///     JSON-encoded spec.
    fn to_json(&self) -> PyResult<String> { ... }

    /// Reconstruct a ``ChartSpec`` from JSON.
    ///
    /// Parameters
    /// ----------
    /// json : str
    ///     JSON string produced by ``to_json``.
    ///
    /// Returns
    /// -------
    /// ChartSpec
    ///     Reconstructed spec; ``s == ChartSpec.from_json(s.to_json())``.
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> { ... }
}
```

- [ ] **Step 7.4: Rebuild**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
```

Expected: clean build. If `signature` syntax errors surface, check PyO3 0.28 syntax against `https://pyo3.rs/v0.28/class.html`.

- [ ] **Step 7.5: Verify `help()` renders correctly**

```bash
uv run --no-sync python -c "import ferrum; help(ferrum.ChartSpec)" | head -40
uv run --no-sync python -c "import ferrum; help(ferrum.EncodingSpec)" | head -20
uv run --no-sync python -c "import ferrum; help(ferrum.ChartSpec.to_json)"
```

Expected: NumPy-formatted docstring on the class; named params (not `*args, **kwargs`) on the signature line.

- [ ] **Step 7.6: Extend allowlist**

```python
# Task 7 — spec types (Rust)
"ChartSpec", "EncodingSpec",
```

- [ ] **Step 7.7: Run smoke verify**

```bash
unset CONDA_PREFIX && uv run --no-sync python -c "from ferrum._core import ChartSpec; s=ChartSpec(mark='point', x='a', y='b'); assert s == ChartSpec.from_json(s.to_json()); print('OK')"
```

Expected: `OK`.

- [ ] **Step 7.8: Lint + test + commit**

```bash
uv run --no-sync ruff check src/ tests/
uv run pytest
git add crates/ferrum-core/src/spec/ tests/test_docstring_coverage.py
git commit -m "docs: ChartSpec, EncodingSpec (Rust)

PyO3 /// doc-comments on the #[pyclass] items per NumPy convention;
#[pyo3(signature = (...))] on every #[new] and #[pymethods] block so
help() renders named parameters instead of (*args, **kwargs).
Extends the coverage allowlist."
```

---

## Task 8: Transforms (Rust)

**Files:**
- Modify: `crates/ferrum-core/src/transform/*.rs` (one file per transform; ~24 files)
- Modify: `tests/test_docstring_coverage.py`

### Steps

- [ ] **Step 8.1: Enumerate transform files**

```bash
ls crates/ferrum-core/src/transform/
```

Expected files (one transform each, roughly): `aggregate.rs`, `bin.rs`, `bin_2d.rs`, `boxstats.rs`, `contour.rs`, `error_extent.rs`, `hex.rs`, `kde.rs`, `kde_2d.rs`, `glm.rs`, `letter_value.rs`, `linkage.rs`, `logistic.rs`, `outliers.rs`, `qq.rs`, `raster.rs`, `reorder.rs`, `robust.rs`, `smooth.rs`, `summary.rs`, `swarm.rs`, `unpivot.rs`, `violin.rs`, plus a `mod.rs`.

`AggregateOp` and `ErrorExtent` are enums; covered alongside their parent transforms.

- [ ] **Step 8.2: For each transform file, add `///` on the `#[pyclass]` and `#[pyo3(signature = ...)]` on `#[new]`**

Example for `bin.rs`:

```rust
/// Equal-width or quantile binning of a numeric field.
///
/// Discretizes a continuous column into intervals; produces a new field
/// ``<field>_bin`` (centers) suitable for histogram-style encodings.
///
/// Parameters
/// ----------
/// field : str
///     Column to bin.
/// bins : int, default 10
///     Number of bins.
/// method : {"equal-width", "quantile"}, default "equal-width"
///     Binning method. Quantile bins have equal sample counts; equal-width
///     bins have equal range.
/// groupby : list of str, optional
///     Group keys. When set, bins are computed within each group.
///
/// Notes
/// -----
/// See spec §"byte-deterministic randomness" — no randomness involved
/// in binning, but downstream stat transforms may consume binned output.
#[pyclass]
pub struct Bin { ... }

#[pymethods]
impl Bin {
    #[new]
    #[pyo3(signature = (field, bins=10, method="equal-width", groupby=None))]
    fn new(field: &str, bins: u32, method: &str, groupby: Option<Vec<String>>) -> PyResult<Self> { ... }
}
```

Apply the same pattern to every transform. Reference signatures live in the actual Rust source — grep each `pub struct` block before writing the docstring to ensure params match exactly.

**Enums** (`AggregateOp`, `ErrorExtent`):

```rust
/// Aggregation operations for the ``Aggregate`` transform.
///
/// Used as the ``op`` argument; mapped to scalar reducers in Rust.
#[pyclass(eq, eq_int)]
pub enum AggregateOp {
    /// Arithmetic mean of values.
    Mean,
    /// Median (50th percentile) of values.
    Median,
    /// Sum of values.
    Sum,
    /// Count of non-null values.
    Count,
    // ... one /// line per variant
}
```

- [ ] **Step 8.3: Rebuild once after editing all transform files**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
```

Expected: clean build.

- [ ] **Step 8.4: Spot-check across a few transforms**

```bash
uv run --no-sync python -c "import ferrum; help(ferrum.Bin)"
uv run --no-sync python -c "import ferrum; help(ferrum.Smooth)"
uv run --no-sync python -c "import ferrum; help(ferrum.AggregateOp)"
```

Expected: NumPy-formatted output with named-param signatures for the classes; one-line summaries per variant for the enum.

- [ ] **Step 8.5: Extend allowlist**

```python
# Task 8 — transforms (Rust)
"Aggregate", "AggregateOp", "Bin", "Bin2D", "BoxStats", "Contour",
"ErrorExtent", "Hex", "Kde", "Kde2D", "Glm", "LetterValue", "Linkage",
"Logistic", "Outliers", "QQ", "Raster", "Reorder", "Robust", "Smooth",
"Summary", "Swarm", "Unpivot", "Violin",
```

- [ ] **Step 8.6: Lint + test + commit**

```bash
uv run --no-sync ruff check src/ tests/
uv run pytest
git add crates/ferrum-core/src/transform/ tests/test_docstring_coverage.py
git commit -m "docs: transforms (Rust)

PyO3 /// doc-comments and signature attributes for all 24 transforms.
Enums (AggregateOp, ErrorExtent) get per-variant one-liners. Extends
the coverage allowlist."
```

---

## Task 9: Scales and schemes (Rust)

**Files:**
- Modify: `crates/ferrum-core/src/scale/linear.rs`, `log.rs`, `time.rs`, `symlog.rs`, `ordinal.rs`, `quantile.rs`, `threshold.rs`
- Modify: `crates/ferrum-core/src/render/color/scheme.rs`, `continuous.rs`
- Modify: `tests/test_docstring_coverage.py`

### Steps

- [ ] **Step 9.1: Enumerate**

```bash
ls crates/ferrum-core/src/scale/
ls crates/ferrum-core/src/render/color/
grep -rn "pub struct ContinuousScheme\|pub struct Gradient" crates/ferrum-core/src/
```

- [ ] **Step 9.2: Apply NumPy `///` docstrings to each scale**

Example for `LinearScale` in `scale/linear.rs`:

```rust
/// Continuous linear scale.
///
/// Maps a numeric domain to a numeric range via affine transformation.
/// Domain endpoints are typically derived from data; range endpoints
/// from the axis pixel extent.
///
/// Parameters
/// ----------
/// domain : tuple of (float, float), optional
///     Input domain. Defaults to data min/max when omitted.
/// range : tuple of (float, float), optional
///     Output range. Defaults to the axis pixel extent.
/// nice : bool, default True
///     Round domain endpoints to "nice" values for tick generation.
/// clamp : bool, default False
///     Clamp out-of-domain inputs to the range endpoints.
///
/// Examples
/// --------
/// Scales are typically constructed implicitly by ``Chart.encode(...)``;
/// pass an instance to override defaults:
///
/// ::
///
///     chart = chart.encode(x="hp", y="mpg")
#[pyclass]
pub struct LinearScale { ... }
```

Repeat for `LogScale`, `TimeScale`, `SymlogScale`, `OrdinalScale`, `QuantileScale`, `ThresholdScale`. Note the `Examples` block uses `::` literal blocks (Rust doc-comment quirk) rather than `>>>` because some scales aren't naturally usable in a one-liner.

- [ ] **Step 9.3: Apply NumPy `///` docstrings to `ContinuousScheme` and `Gradient`**

Example for `ContinuousScheme`:

```rust
/// A continuous color scheme used by quantitative color encodings.
///
/// Stores either a named built-in (``"viridis"``, ``"plasma"``, ...) or
/// a custom gradient. Constructed via ``continuous_palette(name)`` or
/// ``Gradient(stops)``.
///
/// Parameters
/// ----------
/// name : str, optional
///     Name of a built-in colormap.
///
/// See Also
/// --------
/// ferrum.continuous_palette : Look up a built-in by name.
/// ferrum.Gradient : Construct a custom gradient from (t, color) stops.
#[pyclass]
pub struct ContinuousScheme { ... }
```

For `Gradient`:

```rust
/// Construct a ``ContinuousScheme`` from a list of (t, color) stops.
///
/// Parameters
/// ----------
/// stops : list of (float, str)
///     Pairs of ``t`` (in [0, 1]) and CSS color strings. Endpoints
///     ``(0.0, ...)`` and ``(1.0, ...)`` should be present.
///
/// Returns
/// -------
/// ContinuousScheme
///     Scheme that interpolates linearly between adjacent stops in
///     CIE Lab color space.
///
/// Examples
/// --------
/// ::
///
///     scheme = Gradient([(0.0, "#fff"), (1.0, "#000")])
#[pyclass]
pub struct Gradient { ... }
```

Mandatory `#[pyo3(signature = (...))]` on every `#[new]` and `#[pymethods]` block in all of these files.

- [ ] **Step 9.4: Rebuild**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
```

- [ ] **Step 9.5: Spot-check**

```bash
uv run --no-sync python -c "import ferrum; help(ferrum.LinearScale)"
uv run --no-sync python -c "import ferrum; help(ferrum.ContinuousScheme)"
uv run --no-sync python -c "import ferrum; help(ferrum.Gradient)"
```

- [ ] **Step 9.6: Extend allowlist**

```python
# Task 9 — scales + schemes (Rust)
"LinearScale", "LogScale", "TimeScale", "SymlogScale", "OrdinalScale",
"QuantileScale", "ThresholdScale",
"ContinuousScheme", "Gradient",
```

- [ ] **Step 9.7: Lint + test + commit**

```bash
uv run --no-sync ruff check src/ tests/
uv run pytest
git add crates/ferrum-core/src/scale/ crates/ferrum-core/src/render/color/ tests/test_docstring_coverage.py
git commit -m "docs: scales and schemes (Rust)

PyO3 /// doc-comments and signature attributes for 7 scales
(Linear/Log/Time/Symlog/Ordinal/Quantile/Threshold) and the 2 color
scheme classes (ContinuousScheme, Gradient). Extends the coverage
allowlist."
```

---

## Task 10: Render, layout, compose, transport (Rust)

**Files:**
- Modify: `crates/ferrum-core/src/render/binding.rs`, `compositor.rs`, `grid_compose.rs`, `svg.rs`, `png.rs`
- Modify: `crates/ferrum-core/src/layout/binding.rs`
- Modify: `crates/ferrum-core/src/transport.rs`
- Modify: `tests/test_docstring_coverage.py`

### Steps

- [ ] **Step 10.1: Locate the seven free functions**

```bash
grep -rn "#\[pyfunction\]\|fn render_svg\|fn render_png\|fn process_batch\|fn compute_layout\|fn compose_svg" crates/ferrum-core/src/
```

Expected: `process_batch`, `compute_layout`, `render_svg`, `render_png`, `compose_svg_horizontal`, `compose_svg_vertical`, `compose_svg_grid`. Exact files vary — confirm via grep, not assumption.

- [ ] **Step 10.2: Add `///` and `#[pyo3(signature = (...))]` to each**

Example for `render_svg`:

```rust
/// Render a ``ChartSpec`` and Arrow batch to an SVG string.
///
/// Parameters
/// ----------
/// spec : ChartSpec
///     Chart specification.
/// data : pyarrow.RecordBatch
///     Input data; columns must satisfy the spec's encoding fields.
/// width : int, default 400
///     Output SVG width in pixels.
/// height : int, default 300
///     Output SVG height in pixels.
/// theme : dict, optional
///     Theme override.
///
/// Returns
/// -------
/// str
///     SVG document as a string.
///
/// Notes
/// -----
/// Byte-deterministic given the same spec, data, and theme inputs.
#[pyfunction]
#[pyo3(signature = (spec, data, width=400, height=300, theme=None))]
pub fn render_svg(...) -> PyResult<String> { ... }
```

Apply the same shape to:
- `render_png` — note the `dpi` parameter; returns bytes.
- `process_batch` — exchange entry point; document the Arrow CDI flow.
- `compute_layout` — returns a layout object; document role in the renderer.
- `compose_svg_horizontal`, `compose_svg_vertical`, `compose_svg_grid` — multi-SVG combinators; document the inputs (list of SVG strings) and the spacing param.

- [ ] **Step 10.3: Rebuild**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
```

- [ ] **Step 10.4: Spot-check**

```bash
uv run --no-sync python -c "import ferrum; help(ferrum.render_svg)"
uv run --no-sync python -c "import ferrum; help(ferrum.process_batch)"
uv run --no-sync python -c "import ferrum; help(ferrum.compose_svg_grid)"
```

- [ ] **Step 10.5: Extend allowlist**

```python
# Task 10 — free functions (Rust)
"process_batch", "compute_layout", "render_svg", "render_png",
"compose_svg_horizontal", "compose_svg_vertical", "compose_svg_grid",
```

- [ ] **Step 10.6: Lint + test + commit**

```bash
uv run --no-sync ruff check src/ tests/
uv run pytest
git add crates/ferrum-core/src/render/ crates/ferrum-core/src/layout/ crates/ferrum-core/src/transport.rs tests/test_docstring_coverage.py
git commit -m "docs: render, layout, compose, transport (Rust)

PyO3 /// doc-comments and signature attributes for 7 free functions:
render_svg, render_png, compute_layout, process_batch, and the three
compose_svg_* combinators. Extends the coverage allowlist."
```

---

## Task 11: Module-level docstrings + final lint sweep

**Files:**
- Modify: `src/ferrum/__init__.py` (module docstring + final ratchet removal)
- Modify: any module from Tasks 2–6 that's missing a top-of-file `"""..."""`
- Modify: `pyproject.toml` (remove `__init__.py` from per-file-ignores)
- Modify: `tests/test_docstring_coverage.py` (final assertion enabled)

### Steps

- [ ] **Step 11.1: Audit module-level docstrings**

```bash
for f in src/ferrum/*.py src/ferrum/encoding/*.py src/ferrum/figure/*.py src/ferrum/themes/*.py src/ferrum/marks/*.py; do
  head -1 "$f" | grep -q '^"""' || echo "MISSING: $f"
done
```

Expected: list any file without a top-of-file `"""..."""`. Fix each by prepending a one-line module docstring.

- [ ] **Step 11.2: Confirm `src/ferrum/__init__.py` has a docstring**

The current first line is `"""Ferrum — a statistical visualization library with a Rust core."""`. That's fine; no change needed unless it's missing.

- [ ] **Step 11.3: Remove `__init__.py` from per-file-ignores**

In `pyproject.toml`, delete:

```toml
"src/ferrum/__init__.py" = ["D"]
```

The final state of `[tool.ruff.lint.per-file-ignores]` should be:

```toml
[tool.ruff.lint.per-file-ignores]
"src/ferrum/_*.py" = ["D"]
"tests/**" = ["D"]
"src/ferrum/_core.pyi" = ["D"]
```

(All ratcheting entries gone.)

- [ ] **Step 11.4: Tighten the coverage test final assertion**

In `tests/test_docstring_coverage.py`, replace the body of `test_allowlist_covers_all_public_api_after_sweep` with the active form (drop the `if not >= expected: pytest.skip(...)` guard):

```python
def test_allowlist_covers_all_public_api_after_sweep() -> None:
    """The allowlist must guard every entry in ferrum.__all__ except namespaces."""
    expected = set(ferrum.__all__) - _NAMESPACE_EXEMPT
    missing = expected - _DOC_ALLOWLIST
    assert not missing, f"Allowlist missing public API entries: {sorted(missing)}"
```

- [ ] **Step 11.5: Run the full pipeline**

```bash
uv run --no-sync ruff check src/ tests/
uv run pytest
```

Both must be green. If lint fails on any module-level docstring (D100), fix the file before re-running.

- [ ] **Step 11.6: Final full-coverage verification**

```bash
uv run --no-sync python -c "
import ferrum
namespaces = {'themes', 'encoding', 'figure'}
missing = []
for name in sorted(set(ferrum.__all__) - namespaces):
    obj = getattr(ferrum, name)
    if not (obj.__doc__ or '').strip():
        missing.append(name)
print('Missing:', missing) if missing else print('ALL DOCSTRINGS PRESENT')
"
```

Expected: `ALL DOCSTRINGS PRESENT`.

- [ ] **Step 11.7: Commit**

```bash
git add src/ferrum pyproject.toml tests/test_docstring_coverage.py
git commit -m "docs: module-level docstrings + final lint sweep

Adds module-level docstrings to any Python module missing one.
Removes the last ratcheting entry (__init__.py) from ruff
per-file-ignores. Enables the active form of the coverage final
assertion so future PRs that add public symbols without docstrings
fail CI."
```

---

## Task 12: Add `ferrum-docstrings` skill for follow-on updates

**Files:**
- Create: `.claude/skills/ferrum-docstrings/SKILL.md`

### Steps

- [ ] **Step 12.1: Verify `.claude/skills/` exists**

```bash
ls .claude/skills/ 2>/dev/null || mkdir -p .claude/skills/
```

- [ ] **Step 12.2: Create the skill file**

Write `.claude/skills/ferrum-docstrings/SKILL.md`:

````markdown
---
name: ferrum-docstrings
description: Use when adding or updating docstrings in ferrum — applies the NumPy convention, PyO3 placement rules, and ferrum-specific example shapes. Trigger phrases include "add docstring", "document this method", "new public class", "new PyO3 class", "document this transform", "document this encoding channel".
---

# Ferrum docstring conventions

Use this when adding a new public symbol to ferrum, or when updating an existing docstring. The full taxonomy and rationale live in `docs/superpowers/specs/2026-05-11-docstrings-design.md`; this skill captures the rules you need at the keyboard.

## The NumPy template

```
Summary line — one sentence, ends with a period.

Optional extended description.

Parameters
----------
name : type, default value
    Description.
choice_param : {"a", "b"}, default "a"
    Brace literals for enum-like params.

Returns
-------
ReturnType
    Description (omit if return is None).

Raises
------
ValueError
    Only for intentional, user-handlable exceptions.

Examples
--------
>>> import ferrum as fm
>>> fm.Chart(df).point()
```

Section order is fixed. `self` is never in `Parameters`. Class docstring (not `__init__`) owns the constructor `Parameters`.

## Pure-Python symbols

Standard `"""..."""` docstrings in `src/ferrum/*.py`.

- **User-facing symbols** (Chart methods, figure-level functions, encoding channels, position, coord, themes, annotations, schemes): at least one `Examples` block.
- **Internal helpers**: prose only.

## Encoding channels — use the contextual example shape

```python
class X:
    """Positional X channel — maps a field to the horizontal axis.

    Parameters
    ----------
    field : str
        Column name in the input DataFrame.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type. Inferred from column dtype when omitted.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x=fm.X("hp", type_="Q"))
    """
```

The example always shows the channel **inside** `Chart.encode(...)` (or `Chart.facet(...)` for facet channels).

## Rust-backed classes — PyO3 mechanics

Two rules, both load-bearing for `help()` and mkdocstrings rendering:

### Rule 1: `///` on the `#[pyclass]` item, NOT on `#[new]`

```rust
/// One-line summary.
///
/// Parameters
/// ----------
/// field : str
///     Description.
#[pyclass]
pub struct MyTransform { ... }
```

### Rule 2: `#[pyo3(signature = (...))]` is mandatory

```rust
#[pymethods]
impl MyTransform {
    #[new]
    #[pyo3(signature = (field, bins=10, method="equal-width"))]
    fn new(field: &str, bins: u32, method: &str) -> PyResult<Self> { ... }
}
```

Without `signature`, `help()` and the docs site collapse the signature to `(*args, **kwargs)`. PyO3 ≥ 0.20 supports this form (ferrum pins 0.28).

### Rule 3: Per-method `///` for documented `#[pymethods]`

Method-level `///` blocks above each function inside `#[pymethods]`. Same NumPy template.

### Rule 4: Rebuild discipline

Batch all `///` edits in a Rust file before rebuilding once:

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
```

## Adding a new public symbol — checklist

When you add a new public class or function to `ferrum.__all__`:

- [ ] Write a NumPy-format docstring (template above).
- [ ] For Rust-backed: `///` on `#[pyclass]` + `#[pyo3(signature = (...))]` on `#[new]`.
- [ ] For user-facing symbols: include an `Examples` block.
- [ ] Add the symbol name to `_DOC_ALLOWLIST` in `tests/test_docstring_coverage.py`.
- [ ] Run `uv run pytest` — the `test_allowlist_covers_all_public_api_after_sweep` test fails if you skip the allowlist.
- [ ] Run `uv run --no-sync ruff check src/ tests/` — D-rules will flag missing or malformed docstrings.

## When docstrings drift from signatures

If a method gains/loses a parameter, the docstring `Parameters` section must be updated in the same commit. Lint catches *presence*, not drift — review owns this.

## Where to read the long-form rationale

`docs/superpowers/specs/2026-05-11-docstrings-design.md` covers:
- Why these decisions and not others (§2 decisions table).
- The full taxonomy of public symbols and where each lives (§3).
- Lint configuration details (§6).
- The ratcheting strategy used during the initial sweep (§7).
````

- [ ] **Step 12.3: Verify the skill is discoverable**

```bash
ls .claude/skills/ferrum-docstrings/SKILL.md
```

Expected: file exists.

- [ ] **Step 12.4: Commit**

```bash
git add .claude/skills/ferrum-docstrings/SKILL.md
git commit -m "chore: add ferrum-docstrings skill for follow-on updates

Captures the NumPy convention, PyO3 placement rules, and ferrum-
specific example shapes as a project-local skill. Triggers on
docstring-related phrasing; cross-links the design spec for the
full taxonomy."
```

---

## Post-sweep verification

After Task 12 commits, run the full verification suite from a clean state:

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
uv run pytest
uv run --no-sync ruff check src/ tests/
uv run --no-sync python -c "
import ferrum
namespaces = {'themes', 'encoding', 'figure'}
missing = []
for name in sorted(set(ferrum.__all__) - namespaces):
    obj = getattr(ferrum, name)
    if not (obj.__doc__ or '').strip():
        missing.append(name)
assert not missing, f'Missing: {missing}'
print('ALL DOCSTRINGS PRESENT')
"
git log --oneline worktree-chore+docs | head -15
```

Expected:
- `maturin develop`: clean build.
- `pytest`: green, including `tests/test_docstring_coverage.py`.
- `ruff check`: `All checks passed!`.
- Python loop: `ALL DOCSTRINGS PRESENT`.
- `git log`: shows 12 commits in the expected order plus the prior `chore: add zensical as dev dep` and `docs(specs)` commits.

When all four pass, the sweep is complete. Open a PR from `worktree-chore+docs` to `main` (do not push until the user confirms).

## Spec coverage notes

- **§2 decisions 1–11**: each is enforced by at least one task (Tasks 1–11 implement it; Task 12 documents it).
- **§3 taxonomy**: Tasks 2–6 (Python), 7–10 (Rust) cover all three buckets.
- **§4 template**: shown in every Python task as a code block; canonical version lives in the skill (Task 12).
- **§5 PyO3 mechanics**: Tasks 7–10 each apply rules 5.1–5.5.
- **§6 lint config**: Task 1 lands the config; Tasks 2–6 + 11 do the ratchet removal.
- **§7 commit table**: maps 1:1 onto Tasks 1–12.
- **§8 DoD**: §8.1 per-symbol is verified by ruff D-rules; §8.2 per-module is verified by D100 + the manual audit in Task 11; §8.3 repo-wide is verified by the post-sweep block above; §8.4 commit 12 is Task 12.
