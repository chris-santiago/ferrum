# Docstring sweep — design

**Date:** 2026-05-11
**Status:** Approved (pending implementation plan)
**Author:** Chris Santiago + Claude
**Scope:** Add complete NumPy-style docstrings to every public symbol in `ferrum.__all__` (~100 entries plus their methods), and stand up the lint/build mechanics that keep them honest going forward.

---

## 1. Motivation

`ferrum` is about to grow a docs site (zensical, added as a dev dep in `bdb311b`). Before any site scaffolding, the public API needs complete inline docstrings so that:

- `help(ferrum.<X>)` returns useful content for every public symbol.
- IDE tooltips (Pyright/Pylance) surface parameter docs and return types.
- The eventual docs site can auto-extract reference pages via `mkdocstrings` without hand-authored stubs.
- A new contributor adding a public symbol cannot ship without a docstring (lint guardrail).

Today (snapshot 2026-05-11): `Chart` has decent prose coverage but in free-form style; `composition.py`, `themes/`, `position.py`, `coord.py` are partial; **~42 Rust-backed classes and free functions have zero runtime docstrings**; 26 encoding channels are bare. The `__all__` tuple also re-exports three namespace shortcuts (`themes`, `encoding`, `figure`) which require module-level docstrings only, not symbol docstrings of their own.

## 2. Decisions

| # | Decision | Notes |
|---|---|---|
| 1 | NumPy docstring style throughout | Standard in scientific-Python; renders cleanly in `mkdocstrings` with `convention = "numpy"`. |
| 2 | Rust-backed docstrings live in Rust as PyO3 `///` doc-comments | Single source of truth; reaches `help()`, IDE, and `mkdocstrings` simultaneously. |
| 3 | `///` lives on the `#[pyclass]` item, **not** on `#[new]` | NumPy convention: class docstring owns the `Parameters` section for the constructor. |
| 4 | `#[pyo3(signature = (...))]` mandatory on every `#[new]` and `#[pymethods]` block | Without it, `help()` and `mkdocstrings` render `(*args, **kwargs)` instead of named params. |
| 5 | Examples are illustrative `>>>` blocks on user-facing symbols only; not run by doctest | Existing `tests/` already exercises the real pipeline; doctest would force contrived assertions on a `Chart` repr. |
| 6 | Encoding channels use contextual examples inside `Chart.encode(...)` | `>>> fm.Chart(df).encode(x=fm.X("hp", type_="Q"))` — applied uniformly across all 31 channels. |
| 7 | Rewrite all existing prose docstrings to NumPy format during the sweep | Mixed styles render badly in `mkdocstrings`; one consistent format is worth the extra ~25% effort. |
| 8 | Lint via ruff D-rules, NumPy convention, scoped to `src/ferrum/` excluding `_*` modules | Ruff D-rules catch presence + basic format, not section ordering — review still owns content quality. |
| 9 | Module-grouped sweep (~11 commits) with one `maturin develop` per Rust file batch | Approach A from brainstorm. Each commit lands independently with green tests + lint. |
| 10 | Docs site scaffolding is **out of scope** for this sweep | Follows a separate spec once docstrings are complete. |
| 11 | Lint + coverage-test enforcement ratchets file-by-file | Commit 1 enables the rules with all of `src/ferrum/` in `per-file-ignores`; each later commit removes its scope. Keeps every commit green while making the guardrail visible from day one. |

## 3. Public surface taxonomy

The ~100 entries in `ferrum.__all__` plus their methods, split by docstring home:

### 3.1 Pure-Python user-facing (~34 symbols + methods)

Home: `src/ferrum/*.py`. Standard `"""..."""` docstrings.

- `Chart` (`chart.py`) — ~30 methods including `point`, `line`, `bar`, `area`, `rect`, `rule`, `text`, `tick`, `polygon`, `image`, `ribbon`, `density`, `histogram`, `smooth`, `boxplot`, `boxen`, `errorbar`, `errorband`, `ribbon_mark`, `contour`, `violin`, `qq`, `swarm`, `heatmap`, `clustermap`, `encode`, `transform`, `facet`, `theme`, `width`, `height`, `layer`, `to_json`, `render_svg`, etc.
- `Layer` (`layer.py`)
- `HConcatChart`, `VConcatChart`, `JointChart`, `RepeatChart`, `ClusterMapChart` (`composition.py`)
- `Repeat` (`repeat.py`)
- `CoordFlip`, `CoordCartesian`, `CoordPolar`, `CoordGeo`, `CoordFixed` (`coord.py`)
- `Theme`, `set_default_theme`, `get_default_theme`, `theme_context` (`themes/__init__.py`, `themes/_defaults.py`, `themes/builtins.py`)
- `Identity`, `Dodge`, `Jitter`, `Stack` (`position.py`)
- `annotate_hline`, `annotate_vline`, `annotate_rect`, `annotate_text` (`annotations.py`)
- `continuous_palette` (`schemes.py`) — already has a partial NumPy docstring; bring it to spec.
- `displot`, `catplot`, `lmplot`, `residplot`, `pairplot`, `heatmap`, `clustermap`, `jointplot` (`figure/*.py`)

### 3.2 Pure-Python encoding channels (31 classes)

Home: `src/ferrum/encoding/*.py`. Contextual example shape (see §4.4). All 31 channels listed:

`X`, `Y`, `X2`, `Y2`, `XError`, `YError`, `XError2`, `YError2`, `Theta`, `Radius`, `Color`, `Fill`, `Stroke`, `Opacity`, `FillOpacity`, `StrokeOpacity`, `StrokeWidth`, `StrokeDash`, `Size`, `Shape`, `Angle`, `Text`, `Detail`, `Tooltip`, `TooltipField`, `Href`, `Description`, `Key`, `Facet`, `FacetRow`, `FacetCol`.

### 3.3 Rust-backed (~42 symbols + methods)

Home: `crates/ferrum-core/src/...` via PyO3 `///` doc-comments. Resolved paths from the current tree:

- **Spec types** (`crates/ferrum-core/src/spec/chart.rs`, `.../spec/encoding.rs`): `ChartSpec`, `EncodingSpec`.
- **Transforms** (`crates/ferrum-core/src/transform/*.rs`, singular `transform/`): `Aggregate`, `AggregateOp`, `Bin`, `Bin2D`, `BoxStats`, `Contour`, `ErrorExtent`, `Hex`, `Kde`, `Kde2D`, `Glm`, `LetterValue`, `Linkage`, `Logistic`, `Outliers`, `QQ`, `Raster`, `Reorder`, `Robust`, `Smooth`, `Summary`, `Swarm`, `Unpivot`, `Violin`.
- **Scales** (`crates/ferrum-core/src/scale/*.rs`, singular `scale/`): `LinearScale`, `LogScale`, `TimeScale`, `SymlogScale`, `OrdinalScale`, `QuantileScale`, `ThresholdScale`.
- **Schemes** (`crates/ferrum-core/src/render/color/*.rs`): `ContinuousScheme`, `Gradient`. (`Gradient` is exposed at `ferrum.Gradient` via `schemes.py:47` doing `Gradient = _Gradient`, so the runtime object is the Rust-backed class — the `///` doc-comment is the source of truth.)
- **Free functions**: `process_batch` (`transport.rs`), `compute_layout` (`layout/binding.rs`), `render_svg`/`render_png` (`render/binding.rs`, `render/svg.rs`, `render/png.rs`), `compose_svg_horizontal`/`compose_svg_vertical`/`compose_svg_grid` (`render/compositor.rs`, `render/grid_compose.rs`).

## 4. Docstring template

### 4.1 Section structure (NumPy)

Sections are optional individually but appear in this fixed order when present:

```
Summary line — one sentence.

Extended description, free prose, optional.

Parameters
----------
name : type, default value
    Description.
choice_param : {"a", "b"}, default "a"
    Brace literals for enum-like params.

Returns
-------
Chart
    Description of the return value.

Raises
------
ValueError
    When and why (only document intentional exceptions).

Notes
-----
Algorithmic / mathematical / architectural commentary.

See Also
--------
Chart.smooth : Short reason this is related.
ferrum.Bin : Full dotted name for cross-refs.

Examples
--------
>>> import ferrum as fm
>>> chart = fm.Chart(df).encode(x="hp", y="mpg").point()
```

### 4.2 Placement rules

- **Class docstring** owns the `Parameters` section for constructor arguments.
- **Method docstring** documents its own params, never `self`.
- **Module docstring** is a one-line `"""..."""` at the top of every `src/ferrum/*.py`.

### 4.3 Examples policy

- User-facing symbols (§3.1, §3.2, plus figure-level functions): at least one `Examples` block.
- Internal/Rust transforms/scales/spec types (§3.3): prose only — no `>>>`.
- All examples are **illustrative**, never run by `pytest --doctest-modules`. They live in docstrings; they are not assertion-bound.

### 4.4 Encoding channel example shape

Every channel docstring uses a contextual example showing the channel inside `Chart.encode(...)`:

```python
class X:
    """Positional X channel — maps a field to the horizontal axis.

    Parameters
    ----------
    field : str
        Column name in the input DataFrame.
    type_ : {"Q", "N", "O", "T"}, optional
        Data type: quantitative, nominal, ordinal, temporal. Inferred when omitted.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.Chart(df).encode(x=fm.X("hp", type_="Q"))
    """
```

Applied uniformly across all 26 channels.

## 5. PyO3 mechanics for Rust-backed classes

### 5.1 Class doc on `#[pyclass]`

```rust
/// Equal-width or quantile binning of a numeric field.
///
/// Parameters
/// ----------
/// field : str
///     Column to bin.
/// bins : int, default 10
///     Number of bins.
/// method : {"equal-width", "quantile"}, default "equal-width"
///     Binning method.
#[pyclass]
pub struct Bin {
    pub field: String,
    pub bins: u32,
    pub method: BinMethod,
}
```

`///` lives on the struct, not on `#[new]`. Matches NumPy class-doc convention; ensures `help(ferrum.Bin)` and `mkdocstrings` both find the same content.

### 5.2 `signature = (...)` is mandatory

```rust
#[pymethods]
impl Bin {
    #[new]
    #[pyo3(signature = (field, bins=10, method="equal-width"))]
    fn new(field: &str, bins: u32, method: &str) -> PyResult<Self> { ... }
}
```

Without it, the rendered signature collapses to `(*args, **kwargs)`. The newer `signature = (...)` form is preferred over the older `text_signature = "..."` string form because it derives from real Rust types.

### 5.3 Per-method `///` for `#[pymethods]`

Methods on Rust classes that need documentation (e.g. `ChartSpec.to_json`, `ChartSpec.from_json`) get their own `///` block above the function, following the same NumPy template.

### 5.4 PyO3 enums

Enums exposed as classes (e.g. `AggregateOp`, `ErrorExtent`) get:
- A `///` block on the enum describing the enum's purpose.
- A one-line `///` on each variant.

### 5.5 Rebuild discipline

Each Rust source file is fully edited before rebuilding once with:

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
```

~5–6 rebuilds total across the Rust sweep, grouped by commit (§7).

## 6. Lint configuration

### 6.1 `pyproject.toml` additions

Commit 1 lands the full ruff config with `select = ["D"]` **enabled** but `per-file-ignores` initially covers **all of `src/ferrum/`**. Each subsequent commit removes the files it has finished documenting from the ignore list, ratcheting enforcement file-by-file. Every commit is therefore green; the lint config is visible and load-bearing from day one.

Commit-1 form:

```toml
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
"src/ferrum/_*.py" = ["D"]            # private modules
"tests/**" = ["D"]                    # tests
"src/ferrum/_core.pyi" = ["D"]        # stub file; docstrings live in Rust

# Ratcheting exemptions — each row is removed by the commit that completes that scope.
# See §7 for the removal schedule.
"src/ferrum/chart.py" = ["D"]                  # removed by commit 2
"src/ferrum/figure/**" = ["D"]                 # removed by commit 3
"src/ferrum/encoding/**" = ["D"]               # removed by commit 4
"src/ferrum/composition.py" = ["D"]            # removed by commit 5
"src/ferrum/layer.py" = ["D"]                  # removed by commit 5
"src/ferrum/repeat.py" = ["D"]                 # removed by commit 5
"src/ferrum/themes/**" = ["D"]                 # removed by commit 6
"src/ferrum/position.py" = ["D"]               # removed by commit 6
"src/ferrum/coord.py" = ["D"]                  # removed by commit 6
"src/ferrum/annotations.py" = ["D"]            # removed by commit 6
"src/ferrum/schemes.py" = ["D"]                # removed by commit 6
"src/ferrum/__init__.py" = ["D"]               # removed by commit 11 (module docstring)
```

The Rust-backed symbols don't show up in ruff (they live in `_core` which has no `.py` source); their guardrail is the runtime coverage test (§8.3 #11) using the same ratcheting allowlist pattern.

### 6.2 Dev dep

Add `ruff>=0.6` to `[dependency-groups].dev` alongside zensical.

### 6.3 Invocation

```bash
uv run --no-sync ruff check src/ tests/
```

Joins the existing `uv run pytest` step in CI (if/when CI is wired — currently local-only).

### 6.4 Scope of enforcement (honest)

- **Catches**: D100 module, D101 class, D102 method, D103 function — missing docstrings on public symbols. D200–D212 — summary-line and blank-line formatting. D400/D401 — period-at-end and imperative-mood basics.
- **Does NOT catch**: section ordering, `Parameters`-to-signature drift, broken cross-references, example correctness. Those remain review-time concerns.

## 7. Execution order

Twelve commits along Approach A — eleven for the sweep, one trailing commit that captures the conventions as a project-local skill so future docstring work follows the same rules without re-deriving them. Each commit passes `uv run pytest` and `uv run --no-sync ruff check src/ tests/` before landing.

Each Python commit also **removes its scope from `[tool.ruff.lint.per-file-ignores]`** so lint enforcement ratchets module by module. Rust commits **expand `_DOC_ALLOWLIST` in `tests/test_docstring_coverage.py`** for the same reason.

| # | Commit subject | Files | Ratchet | Rebuild |
|---|---|---|---|---|
| 1 | `chore: enable ruff D-rules + add docstring coverage test` | `pyproject.toml`, `uv.lock`, `tests/test_docstring_coverage.py` | Initial config; allowlist empty | No |
| 2 | `docs: rewrite Chart docstrings in NumPy format` | `src/ferrum/chart.py` | Remove `chart.py` from ignores | No |
| 3 | `docs: figure-level convenience functions` | `src/ferrum/figure/*.py` | Remove `figure/**` from ignores | No |
| 4 | `docs: encoding channels` | `src/ferrum/encoding/*.py` | Remove `encoding/**` from ignores | No |
| 5 | `docs: composition, layer, repeat` | `src/ferrum/composition.py`, `layer.py`, `repeat.py` | Remove those three rows | No |
| 6 | `docs: themes, position, coord, annotations, schemes` | `src/ferrum/themes/*.py`, `position.py`, `coord.py`, `annotations.py`, `schemes.py` | Remove those five rows | No |
| 7 | `docs: ChartSpec, EncodingSpec (Rust)` | `crates/ferrum-core/src/spec/chart.rs`, `.../spec/encoding.rs` | Add `ChartSpec`, `EncodingSpec` to `_DOC_ALLOWLIST` | Yes |
| 8 | `docs: transforms (Rust)` | `crates/ferrum-core/src/transform/*.rs` (singular `transform/`) | Add 24 transform symbols to allowlist | Yes |
| 9 | `docs: scales and schemes (Rust)` | `crates/ferrum-core/src/scale/*.rs` (singular `scale/`), `crates/ferrum-core/src/render/color/*.rs` | Add 7 scales + `ContinuousScheme`/`Gradient` to allowlist | Yes |
| 10 | `docs: render, layout, compose, transport (Rust)` | `crates/ferrum-core/src/render/binding.rs`, `compositor.rs`, `grid_compose.rs`, `svg.rs`, `png.rs`; `crates/ferrum-core/src/layout/binding.rs`; `crates/ferrum-core/src/transport.rs` | Add 7 free-function symbols to allowlist | Yes |
| 11 | `docs: module-level docstrings + final lint sweep` | `src/ferrum/__init__.py` + any touched module missing a top-of-file `"""..."""` | Remove `__init__.py` from ignores; assert allowlist covers all of `ferrum.__all__` | No |
| 12 | `chore: add ferrum-docstrings skill for follow-on updates` | `.claude/skills/ferrum-docstrings/SKILL.md` | — | No |

**Branch**: `worktree-chore+docs` (current). PR/merge after commit 11.

**Interleaving rule**: each commit must leave the tree in a green state. If a commit touches Rust files, `maturin develop` runs before commit and the resulting `help()` output is spot-checked for one representative class in the group.

## 8. Definition of done

### 8.1 Per-symbol

1. Summary line is present.
2. `Parameters` enumerates every argument (no `self`), names and types match signatures.
3. `Returns` is present when return type is not `None`.
4. User-facing symbols (§3.1, §3.2, figure-level) have at least one `Examples` block.
5. `Raises` documents only intentional, user-handlable exceptions.
6. Cross-references use full dotted names.

### 8.2 Per-module

7. Top-of-file module docstring exists.

### 8.3 Repo-wide

8. `uv run pytest` passes.
9. `uv run --no-sync ruff check src/ tests/` reports zero D-rule violations.
10. `unset CONDA_PREFIX && uv run --no-sync maturin develop` succeeds after every Rust-touching commit.
11. `tests/test_docstring_coverage.py` (new — added in commit 1) asserts that every symbol in an explicit `_DOC_ALLOWLIST` set has a non-empty `__doc__`. The allowlist starts empty in commit 1 and grows commit-by-commit (see §7 "Ratchet" column); the test passes throughout the sweep. Commit 11 contains a final assertion that `set(_DOC_ALLOWLIST) >= set(ferrum.__all__) - {"themes", "encoding", "figure"}` (the three namespace re-exports are exempt). After commit 11 the test effectively guards all of `ferrum.__all__`.
12. Every `#[pyclass]` carries a `///` block; every `#[new]` and `#[pymethods]` carries a `#[pyo3(signature = (...))]` attribute.

### 8.4 The trailing skill commit (commit 12)

After commit 11 lands and the conventions have been validated against every real public symbol, commit 12 distills them into `.claude/skills/ferrum-docstrings/SKILL.md`. The skill:

- **Triggers** on phrases like "add a docstring", "document this method", "new public class", "add a PyO3 class" — concrete activations from the Anthropic skills guidance.
- **Body** captures: the §4 NumPy template, the §5 PyO3 rules (class-not-init, mandatory `#[pyo3(signature = (...))]`, batched-rebuild discipline), the §3.2 contextual example shape for channels, and a one-line link back to this spec for the full taxonomy.
- **Scope** is project-local (lives in `.claude/skills/`, ships with the repo); not a generic docstring skill.
- **Timing rationale** for landing it last: writing the skill before the sweep risks encoding rules we end up softening during real implementation; writing it after means every example in the skill is grounded in code that actually merged.

The skill is the *trigger*; this spec remains the long-form *reference*. They are complementary.

### 8.5 Out of scope (deferred specs)

- mkdocstrings/zensical site configuration.
- API reference page generation.
- Notebook-style examples in `docs/examples/`.
- README rewrite to reference the new docs.

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| PyO3 signature attribute syntax differs across versions | **Verified 2026-05-11**: workspace `Cargo.toml` pins PyO3 to `0.28`, well past the `signature = (...)` cutoff. The newer attribute form is the only one used in this spec. |
| Docstring-vs-signature drift over time | Lint catches *presence*, not drift. The `test_docstring_coverage.py` asserts presence at runtime; future PRs are expected to update both together as part of review. |
| Existing tests assume specific `__doc__` content | Unlikely (Ferrum tests check behavior, not docstrings) but a quick `grep -r "__doc__" tests/` during commit 1 confirms. |
| Rust rebuild times slow the sweep | Acceptable: ~5–6 rebuilds total. Use `maturin develop` (debug build) during the sweep; only `--release` if a benchmark is needed. |
| Ruff D-rule false positives | Iteratively expand `ignore` list in commit 1; lock list before commit 2. |

## 10. References

- NumPy docstring guide: <https://numpydoc.readthedocs.io/en/latest/format.html>
- Ruff pydocstyle rules: <https://docs.astral.sh/ruff/rules/#pydocstyle-d>
- PyO3 docs on classes and signatures: <https://pyo3.rs/v0.22.0/class.html>
- mkdocstrings Python handler: <https://mkdocstrings.github.io/python/>
