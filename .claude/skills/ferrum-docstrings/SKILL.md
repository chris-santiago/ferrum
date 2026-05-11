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
>>> fm.Chart(df).mark_point()
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

Always show the channel **inside** `Chart.encode(...)` (or `Chart.facet(...)` for facet channels via `facet=`, `facet_row=`, `facet_col=` kwargs on `encode`).

## Rust-backed classes — PyO3 mechanics

Three rules, all load-bearing for `help()` and mkdocstrings rendering:

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

Without `signature`, `help()` and the docs site collapse the signature to `(*args, **kwargs)`. PyO3 0.28 (ferrum's pinned version) supports the `signature = (...)` form.

### Rule 3: Per-method `///` for documented `#[pymethods]`

Method-level `///` blocks above each function. Same NumPy template.

### Rule 4: Rebuild discipline

Batch all `///` edits in a Rust file before rebuilding once:

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
```

## Stub-param honesty (CRITICAL)

If a parameter is accepted by the function signature but never used in the body, document it as **"Reserved for future use (no-op today)"**, NOT as if it works. This is the most common docstring bug — the initial sweep needed amendments on every task for stub-param drift.

Example:
```python
def some_fn(
    field: str,
    real_param: int = 10,
    deferred_param: str | None = None,  # accepted but body never reads it
) -> Chart:
    """Summary.

    Parameters
    ----------
    field : str
        Column name.
    real_param : int, default 10
        Controls X (verified used in body).
    deferred_param : str, optional
        Reserved for future use (no-op today). When wired, will control Y.
    """
```

Always read the function body before documenting any parameter.

## Output-column accuracy (Rust transforms)

For transforms that emit new columns (Bin, Aggregate, BoxStats, Smooth, KDE, etc.), the `///` doc-comment MUST list the actual output column names. Read the Rust struct's output schema before writing the "Output columns" sentence. Multiple transforms had their docstrings amended during the sweep because they listed wrong column names — users binding `Chart.encode(x="<output_column>")` will get KeyError when the column name is documented incorrectly.

## Adding a new public symbol — checklist

When you add a new public class or function to `ferrum.__all__`:

- [ ] Write a NumPy-format docstring (template above).
- [ ] For Rust-backed: `///` on `#[pyclass]` + `#[pyo3(signature = (...))]` on `#[new]`.
- [ ] For user-facing symbols: include an `Examples` block with realistic column names.
- [ ] Read the function body — flag any unused params as "Reserved for future use (no-op today)".
- [ ] Verify any output column names match the actual Rust emit schema.
- [ ] Add the symbol name to `_DOC_ALLOWLIST` in `tests/test_docstring_coverage.py`.
- [ ] Run `uv run pytest` — the `test_allowlist_covers_all_public_api_after_sweep` test fails if you skip the allowlist.
- [ ] Run `uv run --no-sync ruff check src/ tests/` — D-rules will flag missing docstrings.

## Ferrum-specific style notes

- **Method naming**: marks are `mark_point`, `mark_line`, etc. — NOT `point`, `line`.
- **Output methods**: `show_svg`, `show_png`, `save` — NOT `render_svg` on Chart objects (that's a free function in `_core`).
- **Composition operators**: `|` is HConcat, `&` is VConcat, `+` is overlay/layer. Don't confuse "concatenation" with "overlay" in docstrings.
- **Coord declarations**: `CoordFlip` works in Phase 8a+; `CoordPolar`/`Geo`/`Fixed`/`Cartesian` raise `NotImplementedError` and are deferred.
- **Determinism**: any randomness uses seeded ChaCha8 RNG (per project CLAUDE.md). Document `seed` params honestly.

## When docstrings drift from signatures

If a method gains/loses a parameter, the docstring `Parameters` section must be updated in the same commit. Lint catches *presence*, not drift — review owns this.

## Where to read the long-form rationale

`docs/superpowers/specs/2026-05-11-docstrings-design.md` covers:
- Why these decisions and not others (§2 decisions table).
- The full taxonomy of public symbols and where each lives (§3).
- Lint configuration details (§6).
- The ratcheting strategy used during the initial sweep (§7).
