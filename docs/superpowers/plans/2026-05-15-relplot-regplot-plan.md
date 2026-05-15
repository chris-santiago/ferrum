# `relplot` + `regplot` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

## 1. Objective

Add `relplot` (figure-level relational scatter/line with faceting) and `regplot` (axes-level regression scatter, no faceting) to close the two meaningful seaborn API gaps in ferrum's EDA surface.

## 2. Spec references

No standalone spec — contracts are stated below. Follow existing patterns in:
- `src/ferrum/plots/distribution.py` — `displot` / `catplot` structure for `relplot`
- `src/ferrum/plots/regression.py` — `lmplot` structure for `regplot`

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `src/ferrum/plots/distribution.py` | add `relplot` |
| Modify | `src/ferrum/plots/regression.py` | add `regplot` |
| Modify | `src/ferrum/plots/__init__.py` | export both |
| Modify | `src/ferrum/__init__.py` | export both |
| Modify | `ferrum-spec.md` | add §3.x signatures for both |
| Create | `tests/test_relplot_regplot.py` | smoke + validation tests |

## 4. Constraints

- **No matplotlib** in ferrum's venv — ever.
- `relplot` and `regplot` are pure Python wrappers over existing marks/transforms; no new Rust code.
- `regplot` must delegate to `lmplot` internally — no parallel regression logic. Strip `row=`, `col=` from the call surface; everything else passes through.
- `relplot` valid kinds: `"scatter"` (default) and `"line"` only. Any other value raises `ValueError`.
- `size` encodes to `Size` channel; `style` encodes to `Shape` (scatter) or `StrokeDash` (line).
- Faceting in `relplot` via existing `FacetCol`/`FacetRow` mechanism — same as `catplot`/`displot`.
- Both functions follow the `(data, *, x, y, ..., mark, encode, properties, layers, theme, **encode_kwargs)` signature convention used by all ferrum figure-level functions.
- Both exported in `ferrum.__all__` and `ferrum.plots.__all__`.

## 5. Tasks

### Task 1: Implement `relplot`

- [ ] Add `relplot` to `src/ferrum/plots/distribution.py` after `catplot`
- [ ] Signature: `relplot(data, *, x, y, hue=None, size=None, style=None, col=None, row=None, kind="scatter", height=None, aspect=None, mark=None, encode=None, properties=None, layers=None, theme=None, **encode_kwargs) -> Chart`
- [ ] `kind="scatter"` → `mark_point()`; `kind="line"` → `mark_line()` (with `Aggregate` sort on x to enforce line ordering)
- [ ] Map `size` → `Size` encoding; `style` → `Shape` for scatter, `StrokeDash` for line
- [ ] `col`/`row` → `FacetCol`/`FacetRow` via same helper used by `catplot`
- [ ] Invalid `kind` raises `ValueError` with valid-values message
- [ ] Docstring: parameters, kinds table, one usage example per kind
- [ ] Verify: `uv run python -c "import ferrum; help(ferrum.relplot)"`

### Task 2: Implement `regplot`

- [ ] Add `regplot` to `src/ferrum/plots/regression.py` after `lmplot`
- [ ] Signature: `regplot(data, *, x, y, hue=None, method="lm", ci=95, order=1, scatter=True, scatter_kws=None, line_kws=None, truncate=True, x_jitter=None, mark=None, encode=None, properties=None, layers=None, theme=None, **encode_kwargs) -> Chart`
- [ ] Delegate to `lmplot(data, x=x, y=y, hue=hue, method=method, ci=ci, order=order, scatter=scatter, scatter_kws=scatter_kws, line_kws=line_kws, truncate=truncate, x_jitter=x_jitter, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme, **encode_kwargs)`
- [ ] Docstring: note it is the axes-level equivalent of `lmplot` (no faceting)
- [ ] Verify: `uv run python -c "import ferrum; help(ferrum.regplot)"`

### Task 3: Wire exports

- [ ] Add `relplot` to `src/ferrum/plots/__init__.py` imports and `__all__`
- [ ] Add `regplot` to `src/ferrum/plots/__init__.py` imports and `__all__`
- [ ] Add both to `src/ferrum/__init__.py` `from ferrum.plots import (...)` block
- [ ] Verify: `uv run python -c "import ferrum; print(ferrum.relplot, ferrum.regplot)"`

### Task 4: Tests

- [ ] Create `tests/test_relplot_regplot.py`
- [ ] `relplot` smoke: `kind="scatter"` with `hue`, `col` faceting — assert returns `Chart`
- [ ] `relplot` smoke: `kind="line"` — assert returns `Chart`
- [ ] `relplot` invalid kind: assert `ValueError`
- [ ] `regplot` smoke: basic `x`, `y` call — assert returns `Chart`
- [ ] `regplot` smoke: with `hue` and `method="loess"` — assert returns `Chart`
- [ ] Verify: `uv run pytest tests/test_relplot_regplot.py -v`

### Task 5: Update `ferrum-spec.md`

- [ ] Add `relplot` and `regplot` signatures to `ferrum-spec.md` §3.14 (figure functions) alongside `lmplot`
- [ ] Verify: `grep -n "relplot\|regplot" ferrum-spec.md` shows both

## 6. Acceptance checks

- `uv run pytest tests/test_relplot_regplot.py -v` — all pass
- `uv run pytest -x` — full suite green
- `uv run python -c "import ferrum; ferrum.relplot.__doc__"` — non-empty
- `uv run python -c "import ferrum; ferrum.regplot.__doc__"` — non-empty
