# Output-Converter Rename Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: chris-code:subagent-driven-development. All edits are `.py`/`.md` → `python-coder` for source/tests, orchestrator for docs prose. Task 1 (the source rename) must land and pass before Stage 2; Stage 2 tasks are file-disjoint and run in parallel.

## 1. Objective

Rename the value-returning render methods to `to_svg`/`to_png` + add `to_html`, keep `show_svg`/`show_png` as deprecated aliases, and update all docstrings, guide docs, recipe scripts, and tests to match.

## 2. Spec references

- `design-docs/superpowers/specs/2026-06-04-output-converter-rename-design.md` §4–§7 (behavior, architecture, interfaces, invariants), §11 (doc file groups).

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `src/ferrum/_render.py` | rename `show_svg`/`show_png`→`to_svg`/`to_png`; add `to_html`; add deprecated `show_*` aliases; switch internal callers (`_repr_svg_`, `_repr_html_`, `to_png`, inset assembly) |
| Modify | `src/ferrum/composition.py` | rename `show_svg`→`to_svg` on the abstract base + all 7 subclasses; add `to_png`/`to_html` + deprecated aliases on base; switch internal child-SVG callers + `save` |
| Modify | `src/ferrum/display.py` | `save_chart`/`show_chart`/`save_chart_svg` call `to_svg`/`to_png` |
| Modify | `src/ferrum/chart.py`, `src/ferrum/render_config.py` | update docstring/doctest references naming `show_svg` |
| Test | `tests/test_output_converters.py` (new) | behavioral parity (`to_*` == prior output; `to_html()` == `save(.html)`) + `pytest.warns(DeprecationWarning)` on `show_*` |
| Modify | `tests/**` (mechanical) | sweep `show_svg`/`show_png` → `to_svg`/`to_png` in assertions (~139 files) |
| Modify | `scripts/{generate-showcase-pngs,render-recipe-pngs,generate-guide-pngs,gen_concept_pngs,profile_scatter}.py` + `design-docs/demos/{demo_grammar_marks,demo_diagnostics}.py` + `design-docs/themes/demo_themes.py` | switch runnable example/demo scripts to `to_png`/`to_svg` (8 files) |
| Modify | `README.md` | quick-start example `chart.show_png()` → `to_png()` |
| Modify | guide docs (10, see spec §11) | prose/examples use `to_*`; document family in `saving-and-export.md`, `to_html` in `interactive.md` |
| Modify | `.claude/skills/ferrum-docstrings/SKILL.md`, `.claude/agents/viz-power-user.md` | update internal automation instructions naming `show_svg`/`show_png` → `to_*` |
| Modify | `ferrum-spec.md` | dated note documenting `to_*` surface + deprecated aliases |
| Out of scope | `.claude/skills/audit-gallery/plots/*/ferrum_panel.py` (40) + `TODO.md` notes (10) + `audit.py` | ride the deprecated aliases until removal; see spec §11 (internal tooling, swept at alias removal) |
| Out of scope | `design-docs/superpowers/{plans,specs,audits}/*`, Rust doc-comments in `render/{png,binding}.rs` | historical records / cosmetic comments — not rewritten |

## 4. Constraints

- **Output byte-identical:** `to_svg`/`to_png` produce the exact prior artifacts; `to_html()` equals `save(.html)` content. Existing goldens / HTML-export tests pass unchanged.
- **Single implementation per format:** `show_*` aliases only `warnings.warn(DeprecationWarning, stacklevel=2)` + forward; no duplicated render logic.
- **Every composition subclass overrides `to_svg`** (none left implementing only `show_svg`); the abstract base's `NotImplementedError` names `to_svg`.
- `to_html` delegates to the same `_html.assemble_html` path `save(.html)` uses; `embed_wasm`/`toolbar` forward.
- `show()`, `save()`, `interactive()`, and `*Visualizer.show()` unchanged.
- Composition `to_png` keeps its `scale`-only signature (no `raster`), matching today's `show_png`.

## 5. Tasks

### Task 1: Source rename + aliases + `to_html` (single cohesive change)
- [ ] `_render.py`: rename method bodies to `to_svg`/`to_png`; add `to_html(*, embed_wasm=True, toolbar=True, raster=None)` delegating to the interactive assemble path; add `show_svg`/`show_png` deprecated shims; repoint `_repr_svg_`/`_repr_html_`/`to_png`/inset callers to `to_svg`.
- [ ] `composition.py`: rename `to_svg` on base (abstract) + all subclasses; repoint all child `*.show_svg()` calls and `save` to `to_svg`/`to_png`; add `to_png`/`to_html` + deprecated aliases on the base.
- [ ] `display.py`: `save_chart`/`show_chart`/`save_chart_svg` use `to_svg`/`to_png`.
- [ ] `chart.py`, `render_config.py`: update docstring/doctest text referencing `show_svg`.
- [ ] New docstrings state `to_*` **returns** and does not display; `show_*` docstrings note deprecation + replacement.
- [ ] Add `tests/test_output_converters.py`: `to_svg`/`to_png` parity, `to_html()` == `save(tmp.html)`, composition-view coverage, `pytest.warns(DeprecationWarning)` for `show_*`.
- [ ] Verify: `uv run pytest tests/test_output_converters.py -v`; `uv run pytest -n auto` (aliases keep the rest green pre-sweep).

### Task 2: Test-suite sweep (after Task 1)
- [ ] Mechanically replace `show_svg`/`show_png` → `to_svg`/`to_png` across `tests/**` EXCEPT `tests/test_output_converters.py` (which must keep exercising the deprecated names).
- [ ] Verify: `uv run pytest -n auto` green with no `DeprecationWarning` from the suite (except the dedicated test).

### Task 3: Recipe/example/demo scripts (parallel with 2, 4, 5)
- [ ] Switch the 5 `scripts/` files + the 3 `design-docs/` demos (`demos/demo_grammar_marks.py`, `demos/demo_diagnostics.py`, `themes/demo_themes.py`) to `to_png`/`to_svg` (8 files).
- [ ] Verify: run each script (or its `--check`/dry path) without error.

### Task 4: Guide docs + README + internal instructions (parallel)
- [ ] Update the 10 guide files (spec §11): replace `show_svg`/`show_png` with `to_*`; document the family in `saving-and-export.md`; add `to_html` to `interactive.md`.
- [ ] `README.md`: quick-start example `chart.show_png()` → `to_png()`.
- [ ] `.claude/skills/ferrum-docstrings/SKILL.md` + `.claude/agents/viz-power-user.md`: update `show_svg`/`show_png` references → `to_*`.
- [ ] Verify: `nox -s docs` (`zensical build --strict`) green; runnable code blocks execute.

### Task 5: Spec contract (parallel)
- [ ] `ferrum-spec.md`: dated note documenting `to_svg`/`to_png`/`to_html` and the deprecated `show_*` aliases.

## 6. Acceptance checks

- `uv run pytest -n auto` — all pass; deprecation path covered by `pytest.warns`.
- `nox -s docs` — strict build green; no `show_svg`/`show_png` in guide prose/examples or `README.md` except the deprecation note.
- `to_html()` output equals `save(.html)` file content (asserted).
- Grep: no `show_svg`/`show_png` remain in `src/ferrum` except the alias definitions + deprecation docstrings; none remain in `README.md`, `scripts/`, the 3 `design-docs/` demos, or the two `.claude` instruction files. (Historical `design-docs/superpowers/{plans,specs,audits}/*`, audit-gallery panels, and Rust doc-comments are out of scope per spec §11.)

## 7. Open questions

- None blocking. `to_json`/`to_pdf` are out of scope per spec §8.
