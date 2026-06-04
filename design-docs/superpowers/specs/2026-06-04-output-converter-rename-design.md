# Output-Converter Rename (`show_svg`/`show_png` → `to_svg`/`to_png`/`to_html`) Design Spec

> Status: **APPROVED 2026-06-04.** Resolves an API-coherence smell: the `show_`
> prefix conflates *display* (`show()` → `None`) with *conversion*
> (`show_svg()` → `str`, `show_png()` → `bytes`). The rename makes the verb match
> the behavior (`to_*` returns a value; `show` displays; `save` writes disk),
> matching Plotly/Altair/matplotlib conventions, and adds the missing `to_html()`.

## 1. Scope

Rename the value-returning render methods on `Chart` and every composition view
(`HConcatChart`, `VConcatChart`, `LayerChart`, `ConcatChart`, `JointChart`,
`RepeatChart`, `ClusterMapChart`, and the abstract base) to the conventional
`to_*` family, add `to_html()`, and keep the old `show_*` names as deprecated
back-compatible aliases. `show()` (display) and `save(path)` (disk) are unchanged.
Includes the docstring, guide-doc, and recipe/example updates the rename requires.

## 2. Goals

- `to_svg() -> str`, `to_png(*, scale=2.0) -> bytes`, `to_html() -> str` are the
  canonical converters on `Chart` and all composition views.
- `to_html()` returns the **interactive** HTML string — byte-identical to what
  `save(path.html)` writes — with the same `embed_wasm` / `toolbar` options.
- `show_svg()` / `show_png()` keep working (deprecated aliases) for at least one
  minor release; calling them emits a `DeprecationWarning` and forwards to `to_*`.
- `show()` still displays (Jupyter inline / browser); `save()` still writes disk.
- Docstrings, guide docs, and recipe/example scripts use the new names and state
  plainly that `to_*` **returns** (does not display).

## 3. Non-goals

- No change to rendering behavior, the auto-raster policy, the `raster=`/`scale=`
  semantics, or output bytes — same artifacts, new method names.
- No change to `save()`, `show()`, `interactive()`, or the WASM/SVG renderers.
- No change to the `*Visualizer.show()` sklearn-protocol method (different family).
- Not removing the deprecated aliases in this change (removal is a later release).

## 4. System behavior

- `chart.to_svg(*, raster=None)` returns SVG markup — identical output to today's
  `show_svg`.
- `chart.to_png(*, raster=None, scale=2.0)` returns PNG bytes — identical to
  today's `show_png`.
- `chart.to_html(*, embed_wasm=True, toolbar=True, raster=None)` returns the
  self-contained interactive HTML document as a string. `chart.to_html()` and
  `chart.save("x.html")` produce byte-identical content (string vs. file). The
  static SVG-in-HTML wrapper (`_wrap_svg_in_html`, `_repr_html_`) stays an internal
  detail of `show()`'s browser fallback and the Jupyter repr — it is **not** what
  `to_html()` returns.
- `chart.show_svg()` / `chart.show_png()` return the same values as before and emit
  a single `DeprecationWarning` naming the replacement.
- Composition views expose the same `to_svg`/`to_png`/`to_html` surface; the
  abstract contract a subclass implements is now `to_svg` (was `show_svg`).

## 5. Architecture

- **Canonical SVG producer is `to_svg`.** The method body currently named
  `show_svg` becomes `to_svg` on `Chart` (`_render.py`) and on every composition
  class (`composition.py`), including the abstract base's `NotImplementedError`
  contract. Every internal caller that currently invokes `show_svg`/`show_png`
  switches to `to_svg`/`to_png`: `Chart._repr_svg_`/`_repr_html_`, `to_png`'s
  internal `to_svg(...)` call, the inset/child-SVG assembly in `_render.py` and
  `composition.py`, and `display.save_chart` / `show_chart` / `save_chart_svg`.
- **`to_html` delegates to the existing interactive path.** It calls the same
  `ferrum._html.assemble_html(...)` pipeline `save_chart` uses for the `html`
  format (via `_render_scene` / the interactive render), returning the string
  instead of writing it. `embed_wasm`/`toolbar` forward unchanged.
- **Deprecated aliases are thin shims.** On each class, `show_svg`/`show_png`
  become one-line methods that `warnings.warn(..., DeprecationWarning,
  stacklevel=2)` and `return self.to_svg(...)` / `self.to_png(...)`. They carry no
  logic, so there is exactly one implementation per format.

## 6. Canonical interfaces / data contracts

```python
class Chart:                       # and each composition view
    def to_svg(self, *, raster: bool | None = None) -> str: ...
    def to_png(self, *, raster: bool | None = None, scale: float = 2.0) -> bytes: ...
    def to_html(self, *, embed_wasm: bool = True, toolbar: bool = True,
                raster: bool | None = None) -> str: ...

    # Deprecated aliases (warn + forward); kept ≥1 minor release.
    def show_svg(self, *, raster: bool | None = None) -> str: ...
    def show_png(self, *, raster: bool | None = None, scale: float = 2.0) -> bytes: ...

    def show(self, *, raster: bool | None = None) -> None: ...   # unchanged (display)
    def save(self, path, *, format=None, embed_wasm=True,
             raster=None, scale=2.0, toolbar=True) -> None: ...  # unchanged (disk)
```

Composition `to_png`/`to_html` keep their current signatures (composition `to_png`
takes only `scale`, not `raster`, matching today's `show_png`).

## 7. Invariants and constraints

- **Output-identical:** `to_svg`/`to_png` produce byte-for-byte the same artifacts
  as today's `show_svg`/`show_png`; `to_html` == `save(.html)` content. Existing
  golden/snapshot and HTML-export tests pass unchanged.
- **Back-compat:** `show_svg`/`show_png` remain callable with identical return
  types and values; only a `DeprecationWarning` is added.
- **Single implementation per format:** the alias delegates; no copy-paste of
  render logic.
- **Abstract-contract consistency:** every composition subclass overrides `to_svg`
  (no subclass left implementing only `show_svg`).
- `*Visualizer.show()` is untouched.

## 8. Key decisions and tradeoffs

- **`to_*` over keeping `show_*`.** Matches the cross-library convention (`to_*`
  returns, `show`/`display` is a side effect, `save`/`write` is disk) and removes
  the notebook footgun where `show_png()` dumps raw bytes instead of displaying.
- **`to_html` = interactive export, not the static SVG wrapper.** It is the string
  twin of `save(.html)`; the static wrapper has no independent value (it is
  `to_svg()` + boilerplate) and would create a second, conflicting meaning of
  "HTML". Lives on `Chart`, mirroring `save`.
- **Deprecate, don't break.** Aliases warn and forward so downstream code and the
  existing test suite keep working; removal is scheduled for a later release and
  tracked in the changelog/archaeology doc.
- **The `to_*` family is justified by in-memory-value vs. `save`'s disk-write**, for
  every format. `save()` cannot return a string/bytes to embed in a response, pass
  on, hash, or assert in a test — that is exactly what `show_svg`/`show_png` already
  do, and the rename only fixes their name. "`save` already supports format X" is
  therefore *not* a reason to omit `to_X` (it supports svg/png/html too).
- **`to_json` deliberately omitted — name collision, not redundancy with `save`.**
  `Chart.to_json()` already exists and returns the **ChartSpec** (the declaration),
  whereas `save(".json")` writes the **rendered scene graph** (`_render_scene_json`)
  — a different artifact. There is no clean slot for a scene-JSON converter under
  `to_json`, and in-memory scene-JSON has no demonstrated demand. (The latent
  `to_json`=spec vs. `save(.json)`=scene inconsistency is noted as a separate smell,
  out of scope here.)
- **`to_pdf` deferred — additive but niche.** Unlike the others, there is *no*
  in-memory PDF path today (`save(".pdf")` rasterizes SVG→PDF straight to disk via
  `save_chart_svg`), so `to_pdf() -> bytes` would add a genuine capability rather
  than duplicate one. Deferred for lack of a demonstrated in-memory-PDF use case;
  revisit if one appears.

## 9. Acceptance criteria

- `to_svg`/`to_png`/`to_html` exist on `Chart` and all composition views and return
  the correct types; `to_html()` output equals `save(.html)` output.
- `show_svg`/`show_png` still return identical values and raise `DeprecationWarning`
  (asserted by a dedicated test using `pytest.warns`).
- `uv run pytest` green (suite migrated to `to_*` where it asserts output; the
  deprecation path covered by its own test).
- Docstrings for `to_*` state "returns … does not display"; `show_*` docstrings
  note the deprecation and the replacement.
- Guide docs and recipe/example scripts reference `to_svg`/`to_png`/`to_html`; no
  `show_svg`/`show_png` remain in prose or runnable examples except where
  illustrating the deprecation.
- `ferrum-spec.md` updated (dated note) to document the `to_*` surface and the
  deprecated aliases.

## 10. Validation strategy

- **Behavioral:** assert `to_svg`/`to_png` equal the prior `show_*` outputs for a
  representative chart; assert `to_html()` string equals `save(tmp.html)` file
  content; assert composition views expose and correctly implement the family.
- **Deprecation:** `pytest.warns(DeprecationWarning)` on `show_svg`/`show_png`,
  confirming they still return the right value.
- **Docs build:** `nox -s docs` (`zensical build --strict`) green; any runnable
  code blocks using `to_*` execute (pytest-codeblocks).
- **Full gate:** `nox` (lint + tests + build) before release.

## 11. Documentation work (in-scope file groups)

- **Docstrings (`src/ferrum`):** new `to_*` docstrings on `Chart` (`_render.py`)
  and composition views (`composition.py`); deprecation notes on `show_*`; fix the
  doctest examples (`_render.py`, `chart.py:754`) and the `RenderConfig` /
  `chart.py:2912` references that name `show_svg`.
- **Guide docs (10 files):** `guide/saving-and-export.md` (primary — document the
  `to_*` family + `save` + `show`, and add `to_html`), `guide/interactive.md`
  (mention `to_html` as the string form of the interactive export),
  `guide/{marks-encodings,figure-helpers,recipes,themes,model-diagnostics,
  composition}.md`, `getting-started/first-plot.md`, `changelog.md`.
- **Recipe/example scripts (5):** `scripts/{generate-showcase-pngs,
  render-recipe-pngs,generate-guide-pngs,gen_concept_pngs,profile_scatter}.py`
  switch `show_png`/`show_svg` → `to_png`/`to_svg`.
- **Tests (~139 files):** mechanically migrate output-assertion calls to `to_*`
  (scriptable), leaving one dedicated deprecation test on the `show_*` aliases.
