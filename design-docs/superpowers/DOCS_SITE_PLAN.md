# Ferrum Documentation Site — Plan

> Spec for the first cut of the public ferrum docs site. Status: draft, pending approval.

## Goals

1. Ship a coherent docs site that lets a new reader move from "what is this" through to "I can build my own chart" without leaving the site.
2. Treat **Concepts**, **Usage**, **Convenience API**, and **API Reference** as equally first-class — no thin guides, no docstring-dump fallback.
3. Publish only when all sections are populated (no visible "TODO" pages at launch).

## Non-goals

- Rust crate API docs (internal implementation; `cargo doc` only, not published).
- Versioned docs via `mike` or similar — defer until 1.0 cut.
- A custom theme. Use Zensical's default Material-style theme with light branding only.
- Live executable code blocks (Jupyter-style). Code snippets are static markdown, validated via CI doctest where practical.

## Tooling

| Concern | Choice | Notes |
|---|---|---|
| Site generator | **Zensical** (`>=0.0.40`, already a dev dep) | MkDocs-Material-derived; config in `zensical.toml`. |
| API reference | **`mkdocstrings` + `mkdocstrings-python`** | Officially supported in Zensical since v0.0.11 (preliminary; backlinks not yet wired). Style: NumPy (matches `tool.ruff.lint.pydocstyle.convention`). See **mkdocstrings migration discipline** below. |
| Hosting | **GitHub Pages** via GitHub Actions | Deploy on push to `main`. Custom domain TBD. |
| Diagrams | Mermaid (default in Zensical superfences) | For architecture/data-flow figures. |
| Math | Arithmatex (default; KaTeX/MathJax in browser) | For statistical notation. |
| Snippet validation | **`pytest-codeblocks`** in CI | Runs fenced ` ```python ` blocks in docs as tests. Chosen over `>>>`-doctest so prose stays readable. Opt-in per block via `<!--pytest-codeblocks:skip-->` for snippets that require fixtures. |

## Information architecture

Approach A (hybrid grouped). Top-level sections, with `Guide` as the only multi-level group:

```
Home
Get Started
  ├─ Install
  ├─ First plot (60-second tour)
  └─ Why Ferrum
Guide
  ├─ Concepts
  │   ├─ One chart model
  │   ├─ Stats in the rendering pipeline
  │   ├─ Dataframe pluralism
  │   ├─ Interactivity is a renderer, not a rewrite
  │   ├─ Model outputs are data
  │   └─ Performance & scale
  ├─ Marks & encodings
  ├─ Composition (layer · concat · facet)
  ├─ Themes
  ├─ Figure-level helpers (displot, lmplot, rocchart, …)   ← convenience API
  ├─ Model diagnostics
  └─ Interactive rendering
Gallery
  ├─ Index
  └─ 5 hand-crafted examples (see Gallery section below)
Comparison
  ├─ vs seaborn
  ├─ vs yellowbrick
  └─ vs scikit-plot
API Reference
  ├─ ferrum
  ├─ ferrum.encoding
  ├─ ferrum.figure
  └─ ferrum.themes
Changelog
```

### Why a dedicated "Figure-level helpers" page

The quality bar memory makes higher-level convenience functions (`displot`, `lmplot`, `rocchart`) first-class. Putting them in their own Guide page — separate from the raw API reference and separate from low-level Marks & Encodings — gives them the discoverability the user explicitly asked for. The page should answer: when to use the helper vs. the grammar, what the helper compiles into, and how to drop down into the grammar when you outgrow the helper.

## Page count and scope

Total authored pages: **~25 prose pages + 5 gallery pages + 1 API index + auto-generated API pages per module + 1 changelog ≈ 35 files**.

- Prose pages: Home (1), Get Started (3), Concepts (6), Guide (6), Comparison (3), Changelog (1), API index (1), Gallery index (1) = **22 hand-written**.
- Gallery example pages: **5**.
- API reference module pages: 4 (auto-rendered via mkdocstrings).

This is a multi-week content effort. "Build everything before publishing" stands, but we should treat it as a phased internal build (see Build phases below), not a single sitting.

## Per-section source mapping

| Site section | Primary source | Notes |
|---|---|---|
| Home | `ferrum-homepage-philosophy.md` (intro paragraphs) + hooks from `ferrum-marketing-copy.md` | Single landing page with hero + 3-card feature row drawn from `features.md`. |
| Why Ferrum | `features.md` "three things none of them have individually" | Pull verbatim with light editing. |
| Concepts (5 pages) | `FERRUM_PHILOSOPHY.md`, one ### section per page | Trim each to 200–400 words; link forward to relevant Guide/API. |
| Comparison | New content + `ferrum-homepage-philosophy.md` "What Ferrum takes from prior art" paragraph | Each migration page: small mapping table + 2–3 worked examples. |
| All Guide non-Concepts pages | New content | Author from current API + design intent. |
| Gallery | New content + curated subset of `tests/` figures | Hand-craft for clarity, not exhaustiveness. |
| API Reference | Auto-rendered from docstrings (107 documented symbols) | mkdocstrings consumes `src/ferrum/**`. |

## mkdocstrings migration discipline

Zensical's mkdocstrings support is explicitly marked preliminary, and Zensical plans to replace it with a native API reference system in the coming months. To keep that migration cheap:

- **Do not extend or customize mkdocstrings beyond config.** No custom Jinja templates, no handler overrides, no reliance on mkdocstrings-internal behavior or undocumented options.
- **Keep API content authoritative in NumPy-style docstrings in `src/ferrum/**`.** The renderer is replaceable; the docstrings are the source of truth.
- **Use vanilla directive syntax only** (`::: ferrum.foo` with documented `options:` keys). No Zensical-specific extensions; nothing that depends on plugin internals.
- **Confine all renderer-specific surface to two places**: the `[project.plugins.mkdocstrings.*]` block in `zensical.toml`, and the `::: ` directive blocks at the top of each `docs/api/*.md` page. Migration target = swap those two surfaces, leave everything else alone.

## Cross-reference strategy

- API ref symbols link back into Guide pages by anchor (e.g., the `Layer` API entry links to `guide/composition/#layer`).
- Guide and Concepts pages embed inline API references using mkdocstrings' identifier syntax where a single symbol is being introduced, instead of duplicating docstring prose.
- Gallery pages link both to the API symbols they use and to the Guide page that teaches the technique.
- Cross-links are designed in up front; retrofitting them later is more painful than building the convention now.

Caveat: mkdocstrings-on-Zensical does not yet support backlinks (auto-generated "Used by" lists). We will rely on hand-curated forward links until upstream lands backlinks.

## Build phases (internal, not staged releases)

We don't publish until everything is done, but we build in phases for sanity:

1. **Scaffold** — install `mkdocstrings`/`mkdocstrings-python`/`pytest-codeblocks`, write `zensical.toml`, wire nav skeleton, render API reference, run `zensical serve` locally. Deliverable: empty-content site that builds clean.
2. **Surface-area inventory (checkpoint, not a phase)** — before authoring any Guide pages, walk `src/ferrum/__init__.py`'s `__all__` and confirm every Guide page name maps to real, currently-public API on `main`. Specifically check `displot` / `lmplot` / `rocchart` (figure-level helpers), boxplot / silhouette / decision-boundary marks (Phase 10 features, unmerged at time of writing), and the Interactive rendering surface. Any Guide page without backing surface either waits for the feature to land, or its scope is narrowed to what `main` actually exposes. This is the discriminating gate for whether "build everything before publishing" holds against current `main`.
3. **Concepts + Home + Get Started** — port philosophy/homepage/features content into the right pages. Deliverable: a reader can land, understand the pitch, install, and read all 5 concept pages.
4. **Guide pages** — author the subset confirmed in the inventory checkpoint.
5. **Gallery** — hand-craft 5 examples covering: scatter w/ regression, faceted distribution, ROC + confusion matrix composition, SHAP-style diagnostic, interactive view.
6. **Comparison** — author the three migration pages.
7. **CI + deploy** — GitHub Actions workflow with: (a) Rust toolchain install, (b) `maturin build --release` + wheel install (required for mkdocstrings to import `ferrum._core`), (c) `zensical build`, (d) `pytest-codeblocks` against `docs/**`, (e) deploy to `gh-pages` on push to `main`. Local dev needs the same build chain (already paid for in this worktree).
8. **Polish & review** — link audit, broken-anchor check, snippet validation pass, dark/light theme pass.

## Configuration sketch

`zensical.toml` (key sections; full file derived from `zensical new` template + edits):

```toml
[project]
site_name = "Ferrum"
site_description = "Grammar-of-graphics statistical visualization for Python, with a Rust core."
site_url = "https://<github-org>.github.io/ferrum/"

nav = [
  { Home = "index.md" },
  { "Get Started" = [
      { Install = "getting-started/install.md" },
      { "First plot" = "getting-started/first-plot.md" },
      { "Why Ferrum" = "getting-started/why-ferrum.md" },
  ] },
  { Guide = [
      { Concepts = [ ... five pages ... ] },
      { "Marks & encodings" = "guide/marks-encodings.md" },
      { Composition = "guide/composition.md" },
      { Themes = "guide/themes.md" },
      { "Figure-level helpers" = "guide/figure-helpers.md" },
      { "Model diagnostics" = "guide/model-diagnostics.md" },
      { "Interactive rendering" = "guide/interactive.md" },
  ] },
  { Gallery = "gallery/index.md" },
  { Comparison = [ ... three pages ... ] },
  { "API Reference" = [ ... four module pages ... ] },
  { Changelog = "changelog.md" },
]

[project.plugins.mkdocstrings.handlers.python]
paths = ["src"]
inventories = ["https://docs.python.org/3/objects.inv"]

[project.plugins.mkdocstrings.handlers.python.options]
docstring_style = "numpy"
inherited_members = true
show_source = false
show_root_heading = true
```

## Directory layout

```
docs/
  index.md
  getting-started/
    install.md
    first-plot.md
    why-ferrum.md
  guide/
    concepts/
      one-chart-model.md
      stats-pipeline.md
      interactivity.md
      model-outputs-as-data.md
      performance-scale.md
    marks-encodings.md
    composition.md
    themes.md
    figure-helpers.md
    model-diagnostics.md
    interactive.md
  gallery/
    index.md
    01-scatter-regression.md
    02-faceted-distribution.md
    03-roc-confusion-composition.md
    04-shap-diagnostic.md
    05-interactive-view.md
  comparison/
    seaborn.md
    yellowbrick.md
    scikit-plot.md
  api/
    ferrum.md
    encoding.md
    figure.md
    themes.md
  changelog.md
  assets/
    images/
    stylesheets/
zensical.toml
.github/workflows/docs.yml
```

## Operational notes

- **Worktree**: docs work lives on `docs/continue` in `.claude/worktrees/docs-continue/`. The user explicitly requested a worktree this time despite the project's normal plain-branch convention; the `.venv` rebuild cost is already paid.
- **Dependencies to add** (dev group only): `mkdocstrings>=1.0`, `mkdocstrings-python>=2.0`, `pytest-codeblocks`. No runtime deps change.
- **Build chain is heavier than a typical docs site**: mkdocstrings imports `ferrum._core`, which is the compiled Rust extension. Both local builds and CI must run `maturin build` + wheel install before `zensical build`. Plan for 4–5 min CI runs (Rust compile dominates), not 30 seconds.
- **Snippet validation scope**: validate fenced ` ```python ` blocks in Get Started + Guide + Comparison. Gallery is opt-out (`<!--pytest-codeblocks:skip-->`) since those examples need fixtures and pre-rendered output.
- **Image assets**: gallery figures rendered offline and committed to `docs/assets/images/`. Source scripts live under `docs/assets/scripts/` so they're reproducible.
- **Enable "edit on GitHub" link**: turn on `content.action.edit` + `content.action.view` in the Zensical features list to encourage contribution. Cheap to enable, raises the contribution flywheel.

## Open questions (to confirm before execution)

1. **Site URL** — what's the GitHub Pages target (`https://<org>/<repo>` vs custom domain)? Affects `site_url`.
2. **Versioning of API ref** — first cut renders against the version on `main`. OK to leave unversioned for now?
3. **Search** — Zensical ships search by default; keep on.
4. **Theme palette** — accept Zensical default light/dark, or want a brand-specific accent color?
5. **Examples for "First plot"** — confirm we have a working `import ferrum as fr; ...` snippet that renders today, or do we need to author one against the current API surface as part of phase 1?
