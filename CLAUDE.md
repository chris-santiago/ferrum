# Ferrum — Project Instructions

Ferrum is a Rust-backed Python statistical visualization library. The Python layer is the declaration API; the Rust layer (`crates/ferrum-core`, compiled to `ferrum._core`) is the computation engine. Data moves between them once, over the Arrow C Data Interface (CDI).

---

## Start every session here

1. **Read `design-docs/superpowers/ferrum-phases.md`** — it lists all 12 implementation phases, their dependency order, done criteria, and the current status of each. Find the first phase that is not `done` and start there.

2. **Read the phase's spec doc** (linked in the phases table) before writing any code. If no spec exists yet, run `chris-code:brainstorming` before touching anything.

3. **Read `ferrum-spec.md`** (repo root) only if you need the user-facing API contract for the phase you are working on. It is the concept spec, not the implementation guide.

---

## Build commands

| Action | Command |
|---|---|
| Install / rebuild Rust extension | `unset CONDA_PREFIX && uv run --no-sync maturin develop` |
| Release build | `unset CONDA_PREFIX && uv run --no-sync maturin develop --release` |
| Run tests | `uv run pytest -n auto` |
| Run scale tests | `uv run pytest -m slow` (10k–50k row tests, skipped by default) |
| Rust-side tests | `DYLD_LIBRARY_PATH=$(uv run python -c "import sys; print(sys.base_prefix + '/lib')") cargo test` |
| Clippy (core) | `DYLD_LIBRARY_PATH=$(uv run python -c "import sys; print(sys.base_prefix + '/lib')") cargo clippy -p ferrum-core -- -D warnings` |
| Verify skeleton | `unset CONDA_PREFIX && uv run --no-sync python -c "from ferrum._core import ChartSpec; s=ChartSpec(mark='point', x='a', y='b'); assert s == ChartSpec.from_json(s.to_json()); print('OK')"` |
| Build WASM module | `source ~/.cargo/env && wasm-pack build crates/ferrum-wasm --target web --out-dir ../../src/ferrum/_wasm/` |
| Build WASM (release) | `source ~/.cargo/env && wasm-pack build crates/ferrum-wasm --target web --release --out-dir ../../src/ferrum/_wasm/` |
| WASM clippy | `source ~/.cargo/env && cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings` |

> **Note:** `--no-sync` is required for `maturin` commands to avoid a conflict between
> conda's `CONDA_PREFIX` and uv's `VIRTUAL_ENV`. Miniforge base sets `CONDA_PREFIX` even
> outside conda envs, which maturin rejects when uv also sets `VIRTUAL_ENV`. The
> `unset CONDA_PREFIX` prefix clears it for that shell invocation only. Source
> `~/.cargo/env` first if `cargo` is not on your PATH (`source ~/.cargo/env`).

> **macOS `cargo test` note:** On macOS with uv-managed Python, the test binary cannot
> resolve `@rpath/libpython3.x.dylib` at runtime without `DYLD_LIBRARY_PATH` pointing to
> the Python lib directory. The command above uses `sys.base_prefix` (not `sysconfig.get_config_var('LIBDIR')`,
> which returns a bogus `/install/lib` on uv-managed cpython builds).
> **The `sys.base_prefix` recipe fails when the test binary links a different libpython than
> uv's base Python** (check with `otool -L target/debug/deps/ferrum_core-*`): pyo3 links
> whichever Python its build found, e.g. Homebrew's. In that case point `DYLD_LIBRARY_PATH`
> at *that* Python's lib dir instead — on a Homebrew python@3.13 machine:
> `DYLD_LIBRARY_PATH=/opt/homebrew/opt/python@3.13/Frameworks/Python.framework/Versions/3.13/lib cargo test -p ferrum-core`.
> This is a macOS SIP + uv RPATH constraint; it does not affect `maturin develop` or pytest.
> **Clippy: never add `--tests`** — it can mask dead-code lints on production items (observed
> 2026-09-03); measure with the documented no-`--tests` recipe and judge by delta against the
> pre-existing in-crate baseline (~166).

`pip install -e .` will **not** compile the Rust extension. Always use `maturin develop`.

---

## Nox sessions

| Session | Command | What it does |
|---|---|---|
| Lint | `nox -s lint` | Runs `ruff check --fix` + `ruff format` on `src/` and `tests/` (current env) |
| Test | `nox -s test` | `uv sync --all-extras --all-groups` then `pytest` (isolated) |
| Build | `nox -s build` | `uv build` — produces wheel + sdist in `dist/` (isolated) |
| Docs | `nox -s docs` | `zensical build --strict` — fails on warnings (current env) |

Pass extra pytest args via `nox -s test -- -k test_name`.

---

## API reference pages

The split API reference pages under `docs/site/api/` (`chart.md`, `marks.md`,
`composition.md`, `statistics.md`, etc.) are **generated** by
`scripts/gen_api_pages.py` — do not hand-edit them. Run it after any change to the
public API surface (new/renamed/moved `ferrum.__all__` symbol, new submodule):

```
unset CONDA_PREFIX && uv run --no-sync python scripts/gen_api_pages.py        # write pages
unset CONDA_PREFIX && uv run --no-sync python scripts/gen_api_pages.py --check # report partition only
```

It partitions `ferrum.__all__` by each symbol's *defining* module (`obj.__module__`)
and emits each page as `::: ferrum` with an explicit `members:` list, so the pages
own the canonical **`ferrum.X`** anchors that the docs' `[ferrum.X]` autorefs target.
This is load-bearing: the old monolithic `api/ferrum.md` (`::: ferrum`) is retired to
a redirect stub precisely because it owned every `ferrum.X` anchor and forced every
cross-reference onto one too-large-to-render page. If you add a public symbol whose
defining module isn't mapped, the script prints it under `UNHOMED` — add a rule (or a
new page in `PAGES`) and update the nav in `zensical.toml` + the `api/ferrum-toc.md`
table. Verify with a **cache-cleared** build (`rm -rf site .cache && zensical build`),
since autoref "unresolved" warnings are non-deterministic two-pass false positives —
trust the final HTML (`grep 'href=.*api/ferrum/#' site/` should return nothing).

---

## Releasing

Use the `/release` skill. It bumps the version in `pyproject.toml` and `Cargo.toml`, generates a changelog from conventional commits, updates `docs/site/changelog.md`, and creates a GitHub release after confirmation. The `publish.yaml` workflow then builds manylinux/macOS/Windows wheels via `maturin-action` and publishes to PyPI via trusted OIDC publishing.

- **PyPI package name:** `ferrum-viz` (import name stays `ferrum`)
- **Version lives in:** `pyproject.toml` + `Cargo.toml` (workspace root)
- **Workflow:** `.github/workflows/publish.yaml` — triggers on release or `workflow_dispatch`

---

## Hard constraints (never violate)

- **No matplotlib.** Not as a dependency, not as a dev dependency, not as an optional extra. Ever.
- **No global mutable state.** No module-level config objects, no module-level theme rebinding. Themes are values passed to `Chart`; per-chart `.theme()` always wins. The single documented exception is `ferrum.set_default_theme()` (phase 8a+), which mutates a per-thread `contextvars.ContextVar` — scope-bounded, automatic-revert when used as a context manager, and overridden by per-chart `.theme()` at render time. Do not introduce other process-scoped mutators.
- **`ferrum-spec.md` is the API contract.** If implementation diverges, update the spec with a dated note. Never silently drift.
- **`cargo test` must pass** before any phase (2+) is marked done. Phase 1 is the only exception.
- **No seaborn, sklearn, yellowbrick, or scikit-plot imports in `src/ferrum/`.** They exist solely as gallery-audit comparators in isolated PEP 723 envs (see Gallery audit).
- **No `unsafe` blocks** without clear justification documented in a comment.
- **Seeded randomness only in transforms** (`rand_chacha`, seed threaded from the spec). Non-seeded randomness breaks byte-deterministic rendering.
- **No unconditional `extension-module` feature.** Guard with `#[cfg(feature = "extension-module")]`; unconditional use breaks `cargo test`.
- **Goldens are not blessed until visually inspected.** SVG byte-equality is necessary but not sufficient — historically goldens were committed that rendered with missing elements, blank panels, or mis-stacked bars, and the byte-diff tests still passed because the implementation matched the *broken* golden. Whenever you add or regenerate any `tests/goldens/**/*.svg` (or `tests/test_phase_9_e2e/goldens/*.svg`), you must rasterize it to PNG via `python scripts/snapshot-goldens.py <name>` (or `python scripts/snapshot-goldens.py` for all), `Read` each resulting PNG, and confirm the chart renders correctly **before committing**. The helpers live in `tests/_snapshots.py` (`snapshot_golden()`, `rasterize_svg()`, `find_goldens()`, and `regen_and_verify(golden_path, svg)` — the preferred entry point for regen scripts: writes the SVG, rasterizes the PNG, and prints both paths to stdout in a single call so the inspection PNG cannot be silently skipped). `resvg-py` (in the dev dependency group) is the rasterizer. **Caveat:** `resvg-py` silently drops paths when an SVG contains many thousands of polygon/path elements (observed at ~9.5k paths in dense KDE-contour fills). Before concluding a chart is broken from the PNG, sanity-check the SVG itself — `grep -oE 'd="M' tests/.../foo.svg | wc -l` — and look at the x-range of the first numeric coord on each path. A chart that looks like a tiny patch in the PNG but has thousands of paths spanning the plot extent in the SVG is a *renderer-side* truncation, not a real bug.
- **Do not `git push`** unless the user explicitly asks.
- **Confirm before committing to `main`** on non-trivial work. Phase 1 commits directly to main by user decision (greenfield); subsequent phases use feature branches unless the user says otherwise.
- **Regression tests after every bug fix.** Invoke `/regression-test` before declaring any fix complete or committing. Enforced by a PreToolUse hook on `fix/` branches; on other branches, the hook reminds but doesn't block.

---

## Implementation philosophy (Phase 12 and beyond)

**Do the work now. Do it the right way. Enable a better end-user experience now.**

- Do NOT propose "defer X to a later phase / follow-up ticket" as a scope-reduction strategy.
- If a `ferrum-spec.md` parameter is hard to ship completely, ship it completely (with whatever Rust transform, mark, encoding, or position-adjustment subsystem it needs) — not a warn-fallback or `NotImplementedError`.
- Use sub-phase decomposition to manage build order, not to drop scope.
- "Implement everything fully" is the default. No Warn-fallbacks. **NotImplementedErrors ARE NOT ACCEPTABLE.**

This rule governs all open phases; it does not retroactively reopen closed phases (1–12 spec completeness are done; Phase 12 extension points is the current frontier).

---

## Coding agent dispatch rule

**All coding tasks in this project must be delegated to the language-specific coding agent.** Do not use `general-purpose`, `claude`, or `Explore` agents for code that writes or modifies `.py` or `.rs` files.

| Task touches | Dispatch to |
|---|---|
| `src/ferrum/`, `tests/`, `scripts/` (Python) | `python-coder` agent |
| `crates/ferrum-core/`, `crates/ferrum-wasm/` (Rust) | `rust-coder` agent |
| Both Python and Rust | Dispatch both agents with clear boundaries; Python agent handles `.py`, Rust agent handles `.rs` |
| Read-only exploration, search, or analysis | `Explore` agent (unchanged) |

The coding agents are the chris-code plugin's `python-coder` / `rust-coder` (the repo-local copies were retired 2026-08-28 — the plugin versions carry the same inlined review principles plus the typed-record contract). They internalize their review principles in their own system prompts, so code should pass the lite-review gate on first attempt; the ferrum hard constraints and severity escalations in this file bind them too. The orchestrator still handles staging, lite-review dispatch, and commits.

**Model selection:** Coding agents default to Sonnet (set in their frontmatter). Override to Opus via `model: "opus"` on the Agent call when the task requires significant architectural judgment — e.g., cross-subsystem refactors, complex Rust lifetime/type reasoning, or tasks that would otherwise need multiple re-dispatches.

---

## Bug fixes and cascading issues

Bug fixes must be **cohesive and paradigm-respecting** — do not paper over a symptom in a way that violates the existing design (layers vs. chart-level computation, Python vs. Rust responsibilities, etc.). When a fix reveals a cascading issue, understand *why* before touching anything else.

- Identify the root cause in the existing architecture before writing any code.
- A fix that works around a paradigm (e.g., moving data the wrong direction across the PyO3 boundary, duplicating transform logic in Python that belongs in Rust) is not a fix — it is deferred complexity.
- If the correct fix requires changing a foundational invariant, surface that to the user first.

---

## Where things live

| Artifact | Path |
|---|---|
| Implementation phases & roadmap | `design-docs/superpowers/ferrum-phases.md` |
| Concept + API specification | `ferrum-spec.md` |
| Per-phase design specs | `design-docs/superpowers/specs/YYYY-MM-DD-<slug>-design.md` |
| Per-phase implementation plans | `design-docs/superpowers/plans/YYYY-MM-DD-<slug>-plan.md` |
| Python package source | `src/ferrum/` |
| Rust extension crate | `crates/ferrum-core/` |
| Python tests | `tests/` |
| Golden SVG → PNG snapshot helper | `scripts/snapshot-goldens.py`, `tests/_snapshots.py` |
| **Coding agents** | **chris-code plugin `python-coder` / `rust-coder`** (local copies retired 2026-08-28) |
| Lightweight review agents | chris-code plugin `{python,rust}-review-lite` (local copies retired 2026-08-28) |
| Review verdicts (gitignored) | `.claude/output/review-lite/` |
| Heavyweight code-review skills | `.claude/skills/{python,rust}-review/` |
| Gallery audit skill | `.claude/skills/audit-gallery/` |
| Gallery audit agents | `.claude/agents/gallery-{judge,fixer}.md` |
| Bug-hunt skill | `.claude/skills/bug-hunt/` |
| Bug-hunter agent | `.claude/agents/bug-hunter.md` |
| Test-sweep skill | `.claude/skills/test-sweep/` |
| Interactive wiring audit | `.claude/skills/audit-interactive/` |
| Interactive auditor agent | `.claude/agents/auditor-interactive.md` |
| PyO3 binding audit | `.claude/skills/audit-pyo3/` |
| PyO3 binding auditor agent | `.claude/agents/auditor-pyo3-binding.md` |
| Scene pipeline audit | `.claude/skills/audit-scene-pipeline/` |
| Scene pipeline auditor agent | `.claude/agents/auditor-scene-pipeline.md` |
| Theme wiring audit | `.claude/skills/audit-theme/` |
| Theme wiring auditor agent | `.claude/agents/auditor-theme-wiring.md` |
| Schwabish text-integration skill | `.claude/skills/schwabish/` |
| Schwabish agents | `.claude/agents/schwabish-{judge,fixer}.md` |
| Code archaeology skill | `.claude/skills/code-archaeology/` |
| **Code archaeology report** | **`design-docs/superpowers/followups/2026-05-15-code-archaeology.md`** |
| Docs audit skill | `.claude/skills/audit-docs/` |
| **API reference page generator** | **`scripts/gen_api_pages.py`** |
| Regression test skill | `.claude/skills/regression-test/` |
| Release skill | `.claude/skills/release/` |
| Nox sessions | `noxfile.py` |
| Publish workflow | `.github/workflows/publish.yaml` |
| Automations index | `.claude/README.md` |

---

## Composite rendering (Phase B, 2026-07-05)

All composition (HConcat/VConcat/Concat/Joint/ClusterMap/Repeat/Layer) renders
through ONE Rust composite entry per output kind (`render_composite_svg` /
`render_composite_interactive`): Python lowers the composition to a composite
spec tree (`src/ferrum/composition.py`, `_lower_any`/`_build_grid_tree`), Rust
resolves shared scales across leaves, plans the layout, and emits one
SceneGraph. There is NO legacy fallback — a composition that cannot lower
raises a typed `ValueError`. The former W4/W5 interactive-export limitations
(raw-node offsets; JointChart's flat interactive layout) were subsumed:
raw-node defs place per-panel via Rust clip uniquification, and JointChart
interactive is a true ratio grid (verified by headless WASM captures,
`.claude/output/phase-b-captures/`). The one deliberate interactive exception:
`LayerChart._render_interactive` renders the merged single-panel Chart (the
flat path) because selections/hit-testing require overlays to be ONE scene
panel. Deliberate behavior change: an all-empty composition (e.g. zero-row
pairplot) raises instead of rendering a blank grid.

**Overlay chrome dedup is total (residuals batch, #89, 2026-08-28).** For an
Overlay group whose children are all leaves, every leaf lays out against one
shared plot rect — the per-side intersection of the leaves' natural gutters —
computed in a composite planning pre-pass and threaded into `compute_layout`,
so layout never needs post-hoc correction. Suppression of non-primary-leaf
chrome (grid, axes, above-marks axis chrome, scene title) is coupled to
imposition: it applies to every leaf that actually laid out against the
shared rect, which is every leaf in a well-formed group. The former
per-leaf safety-check gate (`overlay_imposition_safe`) and its
`chrome_suppressed` plumbing are deleted — the three former refusal shapes
(a non-primary leaf with its own legend, a `zindex >= 1` axis, or a
`z="below_marks"` annotation) now render one chrome with content preserved,
not per-leaf duplicated chrome. `Panel` gained two typed, serde-default slots
for this: `below_marks: Vec<SceneNode>` (painted after `grid`) and
`chrome_above: Vec<SceneNode>` (zindex ≥ 1 axis/grid nodes, painted after
`axes`, before `annotations`); both SVG and WASM walkers paint them
identically, and scenes that don't use these slots serialize byte-identically.
Three cases still keep per-leaf chrome: an Overlay node whose children are
NOT uniformly direct leaves (a nested-composite child) is structurally
excluded from grouping up front, verified unreachable from Python lowering;
a degenerate per-side-max intersection (no common plot area) drops that
group's suppression and emits `RenderWarning::OverlayGuttersDiverged` naming
the layer count, so a doubled-chrome render is diagnosable; and a member
whose own layout fails during the imposition pre-pass also drops its
group's suppression, with no warning, since that leaf's render re-runs the
same layout moments later and raises the failure as a typed error instead.
See `ferrum-spec.md`'s composition section
(2026-08-28 note) for the user-facing behavior contract.

**Independent-y LayerChart extends this exception to BOTH kinds (secondary
y-axis, 2026-07-11, #52).** `LayerChart(a, b, resolve={"y": "independent"})`
(and its `SecondaryY` sugar, which desugars to an appended independent-y
layer) routes **static `to_svg` and interactive `.interactive()` both**
through the merged flat path — not just interactive. This is a documented,
symmetric extension of the interactive-only exception above: since the
overlay composite tree cannot serve interactive's one-panel contract, a
composite-side implementation of per-layer y-scales would mean two
mechanisms for one feature, so both output kinds resolve per-layer y-scale
slots from the same single-panel spec. This resolves, for the
independent-y case specifically, the former S2 archaeology finding that
static (Rust overlay tree) vs. interactive (merged flat chart) shared
color/size math could diverge — see
`design-docs/superpowers/followups/2026-05-15-code-archaeology.md`. Default
and `resolve={"y": "shared"}` LayerChart output is unaffected: static still
renders via the composite overlay tree, interactive via the merged flat
path, unchanged from before.

---

## Rust test-module convention

Default to an inline `#[cfg(test)] mod tests { ... }` at the bottom of a
single-file module. Promote to a directory module (`mod.rs` + a sibling
`tests.rs`) when the ported/test body would exceed the source module —
not on a line-count threshold, but when a module accumulates enough
relocated or generated test coverage that inlining it would make the
source file harder to read than the split. Precedents: `render/scale_resolve/`
(`mod.rs` + `auxiliary.rs`/`color.rs`/`domain.rs`/`positional.rs`/`seam.rs` +
`tests.rs`), `projection/` (`mod.rs` + `tests.rs`), `transform/stats/`
(`mod.rs` + `tests.rs`). When promoting, leave a short rationale comment at
the top of the new `tests.rs` naming what it carries and why it couldn't stay
inline (see `projection/tests.rs`'s and `transform/stats/tests.rs`'s doc
comments for the pattern to mirror).

**Python test-file convention** (the parallel rule, stated in `tests/test_boxen_palette.py`'s module docstring): a findings-ID-named test file (`test_finding_p*.py`) stays scoped to pinning the specific regression or disposition the finding describes; net-new feature coverage for a mark or figure function gets its own `test_<feature>.py` module, even when the feature landed in the same batch as a findings fix.

---

## Known open gaps (code archaeology)

**Read `design-docs/superpowers/followups/2026-05-15-code-archaeology.md` before working on any of the subsystems below.** It is the living tracker of unimplemented features, silently dropped parameters, dead code paths, and spec-vs-implementation gaps discovered via a full-source sweep. Many items are resolved; the remaining open items are:

- **Features:** `mark_ribbon(interpolate=...)`, `mark_hex(stroke=)`, `mark_function(clip=False)`
- **Missing spec implementations:** full palette library (`ferrum.Grid`, `SceneNode::Raw` WASM, and `compare=` diagnostic routing were resolved by the render-gaps work, compare-aggregate work, and Phase B composite unification, 2026-05/2026-07)
- **Rust dead code:** `ticks.rs` blanket `#[allow(dead_code)]`, `CategoricalPalette`/`Scheme` module, `OutlierRow`, `apply_transforms*`, label `MarkBatchKind::Text`
- **Resolved (2026-05-22):** Opacity semantics (`fill_opacity`/`stroke_opacity` now applied per-channel in WASM shaders; `opacity` no longer double-applied to stroke) and annotation z-order (annotation mesh drawn after marks, not before) were fixed in the `feat/rtree-toolbar` merge.
- **Resolved (2026-08-28, residuals batch #93):** `Description`/`Href`/`Key` encoding all now work end to end. `Key` reached the scene graph without a consumer as of the 2026-08-27 P1 remediation; it now has one — the WASM interactive runtime pairs marks by `key` for transition matching, with enter/exit fades for keys present on only one side of an update, including for packed (≥1000-mark) batches via a `HAS_KEYS` sidecar. Static SVG output is unaffected and stays byte-identical with and without `key=`; unkeyed interactive scenes are unaffected too. Key-based selection membership and key-addressed hit-testing remain unbuilt (logged follow-ups, not this batch). See the archaeology doc's `Key` row and `ferrum-spec.md` §3.2's 2026-08-28 note.

When fixing a bug or adding a feature that overlaps with an open item, update the archaeology doc's status column and check off the corresponding action-list entry.

---

## Gallery audit (default-output comparison)

A reproducible side-by-side audit of ferrum's default plot output against canonical Python libraries (sklearn, seaborn, yellowbrick, scikit-plot). Use it to find where ferrum's defaults lack information or visual quality that competitors ship out of the box — missing AUC annotations, missing reference lines, wrong axis labels, missing per-cell counts on confusion matrices, etc.

- **Skill** — `.claude/skills/audit-gallery/`. Trigger with `/audit-gallery` or "audit our plots / compare ferrum to seaborn-sklearn-yellowbrick / what's missing from our default plots". 38 rows, all wired (see `RESUME.md` for the full table). Generation is a PEP 723 script (`audit.py generate`); judging runs as `gallery-judge` subagents in-session (no `ANTHROPIC_API_KEY` needed); report is a script (`audit.py report`).
- **Agent `gallery-judge`** — judges one row by reading panel PNGs and applying `rubric.md`. Dispatched in parallel, one per row, to keep parent context clean. Writes `verdict.md` with YAML frontmatter + prose.
- **Agent `gallery-fixer`** — works through `REPORT.md`'s prioritized punchlist autonomously after an audit run, closing default-behavior gaps (Python composite-mark expansion preferred over Rust changes — see `design-docs/architecture/ARCHITECTURE.md` "Composite marks" section).
- **Output** — `gallery/` symlink at repo root → `.claude/output/audit-gallery/`. Contains `REPORT.md`, per-row PNGs, per-row `verdict.md`. Gitignored.
- **Comparator isolation** — sklearn, seaborn, yellowbrick, scikit-plot run in isolated PEP 723 envs via `uv run --no-project --script`. **Never add any of them to `pyproject.toml`** — they exist solely as audit comparators. Matplotlib stays out of ferrum's deps per the hard constraint above.
- **When new ferrum APIs land** that unblock previously-BLOCKED rows, kick off a session with `"Wire row <N> — ferrum.<func> just landed"`. Claude reads `RESUME.md`, follows the Resume protocol there (copy `plots/01_roc/<library>_panel.py` as a template, swap calls, update the row's `config.toml`, regenerate). The skill auto-detects unwired READY rows on invocation and offers to wire them before running.

---

## Code-quality guardrails

Commit gates and review-lite dispatch are handled by the chris-code plugin (`executing-plans`, `subagent-driven-development`). The rules below are ferrum-specific escalation triggers that go beyond the plugin's defaults.

### Ferrum severity escalations (applied by the commit gates)

The review gates read this file; these hard-constraint violations map to fixed severities regardless of how small the diff is:

- New `matplotlib` / `seaborn` / `sklearn` / `yellowbrick` / `scikit-plot` import in `src/ferrum/**` → **S5**.
- New `NotImplementedError` or warn-fallback in a chart factory or other Phase 9+ code → **S4**.
- New `unsafe` block in `ferrum-core` → **S4** minimum; **S5** if it touches data crossing the PyO3 boundary.
- New non-seeded randomness (`rand::thread_rng`, `SystemRandom`, platform RNG) inside a transform → **S5** (byte-deterministic rule).
- Unconditional `extension-module` feature use → **S5** (breaks `cargo test`).

**Exception:** Documentation-only changes (`*.md`, `docs/**`, or comments-only diffs in source files) can commit without review-lite dispatch.

### When to escalate to heavyweight review

The lite agents gate every commit; the heavyweight skills (`python-review` / `rust-review`) are subsystem-level audits that produce a six-section report and a refactor roadmap. They are deliberate interventions — not appropriate for every diff. The orchestrator should **offer the heavyweight skill proactively, without waiting for the user to type `/python-review` or `/rust-review`**, when any of these conditions hold:

1. **A phase is about to be marked done.** Before transitioning a phase in `design-docs/superpowers/ferrum-phases.md` from in-progress to done, run the matching heavyweight skill on the subsystems that phase modified. The lite agent sees one commit at a time; the heavy skill sees how the subsystem hangs together after a phase's worth of accumulated work.
2. **The lite agent has escalated 2+ times on the same module within the session.** Repeated escalation means the band-aid pattern is broken — lite is treating diff-level symptoms while the structural issue lives in the module. Escalate to a heavy review scoped to that module.
3. **A new public-API surface is added to an existing family.** A new `*_chart` in `src/ferrum/figures.py`, a new mark in `src/ferrum/marks/`, a new transform in `crates/ferrum-core/src/transform/`, a new encoding channel, or a new visualizer subclass all warrant a heavy review of the whole family afterwards to catch sibling drift (signature mismatches, naming drift, inconsistent return shapes, parallel-API mismatch) before the new member becomes entrenched. The `calibration_chart` signature drift caught in May 2026 is the canonical example: it lived for a phase before anyone audited the family.
4. **The user expresses a coherence smell in natural language.** "This feels off", "sibling drift", "X drifted from Y", "utility module sprawl", "inconsistent return shapes", "parallel-API mismatch", "this module feels overgrown" — the heavyweight skill descriptions already list these triggers. Offer the heavyweight review immediately rather than waiting for the slash command.

Always announce the offer before running — e.g. "This feels like sibling drift across the figure-function family; want me to run `python-review` on `figures.py`?" — and wait for user approval. The user opts in before any architectural audit begins; heavy review is invasive and produces a roadmap the user has to act on.

For the full surface comparison table, severity rubric (S1–S5), audit trail paths, and detailed descriptions of what each agent does — see **`.claude/README.md`**.

---

## Key architectural decisions

See **`design-docs/architecture/ARCHITECTURE.md`** for decisions (transport, serialization, layer/transform pipeline, composite-mark desugaring, linalg backend, randomness contract, etc.) and **`design-docs/architecture/computation-layer.md`** for the concrete data-flow diagram. Read either before touching those subsystems.
