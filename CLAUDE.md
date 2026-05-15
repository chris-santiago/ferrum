# Ferrum — Project Instructions

Ferrum is a Rust-backed Python statistical visualization library. The Python layer is the declaration API; the Rust layer (`crates/ferrum-core`, compiled to `ferrum._core`) is the computation engine. Data moves between them once, over Arrow IPC.

---

## Start every session here

1. **Read `docs/superpowers/ferrum-phases.md`** — it lists all 12 implementation phases, their dependency order, done criteria, and the current status of each. Find the first phase that is not `done` and start there.

2. **Read the phase's spec doc** (linked in the phases table) before writing any code. If no spec exists yet, run `superpowers:brainstorming` before touching anything.

3. **Read `ferrum-spec.md`** (repo root) only if you need the user-facing API contract for the phase you are working on. It is the concept spec, not the implementation guide.

---

## Build commands

| Action | Command |
|---|---|
| Install / rebuild Rust extension | `unset CONDA_PREFIX && uv run --no-sync maturin develop` |
| Release build | `unset CONDA_PREFIX && uv run --no-sync maturin develop --release` |
| Run tests | `uv run pytest` |
| Rust-side tests | `DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test` |
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
> resolve `@rpath/libpython3.10.dylib` at runtime without `DYLD_LIBRARY_PATH` pointing to
> the Python lib directory. The command above uses `sysconfig` to find the path dynamically.
> This is a macOS SIP + uv RPATH constraint; it does not affect `maturin develop` or pytest.

`pip install -e .` will **not** compile the Rust extension. Always use `maturin develop`.

---

## Hard constraints (never violate)

- **No matplotlib.** Not as a dependency, not as a dev dependency, not as an optional extra. Ever.
- **No global mutable state.** No module-level config objects, no module-level theme rebinding. Themes are values passed to `Chart`; per-chart `.theme()` always wins. The single documented exception is `ferrum.set_default_theme()` (phase 8a+), which mutates a per-thread `contextvars.ContextVar` — scope-bounded, automatic-revert when used as a context manager, and overridden by per-chart `.theme()` at render time. Do not introduce other process-scoped mutators.
- **`ferrum-spec.md` is the API contract.** If implementation diverges, update the spec with a dated note. Never silently drift.
- **`cargo test` must pass** before any phase (2+) is marked done. Phase 1 is the only exception.
- **Goldens are not blessed until visually inspected.** SVG byte-equality is necessary but not sufficient — historically goldens were committed that rendered with missing elements, blank panels, or mis-stacked bars, and the byte-diff tests still passed because the implementation matched the *broken* golden. Whenever you add or regenerate any `tests/goldens/**/*.svg` (or `tests/test_phase_9_e2e/goldens/*.svg`), you must rasterize it to PNG via `python scripts/snapshot-goldens.py <name>` (or `python scripts/snapshot-goldens.py` for all), `Read` each resulting PNG, and confirm the chart renders correctly **before committing**. The helpers live in `tests/_snapshots.py` (`snapshot_golden()`, `rasterize_svg()`, `find_goldens()`, and `regen_and_verify(golden_path, svg)` — the preferred entry point for regen scripts: writes the SVG, rasterizes the PNG, and prints both paths to stdout in a single call so the inspection PNG cannot be silently skipped). `resvg-py` (in the dev dependency group) is the rasterizer. **Caveat:** `resvg-py` silently drops paths when an SVG contains many thousands of polygon/path elements (observed at ~9.5k paths in dense KDE-contour fills). Before concluding a chart is broken from the PNG, sanity-check the SVG itself — `grep -oE 'd="M' tests/.../foo.svg | wc -l` — and look at the x-range of the first numeric coord on each path. A chart that looks like a tiny patch in the PNG but has thousands of paths spanning the plot extent in the SVG is a *renderer-side* truncation, not a real bug.
- **Do not `git push`** unless the user explicitly asks.
- **Confirm before committing to `main`** on non-trivial work. Phase 1 commits directly to main by user decision (greenfield); subsequent phases use feature branches unless the user says otherwise.

---

## Implementation philosophy (Phase 9 and beyond)

**Do the work now. Do it the right way. Enable a better end-user experience now.**

- Do NOT propose "defer X to a later phase / follow-up ticket" as a scope-reduction strategy.
- If a `ferrum-spec.md` parameter is hard to ship completely, ship it completely (with whatever Rust transform, mark, encoding, or position-adjustment subsystem it needs) — not a warn-fallback or `NotImplementedError`.
- Use sub-phase decomposition (e.g. 9a / 9b / 9c / 9d) to manage build order, not to drop scope.
- "Implement everything fully" is the default. No Warn-fallbacks. **NotImplementedErrors ARE NOT ACCEPTABLE.**
- Review `PHASE_9_PLUS_MARKS` in `src/ferrum/marks/deferred.py` at each new phase — pull marks into scope if they're needed by `ferrum-spec.md` §3.14 figure-level functions or any other in-phase contract.

This rule governs Phase 9 forward; it does not retroactively reopen closed phases.

---

## Where things live

| Artifact | Path |
|---|---|
| Implementation phases & roadmap | `docs/superpowers/ferrum-phases.md` |
| Concept + API specification | `ferrum-spec.md` |
| Per-phase design specs | `docs/superpowers/specs/YYYY-MM-DD-<slug>-design.md` |
| Per-phase implementation plans | `docs/superpowers/plans/YYYY-MM-DD-<slug>-plan.md` |
| Python package source | `src/ferrum/` |
| Rust extension crate | `crates/ferrum-core/` |
| Python tests | `tests/` |
| Golden SVG → PNG snapshot helper | `scripts/snapshot-goldens.py`, `tests/_snapshots.py` |
| Gallery audit skill | `.claude/skills/gallery-audit/` |
| Gallery audit agents | `.claude/agents/gallery-{judge,fixer}.md` |
| Heavyweight code-review skills | `.claude/skills/{python,rust}-review/` |
| Lightweight review agents | `.claude/agents/{python,rust}-review-lite.md` |
| Bug-hunt skill | `.claude/skills/bug-hunt/` |
| Bug-hunter agent | `.claude/agents/bug-hunter.md` |
| Schwabish text-integration skill | `.claude/skills/schwabish/` |
| Schwabish agents | `.claude/agents/schwabish-{judge,fixer}.md` |
| Automations index | `.claude/README.md` |

---

## Gallery audit (default-output comparison)

A reproducible side-by-side audit of ferrum's default plot output against canonical Python libraries (sklearn, seaborn, yellowbrick, scikit-plot). Use it to find where ferrum's defaults lack information or visual quality that competitors ship out of the box — missing AUC annotations, missing reference lines, wrong axis labels, missing per-cell counts on confusion matrices, etc.

- **Skill** — `.claude/skills/gallery-audit/`. Trigger with `/gallery-audit` or "audit our plots / compare ferrum to seaborn-sklearn-yellowbrick / what's missing from our default plots". 38 rows, all wired (see `RESUME.md` for the full table). Generation is a PEP 723 script (`audit.py generate`); judging runs as `gallery-judge` subagents in-session (no `ANTHROPIC_API_KEY` needed); report is a script (`audit.py report`).
- **Agent `gallery-judge`** — judges one row by reading panel PNGs and applying `rubric.md`. Dispatched in parallel, one per row, to keep parent context clean. Writes `verdict.md` with YAML frontmatter + prose.
- **Agent `gallery-fixer`** — works through `REPORT.md`'s prioritized punchlist autonomously after an audit run, closing default-behavior gaps (Python composite-mark expansion preferred over Rust changes — see "Composite marks desugar Python-side" below).
- **Output** — `gallery/` symlink at repo root → `.claude/skills/gallery-audit/output/`. Contains `REPORT.md`, per-row PNGs, per-row `verdict.md`. Gitignored.
- **Comparator isolation** — sklearn, seaborn, yellowbrick, scikit-plot run in isolated PEP 723 envs via `uv run --no-project --script`. **Never add any of them to `pyproject.toml`** — they exist solely as audit comparators. Matplotlib stays out of ferrum's deps per the hard constraint above.
- **When new ferrum APIs land** that unblock previously-BLOCKED rows, kick off a session with `"Wire row <N> — ferrum.<func> just landed"`. Claude reads `RESUME.md`, follows the Resume protocol there (copy `plots/01_roc/<library>_panel.py` as a template, swap calls, update the row's `config.toml`, regenerate). The skill auto-detects unwired READY rows on invocation and offers to wire them before running.

---

## Code-quality guardrails

### Before writing code — read the review principles

Before any session or subagent writes or modifies Python or Rust in this repo, it must first read the corresponding review skill so new code aligns with the same idioms, severity rubric, and architectural expectations the review surfaces will later enforce:

- **Writing Python?** Read `.claude/skills/python-review/SKILL.md` (and the relevant files under `.claude/skills/python-review/references/`) before editing any `*.py`.
- **Writing Rust?** Read `.claude/skills/rust-review/SKILL.md` (and the relevant files under `.claude/skills/rust-review/references/`) before editing any `*.rs`.

This is a read-only orientation step — do not invoke the full multi-phase review skill, just internalize the principles. The goal is to write code that would pass a review on the first pass, not to discover violations after the fact. The same rule applies to subagents dispatched to implement code: brief them with the relevant principles in their prompt, or instruct them to read the skill before they write.

### Before committing code — run the lite-review gate

Before any `git commit` that touches `*.py` or `*.rs`, the orchestrator must dispatch the matching lite-review agent on the staged diff and act on its verdict. This is the same protocol the `gallery-fixer` flow already uses; the rule generalizes it to all commits, not just post-fixer ones.

- **Committing Python changes?** Stage the diff, dispatch `python-review-lite`, wait for the verdict.
- **Committing Rust changes?** Stage the diff, dispatch `rust-review-lite`, wait for the verdict.
- **Both languages touched?** Dispatch both lite agents in parallel in a single tool-call block.

Act on the returned signal exactly as the gallery-fixer flow already does:

- **clean** → proceed with the commit.
- **block** → un-stage the offending files, address the verdict's findings, re-stage, re-dispatch. Three consecutive blocks on the same area escalates to the heavyweight skill (see next subsection).
- **escalate** → surface the verdict to the user and halt; do not commit.

The single sanctioned exception is documentation-only changes (`*.md`, `docs/**`, or comments-only diffs in source files) — those can commit without dispatch. When in doubt, dispatch.

### When to escalate to heavyweight review

The lite agents gate every commit; the heavyweight skills (`python-review` / `rust-review`) are subsystem-level audits that produce a six-section report and a refactor roadmap. They are deliberate interventions — not appropriate for every diff. The orchestrator should **offer the heavyweight skill proactively, without waiting for the user to type `/python-review` or `/rust-review`**, when any of these conditions hold:

1. **A phase is about to be marked done.** Before transitioning a phase in `docs/superpowers/ferrum-phases.md` from in-progress to done, run the matching heavyweight skill on the subsystems that phase modified. The lite agent sees one commit at a time; the heavy skill sees how the subsystem hangs together after a phase's worth of accumulated work.
2. **The lite agent has escalated 2+ times on the same module within the session.** Repeated escalation means the band-aid pattern is broken — lite is treating diff-level symptoms while the structural issue lives in the module. Escalate to a heavy review scoped to that module.
3. **A new public-API surface is added to an existing family.** A new `*_chart` in `src/ferrum/figures.py`, a new mark in `src/ferrum/marks/`, a new transform in `crates/ferrum-core/src/transform/`, a new encoding channel, or a new visualizer subclass all warrant a heavy review of the whole family afterwards to catch sibling drift (signature mismatches, naming drift, inconsistent return shapes, parallel-API mismatch) before the new member becomes entrenched. The `calibration_chart` signature drift caught in May 2026 is the canonical example: it lived for a phase before anyone audited the family.
4. **The user expresses a coherence smell in natural language.** "This feels off", "sibling drift", "X drifted from Y", "utility module sprawl", "inconsistent return shapes", "parallel-API mismatch", "this module feels overgrown" — the heavyweight skill descriptions already list these triggers. Offer the heavyweight review immediately rather than waiting for the slash command.

Always announce the offer before running — e.g. "This feels like sibling drift across the figure-function family; want me to run `python-review` on `figures.py`?" — and wait for user approval. The user opts in before any architectural audit begins; heavy review is invasive and produces a roadmap the user has to act on.

### Review surfaces

Ferrum has four code-review surfaces: two heavyweight interactive skills for human-invoked audits, and two lightweight autonomous agents for regression-gating fixes the orchestrator is about to commit. Pick the right one for the job.

| Surface | Type | Invoked by | Scope | Writes code? |
|---|---|---|---|---|
| `python-review` | skill | human (`/python-review`) | whole package or named subsystem | yes, with approval |
| `rust-review` | skill | human (`/rust-review`) | whole crate or named subsystem | yes, with approval |
| `python-review-lite` | agent | orchestrator (before any `*.py` commit) | only staged `*.py` diff | **never** |
| `rust-review-lite` | agent | orchestrator (before any `*.rs` commit) | only staged `*.rs` diff | **never** |

### Heavyweight skills (`.claude/skills/{python,rust}-review/`)

Multi-phase reviews (orient → diagnose → propose → execute → review) that produce a six-section report with architecture map, drift findings tagged S1–S5, refactor roadmap, and a proposed first patch. Use them when you want a senior pass over a subsystem ("review this package", "this module feels off", "audit our Rust API"). They are interactive and propose before they edit.

### Lightweight agents (`.claude/agents/{python,rust}-review-lite.md`)

Autonomous read-only quality gates that run **before any commit that touches Python or Rust source** — the gallery-fixer flow is one caller, but the gate now applies to every commit (see "Before committing code" above). They read only `git diff --cached`, apply a trimmed diff-level idiom checklist, run `ruff` / `cargo clippy -D warnings` on the affected files, and return one of three signals:

- **clean** — no S3+ findings, linters pass → orchestrator commits
- **block** — ≥1 S3 finding OR linter failed → orchestrator un-stages, the orchestrator (or the editing subagent, e.g. `gallery-fixer`) addresses the verdict, re-stages, re-dispatches
- **escalate** — ≥1 S4+ finding, OR 3 consecutive block cycles on the same area → orchestrator surfaces to user and halts; consider escalating to the heavyweight skill (see "When to escalate to heavyweight review" above)

Both lite agents are **read-only by design** — their `tools:` frontmatter restricts them to `Read`, `Grep`, `Glob`, `Bash`. They cannot modify code; only the orchestrator (or an editing subagent like `gallery-fixer`) does. The lite agents never speculate refactors beyond a single-sentence "suggested fix" per finding.

The orchestrator (parent Claude session) is responsible for: staging the changes, dispatching both lite agents in parallel when both languages were touched, tracking the cycle count across loops, and acting on the returned status.

### Audit trail

Lite-agent verdicts land at `.claude/skills/gallery-audit/output/_review_lite/<ISO-timestamp>_{python,rust}.md` regardless of trigger — the path is historical (lite started life as a post-`gallery-fixer` gate) but the directory now serves as the canonical verdict log for *all* lite-agent runs, including the commit gate above. Each verdict carries YAML frontmatter (`status`, `cycle`, `n_findings` by severity, `linters` state, `files_reviewed`) followed by per-finding prose. The directory is gitignored alongside the rest of `output/`; the verdicts exist for the orchestrator and for the human reviewing a multi-cycle session, not as permanent artifacts.

### Severity rubric (shared across all four surfaces)

| Tag | Meaning |
|---|---|
| S1 | cosmetic inconsistency; low risk, low impact |
| S2 | readability / maintainability issue; moderate leverage |
| S3 | structural cohesion issue; high leverage — **blocks lite agents** |
| S4 | risky design flaw or bug-prone seam — **escalates lite agents** |
| S5 | critical correctness or API hazard — **escalates lite agents** |

The lite agents apply the same rubric as the heavyweight skills so reading both outputs is calibrated to the same scale.

---

## Key architectural decisions

- **Build backend:** `maturin >= 1.7`
- **Workspace:** Cargo workspace at repo root; `ferrum-core` is the computation engine; `ferrum-wasm` is the WASM renderer (Phase 11, active); `ferrum-shared` is reserved for future phases
- **ABI:** `abi3-py310` — one wheel per platform-arch, works for Python ≥ 3.10
- **`extension-module` feature:** feature-gated (not unconditional) so `cargo test` can link libpython
- **PyO3 version:** pinned in `[workspace.dependencies]`; re-verify against crates.io at the start of any session that adds a new PyO3 API
- **Data transport:** Arrow C Data Interface via `pyo3-arrow` crate (phase 2+); NOT Arrow IPC bytes. Polars DataFrames implement `__arrow_c_stream__` natively — CDI hands off the buffer pointer directly with zero copies. Before phase 2, no DataFrames cross the boundary. (Decision made 2026-05-09: spec originally said "Arrow IPC" but CDI was chosen for zero-copy polars support.)
- **Chart spec serialization:** JSON via `serde` + `serde_json` (phase 3+); NOT Arrow schema metadata, NOT a binary codec. The `ChartSpec` IR is a tree-structured config (mark, encodings, scales, transforms, layers) — JSON matches the public `chart.to_json()` API in `ferrum-spec.md §3.1` and `§3.16`, evolves cleanly across phases as new optional fields are added, and stays human-readable for debugging and test fixtures. Spec size is small (KB), so binary-codec performance gains are irrelevant. Vega-Lite interop (phase 7+ `engine="vega-lite"` output) stays open without translation. (Decision made 2026-05-09: phases doc said "JSON or Arrow schema" — Arrow schema rejected because it describes columns, not config trees, and would require flattening to dotted keys or embedding JSON inside metadata.)
- **DataFrame compatibility layer:** `narwhals` (~1.x) added (phase 8a+) for non-polars DataFrame inputs (pandas, modin, cuDF, dask, ibis). Direct CDI path preserved for `polars.DataFrame` and pyarrow `Table`/`RecordBatch`; everything else flows through `narwhals.from_native(data, eager_only=True).to_arrow()`. Dict-of-arrays, list-of-records, and 2D numpy handled by direct `pyarrow.Table.from_*` branches in `src/ferrum/_coerce.py`. (Decision made 2026-05-10: alternative was ~250 LOC of in-house pandas dtype normalization; narwhals owns those bugs, ships modin/cuDF/dask/ibis support for free, and is the same compatibility layer altair adopted in 2024 for an identical problem.)
- **Multi-layer `ChartSpec`:** `layers: Option<Vec<Layer>>` additive field on `ChartSpec` (phase 8a+). When `layers.is_none()`, the renderer uses single-layer `mark` + `encoding` and the JSON shape is byte-identical to phases 3–7 — existing goldens stay valid. When `layers.is_some()`, the renderer iterates layers within each panel, sharing x/y/color scales by default. **One `ChartSpec` = one `RecordBatch`** is load-bearing: mixed-data layered charts (`Chart(df1) + Chart(df2)`) route through the SVG compositor instead of growing multi-batch logic in the renderer. (Decision made 2026-05-10.)
- **Themes are values; one documented contextvar exception.** `Theme` is an immutable Python value class (phase 8a+). `Chart.theme(t)` per-chart override always wins. `ferrum.set_default_theme(t)` returns a context manager backed by a per-thread `contextvars.ContextVar` for notebook ergonomics — the only sanctioned process-scoped theme state. See the **Hard constraints** section above for the full statement. (Decision made 2026-05-10.)
- **Composite marks desugar Python-side; no Rust `Composite` Mark variant.** Composite marks (`mark_boxplot`, `mark_errorbar`, `mark_errorband`, `mark_ribbon`, `mark_violin`, `mark_boxen`, …) build multi-layer `ChartSpec`s in Python via `chart.layer(...)` over primitive marks (rect, rule, point, line, ribbon). Rust has no awareness that a layered spec came from a composite — it just renders layers. Multi-output transforms (`BoxStats`, `Outliers`, `Violin`, `Hex`, …) feed individual layers via `Layer.data_source: Option<String>` matched against `TransformSpec.name: Option<String>`; when both are `None`, behavior is byte-identical to single-layer 8a. (Decision made 2026-05-10: alternative was a Rust `MarkSpec::Composite { layers: Vec<Mark> }` variant — rejected because it duplicates the multi-layer machinery already in `ChartSpec.layers`, and would force every composite-mark expansion to cross the PyO3 boundary as opaque payloads. Same pattern applies to Phase 9's `mark_boxen` and any future composite.)
- **Pure-Rust linear algebra via `faer`.** `faer` (0.24+) provides Cholesky, SVD, and eigendecomposition with zero external dependencies — no LAPACK, no OpenBLAS, no platform-specific linking. Used by `transform/linalg.rs` (hat-matrix diagonal, correlation matrices, shared `invert_2x2` / `solve_3x3_spd`) and `transform/stats.rs` (PCA via thin SVD, classical MDS via eigendecomposition). Chosen over `ndarray-linalg` to avoid the LAPACK build-system cost; polars uses the same approach. (Decision made 2026-05-13.)
- **Diagnostic statistics run in Rust, not Python.** `src/ferrum/_diagnostics/stats.py` was eliminated; all statistical functions (hat matrix, studentized residuals, Cook's distance, Pearson/Spearman correlation, Shapiro-Wilk, rankdata, rank1d/rank2d, PCA, classical MDS, silhouette, Calinski-Harabasz) live in `transform/stats.rs` and accept/return Arrow via `pyo3-arrow`. Python callers pass polars DataFrames through Arrow CDI — no numpy intermediary. The only numpy usage remaining in the diagnostics subsystem is at sklearn API boundaries (`model.predict()`, `model.predict_proba()` return numpy) and for genuinely 2D operations (CV split indexing, decision-boundary mesh grids). t-SNE and UMAP run in Rust via `manifolds-rs` (0.2.4+, pure Rust, MIT) — Barnes-Hut t-SNE and UMAP with HNSW/NNDescent approximate nearest neighbors. `umap-learn` is no longer a runtime dependency; UMAP works out of the box with `pip install ferrum`. `manifolds-rs` depends on faer 0.23; a `faer-compat` renamed dependency bridges the version gap at the interop boundary (remove when manifolds-rs supports faer >=0.24). (Decision made 2026-05-13.)
- **Byte-deterministic randomness via seeded `rand_chacha`.** Every transform that uses randomness — bootstrap CI (`Smooth`, `Aggregate`), beeswarm tiebreak, Phase 9 `Jitter` — seeds `ChaCha8Rng` from a `u64` (transform's `seed` field or `spec.seed`, default `0`). Never `rand::thread_rng()`, never `SystemRandom`, never platform RNG. This makes SVG goldens byte-identical across macOS / Linux / CI and across Rust toolchain versions. The same rule applies to any future transform or mark that introduces randomness — pick a seed field, document the default, plumb it through `ChaCha8Rng`. (Decision made 2026-05-10: existing 8b transforms ship this way; codified here so Phase 9+ doesn't accidentally reintroduce non-determinism.)


## Docs site work in progress

  - **Worktree**: `../ferrum-worktree-docs-continue/` (sibling of repo root — `git worktree list` to confirm)
  - **Branch**: `docs/continue` (based on `main`)
  - **Spec**: `design-docs/DOCS_SITE_PLAN.md` (in worktree, not main branch)
  - **Zensical config**: `zensical.toml` (in worktree root)

  **Status (paused 2026-05-11):**
  - Phase 1 scaffold landed: Zensical at repo root (`zensical.toml`), docs source at `docs/site/`, `docs_dir = "docs/site"` set so the legacy
  `docs/superpowers/` tree stays out of scope. mkdocstrings + mkdocstrings-python wired with NumPy docstring style.
  - 6 source-independent pages authored: Home, Get Started/{Install, Why Ferrum}, Concepts/{One chart model, Stats in the pipeline, Performance & scale}.
  - Remaining stubs are blocked on either (a) source-backed code examples — First plot, the six Guide pages — or (b) unmerged Phase 10 surface (Model
  diagnostics, Model outputs as data, Interactive rendering, two gallery examples, the yellowbrick + scikit-plot comparisons).
  
  **Resume path after Phase 10 merges to main:**
  1. From the docs worktree: `git fetch && git rebase origin/main`.
  2. Expect conflicts in `pyproject.toml` (dev deps list — usually auto-mergeable), `uv.lock` (resolve by `git checkout --theirs uv.lock && uv sync`), and
  `.gitignore` (additive). No other overlap.
  3. Run `uv run zensical build --clean` to verify (~5–9s) and check the new `griffe` warning count from Phase 10 visualizers.
  4. Author the unblocked pages against the now-real surface.
  
  **Do not** delete the worktree or branch — both docs commits live in `.git/` and survive worktree removal, but the convention is to leave both in place
  until the work merges.
