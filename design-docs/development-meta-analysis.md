# Ferrum: Development Meta-Analysis

## The design problem

Ferrum's original spec was not primarily an exercise in enumerating chart features; it was an attempt to solve a deeper design problem in the Python visualization ecosystem: users are routinely forced to switch APIs, object models, and conceptual frames as soon as their work moves from ordinary plotting to statistical graphics, interactivity, convenience plots, or model diagnostics. The spec's real contribution was to treat that fragmentation itself as the target of design, which is why it reads less like a product checklist and more like a philosophical and architectural argument for a single coherent chart system.

### Starting problem

Existing tools each solve part of the workflow well, but they do so in different worlds. Plotnine and ggplot-style systems are strong at grammar and layering, Altair is strong at typed encodings and composition, Seaborn is strong at statistical convenience, Plotly is strong at interactivity, and Yellowbrick or scikit-plot are useful for model evaluation, yet each demands different abstractions and different compromises once the user's task changes.

That observation became the seed of Ferrum's design. Instead of asking which charts were missing from one existing library, the spec asked what kind of system would let a user stay inside one conceptual model from exploration to explanation to model evaluation, and from small data to much larger data, without rewriting plots or adopting a different mental model at each stage.

### Spec as design compression

For a meta document about agentic coding, this initial spec matters because it compressed a large amount of design intent into a stable artifact before implementation work accelerated. It fixed the project's philosophical center early: every visualization from a scatter plot to a SHAP beeswarm should be expressible as a composition of the same primitives, and high-level helpers should be sugar over that grammar rather than parallel APIs with their own rules.

That compression reduced ambiguity for later implementation work. Agents and implementation passes did not need to repeatedly rediscover what counted as a chart, whether diagnostics should be special objects, whether interactivity should introduce a new authoring model, or whether statistical transforms belonged in user code or in the engine, because the original spec had already answered those questions at the level of principle.

### Core beliefs

The design becomes coherent when read through its core beliefs rather than its feature inventory. The spec explicitly commits to grammar first, convenience second; model artifacts as data; statistical transforms as part of rendering intent; interactivity as a renderer rather than a rewrite; zero unnecessary copies; and defaults chosen for correctness rather than mere aesthetics.

These beliefs are stronger than ordinary product requirements. They imply that a confusion matrix should be composable like any other heatmap, that a ROC curve should be themeable and concatenable like any other layered chart, that KDE and smoothing should be declared in the chart spec rather than precomputed in SciPy, and that calling `.interactive()` should not force the user into a second object model.

### Synthesis from prior art

The original spec is best understood as a synthesis of prior art under stricter unification rules. It inherits layering and explicit scales from plotnine and ggplot2, typed channels and selection ideas from Altair, figure-level convenience and statistical vocabulary from Seaborn, interactive expectations from Plotly, and ML diagnostic vocabulary from Yellowbrick and scikit-plot, while explicitly rejecting the backend coupling, split static-versus-interactive models, row limits, and non-composable helper surfaces that come with those systems.

This is what gives Ferrum's spec its particular shape. It is not arguing that existing libraries are useless; it is arguing that their strengths should not remain partitioned across separate systems, especially when those partitions force users to leave a chart grammar for diagnostics, leave static rendering for interactive rendering, or abandon a declarative interface once the data grows.

### Three central claims

Although the spec is broad, the center of gravity is actually narrow. The clearest synthesis of the original design is the three-part claim: one grammar that scales to production-sized data, model diagnostics as first-class grammar objects rather than a parallel API, and statistical computation in the render pipeline rather than in user preprocessing code.

Those three claims explain most of the rest. Zero-copy ingestion, headless rendering, GPU-backed interactivity, pure-Rust backends, and dataframe interoperability via Narwhals are important, but they function mainly as enabling mechanisms that make the three larger promises believable.

### Why Rust became necessary

Rust enters the spec as a consequence of design ambition, not as the starting identity of the project. Once the library promises one grammar from small to very large data, in-engine statistics, headless rendering, and minimal Python involvement after data handoff, a Rust computation core paired with Arrow-based columnar interchange becomes the natural architectural answer.

That is why the spec's architecture section feels downstream of the philosophy rather than separate from it. Python is defined as the declaration layer, Rust as the computation layer, data crosses the boundary once, and stat transforms, layout, aggregation, and rendering preparation happen after that boundary crossing; even the later shift from Arrow IPC to Arrow CDI preserves this same principle while improving the zero-copy story.

### Why the API got so broad

The breadth of the original spec can initially look excessive, but the reason is clear: Ferrum wants to be complete enough that a practitioner coming from Altair, Seaborn, or Yellowbrick does not immediately fall back to matplotlib when they need a practical chart type or diagnostic workflow. That naturally expands the surface to include primitive marks, composite marks, stat transforms, themes, compound views, figure-level functions, diagnostics, visualizers, and multiple rendering backends.

Seen this way, the large scope is not evidence of aimlessness. It is the cost of taking the unification promise seriously: if Ferrum is meant to remove library-switching as a normal part of analysis, then it must cover the neighboring terrain that currently causes those switches.

### Model diagnostics as a decisive idea

One of the most original parts of the spec is its treatment of diagnostics. In many ecosystems, model evaluation is handled by separate plotting helpers or estimator-bound visualization objects that are useful but non-composable; Ferrum instead treats model-derived artifacts as ordinary tables and diagnostic views as ordinary chart constructions, which is why ROC curves, calibration plots, SHAP charts, residual plots, and learning curves can sit naturally inside the same grammar and composition operators as any other chart.

This idea explains why the spec feels more ambitious than a typical plotting-library design. It is not merely trying to compete on chart aesthetics or syntax; it is trying to remove one of the most persistent conceptual boundaries in data science tooling, the boundary between "charts" and "model visualizations."

### Agentic coding relevance

In the context of agentic coding, the original spec served as a stabilizer. It made it possible for later coding agents to work incrementally on architecture, rendering, compatibility, and phased implementation without constantly renegotiating what the project was for, because the spec had already turned a broad frustration into a load-bearing design thesis.

That is especially important because some implementation details evolved later. The move from Arrow IPC to Arrow CDI and the incorporation of Narwhals-based dataframe compatibility show that the implementation could change while the core design remained intact: one chart system, minimal copies, broad ecosystem interoperability, and no return to fragmented APIs.

---

## The numbers

| Metric | Value |
|---|---|
| Calendar days | **9** (May 9–17, 2026) |
| Total commits | **918** |
| Python source | **33,460 lines** across 89 files |
| Rust source | **64,131 lines** across 153 files |
| Python tests | **2,278** test functions in 125 files |
| Rust tests | **1,080** `#[test]` functions |
| Design specs | **36** documents |
| Implementation plans | **38** documents |
| Phases completed | **12 of 12** |
| Peak day | **213 commits** (May 11) |

Phases 1–7 (skeleton through first rendered chart) landed on **day 1**. Phase 8a (full grammar API with 31 encoding channels) landed **day 2**. Phase 10 (26 model-diagnostic marks, 21 figure functions, 25 sklearn visualizers) landed **day 3**. Phase 11 (WASM interactive renderer with selections, zoom, pan) landed by **day 6**. Phase 12 (17 data transforms, 7 new scale types, `ferrum.color`, `ferrum.config`, `LayerChart`, `ConcatChart`, `Axis`/`Legend` value classes) landed on **day 9**.

## What made it work

### 1. Spec-first, plan-second, code-third — no exceptions

Every phase followed the same ritual: brainstorm → write design spec → write implementation plan → execute plan → mark done. The spec (`ferrum-spec.md`, 1,653 lines) was written before line one of code. Phase-level specs decompose the concept spec into buildable units. Plans decompose specs into task lists. No phase was ever started without both documents approved.

This front-loaded the hard decisions (Arrow CDI vs. IPC, JSON vs. binary serialization, no-matplotlib constraint, themes-as-values) into a single brainstorming session on day 1. Every subsequent session executed against settled architecture — zero time re-litigating.

### 2. Six-layer automation architecture

The `.claude/` directory is as engineered as the library itself — 9 agent definitions, 12 skills, a shared severity rubric (S1–S5), and explicit dispatch rules:

- **Layer 1 (coding agents):** `python-coder` and `rust-coder` embed the full review principles from their respective heavyweight review skills. Code is written to pass review on the first attempt, not iteratively corrected.
- **Layer 2 (commit gates):** `python-review-lite` and `rust-review-lite` run on every staged diff before commit. Read-only. Three consecutive blocks escalate to heavyweight review. The orchestrator (Opus) never commits without a gate pass.
- **Layer 3 (heavyweight reviews):** Full subsystem audits at phase boundaries. Catch sibling drift, API inconsistency, and structural decay that accumulate across a phase's worth of commits.
- **Layer 4 (quality campaigns):** `/bug-hunt` dispatches 11 parallel agents across subsystems. `/test-sweep` runs multi-round combinatorial TDD. `/gallery-audit` renders 38 plot types against sklearn/seaborn/yellowbrick and judges them with a rubric. `/code-archaeology` sweeps the entire codebase for unimplemented features and spec drift.
- **Layer 5 (remediation agents):** `gallery-fixer`, `schwabish-fixer`, `bug-hunter` — each reads campaign output and closes findings autonomously, delegating code changes back to the Layer 1 coding agents.
- **Layer 6 (utility skills):** `/regression-test` (auto-triggered after every bug fix), `/ferrum-docstrings`, `/docs-audit`, `/release`.

The key insight: **coding agents never commit**. The orchestrator handles staging, gate dispatch, and commit. This separation means the review pipeline is structurally unforgeable — you can't skip it.

### 3. Orchestrator + specialist model split

Opus orchestrates: it reads specs, decomposes work, dispatches agents, interprets results, handles cross-cutting decisions. Sonnet executes: it writes Python, writes Rust, runs tests. This matches the cost/capability curve — architectural judgment is expensive and rare, line-by-line coding is cheap and frequent. A single Opus context window manages the session while parallel Sonnet agents do the mechanical work.

The dispatch rule is enforced in `CLAUDE.md`: "Never use `general-purpose`, `claude`, or `Explore` agents for code that writes or modifies `.py` or `.rs` files." This prevents the orchestrator from doing coding work itself and ensures every line of code goes through an agent that has internalized the review principles.

### 4. The CLAUDE.md as institutional memory

At 245 lines, `CLAUDE.md` is the project's constitution. It encodes:
- Build commands (with known platform gotchas)
- Hard constraints ("no matplotlib, ever")
- Dispatch rules (which agent handles which files)
- Review escalation protocol (when lite → heavyweight)
- The implementation philosophy ("do the work now, do it the right way")
- Where everything lives (a lookup table, not prose)

Every session — every agent — starts by reading this file. It's the mechanism by which decisions made in session 1 are enforced in session 50. The `memory/` system supplements it with cross-session context that doesn't belong in committed code (user preferences, workflow feedback, stale-state warnings).

### 5. Quality campaigns as ratchets

The project didn't just write code and move on. After phases stabilized, systematic sweeps found what human review missed:

- `/test-sweep` wrote **132 combinatorial tests** across 5 rounds (mark×channel, facet×layer, coord×position, theme×mark, encoding×facet×theme), found and fixed **2 bugs**.
- `/bug-hunt` dispatched **11 parallel agents** to write edge-case tests per subsystem.
- `/gallery-audit` rendered **38 plot types** against 4 reference libraries, scored them against a rubric, and fed findings to `gallery-fixer`.
- `/code-archaeology` swept the **entire codebase** for silent drops, dead code, and spec drift — found 4 active bugs, 7 high-severity Rust gaps, 11 silent-drop mark kwargs, and 6 stale doc references. All fixed.

These aren't one-time runs. They're repeatable skills that can be re-invoked after any significant change. Each run either confirms quality or surfaces regressions.

### 6. Feedback loops that actually close

The `memory/` system captures operational lessons: "subagents falsely claimed file deletions" → always verify independently. "Plans with inline code blocks waste tokens" → plans describe WHAT, not HOW. "Integration tests must not mock the database" → test against real state.

These aren't suggestions — they're loaded into every session and shape agent behavior. The feedback loop is: something goes wrong → save a memory → next session reads it → the failure mode is structurally prevented.

## What this architecture produces

A Rust-backed Python visualization library with:
- 31 encoding channels, 20+ mark types, 21 figure functions, 25 sklearn visualizers
- A WASM interactive renderer with selections, zoom, pan, and linked views
- Zero matplotlib dependency (hard constraint from day 1)
- 17 data transforms, 7 new scale types, and utility modules (`ferrum.color`, `ferrum.config`) closing the spec gap
- 3,358 tests across Python and Rust
- A docs site (in-progress on a worktree branch)
- A release pipeline with conventional commits, changelog generation, and PyPI publishing

Built in 9 days by one human and an agentic Claude framework.

## The meta-lesson

The velocity didn't come from typing faster. It came from:
1. **Never starting without a spec** — eliminates rework from misunderstood requirements
2. **Enforcing review structurally** — gates on every commit, not periodic audits
3. **Separating judgment from execution** — Opus reasons, Sonnet codes
4. **Making quality campaigns repeatable** — sweeps are skills, not one-time heroics
5. **Treating agent infrastructure as product** — the `.claude/` directory has its own README, architecture diagram, and severity rubric

The 918 commits aren't 918 manual actions. They're the output of a system that was designed to produce correct code at high throughput, with the human providing direction, constraints, and architectural taste — not keystrokes.
