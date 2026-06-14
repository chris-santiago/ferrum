# Post-mortem — how documented, tested, specced features shipped as silent no-ops

- **Date:** 2026-06-14
- **Scope:** three classes of silent-drop defect found in one session — B4 (`Chart.override()`), B5 (per-channel `axis=`/`legend=`), and the residual "read-but-never-rendered" fields surfaced while fixing B5.
- **Audience:** anyone building a declaration-API-over-a-render-engine feature in ferrum.
- **Related:** `2026-06-14-override-silent-drop-rca.md` (B4, §8 provenance), `2026-06-14-per-channel-axis-legend-silent-drop-rca.md` (B5, §5-6), tracker `2026-05-15-code-archaeology.md` (B4/B5).

---

## 1. What happened

Three groups of fully documented, typed, and "tested" features did nothing at render time:

| Incident | Surface | Failure | Reach |
|---|---|---|---|
| **B4** | `Chart.override(**kwargs)` | stored kwargs in `self._overrides`, never read by the render pipeline | **every** key |
| **B5** | per-channel `encode(x=fm.X("f", axis=fm.Axis(...)))` / `legend=` | crossed to Rust as an opaque `serde_json::Map`; the renderer hand-read a subset | **most** `fm.Axis` (~22) / `fm.Legend` (~13) fields |
| **Residual** | `symbol_size`, `label_color`, `offset`, `padding`, `title_padding`, `column_padding` (legend); `offset`, `label_overlap`, `label_flush` (axis) | the value was *read into an overrides struct* but no layout/render code consumed it | ~9 fields, on **both** per-channel and chart-level paths |

Each had a spec, a plan, passing tests, and user-facing docs. None rendered. The user's question — *how did all of this go unimplemented despite specs, plans, tests, and docs?* — is the subject of this post-mortem.

## 2. The shared root cause: the two-halves trap

Every one of these features has the same shape — **two halves that must both exist:**

1. **Declaration half** — Python API → dataclass/value-class → `to_dict()`/spec → serialize across the PyO3 boundary. Easy, visible, fast to write. Produces a value that round-trips.
2. **Consumption half** — Rust layout/render reads that value out of the spec and *changes the output*. Harder, less visible, the actual feature.

In all three incidents the **declaration half shipped and the consumption half was absent or partial.** And the gap is structurally invisible:

- The value **serializes** → round-trip/`to_dict()` tests pass.
- The value is **accepted** (no `deny_unknown_fields`, or a permissive `**kwargs`/opaque map) → no error is raised.
- **Nothing renders and checks** → no test fails.

So the feature looks done from every angle except the one that matters: the rendered pixels. The declaration half is a convincing decoy.

## 3. Why each safety net failed

The user's framing is the important part: these had *specs, plans, tests, and docs*. Each was present and each failed to catch the gap, for a specific reason.

### Specs — correct, but aspirational
The B4 design (`40cbf18`) was *right*: it specified render-time application and stated, verbatim, **"No silent no-ops."** The thing that shipped is exactly what the spec forbade. **A spec describes intent; nothing verifies the implementation matches it.** Absent a gate that ties a spec claim to a passing behavioral test, a spec is a wish. The B5 spec layer is worse: `AxisSpec`/`LegendSpec` were introduced (`31ceee4`) explicitly as *"deferred kwargs… renderer ignores in 8a"* — the spec itself encoded the half-built state, and "deferred" silently became "forgotten."

### Plans — right tasks, fatal packaging
The B4 plan had the correct tasks: *"Add `.override()` — store overrides, **validate at render time**"* and *"Wire cascade resolution in `_render.py`."* Three failures compounded:
- **Build and verify were conflated into one checklist item** ("store overrides, validate at render time") — one checkbox covering two halves, so partial completion was inexpressible.
- **The checklist was never maintained** — `grep -c '[x]'` on the plan returns **0**. No box was ever ticked, so the plan gave zero completion signal; the phase was judged done by "method exists + tests green + docs present."
- **For the residual fields, the plan's premise was wrong.** "Implement all advertised fields" assumed each field had a renderer to *route into*. Some had no renderer at all — the plan couldn't surface a task that nobody knew was missing.

### Tests — the load-bearing failure: structure asserted, behavior not
This is the single most important line. Every one of these features had tests, and every test asserted the **declaration half**:
- B4: `assert c._overrides == {"x_axis_label_angle": -45}` — the dict was populated. (`grep -cE 'to_svg|to_spec|render' test_override.py` → **0**.)
- B5: `assert Axis(grid_color="#ccc").to_dict() == {"grid_color": "#ccc"}` — the dataclass serialized. (`test_phase_12_axis_legend.py` — 0 render assertions before this session.)
- Residual: same `.to_dict()` pattern; the fields serialized fine.

**A test that never calls a render method cannot detect a missing render consumer.** These tests pass identically whether the feature works or does nothing. CI stayed green on hollow coverage. The anti-pattern is seductive because the storage tests are *true* — they just test the wrong half.

### Docs — written from the design, not the code
B4's docs (`b289c67`) documented the full *designed* contract (validation, `FerrumOverrideError`, deprecation, a 6-level cascade) — none implemented. The smoking-gun tell: the doc author hit the dead feature while producing the example screenshot and worked around it, leaving `# uses typed method in practice; shown for illustration` in the doc. **The gap was observed at docs-writing time and papered over instead of filed.** There is no doc-vs-implementation liveness gate; `/audit-docs` checks staleness, not "does the documented behavior actually render."

### Reviews — see a diff, not a render
Code review sees one commit: a method, tests, docs. All three are present, so it reads as wired. A reviewer doesn't *run* the feature. (Counter-evidence worth noting: the heavyweight cohesion reviews **this session** caught real bugs — the Unit 1 show-toggle precedence inversion, the Unit 3 clip-id collision. Review *can* catch these, but only when it traces behavior to the render output, not when it confirms structure exists.)

### Merges / CI — a missing consumer is invisible
There is nothing to conflict on (the consumer never existed, so no merge "dropped" it) and nothing to fail (hollow tests are green). A missing consumer is invisible to both conflict detection and CI. Every subsequent merge carried the dead code + hollow tests + aspirational docs forward unchallenged.

## 4. The investigation layer can be fooled too — "honored ≠ rendered"

The residual gap is the subtlest and the most instructive. When auditing B5, the analysis (including this project's own RCA) classified a field as *honored* if `prepare.rs` did `.extra.get("field")` on it. But **a read is not a render.** A value can be read into a `LegendPreparedOverrides` struct and then never consumed by any layout/render code — exactly what happened to `symbol_size`, `offset`, `padding`, and friends. Grepping for reads finds the declaration half a second time, not the consumption half.

**The only ground truth is: set the field, render, and check the output changed.** Every layer above the pixels — serialization, struct population, "the value is present in the spec" — can lie.

## 5. Why it recurred (not a one-off)

`override`, per-channel `axis`/`legend`, and the residual fields were built the same way, in the same vintage, by the same workflow. The declaration-first habit is *natural*: you design the API, add the value class, write the storage test, write the docs — and the render wiring feels obvious / "will get done." "Obvious" wiring silently doesn't happen, and nothing in the pipeline notices. It recurs because the workflow optimizes for the visible half and has no gate on the invisible one.

## 6. Remediations (process + tooling)

Ordered by leverage:

1. **Behavioral tests are mandatory for any render-affecting field; structure tests are never sufficient.** A test must set the field and assert the **rendered output** (SVG attribute, parsed element, pixel) changed. Make it a reviewable rule: *a test module for a render feature that never calls `to_svg`/`to_png`/`to_spec`-then-render is a red flag.* Consider a CI heuristic that flags new `*config*`/`*encoding*`/`*axis*`/`*legend*` test files with zero render calls.
2. **Field-coverage matrix.** Enumerate every public field of the config/encoding/style surface and assert each one changes the output (a generated parametric test). This is the only mechanical defense against "a field nobody wired." It would have caught all three incidents at once. (B5 unit 5 added per-field render tests + parity tests; institutionalize the parity test — *per-channel and chart-level produce identical output* — as the standing pattern.)
3. **Fail loud by default.** Type the spec and use `deny_unknown_fields` (done for B5) and a validation registry with did-you-mean (done for B4) so the *accepted-but-dropped* case becomes *rejected at render*. A silent drop is a wrong-by-construction default; rejection turns an invisible no-op into a visible error.
4. **Doc/spec liveness gate.** A doc or spec that claims a feature renders should reference a passing render test. Extend the docs audit from "is this stale?" to "does the documented behavior actually render?" The `# uses typed method in practice` workaround should have been an automatic stop-and-file.
5. **Plan hygiene.** Separate "build" from "verify it renders" as distinct checklist items with a done-criterion of *"renders, asserted by test X"* — never *"method exists."* Maintain the checklist as the source of truth; an all-`[ ]` plan at "done" is itself a signal.
6. **Audit by rendering, not grepping.** Silent-drop audits must render-and-check. Treat `.get("field")` / struct population as *suspicious*, not *confirming*.

## 7. What this session did about it

- **B4:** implemented the consumer, validation registry, fail-loud `FerrumOverrideError`, deprecation routing; replaced the storage-only tests with render-level + cascade + error tests. Merged.
- **B5:** typed the per-channel specs (one shared `AxisStyleSpec`/`LegendStyleSpec`, `deny_unknown_fields`), routed them into the chart-level consumer at per-channel-wins, wired every orphan field with a renderer, exposed them chart-level, and **upgraded `test_phase_12_axis_legend.py` from `.to_dict()` to render-level + parity + fail-loud**. (`fix/per-channel-axis-legend`.)
- **Residual:** the render-level upgrade *surfaced* the read-but-unrendered fields (the parametric render assertions failed to find an effect) — exactly the field-coverage-matrix defense working. Their renderers are being implemented (legend geometry done; axis `offset`/`label_overlap`/`label_flush` pending).

The through-line of the fix mirrors the through-line of the failure: **the bug was hidden by structure-only tests, and it was found and closed the moment a test rendered and checked.**
