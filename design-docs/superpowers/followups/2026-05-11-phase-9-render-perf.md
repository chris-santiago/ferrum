# Follow-up: pytest SVG-diff renderer is slow on failing golden tests

**Surfaced:** 2026-05-11 during themes-T4 golden regen.
**Verified:** 2026-05-11 (post feat/themes merge to main).
**Resolved:** 2026-05-11 in commit `c8f0894` —
`tests._snapshots.assert_svg_eq` helper replaces the two `assert svg ==
expected` call sites. Mid-string-mutation failure path measured at
**4.35 s** (was ~100 s). Steady-state unchanged.
**Severity:** ~~Maintenance smell on the failure path.~~ Closed.

## TL;DR

There is **no** ferrum render-perf problem. Charts render fast (47 s for
all 921 phase-9 / phase-10 / theme goldens combined; ~50 ms / chart
average, ~1-4 s on the heaviest KDE charts). The earlier 104 s / chart
observation was pytest's failure-diff machinery formatting ~500 KB
SVG-mismatch strings, not chart rendering.

**Practical impact:** when a renderer change invalidates the goldens,
the first `pytest -v` or `pytest --tb=short` sweep hangs for many
minutes per failing test while pytest builds the diff display.
`pytest --tb=no` and the regen path (`FERRUM_UPDATE_GOLDENS=1` /
`FERRUM_REGENERATE_GOLDENS=1`) bypass that machinery entirely and
finish in seconds.

## Symptom

Reproduce the diff-display slowness:
```bash
# An intentionally-failing test (defaults differ from committed golden):
unset CONDA_PREFIX && uv run --no-sync pytest \
  tests/test_phase_9_e2e.py::test_displot_hist_golden -v
# 100+ s — almost all of it in the assertion-diff renderer.
```

Reproduce the actual render cost (no failure → no diff):
```bash
unset CONDA_PREFIX && FERRUM_UPDATE_GOLDENS=1 \
  FERRUM_REGENERATE_GOLDENS=1 uv run --no-sync pytest tests/ --tb=no
# 921 tests in ~47 s; 980 in ~10 s on the post-merge run.
```

Same suite with `--tb=no` and no regen (so failing tests still fail,
just without the diff display):
```bash
unset CONDA_PREFIX && uv run --no-sync pytest tests/ -q --no-header --tb=no
# Failing tests report F/. in 4-7 s each; whole suite completes in
# the same ~10 s order as the green case.
```

## Root cause

Each `_check_golden` style helper does `assert svg == expected` on two
strings that are typically 200 KB - 1 MB. When the byte-equality fails,
pytest's pytest-pluggable rewrite + `pytest._diff_pp` machinery walks
both strings character-by-character to build a colorised unified diff
for the terminal. For SVG strings dominated by long contiguous runs of
floating-point coordinates this is `O(n²)` work in the diff algorithm
and `O(n)` allocations in the formatter — multiple minutes per failing
test on a modern Mac.

**The chart rendering itself is not the cost.** Profiling the regen
path (no assertion failures, so no diff display) shows the heaviest
charts (`jointplot_kde_marginals`, `pairplot_3x3_hue`) at 1-4 s each;
the median chart is well under 100 ms.

## What another session should do

1. **Add a custom assertion helper** that produces a small failure
   message for SVG golden mismatches: byte count delta, first
   divergence offset, ~80 chars of context on each side. Use it in
   `tests/test_phase_9_e2e.py::_check_or_update` and
   `tests/diagnostics/test_goldens_phase_10.py::_check_golden`. This
   makes failure-path runs as fast as success-path runs.

   Sketch:
   ```python
   def _assert_svg_eq(actual: str, expected: str, *, name: str) -> None:
       if actual == expected:
           return
       # Find first divergence cheaply.
       n = min(len(actual), len(expected))
       i = next((k for k in range(n) if actual[k] != expected[k]), n)
       ctx = 80
       a_ctx = actual[max(0, i - ctx) : i + ctx]
       b_ctx = expected[max(0, i - ctx) : i + ctx]
       raise AssertionError(
           f"golden mismatch for {name!r}: got {len(actual)} bytes, "
           f"expected {len(expected)} bytes; first divergence at offset {i}.\n"
           f"actual:   ...{a_ctx!r}...\n"
           f"expected: ...{b_ctx!r}...\n"
           f"Set FERRUM_UPDATE_GOLDENS=1 (or FERRUM_REGENERATE_GOLDENS=1 for "
           f"the phase_10 file) to refresh after intentional changes."
       )
   ```

2. **Bonus, only if (1) doesn't fully fix it:** add a `pytest.mark.slow`
   tier and tag the >5 s charts. The bottleneck is the diff display so
   (1) alone should suffice; only fall through to slow-tiering if some
   intrinsic chart-render cost remains after the diff display is fixed.

## What another session should NOT do

The earlier framing of this issue as "phase_9 / phase_10 charts are too
slow to render" was wrong. **Do not** spend time:

- Lowering KDE grid resolution in `crates/ferrum-core/src/transform/kde2d.rs`
- Reducing bootstrap CI iterations in `crates/ferrum-core/src/transform/smooth.rs`
- Optimising `fmt_f` in `crates/ferrum-core/src/render/svg.rs`

unless you have a profiler trace showing one of them dominates on
the **regen path** (the path that doesn't trigger pytest's diff
display). The regen-path numbers above suggest none of them are
binding.

## Touch points

- `tests/_snapshots.py` — likely where the helper lands.
- `tests/test_phase_9_e2e.py::_check_or_update` — replace `assert ==`
  call site.
- `tests/diagnostics/test_goldens_phase_10.py::_check_golden` — same.

## Suggested session prompt

> The pytest assertion-diff renderer takes 60-120s per failing SVG
> golden test because the two strings are 200 KB - 1 MB and pytest
> walks them to build a unified diff. The chart renders themselves are
> fast (~50 ms - 4 s; full 921-test suite in 47 s when not failing).
> Replace the `assert svg == expected` call sites in
> `tests/test_phase_9_e2e.py::_check_or_update` and
> `tests/diagnostics/test_goldens_phase_10.py::_check_golden` with a
> custom helper that reports byte-count delta + first-divergence offset
> + ~80 chars of context on each side, skipping pytest's diff display
> entirely. Verify by intentionally breaking one phase_10 golden and
> timing the failing test before and after. Target: failure-path test
> latency under 5 s. See
> `docs/superpowers/followups/2026-05-11-phase-9-render-perf.md` for
> handoff context.
