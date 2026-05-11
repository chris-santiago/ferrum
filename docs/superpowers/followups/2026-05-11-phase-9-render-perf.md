# Follow-up: phase_9_e2e tests render at ~100s/chart

**Surfaced:** 2026-05-11 during themes-T4 golden regen.
**Worktree:** any (issue is in `main`-merged code).
**Severity:** Maintenance smell — every theme-or-renderer change forces a
20+ minute golden regeneration sweep across `tests/test_phase_9_e2e.py` and
the heavy phase_10 KDE/contour goldens.

## Symptom

**Caveat (2026-05-11 update):** initial measurement of 100s/chart was
misleading. Re-running with `--tb=no` showed a full regen sweep of 921
tests in **47 seconds**. The 104s observed for a single test was
pytest's failure-diff renderer formatting a ~500 KB SVG mismatch, not
the chart-render cost. Symptom is real but narrower than first read.

What's actually slow:
- **Failure-path display.** When a golden mismatches, pytest assembles a
  unified diff of the entire SVG string (hundreds of KB) and chokes for
  60–120 s. The render itself takes ~1–4 s.
- **The first run after a renderer change is therefore much slower than
  steady state** — every golden mismatch costs the diff display.

If the goldens are pre-blessed and matching, the suite is fine. The
worst case is "I just changed a default and haven't regen'd yet".

Reproduce the diff-display slowness:
```bash
unset CONDA_PREFIX && uv run --no-sync pytest \
  tests/test_phase_9_e2e.py::test_displot_hist_golden -v
# 104s — almost all of it in the assertion diff renderer.
```

Reproduce the actual render cost (regen path):
```bash
unset CONDA_PREFIX && FERRUM_UPDATE_GOLDENS=1 \
  FERRUM_REGENERATE_GOLDENS=1 uv run --no-sync pytest tests/ --tb=no
# 921 tests in ~47 s.
```

## Root cause (hypothesis — needs verification)

The expensive phase_9 tests all chain through KDE or contour:
- `displot_hist`, `displot_stacked_hist`: simple histograms — these should
  NOT be slow, investigate separately.
- `jointplot_kde_hist`, `jointplot_kde_marginals`: 2D KDE.
- `clustermap_basic`, `clustermap_row_col_dendrograms`: hierarchical
  clustering + heatmap rect grid.
- `lmplot_lm_ci`, `residplot_lowess`: bootstrap CI bands (1000 iters
  default).
- `pairplot_3x3`, `pairplot_3x3_hue`: 9-panel grid each rendering its own
  KDE/scatter/hist.
- `catplot_box`, `heatmap_annot`: less obvious.

Three plausible cost centers per chart:

1. **KDE grid resolution.** `Kde2D` and `Contour` likely default to a
   high-resolution evaluation grid (256×256?). For tiny test data this is
   ~65k grid evaluations producing thousands of contour polygons. The
   contour-fill polygons then walk through SVG path emission — every vertex
   formatted to 3 decimal places via `fmt_f`. Resolution is theoretical,
   not data-driven.
2. **Bootstrap CI default iterations.** `Smooth.bootstrap_ci=true` (the
   default for `lmplot` / `residplot`) runs `n_bootstrap=1000` resamples
   per chart. For 150-row datasets the iteration count is hugely over-
   sampled relative to the precision needed for a CI band.
3. **SVG hand-formatting.** Every coordinate goes through `fmt_f` which
   uses `format!("{x:.*}", FLOAT_PRECISION)` then trims trailing zeros.
   For a 9000-polygon density contour fill this is millions of small
   allocations.

## What another session should do

**Top priority is the diff-display slowness, not the render itself.** A
500 KB SVG mismatch shouldn't take 104 seconds for pytest to format. The
fix is likely a custom `__eq__` / pytest hook that emits "byte counts
differ; first divergence at offset N" instead of a unified diff, OR
storing goldens as gzipped files and comparing hashes.

1. **Verify the diff hypothesis first.** Time a failing test with
   `--tb=line` vs `-v`; if `--tb=line` is 1–5 s and `-v` is 100+ s, the
   diff renderer is the cost and the chart render itself is fine.
2. **Quick win, if KDE resolution is the culprit:** Drop default grid from
   N×N to ceil(sqrt(n))×ceil(sqrt(n)) or some n-aware heuristic for any
   `Kde2D`/`Contour` invocation in tests; expose a public knob if not
   already there.
3. **Quick win, if bootstrap iters dominate:** Per-fixture override of
   `n_bootstrap` to 100 (or `seed=0` + deterministic n=200). Visual
   fidelity of CI bands at n=200 is indistinguishable from n=1000 on tiny
   data; the goldens just need byte-stability.
4. **Long-term, if SVG formatting dominates:** Pre-allocate the buffer
   (`String::with_capacity` already in `SvgBuffer::new` at 8192 bytes —
   bump for KDE-heavy charts) or replace the `format!` per-coord with a
   manual digit writer. This is touchy because golden goldens are
   byte-exact; investigate only after profiling confirms.
5. **Either way, add a `pytest.mark.slow` tier.** Tag the >10s tests, add
   a `--runslow` flag, exclude from the default `pytest` sweep used in
   development. CI runs `pytest --runslow`. This decouples iteration speed
   from coverage.

## Touch points

- `crates/ferrum-core/src/transform/kde2d.rs` (grid resolution default)
- `crates/ferrum-core/src/transform/contour.rs` (polygon emission)
- `crates/ferrum-core/src/transform/smooth.rs` (bootstrap iter default)
- `crates/ferrum-core/src/render/svg.rs::fmt_f` (per-coord formatter)
- `tests/test_phase_9_e2e.py` (mark slow)
- `tests/diagnostics/test_goldens_phase_10.py` (mark slow on KDE/contour
  rows: pdp_chart_three_features, decision_boundary_binary_logistic,
  parallel_coordinates_multiclass, shap_chart_beeswarm — all 30s+)

## Why it's not in scope for T4

T4 is visual defaults + per-scale padding plumbing. Touching KDE/contour
internals is unrelated and would balloon the worktree. Documenting and
moving on is the right call.

## Suggested session prompt

> Profile and reduce the per-chart cost in `tests/test_phase_9_e2e.py`
> and the KDE-heavy rows of `tests/diagnostics/test_goldens_phase_10.py`.
> Each chart takes 60–120s; full regen takes ~30 min. Start with
> `cargo flamegraph` on `test_displot_hist_golden` to identify the
> dominant cost center (KDE grid resolution, bootstrap iterations, or
> SVG formatting). Apply the smallest fix that brings the mean per-chart
> cost under 5 seconds. Add a `pytest.mark.slow` tier for anything that
> remains over 10s. See
> `docs/superpowers/followups/2026-05-11-phase-9-render-perf.md` for the
> handoff context.
