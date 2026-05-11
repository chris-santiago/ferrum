# Phase 9 e2e goldens — visual bugs to investigate

You're a fresh session on the **ferrum** repo. Branch `feat/phase-10` has Phase 10a + 10b complete and 10c in progress. Several Phase 9 e2e goldens at `tests/test_phase_9_e2e/goldens/` look visually wrong despite passing byte-equality tests (the tests only check `svg == golden`, not "does it look right"). Your job: find root causes and fix.

## Goldens flagged as broken

Look at each in a browser / image viewer:

| File | Reported symptom |
| --- | --- |
| `catplot_box.svg` | Entire plot is blank. |
| `clustermap_basic.svg` | Heatmap cells are blank; values unlabeled; some dendrogram lines look nonsensical. |
| `displot_hist.svg`, `displot_stacked_hist.svg` | Both blank. |
| `pairplot_3x3.svg`, `pairplot_3x3_hue.svg` | Diagonals (the per-variable density/hist panels) are blank. |
| `jointplot_kde_hist.svg`, `jointplot_kde_marginals.svg` | Some sub-plots are blank. |
| (Phase 10c class_prediction_error chart — no golden yet) | `mark_bar` + `Stack(offset='zero')` only emits 1 segment per bar instead of one per (actual, predicted) cell. `Stack(offset='normalize')` emits 0 rects. Verified on a 3x3 confusion matrix: only 5 of 9 nonzero cells render. Phase 10c Task 19's golden is held back until this is fixed. |

The goldens at `tests/test_phase_9_e2e/goldens/heatmap_annot.svg`, `tests/diagnostics/test_goldens_phase_10/...` were recently fixed and look fine — use those as reference for what "correct" rendering looks like in this codebase.

## Recent context — read these first

Two commits in the last session fixed *some* Phase 7 rendering stubs while building Phase 10c:

1. `c9e2531` — **feat(phase-10c-pre): finish rect+text Rust stubs.**
   Fixed `mark_rect` continuous color (Float64 column was silently failing
   `col_as_str` ⇒ all cells solid orange). Added a `text` encoding channel
   to `Encoding` struct + `mark_text` drawer + `_build_layers_list`
   whitelist. Phase 9 heatmap/clustermap goldens were regenerated.

2. `6ec1d8e` — **feat(phase-10c): mark_class_prediction_error + text channel honors numeric+format.**
   Extended the new text channel to read Float64/Int64 columns (annotations
   on numeric values were dropping); threaded `Text(field, format='.2f')`
   format spec from Python through `_build_layers_list` into the Rust
   drawer. Re-regenerated the Phase 9 heatmap/clustermap goldens.

So `clustermap_basic.svg` and the heatmap pair were regenerated **twice**. The current state should have viridis colors and reasonable values — if cells still look blank, there's another rendering bug downstream that I haven't found.

Don't blindly regenerate goldens. **Find the root cause first.** The goldens were locked to broken output before — refreshing without understanding the bug just locks in a different broken state.

## How to investigate efficiently

Read what each chart actually emits:

```python
uv run --no-sync python -c "
import polars as pl, ferrum as fe, re
# Reproduce the chart used by the failing golden by reading
# tests/test_phase_9_e2e.py (look up the fixture and the call).
# Then:
chart = fe.<the_call>(...)
svg = chart.show_svg()
# Inspect rects, texts, lines
print('rects:', svg.count('<rect '))
print('texts:', svg.count('<text '))
print('lines:', svg.count('<line '))
# Spec dump shows what's being sent to Rust:
import json
print(json.dumps(json.loads(chart.to_spec().to_json()), indent=2))
"
```

Compare what's in the spec to what should render. Then drop into the
relevant `crates/ferrum-core/src/render/marks/<mark>.rs` drawer and trace
the path.

## Likely root causes — check these first

These are guesses; verify before fixing.

1. **`mark_box` is a Phase 7 stub.** `catplot_box.svg` likely renders as blank because the box drawer is incomplete. Check `crates/ferrum-core/src/render/marks/box.rs` or whichever mark catplot uses. The composite `mark_boxplot` (`src/ferrum/marks/composite.py:15`) desugars to rect/rule/tick layers — verify those all draw correctly. Phase 7 `mark_tick` and `mark_rule` should be functional, but worth double-checking.

2. **`displot` uses `mark_density` (KDE) or `mark_histogram`.** Both go through transforms (`Kde`, `Bin`) that emit named outputs. Check whether those outputs are reaching the renderer. The histogram path produces `mark_bar` which should work. The KDE path produces `mark_area` — verify the area drawer handles the post-`Kde` `x` / `y` columns correctly.

3. **`pairplot` diagonals use `mark_density` or histograms** depending on `kind`. Same KDE/histogram concern. Plus pairplot uses `RepeatChart` which expands templates — make sure the diagonal cell's diagonal-only mark (where row_var == col_var) is being constructed and rendered.

4. **`jointplot` blank sub-plots** — composite of joint (center) + marginals (top/right). Marginals are typically `mark_density` or `mark_histogram` again. Probable shared root cause with pairplot diagonals.

5. **`clustermap_basic.svg` "cells blank"** — my last fix made the rects viridis-colored, so something downstream (recent maturin rebuild not picked up?) may be the issue. Run `unset CONDA_PREFIX PYTHONHOME DYLD_LIBRARY_PATH; PATH=$HOME/.cargo/bin:$PATH uv run --no-sync maturin develop --quiet` then re-render. If still blank, dig into `render/marks/rect.rs` for the clustermap-specific code path. The clustermap reorders rows/columns via a `Reorder` transform — verify the reordered batch reaches mark_rect with the right ordinal scale domain.

6. **"Dendrogram lines nonsensical"** — clustermap dendrograms use `mark_segment`. Check `crates/ferrum-core/src/render/marks/segment.rs` and `src/ferrum/composition.py` / `src/ferrum/figure/matrix.py` for the segment data source.

## Workflow

1. Read `tests/test_phase_9_e2e.py` and find the exact call producing each broken golden.
2. Render that exact call's chart, dump the spec, inspect the SVG.
3. If the spec looks right, the bug is in the Rust drawer. If the spec is wrong, the bug is in the Python figure builder.
4. Fix the root cause. **Don't refresh goldens until the fix is verified visually.**
5. After fix, run:

   ```bash
   PYHOME=~/.local/share/uv/python/cpython-3.10.14-macos-aarch64-none \
   PATH=$HOME/.cargo/bin:$PATH PYTHONHOME=$PYHOME \
   DYLD_LIBRARY_PATH=$PYHOME/lib \
   cargo test -p ferrum-core --quiet
   # then
   unset CONDA_PREFIX PYTHONHOME DYLD_LIBRARY_PATH
   PATH=$HOME/.cargo/bin:$PATH uv run --no-sync maturin develop --quiet
   uv run --no-sync pytest
   ```

6. Regenerate the affected goldens with `FERRUM_UPDATE_GOLDENS=1` and visually re-verify in a browser.

## Out of scope for this session

- Don't touch `feat/phase-10` task progress for Phase 10c-10h (Tasks 18-44). Branch is in-flight.
- Don't touch `tests/goldens/phase_10/` goldens — those are Phase 10 work and look correct.
- Don't refactor the rendering pipeline broadly; focused per-mark fixes only.
- Commit as you fix individual marks/figures, with `fix(phase-7-stubs): ...` or `fix(phase-9-e2e): ...` messages.

## Memory note for after the fix

When done, update the `feat/phase-10` working notes (the memory entry
`project_phase_10_in_progress.md`) with a one-line entry under "Key 10c
findings" noting which Phase 7 mark stubs were also incomplete and got
fixed — so when 10g (parallel coords, feature ranking) needs them, the
next session knows they're working.
