# Followup — mark_text multi-line rendering via `<tspan>`

**Filed:** 2026-05-12
**Severity:** S2 (readability — info is conveyed, just denser than designed)

## What's broken

The Rust SVG renderer emits `<text>...with embedded \n...</text>` for any
`mark_text` value containing a literal newline. SVG `<text>` collapses raw
newlines to a single space, so the visual layout designed as

```
R² 0.701
RMSE 0.928
MAE 0.680
```

renders in the SVG as

```
R² 0.701 RMSE 0.928 MAE 0.680
```

This affects every chart that uses multi-line annotations:

- `residplot` 4-panel R²/RMSE/MAE corner (committed 2026-05-12 in `d8790ac`)
- `lmplot(show_metrics=True)` corner (committed 2026-05-12 in Candidate 3a)
- Future Schwabish corner-metric annotations on lift, classification, etc.

## Fix shape

In the SVG renderer's `mark_text` path, when the rendered string contains
`\n`, emit one `<tspan x="..." dy="1.2em">` per line under a single parent
`<text>`. First line keeps `dy="0"` (or omits `dy`); subsequent lines bump
by `1.2em`. Anchor (`x`, `y`) is read from the encoded position;
`align`/`dx`/`dy` are honored on the parent.

Existing `mark_text` tests that pass single-line strings stay byte-identical
because the rendering branch only fires when `'\n' in text`.

## Goldens to regen + visually inspect

- `tests/test_phase_9_e2e/goldens/residplot_lowess.svg`
- `tests/test_phase_9_e2e/goldens/lmplot_lm_ci.svg`
- any other goldens whose `<text>` content contains a literal `\n`

## Why not fixed inline

Discovered mid-flight during Candidate 3a (lmplot show_metrics). The
single-line rendering is the same as residplot ships today (committed 30
minutes prior); the lmplot Python layer is correct independent of the
renderer fix. Splitting the renderer change into its own commit keeps
Candidate 3a's diff confined to the figure-level default and the renderer
change scoped to the SVG backend.
