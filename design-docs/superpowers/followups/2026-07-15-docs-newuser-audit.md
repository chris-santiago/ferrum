# Docs Site Audit — New-User Perspective (2026-07-15)

> Eight parallel agents read `docs/site/` as different newcomers (first-plot user, mental-model builder, chart builder, power user, diagnostics/export user, library evaluator, and two API-reference consumers). Every doc claim, code snippet, and API page was cross-checked against the live source in `src/ferrum/` (and `crates/ferrum-core/` where relevant) at v0.20.1, not just read for prose. Internal links and image paths were resolved on disk.
>
> Baseline: `docs/site/DOCS_REVIEW.md` (2026-05-17). Most of its HIGH items are now fixed (sklearn prerequisite callout, dynamic `__version__`, the 24 missing visualizer pages, the `groupby` explanation, the wrong package name in `interactive.md`). This audit is the fresh set.
>
> Totals: **14 HIGH, 25 MED, 22 LOW.** Full per-slice detail lives in the scratchpad files `docs-audit/01`–`08`.

---

## Cross-cutting themes (fix once, closes many)

1. **Wrong PyPI package name `pip install ferrum`.** The dist is `ferrum-viz`. The 2026-05-17 fix to `interactive.md` was not swept: it survives in `performance-scale.md:112` and `saving-and-export.md:161`. → grep the whole site for `pip install ferrum` (word-boundary, not `ferrum-viz`).

2. **Autorefs to deprecated symbols missing from `__all__`.** `[ferrum.shap_chart]` and `[ferrum.rank_chart]` are used as cross-references but neither is in `__all__`, so `gen_api_pages.py` emits no anchor and the links are dead. `shap_chart` is a live (non-deprecated) dispatcher that should be added to `__all__`; `rank_chart` is deprecated and should be replaced by `rank1d_chart`/`rank2d_chart` in prose. (model-outputs-as-data.md:31, model-diagnostics.md:9/79/81.)

3. **Links to the retired `api/ferrum.md` redirect stub.** At least 6 links across `index.md:95`, `marks-encodings.md:708`, `data-transforms.md:565`, `composition.md:381`, `model-diagnostics.md:83/313` point at the meta-refresh bounce page instead of `api/ferrum-toc.md` (or the split `api/plots.md`/`api/visualizers.md`). → sweep `../api/ferrum.md` and `api/ferrum.md`.

4. **Comparison-page snippets are `pytest.mark.skip`, so they drifted hard.** Root cause of 4 of the HIGH findings. The Altair page teaches three whole subsystems (`fm.Scale`, `fm.Filter/Calculate/...`, `.repeat()`) that do not exist, and the plotnine page teaches string coords that raise. → un-skip the comparison snippets in CI, or at minimum re-verify each against the live API.

5. **`**kwargs` / honored params not enumerated in docstrings.** Because the API pages are auto-generated, these are docstring gaps. Worst offender is the **X/Y positional channels** (six honored kwargs undocumented). Then `Fill`/`Stroke` (scheme/scale), the composite statistical marks, `ModelSource` derived-data methods, and `Theme`'s per-level grid keys. Sibling symbols in each family already enumerate, so these are inconsistencies as well as gaps.

6. **Doc code blocks that don't match the recipe that generated their screenshot.** `annotations.md` and `secondary-axes.md` show code that differs from `recipes/customization/*.py` (and in two cases the doc code errors). → make the doc block identical to the recipe, or regenerate the image from the doc block.

7. **Hardcoded mark count is stale.** "54 marks" appears in `first-plot.md:156` and `marks-encodings.md:309`; the live count is 56. → drop the exact number or generate it.

---

## HIGH (blocks a user or documents non-existent behavior)

| # | File:line | Problem | Fix |
|---|---|---|---|
| H1 | install.md:11,16,23,28,66-69; first-plot.md:9 | Unquoted `ferrum-viz[all]` glob-fails on zsh (default macOS shell): `zsh: no matches found`. Blocks the primary install path at step one. | Quote everywhere: `pip install "ferrum-viz[all]"`, `uv add "ferrum-viz[all]"`, etc. |
| H2 | performance-scale.md:112 | `pip install ferrum` — wrong package name. | `ferrum-viz`. |
| H3 | saving-and-export.md:161 | `pip install ferrum` — wrong package name. | `ferrum-viz`. |
| H4 | saving-and-export.md:30,92 | PDF export called a "vector PDF using resvg-py"; it is actually a **raster** PNG (Rust `resvg`) wrapped by a pure-Python codec in `_pdf.py`. `resvg-py` is dev-only. Lines 30 and 92 also contradict each other. | State PDF embeds a rasterized PNG in a dependency-free PDF wrapper; drop "vector" and the `resvg-py` attribution. |
| H5 | model-outputs-as-data.md:31 | `[ferrum.shap_chart]` autoref → deprecated, non-`__all__` symbol; link is dead and steers to a deprecated API. | Use `shap_beeswarm_chart` (or the three `shap_*_chart` helpers). |
| H6 | model-diagnostics.md:9,79,81 | `[ferrum.shap_chart]`/`[ferrum.rank_chart]` dead autorefs (not in `__all__`). | Add `shap_chart` to `__all__`; replace `rank_chart` with `rank1d_chart`/`rank2d_chart`. |
| H7 | marks-encodings.md:297 | Says `Size`/`Shape` `legend` kwarg is "reserved for future use." It works — `fm.Size(..., legend=False)` demonstrably changes the SVG. Steers users off a live feature. | Document legend suppression on `Color`, `Size`, `Shape`. |
| H8 | annotations.md:22,155,163 | `annotation.arrow(curve=...)` has no such param; the intro example raises `TypeError`. | Remove `curve=` from the signature block and both examples (arrows are straight). |
| H9 | annotations.md:74-83,124,181 | `annotation.rect(z=...)` has no such param; Z-order section claims universal `z=` but only `text` supports it. "Highlight behind marks" example errors. | Scope the Z-order section to `text`, or implement `z` on shape primitives. |
| H10 | comparison/altair.md:193-216 | Entire Scales section uses `fm.Scale(...)`, which does not exist. Contradicts the plotnine page's correct `fm.LogScale()`. | Rewrite against typed scales (`fm.LogScale()`, `fm.LinearScale(domain=...)`) / `scale={"scheme":...}`. |
| H11 | comparison/altair.md:100-120 | `fm.Filter/Calculate/Fold/Window/Sample/Flatten` — none exist. | Use `.transform(fm.transform_filter("value > 0"))` and the `transform_*` functions. |
| H12 | comparison/altair.md:147-160,271 | `.repeat(column=[...])` — `Chart.repeat` does not exist. | Use `fm.RepeatChart(template, column=[...])` (as the gallery does). |
| H13 | comparison/plotnine.md:36,128 | `.coord("flip")` and string coords (`"flip"`,`"polar"`,`"theta"`,`"radial"`) raise `TypeError`; no theta/radial coords exist. | `.coord(fm.CoordFlip())`, `fm.CoordPolar(theta=...)`; drop theta/radial. |
| H14 | api/encoding.md (X, Y) | The two most-used channels honor six kwargs (`sort`, `axis`, `stack`, `impute`, `format`/`format_type`, `legend`) that the docstrings never mention; unhonored kwargs are silently warned-and-dropped, so they're undiscoverable. | Add the six params to `X`/`Y` docstrings (mirror the `Color` Notes block). |

---

## MED (real friction or inaccuracy)

- **model-diagnostics.md:117** — says `ModelSource` lazy-imports `umap`; UMAP is pure-Rust (`_core.umap_embedding`), no Python `umap`. Contradicts line 302. (Same stale phrase in `diagnostics/source.py:45`.)
- **model-diagnostics.md:81** — `rank_chart` listed as a first-class helper without a deprecation note.
- **marks-encodings.md:628** — `mark_density(multiple=...)` list omits the working `"fill"` and advertises `"dodge"`, which the source flags as not-yet-implemented.
- **themes.md:221** — "Twelve" sequential ramps; `SEQUENTIAL_SCHEMES` has 16 (omits `reds`/`greens`/`oranges`/`purples`).
- **composition.md:180** — `.share_scale()` prose promises `color`/`size`, but the autoref target `Chart.share_scale` is x/y-only and raises on those; only composition classes accept them.
- **composition.md:209** — JointChart `right=` "correct orientation" never explained; the real `orientation="horizontal"` mechanism is unmentioned.
- **configuration.md:46-74** — `AxisConfig` table presented as complete but omits ~10 real params (incl. `label_padding`, which a sibling recipe uses).
- **format-presets.md:56,95** — `$.1fM` and `.1f%%` render wrong (`$1400000.0`, `1400000.0`); ferrum silently drops trailing literals (not valid d3-format).
- **annotations.md:17-26** — intro block differs from the recipe that made its screenshot (coords/font + the `curve` error).
- **secondary-axes.md:61-89** — recipe omits the `configure_padding(right=80)` its screenshot needs; y2 axis may clip.
- **gallery/gallery.md:383** — residplot "grammar" uses `mark_smooth(inject_residuals=True)`, which raises `TypeError`.
- **comparison/scikit-plot.md:18, yellowbrick.md:23** — `elbow_chart(model, X)` omits the required keyword-only `ks=`; the two pages also disagree on the first arg name.
- **model-diagnostics.md:83,313** — links to the retired `api/ferrum.md` stub (see theme 3).
- **api/encoding.md (Fill, Stroke)** — omit `scheme`/`scale`/`title`/`legend`/`condition`; the point of a color channel is undiscoverable.
- **api/encoding.md (FillOpacity/StrokeOpacity/StrokeWidth/StrokeDash/Angle)** — `legend` honored but undocumented.
- **api/encoding.md (Color)** — `scale` example cites `ColorScale.Continuous("viridis")`, which doesn't exist (real: `ContinuousScheme`/`continuous_palette`/`Gradient`).
- **api/marks.md (mark_boxen)** — `k_depth` docstring drift: signature default is `"tukey"`, docstring says default `"proportion"` and omits `"tukey"` from the enum.
- **api/marks.md (mark_arc/mark_image/mark_geoshape)** — `**kwargs` documented as a guide-pointer only, unlike sibling primitive marks. Real set: `_VALID_MARK_KWARGS`.
- **api/marks.md (mark_boxplot/boxen/errorbar/errorband/contour/violin/swarm/qq/raster)** — `**mark_kwargs` documented as "forwarded overrides" only; density/histogram/smooth enumerate theirs.
- **api/model_sources.md** — ~23 derived-data methods document kwargs in free prose, so mkdocstrings renders no parameter table (only `compare` has a `Parameters` section).
- **api/model_sources.md** — `calibration_curve` (`n_bins`, `strategy`) and `roc_curve` (`drop_intermediate`) kwargs entirely absent from docstrings.
- **api/themes.md (Theme)** — 9 per-level grid keys (`major_grid_*`, `minor_grid_*`, `minor`) accepted but not in the Parameters block; cross-reference `fm.Grid`.
- **guide/concepts/interactivity.md** — still no "how interactive output is delivered" note (Jupyter extra vs `.save()` HTML vs in-wheel WASM); never mentions `.interactive(toolbar=...)`.
- **guide/concepts (five pages)** — pure prose, no runnable snippet grounding the central claim (only dataframe-pluralism.md has code).

---

## LOW (polish)

- Stale "54 marks" count (first-plot.md:156, marks-encodings.md:309) → 56, or de-numeric it.
- `index.md:95` "API Reference" links point at the `api/ferrum.md` redirect stub → `api/ferrum-toc.md`.
- install.md "Standard install" leads with `[all]` (pulls the heavy shap/numba stack) → lead with base install.
- why-ferrum.md:13 competitor scaling limits stated as hard facts (Altair's 5k is a configurable `max_rows` guard).
- performance-scale.md benchmark table has no ferrum version/date.
- modin/cuDF/dask/ibis asserted as first-class inputs but never demonstrated.
- first-plot.md:144 `.xlim()/.ylim()` block is the only non-self-contained snippet (copy-paste → `NameError`).
- themes.md key tables omit `cull_threshold`, `strip_padding`, `strip_text_size`.
- composition.md:316 leaks the internal "Phase 12" roadmap term into user docs.
- marks-encodings.md:189 "five scale classes" undercounts (self-corrected next line).
- configuration.md:186 `LegendConfig` table omits ~10 real fields.
- annotations.md:245-277 span "target zone" example doesn't match its rect-generated image.
- interactive.md:17 `[jupyter]` extra described as adding `ipywidgets` (it's transitive via anywidget).
- model-diagnostics.md fitted-model path never states it needs `[models]`/sklearn.
- seaborn.md:13 "`method=` replaces `order=`" — both exist on `lmplot`.
- plotnine.md:124 lists "step" as a mark; it's `mark_line(interpolate="step")`.
- plotnine.md:18 geom_smooth `method=` claims `"lowess"`/`"poly"`; wrapper only surfaces `"lm"`/`"loess"` (verify against Rust).
- api/transforms.md aggregate `fn`/pivot `op` value sets shown by example only.
- api/model_sources.md `ComparedModelSource` proxied methods don't render (dynamic `__getattr__`; deliberate).
- api/visualizers.md `SHAPBarVisualizer` defers params by-reference → thin rendered page.
- api/ferrum-toc.md:17 references a non-existent `annotate_line` (real: `annotate_hline`/`annotate_vline`).

---

## What was verified clean (no action)

All 28 Visualizer classes now have complete docstrings; the 24-missing-pages baseline finding is fully resolved. All gallery/showcase images exist on disk. All 12 `plots` figure functions, 25 statistics + 16 scales value-objects, and the `transform_*` family are well-documented. The auto-generation pipeline itself is correct — every API gap above is a source docstring gap.
