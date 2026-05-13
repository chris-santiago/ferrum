# Facet-Before-Transform Pipeline Order

**Date:** 2026-05-13  
**Status:** Proposed

## Problem

Transforms run on the full RecordBatch, then faceting partitions the output. If a transform (Smooth, Histogram, KDE, etc.) replaces the batch with computed output, the facet column is lost → "unknown column" error.

## Decision

Faceting partitions data **before** transforms run. Each facet panel is an independent view with its own data subset. Transforms execute per-panel.

This matches Vega-Lite's model and user expectations: "facet by dose, then smooth" means each dose group gets its own smooth — no explicit `groupby="dose"` needed on the transform.

## Current pipeline (broken)

```
full_data → transforms → transformed_batch → facet partition → per-panel render
                                              ↑ facet column gone
```

## Proposed pipeline

```
full_data → facet partition → per-panel subset → transforms → per-panel render
```

## Where to change

`crates/ferrum-core/src/render/prepare.rs` — the facet loop currently partitions the *rendered* batch. Move the partition upstream so each panel's transform pipeline receives only its subset.

Specifically: when `spec.facet` is `Some`, partition the input `RecordBatch` by the facet column(s) first, then run `apply_transforms` independently for each partition. The non-faceted path is unchanged.

## Edge cases

- **Named transforms with `data_source`**: each panel's transform outputs are panel-scoped. Layer `data_source` lookups resolve within the panel's transform namespace — no cross-panel leakage.
- **Shared scales**: faceted panels already share x/y scales via `resolve_scales` across all panels. This doesn't change — scale resolution still sees all panels' data.
- **No facet**: pipeline is identical to today (single partition = full batch).
