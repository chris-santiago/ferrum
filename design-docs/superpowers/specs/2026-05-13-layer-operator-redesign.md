# `+` Operator Redesign: Always Layer

**Date:** 2026-05-13  
**Status:** Proposed

## Problem

`Chart.__add__` is ambiguous: same data → layer, different data → hconcat fallback. Users can't predict which they'll get.

## Decision

`+` always means **layer**. Concat becomes explicit.

| Operation | API |
|---|---|
| Layer (same data) | `c1 + c2` or `.layer(Layer(...))` or method chaining |
| Layer (different data) | `c1 + c2` (auto null-pad merge) |
| Horizontal concat | `fm.hconcat(c1, c2)` |
| Vertical concat | `fm.vconcat(c1, c2)` |

## How different-data layering works

When `c1 + c2` detects different data, it builds a unified DataFrame by null-padding:

```
c1.data: {x, y, class}        → {x, y, class, ref_y: null}
c2.data: {ref_y}              → {x: null, y: null, class: null, ref_y}
unified = pl.concat([padded_c1, padded_c2], how="diagonal")
```

Each layer's encoding references only its own columns. Nulls in the other layer's columns are skipped by the renderer (marks already skip null values).

This is what `decision_boundary_chart` does manually today — the `+` operator automates it.

## Changes required

1. **`Chart.__add__`** — remove `_shares_data_with` gate and hconcat fallback. Always call `_merge_as_layers`. When data differs, run `pl.concat([df1, df2], how="diagonal")` to produce the unified batch.
2. **`Chart.__or__`** — keep as hconcat (or remove if we want `fm.hconcat()` only).
3. **`fm.hconcat()` / `fm.vconcat()`** — new top-level functions returning `HConcatChart` / `VConcatChart`. These already exist as classes; just add the convenience constructors.
4. **Remove the "Layered charts with differing data" warning** — it becomes normal behavior.
5. **Update `HConcatChart`/`VConcatChart`** — no longer produced by `+`.

## Migration

The only code that relies on `+` producing hconcat is the `build_hconcat` demo — change it to `fm.hconcat(c1, c2)`. No public API contract promises hconcat from `+`.
