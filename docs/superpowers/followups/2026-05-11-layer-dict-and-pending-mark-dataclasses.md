# Follow-up: P1.3b/c layer dict + `_pending_stat_mark` dataclasses

**Filed**: 2026-05-11
**Origin**: deferred from `docs/superpowers/plans/2026-05-11-python-coherence-pass-plan.md` P1.3 (C5)
**Status**: scoped, not yet done

P1.1 (C2 — collapse `_resolve_pending` 2/3-tuple protocol) was originally
filed here as deferred. It landed in commit `bbe518c` with the closure
adapters described in the original deferral note; this section is removed.

## P1.3 — frozen dataclasses for layer dicts and `_pending_stat_mark`

### What landed under P1.3a

`Chart._facet` was converted from an ad-hoc 4-shape `dict` (`"wrap"`/`"grid"`
tagged-union) to a frozen `_Facet` dataclass. See commit `226ba24` and
`tests/test_facet.py` for the migration pattern.

### What remains under P1.3

### Layer dicts (P1.3b)

The "layer" shape — a `dict` with `"mark"`, `"encoding"`, `"transforms"`,
`"mark_style"`/`"mark_kwargs"`, `"data_source"`, `"position"` keys — is
constructed across all four mark-desugar modules and consumed in three
places:

**Producers (~50 sites)**:

- `src/ferrum/marks/statistical.py` — smooth ribbon+line layers (~5)
- `src/ferrum/marks/heavy_stat.py` — contour, violin, qq, raster, hex,
  swarm (~25)
- `src/ferrum/marks/composite.py` — boxplot, boxen, errorbar, errorband,
  ribbon (~15)
- `src/ferrum/marks/diagnostic.py` — residuals, prediction_error, roc,
  pr, calibration, gain, lift, discrimination_threshold, confusion,
  class_prediction_error, importance, shap_*, pdp, learning_curve,
  validation_curve, cv_scores, alpha_selection, silhouette, pca_scree,
  intercluster_distance, decision_boundary, rank1d, rank2d,
  parallel_coordinates (many)

**Consumers**:

- `Chart._build_layers_list()` (in `src/ferrum/chart.py`)
- `_SpecView.layers` (in `src/ferrum/_spec_view.py`) — already reads
  via `.get(key)`
- `_expand_layers()` (in `src/ferrum/chart.py`, module-level helper)

**Dual-key handling to resolve in the conversion**: the layer shape has
both `mark_kwargs` and `mark_style` keys (legacy alias). `_SpecView`
reads `d.get("mark_kwargs") or d.get("mark_style")`. The dataclass
should pick one canonical name and update all producers.

**Approach**: introduce `_Layer` dataclass alongside the dict shape;
migrate producer-by-producer; run the golden suite after each module's
producers are migrated; remove the dict path once all producers emit
`_Layer` instances.

### `_pending_stat_mark` (P1.3c)

Currently a tuple — either 2-tuple `(kind, kwargs)` (legacy form for
`mark_density`/`mark_histogram`/`mark_smooth`) or 3-tuple
`(kind, kwargs, desugar_fn)` (the helper-driven form).

**Blocked on P1.1**: `_resolve_pending`'s 2-tuple branch must be
collapsed into the 3-tuple branch before the dataclass conversion makes
sense. Otherwise the dataclass would need to support two shapes.

**Once P1.1 lands**:

```python
@dataclass(frozen=True)
class _PendingMark:
    kind: str
    kwargs: dict
    desugar_fn: Any  # Callable[[str|None, str|None, **kwargs], tuple]
```

Touched sites:

- `Chart._set_composite_mark()` — single producer
- `Chart._resolve_pending()` — single consumer (3-tuple branch)
- `Chart._clone()` — passes the slot through; no change needed
- The two legacy 2-tuple producers in `mark_density`/`mark_histogram`/
  `mark_smooth` — disappear under P1.1.

## Why deferred

The single-session refactor budget for P1.3 was 2 commits. `_facet`
took one. The layer-dict conversion's surface area (50+ producers
across 4 files plus the dual-key migration) is single-commit-hostile —
it should land as a sequence of producer-batch commits, each verified
against the golden suite, to keep regressions tractable.
