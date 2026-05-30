# Composite-mark constant style Design Spec

## 1. Scope

Make ferrum's composite/statistical marks (`mark_density`, `mark_histogram`, `mark_smooth`, `mark_hex`, `mark_contour`, `mark_ribbon`, `mark_errorbar`, `mark_errorband`, `mark_boxplot`, `mark_boxen`, `mark_violin`, `mark_qq`, `mark_raster`, `mark_swarm`, `mark_function`) accept **constant mark-style kwargs** (`opacity`, `stroke_width`, `fill`, `stroke`, `stroke_dash`, `size`, …) the same way simple marks already do, instead of funneling every kwarg into the transform desugar function (which has a fixed signature and rejects style kwargs). After this change `mark_density(opacity=0.4)` works like `mark_area(opacity=0.4)`.

## 2. Goals

- A constant style kwarg passed to any composite mark is applied to the rendered mark(s), not rejected.
- The set of "style" vs "transform" kwargs is determined per-mark with no hand-maintained collision table.
- Style validation, alias handling, and error messages match the simple-mark path (reuse `MarkBase`).
- Composite marks invoked with **no** style kwargs produce byte-identical output to today.
- The whole composite-mark family is covered in one coherent mechanism.

## 3. Non-goals

- Constant-encoding support (`encode(opacity=fm.value(0.4))`) — a separate encoding-layer feature, explicitly deferred.
- Per-layer style targeting from the grammar API (e.g. `mark_smooth(line={...}, band={...})`). The figure-function `mark={layer: {...}}` dict path (`_overrides.py`) already covers per-layer targeting; the grammar path applies flat style to all emitted layers.
- Any change to the figure-function override path (`_apply_overrides` / `_apply_mark_overrides`).
- Any Rust change. This is Python-layer kwarg routing only.

## 4. System behavior

When a composite mark is resolved (`Chart._resolve_pending`), its stored kwargs are partitioned into transform kwargs and style kwargs. Transform kwargs drive the desugar (unchanged behavior). Style kwargs are validated and applied to the resulting mark or layers.

- **Single-mark composite** (e.g. 1-D `mark_density`, `mark_histogram`, `mark_hex`): style is applied to the chart's `_mark_kwargs`. `mark_density(opacity=0.4).encode(x="v", color="g")` renders each density area with a translucent fill.
- **Layered composite** (e.g. `mark_smooth` → CI ribbon + line; 2-D `mark_density`/`mark_contour` → contour layers): the flat style kwargs are applied to **every** emitted layer. `mark_smooth(opacity=0.4)` dims both the ribbon and the line.
- **Composite layered onto a prior primitive mark** (e.g. `chart.mark_point().mark_smooth(opacity=0.4)`): style applies only to the composite's emitted layers (the smooth ribbon/line); the pre-existing primitive layer (the scatter) keeps its own kwargs.
- **Collisions resolve toward the transform**: `mark_density(fill=False)` keeps its existing meaning (line instead of area) because `fill` is a real `desugar_density` parameter; `mark_hex(cmap="viridis")` keeps `cmap` as a transform parameter. A fill *color* on a density is set via the `color=` alias (`mark_density(color="#888")`), which is not a desugar parameter and therefore routes to style.
- **Errors**: an unknown kwarg that is neither a desugar parameter nor a valid mark-style key raises `TypeError` from `MarkBase`, before the transform runs.

## 5. Architecture

Single change point: `Chart._resolve_pending` in `src/ferrum/chart.py`, around the `desugar_fn(x_field, y_field, **kwargs)` call.

Data flow per composite mark:

1. **Split** the pending mark's user kwargs into `(transform_kwargs, style_kwargs)` by introspecting the desugar function's signature. Keys naming a declared desugar parameter → `transform_kwargs`; all others → `style_kwargs`.
2. **Validate/normalize** `style_kwargs` through `MarkBase` (alias resolution, typo rejection) into a canonical style dict.
3. The existing auto-injection of transform-only keys (`groupby`, `y2_field`, `field`, smooth `name`) is added to `transform_kwargs` (these are all desugar parameters).
4. **Desugar** with `transform_kwargs` only (today's behavior for the transform side).
5. **Apply** the canonical style dict: to `_mark_kwargs` for the single-mark result, or merged into every emitted layer's `mark_kwargs` for the layered result, with the user style taking precedence over any desugar-set per-layer default. The prior primitive layer, when present, is left untouched.

Responsibilities stay where they belong: desugar functions own the transform; `MarkBase` owns style validation; `_resolve_pending` owns the routing. No new module is required; a small private helper (`_split_style_kwargs`) encapsulates the introspection.

## 6. Canonical interfaces / data contracts

Splitting helper (private; signature is the contract, not the body):

```python
def _split_style_kwargs(desugar_fn, user_kwargs: dict) -> tuple[dict, dict]:
    """Return (transform_kwargs, style_kwargs).

    transform_kwargs: keys that name a parameter of desugar_fn.
    style_kwargs: the remainder (to be validated by MarkBase).
    """
```

Style classification reuses the existing authoritative source:
- Valid style keys and aliases: `ferrum.marks.base.MarkBase` (`_VALID_MARK_KWARGS`, `_MARK_KWARG_ALIASES`, `to_mark_kwargs_dict()`).

## 7. Invariants and constraints

- **Byte-identical SVG** for any composite mark invoked without style kwargs (`style_kwargs` empty ⇒ `_mark_kwargs` unchanged). All existing goldens must stay byte-identical.
- **Transform behavior unchanged**: every existing desugar parameter still routes to the desugar; collision names (`fill`, `cmap`, `density`, `multiple`, `method`, `ci`, …) resolve to the transform whenever they are declared desugar parameters.
- **Parity with simple marks**: the accepted style keys, alias names, and `TypeError` behavior are exactly those of `MarkBase` as used by `_set_mark`.
- **Python-only**: no Rust, no Arrow, no spec-schema change.
- **No silent drops**: a kwarg that is neither transform nor valid style raises, never disappears.

## 8. Key decisions and tradeoffs

- **Signature introspection over a manual style allowlist.** Chosen because it is per-mark correct and needs no collision table; the alternative (a hand-maintained style set siphoned before desugar) duplicates `MarkBase` knowledge and must track per-mark collisions (`fill`, `cmap`). Relies on all desugar functions having explicit signatures (verified: none use `**kwargs`).
- **Flat style applies to all emitted layers** on layered composites, with user value winning over desugar defaults. Predictable and matches "a mark property applies to the mark." Per-layer precision remains available through the existing figure-function dict path; a grammar-level per-layer form is deferred (YAGNI).
- **Prior primitive layer untouched.** Style from a composite mark scopes to that composite's output, not to a separately-configured primitive layer added earlier in the chain.
- **Reuse `MarkBase` rather than re-validate.** Keeps one source of truth for valid style keys, aliases, and error messages; avoids drift between simple and composite mark validation.
- **Do not unify with `_overrides._apply_mark_overrides`.** It runs post-resolution on named layers with catalog validation; the grammar path runs during resolution with flat style. They share intent but differ in timing and keying; forcing a shared abstraction now would be premature. A comment will note the relationship.
- **Constant-encoding path deferred.** `encode(<channel>=fm.value(...))` is a distinct mechanism in the encoding/serialization subsystem; bundling it would widen scope without being required to solve the mark-style gap.

## 9. Acceptance criteria

- `mark_density(opacity=0.4).encode(x=..., color=...)` renders translucent area fills; the same kwarg works across the composite-mark family.
- `mark_smooth(opacity=0.4)` applies opacity to both the CI ribbon and the line; a prior `mark_point()` layer is unaffected.
- `mark_density(fill=False)` still renders a line (not an area); `mark_hex(cmap=...)` still routes `cmap` to the transform.
- `mark_density(opacity=0.4)` no longer raises; `mark_density(<typo>)` raises `TypeError` via `MarkBase`.
- All existing goldens are byte-identical; `cargo test`, `uv run pytest -n auto`, and wasm clippy stay green.
- A new styled-density golden is added and visually inspected per the CLAUDE.md golden rule.

## 10. Validation strategy

- **Behavioral**: per composite-mark family, assert a representative style kwarg reaches the rendered output (e.g. `rgba` fill for `opacity`, `stroke-width` attribute for `stroke_width`); assert transform kwargs still take effect (`multiple="stack"` changes layout); assert collisions resolve to the transform (`fill` line/area on density, `cmap` on hex); assert layered application hits all emitted layers; assert prior-layer isolation; assert typo raises.
- **Regression/byte-identity**: full golden suite must show zero diffs for unaffected charts (the proof of behavior preservation); one new inspected golden pins the styled-density output.
- **Gates**: `uv run pytest -n auto`, `cargo test`, `cargo clippy -p ferrum-wasm … -D warnings`, plus `/regression-test` after implementation.

## 11. Open questions

None.
