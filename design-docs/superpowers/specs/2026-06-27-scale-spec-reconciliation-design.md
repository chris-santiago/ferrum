# ScaleSpec ↔ PyO3 *Scale Reconciliation Design Spec

> SPEC-04 / GitHub issue #38. Resolves the dual (really triple) modelling of scales
> into one canonical wire/render representation with a compiler- and test-enforced
> link to the user-facing construction surface.

## 1. Scope

Today a scale is modelled three times: the `ScaleSpec` serde enum (`spec/encoding.rs`,
the wire/render form), the 16 user-facing PyO3 `*Scale` classes (`scale/*.rs`,
construction + compute facades), and a hand-written Python bridge `_scale_to_dict`
(`src/ferrum/encoding/_scale.py`) that copies fields off each pyclass into a wire
dict. The three can drift independently; `QuantileScale`/`ThresholdScale` have no
`ScaleSpec` counterpart and raise `TypeError` when passed to `encode(scale=...)`. This
work makes `ScaleSpec` the single canonical representation, links each `*Scale` class
to it through a Rust `to_scale_spec` method, collapses the Python bridge to a thin
delegation, and adds the two missing wire variants — all without breaking the public
API or changing any render/`to_json` output.

## 2. Goals

- One canonical wire/render representation: `ScaleSpec`.
- Each `*Scale` pyclass is explicitly linked to its `ScaleSpec` variant by an
  inherent Rust `to_scale_spec(&self) -> ScaleSpec`; the link is enforced at compile
  time (struct-literal field coverage) and test time (a parity guard).
- `_scale_to_dict`'s per-class field copying is eliminated; the Python bridge
  delegates to the Rust-emitted canonical dict.
- `QuantileScale` and `ThresholdScale` round-trip through `encode(scale=...)` and
  render (instead of raising).
- The public API surface (`fr.LinearScale(...)`, `.bandwidth()`, `.ticks()`, `.scale()`,
  every constructor) is unchanged. Strictly non-breaking.
- Render output and `Chart.to_json()` are byte-identical for every pre-existing scale.

## 3. Non-goals

- **Discrete-color (binned) rendering for Quantize/Quantile/Threshold.** All three
  degrade to continuous color today; Quantile/Threshold inherit exactly the parity
  their sibling Quantize already has. True per-bin color mapping is a separable
  follow-up that spans the color renderer uniformly, logged as the north-star.
- Changing the compute-facade math (`.bandwidth()`, `.scale()`, `.ticks()`, `.invert*`).
- Collapsing the `*Scale` structs to store a `ScaleSpec` internally (their compute
  methods need resolved numeric domain/range that `ScaleSpec` does not carry). They
  remain thin builders that *emit* the canonical form.
- Any change to the dict-form scale path (`scale={"type": ...}`) or the reactive
  `domain=Parameter(...)` rewrite.

## 4. System behavior

- A user constructing `fr.QuantileScale(domain=..., range=...)` or
  `fr.ThresholdScale(...)` and passing it to an encoding's `scale=` produces a valid
  chart that serializes and renders. Positionally (x/y) these degrade to a Linear
  scale, matching the existing Quantize/Sequential/Diverging behavior.
- For every other `*Scale` class, observable behavior is unchanged: identical render
  SVG and identical `Chart.to_json()`.
- The transient dict returned by `_scale_to_dict(scale)` for a pyclass instance
  changes shape — it becomes the canonical `ScaleSpec` serialization (which includes
  serde-defaulted keys such as band `padding`/`align`). This dict is an internal
  input to deserialization and never appears in `to_json()`; only direct callers of
  `_scale_to_dict` (the wire-shape unit tests) observe it.

## 5. Architecture

Single-source the scale wire form the way every other declaration family already is
(Marks `to_mark_kwargs_dict`, Axis/Legend `to_dict`, Theme `to_spec_dict`, Channels
`to_encoding_spec_dict`), with the difference that the `*Scale` classes are
Rust-backed pyclasses — so the "emit my wire dict" method lives in Rust.

Data flow (pyclass scale → wire), after the change:

```
fr.BandScale(...)               PyO3 pyclass (Rust-backed; compute facade)
  │  _scale_to_dict(scale)      Python: single delegation for pyclass instances
  ▼
scale._to_scale_spec_dict()     #[pymethods] wrapper:
  = encode_serde_value_for_py(  serde-serialize the canonical form
        self.to_scale_spec())
  ▼  to_scale_spec(&self)       inherent Rust: struct fields → ScaleSpec variant
ScaleSpec                       the single canonical representation
  ▼  serde_json round-trip via EncodingSpec::new's json_round (unchanged)
stored ScaleSpec → render / to_json   (byte-identical to today)
```

The dict / `Parameter` / `None` branches of `_scale_to_dict` are unchanged; only the
13 `isinstance(*Scale)` branches collapse to the single delegation.

Resolution path (`build_from_scale_spec`) gains arms for the two new variants; it is
the one compiler-enforced match site. The 10 catch-all `_ =>` consumers
(`domain_param`, `set_domain`, the `color.rs` scheme/domain/midpoint/reverse helpers,
`auxiliary.rs` size range, `scene_build.rs` x_domain) are semantically correct as-is
for discrete-binning scales and stay unchanged.

## 6. Canonical interfaces / data contracts

**New `ScaleSpec` variants** (mirror `Quantize`'s shape, but numeric range):

```rust
// numeric range distinguishes these from Quantize { range: Option<Vec<String>> }
Quantile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    domain: Option<Vec<f64>>,   // sorted sample values
    #[serde(default, skip_serializing_if = "Option::is_none")]
    range: Option<Vec<f64>>,    // discrete numeric outputs
},
Threshold {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    domain: Option<Vec<f64>>,   // threshold boundaries
    #[serde(default, skip_serializing_if = "Option::is_none")]
    range: Option<Vec<f64>>,    // discrete numeric outputs; len == domain.len() + 1
},
```

Serde tag follows the enum's existing `tag = "type", rename_all = "lowercase"`, so the
wire types are `"quantile"` and `"threshold"`. The computed `quantiles` cut-points are
**not** transmitted (deterministic from the sample domain).

**The bridge method (contract, per pyclass):**

```rust
impl <EachScale> {
    // inherent: reads this struct's own fields into the matching variant
    pub(crate) fn to_scale_spec(&self) -> ScaleSpec { /* ScaleSpec::<Variant> { .. } */ }
}
#[pymethods]
impl <EachScale> {
    // Python-facing: serialize the canonical form to a wire dict
    fn _to_scale_spec_dict(&self, py: Python) -> PyResult<Py<PyAny>>; // encode_serde_value_for_py
}
```

**Correctness contract for already-bridged classes** — the invariant the implementation
must satisfy for byte-identity:

```
for every existing *Scale s:
    to_scale_spec(s)  ==  deserialize_as_ScaleSpec( old _scale_to_dict(s) )
```

i.e. the new path stores the *same* `ScaleSpec` the old path stored, so render and
`to_json()` are unchanged.

**Python bridge, after collapse** (shape contract, not an implementation transcript):

```python
def _scale_to_dict(scale):
    if scale is None: return None
    if isinstance(scale, dict): ...        # Parameter / type-default branches UNCHANGED
    if hasattr(scale, "_to_scale_spec_dict"):
        return scale._to_scale_spec_dict()  # replaces all 13 isinstance branches
    return scale                            # unknown → let Rust raise
```

## 7. Invariants and constraints

- **Strictly non-breaking.** No public symbol removed or renamed; no constructor
  signature changed. (`ferrum-spec.md` is the API contract.)
- **Render + `to_json()` byte-identical** for every scale that worked before. Proven,
  not assumed (see §10).
- **Positional fallback parity:** Quantile/Threshold resolve to `ScaleKind::Linear`
  for x/y, identical to Quantize/Sequential/Diverging.
- **Threshold validity:** `range.len() == domain.len() + 1` (already enforced by the
  pyclass constructor; the wire variant carries no additional validation duty).
- **No new serialization helper:** reuse `encode_serde_value_for_py` (the existing
  getter serializer).
- **`cargo test` and `pytest -n auto` green** before the issue is marked done
  (project hard constraint for any phase ≥ 2 change).

## 8. Key decisions and tradeoffs

- **Decision: canonical Rust bridge (`to_scale_spec`), not a Python-only patch.**
  Defended via `chris-code:coherent-change` (decision-only). Making the wire builder
  live in Rust next to `ScaleSpec` means extending a variant breaks the builder until
  updated — the compile-time drift guard. A Python-only fix (add two `isinstance`
  branches + a parity test) was rejected: it fixes the Quantile/Threshold symptom but
  leaves scales the lone dual-sourced family, with only a test-time guard.
- **Decision: keep the public `*Scale` classes (reject breaking demotion, issue
  Option 2).** Demotion deletes useful compute facades (`.bandwidth()`, `.ticks()`)
  and breaks the API for no gain over the bridge.
- **Decision: the transient `_scale_to_dict` dict adopts canonical `ScaleSpec`
  serialization** rather than preserving its current ad-hoc byte shape. Justified
  because that dict never reaches `to_json()` (the getter re-serializes the stored
  `ScaleSpec`); only wire-shape unit tests observe it, and they move to the canonical
  expectation. This is what dissolves the issue's stated "byte-identity risk."
- **Decision: Quantile/Threshold get their own variant shape** (numeric `range`),
  not a reuse of `Quantize` (color-string `range`). Their range types differ.
- **Decision: discrete-color rendering is out of scope** (§3). Quantile/Threshold
  reach exactly Quantize's parity; the binned-color feature is the logged north-star.
- **Decision: `to_scale_spec` as an inherent method**, not `From<&Scale> for
  ScaleSpec`. `From` is idiomatic but still needs a `#[pymethods]` wrapper to reach
  Python; the inherent method reads more directly. Stylistic, not behavioral.

## 9. Acceptance criteria

1. `fr.QuantileScale(domain=[...], range=[...])` and `fr.ThresholdScale(domain=[...],
   range=[...])` passed to `encode(scale=...)` build, serialize, and render without
   error (the reproduced `TypeError` is gone).
2. `ScaleSpec::Quantile` and `ScaleSpec::Threshold` exist, deserialize from
   `{"type":"quantile",...}` / `{"type":"threshold",...}`, and serde round-trip.
3. Every `*Scale` pyclass has an inherent `to_scale_spec`; `_scale_to_dict` contains
   no per-class `isinstance(*Scale)` branch (only the dict/Parameter/None branches and
   the single delegation remain).
4. A drift-guard test enumerates every `*Scale` pyclass and asserts each yields a
   deserializable `ScaleSpec` that round-trips; the pyclass set ↔ variant set is
   complete (every class maps to a variant; documented color-only exceptions noted).
5. Render SVG and `Chart.to_json()` are byte-identical for a representative chart per
   pre-existing scale type (Linear, Log, Band, Point, Ordinal, Sequential, Diverging,
   Quantize, BinOrdinal, Pow, Sqrt, Time, Symlog).
6. Full `pytest -n auto` and `cargo test` green; lite-review gates clean.

## 10. Validation strategy

- **RED proof (the bug):** the reproduction `Chart(df).mark_point().encode(color=fr.X(
  "c", scale=fr.QuantileScale(...)))` raises `TypeError` on pre-change code and builds
  + renders after — captured as a regression test for both Quantile and Threshold.
- **Byte-identity guard:** for each pre-existing scale type, assert `to_json()` (and a
  rendered SVG where cheap) is unchanged across the change. This directly validates
  the §7 non-breaking invariant and the §6 correctness contract.
- **Drift guard:** the parity test (§9.4) — failing if any pyclass lacks a variant or a
  round-trip — plus the compiler's exhaustive `build_from_scale_spec` match.
- **Wire-shape tests:** Rust `scale_spec_quantile_round_trip` / `_threshold_round_trip`
  mirroring the existing `scale_spec_*_round_trip` idiom; Python `_scale_to_dict`
  assertions updated to the canonical serialization and extended to Quantile/Threshold.

## 11. Open questions

None blocking. (Whether to fold in discrete-color rendering is a deliberate non-goal,
§3 — a separate follow-up the maintainer may schedule independently.)
