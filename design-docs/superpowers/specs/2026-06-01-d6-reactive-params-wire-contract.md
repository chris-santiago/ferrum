# D6 Reactive Parameters — Wire Contract (build addendum)

> Sharpens the open wire contract of `2026-06-01-flexibility-new-capabilities-design.md` §5–§6 for D6 (Task 5 of the new-capabilities plan). This is the single source of truth the four layers (Python serialize → Rust spec → static resolve → WASM runtime) must agree on. Grounded in the actual code at each seam (read 2026-06-01).
>
> **Guiding invariant:** a chart that declares **no** parameters serializes and renders **byte-identically** to today. Every new field is `skip_serializing_if` empty/None. Existing `selections`/`conditionals` keys are unchanged (the WASM runtime already reads them).

## 1. Python `Parameter` model (`src/ferrum/parameter.py`, NEW)

```python
class Parameter:                       # plain base (NOT a dataclass)
    name: str
    def ref(self) -> dict: return {"param": self.name}   # marker when referenced
    def to_param_spec_dict(self) -> dict: ...            # entry in the params section

@dataclass(frozen=True)
class VariableParameter(Parameter):
    name: str
    value: Any = None
    bind: Any = None                   # None | "legend" | a bind-input dict
```

`fm.param(name, value=None, bind=None) -> VariableParameter`.

`parameter.py` is the lower-level module: it must **not** import `selection.py`. `selection.py` imports `Parameter` from it and makes `Selection(Parameter)` (frozen dataclass subclassing the plain base — legal in Python). `Selection` implements:
- `ref()` → `{"param": self.name}` (inherited).
- `to_param_spec_dict()` → `{"name": name, "kind": self.kind, "select": {...self.params}}` plus `"bind"` when set.

`isinstance(x, Parameter)` is the reference-site discriminator and must be true for both `VariableParameter` and `Selection`.

## 2. `params` section (top-level spec JSON)

`Chart` grows `self._params: list[Parameter] | None`. `Chart.to_spec()` (chart.py ~3142, just before `return ChartSpec(**kw)`) emits, when non-empty:

```python
kw["params"] = json.dumps([p.to_param_spec_dict() for p in resolved._params])
```

`ChartSpec.__new__` gains `params: Option<&str>` (JSON), deserialized exactly like `selections`/`conditionals`. The wire array:

```json
[
  {"name": "thresh", "kind": "variable", "value": 50.0, "bind": null},
  {"name": "brush",  "kind": "interval", "select": {"translate": true, "zoom": true, "resolve": "global", "encodings": ["x"]}},
  {"name": "sel",    "kind": "point",    "select": {...}, "bind": "legend"}
]
```

Selections continue to ALSO serialize into the existing `selections` key (WASM reads that today). The `params` section is additive: it is the unified declaration the static resolver and new WASM wiring read. A `Selection` is auto-promoted into `_params` whenever it is registered (`add_selection`) OR referenced; `fm.param` variables go only into `_params`.

## 3. Rust `ParameterSpec` (`crates/ferrum-core/src/spec/parameter.rs`, NEW)

Lean — static resolve only needs initial values. Full selection projection stays in `selections` for WASM.

```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ParamKind { Variable, Point, Interval }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ParameterSpec {
    pub name: String,
    pub kind: ParamKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<serde_json::Value>,   // variable initial value
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bind: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub select: Option<serde_json::Value>,  // opaque to static; WASM-bound
}
```

`ChartSpec` gains `#[serde(default, skip_serializing_if = "Vec::is_empty")] pub params: Vec<ParameterSpec>`. Add to `__new__` signature + the struct literal in `new()` AND in every `ChartSpec { ... }` literal in the test module (there are several — `params: Vec::new()`).

A `ParamStore` (built from `spec.params`) resolves a name to its static initial value: variable → `value`; point/interval → `None` (empty selection statically).

## 4. Reference markers (the four reference sites)

### 4a. `scale.domain = param` → reactive rescale
Carried as a **sibling** of `domain` inside the scale dict, NOT by retyping `domain` (keeps every scale struct byte-stable):

```json
"scale": {"type": "linear", "domainParam": "brush"}    // literal `domain` omitted
```

- **Python:** when `scale={"domain": <Parameter>}`, emit `domainParam: param.name`, drop the literal `domain`. (`_scale_to_dict` / encoding scale build.)
- **Rust:** add `#[serde(rename = "domainParam", default, skip_serializing_if = "Option::is_none")] pub domain_param: Option<String>` to `ContinuousScaleCommon` (the 7 continuous variants Linear/Log/Time/Symlog/Pow/Sqrt/Utc — these are the overview+detail reactive-rescale case in §9). Static resolver: before scale resolution (scene_build.rs), if `domain_param` set and the `ParamStore` yields a numeric array value → set `domain = Some(value)`; else leave `domain = None` (auto-infer from data — the correct static semantics for an empty selection).
- **Scope note:** ordinal/band/point/sequential/diverging domain-params (categorical reactive rescale) are NOT a §9 acceptance item and are a recorded follow-up, not part of D6. The `ScaleSpec` enum has no shared ordinal common; adding `domainParam` to each categorical variant is deferred.

### 4b. `transform_filter(param)` → crossfilter
Keep `predicate` **required** (no Optional ripple). A param filter emits a pass-through predicate plus a marker:

```json
{"type": "filter", "predicate": "true", "param": "brush"}
```

- **Python:** `transform_filter` accepts a `Parameter`; emits `{"type":"filter","predicate":"true","param":param.name}`.
- **Rust:** `FilterSpec` gains `#[serde(default, skip_serializing_if = "Option::is_none")] pub param: Option<String>`. Static `apply()` ignores `param` and runs `predicate` ("true" → keeps all rows: correct static semantics). WASM reads `param` to crossfilter live.

### 4c. `value = param` — DEFERRED follow-up (NOT in D6)
A parameter bound to a standalone constant encoding (`encode(size=fm.value(param))`) would require a value-only (datum-free constant) encoding channel, which ferrum has never had — `fm.value(...)` is consumed only inside conditionals today. No §9 acceptance criterion needs it: "a variable parameter drives an encoding" (§9 #4) is satisfied by `scale.domainParam` with a variable array value (the slider sets the domain → drives the mapping → static uses the initial value). Building a constant-value channel + an `EncodingValue::Param` wire variant (which would force non-exhaustive-match churn across ferrum-core and ferrum-wasm) is out of D6 scope and is recorded in the code-archaeology doc as a follow-up. `fm.value(...)` keeps its current literal-only behavior.

### 4d. conditional test on a parameter — `fm.when(...).then(...).otherwise(...)`
New module-level builder in `selection.py` (or `parameter.py`), additive to the existing `Selection.when(if_encoding).otherwise(else_encoding)`:

```python
fm.when(selection).then(v_if).otherwise(v_else)   # -> ConditionalSpec
```

Produces a `ConditionalSpec(selection_name=param.name, if_selected=value(v_if), if_not=value(v_else))` — reuses the existing ConditionalEncoding wire. The conditional test parameter is a **selection** (the test is "datum ∈ selection"); variable params drive value/domain/bind, not the conditional predicate. Back-compat: existing `sel.when(enc).otherwise(enc)` unchanged.

### 4e. `bind="legend"`
`selection_point(bind: str | None = None)`. `bind="legend"` flows into `to_param_spec_dict()` (`"bind": "legend"`). WASM wires legend-entry clicks to toggle the point selection (and thus the series via the conditional). `selection_interval`/`selection_single`/`selection_multi` signatures otherwise unchanged.

## 5. Static-render semantics (Rust, 5d) — determinism

| Reference | Static resolution |
|---|---|
| `domainParam` → variable (numeric array value) | use the array as the scale domain |
| `domainParam` → selection (empty) | `domain = None` → auto-infer from data |
| `filter` `param` | ignore marker; run `predicate:"true"` → keep all rows |

Conditionals are interactive-only (the static SVG path does not apply them), so the static resolver's sole job is `domainParam` substitution; the filter `param` marker is inert statically because its predicate is already `"true"`.

A spec with empty `params` and no markers takes every existing code path unchanged → byte-identical output. **This is the gate**: param-free goldens must not move.

## 6. Interactive runtime (WASM/JS, 5e) — what the emitted artifacts must contain

Validated by inspecting emitted HTML/JS + scene JSON (no browser in CI). The interaction config / scene JSON delivered to `WasmRenderer` must carry the `params` section and the reference markers so the runtime can:
- **Reactive rescale (overview+detail):** a brush param feeding a `domainParam` on a linked panel rescales that panel on brush change.
- **Crossfilter:** a `filter` `param` removes non-matching rows from the linked panel on brush change.
- **Legend toggle:** a `bind="legend"` point selection toggles the matching series on legend-entry click.

Tests assert the param, its references, and its event bindings are present and correct in the emitted artifacts — not merely that export succeeded. `cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings` clean; WASM rebuilt via the documented `wasm-pack` command.

## 7. Public API additions (exports)

`fm.param`, `fm.when` (module-level), `Parameter`, `VariableParameter` added to `ferrum/__init__.py` `__all__` and to `tests/test_docstring_coverage.py` `_DOC_ALLOWLIST` (docstrings required). `value`/`selection_*` unchanged.
