# Refactoring Heuristics — Full Catalog

Each heuristic includes the smell, why it happens, the typical fix, and counter-indications (when *not* to apply it). Use these to classify findings in Phase 2 and to justify proposals in Phase 3.

## 1. Boolean and mode-flag creep

**Smell.** A function has multiple booleans or magic mode strings that substantially change behavior. Call sites like `process(data, strict=True, validate=False, async_=True)` force readers to look up every flag.

**Why it happens.** Each new requirement bolted on as a flag rather than rethinking the entry point.

**Fix.** Pick the smallest option that improves readability:
- Split into separately named functions (`render_strict` vs `render_lenient`).
- A small config dataclass (`RenderOptions`) — best when there are ≥3 flags or callers commonly omit defaults.
- An enum-like `Literal` type (`Mode = Literal["strict", "lenient"]`) — good middle ground.
- Strategy functions passed in — only when the behavior is genuinely pluggable.

**Counter-indication.** A single, self-documenting `bool` at a call site doesn't need a dedicated type. `sort(reverse=True)` is fine.

## 2. Dict-shaped domain data

**Smell.** Functions pass around loosely-structured dicts whose keys are required *by convention*. Typos silently produce `KeyError` or `None`-poisoned downstream values.

**Why it happens.** Easy to start with a dict; nobody promoted it to a type when it earned the right to be one.

**Fix.** Promote to a structured type *only when it clarifies*:
- `dataclass` (with `slots=True` if hot) — best when there's behavior or invariants.
- `TypedDict` — best when the shape stays a dict (JSON in/out, kwargs).
- `NamedTuple` — best for small, immutable, tuple-friendly values.
- Pydantic / `attrs` if the codebase already uses them.

**Counter-indication.** A truly heterogeneous payload — like a JSON blob from an external API with optional/unknown fields — may be clearer as a dict with explicit `.get()` calls.

## 3. Hidden side effects

**Smell.** A "helper" reads environment variables, touches the filesystem, performs logging at unexpected levels, mutates module globals, or makes network/DB calls. Tests that look local turn out to require fixtures.

**Why it happens.** Convenience: the side effect was easier to do here than to plumb through.

**Fix.** Surface the behavior. Options, in order of preference:
- Move the effect to the boundary (the entry point, not the helper).
- Inject the effectful thing (`fs`, `clock`, `db`) as a parameter with a sensible default.
- Rename the function to make the effect obvious (`load_config_from_env` not `get_config`).
- At minimum, document the effect in the docstring.

**Counter-indication.** Don't push effects to the boundary if doing so threads parameters through 6 layers — that's worse than a clearly-named effectful helper.

## 4. Orchestration mixed with implementation

**Smell.** One function reads config, opens a file, parses it, validates it, transforms it, and writes the result. The high-level flow is buried in low-level mechanics.

**Why it happens.** Started as one job, accrued sub-tasks over time, no one extracted them.

**Fix.** Extract leaf operations. The orchestrator becomes a short, sequential reading of high-level steps. Each leaf is independently testable.

**Counter-indication.** Don't extract a "step" that is one line and will never be reused — that's just indirection.

## 5. Utility dump modules

**Smell.** `utils.py`, `helpers.py`, `common.py`, `misc.py` — files whose name reveals nothing about what's inside. Functions accumulate that have nothing to do with each other.

**Why it happens.** "Add it to utils for now" without later sorting.

**Fix.** Split by *domain responsibility*. `string_utils`, `time_utils`, `db_helpers` are still weak; better to find the responsibility (`paths`, `retry`, `pagination`) and let related functions live near the code that uses them. If a utility has exactly one caller, inline it.

**Counter-indication.** Don't split if the resulting modules would have circular imports or near-identical contents — that's the wrong cut line.

## 6. Return shape drift

**Smell.** Sibling functions return inconsistently — one returns `(value, error)`, the next returns `{"value": x, "error": None}`, the next returns `None` to signal failure, the next raises. Callers wrap each one differently.

**Why it happens.** Each function made a local decision.

**Fix.** Pick a canonical contract. Common Pythonic options:
- Always return the value, raise on failure (most idiomatic).
- Always return a small result object (`Result` / `Maybe` / domain-specific dataclass).
- Always return a tuple of fixed arity.

**Counter-indication.** Functions that span very different domains can legitimately have different contracts.

## 7. Exception drift

**Smell.** Similar failure modes raise wildly different exception types — `ValueError`, `RuntimeError`, custom subclass, `KeyError`, sometimes a `dict` with `"error"`. Callers can't write uniform handlers.

**Why it happens.** Each module made a local error decision.

**Fix.** Unify the error story for the package:
- Define a small hierarchy of domain exceptions (often 2–4 types is enough).
- Raise them consistently at boundaries.
- Catch the base class in callers when they want broad recovery.

**Counter-indication.** Don't paper over genuinely different errors with one type — keep `ValidationError` and `NetworkError` distinct.

## 8. Overgrown classes

**Smell.** A class with 30 methods, weak invariants, and a name that ends in `Manager`, `Service`, `Handler`, or `Helper`. Usually mostly a namespace.

**Why it happens.** "Let's group these as a class" replaced module organization.

**Fix.** Ask: does this class hold durable state? Does it enforce invariants? Does it implement a protocol? If not three "yes"es, prefer functions or smaller collaborating objects. A module of related functions is often the right answer.

**Counter-indication.** If the class is a stateful resource (`Database`, `HttpClient`) or a protocol implementation, leave it alone.

## 9. Stateful code without need

**Smell.** A class is instantiated, has two methods called on it in sequence, then discarded. The "state" is just intermediate values from one call to the next.

**Why it happens.** Defaulted to a class because Python objects feel more "structured."

**Fix.** Collapse to one or two pure functions. Pass intermediate values explicitly.

**Counter-indication.** If the construction validates inputs and the methods enforce that validation, keep the class.

## 10. Under-modeled state

**Smell.** Correctness depends on call order (`init()`, then `configure()`, then `run()`). Mid-construction objects exist where attributes are `None` until "ready."

**Why it happens.** Class grew incrementally; nobody encoded the lifecycle.

**Fix.** Options, lightest first:
- Move all required state into `__init__` and remove the lifecycle.
- Use a factory function/classmethod that returns a fully-constructed object.
- Split into two types if the "before" and "after" states are truly distinct (typestate-lite).
- Document the lifecycle if it can't be eliminated.

**Counter-indication.** Some lifecycles are domain-essential (e.g., `AsyncContextManager` patterns). Don't fight them.

## 11. Public API leakage

**Smell.** Callers import from `mypackage.internal.submodule._helpers`. The intended public surface is unclear. Refactoring the internals breaks downstream code.

**Why it happens.** No one curated `__init__.py`; callers reached for whatever path worked.

**Fix.**
- Define the intended public API in `__init__.py` via explicit re-exports and `__all__`.
- Prefix truly internal modules with `_` (`_internal.py`).
- Document the supported import paths.
- Provide deprecation shims if changing public paths.

**Counter-indication.** For pure-internal-use packages (private monorepo), strict curation may be friction without payoff.

## 12. Framework overreach

**Smell.** Decorators wrap decorators; a registry collects classes at import time; a metaclass adds methods invisibly; control flow goes through callback chains that don't appear in any one function.

**Why it happens.** Pattern lust, or one real need that was over-generalized.

**Fix.** Simplify toward direct, traceable control flow. Ask: can this be a plain function call? A loop over an explicit list? A small dispatch dict?

**Counter-indication.** Established framework integrations (Django models, pytest fixtures, FastAPI routes) earn their magic. Don't fight the framework.

---

## Quick triage table

| Smell                          | Default fix                                | Watch out for                              |
|--------------------------------|---------------------------------------------|--------------------------------------------|
| Boolean / mode-flag creep      | Split / Literal type / config dataclass     | Self-documenting single bool is fine       |
| Dict-shaped domain data        | dataclass / TypedDict — when it clarifies   | External JSON with unknown fields          |
| Hidden side effects            | Surface at boundary / inject / rename       | Don't thread 6 layers of params            |
| Orchestration + impl mixed     | Extract leaf operations                     | Don't extract single-use one-liners        |
| Utility dump module            | Split by domain responsibility              | Avoid circular imports after split         |
| Return shape drift             | Pick one canonical contract                 | Genuinely different domains may differ     |
| Exception drift                | Small domain hierarchy + consistent raise   | Don't merge genuinely different errors     |
| Overgrown class                | Module of functions or smaller objects      | Stateful resources stay as classes         |
| Stateful code without need     | Collapse to pure functions                  | Keep if construction enforces invariants   |
| Under-modeled state            | All state in `__init__` / factory / split   | Some lifecycles are domain-essential       |
| Public API leakage             | `__init__.py` + `__all__` + `_` prefix      | Private monorepo may not need curation     |
| Framework overreach            | Direct control flow                         | Don't fight established frameworks         |
