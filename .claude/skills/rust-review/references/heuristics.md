# Refactoring Heuristics — Full Catalog

Each heuristic includes the smell, why it happens, the typical fix, and counter-indications (when *not* to apply it). Use these to classify findings in the Phase 2 diagnosis and to justify proposals in Phase 3.

## 1. Boolean parameter smell

**Smell.** Functions that take one or more `bool` parameters whose meaning is non-obvious at the call site. `frobnicate(buf, true, false, true)` forces every reader to jump to the signature.

**Why it happens.** Incremental feature additions — each new behavior bolted on as a flag rather than rethinking the entry point.

**Fix.** Choose the *least invasive* option that improves readability:
- Split into separately named functions (`render_strict` vs `render_lenient`).
- Replace with a small enum (`Mode::Strict | Mode::Lenient`).
- Introduce a typed options struct (`RenderOptions { strict: bool, .. }`) — but only when there are ≥3 flags or callers commonly omit defaults.

**Counter-indication.** A single, self-documenting `bool` at a call site like `iter.collect::<Vec<_>>().sort_by(|a, b| ..)` does not need a dedicated type.

## 2. Repeated normalization

**Smell.** Multiple callers each sanitize / validate / convert the same input in slightly different ways. Bugs creep in when one branch normalizes and another doesn't.

**Why it happens.** Callers absorbed the responsibility because the boundary type didn't enforce the invariant.

**Fix.** Move normalization to the constructor or boundary type. Make it impossible to hold an un-normalized value. Newtype wrappers (`NonEmptyString`, `CanonicalPath`) are often the simplest mechanism.

**Counter-indication.** When the "normalization" is actually context-dependent and the variants encode meaningful differences, unification destroys signal.

## 3. Orchestrator-implementation mixing

**Smell.** A function that both coordinates workflow (open → parse → validate → write) and does the low-level work of one of those steps. Hard to test in isolation, hard to swap pieces.

**Why it happens.** Started as one job, accrued sub-tasks over time, no one extracted them.

**Fix.** Extract the leaf operations into their own functions / types. The orchestrator becomes a short, sequential reading of high-level steps.

**Counter-indication.** Don't extract a "step" that is one line and will never be reused — that's just indirection.

## 4. Parallel APIs that drifted

**Smell.** Two or more variants of "do roughly the same thing" with subtly different signatures, error behavior, or default values. Callers pick one by accident.

**Why it happens.** Convergent evolution: someone needed a slight variation, copied the function, then both versions kept evolving.

**Fix.** Pick one canonical form, port the meaningful differences into parameters or a small enum, and remove the duplicate. Deprecate the old name(s) with a `#[deprecated]` migration path if public.

**Counter-indication.** If the variants encode genuinely different domain concepts, unifying them muddles the model.

## 5. Data-model leakage

**Smell.** Callers of an API need to know about internal representation — they reach into fields, depend on internal enum variants, or have to invoke a sequence of methods in the right order.

**Why it happens.** The API exposed concrete types early; refactoring the internals would now break every caller.

**Fix.** Redesign the boundary. Hide the internal type behind a stable surface (newtype, opaque struct, trait object). Provide intent-named methods. Consider sealing the trait if it should not be implemented externally.

**Counter-indication.** Don't add layers solely to "hide" things callers actually need; that's just friction.

## 6. Error fragmentation

**Smell.** Similar failure modes represented many ways — sometimes `Result<T, String>`, sometimes a per-module enum, sometimes a panic. Callers can't write uniform handlers.

**Why it happens.** Each module made a local error decision.

**Fix.** Adopt one error story for the crate. `thiserror`-style enums for libraries, `anyhow`-style for binaries. Use `From` impls for clean propagation. Make panics a *bug signal*, never a control-flow tool in recoverable paths.

**Counter-indication.** A pure FFI boundary may genuinely need a flat error code; respect that boundary.

## 7. Compatibility scar tissue

**Smell.** `if version < 2 { .. } else { .. }`, `// TODO: remove after migration` from years ago, dead enum variants kept "just in case."

**Why it happens.** Migrations land, but the cleanup half doesn't.

**Fix.** First, isolate the scar tissue behind a single function or module. Then, if the migration is provably complete, delete it. Capture the deletion in the commit message so the rationale is preserved.

**Counter-indication.** If callers outside this repo may still depend on the old path, the scar is load-bearing. Confirm before cutting.

## 8. Overgrown modules

**Smell.** A single file or module has multiple, unrelated reasons to change. Its name has become vague (`util.rs`, `helpers.rs`, `common.rs`).

**Why it happens.** "Add it to utils for now" with no later sort.

**Fix.** Split by *responsibility*, not by line count. A 2000-line file with one cohesive purpose is fine; a 300-line `util.rs` with eight unrelated functions is not.

**Counter-indication.** Don't split if the resulting modules would have circular dependencies or near-identical contents — that signals the wrong cut line.

## 9. Premature generality

**Smell.** A trait, generic, or macro that has exactly one implementor / instantiation / call site. The abstraction obscures the single real use case.

**Why it happens.** Speculative design for hypothetical future variants.

**Fix.** Collapse to the concrete case. Generics can be re-introduced when a second real instantiation appears.

**Counter-indication.** If the single use case genuinely makes invalid states unrepresentable (e.g., typestate), the abstraction earns its keep.

## 10. Hidden invariants

**Smell.** Correctness depends on call order, undocumented assumptions, or "you must also remember to do X." Bug magnet.

**Why it happens.** Discipline-by-convention crept in instead of being encoded.

**Fix.** Move the invariant into the type system or a constructor path. Typestate, newtypes, builder patterns, RAII guards, `Drop` impls — pick the lightest tool that makes the invariant unforgettable.

**Counter-indication.** Don't add a typestate ladder for an invariant that's already obvious from the public surface and stable.

---

## Quick triage table

| Smell                          | Default fix                       | Watch out for                            |
|--------------------------------|-----------------------------------|------------------------------------------|
| Boolean parameter              | Enum / split / options struct     | Single self-documenting bool is fine     |
| Repeated normalization         | Newtype / boundary constructor    | Context-dependent variants               |
| Orchestrator + impl mixed      | Extract leaf operations           | Don't extract single-use one-liners      |
| Parallel APIs                  | Pick canonical, deprecate others  | Genuinely different domain concepts      |
| Data-model leakage             | Opaque type + intent-named API    | Adding layers callers don't need         |
| Error fragmentation            | One crate-wide error story        | FFI / pure boundaries                    |
| Compatibility scar             | Isolate, then delete              | External callers may still depend on it  |
| Overgrown module               | Split by responsibility           | Avoid circular deps after split          |
| Premature generality           | Collapse to concrete              | Typestate that enforces real invariants  |
| Hidden invariant               | Encode in types/constructor       | Already-obvious invariants               |
