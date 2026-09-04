"""Feature tests for the raw-dict scale key gate (F-L04-07 completeness half
+ F-L04-10 raw-dict coverage, batch-C task 4).

Batch-C task 4's Rust half (``.sdd/task-4-report.md``) closed the last
documented ``#[serde(flatten)]`` silent-drop carve-out on ``ScaleSpec``: every
raw ``scale={...}`` dict is now validated against a schema-derived, per-type
accepted-key set at ``ScaleSpec``'s own ``Deserialize`` boundary — the single
chokepoint every ``fm.X(...)``/``fm.Y(...)``/``ChartSpec.from_json`` scale
dict passes through, regardless of caller (an UNTYPED raw dict, e.g.
``{"zero": False}`` with no ``"type"`` key, reaches this chokepoint as
``linear`` because ``ferrum.encoding._scale._scale_to_dict`` injects
``{"type": "linear", ...}`` first — see item 7 below, which names that
injection explicitly rather than leaving it implicit). An unknown key (a
typo like ``clammp`` for ``clamp``, or ``reveres`` for ``reverse``) now
refuses, naming the offending key, the scale type, and the sorted
accepted-key list — where it previously vanished silently
(``tests/test_bug_hunt_encoding_step4.py::test_scale_dict_typo_key_is_rejected``
is the flipped positive mirror of that old tolerance pin).

The same Rust change also made a raw-dict temporal domain
(``{"type": "time"/"utc", "domain": [datetime.date(...), ...]}``) convert to
epoch-ms before ``json.dumps`` would otherwise choke on a non-JSON-serializable
``datetime`` object — but only on the chart-level channel path
(``EncodingSpec::new``). A quality-review remediation cycle found and closed
two gate-interaction regressions this module's first cut missed:

- **S4** — ``_spec_build.py``'s bar y-axis zero-anchor used to stamp
  ``"zero": True`` onto ANY y-channel scale dict regardless of type. Before
  the gate, that was a harmless no-op for every non-linear type (``zero``
  isn't a field on ``ScaleSpec::Log``/``Symlog``/``Sqrt``/``Band``/etc., so
  serde's flatten silently dropped it); after the gate, it refused outright
  — a real regression this module's original "no over-refusal" sweep could
  not see, because that sweep only exercised hand-written literals, never a
  dict as ferrum's own producers actually emit one (item 8 below closes
  that blind spot generically; ``tests/test_bar_zero.py`` carries the
  full, mark-specific regression suite).
- **S3** — the raw-dict temporal-domain conversion (item 5 below) only ran
  on the chart-level channel path (Rust's ``EncodingSpec::new`` hook). A
  ``Layer``/composite-mark channel bypasses that constructor entirely
  (``coerce_layers`` -> ``pyo3_serde::from_py`` json-dumps the dict
  directly), so the identical ``scale={"type": "time", "domain":
  [date(...), ...]}`` that rendered on a bare chart crashed with an opaque
  ``TypeError: Object of type date is not JSON serializable`` on
  ``chart_a + chart_b`` or any composite mark. Fixed at the ONE Python seam
  both routes share — ``ferrum.encoding._scale._scale_to_dict`` — since
  both ``_spec_build.py``'s chart-level and per-layer encoding-dict builders
  call the identical ``ChannelBase.to_encoding_spec_dict()`` method, which
  dispatches ``scale=`` through ``_scale_to_dict`` either way (item 9
  below proves the layer path directly).

Covers:
  1. Unknown-key refusal: the typo case names the real key among the sorted
     accepted list, at both wire-boundary entry points (``ChartSpec.from_json``
     and the ``EncodingSpec::new`` constructor path), and generalizes across
     all 16 known ``ScaleSpec`` types (not just ``linear``).
  2. A valid-keys sweep: every accepted key, for every scale type, populated
     in one hand-authored raw dict, still parses and builds a ``ChartSpec``.
     This proves the gate accepts every key IT ENUMERATES for a type — it is
     necessarily silent on a legal shape the hand-authored fixture omits (see
     item 8, which closes that specific blind spot with producer-emitted
     dicts instead of literals).
  3. ``reverse`` accepted (no refusal) via raw dict on all seven continuous
     types, including an end-to-end render-order flip proof for ``utc``
     (the one continuous type with no dedicated Python scale class, so
     ``tests/test_scale_reverse.py``'s class-based sweep cannot reach it).
  4. ``{"type": "diverging", "reverse": true}`` refused (no such field) —
     the third silent-no-op case the gate closes, named explicitly.
  5. Raw-dict temporal domains (``{"type": "time"/"utc", "domain":
     [datetime.date(...), ...]}``) render byte-identical SVG to the
     epoch-float equivalent — the raw-dict path specifically, as distinct
     from ``tests/test_timescale_domain.py``'s class-constructor path (T3),
     which the raw-dict conversion added by this task does not go through.
  6. A non-temporal domain element on a raw-dict time/utc scale refuses,
     naming the accepted forms.
  7. The untyped raw-dict spelling (no ``"type"`` key at all) — the most
     common one in the codebase — reaches the gate as ``linear`` via the
     ``_scale_to_dict`` injection named above, both for a legal key
     (``{"zero": False}``) and a typo (``{"clammp": True}``, refused naming
     ``linear``'s accepted list).
  8. A producer-emitted-dict arm in the "no over-refusal" sweep: dicts built
     by ``_scale_to_dict``/a real ``Chart.to_spec()`` call, not hand-written
     literals — the shape of coverage that would have caught the S4 bar
     zero-anchor regression inside this module, not just in
     ``tests/test_bar_zero.py``.
  9. Raw-dict temporal domains render on the LAYER path (``chart.layer(...)``
     and ``chart_a + chart_b``), not just the chart-level channel path — the
     S3 regression above.

**Round 3** (a second quality-review remediation cycle) closed one
RECURRING S4 finding and two smaller ones at the same edited block:

- **S4, recurring** — ``_spec_build.py``'s override-scale merge
  (``{**base_scale, **scale_overrides}``) still emitted gate-refused dicts
  whenever ``Chart.override(<channel>_scale_type=...)`` switched the
  effective type over an EXPLICIT base scale — e.g. ``fm.LinearScale()``'s
  own emitted dict carries ``"zero": False`` (a real ``ScaleSpec::Linear``
  field, always serialized), which survived unfiltered onto a
  ``.override(y_scale_type="log")`` switch and got refused. Round 3 fixed
  this with a Python mirror of Rust's ``ContinuousScaleCommon`` flatten-key
  set (``_spec_build._CONTINUOUS_COMMON_SCALE_KEYS``); item 11 below is the
  regression coverage that mirror's fix earned.
- **S3, exception drift** — moving temporal-domain conversion into Python
  (item 5/9 above) meant a bad ISO string in a scale domain reported from
  ``ferrum.annotation.coords``'s "annotation coordinate" vocabulary instead
  of a scale-domain one. Item 12 below pins the fix (re-raised in Rust's
  own ``TimeScale domain value`` wording, though NOT byte-identical to it —
  see item 12's own note) alongside the sibling non-ISO-junk-element case,
  which was already coherent and unchanged.
- The ``mark_bar()`` + ``.override(<y>_scale_domain=...)`` behavior change
  from round 2's reorder (zero-anchor no longer widens an
  override-supplied domain) is ADJUDICATED KEPT — item 13 below pins it
  with the sharper equals-explicit-zero-False / differs-from-explicit-
  zero-True repro pair rather than a tick-label assertion, and flags it
  for a changelog callout (T8).

  11. Regression coverage for the recurring S4 finding: every one of the
      reviewer's repro spellings (class, typed-dict, untyped-dict), both
      directions of the type switch (linear->log and log->linear), and two
      marks (bar and point, since the bug is not bar-specific) — plus a
      same-type-override control proving the filter does NOT over-drop a
      variant-specific key when the type isn't actually changing.
  12. The ISO-string and non-ISO-junk sibling failures now read one
      coherent ``TimeScale domain value`` vocabulary regardless of which
      subsystem (Python or Rust) catches them — NOT byte-identical wording
      (Rust formats the offending value with ``{s:?}``, double quotes;
      Python with ``{value!r}``, single quotes), only the surrounding
      sentence.
  13. ``mark_bar()`` + ``.override(y_scale_domain=...)`` matches the
      explicit ``zero: False`` spelling and differs from the explicit
      ``zero: True`` spelling — the deliberate, disclosed round-2 behavior
      change.

**Round 4** (a third quality-review remediation cycle) replaced round 3's
``_CONTINUOUS_COMMON_SCALE_KEYS`` mechanism outright, rather than patching
it, because both reviewers converged on the same root cause: filtering
``base_scale`` against a hand-maintained INTERSECTION of the 7 continuous
types' accepted keys is simultaneously too narrow (it drops a key the
*target* type genuinely accepts — ``nice`` is on ``linear``/``log``/
``time``/``symlog``/``utc`` but not in their intersection, since it's
absent from ``pow``/``sqrt``, so a ``linear(nice=True)`` -> ``log`` switch
silently lost ``nice``) and does nothing at all for a continuous <->
non-continuous switch (``band``/``point``/``sequential``/etc., which share
no ``ContinuousScaleCommon``-equivalent struct with anything, so round 3's
filter never even fired for them and they kept hard-raising on shapes that
rendered pre-gate).

``_merge_override_scale`` (``src/ferrum/_spec_build.py``) now filters
``base_scale`` against ``ferrum._core.scale_accepted_keys(new_type)`` — the
gate's OWN per-type accepted-key table, published from Rust
(``crates/ferrum-core/src/spec/encoding.rs``'s ``accepted_keys_for_scale_type``,
already this task's own Rust half) — for ANY type-changing switch, not just
a continuous-to-continuous one. A key the new type accepts survives; a key
it doesn't is dropped, in both cases matching pre-gate behavior (serde's
flatten either kept or silently dropped the key, depending on whether the
target variant declared it). ``_CONTINUOUS_SCALE_TYPES``,
``_CONTINUOUS_COMMON_SCALE_KEYS``, and the cross-language guard test that
existed only to police that mirror are deleted — there is nothing left to
mirror once Python asks Rust directly.

  11b. Survival coverage: a key both the old and new type accept
       (``nice``, on a ``linear``->``log`` and a ``time``->``utc`` switch)
       must SURVIVE the type-changing override, using the same
       equals-explicit / differs-from-explicit-omitted shape as item 13.
       Round 3's fix failed this pair silently (no test caught it).
  11c. Mixed-family switch coverage: every repro from both cycle-3
       verdicts (``point``<->``band``, ``band``<->``point``,
       ``band``->``log``, ``log``->``band``, ``point``->``linear``,
       ``linear``->``band``, and ``Color(LinearScale())``->``sequential``)
       renders and drops exactly the stale key named in each verdict.
       Round 3's fix could not reach any of these (its gate required BOTH
       sides to be in the 7-member continuous family).
  12b. A live cross-language guard for item 12's ISO-parse message (there
       was none before round 4): Rust's wording is triggered through the
       real ``ferrum._core.EncodingSpec`` PyO3 constructor directly (not a
       hand-built mirror), and compared against Python's own message with
       the one known quoting difference normalized — so a future drift in
       either side's wording fails this test instead of silently
       diverging.

**Round 5** (a fourth quality-review remediation cycle) closed the last gap
in ``_merge_override_scale``: round 4's filter only ever asked whether the
NEW type accepts a base-scale key, never whether the OLD type did. A key
outside the OLD type's accepted set is not a legitimate leftover a type
switch is entitled to carry — it is a typo (or a plain-inapplicable key)
the gate would already refuse with no override at all. Two live shapes
proved the gap:

- a typo'd key (``clammp``) that happens to be invalid for BOTH the old
  and the new type silently vanished behind a type-changing override,
  restoring the exact silent-drop carve-out this whole task exists to
  close;
- sharper: a key invalid for the OLD type but that coincidentally names a
  REAL field of the NEW type (``nice`` is not a ``band`` field, but IS a
  ``log`` field) got silently PROMOTED to an effective setting after the
  switch — worse than the pre-gate baseline, where ``band``'s flatten just
  dropped it.

``_merge_override_scale`` now validates ``base_scale``'s own keys against
``scale_accepted_keys(old_type)`` FIRST, and refuses (matching the gate's
own non-switch-path refusal) before any type-switch filtering ever runs
— see item 14 below for both regression pins, plus the case-A no-override
control asserted alongside each. A second, independent gap in the SAME
helper: a non-``str`` ``<channel>_scale_type=`` override value (e.g. a user
writing ``y_scale_type=fm.LogScale()`` by mistake, reaching for
``y_scale=``) used to reach ``scale_accepted_keys`` directly whenever the
channel carried an explicit base scale, and PyO3's own argument-coercion
error surfaced instead of the gate's own message — see item 15 below.

  14. Old-type-validity regression pins: the typo-escapes-via-override case
      (case B) and the sharper invalid-key-gets-promoted case (case C),
      each asserted against the no-override control (case A) to prove the
      override path now refuses identically rather than divergently.
  15. Non-string override-type regression pins: the PyO3 argument-coercion
      error must not leak through an explicit base scale (regardless of
      which downstream error the malformed value ultimately produces), an
      unknown-string override type over an explicit base scale still
      surfaces the gate's own "unknown variant" refusal (round 4's
      fallback branch, previously executed by zero tests), and a
      non-string spelling now produces the identical error class with and
      without an explicit base scale on the channel.

**Round 6** (a fifth quality-review remediation cycle) closed the last two
gaps at ``_merge_override_scale``, both at the old-type-validity check round
5 added:

- Round 5's ``isinstance(new_type, str)`` guard protected only the NEW-type
  ``scale_accepted_keys`` call. The OLD-type call one branch earlier had no
  such guard, so a non-string base ``"type"`` (an int, a float, ``bytes``,
  or a scale pyclass instance under ``"type"`` by mistake) behind a
  type-changing override raised PyO3's raw ``TypeError: argument
  'scale_type': ...`` instead of whatever the real gate raises for that
  base standalone — the exact leak round 5's own item-15 tests forbid, on
  the sibling call those tests never parametrized.
- Round 5's docstring claimed an unknown old-type tag (``{"type":
  "linearr"}``) "falls through unfiltered, since the gate will refuse the
  base type's own tag downstream regardless" — false on a type-changing
  override, because the override supplies its OWN ``"type"`` and the base's
  invalid tag never reaches the wire on that spelling; the user's actual
  mistake would go unreported.

Rather than patch both gaps with more per-call ``isinstance`` guards (a
growing list of patched call sites), round 6 replaces the WHOLE
old-type-validity check with a probe of the real gate:
``ferrum._core.EncodingSpec("<override-scale-probe>", scale=base_scale)``
validates ``base_scale`` AS-AUTHORED — before any filtering or merging —
through the exact production deserialize path, for ANY type-changing
override. This is preferred over a hand-synthesized message (round 5's
``f"unknown key '{k}' for type '{t}'; accepted: ..."`` literal, now
deleted) because it gives byte-true gate messages for every invalid-base
shape at once — bad key, unknown type tag, non-string/non-hashable type
value — with no drift guard to maintain, since it IS the gate raising it,
not a Python copy of its wording. It is a cold path (only type-changing
overrides reach it), so the extra ``EncodingSpec`` construction has no
render-path cost. Once the probe passes, ``old_type`` is guaranteed to be
a real ``ScaleSpec`` variant tag and every one of ``base_scale``'s keys is
valid for it, so the type-switch filtering in part 2 of the docstring can
call ``scale_accepted_keys(old_type)`` unconditionally.

A companion Python-side gap closed in the same round: ``_scale_to_dict``
(``src/ferrum/encoding/_scale.py``) tested a user-supplied ``"type"``
value for frozenset membership with no type guard, so an unhashable
``"type"`` (a scale pyclass instance) raised ``TypeError: unhashable
type: ...`` from inside a temporal-conversion helper — a message naming
no ferrum concept, on a line this task's own cycle-2 diff added (not
pre-existing, despite round 5's report attributing it to a pre-task
cause). Guarding the membership test with ``isinstance(..., str)``
restores the pre-task JSON-serialization ``TypeError`` shape.

  16. Old-type non-string parity (every malformed ``"type"`` value from
      item 15's list, now on the BASE side): the override spelling and the
      no-override control must raise the identical exception type and
      message. The unknown-old-type-tag case (round 5's false docstring
      claim): a type-changing override over a typo'd base type must refuse
      identically to the no-override control, not fall through. The
      ``_scale_to_dict`` membership-check guard: a scale pyclass instance
      under a raw dict's ``"type"`` key produces the pre-task
      JSON-serialization ``TypeError``, not ``unhashable type``.

**Round 7** (a sixth quality-review remediation cycle) closed one recurring
gap and one over-refusal, both at ``_merge_override_scale``:

- **Recurring — the old-type validity check's own entry guard laundered
  ``{"type": None}``.** ``old_type is None`` conflated "no scale on the
  channel at all" (``base_scale == {}``, which correctly must keep
  short-circuiting) with "the channel's scale explicitly claims ``"type":
  None``" (a real, present claim — ``scale={"type": maybe_type}`` where
  ``maybe_type`` resolved to ``None`` is not an exotic spelling), so the
  latter's type-changing override skipped the round-6 probe entirely and
  rendered — including a bad key getting silently PROMOTED to an effective
  new-type setting, the exact outcome the probe exists to refuse. Fixed by
  keying the guard on ``"type" not in base_scale`` instead: item 16's
  parametrization gains ``("none:None", None)``.
- **Over-refusal — the round-6 probe validated base_scale's VALUES, not
  just its tag and keys, under a type the switch is actively replacing.**
  A key both the old and new type accept (most commonly ``domain``) got
  refused whenever its value only type-checked under the NEW type —
  ``{"domain": ["a","b","c"]}`` stamped ``"linear"`` by ``_scale_to_dict``'s
  untyped-dict default, switched to ``"band"``, used to refuse ``invalid
  type: string "a", expected f64`` even though the final, merged dict is a
  perfectly legal ``band`` scale. Measured blast radius (240 ordered
  type-pair sweep, base populated with every key the old type accepts): 48
  of 240 pairs flipped from render (round 4/5 mechanism) to refuse (round
  6), 0 newly accepted. Fixed by splitting the single whole-dict probe into
  two narrower ones — a tag-only probe (``{"type": old_type}``) plus a
  key-membership probe over ONLY the keys ``old_type`` doesn't recognize
  (``unknown_under_old`` in the source) — so a key ``old_type`` DOES
  recognize is never value-validated here at all; its value reaches
  ``new_type``'s own downstream validation on the final merged dict,
  exactly as production always ran for a key that survives the filter.
  Item 17 below is the regression coverage for this, in the
  producer-composition shape (``_scale_to_dict``'s untyped-dict default
  composed with a type-changing override), not isolated literals.

Both fixes are shape-preserving for every previously-closed case: the
tag-only probe still raises byte-true gate messages for a non-string/
non-hashable/unknown-tag ``old_type`` (round 6, item 16); the
key-membership probe still refuses the two round-5 "old-type validity"
regressions (item 14's case B ``clammp`` and case C ``nice``-not-a-``band``-
field, both keys the OLD type does not recognize either way).

  17. Producer-composition regression coverage for the round-7 over-refusal
      fix: an untyped raw-dict scale (``_scale_to_dict``'s ``"linear"``
      default) whose ``domain`` is only legal under the NEW type, composed
      with a type-changing ``Chart.override(<channel>_scale_type=...)``,
      renders and carries the domain through — parametrized over
      ``x_scale_type="band"``/``"point"``/``"ordinal"`` (a string domain)
      and ``y_scale_type="time"`` (an ISO-string domain, asserting the
      epoch-ms conversion still runs), plus the explicit-``"type":"linear"``
      spelling of the same shape.

**Round 8** (a narrow, single-predicate closing round) closed the last of
the three buckets ``_merge_override_scale``'s key partition could produce: a
key ``old_type`` accepts but ``new_type`` doesn't — the ``drop =
accepted_old - accepted_new`` set round 4 introduced — was silently
dropped by the type-switch filter with NO validation anywhere: not by
``unknown_under_old``'s probe (the key IS a member of ``accepted_old``, so
it never enters that set), and not by ``new_type``'s own downstream gate
(the key is gone from the merged dict by the time that gate runs). A key
in this bucket whose VALUE is invalid under ``old_type`` therefore rendered
silently behind a type-changing override, even though the same base scale
refused standalone — e.g. ``{"type": "linear", "zero": "yes"}`` (an
invalid ``bool`` value) + ``.override(y_scale_type="log")`` rendered,
because ``zero`` is a ``linear`` field, not a ``log`` one, so the filter
dropped it before anyone checked it was garbage. Fixed by probing
``base_scale.keys() & (accepted_old - accepted_new)`` under ``old_type``,
the same shape as ``unknown_under_old``'s probe, before dropping those
keys — so every key of ``base_scale`` is now validated exactly once: under
``old_type`` if it is dropped or unknown, under ``new_type`` if it
survives. Verified this touches none of round 7's 48 restored survival
pairs: a survivor is, by definition, accepted by BOTH types, so it is
never a member of ``drop`` and is never sent to this probe.

  18. Dropped-bucket regression coverage: a key accepted by ``old_type``
      but not ``new_type``, whose value is invalid under ``old_type``,
      must refuse identically with and without the type-changing override
      — parametrized over three repro shapes (``zero`` invalid on
      ``linear``, switched to both ``log`` and ``band``; ``base`` invalid
      on ``log``, switched to ``band``; ``align`` invalid on ``band``,
      switched to ``linear``) — plus a control proving a SURVIVING key
      (accepted by both types) is untouched by the new probe.

RED-proof note (discriminating by construction, not a toggled runtime
check): before this batch's Rust half landed, ``ScaleSpec``'s internally
tagged, ``#[serde(flatten)]``-based deserialize had no way to enforce
``deny_unknown_fields`` — a typo'd scale key silently round-tripped as a
no-op. ``tests/test_bug_hunt_encoding_step4.py``'s now-flipped test proved
this directly: it asserted ``ChartSpec.from_json(...)`` did NOT raise for a
``clammp`` typo and that the typo'd key was silently dropped from the
round-trip. Every refusal test below is the exact positive mirror of that
prior assertion — non-vacuously RED against any pre-gate build by
construction, since the assertions here (``pytest.raises`` where the old
test asserted no raise) fail outright without the gate. Items 8 and 9 above
are separately RED-proofed against the actual pre-remediation code in this
cycle's own history (not simulated): the S4 bar+``LogScale`` case and the
layer-path temporal-date case both raised on the working tree before the
``_spec_build.py``/``_scale.py`` fixes in this cycle landed, confirmed by
direct reproduction before writing the fix (see the coordinator's
remediation-cycle record for the exact repro commands).
"""

from __future__ import annotations

import calendar
import datetime as dt
import json

import polars as pl
import pytest

import ferrum as fm
from ferrum._core import ChartSpec
from tests._svg_extents import axis_tick_labels

# ---------------------------------------------------------------------------
# Schema-derived accepted-key sets (mirrors
# ``crates/ferrum-core/src/spec/encoding.rs::accepted_keys_for_scale_type``,
# whose own drift guard
# (``accepted_keys_for_scale_type_matches_every_variants_serialized_keys``)
# proves this hand-maintained Rust set matches every ``ScaleSpec`` variant's
# real serialized keys). Sorted the same way the gate's own error message
# sorts them (``sort_unstable`` on ``&str`` — plain lexicographic order,
# matching Python's default string sort for ASCII).
# ---------------------------------------------------------------------------

ACCEPTED_KEYS: dict[str, list[str]] = {
    "linear": [
        "clamp",
        "domain",
        "domainParam",
        "nice",
        "padding",
        "range",
        "reverse",
        "scheme",
        "zero",
    ],
    "log": [
        "base",
        "clamp",
        "domain",
        "domainParam",
        "nice",
        "padding",
        "range",
        "reverse",
        "scheme",
    ],
    "time": ["clamp", "domain", "domainParam", "nice", "padding", "range", "reverse", "scheme"],
    "symlog": [
        "clamp",
        "constant",
        "domain",
        "domainParam",
        "nice",
        "padding",
        "range",
        "reverse",
        "scheme",
    ],
    "pow": ["clamp", "domain", "domainParam", "exponent", "padding", "range", "reverse", "scheme"],
    "sqrt": ["clamp", "domain", "domainParam", "padding", "range", "reverse", "scheme"],
    "utc": ["clamp", "domain", "domainParam", "nice", "padding", "range", "reverse", "scheme"],
    "ordinal": ["domain", "padding", "range"],
    "band": ["align", "domain", "padding", "paddingInner", "paddingOuter", "range"],
    "point": ["align", "domain", "padding", "range", "reverse"],
    "sequential": ["domain", "reverse", "scheme", "stops"],
    "diverging": ["domain", "domainMid", "scheme"],
    "quantize": ["domain", "range"],
    "quantile": ["domain", "range"],
    "threshold": ["domain", "range"],
    "bin-ordinal": ["bins", "scheme"],
}

# One raw dict per type with EVERY accepted key populated (the valid-keys
# sweep, item 2 above). These are hand-authored wire shapes, not derived from
# a Python ``*Scale`` class's ``_to_scale_spec_dict()`` — several accepted
# keys (e.g. ``linear``'s ``zero``) are raw-dict-only spellings with no
# constructor kwarg on the corresponding class at all (``LinearScale`` has no
# ``zero=`` parameter; ``{"zero": False}`` is a real, currently-legal
# ``tests/test_bar_zero.py`` shape reachable only as a raw dict), so a
# class-derived sweep would under-cover the gate's own accepted-key sets.
VALID_DICTS: dict[str, dict[str, object]] = {
    "linear": {
        "type": "linear",
        "domain": [0, 10],
        "range": [0, 100],
        "clamp": True,
        "padding": 1.0,
        "reverse": True,
        "nice": True,
        "zero": True,
        "scheme": "viridis",
        "domainParam": "unused_param_name",
    },
    "log": {
        "type": "log",
        "domain": [1, 100],
        "range": [0, 100],
        "clamp": True,
        "padding": 1.0,
        "reverse": True,
        "nice": True,
        "base": 2.0,
        "scheme": "viridis",
        "domainParam": "unused_param_name",
    },
    "time": {
        "type": "time",
        "domain": [0, 1000],
        "range": [0, 100],
        "clamp": True,
        "padding": 1.0,
        "reverse": True,
        "nice": True,
        "scheme": "viridis",
        "domainParam": "unused_param_name",
    },
    "symlog": {
        "type": "symlog",
        "domain": [-10, 10],
        "range": [0, 100],
        "clamp": True,
        "padding": 1.0,
        "reverse": True,
        "nice": True,
        "constant": 2.0,
        "scheme": "viridis",
        "domainParam": "unused_param_name",
    },
    "pow": {
        "type": "pow",
        "domain": [0, 10],
        "range": [0, 100],
        "clamp": True,
        "padding": 1.0,
        "reverse": True,
        "exponent": 3.0,
        "scheme": "viridis",
        "domainParam": "unused_param_name",
    },
    "sqrt": {
        "type": "sqrt",
        "domain": [0, 10],
        "range": [0, 100],
        "clamp": True,
        "padding": 1.0,
        "reverse": True,
        "scheme": "viridis",
        "domainParam": "unused_param_name",
    },
    "utc": {
        "type": "utc",
        "domain": [0, 1000],
        "range": [0, 100],
        "clamp": True,
        "padding": 1.0,
        "reverse": True,
        "nice": True,
        "scheme": "viridis",
        "domainParam": "unused_param_name",
    },
    "ordinal": {"type": "ordinal", "domain": ["a", "b", "c"], "range": [0, 100], "padding": 0.1},
    "band": {
        "type": "band",
        "domain": ["a", "b", "c"],
        "padding": 0.2,
        "paddingInner": 0.1,
        "paddingOuter": 0.3,
        "align": 0.6,
        "range": [0, 100],
    },
    "point": {
        "type": "point",
        "domain": ["a", "b", "c"],
        "padding": 0.3,
        "align": 0.4,
        "reverse": True,
        "range": [0, 100],
    },
    "sequential": {
        "type": "sequential",
        "scheme": "viridis",
        "domain": [0, 1],
        "reverse": True,
        "stops": [[0.0, "#ff0000"], [1.0, "#00ff00"]],
    },
    "diverging": {"type": "diverging", "scheme": "redblue", "domain": [-1, 0, 1], "domainMid": 0.0},
    "quantize": {"type": "quantize", "domain": [0, 100], "range": ["low", "mid", "high"]},
    "quantile": {"type": "quantile", "domain": [1.0, 2.0, 3.0, 4.0], "range": [0.0, 0.5, 1.0]},
    "threshold": {"type": "threshold", "domain": [0.5, 1.0], "range": [0.0, 0.5, 1.0]},
    "bin-ordinal": {"type": "bin-ordinal", "bins": [0, 10, 20, 30], "scheme": "blues"},
}

# Types that resolve positionally (attach to x); the rest are color-channel
# scales (categorical/binned color mapping) and are exercised on Color
# instead, matching how each type is actually used in the wider suite (e.g.
# ``tests/test_flexibility_campaign.py``'s ordinal dict-form color test,
# ``tests/test_phase_12_scales.py::TestScaleIntegration``'s Sequential-on-
# Color case).
POSITIONAL_TYPES = {
    "linear",
    "log",
    "time",
    "symlog",
    "pow",
    "sqrt",
    "utc",
    "ordinal",
    "band",
    "point",
}
COLOR_TYPES = {"sequential", "diverging", "quantize", "quantile", "threshold", "bin-ordinal"}
assert POSITIONAL_TYPES | COLOR_TYPES == set(ACCEPTED_KEYS)

_ALL_TYPES = sorted(ACCEPTED_KEYS)


def _base_spec_json() -> dict:
    return json.loads(ChartSpec(mark="point", x="a", y="b").to_json())


def _accepted_suffix(scale_type: str) -> str:
    """The gate's pinned ``"accepted: ..."`` suffix for one scale type."""
    return "accepted: " + ", ".join(ACCEPTED_KEYS[scale_type])


# ---------------------------------------------------------------------------
# 1. Unknown-key refusal
# ---------------------------------------------------------------------------


def test_typo_key_refused_names_key_type_and_accepted_list() -> None:
    """``clammp`` (typo of ``clamp``) on a ``linear`` scale refuses, naming
    the bad key, the scale type, and the full sorted accepted-key list —
    the spec §9 repro this gate exists to close.

    Verbatim-substring pins (not a loose ``match=``) per the batch-B
    mutation-testing lesson: a gate that named the wrong key, the wrong
    type, or a subtly-wrong accepted list would still satisfy a loose regex
    but fail these exact substring checks.
    """
    j = _base_spec_json()
    j["encoding"]["x"]["scale"] = {"type": "linear", "clammp": True}
    with pytest.raises(ValueError) as exc_info:
        ChartSpec.from_json(json.dumps(j))
    message = str(exc_info.value)
    assert "unknown key 'clammp'" in message
    assert "for type 'linear'" in message
    assert _accepted_suffix("linear") in message


def test_reveres_typo_of_reverse_refused() -> None:
    """``reveres`` (typo of ``reverse``) refuses, naming ``reverse`` among
    the accepted keys — the finding's own motivating example.
    """
    j = _base_spec_json()
    j["encoding"]["x"]["scale"] = {"type": "linear", "reveres": True}
    with pytest.raises(ValueError) as exc_info:
        ChartSpec.from_json(json.dumps(j))
    message = str(exc_info.value)
    assert "unknown key 'reveres'" in message
    assert "for type 'linear'" in message
    assert "reverse" in message


def test_typo_key_refused_via_encoding_constructor_path() -> None:
    """The same typo refuses through the OTHER wire chokepoint entry point:
    ``fm.X(..., scale=...)`` building a ``ChartSpec`` via ``EncodingSpec::new``,
    not just ``ChartSpec.from_json``.

    Both paths bottom out at the identical ``ScaleSpec::deserialize`` call
    (see ``.sdd/task-4-report.md``'s "chokepoint" section), so a gate placed
    anywhere else could plausibly cover one path and silently miss the
    other. This proves both are covered. Substring-only (not the exact
    ``"scale: scale: unknown key ..."`` double-prefixed message the
    ``json_round`` field-error wrapper produces on this specific path) —
    that cosmetic double prefix is a documented, harmless quirk (task-4
    report, "Concerns" #3), not part of this test's contract.
    """
    df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("a", scale={"type": "linear", "clammp": True}), y="b")
    )
    with pytest.raises(ValueError) as exc_info:
        chart.to_spec()
    message = str(exc_info.value)
    assert "unknown key 'clammp'" in message
    assert "for type 'linear'" in message


@pytest.mark.parametrize("scale_type", _ALL_TYPES)
def test_unknown_key_refused_for_every_type(scale_type: str) -> None:
    """A bogus key refuses for every one of the 16 known scale types, each
    naming that type's own exact sorted accepted-key list.

    Generalizes the ``linear``-only typo case above across the full type
    taxonomy — the gate is schema-derived per type, so each type's accepted
    list must be independently correct, not just the one spec-§9 example.
    """
    j = _base_spec_json()
    j["encoding"]["x"]["scale"] = {"type": scale_type, "boguskey": True}
    with pytest.raises(ValueError) as exc_info:
        ChartSpec.from_json(json.dumps(j))
    message = str(exc_info.value)
    assert f"unknown key 'boguskey' for type '{scale_type}'" in message
    assert _accepted_suffix(scale_type) in message


def test_unknown_type_falls_through_to_variant_error_not_gate_message() -> None:
    """An unrecognized ``"type"`` value gets serde's own "unknown variant"
    error, not a gate-authored message — the gate is deliberately
    permissive on shapes it cannot judge (see
    ``validate_scale_spec_keys``'s doc).
    """
    j = _base_spec_json()
    j["encoding"]["x"]["scale"] = {"type": "bogus-scale-type", "clammp": True}
    with pytest.raises(ValueError, match="unknown variant"):
        ChartSpec.from_json(json.dumps(j))


# ---------------------------------------------------------------------------
# 2. Valid-keys sweep: every key the gate itself enumerates is accepted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale_type", _ALL_TYPES)
def test_every_accepted_key_for_every_type_still_parses(scale_type: str) -> None:
    """Every accepted key, for every scale type, populated together in one
    HAND-AUTHORED raw dict, still builds a ``ChartSpec`` without error.

    Scoped claim (quality-review remediation): this proves the gate accepts
    every key its OWN ``ACCEPTED_KEYS``/``VALID_DICTS`` mirror enumerates —
    it is a near-circular check (``VALID_DICTS`` is authored from
    ``ACCEPTED_KEYS``, enforced by the fixture-integrity assertion below) and
    cannot detect over-refusal of a legal shape neither list happens to
    include, or of a dict as a PRODUCER (not a human) actually assembles —
    that is exactly how the S4 bar-zero-anchor regression slipped past this
    sweep (a real bug: ``_spec_build.py`` stamped a ``zero`` key onto every
    bar y-scale regardless of type, refused for every non-linear type once
    the gate landed). ``test_producer_emitted_scale_dicts_do_not_over_refuse``
    below closes that specific gap with producer-emitted dicts instead of
    literals.
    """
    scale = VALID_DICTS[scale_type]
    assert set(scale) - {"type"} == set(ACCEPTED_KEYS[scale_type]), (
        f"{scale_type}: VALID_DICTS fixture must populate every accepted key exactly once"
    )
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "val": [1.0, 2.0, 3.0]})
    if scale_type in POSITIONAL_TYPES:
        chart = fm.Chart(df).mark_point().encode(x=fm.X("x", scale=scale), y="val")
    else:
        chart = fm.Chart(df).mark_point().encode(x="x", y="val", color=fm.Color("val", scale=scale))
    spec = chart.to_spec()  # must not raise
    assert spec is not None


# ---------------------------------------------------------------------------
# 3. reverse accepted via raw dict on all 7 continuous types
# ---------------------------------------------------------------------------

_CONTINUOUS_TYPES = ["linear", "log", "time", "symlog", "pow", "sqrt", "utc"]


@pytest.mark.parametrize("scale_type", _CONTINUOUS_TYPES)
def test_reverse_accepted_via_raw_dict_on_every_continuous_type(scale_type: str) -> None:
    """``{"type": <continuous>, "reverse": true}`` is accepted (no refusal)
    for all 7 continuous types, including ``utc`` — the one continuous type
    with no dedicated Python scale class
    (``tests/test_scale_reverse.py``'s class-based sweep covers the other
    six directly via ``fc.<X>Scale(reverse=True)`` but cannot reach ``utc``,
    which is only ever a raw-dict ``"type"`` tag or ``TimeScale(utc=True)``).
    """
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    domain = [-10, 10] if scale_type == "symlog" else [1, 100]
    scale = {"type": scale_type, "domain": domain, "reverse": True}
    chart = fm.Chart(df).mark_point().encode(x=fm.X("x", scale=scale), y="y")
    assert chart.to_spec() is not None  # must not raise


def test_utc_reverse_via_raw_dict_flips_rendered_tick_order() -> None:
    """``{"type": "utc", "reverse": true}`` doesn't just parse — it actually
    flips rendered tick-label order, mirroring
    ``tests/test_scale_reverse.py``'s discriminating per-class assertions
    for the one continuous type that module's class-based sweep cannot
    reach (see above).
    """

    def _svg(reverse: bool) -> str:
        df = pl.DataFrame({"x": [0.0, 1_000_000.0], "y": [0.0, 100.0]})
        scale = {"type": "utc", "domain": [0.0, 1_000_000.0], "nice": False, "reverse": reverse}
        chart = fm.Chart(df).mark_point().encode(x=fm.X("x", scale=scale), y="y")
        return chart.to_svg()

    forward = axis_tick_labels(_svg(False), axis="x")
    reversed_ = axis_tick_labels(_svg(True), axis="x")
    assert len(forward) >= 3, "too few forward tick labels to be discriminating"
    assert reversed_ == list(reversed(forward)), "reverse=True must flip utc tick order end to end"


# ---------------------------------------------------------------------------
# 4. diverging + reverse refused
# ---------------------------------------------------------------------------


def test_diverging_reverse_refused_with_accepted_list() -> None:
    """``{"type": "diverging", "reverse": true}`` refuses — ``Diverging`` has
    no ``reverse`` field (T1's third, more-silent carve-out this gate
    closes: without the gate, ``reverse`` on a ``Diverging`` scale would not
    even round-trip, let alone raise).
    """
    j = _base_spec_json()
    j["encoding"]["x"]["scale"] = {"type": "diverging", "reverse": True}
    with pytest.raises(ValueError) as exc_info:
        ChartSpec.from_json(json.dumps(j))
    message = str(exc_info.value)
    assert "unknown key 'reverse' for type 'diverging'" in message
    assert _accepted_suffix("diverging") in message


# ---------------------------------------------------------------------------
# 5. Raw-dict temporal domain renders identically to epoch-float equivalent
# ---------------------------------------------------------------------------


def _epoch_ms(d: dt.date) -> float:
    """Independent (non-ferrum) UTC epoch-ms conversion for a naive date, so
    the render-identity assertions below don't depend on the very converter
    (``temporal_value_to_epoch_ms``) whose output they're checking.
    """
    return calendar.timegm(d.timetuple()) * 1000.0


def _svg_for_raw_dict_domain(domain: list[object], *, scale_type: str) -> str:
    lo_ms, hi_ms = _epoch_ms(dt.date(2021, 3, 1)), _epoch_ms(dt.date(2021, 3, 2))
    df = pl.DataFrame({"x": [lo_ms, hi_ms], "y": [0.0, 100.0]})
    scale = {"type": scale_type, "domain": domain, "nice": False}
    chart = fm.Chart(df).mark_point().encode(x=fm.X("x", scale=scale), y="y")
    return chart.to_svg()


@pytest.mark.parametrize("scale_type", ["time", "utc"])
def test_raw_dict_date_domain_renders_identically_to_epoch_float(scale_type: str) -> None:
    """``scale={"type": "time"/"utc", "domain": [datetime.date(...), ...]}``
    (a RAW DICT, not a ``TimeScale(...)`` instance) renders byte-identical
    SVG to the equivalent explicit epoch-ms float domain.

    Distinct from ``tests/test_timescale_domain.py``'s taxonomy (T3): that
    module's ``_svg_for_domain`` always goes through the ``TimeScale``
    PyO3 constructor, which already had its own datetime-accepting
    extraction before this task. This test is the raw-dict-specific path
    this task actually adds (``convert_raw_dict_temporal_domain``, called
    from ``EncodingSpec::new`` BEFORE ``json.dumps`` ever sees the dict).
    """
    lo, hi = dt.date(2021, 3, 1), dt.date(2021, 3, 2)
    lo_ms, hi_ms = _epoch_ms(lo), _epoch_ms(hi)

    svg_date = _svg_for_raw_dict_domain([lo, hi], scale_type=scale_type)
    svg_float = _svg_for_raw_dict_domain([lo_ms, hi_ms], scale_type=scale_type)
    assert svg_date == svg_float

    labels = axis_tick_labels(svg_date, axis="x")
    assert len(labels) >= 2, "byte-identity must not be vacuous over a blank render"


def test_raw_dict_datetime_domain_renders_identically_to_epoch_float() -> None:
    """The ``datetime.datetime`` (not just ``datetime.date``) element case,
    for the ``time`` type — proves the raw-dict conversion routes through
    the same ``temporal_value_to_epoch_ms`` taxonomy the class path uses,
    not a narrower date-only special case.

    Non-midnight datetimes (13:30, not 00:00:00) are load-bearing here
    (quality-review remediation): a converter implemented as "take
    ``.date()`` then midnight-UTC" — a narrower date-only special case, the
    exact failure mode this test's docstring claims to exclude — would
    silently truncate the time-of-day component and still produce
    byte-identical output against a midnight-only baseline, passing
    vacuously. ``calendar.timegm(lo.timetuple())`` (not ``_epoch_ms``, which
    takes a ``date`` and is midnight-only by construction) carries H:M:S
    from a real ``datetime``, so a truncating converter now disagrees with
    the baseline and this test would fail.
    """
    lo = dt.datetime(2021, 3, 1, 13, 30, 0)
    hi = dt.datetime(2021, 3, 2, 7, 45, 0)
    lo_ms = calendar.timegm(lo.timetuple()) * 1000.0
    hi_ms = calendar.timegm(hi.timetuple()) * 1000.0

    svg_datetime = _svg_for_raw_dict_domain([lo, hi], scale_type="time")
    svg_float = _svg_for_raw_dict_domain([lo_ms, hi_ms], scale_type="time")
    assert svg_datetime == svg_float

    # Non-vacuity: the non-midnight baseline must actually differ from the
    # midnight-truncated one, or a date-only-truncating converter would pass
    # both this test and the weaker midnight-based version it replaces.
    midnight_ms = _epoch_ms(lo.date())
    assert lo_ms != midnight_ms, "13:30 must not collapse to midnight in the independent baseline"


# ---------------------------------------------------------------------------
# 6. Non-temporal junk domain element refuses, naming accepted forms
# ---------------------------------------------------------------------------


def test_non_temporal_domain_element_refuses_naming_accepted_forms() -> None:
    """A raw-dict ``time`` scale's ``domain`` list containing a value that is
    neither float, ``datetime.date``, ``datetime.datetime``, nor an
    ISO-8601 string refuses with ``TypeError``, naming the accepted forms —
    the same accepted-forms taxonomy ``TimeScale(domain=...)``'s own PyO3
    constructor uses (``temporal_value_to_epoch_ms``, reused verbatim by the
    raw-dict conversion path).
    """
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("x", scale={"type": "time", "domain": [object(), 1.0]}), y="y")
    )
    with pytest.raises(TypeError, match="datetime.date, datetime.datetime, or an ISO-8601"):
        chart.to_spec()


# ---------------------------------------------------------------------------
# 7. Untyped raw-dict spelling (no "type" key) reaches the gate as linear
# ---------------------------------------------------------------------------


def test_untyped_raw_dict_with_legal_key_parses_as_linear() -> None:
    """``scale={"zero": False}`` — no ``"type"`` key at all — is the single
    most common raw-dict spelling in the codebase (dozens of sites across
    ``tests/`` and ``src/ferrum/plots/``, e.g.
    ``tests/test_bar_zero.py::test_bar_explicit_zero_false``). It reaches
    the gate as ``linear`` because
    ``ferrum.encoding._scale._scale_to_dict``'s dict branch injects
    ``{"type": "linear", **scale}`` before the wire boundary ever sees it —
    named explicitly here rather than left as an implicit assumption every
    other test in this module quietly depends on.
    """
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    chart = fm.Chart(df).mark_point().encode(x=fm.X("a", scale={"zero": False}), y="b")
    spec_json = json.loads(chart.to_spec().to_json())
    assert spec_json["encoding"]["x"]["scale"]["type"] == "linear"
    assert spec_json["encoding"]["x"]["scale"]["zero"] is False


def test_untyped_raw_dict_typo_refuses_naming_linear() -> None:
    """The same untyped-dict injection means an untyped typo (``{"clammp":
    True}``, no ``"type"`` key) refuses naming ``linear``'s accepted list —
    the gate sees the exact same ``{"type": "linear", "clammp": true}``
    shape the explicitly-typed case (``test_typo_key_refused_names_key_type_and_accepted_list``)
    exercises, just reached through the untyped spelling instead.
    """
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    chart = fm.Chart(df).mark_point().encode(x=fm.X("a", scale={"clammp": True}), y="b")
    with pytest.raises(ValueError) as exc_info:
        chart.to_spec()
    message = str(exc_info.value)
    assert "unknown key 'clammp'" in message
    assert "for type 'linear'" in message
    assert _accepted_suffix("linear") in message


# ---------------------------------------------------------------------------
# 8. Producer-emitted dicts (not hand-written literals) do not over-refuse
# ---------------------------------------------------------------------------


def test_producer_emitted_scale_dicts_do_not_over_refuse() -> None:
    """The "no over-refusal" sweep in section 2 above is authored FROM
    ``ACCEPTED_KEYS`` and can only catch a gate that refuses a key it
    enumerates itself. This test instead feeds the gate dicts actually
    PRODUCED by ferrum's own internal scale-dict assembly code — the shape
    of coverage that would have caught the S4 bar-zero-anchor regression
    (``_spec_build.py`` used to stamp ``"zero": True`` onto every bar
    y-scale regardless of type, which the hand-authored sweep's
    ``VALID_DICTS`` never exercised because it never modeled that
    producer's own composition logic).
    """
    from ferrum.encoding._scale import _scale_to_dict

    # Producer 1: _scale_to_dict's own untyped-dict injection + domainParam
    # sugar (D6 reactive rescale) — a dict shape only a producer emits, not
    # something a user hand-writes as a literal in the sweep above.
    param_domain_scale = _scale_to_dict({"domain": fm.param("d", value=[0.0, 10.0])})
    assert param_domain_scale == {"type": "linear", "domainParam": "d"}
    j = _base_spec_json()
    j["encoding"]["x"]["scale"] = param_domain_scale
    ChartSpec.from_json(json.dumps(j))  # must not raise

    # Producer 2: a real Chart.to_spec() call for every non-linear bar
    # y-scale in the S4 repro matrix — extracts the ACTUAL scale dict
    # _spec_build.py assembles (post zero-anchor logic, post override
    # merge), not a re-typed hand literal, and feeds it back through the
    # gate a second time via ChartSpec.from_json to prove that specific
    # producer output round-trips.
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})
    for scale_factory in (fm.LogScale(), fm.SymlogScale(), fm.SqrtScale()):
        chart = fm.Chart(df).mark_bar().encode(x="cat", y=fm.Y("val", scale=scale_factory))
        produced = json.loads(chart.to_spec().to_json())
        produced_scale = produced["encoding"]["y"]["scale"]
        assert "zero" not in produced_scale, (
            f"producer must not emit a zero key for {produced_scale.get('type')!r}"
        )
        j2 = _base_spec_json()
        j2["encoding"]["x"]["scale"] = produced_scale
        ChartSpec.from_json(json.dumps(j2))  # must not raise — round-trips clean


# ---------------------------------------------------------------------------
# 9. Raw-dict temporal domains render on the LAYER path, not just chart-level
# ---------------------------------------------------------------------------


def test_layer_path_raw_dict_date_domain_renders() -> None:
    """``scale={"type": "time", "domain": [datetime.date(...), ...]}`` on a
    ``chart.layer(...)`` channel renders — the S3 quality-review finding.

    Before the ``ferrum.encoding._scale._scale_to_dict`` fix, this exact
    dict rendered fine on a bare chart-level channel (routed through Rust's
    ``EncodingSpec::new`` hook) but crashed with an opaque ``TypeError:
    Object of type date is not JSON serializable`` on the layer path
    (``coerce_layers`` -> ``pyo3_serde::from_py`` json-dumps the raw dict
    directly, bypassing that Rust hook entirely). Fixed at the shared
    Python seam both paths call identically — see the module docstring.
    """
    from ferrum.layer import Layer

    lo, hi = dt.date(2021, 3, 1), dt.date(2021, 3, 2)
    lo_ms, hi_ms = _epoch_ms(lo), _epoch_ms(hi)
    df = pl.DataFrame({"d": [lo_ms, hi_ms], "y": [0.0, 100.0]})

    base = fm.Chart(df).mark_point().encode(x="d", y="y")

    def _layered_svg(domain: list[object]) -> str:
        layer = Layer(
            mark="line",
            encoding={"x": fm.X("d", type="T", scale={"type": "time", "domain": domain}), "y": "y"},
        )
        return base.layer(layer).to_svg()

    svg_date = _layered_svg([lo, hi])
    svg_float = _layered_svg([lo_ms, hi_ms])
    assert svg_date == svg_float

    labels = axis_tick_labels(svg_date, axis="x")
    assert len(labels) >= 2, "byte-identity must not be vacuous over a blank render"


def test_layer_path_raw_dict_iso_string_domain_renders() -> None:
    """The ISO-8601 string element case on the layer path — the other
    non-numeric spelling the quality review flagged as diverging from the
    chart-level path before the fix.
    """
    from ferrum.layer import Layer

    lo, hi = dt.date(2021, 3, 1), dt.date(2021, 3, 2)
    lo_ms, hi_ms = _epoch_ms(lo), _epoch_ms(hi)
    df = pl.DataFrame({"d": [lo_ms, hi_ms], "y": [0.0, 100.0]})

    base = fm.Chart(df).mark_point().encode(x="d", y="y")

    def _layered_svg(domain: list[object]) -> str:
        layer = Layer(
            mark="line",
            encoding={"x": fm.X("d", type="T", scale={"type": "time", "domain": domain}), "y": "y"},
        )
        return base.layer(layer).to_svg()

    svg_iso = _layered_svg(["2021-03-01", "2021-03-02"])
    svg_float = _layered_svg([lo_ms, hi_ms])
    assert svg_iso == svg_float


def test_chart_plus_chart_composition_raw_dict_date_domain_renders() -> None:
    """``chart_a + chart_b`` (the documented, common layering spelling) with
    a raw-dict temporal domain on one side renders — the exact composition
    shape the coordinator's remediation brief names (``(a+b).to_svg()``).
    """
    lo, hi = dt.date(2021, 3, 1), dt.date(2021, 3, 2)
    lo_ms, hi_ms = _epoch_ms(lo), _epoch_ms(hi)
    df_a = pl.DataFrame({"d": [lo_ms, hi_ms], "y": [0.0, 100.0]})
    df_b = pl.DataFrame({"d": [lo_ms, hi_ms], "y2": [10.0, 20.0]})

    chart_a = (
        fm.Chart(df_a)
        .mark_point()
        .encode(x=fm.X("d", type="T", scale={"type": "time", "domain": [lo, hi]}), y="y")
    )
    chart_b = fm.Chart(df_b).mark_line().encode(x=fm.X("d", type="T"), y="y2")
    svg = (chart_a + chart_b).to_svg()  # must not raise
    assert "<svg" in svg


# ---------------------------------------------------------------------------
# 11. Recurring S4: override-scale-merge type switch over an explicit base
# ---------------------------------------------------------------------------

_TYPE_SWITCH_BASE_SCALES: list[tuple[str, object]] = [
    ("class:LinearScale", fm.LinearScale()),
    ("dict:typed-linear-zero-false", {"type": "linear", "zero": False}),
    ("dict:untyped-zero-false", {"zero": False}),
]
_TYPE_SWITCH_BASE_IDS = [name for name, _scale in _TYPE_SWITCH_BASE_SCALES]

_TYPE_SWITCH_MARKS: list[tuple[str, str]] = [("bar", "cat"), ("point", "cat")]
_TYPE_SWITCH_MARK_IDS = [name for name, _x in _TYPE_SWITCH_MARKS]


def _chart_with_y_scale(mark_name: str, scale: object) -> "fm.Chart":
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})
    chart = fm.Chart(df).encode(x="cat", y=fm.Y("val", scale=scale))
    return chart.mark_bar() if mark_name == "bar" else chart.mark_point()


@pytest.mark.parametrize("mark_name, _x", _TYPE_SWITCH_MARKS, ids=_TYPE_SWITCH_MARK_IDS)
@pytest.mark.parametrize(
    "scale_name, base_scale", _TYPE_SWITCH_BASE_SCALES, ids=_TYPE_SWITCH_BASE_IDS
)
def test_override_type_switch_linear_to_log_drops_stale_zero_key(
    mark_name: str, _x: str, scale_name: str, base_scale: object
) -> None:
    """``.override(y_scale_type="log")`` over an explicit linear-ish base
    scale (class, typed dict, or untyped dict — all three carry/normalize
    to a real ``"zero"`` key) must not carry ``zero`` onto ``log``, on both
    ``bar`` and ``point`` (the bug is not bar-specific: the zero-anchor
    guard only ever protects the bar-injected key, not a user-authored
    ``LinearScale()``'s own ``zero`` field).
    """
    chart = _chart_with_y_scale(mark_name, base_scale).override(y_scale_type="log")
    scale = json.loads(chart.to_spec().to_json())["encoding"]["y"]["scale"]
    assert scale["type"] == "log"
    assert "zero" not in scale
    assert "<svg" in chart.to_svg()


@pytest.mark.parametrize("mark_name, _x", _TYPE_SWITCH_MARKS, ids=_TYPE_SWITCH_MARK_IDS)
def test_override_type_switch_log_to_linear_drops_stale_base_key(mark_name: str, _x: str) -> None:
    """The reverse-direction switch: ``fm.LogScale()`` (carries ``base``)
    overridden to ``linear`` must drop ``base`` — the sibling repro from
    the quality review, proving the fix is not accidentally one-directional.
    """
    chart = _chart_with_y_scale(mark_name, fm.LogScale()).override(y_scale_type="linear")
    scale = json.loads(chart.to_spec().to_json())["encoding"]["y"]["scale"]
    assert scale["type"] == "linear"
    assert "base" not in scale
    assert "<svg" in chart.to_svg()


def test_override_same_type_control_keeps_variant_specific_key() -> None:
    """Control: when the override does NOT change the effective type (only
    ``y_scale_domain`` is overridden, ``fm.LogScale(base=2.0)``'s own
    ``type`` is untouched), the merge must NOT filter — ``base`` must
    survive. Proves the type-aware filter fires only on an actual type
    change, not on every override.
    """
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "val": [10.0, 20.0, 15.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x", y=fm.Y("val", scale=fm.LogScale(base=2.0)))
        .override(y_scale_domain=[1, 100])
    )
    scale = json.loads(chart.to_spec().to_json())["encoding"]["y"]["scale"]
    assert scale["type"] == "log"
    assert scale["base"] == 2.0
    assert scale["domain"] == [1.0, 100.0]
    assert "<svg" in chart.to_svg()


# ---------------------------------------------------------------------------
# 11b. Round 4: filter against the TARGET type's own accepted-key set
# (ferrum._core.scale_accepted_keys), not a hand-mirrored intersection —
# survival of a shared key, and mixed continuous/non-continuous switches
# ---------------------------------------------------------------------------

_SURVIVAL_TYPE_SWITCHES: list[tuple[str, str]] = [
    ("linear", "log"),
    ("time", "utc"),
]
_SURVIVAL_IDS = [f"{old}->{new}" for old, new in _SURVIVAL_TYPE_SWITCHES]


@pytest.mark.parametrize("old_type, new_type", _SURVIVAL_TYPE_SWITCHES, ids=_SURVIVAL_IDS)
def test_override_type_switch_keeps_a_key_the_new_type_also_accepts(
    old_type: str, new_type: str
) -> None:
    """A key both the old and new type accept (``nice``, on every continuous
    type except ``pow``/``sqrt``) must SURVIVE a type-changing
    ``Chart.override(<channel>_scale_type=...)``, not just a variant-specific
    key be dropped. Round 3's ``_CONTINUOUS_COMMON_SCALE_KEYS`` intersection
    mirror filtered ``nice`` out on every one of these switches (it is
    per-variant, not in ``ContinuousScaleCommon``), silently moving the axis
    relative to the explicit spelling — this is the equality/inequality
    shape already used at
    ``test_bar_override_y_scale_domain_matches_explicit_zero_false_not_zero_true``
    above, applied to ``nice`` instead of ``domain``/``zero``.
    """
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})

    def _svg(scale: dict) -> str:
        return fm.Chart(df).mark_point().encode(x=fm.X("x", scale=scale), y="y").to_svg()

    svg_override = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("x", scale={"type": old_type, "nice": True}), y="y")
        .override(x_scale_type=new_type)
        .to_svg()
    )
    svg_explicit_nice_true = _svg({"type": new_type, "nice": True})
    svg_explicit_nice_omitted = _svg({"type": new_type})
    assert svg_override == svg_explicit_nice_true, (
        f"nice=True must survive the {old_type}->{new_type} override switch"
    )
    assert svg_override != svg_explicit_nice_omitted, (
        "the two explicit spellings must render differently, or this pair is vacuous"
    )


_MIXED_FAMILY_SWITCHES: list[tuple[str, object, str, str]] = [
    # (id, base_scale, override_type, stale_key_expected_absent)
    ("point_padding->band", fm.PointScale(padding=0.5), "band", "reverse"),
    ("band_paddingInner->point", fm.BandScale(padding_inner=0.4), "point", "paddingInner"),
    ("band_paddingInner->log", fm.BandScale(padding_inner=0.4), "log", "paddingInner"),
    ("log_base->band", fm.LogScale(base=2.0), "band", "base"),
    ("point_align->linear", fm.PointScale(align=0.0), "linear", "align"),
    ("linear->band", fm.LinearScale(), "band", "zero"),
]
_MIXED_FAMILY_IDS = [case_id for case_id, *_rest in _MIXED_FAMILY_SWITCHES]


@pytest.mark.parametrize(
    "base_scale, override_type, stale_key",
    [c[1:] for c in _MIXED_FAMILY_SWITCHES],
    ids=_MIXED_FAMILY_IDS,
)
def test_override_mixed_family_type_switch_renders_without_stale_key(
    base_scale: object, override_type: str, stale_key: str
) -> None:
    """A type-changing override across the continuous/non-continuous
    boundary (``point``/``band``/``log``/``linear`` mixed pairwise) used to
    hard-raise, since round 3's filter only fired when BOTH the old and new
    type were in the 7-member continuous family. Every one of these shapes
    rendered pre-gate (the carried key is not a field on the target
    variant, so serde's flatten silently dropped it), so filtering by the
    target type's own accepted-key set — which fires on any type change,
    not just a continuous-to-continuous one — must restore that.
    """
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "val": [10.0, 20.0, 15.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("x", scale=base_scale), y="val")
        .override(x_scale_type=override_type)
    )
    scale = json.loads(chart.to_spec().to_json())["encoding"]["x"]["scale"]
    assert scale["type"] == override_type
    assert stale_key not in scale
    assert "<svg" in chart.to_svg()


def test_override_mixed_family_type_switch_on_color_channel_renders() -> None:
    """The non-positional-channel sibling of the mixed-family switches
    above: ``Color(scale=fm.LinearScale())`` (carries ``zero``/``clamp``/
    ``nice``, none of which ``sequential`` has) overridden to
    ``color_scale_type="sequential"`` used to raise
    ``unknown key 'clamp' for type 'sequential'``.
    """
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "val": [10.0, 20.0, 15.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x", y="val", color=fm.Color("val", scale=fm.LinearScale()))
        .override(color_scale_type="sequential")
    )
    scale = json.loads(chart.to_spec().to_json())["encoding"]["color"]["scale"]
    assert scale["type"] == "sequential"
    assert "clamp" not in scale
    assert "zero" not in scale
    assert "<svg" in chart.to_svg()


# ---------------------------------------------------------------------------
# 12. Exception-drift fix: coherent vocabulary for both bad-domain-element
# kinds
# ---------------------------------------------------------------------------


def test_bad_iso_string_in_scale_domain_speaks_scale_domain_vocabulary() -> None:
    """A bad ISO string in a raw-dict temporal domain used to raise
    ``ferrum.annotation.coords``'s "Cannot parse annotation coordinate ..."
    wording, since ``_scale_to_dict``'s Python-side conversion (item 5/9
    above) routes every string element through
    ``temporal_coord_to_epoch_ms`` directly — a subsystem the user never
    named. Now re-raised in the same vocabulary Rust's own
    ``iso8601_string_epoch_ms`` hook uses for the identical mistake, so a
    scale-domain mistake reads as a scale-domain mistake regardless of
    which subsystem (Python, here, or Rust, for the sibling case below)
    happens to catch it.
    """
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("x", scale={"type": "time", "domain": ["notadate", "2021-03-02"]}), y="y")
    )
    with pytest.raises(ValueError) as exc_info:
        chart.to_spec()
    message = str(exc_info.value)
    assert "Cannot parse TimeScale domain value" in message
    assert "annotation coordinate" not in message
    assert "ISO-8601" in message


def test_non_iso_junk_element_speaks_the_same_vocabulary() -> None:
    """The sibling bad-domain-element case — a value that is neither
    numeric nor date/datetime/str at all — is unchanged (still Rust's
    ``TypeError``, since it never reaches the ISO-string branch), and now
    reads coherently alongside the ISO-string case's message above: both
    name ``"TimeScale domain value(s)"``, not two different subsystems.
    """
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("x", scale={"type": "time", "domain": [object(), 1.0]}), y="y")
    )
    with pytest.raises(TypeError, match="TimeScale domain values"):
        chart.to_spec()


def test_iso_parse_message_matches_rust_wording_modulo_quoting() -> None:
    """Cross-language drift guard for the Python copy of Rust's ISO-parse
    message (py-quality cycle-3 required fix 5). The two literals
    (``_scale.py::_convert_temporal_domain_elements`` and Rust's
    ``iso8601_string_epoch_ms``) are NOT byte-identical: Rust quotes the
    offending value with ``{s:?}`` (double quotes), Python with
    ``{value!r}`` (single quotes). This pins the shared sentence with that
    one known quoting difference normalized, rather than claiming
    byte-identity.

    Rust's wording is triggered through the real production PyO3
    constructor, ``ferrum._core.EncodingSpec`` — the exact class
    ``SpecBuildMixin._build_encoding_specs`` instantiates at the
    chart-level channel path (``src/ferrum/_spec_build.py``) — called
    directly with a raw dict so Python's own conversion layer (which
    intercepts every string element before Rust ever sees one on the
    normal ``fm.X(...)``/``fm.Y(...)`` route) is bypassed. This is a real
    production entry point exercised directly, not a hand-built mirror of
    Rust's message.
    """
    from ferrum._core import EncodingSpec

    bad_value = "notadate"
    df = pl.DataFrame({"x": [1.0], "y": [2.0]})

    with pytest.raises(ValueError) as py_exc:
        (
            fm.Chart(df)
            .mark_point()
            .encode(x=fm.X("x", scale={"type": "time", "domain": [bad_value, "2021-03-02"]}), y="y")
            .to_spec()
        )
    py_message = str(py_exc.value)

    with pytest.raises(ValueError) as rust_exc:
        EncodingSpec(
            "x", "quantitative", scale={"type": "time", "domain": [bad_value, "2021-03-02"]}
        )
    rust_message = str(rust_exc.value)

    py_normalized = py_message.replace(repr(bad_value), f'"{bad_value}"')
    assert py_normalized == rust_message, (
        "Python's ISO-parse message must track Rust's wording (quoting difference aside)"
    )


# ---------------------------------------------------------------------------
# 13. Adjudicated-kept: bar + override(y_scale_domain=...) no longer widens
# ---------------------------------------------------------------------------


def test_bar_override_y_scale_domain_matches_explicit_zero_false_not_zero_true() -> None:
    """DELIBERATE BEHAVIOR CHANGE (round 2's override-merge/zero-anchor
    reorder; flag for a T8 changelog callout): before the reorder, a bar's
    ``.override(y_scale_domain=[...])`` got the zero-anchor injected FIRST
    onto an empty scale (``{"type": "linear", "zero": True}``), then had
    ``domain`` merged on top, producing ``{"type": "linear", "zero": True,
    "domain": [lo, hi]}`` — Rust's positional resolver would then widen the
    resolved axis extent to include zero even though the user explicitly
    supplied a domain. Now the override merge runs first, so ``domain`` is
    present by the time the zero-anchor check runs and the injection is
    correctly suppressed — matching the pre-existing, documented rule that
    a channel-level ``scale={"domain": ...}`` already suppresses the
    zero-anchor (``tests/test_bar_zero.py::test_bar_explicit_domain_no_zero``).
    Sharper than a tick-label assertion (per the spec reviewer's addendum):
    equality with the explicit ``zero: False`` spelling AND inequality
    with the explicit ``zero: True`` spelling, over the exact repro pair
    the spec reviewer independently confirmed.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [50.0, 120.0, 90.0]})
    svg_override = (
        fm.Chart(df).mark_bar().encode(x="cat", y="val").override(y_scale_domain=[50, 200]).to_svg()
    )
    svg_zero_false = (
        fm.Chart(df)
        .mark_bar()
        .encode(x="cat", y=fm.Y("val", scale={"domain": [50, 200], "zero": False}))
        .to_svg()
    )
    svg_zero_true = (
        fm.Chart(df)
        .mark_bar()
        .encode(x="cat", y=fm.Y("val", scale={"domain": [50, 200], "zero": True}))
        .to_svg()
    )
    assert svg_override == svg_zero_false, (
        "override(y_scale_domain=...) must match the explicit zero=False spelling"
    )
    assert svg_override != svg_zero_true, (
        "override(y_scale_domain=...) must NOT widen to the old zero=True behavior"
    )


# ---------------------------------------------------------------------------
# 14. Round 5: old-type validity — a key invalid for the OLD type must not
# be laundered through a type-changing override
# ---------------------------------------------------------------------------


def test_override_type_switch_typo_base_key_still_refuses() -> None:
    """Case B (spec review cycle 4, finding 1): ``clammp`` (a typo of
    ``clamp``) is not a real field of ``linear`` OR ``log``, so a
    type-changing override must refuse it exactly as the no-override
    control (case A) does, rather than letting it vanish silently the way
    it did before round 5's old-type-validity check — round 4's filter only
    dropped keys the target type rejects that the SOURCE type accepted, so
    a source-invalid key it happened to also reject was passed through
    unfiltered and then silently dropped by the merge's own ``**`` spread
    once it reached a type with no such field.

    Round 6 replaced the hand-synthesized refusal message with a probe of
    the real gate (``ferrum._core.EncodingSpec``), so the override spelling
    and the no-override control are now literally the same call on the
    same dict — asserted here as exact message equality, not a shared
    substring.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})
    scale = {"type": "linear", "clammp": True}

    with pytest.raises(ValueError, match="unknown key 'clammp' for type 'linear'") as no_override:
        fm.Chart(df).mark_bar().encode(x="cat", y=fm.Y("val", scale=scale)).to_svg()

    with pytest.raises(ValueError) as with_override:
        fm.Chart(df).mark_bar().encode(x="cat", y=fm.Y("val", scale=scale)).override(
            y_scale_type="log"
        ).to_svg()

    assert str(with_override.value) == str(no_override.value), (
        "the override spelling must raise the identical message as the no-override control"
    )


def test_override_type_switch_key_invalid_for_old_type_is_not_promoted() -> None:
    """Case C (spec review cycle 4, finding 1, "sharper"): ``nice`` is not
    a ``band`` field, so ``{"type": "band", "nice": True}`` refuses with no
    override at all (case A control, asserted first). ``nice`` IS a real
    ``log`` field, though, so before round 5's old-type-validity check, a
    ``.override(x_scale_type="log")`` switch let it pass through unfiltered
    (round 4's filter only drops keys against the SOURCE type's own accepted
    set for keys the source type actually has) and ``log``'s own field
    silently absorbed it — worse than the pre-gate baseline, where
    ``band``'s flatten just dropped an inapplicable key. Both the
    no-override and the override spelling must refuse identically.

    Round 6 note: this is now a probe of the real gate rather than a
    synthesized message (see previous test's docstring), so equality is
    asserted directly rather than a shared substring.
    """
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "val": [10.0, 20.0, 15.0]})
    scale = {"type": "band", "nice": True}

    with pytest.raises(ValueError, match="unknown key 'nice' for type 'band'") as no_override:
        fm.Chart(df).mark_point().encode(x=fm.X("x", scale=scale), y="val").to_svg()

    with pytest.raises(ValueError) as with_override:
        fm.Chart(df).mark_point().encode(x=fm.X("x", scale=scale), y="val").override(
            x_scale_type="log"
        ).to_svg()

    assert str(with_override.value) == str(no_override.value), (
        "the override spelling must raise the identical message as the no-override control"
    )


# ---------------------------------------------------------------------------
# 15. Round 5: non-string override-type value must not leak a PyO3
# argument-coercion error through an explicit base scale
#
# Shared with item 16's old-type parametrization below (round 7,
# rust-quality cycle-6 finding 7): one list of malformed "type" values
# feeds both the override slot (this section) and the base slot (item 16),
# so the two positions cannot drift apart the way round 6's two
# independently hand-typed lists did (``float:1.5`` was override-only-
# absent, ``none:None`` was missing from both).
# ---------------------------------------------------------------------------

_NONSTRING_TYPE_VALUES: list[tuple[str, object]] = [
    ("int:5", 5),
    ("float:1.5", 1.5),
    ("bytes:log", b"log"),
    ("class:LogScale", fm.LogScale()),
    ("none:None", None),
]
_NONSTRING_IDS = [name for name, _value in _NONSTRING_TYPE_VALUES]


@pytest.mark.parametrize(
    "new_type_value", [value for _name, value in _NONSTRING_TYPE_VALUES], ids=_NONSTRING_IDS
)
def test_override_nonstring_type_over_explicit_base_does_not_leak_pyo3_argument_error(
    new_type_value: object,
) -> None:
    """A non-``str`` ``<channel>_scale_type=`` override value (a user
    reaching for ``y_scale=`` and mistyping ``y_scale_type=``) used to
    surface PyO3's own argument-coercion error naming an internal
    parameter the user never wrote (``argument 'scale_type': 'LogScale'
    object is not an instance of 'str'``) whenever the channel carried an
    explicit base scale — because ``_merge_override_scale`` handed the raw
    value straight to ``ferrum._core.scale_accepted_keys`` before any gate
    ever saw it. It must not, regardless of which downstream error the
    malformed value ultimately produces once it reaches the real gate.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})
    with pytest.raises(Exception) as excinfo:
        fm.Chart(df).mark_bar().encode(x="cat", y=fm.Y("val", scale=fm.LinearScale())).override(
            y_scale_type=new_type_value
        ).to_svg()
    assert "argument 'scale_type'" not in str(excinfo.value)


@pytest.mark.parametrize(
    "new_type_value", [value for _name, value in _NONSTRING_TYPE_VALUES], ids=_NONSTRING_IDS
)
def test_override_nonstring_type_same_error_class_with_and_without_base_scale(
    new_type_value: object,
) -> None:
    """The same malformed ``<channel>_scale_type=`` value, with vs. without
    an explicit base scale on the channel, must land in the SAME downstream
    error path — not diverge into the PyO3 argument-coercion error on one
    side (explicit base) and the real gate's own message on the other (no
    base), which is exactly the class/no-base inconsistency the round-4
    rust-quality review found and this round's ``isinstance`` guard closes.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})

    with pytest.raises(Exception) as with_base:
        fm.Chart(df).mark_bar().encode(x="cat", y=fm.Y("val", scale=fm.LinearScale())).override(
            y_scale_type=new_type_value
        ).to_svg()
    with pytest.raises(Exception) as without_base:
        fm.Chart(df).mark_bar().encode(x="cat", y="val").override(
            y_scale_type=new_type_value
        ).to_svg()

    assert type(with_base.value) is type(without_base.value), (
        "with-base and no-base spellings must raise the same exception type"
    )
    assert str(with_base.value) == str(without_base.value), (
        "with-base and no-base spellings must raise the same message"
    )


def test_override_bogus_type_over_explicit_base_surfaces_gate_unknown_variant() -> None:
    """Round 4's ``except ValueError`` fallback branch around
    ``scale_accepted_keys`` — reached when the override's type string names
    no known ``ScaleSpec`` variant — was executed by zero tests before this
    round. An unknown override-type string over an EXPLICIT base scale must
    still fall through to the real gate's own "unknown variant" refusal,
    not get swallowed or mis-shaped by the merge helper.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})
    with pytest.raises(ValueError, match=r"unknown variant `bogus`"):
        fm.Chart(df).mark_bar().encode(x="cat", y=fm.Y("val", scale=fm.LogScale())).override(
            y_scale_type="bogus"
        ).to_svg()


# ---------------------------------------------------------------------------
# 16. Round 6: old-type validity is now a probe of the REAL gate (not a
# hand-synthesized message) -- closes the old-type non-string leak (round
# 5's isinstance guard only protected the NEW-type call one branch below)
# and the unknown-old-type-tag fall-through (round 5's docstring claimed
# it was safe; it wasn't, since a type-changing override replaces the tag
# before the gate could ever see the base's own claim). Round 7 (below)
# folded ``None`` into this parametrization -- round 6's entry guard
# (``old_type is None``) skipped the probe entirely for an EXPLICIT
# ``{"type": None}`` base, which is the one shape this section's own
# universality claim did not yet cover; see item 17's narrative note.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "old_type_value", [value for _name, value in _NONSTRING_TYPE_VALUES], ids=_NONSTRING_IDS
)
def test_override_type_switch_nonstring_old_type_same_error_as_no_override_control(
    old_type_value: object,
) -> None:
    """Round 5's ``isinstance(new_type, str)`` guard (item 15) covered only
    the OVERRIDE's own type value. The sibling call one branch earlier --
    validating ``base_scale``'s OWN ``"type"`` -- had no such guard, so a
    non-string base ``"type"`` behind a type-changing override raised
    PyO3's raw ``TypeError: argument 'scale_type': ...`` (naming an
    internal parameter the user never wrote) instead of whatever the real
    gate raises for that exact base scale standalone.

    Round 6 replaced the whole old-type-validity check with a probe of the
    real gate (``ferrum._core.EncodingSpec``) that never hands an
    unvalidated value to the ``&str``-typed ``scale_accepted_keys`` at
    all, so the override spelling and the no-override control are now the
    identical call on the identical dict for every case the probe actually
    reached. The claim was NOT yet true for ``old_type_value is None``,
    though: round 6's entry guard (``old_type is None``) short-circuited
    BEFORE the probe for that one value, skipping it entirely rather than
    validating it -- round 7 replaced the guard with ``"type" not in
    base_scale`` (a base scale with an explicit ``"type": None`` key IS a
    real, present claim, distinct from a channel with no ``scale=`` at
    all), so the probe now runs unconditionally on every malformed
    ``old_type_value`` including ``None`` and the claim is true: this can
    no longer diverge by construction, for ANY malformed ``"type"`` value,
    not just the ones enumerated here.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})

    def render(*, override: bool):
        scale = {"type": old_type_value, "domain": [0.0, 30.0]}
        c = fm.Chart(df).mark_bar().encode(x="cat", y=fm.Y("val", scale=scale))
        if override:
            c = c.override(y_scale_type="log")
        return c.to_svg()

    with pytest.raises(Exception) as no_override:
        render(override=False)
    with pytest.raises(Exception) as with_override:
        render(override=True)

    assert "argument 'scale_type'" not in str(with_override.value)
    assert type(with_override.value) is type(no_override.value), (
        "with-override and no-override spellings must raise the same exception type"
    )
    assert str(with_override.value) == str(no_override.value), (
        "with-override and no-override spellings must raise the same message"
    )


def test_override_type_switch_unknown_old_type_tag_refuses_matching_control() -> None:
    """A base scale whose OWN ``"type"`` names no known ``ScaleSpec``
    variant (a typo like ``"linearr"``) must refuse a type-changing
    override exactly as it refuses standalone -- NOT fall through
    unfiltered. Round 5's docstring claimed the fall-through was safe
    "since the gate will refuse the base type's own tag downstream
    regardless"; that was false, because the override supplies its OWN
    ``"type"`` and the base's invalid tag never reaches the wire on that
    spelling -- the user's actual mistake (the typo'd base type) would
    never be named.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})
    scale = {"type": "linearr", "zero": True}

    with pytest.raises(ValueError, match=r"unknown variant `linearr`") as no_override:
        fm.Chart(df).mark_bar().encode(x="cat", y=fm.Y("val", scale=scale)).to_svg()
    with pytest.raises(ValueError) as with_override:
        fm.Chart(df).mark_bar().encode(x="cat", y=fm.Y("val", scale=scale)).override(
            y_scale_type="log"
        ).to_svg()

    assert str(with_override.value) == str(no_override.value), (
        "an unknown old-type tag must refuse identically with and without the override, "
        "not be silently replaced by the override before the gate ever sees it"
    )


def test_scale_to_dict_temporal_membership_check_tolerates_nonstring_type() -> None:
    """``_scale_to_dict``'s temporal-conversion branch
    (``src/ferrum/encoding/_scale.py``) tested
    ``scale.get("type") in _TEMPORAL_SCALE_TYPES`` -- a frozenset
    membership test -- against a user-supplied value with no type guard.
    An unhashable ``"type"`` (a scale pyclass instance) raised
    ``TypeError: unhashable type: ...`` from INSIDE this conversion
    helper, a message naming no ferrum concept, from a line THIS task
    added (batch-C task 4's own cycle-2 diff -- not pre-existing, despite
    round 5's report attributing it to a pre-task cause). The pre-task
    shape for the identical mistake is a plain JSON-serialization
    ``TypeError`` once the bad ``"type"`` value reaches Rust; guarding the
    membership test with ``isinstance(..., str)`` restores that plain
    shape instead of crashing earlier with a worse message.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})
    scale = {"type": fm.LogScale(), "domain": [0.0, 30.0]}

    with pytest.raises(TypeError) as excinfo:
        fm.Chart(df).mark_bar().encode(x="cat", y=fm.Y("val", scale=scale)).to_svg()

    assert "unhashable" not in str(excinfo.value)
    assert "is not JSON serializable" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 17. Round 7: producer-composition regression coverage for the narrowed
# old-type-validity probe -- a key both the old and new type accept, whose
# VALUE only legal-typechecks under the NEW type, must survive a
# type-changing override rather than being value-refused under the type
# the switch is actively replacing
# ---------------------------------------------------------------------------

_UNTYPED_STRING_DOMAIN_OVERRIDE_TARGETS = ["band", "point", "ordinal"]


@pytest.mark.parametrize("new_type", _UNTYPED_STRING_DOMAIN_OVERRIDE_TARGETS)
def test_override_type_switch_untyped_string_domain_survives_switch_to_categorical(
    new_type: str,
) -> None:
    """An UNTYPED raw-dict scale (``_scale_to_dict``'s ``"linear"`` default,
    stamped by ``_emit_scale`` before ``_merge_override_scale`` ever sees
    it) whose ``domain`` is a string list -- only legal under a categorical
    scale, not the stamped-default ``linear`` -- used to refuse under round
    6's whole-dict old-type probe (``invalid type: string "a", expected
    f64``) even though the final, override-merged scale is a perfectly
    legal ``band``/``point``/``ordinal`` domain. ``domain`` is a key both
    ``linear`` and the target type accept, so round 7's narrower probe
    (tag-only, plus a key-membership probe over only the keys ``linear``
    does NOT recognize) never value-validates it under ``linear`` at all --
    its value reaches the target type's own downstream validation on the
    final merged dict, unchanged.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("cat", scale={"domain": ["a", "b", "c"]}), y="val")
        .override(x_scale_type=new_type)
    )
    scale = json.loads(chart.to_spec().to_json())["encoding"]["x"]["scale"]
    assert scale["type"] == new_type
    assert scale["domain"] == ["a", "b", "c"]
    assert "<svg" in chart.to_svg()


def test_override_type_switch_untyped_iso_domain_survives_switch_to_time() -> None:
    """The temporal sibling of the case above: an UNTYPED raw-dict scale
    (stamped ``"linear"``) whose ``domain`` is a list of ISO date strings
    used to refuse the identical way under round 6's whole-dict probe (a
    string is not a valid ``linear`` domain element). The override-merged
    scale is a legal ``time`` scale, and asserting the domain converted to
    the exact epoch-ms pair also proves ``_scale_to_dict``'s temporal
    conversion runs on the POST-override effective type -- the switch is
    not laundering the ISO strings past the conversion the way it used to
    launder them past validation.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(
            x="cat",
            y=fm.Y("val", scale={"domain": ["2021-01-01", "2021-12-31"]}),
        )
        .override(y_scale_type="time")
    )
    scale = json.loads(chart.to_spec().to_json())["encoding"]["y"]["scale"]
    assert scale["type"] == "time"
    assert scale["domain"] == [1_609_459_200_000.0, 1_640_908_800_000.0]
    assert "<svg" in chart.to_svg()


def test_override_type_switch_explicit_linear_string_domain_survives_switch_to_band() -> None:
    """The explicit-``"type"`` sibling of the untyped case above:
    ``{"type": "linear", "domain": ["a", "b"]}`` -- the user's own claimed
    type, not a ``_scale_to_dict`` default -- switched to ``band`` must
    also survive; the fix is not conditioned on the type having been
    stamped by ``_scale_to_dict`` rather than authored directly.
    """
    df = pl.DataFrame({"cat": ["a", "b"], "val": [10.0, 20.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("cat", scale={"type": "linear", "domain": ["a", "b"]}), y="val")
        .override(x_scale_type="band")
    )
    scale = json.loads(chart.to_spec().to_json())["encoding"]["x"]["scale"]
    assert scale["type"] == "band"
    assert scale["domain"] == ["a", "b"]
    assert "<svg" in chart.to_svg()


# ---------------------------------------------------------------------------
# 18. Round 8: the DROPPED-key bucket -- a key accepted_old recognizes but
# accepted_new doesn't, about to be silently dropped by the type-switch
# filter -- must be validated under old_type before it's dropped, not
# laundered into a silent render. This is the third bucket of the
# partition item 17's docstring names: unknown_under_old (refused),
# survivors (validated downstream under new_type), and now dropped
# (validated here, under old_type, the same shape as unknown_under_old).
# ---------------------------------------------------------------------------

_DROPPED_BUCKET_CASES = [
    pytest.param({"type": "linear", "zero": "yes"}, "log", id="linear_zero-to-log"),
    pytest.param({"type": "linear", "zero": "yes"}, "band", id="linear_zero-to-band"),
    pytest.param({"type": "log", "base": "ten"}, "band", id="log_base-to-band"),
    pytest.param({"type": "band", "align": "left"}, "linear", id="band_align-to-linear"),
]


@pytest.mark.parametrize("base_scale,new_type", _DROPPED_BUCKET_CASES)
def test_override_type_switch_dropped_key_bad_value_refuses_matching_control(
    base_scale: dict, new_type: str
) -> None:
    """A key ``old_type`` accepts but ``new_type`` doesn't is dropped by
    the type-switch filter before ``new_type``'s own downstream gate ever
    sees it -- and ``unknown_under_old``'s probe never checks it either,
    since it IS a member of ``old_type``'s accepted-key set. Before this
    round, nothing validated its VALUE at all: ``{"type": "linear", "zero":
    "yes"}`` (an invalid ``bool``) + ``.override(y_scale_type="log")``
    rendered silently, even though the identical base scale refuses
    standalone. The dropped-bucket probe closes this: the override
    spelling must refuse with the exact same exception type and message as
    the no-override control.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})

    def render(*, override: bool):
        c = fm.Chart(df).mark_bar().encode(x="cat", y=fm.Y("val", scale=base_scale))
        if override:
            c = c.override(y_scale_type=new_type)
        return c.to_svg()

    with pytest.raises(Exception) as no_override:
        render(override=False)
    with pytest.raises(Exception) as with_override:
        render(override=True)

    assert type(with_override.value) is type(no_override.value), (
        "a dropped key's bad value must refuse with the same exception type as the "
        "no-override control, not render silently once the type switch drops it"
    )
    assert str(with_override.value) == str(no_override.value), (
        "a dropped key's bad value must refuse with the same message as the no-override control"
    )


def test_override_type_switch_surviving_key_untouched_by_dropped_bucket_probe() -> None:
    """Control for the fix above: a key BOTH the old and new type accept
    (``domain``, legal on both ``linear`` and ``band``) is a survivor, not
    a member of ``drop = accepted_old - accepted_new`` -- the new
    dropped-bucket probe must never see it, so it keeps reaching
    ``new_type``'s own downstream validation untouched, exactly as item 17
    already pins. A regression that widened the dropped-bucket probe to
    cover survivors too would refuse this case; it must still render.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10.0, 20.0, 15.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("cat", scale={"type": "linear", "domain": ["a", "b", "c"]}), y="val")
        .override(x_scale_type="band")
    )
    scale = json.loads(chart.to_spec().to_json())["encoding"]["x"]["scale"]
    assert scale["type"] == "band"
    assert scale["domain"] == ["a", "b", "c"]
    assert "<svg" in chart.to_svg()
