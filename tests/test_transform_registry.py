"""Batch-D T12: Python-side contract tests for `ferrum._core.transform_registry`.

Three concerns:

1. **Contract shape** -- `transform_registry(kind)` returns the documented
   four-key dict (`methods`/`outputs`/`secondary_outputs`/`params`) with the
   documented types, and an unknown kind raises `ValueError` listing known
   kinds. The Rust side (`crates/ferrum-core/src/transform/registry/`) owns
   the exhaustiveness proof against real `apply()` output for every kind;
   this file only proves the *Python-visible* binding contract.

2. **Mirror deletion is grep-clean** -- the three Python module constants
   this round deleted (`_VALID_KERNELS`, `_VALID_QQ_DISTRIBUTIONS`, the
   local `_VALID_AGGREGATES` in `desugar_hex`) do not reappear anywhere in
   `src/ferrum/`, proving the registry adoption is complete rather than
   leaving a second, unreferenced copy behind.

3. **The output-column lookup helpers** -- `ferrum._registry`'s
   `registry_output_columns` (primary batch) and `registry_secondary_columns`
   (a named secondary batch) validate a consumer's column names against the
   live registry and return them as a lookup (spec §4.6 / decision M6a).
   Tested here against the real registry, not a mock.

The *consumption* side -- which Python consumers read declared output columns,
that none of them still hardcodes one, and that every name they do read is
declared -- is NOT asserted here. Round 1 and round 2 both put a hand-written
consumed-column table in this file and both shipped it incomplete (`hex`'s
`value`; then six consumer sites nobody had enumerated; then a sweep whose own
boundary was hand-counted). It now lives in
`tests/test_registry_output_column_consumers.py`, where both the population
(every string constant in `src/ferrum/**.py` equal to a declared output name)
and its accounting are computed, so the claim cannot outrun the code again.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ferrum import _core

_SRC = Path(__file__).parent.parent / "src" / "ferrum"

# ---------------------------------------------------------------------------
# 1. Contract shape
# ---------------------------------------------------------------------------


def test_returns_documented_four_key_shape():
    entry = _core.transform_registry("smooth")
    assert set(entry.keys()) == {"methods", "outputs", "secondary_outputs", "params"}
    assert isinstance(entry["methods"], list)
    assert all(isinstance(m, str) for m in entry["methods"])
    assert isinstance(entry["outputs"], list)
    for item in entry["outputs"]:
        assert isinstance(item, tuple)
        assert len(item) == 2
        name, dtype = item
        assert isinstance(name, str)
        assert isinstance(dtype, str)
    assert isinstance(entry["secondary_outputs"], list)
    assert isinstance(entry["params"], list)
    assert all(isinstance(p, str) for p in entry["params"])


def test_secondary_outputs_are_keyed_batches_of_typed_columns():
    """`secondary_outputs` is `[(key, [(column, dtype), ...]), ...]` -- the
    shape `registry_secondary_columns` reads. Asserted on `qq`, which
    declares exactly one such batch (the reference line)."""
    entry = _core.transform_registry("qq")
    assert isinstance(entry["secondary_outputs"], list)
    keys = []
    for item in entry["secondary_outputs"]:
        assert isinstance(item, tuple)
        assert len(item) == 2
        key, columns = item
        assert isinstance(key, str)
        keys.append(key)
        assert isinstance(columns, list)
        for name, dtype in columns:
            assert isinstance(name, str)
            assert isinstance(dtype, str)
    assert keys == ["qq_line"]


def test_unknown_kind_raises_listing_known_kinds():
    with pytest.raises(ValueError, match="smooth"):
        _core.transform_registry("not-a-real-transform-kind")


def test_empty_methods_and_outputs_are_real_answers_not_omissions():
    """`data_stack`'s `methods` is non-empty (offset modes) but its
    `outputs` is `[]` by documented design (the additive pair is a
    caller-settable parameter layered on an arbitrary input schema) --
    confirms `[]` is read as a real, present value, not mistaken for a
    missing key."""
    entry = _core.transform_registry("data_stack")
    assert entry["outputs"] == []
    assert entry["methods"] == ["zero", "normalize", "center"]


@pytest.mark.parametrize(
    "kind, as_default",
    [("regression_data", ("x", "y")), ("loess_data", ("x", "y"))],
)
def test_regression_family_default_as_matches_registry_outputs(kind, as_default):
    """`regression_data`/`loess_data` name their output columns from the
    caller's `as_=` tuple (default `("x", "y")`) -- the registry's declared
    `outputs` holds only under that default (T1's documented caveat), which
    this pins directly rather than assuming."""
    declared = {name for name, _dtype in _core.transform_registry(kind)["outputs"]}
    assert declared == set(as_default)


# ---------------------------------------------------------------------------
# 2. Mirror deletion is grep-clean
# ---------------------------------------------------------------------------

_DELETED_MIRROR_LITERALS = (
    "_VALID_KERNELS",
    "_VALID_QQ_DISTRIBUTIONS",
    "_VALID_AGGREGATES",
)


@pytest.mark.parametrize("literal_name", _DELETED_MIRROR_LITERALS)
def test_deleted_mirror_literal_does_not_reappear_anywhere_in_src(literal_name):
    hits = []
    for path in _SRC.rglob("*.py"):
        text = path.read_text()
        if literal_name in text:
            hits.append(str(path.relative_to(_SRC.parent.parent)))
    assert not hits, f"{literal_name!r} still referenced in: {hits}"


# ---------------------------------------------------------------------------
# 3. The output-column lookup helper (spec §4.6 / M6a)
# ---------------------------------------------------------------------------


def test_registry_output_columns_returns_a_lookup_of_the_validated_names():
    from ferrum._registry import registry_output_columns

    cols = registry_output_columns("smooth", ["x", "y", "ci_lower", "ci_upper"])
    assert cols == {"x": "x", "y": "y", "ci_lower": "ci_lower", "ci_upper": "ci_upper"}
    assert cols["ci_lower"] == "ci_lower"


def test_registry_output_columns_raises_on_undeclared_column():
    """RED-proof of the lookup half of M6a: a name that is not declared for
    the kind raises, naming the kind and the stale column -- this is what
    would fire if a Rust-side rename shipped without a matching Python
    update."""
    from ferrum._registry import registry_output_columns

    with pytest.raises(ValueError, match="smooth.*not_a_real_column"):
        registry_output_columns("smooth", ["x", "not_a_real_column"])


def test_registry_output_columns_rejects_an_unknown_kind():
    """The kind is not validated separately -- it reaches the registry, whose
    own refusal names the known kinds."""
    from ferrum._registry import registry_output_columns

    with pytest.raises(ValueError, match="unknown transform kind"):
        registry_output_columns("not-a-real-transform-kind", ["x"])


def test_registry_secondary_columns_returns_a_lookup_of_the_validated_names():
    from ferrum._registry import registry_secondary_columns

    cols = registry_secondary_columns("linkage", "segments", ["x", "y", "x2", "y2"])
    assert cols == {"x": "x", "y": "y", "x2": "x2", "y2": "y2"}


def test_registry_secondary_columns_raises_on_an_undeclared_column():
    """RED-proof of the secondary half: a Rust-side rename of a dendrogram
    segment column raises here, naming the stale name."""
    from ferrum._registry import registry_secondary_columns

    with pytest.raises(ValueError, match="linkage.*segments.*not_a_real_column"):
        registry_secondary_columns("linkage", "segments", ["x", "not_a_real_column"])


def test_registry_secondary_columns_raises_on_an_undeclared_key():
    """The batch key is validated too, so a renamed key fails at desugar time
    rather than yielding a layer bound to a batch nothing publishes."""
    from ferrum._registry import registry_secondary_columns

    with pytest.raises(ValueError, match="declares no 'depth_99' batch"):
        registry_secondary_columns("letter_value", "depth_99", ["lower"])


def test_registry_secondary_columns_refuses_the_primary_schema():
    """The two lookups are not interchangeable: `letter_value`'s PRIMARY
    schema declares `depth`, its secondary depth batches do not -- so a
    primary name checked through the secondary helper raises rather than
    passing on a coincidence."""
    from ferrum._registry import registry_output_columns, registry_secondary_columns

    assert registry_output_columns("letter_value", ["depth"]) == {"depth": "depth"}
    with pytest.raises(ValueError, match="does not declare"):
        registry_secondary_columns("letter_value", "depth_1", ["depth"])
