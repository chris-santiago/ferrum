"""Batch-D T12: Python-side contract tests for `ferrum._core.transform_registry`.

Three concerns:

1. **Contract shape** -- `transform_registry(kind)` returns the documented
   five-key dict (`methods`/`methods_needing_field`/`outputs`/
   `secondary_outputs`/`params`) with the documented types, and an unknown
   kind raises `ValueError` listing known kinds. `methods_needing_field`
   joined the wire in wave 6B (`#137`/`#176`); it is `[]` for a kind whose
   declared methods split no field requirement. The Rust side
   (`crates/ferrum-core/src/transform/registry/`) owns the exhaustiveness
   proof against real `apply()` output for every kind; this file only proves
   the *Python-visible* binding contract.

2. **Mirror deletion is grep-clean** -- the Python module constants this and
   later rounds deleted (`_VALID_KERNELS`, `_VALID_QQ_DISTRIBUTIONS`, the
   local `_VALID_AGGREGATES` in `desugar_hex`, `_AUX_LEGEND_CHANNELS`,
   `_RASTER_AGGREGATES`/`_RASTER_AGGREGATES_NEEDING_FIELD`) do not reappear
   anywhere in `src/ferrum/`, proving the registry adoption is complete
   rather than leaving a second, unreferenced copy behind. `_DELETED_MIRROR_LITERALS`
   also carries one hand-typed message fragment, not a symbol name
   (`'count'|'density'|'any'`, wave 6B T9 cycle 4) -- a raster no-field hint
   that was typed directly into two f-strings rather than named as a
   constant, so the same grep-for-reintroduction guard covers it too.

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

import re
from pathlib import Path

import pytest

from ferrum import _core

_SRC = Path(__file__).parent.parent / "src" / "ferrum"

# ---------------------------------------------------------------------------
# 1. Contract shape
# ---------------------------------------------------------------------------


def test_returns_documented_five_key_shape():
    """spec §4.6 / #137: `methods_needing_field` is the fifth wire key,
    published for every kind (empty for a kind whose methods split no
    field-requirement, e.g. `smooth`; non-empty for raster/hex/aggregate/
    join_aggregate)."""
    entry = _core.transform_registry("smooth")
    assert set(entry.keys()) == {
        "methods",
        "methods_needing_field",
        "outputs",
        "secondary_outputs",
        "params",
    }
    assert isinstance(entry["methods"], list)
    assert all(isinstance(m, str) for m in entry["methods"])
    assert isinstance(entry["methods_needing_field"], list)
    assert all(isinstance(m, str) for m in entry["methods_needing_field"])
    assert set(entry["methods_needing_field"]) <= set(entry["methods"])
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
    # #123, wave 6B, cycle 2: replaced by ferrum._core.aux_legend_channels().
    "_AUX_LEGEND_CHANNELS",
    # #137/#176, wave 6B T9: replaced by transform_registry("raster")'s
    # "methods"/"methods_needing_field" keys.
    "_RASTER_AGGREGATES",
    "_RASTER_AGGREGATES_NEEDING_FIELD",
    # #137/#176, wave 6B T9 cycle 4 (spec-review finding): the hand-typed
    # raster no-field hint in RenderConfig's and mark_raster's needs-field
    # messages, now composed by _registry.require_aggregate_field() from
    # transform_registry("raster") rather than mirrored as a literal.
    "'count'|'density'|'any'",
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
    update. Exact-string pin (not just a substring/regex match) on the
    PRIMARY helper's wording, `"known output columns"` -- distinct from
    `registry_secondary_columns`'s `"known columns"` below (#165(2), wave 6B
    T10 cycle 2 Q1): a regex could pass even if the `{missing}`/`{declared}`
    placeholders were swapped or the wording collapsed to match the
    secondary helper's. `['ci_lower', 'ci_upper', 'x', 'y']` is `smooth`'s
    real declared output set, read directly off `transform_registry("smooth")`
    (see `test_returns_documented_five_key_shape`'s use of the same kind),
    hand-pinned here rather than re-derived so this test cannot pass by
    sharing the same computation as the code it is checking."""
    from ferrum._registry import registry_output_columns

    with pytest.raises(ValueError) as excinfo:
        registry_output_columns("smooth", ["x", "not_a_real_column"])
    assert str(excinfo.value) == (
        "transform_registry('smooth')['outputs'] does not declare "
        "['not_a_real_column']; known output columns: "
        "['ci_lower', 'ci_upper', 'x', 'y']"
    )


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
    segment column raises here, naming the stale name. Exact-string pin on
    the SECONDARY helper's wording, `"known columns"` -- no `"output"`
    qualifier, distinct from `registry_output_columns`'s message above
    (#165(2), wave 6B T10 cycle 2 Q1). `['x', 'x2', 'y', 'y2']` is
    `linkage`'s real `segments` batch schema, hand-pinned rather than
    re-derived from the registry for the same reason as the primary pin
    above."""
    from ferrum._registry import registry_secondary_columns

    with pytest.raises(ValueError) as excinfo:
        registry_secondary_columns("linkage", "segments", ["x", "not_a_real_column"])
    assert str(excinfo.value) == (
        "transform_registry('linkage')['secondary_outputs']['segments'] "
        "does not declare ['not_a_real_column']; known columns: "
        "['x', 'x2', 'y', 'y2']"
    )


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


# ---------------------------------------------------------------------------
# 4. Secondary-key composition + `registry_*_columns` call-site pins
#    (wave 6B T10, `#165`(1)(2)) -- captured from HEAD before `_check_declared`
#    was extracted out of `registry_output_columns`/`registry_secondary_columns`,
#    and before the six Python `f"{name}_{key}"` composition sites
#    (`marks/heavy_stat.py`, `marks/composite.py` x3, `plots/matrix.py` x3)
#    were switched to call `ferrum._core.secondary_key`.
# ---------------------------------------------------------------------------

# The (kind, names) / (kind, key, names) argument tuples every real
# `registry_output_columns`/`registry_secondary_columns` call site in
# `src/ferrum/` passes, copied verbatim from source rather than invented, so
# a renamed/dropped/reordered column at any site would move one of these
# pins. `grep -rnoE "registry_(output|secondary)_columns\(" src/ferrum`
# reports 25 hits, but one (`plots/distribution.py:876`) is inside a comment
# referencing the helper, not a call -- 24 real physical call sites.
# `marks/_fit_layers.py:110` is one of those 24 physical sites but is reached
# with four different `kind`s at runtime (`smooth`/`logistic`/`glm`/`robust`,
# from `statistical.py`'s and `regression.py`'s dedicated-fit dispatch), so
# it is pinned once per real kind below -- 27 total pinned invocations.
_PRIMARY_CALL_SITES = [
    ("bin", ["bin_start", "bin_end", "count", "agg_value"]),  # _layer_transforms.py:434
    (
        "contour",
        ["contour_x", "contour_y", "contour_x2", "contour_y2", "level_id", "level_value"],
    ),  # marks/heavy_stat.py:148
    ("violin", ["violin_y", "group_id"]),  # marks/heavy_stat.py:393
    ("box_stats", ["q1", "median", "q3"]),  # marks/heavy_stat.py:497
    ("qq", ["theoretical", "sample"]),  # marks/heavy_stat.py:668
    ("hex", ["hex_x", "hex_y", "hex_id", "value"]),  # marks/heavy_stat.py:946
    ("kde", ["value", "density"]),  # marks/statistical.py:199
    ("bin", ["bin_start", "bin_end", "count", "density"]),  # marks/statistical.py:417
    ("smooth", ["x", "y"]),  # marks/statistical.py:601
    (
        "box_stats",
        ["q1", "median", "q3", "lower_whisker", "upper_whisker"],
    ),  # marks/composite.py:316
    ("error_extent", ["lower", "upper"]),  # marks/composite.py:515
    ("error_extent", ["lower", "upper"]),  # marks/composite.py:670
    # marks/_fit_layers.py:110 -- one physical call, four real `kind`s:
    ("smooth", ["x", "y", "ci_lower", "ci_upper"]),  # regplot/lmplot smooth path
    ("logistic", ["x", "y", "ci_lower", "ci_upper"]),  # logplot
    ("glm", ["x", "y", "ci_lower", "ci_upper"]),  # glmplot
    ("robust", ["x", "y", "ci_lower", "ci_upper"]),  # residplot / robust fit
    ("bin_2d", ["x_lo", "x_hi", "y_lo", "y_hi", "count"]),  # plots/matrix.py:1474
    ("qq", ["sample"]),  # plots/regression.py:304
    ("smooth", ["x", "y"]),  # plots/regression.py:1620
    ("kde", ["value", "density"]),  # plots/distribution.py:283
    ("bin", ["bin_start", "count"]),  # plots/distribution.py:806
    ("summary", ["mean", "lower", "upper"]),  # plots/distribution.py:1358
]

_SECONDARY_CALL_SITES = [
    (
        "qq",
        "qq_line",
        ["qq_line_x_start", "qq_line_y_start", "qq_line_x_end", "qq_line_y_end"],
    ),  # marks/heavy_stat.py:688
    ("letter_value", "depth_1", ["group", "lower", "upper"]),  # marks/composite.py:1210
    ("letter_value", "outliers", ["group", "value"]),  # marks/composite.py:1213
    ("linkage", "segments", ["x", "y", "x2", "y2"]),  # plots/matrix.py:1130
    ("linkage", "order", ["original_idx"]),  # plots/matrix.py:1131
]


def test_all_registry_columns_call_sites_return_their_identity_lookup():
    """Every argument tuple the real `registry_output_columns`/
    `registry_secondary_columns` call sites in `src/ferrum/` pass (27
    site x kind invocations across 24 physical call sites, see the module
    comment above), replayed directly against the live registry: each must
    still return `{name: name}` after the `_check_declared` tail extraction,
    proving the extraction did not silently drop, reorder, or dedupe a name
    at any site."""
    from ferrum._registry import registry_output_columns, registry_secondary_columns

    assert len(_PRIMARY_CALL_SITES) + len(_SECONDARY_CALL_SITES) == 27
    for kind, names in _PRIMARY_CALL_SITES:
        assert registry_output_columns(kind, names) == {n: n for n in names}
    for kind, key, names in _SECONDARY_CALL_SITES:
        assert registry_secondary_columns(kind, key, names) == {n: n for n in names}


def test_pinned_secondary_keys_from_the_six_composition_sites():
    """The six composed secondary keys, captured from the real desugar /
    `clustermap` paths (not hand-built mirrors): `qq`'s reference line,
    `letter_value`'s depth-1 band and outliers, and `linkage`'s row reorder
    index plus column/row dendrogram segments. All six sites use a fixed
    internal `name=` (never the caller's own), so only the name-set half is
    reachable here -- the name=None half is pinned directly against
    `secondary_key` below."""
    import polars as pl

    import ferrum as fm
    from ferrum.marks.composite import desugar_boxen
    from ferrum.marks.heavy_stat import desugar_qq

    qq_layers = desugar_qq("sample", line=True).layers
    reference = next(layer for layer in qq_layers if layer.name == "reference")
    assert reference.data_source == "qq_main_qq_line"

    boxen_layers = desugar_boxen("cat", "value").layers
    depth_1 = next(layer for layer in boxen_layers if layer.name == "depth_1")
    outlier = next(layer for layer in boxen_layers if layer.name == "outlier")
    assert depth_1.data_source == "lv_depth_1"
    assert outlier.data_source == "lv_outliers"

    frame = pl.DataFrame(
        {
            "row": ["r0", "r1", "r2", "r3"],
            "A": [0.1, 0.4, 0.2, 0.9],
            "B": [1.0, 0.3, 0.6, 0.2],
        }
    )
    cm = fm.clustermap(frame)
    reorder = next(t for t in cm.heatmap._transforms if "Reorder" in repr(t))
    # A repr spells each argument as its constructor parameter (`from_=`, as `AggregateOp`'s `fn_=`), #195.
    assert 'from_=Some("row_link_order")' in repr(reorder)
    assert cm.col_dendrogram._layers[0].data_source == "col_link_segments"
    assert cm.row_dendrogram._layers[0].data_source == "row_link_segments"


@pytest.mark.parametrize("key", ["qq_line", "depth_1", "outliers", "order", "segments"])
def test_secondary_key_with_no_name_is_unprefixed(key):
    """`name=None` is not reachable through any of the six Python
    composition sites above (each uses a fixed internal `name=` constant) --
    pinned directly against the real `ferrum._core.secondary_key` pyfunction
    instead, so the None-name half of the rule is proven against the real
    path even though no Python call site currently exercises it."""
    from ferrum._core import secondary_key

    assert secondary_key(None, key) == key


#: A hand-composed secondary key looks like ``f"{some_name}_{some_key}"`` --
#: an f-string whose entire body is exactly two brace-delimited expressions
#: joined by a literal underscore, where the second expression may itself be
#: a call (`` _lv_depth_key(k)``). This is the shape the six sites had before
#: #165(1); it is unrelated to `plots/_helpers.py`'s/`plots/distribution.py`'s
#: `f"{base}_{n}"` layer-name uniquifier, which is a different concern
#: (deduplicating a layer NAME, not composing a transform's secondary key)
#: and lives outside the three files this test scans.
_HAND_COMPOSED_SECONDARY_KEY_PATTERN = re.compile(
    r'f"\{[A-Za-z_][A-Za-z0-9_]*\}_\{[A-Za-z_][A-Za-z0-9_]*(?:\([^{}]*\))?\}"'
)


def _code_only(text: str) -> str:
    """*text* with every ``#``-prefixed comment line dropped.

    `marks/heavy_stat.py`'s `_QQ_LINE_KEY` doc-comment names
    `` ``secondary_key(_QQ_TRANSFORM_NAME, _QQ_LINE_KEY)`` `` in prose (wave
    6B T10 cycle 2's fix for the stale-docstring false positive against
    `_HAND_COMPOSED_SECONDARY_KEY_PATTERN`). Left in, that same prose
    satisfies a bare `"secondary_key(" in text` / `.count(...) >= 1` check
    on its own -- a reverted call site rewritten as a non-f-string
    concatenation (`_QQ_TRANSFORM_NAME + "_" + _QQ_LINE_KEY`, which the
    f-string-shape regex above also can't see) would then still pass both
    checks (cycle 3, lite-review finding). Stripping comment lines before
    either check runs means both checks only ever see real code."""
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def test_six_composition_sites_call_secondary_key_not_a_hand_composed_f_string():
    """#165(1): every one of the six known secondary-key composition sites
    (`marks/heavy_stat.py`'s qq line; `marks/composite.py`'s depth-band,
    median and outlier `data_source=`; `plots/matrix.py`'s row reorder
    `from_=` and column/row dendrogram `data_source=`) must call
    `ferrum._core.secondary_key` rather than hand-composing
    `f"{name}_{key}"`. A structural pin, not a behavioral one: it would fail
    if a site were reverted to the old f-string even though the composed
    string happens to still match (the RED case a value-only pin can't see).

    Checks BOTH directions (wave 6B T10 cycle 2 Q3): the new calls are
    present, AND the old hand-composed shape is absent -- so a new
    `f"{name}_{key}"` site added *alongside* the existing `secondary_key(`
    calls (rather than replacing one) would still fail this test, not just a
    site that reverted an existing call. Both checks run against
    `_code_only`-filtered text (cycle 3): a `#`-prefixed comment mentioning
    either shape in prose must not be able to satisfy either assertion on
    its own."""
    heavy_stat = _code_only((_SRC / "marks" / "heavy_stat.py").read_text())
    composite = _code_only((_SRC / "marks" / "composite.py").read_text())
    matrix = _code_only((_SRC / "plots" / "matrix.py").read_text())

    assert heavy_stat.count("secondary_key(") >= 1
    assert composite.count("secondary_key(") >= 3
    assert matrix.count("secondary_key(") >= 3

    for relpath, text in (
        ("marks/heavy_stat.py", heavy_stat),
        ("marks/composite.py", composite),
        ("plots/matrix.py", matrix),
    ):
        hits = _HAND_COMPOSED_SECONDARY_KEY_PATTERN.findall(text)
        assert not hits, f"{relpath} still hand-composes a secondary key: {hits}"
