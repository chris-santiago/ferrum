"""`desugar_boxen`'s band layers vs. `letter_value`'s declared secondary batches.

`desugar_boxen` validates ONE depth schema at desugar time (every `depth_{k}`
batch carries the same columns, so one lookup covers all the band layers), and
builds each layer's `data_source` from `_lv_depth_key(k)`. That leaves two
claims the runtime lookup cannot make on its own, both pinned here:

* every `k` the mark builds a layer for has a declared batch -- i.e.
  `_BOXEN_K_MAX` does not exceed the depth count the transform publishes; and
* the key spelling `_lv_depth_key` produces is the one the registry declares,
  so a Rust-side change to the key shape fails here rather than yielding
  layers silently bound to batches nothing publishes.

Per the repo's Python test-file convention this is feature-scoped coverage for
the boxen/letter_value seam, not a findings-ID file.
"""

from __future__ import annotations

import pytest

from ferrum import _core
from ferrum.marks.composite import (
    _BOXEN_K_MAX,
    _LV_DEPTH_1_KEY,
    _LV_OUTLIERS_KEY,
    _LV_TRANSFORM_NAME,
    _lv_depth_key,
)


def _declared_keys() -> list[str]:
    return [key for key, _columns in _core.transform_registry("letter_value")["secondary_outputs"]]


def test_every_band_layer_has_a_declared_secondary_batch():
    declared = set(_declared_keys())
    missing = [
        _lv_depth_key(k) for k in range(1, _BOXEN_K_MAX + 1) if _lv_depth_key(k) not in declared
    ]
    assert not missing, (
        f"desugar_boxen builds a layer for {missing}, which "
        "transform_registry('letter_value')['secondary_outputs'] does not "
        f"declare (declared: {sorted(declared)})"
    )


def test_the_validated_depth_key_is_the_one_the_layers_use():
    """`_LV_DEPTH_1_KEY` is the literal `desugar_boxen` hands the registry
    lookup; `_lv_depth_key` spells the keys the layers bind to. They must be
    the same key, or the validated schema is not the schema being read."""
    assert _LV_DEPTH_1_KEY == _lv_depth_key(1)


def test_the_outliers_key_is_declared():
    assert _LV_OUTLIERS_KEY in _declared_keys()


@pytest.mark.parametrize("k", range(1, _BOXEN_K_MAX + 1))
def test_every_band_layers_data_source_is_the_named_wire_key(k):
    """`letter_value` prefixes every secondary key with the spec's own
    `name=`, so the wire name is `{name}_{key}` -- verified against a real
    rendered spec rather than restated."""
    import polars as pl

    import ferrum as fm

    frame = pl.DataFrame(
        {"cat": ["a"] * 40 + ["b"] * 40, "val": [float(i % 40) for i in range(80)]}
    )
    chart = fm.Chart(frame).mark_boxen().encode(x="cat", y="val")
    sources = {layer.data_source for layer in chart._resolve_pending()._layers}
    assert f"{_LV_TRANSFORM_NAME}_{_lv_depth_key(k)}" in sources
