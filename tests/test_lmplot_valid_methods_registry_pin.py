"""Batch-D T12: drift guard for `lmplot`'s `_VALID_METHODS`.

`ferrum.plots.regression._VALID_METHODS` is a routing-coupled subset (the
five fit branches `lmplot` actually dispatches to), not a direct mirror of
`transform_registry("smooth")["methods"]` -- deriving it live would silently
accept methods `lmplot` has no `elif` branch for (see the module constant's
own comment for the full reasoning; this is the banked cross-task note 4
disposition from the batch-D T12 dispatch). What's still worth pinning is
that the two halves of this hand-picked set stay real: the smooth-routed
names must remain valid `transform_registry("smooth")["methods"]` entries,
and the three non-smooth names must remain valid registry kinds in their own
right -- so a Rust-side rename on either side fails this test instead of
surfacing only when a user hits `lmplot(method=...)`.
"""

from __future__ import annotations

from ferrum import _core
from ferrum.plots.regression import _VALID_METHODS

_SMOOTH_ROUTED_METHODS = {"lm", "loess"}
_NON_SMOOTH_ROUTED_KINDS = {"logistic", "glm", "robust"}


def test_valid_methods_is_exactly_the_routed_smooth_and_non_smooth_split():
    assert _VALID_METHODS == _SMOOTH_ROUTED_METHODS | _NON_SMOOTH_ROUTED_KINDS


def test_smooth_routed_methods_are_real_smooth_registry_methods():
    smooth_methods = set(_core.transform_registry("smooth")["methods"])
    assert _SMOOTH_ROUTED_METHODS <= smooth_methods


def test_non_smooth_routed_names_are_real_registry_kinds():
    # Each must resolve without raising -- an unknown kind raises ValueError.
    for kind in _NON_SMOOTH_ROUTED_KINDS:
        entry = _core.transform_registry(kind)
        assert "outputs" in entry and "params" in entry
