"""Shared helpers for validating user-supplied ``**mark_kwargs`` across
every composite-mark desugar in the marks package.

Diagnostic / statistical / composite marks accept arbitrary keyword
arguments via ``**mark_kwargs`` because Chart's ``mark_<name>`` methods
pass them through. Without validation, an unknown kwarg
(``mark_boxplot(strokr="red")`` — typo) silently disappears: the
renderer never sees it, no error is raised, and the user is left
guessing.

Two helpers enforce the Phase 9+ no-defer principle on this surface:

- ``validate_user_mark_kwargs(mark_name, kwargs)`` checks every key
  against the renderer-level allowlist (``MarkBase._VALID_MARK_KWARGS``)
  and raises ``TypeError`` on unknown keys, mirroring ``MarkBase`` /
  ``_set_mark`` validation for primitive marks.
- ``apply_user_mark_kwargs(layers, kwargs)`` merges validated kwargs
  into every layer's ``mark_kwargs`` dict for layered-mode desugars
  (``MarkDesugarResult.layers`` set), with **user-wins** semantics on
  conflict so explicit user intent overrides a desugar's hard-coded
  defaults (e.g. ribbon's ``opacity=0.3``).

Single-mark desugars (``desugar_density``, ``desugar_histogram``,
``desugar_smooth`` with ``ci=None``) have no per-layer mark_kwargs slot
in ``Chart._resolve_pending``; for those, callers invoke only
``validate_user_mark_kwargs`` to reject unknown kwargs, and forwarding
to the rendered mark is reserved for a future ``_resolve_pending``
extension.

A meta-test in ``tests/test_mark_kwargs_no_silent_drop.py`` AST-scans
every ``desugar_*`` function in the marks package and asserts each one
either has no ``**kwargs`` parameter or calls
``validate_user_mark_kwargs`` in its body — preventing silent-drop
regressions from being reintroduced.
"""

from __future__ import annotations


def validate_user_mark_kwargs(mark_name: str, kwargs: dict) -> dict:
    """Validate user-supplied ``mark_kwargs``; raise ``TypeError`` on unknown keys.

    Returns the dict unchanged so callers can write
    ``user_kw = validate_user_mark_kwargs("foo", mark_kwargs)`` in one
    line.  Passes through an empty dict immediately to keep the common
    no-kwarg path free of the ``MarkBase`` import.

    Parameters
    ----------
    mark_name : str
        Name of the composite mark (e.g. ``"boxplot"``). Used in the
        error message only.
    kwargs : dict
        Caller-supplied keyword arguments to validate.

    Returns
    -------
    dict
        A copy of ``kwargs`` (unchanged) if all keys are valid.

    Raises
    ------
    TypeError
        If any key is not in ``ferrum.marks.base._VALID_MARK_KWARGS``.

    Examples
    --------
    >>> validate_user_mark_kwargs("boxplot", {"opacity": 0.5})
    {'opacity': 0.5}
    >>> validate_user_mark_kwargs("boxplot", {"typo": 1})  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    TypeError: mark_boxplot: unknown keyword argument(s) ['typo']. ...
    """
    if not kwargs:
        return {}
    from ferrum.marks.base import _VALID_MARK_KWARGS  # type: ignore[attr-defined]

    unknown = [k for k in kwargs if k not in _VALID_MARK_KWARGS]
    if unknown:
        raise TypeError(
            f"mark_{mark_name}: unknown keyword argument(s) {unknown!r}. "
            f"Valid: {sorted(_VALID_MARK_KWARGS)}"
        )
    return dict(kwargs)


def apply_user_mark_kwargs(layers: list, user_kwargs: dict) -> list:
    """Merge ``user_kwargs`` into every layer's ``mark_kwargs`` dict.

    User-supplied values win over the desugar's hard-coded defaults so
    explicit user intent is honored (e.g. ``opacity=0.8`` overrides a
    ribbon's built-in ``opacity=0.3``).  The renderer silently ignores
    kwargs that don't apply to a given mark type, so spreading across all
    layers is safe even when only one layer can use the kwarg.

    Parameters
    ----------
    layers : list[_Layer]
        Layer descriptors as returned by a desugar function.
    user_kwargs : dict
        Validated user-supplied kwargs (from ``validate_user_mark_kwargs``).

    Returns
    -------
    list[_Layer]
        New list of layers with ``mark_kwargs`` updated.  The input list
        and its members are not mutated.

    Examples
    --------
    >>> from ferrum._layer import _Layer
    >>> layers = [_Layer(mark="ribbon", mark_kwargs={"opacity": 0.3})]
    >>> apply_user_mark_kwargs(layers, {"opacity": 0.8})[0].mark_kwargs
    {'opacity': 0.8}
    """
    if not user_kwargs:
        return layers
    from dataclasses import replace

    merged: list = []
    for layer in layers:
        existing = dict(layer.mark_kwargs or {})
        existing.update(user_kwargs)  # user wins
        merged.append(replace(layer, mark_kwargs=existing))
    return merged


# Public re-exports — desugars should import from this module, not poke
# at the underscore-prefixed names directly. The canonical list of kwargs
# the renderer actually honors lives in ``ferrum.marks.base._VALID_MARK_KWARGS``.
__all__ = ["validate_user_mark_kwargs", "apply_user_mark_kwargs"]
