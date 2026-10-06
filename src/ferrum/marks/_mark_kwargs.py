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
  against *mark_name*'s own honored-kwarg vocabulary
  (``ferrum.marks.base.mark_honors``) and raises ``ValueError`` on any key
  *mark_name* does not honor, mirroring ``MarkBase`` / ``_set_mark``
  validation for primitive marks.
- ``apply_user_mark_kwargs(mark_name, layers, kwargs)`` merges validated
  kwargs into the ``mark_kwargs`` of every layer whose mark draws them, for
  layered-mode desugars (``MarkDesugarResult.layers`` set), with
  **user-wins** semantics on conflict so explicit user intent overrides a
  desugar's hard-coded defaults (e.g. ribbon's ``opacity=0.3``) -- via
  ``_layer.apply_catch_all_style``, the package's one merger of a
  catch-all style over a layer. A kwarg no layer draws refuses.

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
    """Validate user-supplied ``mark_kwargs``; raise ``ValueError`` on keys *mark_name* doesn't honor.

    Returns the dict unchanged so callers can write
    ``user_kw = validate_user_mark_kwargs("foo", mark_kwargs)`` in one
    line.  Passes through an empty dict immediately to keep the common
    no-kwarg path free of the ``MarkBase`` import.

    Parameters
    ----------
    mark_name : str
        Name of the composite mark (e.g. ``"boxplot"``). Used both to look
        up the honored set (``ferrum.marks.base.mark_honors``) and in the
        error message.
    kwargs : dict
        Caller-supplied keyword arguments to validate. Keys are checked as
        given -- this helper does not resolve aliases (``color``→``fill``,
        ``alpha``→``opacity``); callers that need alias resolution go
        through ``MarkBase`` instead (see the module docstring).

    Returns
    -------
    dict
        A copy of ``kwargs`` (unchanged) if every key is honored.

    Raises
    ------
    EntryRefusal
        A ``ValueError`` voiced as ``mark_<mark_name>``, or as the figure
        function or visualizer that wrote the kwargs, if any key is not
        honored by *mark_name* (see ``ferrum.marks.base.honored_kwargs``).

    Examples
    --------
    >>> validate_user_mark_kwargs("boxplot", {"opacity": 0.5})
    {'opacity': 0.5}
    >>> validate_user_mark_kwargs("boxplot", {"typo": 1})  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    ferrum._entry.EntryRefusal: mark_boxplot: unknown keyword argument(s) ['typo']. ...
    """
    if not kwargs:
        return {}
    from ferrum._entry import EntryRefusal
    from ferrum.marks.base import mark_honors  # type: ignore[attr-defined]
    from ferrum.marks.base import _mark_kwargs_error_plural  # type: ignore[attr-defined]

    unknown = [k for k in kwargs if not mark_honors(mark_name, k)]
    if unknown:
        raise EntryRefusal(
            lambda entry: _mark_kwargs_error_plural(mark_name, unknown, entry=entry),
            f"mark_{mark_name}",
        )
    return dict(kwargs)


def apply_user_mark_kwargs(mark_name: str, layers: list, user_kwargs: dict) -> list:
    """Merge ``user_kwargs`` into the ``mark_kwargs`` of every layer whose mark draws them.

    User-supplied values win over the desugar's hard-coded defaults so
    explicit user intent is honored (e.g. ``opacity=0.8`` overrides a
    ribbon's built-in ``opacity=0.3``).  Each kwarg reaches only the layers
    whose mark draws it (Rust's style table), and a kwarg none of *layers*
    draws refuses, voiced as ``mark_<mark_name>``.

    The merge itself is ``_layer.apply_catch_all_style``, the same function
    ``_desugar._resolve_pending_impl`` uses to spread the chart-level
    catch-all over a layered desugar's output -- so the one exception to
    user-wins holds here too: a key the caller aimed at ONE layer (recorded
    on ``_Layer.caller_style_keys``, e.g. through
    ``mark_smooth(layer_kwargs=...)``) is not overridden by a broader
    instruction.

    Parameters
    ----------
    mark_name : str
        Name of the composite mark that emitted *layers* (e.g.
        ``"boxplot"``); the refusal speaks as ``mark_<mark_name>``.
    layers : list[_Layer]
        Layer descriptors as returned by a desugar function.
    user_kwargs : dict
        Validated user-supplied kwargs (from ``validate_user_mark_kwargs``).

    Returns
    -------
    list[_Layer]
        New list of layers with ``mark_kwargs`` updated.  The input list
        and its members are not mutated.

    Raises
    ------
    EntryRefusal
        A ``ValueError`` if a kwarg is drawn by none of *layers*' marks.

    Examples
    --------
    >>> from ferrum._layer import _Layer
    >>> layers = [_Layer(mark="ribbon", mark_kwargs={"opacity": 0.3})]
    >>> apply_user_mark_kwargs("ribbon", layers, {"opacity": 0.8})[0].mark_kwargs
    {'opacity': 0.8}
    """
    if not user_kwargs:
        return layers
    from ferrum._layer import apply_catch_all_style
    from ferrum.marks.base import refuse_style_no_layer_draws

    refuse_style_no_layer_draws(f"mark_{mark_name}", user_kwargs, [ly.mark for ly in layers])
    return [apply_catch_all_style(layer, user_kwargs) for layer in layers]


# Public re-exports — desugars should import from this module, not poke
# at the underscore-prefixed names directly. The per-mark honored
# vocabularies are ``ferrum.marks.base.honored_kwargs`` / ``mark_honors``,
# derived from Rust's style table.
__all__ = ["validate_user_mark_kwargs", "apply_user_mark_kwargs"]
