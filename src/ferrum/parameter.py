"""Reactive-parameter API for ferrum charts (D6).

This module defines the base Parameter class and the VariableParameter concrete
type.  Selection (in ferrum.selection) also subclasses Parameter; that module
imports from here to keep the dependency one-directional (parameter.py must
never import selection.py).

Classes
-------
Parameter         — plain base class; both VariableParameter and Selection
                    are subtypes.  ``isinstance(x, Parameter)`` is the
                    reference-site discriminator.
VariableParameter — a frozen dataclass for named scalar/array variables with
                    an optional initial value and optional bind target.

Functions
---------
param             — construct a VariableParameter (the fm.param entry point).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class Parameter:
    """Base class for all ferrum reactive parameters.

    Both ``VariableParameter`` and ``Selection`` are subtypes.
    ``isinstance(x, Parameter)`` is the canonical discriminator used at
    reference sites (e.g. scale domain, transform filter, conditional
    encoding) to detect that a value is a parameter reference rather than a
    literal.

    Subclasses must implement ``to_param_spec_dict()``.  The base raises
    ``NotImplementedError`` only as an unreachable guard — every concrete
    type (``VariableParameter``, ``Selection``) provides its own override.

    Attributes
    ----------
    name : str
        Stable identifier used across the chart spec to link parameters to
        their reference sites.

    Examples
    --------
    >>> import ferrum as fm
    >>> p = fm.param("threshold", value=50)
    >>> isinstance(p, fm.Parameter)
    True
    >>> p.ref()
    {'param': 'threshold'}
    """

    name: str

    def ref(self) -> dict:
        """Return the reference marker dict used at parameter reference sites.

        Returns
        -------
        dict
            ``{"param": self.name}`` — consumed by scale.domain, filter
            predicates, and conditional value sites.

        Examples
        --------
        >>> import ferrum as fm
        >>> fm.param("brush").ref()
        {'param': 'brush'}
        """
        return {"param": self.name}

    def to_param_spec_dict(self) -> dict:
        """Serialize to the wire dict emitted into the ``params`` spec array.

        Subclasses must override this.  This base implementation is an
        unreachable guard — concrete types (``VariableParameter``,
        ``Selection``) each supply their own implementation.

        Returns
        -------
        dict
            Entry in the top-level ``params`` spec array.  Shape varies by
            subtype; see ``VariableParameter.to_param_spec_dict`` and
            ``Selection.to_param_spec_dict`` for the concrete shapes.

        Raises
        ------
        NotImplementedError
            Always (guard for subclasses that forget to override).
        """
        raise NotImplementedError(f"{type(self).__name__} must implement to_param_spec_dict()")


def _normalize_bind(bind: Any) -> Any:
    """Normalize a bind argument to its canonical form, or None.

    Passes ``"legend"`` and dict bind objects through unchanged.  Returns
    ``None`` for ``None`` (the common no-bind case).

    Parameters
    ----------
    bind : str, dict, or None
        The raw bind value from ``param(..., bind=...)`` or
        ``selection_point(..., bind=...)``.

    Returns
    -------
    str, dict, or None
        The canonical bind value, or ``None`` when no bind is configured.
    """
    return bind  # pass-through; None, "legend", and dicts are all valid


@dataclass(frozen=True)
class VariableParameter(Parameter):
    """A named scalar or array parameter with an optional initial value.

    Created via ``fm.param()``.  Do not construct directly.

    Attributes
    ----------
    name : str
        Stable parameter identifier.
    value : Any, default None
        Static initial value (serialized into the spec as the starting state
        for the WASM runtime and the static renderer).  ``None`` is a valid
        initial value — the key is always emitted.
    bind : str, dict, or None, default None
        Optional bind target.  ``"legend"`` connects the parameter to a
        legend widget.  A dict specifies a full Vega-Lite-style input
        binding (e.g. ``{"input": "range", "min": 0, "max": 100}``).
        When ``None`` the ``"bind"`` key is omitted from the wire dict.

    Examples
    --------
    >>> import ferrum as fm
    >>> p = fm.param("threshold", value=50)
    >>> p.to_param_spec_dict()
    {'name': 'threshold', 'kind': 'variable', 'value': 50}

    >>> fm.param("t", value=0, bind="legend").to_param_spec_dict()
    {'name': 't', 'kind': 'variable', 'value': 0, 'bind': 'legend'}
    """

    name: str
    value: Any = None
    bind: Any = None

    def to_param_spec_dict(self) -> dict:
        """Serialize to the params-array wire dict.

        ``"bind"`` is omitted when ``None`` to avoid emitting ``null``
        clutter in the spec JSON.  ``"value"`` is always emitted (it is the
        static initial value and may legitimately be ``None``).

        Returns
        -------
        dict
            Wire shape::

                {"name": str, "kind": "variable", "value": Any}
                {"name": str, "kind": "variable", "value": Any, "bind": str | dict}

        Examples
        --------
        >>> import ferrum as fm
        >>> fm.param("x", value=10).to_param_spec_dict()
        {'name': 'x', 'kind': 'variable', 'value': 10}
        """
        normalized = _normalize_bind(self.bind)
        d: dict[str, Any] = {
            "name": self.name,
            "kind": "variable",
            "value": self.value,
        }
        if normalized is not None:
            d["bind"] = normalized
        return d


def param(name: str, value: Any = None, bind: Any = None) -> VariableParameter:
    """Construct a named variable parameter.

    A ``VariableParameter`` declares a named, mutable scalar or array
    parameter that can be referenced at scale domains, filter predicates,
    and conditional encoding values.  At static render time the initial
    ``value`` is used; in interactive WASM exports the parameter is driven
    by its ``bind`` widget or by a linked selection.

    Parameters
    ----------
    name : str
        Stable identifier for the parameter.  Must be unique within the
        chart.  Used in ``ref()`` markers at reference sites and in the
        emitted ``params`` spec array.
    value : Any, default None
        Static initial value.  Serialized as the starting state for both
        the static renderer and the WASM runtime.  ``None`` is a valid
        choice; the key is always emitted.
    bind : str, dict, or None, default None
        Optional bind target controlling how the parameter is driven at
        runtime.

        - ``None`` (default) — no binding; parameter is static unless
          updated programmatically.
        - ``"legend"`` — binds the parameter to legend-entry click
          interactions.
        - ``dict`` — full input-binding spec (e.g.
          ``{"input": "range", "min": 0, "max": 100, "step": 1}``).

    Returns
    -------
    VariableParameter
        Frozen, immutable parameter descriptor.  Pass to
        ``Chart.add_param()`` (coming in a later sub-task) and reference
        via ``p.ref()`` at scale domain or conditional sites.

    Examples
    --------
    Static threshold parameter:

    >>> import ferrum as fm
    >>> thresh = fm.param("threshold", value=50)
    >>> thresh.to_param_spec_dict()
    {'name': 'threshold', 'kind': 'variable', 'value': 50}

    Range-slider binding:

    >>> slider = fm.param(
    ...     "k",
    ...     value=3,
    ...     bind={"input": "range", "min": 1, "max": 20, "step": 1},
    ... )

    Legend bind:

    >>> legend_param = fm.param("series", bind="legend")
    """
    return VariableParameter(name=name, value=value, bind=bind)
