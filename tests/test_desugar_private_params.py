"""A ``_``-prefixed desugar parameter is private state a prep step injects, never a user kwarg.

``_desugar._split_style_kwargs`` leaves every ``_``-prefixed parameter of a
desugar function out of the transform-kwarg set, so a user key of that name
routes to style and ``MarkBase`` refuses it as unknown. That is only sound
while each such parameter has a ``_PENDING_PREP`` step that supplies it from
private chart state: a ``_``-prefixed parameter no step injects would be
unreachable by the caller and silently stuck at its default.

The desugar population is derived (every module-level ``desugar_*`` function
in ``ferrum.marks``), and so is the injected set (every ``_``-prefixed string
key a registered prep step writes into a dict), so a new parameter or a new
step joins the check without editing this file.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import textwrap
from typing import Callable, Iterable

import ferrum.marks
from ferrum._desugar import _PENDING_PREP, _split_style_kwargs

_NAMED_KINDS = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


def _desugar_functions() -> dict[str, Callable]:
    found: dict[str, Callable] = {}
    for info in pkgutil.walk_packages(ferrum.marks.__path__, prefix="ferrum.marks."):
        module = importlib.import_module(info.name)
        for name, obj in vars(module).items():
            if (
                name.startswith("desugar_")
                and inspect.isfunction(obj)
                and obj.__module__ == module.__name__
            ):
                found[f"{module.__name__}.{name}"] = obj
    return found


def _private_params(fn: Callable) -> set[str]:
    params = inspect.signature(fn).parameters.values()
    return {p.name for p in params if p.kind in _NAMED_KINDS and p.name.startswith("_")}


def _injected_names(steps: Iterable[Callable]) -> set[str]:
    """Every ``_``-prefixed string literal a step uses as a dict key."""
    names: set[str] = set()
    for step in steps:
        tree = ast.parse(textwrap.dedent(inspect.getsource(step)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                names.update(
                    key.value
                    for key in node.keys
                    if isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and key.value.startswith("_")
                )
    return names


def _uninjected(functions: dict[str, Callable], injected: set[str]) -> dict[str, set[str]]:
    out = {name: _private_params(fn) - injected for name, fn in functions.items()}
    return {name: missing for name, missing in out.items() if missing}


_PREP_STEPS = {step for prep in _PENDING_PREP.values() for step in prep.pre}
_HEX = "ferrum.marks.heavy_stat.desugar_hex"


def test_every_private_desugar_parameter_is_injected_by_a_prep_step() -> None:
    functions = _desugar_functions()
    injected = _injected_names(_PREP_STEPS)
    # The population is every desugar the dispatcher can reach, and it holds
    # a private parameter: ``desugar_hex``'s ``_color``, which
    # ``_prep_hex_field_from_color`` supplies. The control below shows an
    # unsupplied one would be caught.
    assert _private_params(functions[_HEX]) == {"_color"}
    assert "_color" in injected
    assert _uninjected(functions, injected) == {}


def test_a_private_parameter_no_step_injects_is_caught() -> None:
    """Control: the check fails for a ``_``-prefixed parameter nothing supplies."""

    def desugar_future(x_field, y_field, *, bandwidth=None, _unsupplied=None):
        raise AssertionError("never called")

    injected = _injected_names(_PREP_STEPS)
    assert _uninjected({"desugar_future": desugar_future}, injected) == {
        "desugar_future": {"_unsupplied"}
    }


def test_a_user_key_named_like_a_private_parameter_routes_to_style() -> None:
    """The partition itself: a user ``_color=`` is a style key, not a transform kwarg."""
    hex_desugar = _desugar_functions()[_HEX]
    transform, style = _split_style_kwargs(hex_desugar, {"_color": "spoof", "bin_size": 0.5})
    assert transform == {"bin_size": 0.5}
    assert style == {"_color": "spoof"}
