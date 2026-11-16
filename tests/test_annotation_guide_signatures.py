"""Drift guard: the annotation guide's signature blocks state the shipped signatures.

``docs/site/guide/concepts/annotations.md`` opens each ``ferrum.annotation`` factory's section
with a ``ann.<name>( ... )`` block listing its parameters. Each block must name the parameters
the factory takes, state every keyword default as the code has it (a colour that follows the
theme is ``None``, not a literal colour), and list a bare name only for a parameter with no
default. Example calls elsewhere on the page are not signatures and are not checked.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

import ferrum.annotation as ann

_GUIDE = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "site"
    / "guide"
    / "concepts"
    / ("annotations.md")
)
# A signature block is a python fence whose first line is ``ann.<name>(`` and whose last is ``)``.
_BLOCK = re.compile(r"```python\nann\.(\w+)\(\n(.*?)\n\)\n```", re.DOTALL)


def _blocks() -> dict[str, str]:
    return {name: body for name, body in _BLOCK.findall(_GUIDE.read_text())}


def _parse(body: str) -> tuple[list[str], dict[str, object]]:
    """The bare parameter names and the ``name=default`` pairs a block states."""
    lines = []
    for line in body.splitlines():
        code = re.sub(r"\s+#.*$", "", line).strip()
        if code and code != "*,":
            lines.append(code)
    call = ast.parse("f(" + " ".join(lines) + ")", mode="eval").body
    assert isinstance(call, ast.Call)
    bare = [arg.id for arg in call.args if isinstance(arg, ast.Name)]
    keyword = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords if kw.arg is not None}
    return bare, keyword


def test_the_guide_has_a_signature_block_for_every_factory():
    factories = {
        name
        for name, member in vars(ann).items()
        if inspect.isfunction(member) and member.__module__ == "ferrum.annotation.primitives"
    }
    assert set(_blocks()) == factories


@pytest.mark.parametrize("name", sorted(_blocks()))
def test_the_guide_states_the_shipped_signature(name):
    parameters = inspect.signature(getattr(ann, name)).parameters
    bare, keyword = _parse(_blocks()[name])
    assert set(bare) | set(keyword) == set(parameters)
    for param in bare:
        assert parameters[param].default is inspect.Parameter.empty, (
            f"the guide lists {name}({param}) as having no default"
        )
    for param, stated in keyword.items():
        assert parameters[param].default == stated, (
            f"the guide states {name}({param}={stated!r}); the code has "
            f"{parameters[param].default!r}"
        )
