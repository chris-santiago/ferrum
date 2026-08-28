"""Shared AST leaf-invariant checker for zero-import vocabulary modules.

Several modules deliberately import nothing from ``ferrum`` so they can sit
"beneath" the rest of the package without reintroducing a dependency cycle --
``ferrum._validate`` (P7) and ``ferrum.encoding._channel_policy`` (P1,
relocated from ``ferrum.chart`` in #103) are the two current examples, each
pinned by its own regression test in ``tests/test_finding_p7.py`` and
``tests/test_finding_p1.py``. This module hoists the walk both of those tests
perform (the ``tests/_snapshots.py`` / ``tests/_svg_extents.py`` precedent for
cross-cutting test helpers) so the invariant has one definition instead of
two hand-synced copies.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import ModuleType


def assert_module_is_a_leaf(module: ModuleType, *, package: str = "ferrum") -> None:
    """Assert that ``module`` imports nothing from ``package``.

    Walks the module's source AST directly (rather than inspecting
    ``sys.modules`` after the fact) so an added ``from ferrum...`` import is
    caught even if it happens to be unreachable at runtime.

    Raises
    ------
    AssertionError
        If the module contains an absolute import naming ``package`` (or a
        submodule of it), or any relative import -- a relative import
        necessarily reaches back into the module's own package, so it is
        flagged unconditionally rather than by name-prefix matching.
    """
    tree = ast.parse(Path(module.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            # A relative import (level > 0, e.g. `from . import x`) can carry
            # a `module` that doesn't start with `package` (or is None for
            # `from . import x`) while still reaching back into the package --
            # flag it directly rather than relying on name-prefix matching,
            # which misses it entirely.
            assert node.level == 0, (
                f"leaf module uses a relative import (level={node.level}): "
                f"from {'.' * node.level}{node.module or ''} import ..."
            )
            names = [node.module or ""]
        else:
            continue
        for name in names:
            assert not name.startswith(package), f"leaf module imports {name!r}"
