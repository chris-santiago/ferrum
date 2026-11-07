"""Shared AST import-invariant checkers for modules that must not import a given boundary.

Several modules deliberately import nothing from ``ferrum`` so they can sit
"beneath" the rest of the package without reintroducing a dependency cycle --
``ferrum._validate`` (P7) and ``ferrum.encoding._channel_policy`` (P1,
relocated from ``ferrum.chart`` in #103) are the two current examples, each
pinned by its own regression test in ``tests/test_finding_p7.py`` and
``tests/test_finding_p1.py``. :func:`assert_module_is_a_leaf` hoists the walk
both of those tests perform (the ``tests/_snapshots.py`` / ``tests/_svg_extents.py``
precedent for cross-cutting test helpers) so that invariant has one
definition instead of two hand-synced copies.

:func:`assert_no_module_imports` is the directory-wide sibling: a layering
boundary between two subpackages of ``ferrum`` (the core modules must never
import the figure layer, ``ferrum.plots``) rather than a single leaf module's
total import ban. Both read every name an AST node binds through
:func:`_bound_imports`, the one place either dispatches on ``ast.Import``
vs. ``ast.ImportFrom``; neither re-inspects the node itself afterward.
"""

from __future__ import annotations

import ast
import importlib.util
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType


def _bound_imports(node: ast.AST, anchor_package: str) -> Iterator[tuple[str, str]]:
    """Yield ``(resolved_module, bound_name)`` for every name *node* binds.

    *resolved_module* is the absolute dotted path the binding ultimately
    names: the imported path itself for ``import a.b.c``, or
    ``base.name`` for ``from base import name`` (a relative *base*
    resolved against *anchor_package* via
    :func:`importlib.util.resolve_name` first). *bound_name* is the local
    name the import binds (``alias.asname`` when given, else the written
    name -- ``"*"`` for a star import, which binds every name the target
    module exports). This is the one place either checker below
    dispatches on ``ast.Import`` vs. ``ast.ImportFrom``; every other
    function reads only the pairs it yields, never the node itself.
    Yields nothing for a node that isn't an import.
    """
    if isinstance(node, ast.Import):
        for alias in node.names:
            yield alias.name, (alias.asname or alias.name)
        return
    if isinstance(node, ast.ImportFrom):
        if node.level == 0:
            base = node.module or ""
        else:
            relative = "." * node.level + (node.module or "")
            base = importlib.util.resolve_name(relative, anchor_package)
        for alias in node.names:
            resolved = f"{base}.{alias.name}" if base else alias.name
            yield resolved, (alias.asname or alias.name)


def assert_module_is_a_leaf(module: ModuleType, *, package: str = "ferrum") -> None:
    """Assert that ``module`` imports nothing from ``package``.

    Walks the module's source AST directly (rather than inspecting
    ``sys.modules`` after the fact) so an added ``from ferrum...`` import
    is caught even if it happens to be unreachable at runtime. A relative
    import is resolved against the module's own ``__package__`` first
    (:func:`_bound_imports`), the same way :func:`assert_no_module_imports`
    resolves one, so it is checked by name like any other import rather
    than flagged unconditionally.

    Raises
    ------
    AssertionError
        If the module contains an import -- absolute or relative -- naming
        ``package`` or a submodule of it.
    """
    tree = ast.parse(Path(module.__file__).read_text())
    anchor_package = module.__package__ or ""
    for node in ast.walk(tree):
        for resolved, _bound_name in _bound_imports(node, anchor_package):
            assert not resolved.startswith(package), f"leaf module imports {resolved!r}"


def _dotted_module_name(path: Path, package_dir: Path, root_package: str) -> str:
    """The dotted import name *path* (a file under *package_dir*) is loaded as."""
    rel_parts = path.relative_to(package_dir).with_suffix("").parts
    if rel_parts[-1] == "__init__":
        rel_parts = rel_parts[:-1]
    return ".".join((root_package, *rel_parts))


def _anchor_package(dotted: str, *, is_init: bool) -> str:
    """The ``__package__`` a module named *dotted* resolves its own relative imports against."""
    if is_init or "." not in dotted:
        return dotted
    return dotted.rsplit(".", 1)[0]


def _root_import_origins(package_dir: Path, root_package: str) -> dict[str, str]:
    """Map each name *root_package*'s own ``__init__.py`` binds to the absolute module it imports it from.

    A ``from <root_package> import <name>`` written anywhere else in the
    tree binds whatever ``<root_package>/__init__.py`` itself bound to
    ``<name>`` -- reading that one file's own imports resolves a lazy
    re-export of a figure-layer name (``from ferrum import displot``) to
    its true origin (``ferrum.plots.displot``) without actually importing
    the package, the same way ``displot.__module__`` would at runtime.
    Returns an empty map when *package_dir* has no ``__init__.py``.
    """
    init_path = package_dir / "__init__.py"
    if not init_path.exists():
        return {}
    origins: dict[str, str] = {}
    tree = ast.parse(init_path.read_text())
    for node in ast.walk(tree):
        for resolved, bound_name in _bound_imports(node, root_package):
            origins[bound_name] = resolved
    return origins


def assert_no_module_imports(
    package_dir: Path, *, forbidden: str, allowed: tuple[str, ...] = ()
) -> None:
    """Assert that no module under *package_dir* imports *forbidden* (or a submodule of it).

    Walks every ``.py`` file under *package_dir*, skipping one whose path
    relative to *package_dir* starts with one of *allowed*'s prefixes --
    the subpackage that legitimately owns *forbidden*, or a package's own
    ``__init__.py``, which re-exports it for callers. Each remaining
    file's full AST is walked exactly as :func:`assert_module_is_a_leaf`
    walks a single module (:func:`_bound_imports`), so a lazy,
    function-local import is caught the same way a module-level one is,
    under every spelling: a direct ``import forbidden[.x]``; an absolute
    or relative ``from <base> import <name>`` naming *forbidden* either as
    ``<base>`` or as ``<base>.<name>`` (so ``from ferrum import plots``,
    ``from . import plots`` and ``from .. import plots`` are all caught,
    a relative import resolved to its absolute dotted name via
    :func:`importlib.util.resolve_name` anchored on the importing
    module's own ``__package__``); a name imported straight off the
    package root (``from <root_package> import <name>``) whose own
    ``__init__.py`` re-exports it from *forbidden*
    (:func:`_root_import_origins`), so a lazy ``from ferrum import
    displot`` is caught too, while a root import of a genuine core name
    (``from ferrum import Bin``) stays allowed; and a root-level star
    import (``from ferrum import *``), which binds every name the root
    re-exports, flagged if any of them originates in *forbidden*.

    Raises
    ------
    AssertionError
        If a file outside *allowed* imports *forbidden* or one of its
        submodules, under any of the spellings above.
    """
    root_package = package_dir.name
    root_prefix = f"{root_package}."
    root_origins = _root_import_origins(package_dir, root_package)
    for path in sorted(package_dir.rglob("*.py")):
        rel = path.relative_to(package_dir)
        if any(rel.as_posix().startswith(prefix) for prefix in allowed):
            continue
        is_init = rel.name == "__init__.py"
        dotted = _dotted_module_name(path, package_dir, root_package)
        anchor_package = _anchor_package(dotted, is_init=is_init)
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            for resolved, _bound_name in _bound_imports(node, anchor_package):
                candidates = [resolved]
                if resolved.startswith(root_prefix):
                    root_name = resolved[len(root_prefix) :]
                    if root_name == "*":
                        candidates.extend(root_origins.values())
                    else:
                        origin = root_origins.get(root_name)
                        if origin is not None:
                            candidates.append(origin)
                for name in candidates:
                    assert name != forbidden and not name.startswith(f"{forbidden}."), (
                        f"{rel.as_posix()} imports {name!r}, which only "
                        f"{', '.join(allowed)} may import"
                    )
