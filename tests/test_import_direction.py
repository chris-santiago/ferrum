"""Core modules must never import the figure layer (``ferrum.plots``).

``assert_no_module_imports`` (``tests/_leaf_guard.py``) walks every core
module's full AST and flags an import of ``ferrum.plots`` under any
spelling: a direct ``import ferrum.plots[.x]``; an absolute or relative
``from ... import ...`` naming the submodule directly, module-level or
nested inside a function body; a name bound at the package root whose own
``__init__.py`` re-exports it from ``ferrum.plots``, so a lazy ``from
ferrum import <figure function>`` is caught even though neither the
module name nor the imported name literally mentions ``plots``; and a
root-level star import (``from ferrum import *``), which binds every one
of those re-exported names at once. A root import of a genuine core name
(``from ferrum import Bin``) stays allowed, as does any import of
``ferrum.plots`` from ``plots/``, ``diagnostics/`` or the package's own
``__init__.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import ferrum
from tests._leaf_guard import assert_no_module_imports

_ALLOWED = ("plots/", "diagnostics/", "__init__.py")


def test_core_never_imports_the_figure_layer() -> None:
    package_dir = Path(ferrum.__file__).parent
    assert_no_module_imports(package_dir, forbidden="ferrum.plots", allowed=_ALLOWED)


def _write_package(root: Path, files: dict[str, str]) -> Path:
    package_dir = root / "ferrum"
    for rel_path, source in files.items():
        full_path = package_dir / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(source)
    return package_dir


#: A synthetic ``ferrum`` package shaped like the real one for the guard's
#: purposes: a root ``__init__.py`` that binds one core name from a core
#: module (mirroring ``from ferrum import Bin``) and one figure-layer name
#: from ``plots`` (mirroring ``from ferrum import displot``), plus the
#: ``plots/``, ``diagnostics/`` and ``marks/`` subpackages every case below
#: reads or writes into.
_BASE_TREE = {
    "__init__.py": (
        "from ferrum._core import Bin\n"
        "from ferrum.plots.distribution import displot\n"
        "import ferrum.plots as plots\n"
    ),
    "plots/__init__.py": "from ferrum.plots.distribution import displot\n",
    "plots/distribution.py": "def displot():\n    pass\n",
    "diagnostics/__init__.py": "",
    "marks/__init__.py": "",
}


def _tree_with(tmp_path: Path, extra: dict[str, str]) -> Path:
    return _write_package(tmp_path, {**_BASE_TREE, **extra})


_FLAGGED_SPELLINGS: dict[str, tuple[str, str]] = {
    "function-local absolute import": (
        "marks/_leaky.py",
        "def resolve():\n    from ferrum.plots.distribution import displot\n    return displot\n",
    ),
    "import ferrum.plots": (
        "marks/_leaky.py",
        "import ferrum.plots\n",
    ),
    "import ferrum.plots.distribution": (
        "marks/_leaky.py",
        "import ferrum.plots.distribution\n",
    ),
    "from ferrum import plots": (
        "marks/_leaky.py",
        "def resolve():\n    from ferrum import plots\n    return plots\n",
    ),
    "from . import plots, top-level module": (
        "_leaky.py",
        "def resolve():\n    from . import plots\n    return plots\n",
    ),
    "from .. import plots, nested module": (
        "marks/_leaky.py",
        "def resolve():\n    from .. import plots\n    return plots\n",
    ),
    "a root-level re-export of a figure-layer name": (
        "marks/_leaky.py",
        "def resolve():\n    from ferrum import displot\n    return displot\n",
    ),
    "a root-level star import": (
        "marks/_leaky.py",
        "from ferrum import *\n\n\ndef resolve():\n    return displot\n",
    ),
}


@pytest.mark.parametrize(
    "rel_path, source", _FLAGGED_SPELLINGS.values(), ids=list(_FLAGGED_SPELLINGS)
)
def test_every_spelling_of_a_figure_layer_import_is_flagged(
    tmp_path: Path, rel_path: str, source: str
) -> None:
    package_dir = _tree_with(tmp_path, {rel_path: source})
    with pytest.raises(AssertionError, match=rel_path):
        assert_no_module_imports(package_dir, forbidden="ferrum.plots", allowed=_ALLOWED)


def test_a_root_level_import_of_a_core_name_is_not_flagged(tmp_path: Path) -> None:
    """``from ferrum import Bin`` stays allowed.

    ``Bin`` is bound at the root from a core module, not re-exported from
    ``ferrum.plots``, the same shape as the real package's ``from ferrum
    import Aggregate``/``Repeat``/``config``/``ChartSpec``.
    """
    package_dir = _tree_with(
        tmp_path,
        {
            "marks/_uses_core_root_import.py": (
                "def resolve():\n    from ferrum import Bin\n    return Bin\n"
            )
        },
    )
    assert_no_module_imports(package_dir, forbidden="ferrum.plots", allowed=_ALLOWED)


def test_an_import_of_the_figure_layer_from_an_allowed_location_is_not_flagged(
    tmp_path: Path,
) -> None:
    """``plots/``, ``diagnostics/`` and the package's own ``__init__.py`` may import it."""
    package_dir = _write_package(
        tmp_path,
        {
            "__init__.py": "from ferrum.plots import displot\n",
            "plots/__init__.py": "from ferrum.plots.distribution import displot\n",
            "plots/distribution.py": "def displot():\n    pass\n",
            "diagnostics/__init__.py": "",
            "diagnostics/visualizers.py": (
                "def build():\n"
                "    from ferrum.plots.classification import roc_chart\n"
                "    return roc_chart\n"
            ),
        },
    )
    assert_no_module_imports(package_dir, forbidden="ferrum.plots", allowed=_ALLOWED)
