"""Load a repo script as an importable module, for tests that call into it directly.

``scripts/`` holds plain scripts, not an installed package, so a test that
exercises one (``check_spec.py``, ``regen_docs_images.py``, ``gen_api_pages.py``...)
loads it from its file path rather than importing it normally. Dataclasses
resolve their module through ``sys.modules`` while the class body runs, so
the module is registered there before its body executes.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_script(name: str, path: Path) -> ModuleType:
    """Import the script at ``path`` under ``name`` and return the loaded module."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
