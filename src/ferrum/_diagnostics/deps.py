"""Lazy-import helpers for Phase 10 optional dependencies.

Each `require_*` function is called as the first line of any ModelSource
method that needs the corresponding third-party library. `import ferrum`
and `ModelSource.__init__` never call these helpers.
"""
from __future__ import annotations

from types import ModuleType


def require_sklearn(method_name: str) -> ModuleType:
    """Lazy-import sklearn; raise with `ferrum[models]` hint on failure."""
    try:
        import sklearn
    except ImportError as e:
        raise ImportError(
            f"ferrum.ModelSource.{method_name}() requires scikit-learn. "
            f"Install it with `pip install ferrum[models]` or "
            f"`pip install scikit-learn`."
        ) from e
    return sklearn


def require_shap(method_name: str) -> ModuleType:
    try:
        import shap
    except ImportError as e:
        raise ImportError(
            f"ferrum.ModelSource.{method_name}() requires the shap library. "
            f"Install it with `pip install ferrum[shap]` or `pip install shap`."
        ) from e
    return shap


def require_umap(method_name: str) -> ModuleType:
    try:
        import umap
    except ImportError as e:
        raise ImportError(
            f"ferrum.ModelSource.{method_name}() requires umap-learn. "
            f"Install it with `pip install ferrum[umap]` or `pip install umap-learn`."
        ) from e
    return umap
