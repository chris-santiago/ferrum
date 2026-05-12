"""``ComparedModelSource`` — Phase 10h multi-model wrapper.

Proxies every Phase 10 derived-data method through each underlying
``ModelSource`` and concatenates the per-model frames with a
``model: Utf8`` column stamped on each one, so chart builders can
route ``color="model"`` to overlay one curve per model.

The ``_COMPARED_METHODS`` frozenset is derived automatically from
the six per-domain mixins — when you add a public method to any
mixin it joins the multi-model dispatch surface with no further
bookkeeping.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from ._classification import ClassificationCurvesMixin
from ._clustering import ClusteringMixin
from ._importance import FeatureImportanceMixin
from ._predictions import PredictionsMixin
from ._ranking import RankingMixin
from ._selection import ModelSelectionMixin


_DOMAIN_MIXINS = (
    PredictionsMixin,
    ClassificationCurvesMixin,
    FeatureImportanceMixin,
    ModelSelectionMixin,
    ClusteringMixin,
    RankingMixin,
)


def _collect_compared_methods() -> frozenset[str]:
    """Collect every public, callable method defined directly on a domain
    mixin (not inherited from ``object``).  The frozenset becomes the
    dispatch surface for ``ComparedModelSource.__getattr__`` — adding a
    new method to any mixin makes it multi-model-aware automatically.
    """
    methods: set[str] = set()
    for cls in _DOMAIN_MIXINS:
        for name, attr in vars(cls).items():
            if name.startswith("_"):
                continue
            if callable(attr):
                methods.add(name)
    return frozenset(methods)


_COMPARED_METHODS: frozenset[str] = _collect_compared_methods()


class ComparedModelSource:
    """Multi-model wrapper exposing the same surface as ``ModelSource``.

    Every derived-data method is proxied through each underlying
    ``ModelSource`` and the per-model outputs are concatenated with a
    ``model: Utf8`` column stamped on each frame, so downstream chart
    builders can route ``color="model"`` to render one curve per model.

    ``_X``, ``_y``, ``_feature_names``, and ``_class_names`` resolve to
    the first source's values (every wrapped source shares ``X`` / ``y``
    by construction in ``ModelSource.compare``, so any one will do);
    accessing ``_model`` raises since there is no single estimator.
    ``model_names`` reports the configured ordering.

    Parameters
    ----------
    sources : dict[str, ModelSource]
        Mapping from model name (used for the ``model`` column) to the
        underlying ``ModelSource``. Must contain at least one entry —
        passing an empty dict raises ``ValueError``.

    Examples
    --------
    >>> import ferrum as fm
    >>> cms = fm.ModelSource.compare({"ridge": ridge, "lasso": lasso}, X, y)
    >>> fm.roc_chart(cms)                  # overlay both curves
    >>> cms.model_names
    ['ridge', 'lasso']
    >>> cms.roc_curve()                    # long-form frame with `model` column
    """

    __slots__ = ("_sources",)

    def __init__(self, sources):
        if not sources:
            raise ValueError("ComparedModelSource requires at least one source.")
        self._sources = dict(sources)

    @property
    def model_names(self) -> list[str]:
        """Ordered list of model display names.

        Returns the keys of the ``sources`` dict supplied at construction time,
        in insertion order. Each name corresponds to the value written into the
        ``model`` column on every derived-data DataFrame.

        Returns
        -------
        list[str]
            Model names in the order they were registered.
        """
        return list(self._sources.keys())

    def _dispatch(self, method: str, *args: Any, **kwargs: Any) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []
        for name, src in self._sources.items():
            df = getattr(src, method)(*args, **kwargs)
            frames.append(df.with_columns(pl.lit(name).alias("model")))
        return pl.concat(frames, how="vertical_relaxed")

    def __getattr__(self, name: str) -> Any:
        # __slots__ makes attribute access strict; the runtime falls through
        # to __getattr__ only for unknown names. We route a frozen list of
        # ModelSource methods through `_dispatch`, expose `_X`/`_y` from the
        # first wrapped source (chart builders sometimes need them), and
        # explicitly forbid `_model` access since the answer would be
        # nonsensical for a multi-model wrapper.
        if name in _COMPARED_METHODS:
            method = name
            return lambda *args, **kwargs: self._dispatch(method, *args, **kwargs)
        if name in (
            "_X",
            "_y",
            "_feature_names",
            "_class_names",
            "_capabilities",
            "X",
            "y",
            "feature_names",
            "capabilities",
        ):
            return getattr(next(iter(self._sources.values())), name)
        if name in ("_model", "model"):
            raise AttributeError(
                "ComparedModelSource has no single model. Iterate "
                "ComparedModelSource._sources.values() to access each "
                "wrapped ModelSource's model."
            )
        raise AttributeError(
            f"ComparedModelSource has no attribute {name!r}. "
            f"Methods routed through .compare: {sorted(_COMPARED_METHODS)}"
        )

    def __repr__(self) -> str:
        return f"ComparedModelSource({list(self._sources.keys())!r})"
