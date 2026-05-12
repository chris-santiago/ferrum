"""Figure-level convenience functions for common statistical plots.

Each function returns a ``Chart`` or compound view (``JointChart``,
``RepeatChart``, ``ClusterMapChart``) built from the grammar primitives.
Every parameter listed in ``ferrum-spec.md §3.14`` Group A is honored
— no ``NotImplementedError`` stubs.

Public API
----------
catplot   : Categorical plots (strip, swarm, box, violin, boxen, point, bar, count).
displot   : Univariate distribution plots (hist, kde, ecdf, rug).
jointplot : Joint-distribution view with marginals.
pairplot  : Pairwise scatterplot grid.
heatmap   : 2-D heatmap from a wide-format DataFrame.
clustermap: Clustered heatmap with row/column dendrograms.
lmplot    : Linear (and non-linear) regression scatter overlay.
residplot : Residual-diagnostic scatter plot.
"""

from ferrum.figure.distribution import displot
from ferrum.figure.categorical import catplot
from ferrum.figure.regression import lmplot, residplot
from ferrum.figure.matrix import pairplot, heatmap, clustermap
from ferrum.figure.joint import jointplot

__all__ = [
    "displot",
    "catplot",
    "lmplot",
    "residplot",
    "pairplot",
    "heatmap",
    "clustermap",
    "jointplot",
]
