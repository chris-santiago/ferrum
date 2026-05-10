"""Phase 9e — figure-level convenience functions.

Each function returns a Chart or compound view (JointChart, RepeatChart,
ClusterMapChart) whose .spec / .charts / .expand() is a fully-formed object.
No NotImplementedError — every parameter advertised in ferrum-spec.md §3.14
Group A is honored.
"""
from ferrum.figure.distribution import displot
from ferrum.figure.categorical import catplot
from ferrum.figure.regression import lmplot, residplot
from ferrum.figure.matrix import pairplot, heatmap, clustermap
from ferrum.figure.joint import jointplot

__all__ = [
    "displot", "catplot", "lmplot", "residplot",
    "pairplot", "heatmap", "clustermap", "jointplot",
]
