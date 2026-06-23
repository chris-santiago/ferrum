"""Diagnostic / model-selection mark methods for Chart, composed by domain.

``DiagnosticMarksMixin`` is a facade that composes the six per-domain mixins in
``ferrum.marks._chart_mixins`` -- one per diagnostic domain, mirroring the
desugar split in ``ferrum.marks.diagnostic``.  Each per-domain mixin holds
exactly the ``mark_*`` methods that delegate to that domain's desugars; the
facade unions them into the single mixin ``Chart`` inherits from.

The per-domain mixins are pure method bags (no ``__init__``/``__slots__``) and
do not define overlapping method names, so the base-class order is immaterial to
method resolution.  ``Chart`` imports ``DiagnosticMarksMixin`` from here.
"""

from __future__ import annotations

from ferrum.marks._chart_mixins import (
    ClassificationMarksMixin,
    ClusteringMarksMixin,
    ExplanationMarksMixin,
    RankingMarksMixin,
    RegressionMarksMixin,
    SelectionMarksMixin,
)


class DiagnosticMarksMixin(
    RegressionMarksMixin,
    ClassificationMarksMixin,
    ExplanationMarksMixin,
    SelectionMarksMixin,
    ClusteringMarksMixin,
    RankingMarksMixin,
):
    """Mixin providing diagnostic / model-selection / clustering mark methods.

    Facade composing the six per-domain mixins; carries no methods of its own.
    """
