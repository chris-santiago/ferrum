"""Per-domain Chart mixins for diagnostic / model-selection mark methods.

The diagnostic ``mark_*`` methods are partitioned by the desugar domain each
delegates to, mirroring ``ferrum.marks.diagnostic``'s six domain modules.  The
``DiagnosticMarksMixin`` facade in
``ferrum.marks._chart_methods_diagnostic`` composes these into the single mixin
``Chart`` inherits from.

Each mixin is a pure method bag: no ``__init__``/``__slots__``/shared state.
Methods reference Chart internals (``self._set_composite_mark`` etc.) resolved
at runtime via MRO.
"""

from ferrum.marks._chart_mixins._classification import ClassificationMarksMixin
from ferrum.marks._chart_mixins._clustering import ClusteringMarksMixin
from ferrum.marks._chart_mixins._explanation import ExplanationMarksMixin
from ferrum.marks._chart_mixins._ranking import RankingMarksMixin
from ferrum.marks._chart_mixins._regression import RegressionMarksMixin
from ferrum.marks._chart_mixins._selection import SelectionMarksMixin

__all__ = [
    "ClassificationMarksMixin",
    "ClusteringMarksMixin",
    "ExplanationMarksMixin",
    "RankingMarksMixin",
    "RegressionMarksMixin",
    "SelectionMarksMixin",
]
