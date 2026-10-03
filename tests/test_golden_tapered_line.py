"""Golden SVG tests for tapered lines (`mark_line` with an encoded `size`).

A sized line draws one filled outline per series whose thickness at each
vertex is the dot diameter `2*sqrt(size_scale(v)/pi)` of a point drawn at the
same value, and its size legend draws line swatches at those widths. The
cases (built in ``tests/_tapered_line_charts.py``) cover the straight, curved
and step interpolations, a dashed taper, a half-opacity hairpin whose overlap
must paint once, color groups each tapering on their own, a primary-y break
cutting a taper, and facets. Every chart's size column rises left to right,
so a correct render thickens toward the right and its legend swatches thicken
down the legend.

These are byte-comparison goldens (mirroring
``tests/test_golden_secondary_y_axis.py``), not the behavioral assertions in
``tests/test_tapered_line.py``.

Run with ``FERRUM_UPDATE_GOLDENS=1`` to regenerate the on-disk goldens through
``tests._snapshots.regen_and_verify``, which also rasterizes each PNG for
inspection; the default behavior is byte-identical comparison.
"""

from __future__ import annotations

import pytest

from tests._snapshots import check_or_update_svg_golden
from tests._tapered_line_charts import GOLDEN_CHARTS, GOLDENS_DIR


def test_every_golden_on_disk_has_a_chart() -> None:
    assert {p.name for p in GOLDENS_DIR.glob("*.svg")} == set(GOLDEN_CHARTS)


@pytest.mark.parametrize("name", sorted(GOLDEN_CHARTS))
def test_tapered_line_golden(name: str) -> None:
    check_or_update_svg_golden(GOLDENS_DIR, name, GOLDEN_CHARTS[name]().to_svg())
