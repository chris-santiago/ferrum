"""Tests for the shared SVG parsing helpers in `tests/_svg_rects.py`.

Pins the strokeless-`<line>` edge case in `rug_tick_x1_by_stroke` (review-lite
finding, wave 4 close): a `<line>` carrying no `stroke` attribute must be
excluded by the `keep` predicate, not admitted and then raise `KeyError` from
the unconditional `attrs["stroke"]` lookup. No current `ferrum-core` renderer
path emits a strokeless `<line>` today -- both `SvgBuilder::line` and
`::gridline` (`crates/ferrum-core/src/render/svg.rs`) unconditionally write a
`stroke` attribute, since `Stroke.stroke` is a required `Color`, not an
`Option` -- but `polylines`, parsing the same `svg_elements` machinery ten
lines above `rug_tick_x1_by_stroke` in `_svg_rects.py`, already treats a
missing `stroke` as reachable and states its own handling (`attrs.get`
instead of `attrs[...]`). This test pins `rug_tick_x1_by_stroke` to the same
contract, so a future renderer change that starts emitting a strokeless
`<line>` (e.g. an unstyled chrome or annotation line) hits a stated
exclusion instead of a parser crash.
"""

from __future__ import annotations

from tests._svg_rects import rug_tick_x1_by_stroke


def test_rug_tick_x1_by_stroke_excludes_line_with_no_stroke_attribute():
    svg = (
        '<line x1="1" y1="0" x2="1" y2="10" stroke="#ff0000"/><line x1="2" y1="0" x2="2" y2="10"/>'
    )
    assert rug_tick_x1_by_stroke(svg) == {"#ff0000": {1.0}}
