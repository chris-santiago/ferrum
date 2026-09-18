"""Net-new coverage for #194: ``displot(multiple=)`` decides binning too.

``displot`` built its own position adjustment from ``multiple=`` and never
forwarded the mode into ``mark_histogram``, so the ``Bin`` transform always ran
at ``shared_extent=False``.  Under ``multiple="stack"``/``"fill"`` each group
therefore binned over its own data range: the two groups' bars shared no x
position and were not even the same width, ``apply_stack`` had no pairs to
stack, and the render was a side-by-side chart wearing a stack's name.

Fix: one owner (``src/ferrum/marks/_multiple.py``'s ``multiple_semantics``)
maps the mode to BOTH the position adjustment and the binning extent; the
histogram construction and the kde overlay each read that one pair.
``displot`` forwards the mode and its own ``position_by=`` so the caller's
``by=hue`` grouping survives the desugar's position.

Scope: this module pins the mode-to-geometry contract for the histogram kind,
plus the input contract of the one public door this fix opened
(``mark_histogram(position_by=)``). The overlay's own scaling lives in
``test_displot_kde_overlay.py``.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from tests._svg_rects import data_bar_rects

#: Modes whose groups must land on one shared bin grid, and modes that keep
#: per-group extents. A stack needs shared edges to have anything to stack.
SHARED_EDGE_MODES = ("stack", "fill")
PER_GROUP_EDGE_MODES = ("layer", "dodge")


@pytest.fixture
def two_group_df():
    """Two well-separated normal samples tagged by ``g``.

    Well separated on purpose: with means 0 and 6 the groups' own extents
    barely overlap, so "each group binned over its own range" and "both
    groups binned over the global range" put the bars in unmistakably
    different places.
    """
    rng = np.random.default_rng(0)
    n = 300
    return pl.DataFrame(
        {
            "x": np.concatenate([rng.normal(0, 1, n), rng.normal(6, 1, n)]),
            "g": ["a"] * n + ["b"] * n,
        }
    )


def _bar_geometry_by_fill(svg: str) -> dict[str, tuple[set[float], set[float]]]:
    """Map each bar fill color to its ``(set of x starts, set of widths)``."""
    out: dict[str, tuple[set[float], set[float]]] = {}
    for bar in data_bar_rects(svg):
        xs, widths = out.setdefault(bar["fill"], (set(), set()))
        xs.add(round(float(bar["x"]), 2))
        widths.add(round(float(bar["width"]), 2))
    return out


def _bin_transform(chart) -> dict:
    spec = json.loads(chart.to_spec().to_json())
    return next(t for t in spec["transforms"] if t.get("type") == "bin")


def _kde_overlay_transform(chart) -> dict:
    spec = json.loads(chart.to_spec().to_json())
    return next(t for t in spec["transforms"] if t.get("name") == "kde_overlay")


class TestSharedBinEdges:
    """Under stack and fill, every group's bars share bin edges and widths."""

    @pytest.mark.parametrize("multiple", SHARED_EDGE_MODES)
    def test_groups_share_bin_edge_positions_and_widths(self, two_group_df, multiple):
        """One edge grid, one bar width, and keys both groups actually share.

        RE-EXPRESSED (fix round 1), not re-baselined. This asserted x-start
        SET equality between the groups, which is a claim about a complete
        (bin x group) rect grid reaching the SVG -- something the contract
        never promised and that sibling task 10b deliberately stopped
        emitting (a group with no rows in a bin no longer serializes an
        invisible zero-height rect). The three properties below are what
        "the groups share bin edges and the bars are stackable" actually
        means, and they are equally false on the pre-fix output: pre-fix the
        two groups binned over their own ranges, so the union of their starts
        carried two incommensurable spacings and their widths differed
        (31.34 px vs 34.19 px, measured).
        """
        svg = fm.displot(two_group_df, x="x", hue="g", kind="hist", multiple=multiple).to_svg()
        geometry = _bar_geometry_by_fill(svg)
        assert len(geometry) == 2, f"expected bars for 2 hue levels, got {len(geometry)}"
        (fill_a, (xs_a, widths_a)), (fill_b, (xs_b, widths_b)) = sorted(geometry.items())

        # 1. One edge grid. Every start in the union sits on a single
        #    spacing, so a bin absent from one group (or from both) leaves a
        #    whole-multiple gap rather than shifting the grid.
        starts = sorted(xs_a | xs_b)
        gaps = [b - a for a, b in zip(starts, starts[1:])]
        assert gaps, f"multiple={multiple!r}: fewer than two distinct bar starts"
        spacing = min(gaps)
        off_grid = [g for g in gaps if abs(g / spacing - round(g / spacing)) > 0.01]
        assert not off_grid, (
            f"multiple={multiple!r}: {fill_a} bars start at {sorted(xs_a)} and {fill_b} "
            f"at {sorted(xs_b)}; the union's gaps {gaps} are not all multiples of "
            f"{spacing} -- the groups were binned on two different edge grids"
        )

        # 2. One width, across both groups -- otherwise the bars are not
        #    commensurable even where they do line up.
        assert widths_a == widths_b and len(widths_a) == 1, (
            f"multiple={multiple!r}: bar widths {sorted(widths_a)} vs {sorted(widths_b)} "
            "-- the groups' bars are not even commensurable"
        )

        # 3. Something for apply_stack to pair: at least one bin where both
        #    groups have rows, landing on the same x key.
        assert xs_a & xs_b, (
            f"multiple={multiple!r}: {fill_a} and {fill_b} share no x key "
            f"({sorted(xs_a)} vs {sorted(xs_b)}) -- nothing stacks onto anything"
        )

    @pytest.mark.parametrize("multiple", SHARED_EDGE_MODES)
    def test_wire_states_the_shared_extent(self, two_group_df, multiple):
        """The rendered geometry above comes from `Bin(shared_extent=True)`,
        not from some coincidence of this fixture's numbers."""
        chart = fm.displot(two_group_df, x="x", hue="g", kind="hist", multiple=multiple)
        assert _bin_transform(chart)["shared_extent"] is True

    @pytest.mark.parametrize("multiple", PER_GROUP_EDGE_MODES)
    def test_layer_and_dodge_keep_per_group_extents(self, two_group_df, multiple):
        """The non-stacking modes are unchanged: each group bins over its own
        range, because nothing needs to pair up across groups."""
        chart = fm.displot(two_group_df, x="x", hue="g", kind="hist", multiple=multiple)
        assert _bin_transform(chart)["shared_extent"] is False

        geometry = _bar_geometry_by_fill(chart.to_svg())
        (_fill_a, (xs_a, _wa)), (_fill_b, (xs_b, _wb)) = sorted(geometry.items())
        assert xs_a != xs_b, (
            f"multiple={multiple!r}: the two groups' bars landed on identical x "
            "positions -- this fixture's groups are well separated, so that means "
            "the shared extent leaked into a mode that must not have one"
        )


class TestOverlayFollowsTheBars:
    """The overlay's extent mode is the bars' own, for every mode."""

    @pytest.mark.parametrize("multiple", SHARED_EDGE_MODES + PER_GROUP_EDGE_MODES)
    def test_overlay_extent_mode_equals_the_bars(self, two_group_df, multiple):
        chart = fm.displot(
            two_group_df, x="x", hue="g", kind="hist", kde=True, stat="count", multiple=multiple
        )
        expected = multiple in SHARED_EDGE_MODES
        assert _bin_transform(chart)["shared_extent"] is expected
        assert _kde_overlay_transform(chart)["shared_extent"] is expected, (
            f"multiple={multiple!r}: the kde overlay's extent mode diverged from the "
            "bars' -- its count-scaled bin width no longer matches theirs"
        )


class TestPositionGroupingSurvives:
    """Forwarding the mode must not cost displot its `by=hue` grouping.

    The desugar's own position adjustment overwrites the one `mark_histogram`
    was called with (`_desugar._resolve_pending_impl`), so a naive forward
    would silently replace `Stack(by="g")` with a `by`-less `Stack`. displot
    passes `position_by=`, so the adjustment the desugar builds names the
    same column.
    """

    @pytest.mark.parametrize(
        "multiple,expected",
        [
            ("stack", {"type": "stack", "by": "g", "offset": "zero", "anchor": "top"}),
            ("fill", {"type": "stack", "by": "g", "offset": "normalize", "anchor": "top"}),
            ("dodge", {"type": "dodge", "by": "g", "padding": 0.05}),
            ("layer", {"type": "identity"}),
        ],
    )
    def test_rendered_position_names_the_hue_column(self, two_group_df, multiple, expected):
        chart = fm.displot(two_group_df, x="x", hue="g", kind="hist", multiple=multiple)
        spec = json.loads(chart.to_spec().to_json())
        assert spec["position"] == expected

    def test_channel_object_hue_gives_the_same_grouping(self, two_group_df):
        """`hue=fm.Color("g")` is a documented spelling; the position must
        name the column, not carry the channel object."""
        chart = fm.displot(two_group_df, x="x", hue=fm.Color("g"), kind="hist", multiple="stack")
        spec = json.loads(chart.to_spec().to_json())
        assert spec["position"]["by"] == "g"


class TestPositionByRefusesNonColumnNames:
    """The door this fix opened refuses what it cannot carry.

    `position_by` becomes `Stack(by=)`/`Dodge(by=)`, whose contract is
    `Optional[str]`. Anything else used to reach serialization untyped: a
    channel object produced `TypeError: Object of type Color is not JSON
    serializable` -- the literal failure the catplot dodge rider in this same
    change exists to eliminate, re-opened at a brand-new public parameter --
    and an int or list produced a raw serde message naming an internal wire
    key. It REFUSES rather than unwrapping: this parameter is documented as a
    column name, not as the figure layer's "str or encoding" spelling.
    """

    @pytest.fixture
    def hist_chart(self, two_group_df):
        def build(**kwargs):
            return (
                fm.Chart(two_group_df)
                .mark_histogram(multiple="stack", **kwargs)
                .encode(x="x", color="g")
            )

        return build

    def test_channel_object_refuses_naming_the_entry_and_the_fix(self, hist_chart):
        with pytest.raises(ValueError) as exc:
            hist_chart(position_by=fm.Color("g")).to_svg()
        message = str(exc.value)
        assert message.startswith("mark_histogram: position_by must be a column name")
        assert "Color" in message
        assert "position_by='g'" in message, "the refusal must say what to pass instead"

    @pytest.mark.parametrize("bad", [123, ["g"], 1.5])
    def test_other_non_string_values_refuse_the_same_way(self, hist_chart, bad):
        with pytest.raises(ValueError, match=r"^mark_histogram: position_by must be a column name"):
            hist_chart(position_by=bad).to_svg()

    def test_a_column_name_is_accepted_and_reaches_the_adjustment(self, hist_chart):
        """The accepted shape still works, and still names the column."""
        chart = hist_chart(position_by="g")
        assert chart.to_dict()["position"] == {
            "type": "stack",
            "by": "g",
            "offset": "zero",
            "anchor": "top",
        }

    def test_omitting_it_leaves_the_adjustment_by_less(self, hist_chart):
        """Backward compatibility: a caller who never heard of `position_by`
        gets exactly the `by`-less adjustment they got before it existed."""
        assert hist_chart().to_dict()["position"].get("by") is None


# ---------------------------------------------------------------------------
# Clean inputs stay byte-identical.
# ---------------------------------------------------------------------------


#: SHA-256 of the rendered SVG for each path the fix must NOT move, MEASURED
#: ON THE PRE-FIX BUILD (`src/ferrum/plots/distribution.py`,
#: `src/ferrum/marks/statistical.py` and
#: `src/ferrum/marks/_chart_methods_statistical.py` restored by file copy,
#: `src/ferrum/marks/_multiple.py` removed) -- a record of the old behavior,
#: not a re-baseline of the new one.
_PRE_FIX_SVG_DIGESTS = {
    "dodge": "36598053f906346983ad31d12e02e0e67c9b701e7a2a83cccc398e3785820707",
    "kde_kind_stack": "c5572e1467e762e9009b92f838db248ee80241534022a6a1fc4cbb841605d9ea",
    "layer": "529199fcca368cb3740c509a0aaac7a164a4ce7a49208f4bcd83f5e3c71ba68e",
    "no_hue": "634e2b6966ca6998ce1008efff01279df5f070ab1742ee2cde825d1016302754",
}


@pytest.mark.parametrize("name", sorted(_PRE_FIX_SVG_DIGESTS))
def test_unaffected_paths_render_byte_identically(two_group_df, name):
    """`multiple="layer"`/`"dodge"`, the no-hue path, and the `kind="kde"`
    branch (which does not forward the mode) are unchanged by this fix."""
    calls = {
        "layer": dict(x="x", hue="g", kind="hist", multiple="layer"),
        "dodge": dict(x="x", hue="g", kind="hist", multiple="dodge"),
        "no_hue": dict(x="x", kind="hist", multiple="stack"),
        "kde_kind_stack": dict(x="x", hue="g", kind="kde", multiple="stack"),
    }
    svg = fm.displot(two_group_df, **calls[name]).to_svg()
    digest = hashlib.sha256(svg.encode()).hexdigest()
    assert digest == _PRE_FIX_SVG_DIGESTS[name], f"{name} render moved: {digest}"
