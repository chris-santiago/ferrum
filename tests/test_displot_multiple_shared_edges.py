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

EXTENDED (wave-4 close remediation): ``"dodge"`` binned per-group too, and the
one-owner module ratified it with a rationale the render did not support --
"dodged bars occupy their own slot within a shared band" over an output with no
shared band at all (measured: 9 bars of width 35.37 px against 10 of width
21.05 px, sharing not one x position). The band a dodged bar takes its slot in
IS the bin, so ``"dodge"`` now bins on a shared extent like ``"stack"`` and
``"fill"``. This is a deliberate, user-approved rendering change for
``multiple="dodge"``; the ``"dodge"`` entry in ``_PRE_FIX_SVG_DIGESTS`` below
was retired rather than re-baselined for exactly that reason.

Shared edges were necessary and not sufficient. The Rust ``apply_dodge`` then
had to move BOTH ends of a span bar rather than only its start, before a slot
had a width worth asserting; that landed, so ``TestDodgeSharesTheBand`` pins
the slot geometry too -- equal widths, adjacent rather than overlapping,
tiling the padded band -- and the bar COUNT, because the first group's
negative offset used to push the leftmost bin start below the resolved
x-domain and lose that bar to a ``NaN`` with nothing counting bars.

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
from tests._svg_extents import assert_bars_start_on_the_zero_tick
from tests._svg_rects import data_bar_rects

#: Modes whose groups must land on one shared bin grid, and the one mode that
#: keeps per-group extents. A stack needs shared edges to have anything to
#: stack; a dodge needs them to have a band to take a slot in. Only "layer"
#: asks nothing of its neighbours, so only "layer" may bin per group.
SHARED_EDGE_MODES = ("stack", "fill", "dodge")
PER_GROUP_EDGE_MODES = ("layer",)

#: The subset of SHARED_EDGE_MODES that also draws every group's bars at one
#: width on one set of x keys. "dodge" shares the edges but then subdivides
#: each band into per-group slots, so its bars are the same width as each
#: other but NOT as the band, and they are not co-located -- a different
#: rendered shape from one contract.
STACKING_MODES = ("stack", "fill")

#: `Dodge`'s default padding, eaten at BOTH ends of the band before the slots
#: are cut (`pad_total = band * padding * 2`). Named once: the wire test below
#: pins that this is the value that reaches the spec, and the dodge geometry
#: test predicts the rendered slot width from it, so the two readings cannot
#: drift into two numbers.
DODGE_PADDING = 0.05


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


@pytest.fixture
def overlapping_two_group_df():
    """Two uniform samples over the SAME range, tagged by ``g``.

    For the bar-count property only. On the separated fixture above, a bin
    where a group has (almost) no rows still renders -- the bar carries a
    minimum height (measured 8 px) -- so a count there would be pinning the
    height clamp rather than the thing at issue. Here every group genuinely
    has rows in every bin (measured minimum bar height 212 px), so "10 bins
    requested, 10 bars drawn, for each group" is a statement about bars that
    each have data behind them.
    """
    rng = np.random.default_rng(7)
    n = 300
    return pl.DataFrame(
        {
            "x": np.concatenate([rng.uniform(0, 10, n), rng.uniform(0, 10, n)]),
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


def _one_grid_step(xs: set[float], *, label: str) -> float:
    """Return the single bin step *xs* lies on, asserting that it has one.

    A group whose bars all come off one bin grid has starts that differ by
    whole multiples of one spacing -- a bin nobody landed in leaves a
    double gap, not a shifted grid.
    """
    starts = sorted(xs)
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert gaps, f"{label}: fewer than two distinct bar starts ({starts})"
    step = min(gaps)
    off_grid = [g for g in gaps if abs(g / step - round(g / step)) > 0.01]
    assert not off_grid, (
        f"{label}: bar starts {starts} have gaps {gaps}, not all whole multiples "
        f"of {step} -- these bars did not come off one bin grid"
    )
    # Refine: `min(gaps)` carries the rounding of two 2-dp coordinates, which
    # compounds into a visible drift across ten bands. The span divided by the
    # whole number of steps it covers does not.
    return (starts[-1] - starts[0]) / sum(round(g / step) for g in gaps)


def _slot_offset(xs: set[float], *, step: float, origin: float, label: str) -> float:
    """Return the constant offset of *xs* within its band, asserting it is one.

    Each group's bars sit at the same position inside every band it appears
    in, so ``(x - origin) % step`` is one value across the whole group.  That
    value IS the group's dodge slot.
    """
    residues = [(x - origin) % step for x in sorted(xs)]
    # A residue a hair under `step` is the same slot as one a hair over zero.
    residues = [r - step if r > step - 0.05 else r for r in residues]
    spread = max(residues) - min(residues)
    assert spread < 0.05, (
        f"{label}: bars sit at {sorted(set(round(r, 2) for r in residues))} within "
        "their bands -- one group's bars must take ONE slot, not several"
    )
    return residues[0]


class TestSharedBinEdges:
    """Under stack and fill, every group's bars share bin edges and widths."""

    @pytest.mark.parametrize("multiple", STACKING_MODES)
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
    def test_layer_keeps_per_group_extents(self, two_group_df, multiple):
        """``"layer"`` is the one mode that still bins per group.

        Overlaid distributions ask nothing of each other -- no pairing, no
        slot -- so each group is free to bin over its own range, and does:
        the two groups' grids have DIFFERENT steps, which is the positive
        statement of "these were binned separately" (an `xs_a != xs_b`
        inequality would also pass on two groups that shared a grid and
        merely landed in different bins).
        """
        chart = fm.displot(two_group_df, x="x", hue="g", kind="hist", multiple=multiple)
        assert _bin_transform(chart)["shared_extent"] is False

        geometry = _bar_geometry_by_fill(chart.to_svg())
        (fill_a, (xs_a, _wa)), (fill_b, (xs_b, _wb)) = sorted(geometry.items())
        step_a = _one_grid_step(xs_a, label=f"multiple={multiple!r} {fill_a}")
        step_b = _one_grid_step(xs_b, label=f"multiple={multiple!r} {fill_b}")
        assert step_a != pytest.approx(step_b, rel=1e-3), (
            f"multiple={multiple!r}: both groups binned at step {step_a} -- this "
            "fixture's groups have very different ranges, so one shared step means "
            "a shared extent leaked into the one mode that must not have one"
        )


class TestDodgeSharesTheBand:
    """``multiple="dodge"`` bins every group against one grid, then slots them.

    The mode's own rationale is that each group takes its own slot within a
    shared band. That band is the bin, so the bins have to be the same bins
    first -- and they were not: ``_multiple.py`` mapped ``"dodge"`` to
    ``shared_extent=False`` and this fixture rendered 9 blue bars of width
    35.37 px against 10 red bars of width 21.05 px, sharing not one x
    position. The guarding test asserted only ``xs_a != xs_b``, which is a
    claim about DIFFERENCE and was true precisely because the bug was live.

    The slot geometry is asserted too, now that it means something. Sharing
    the edges was necessary but not sufficient: the continuous-axis
    `apply_dodge` used to add its per-group offset to the band column alone
    and leave the span's second coordinate (`bin_end`) where it was, so a
    span bar's two edges moved by different amounts -- measured 60.91 px and
    38.53 px against a 49.72 px band, overlapping, one of them overrunning
    the band entirely. Both coordinates move now, so a slot is a slot: same
    width for every group, adjacent rather than overlapping, and the whole
    band accounted for by `n_groups` slots plus the padding at each end.

    Scope: these are the figure-level properties a `displot` caller sees.
    The slot arithmetic itself is unit-tested Rust-side; this module does not
    restate it, it checks that what reached the SVG has the shape the
    arithmetic promises.
    """

    def test_wire_states_the_shared_extent(self, two_group_df):
        chart = fm.displot(two_group_df, x="x", hue="g", kind="hist", multiple="dodge")
        assert _bin_transform(chart)["shared_extent"] is True

    def test_both_groups_bin_on_one_grid_and_take_distinct_slots(self, two_group_df):
        """One step for both groups, and a constant sub-band offset between them."""
        svg = fm.displot(two_group_df, x="x", hue="g", kind="hist", multiple="dodge").to_svg()
        geometry = _bar_geometry_by_fill(svg)
        assert len(geometry) == 2, f"expected bars for 2 hue levels, got {len(geometry)}"
        (fill_a, (xs_a, _wa)), (fill_b, (xs_b, _wb)) = sorted(geometry.items())

        # 1. One bin grid, shared. Pre-fix these were 29.25 px and 27.16 px:
        #    two groups binning their own ranges into their own bin widths.
        step_a = _one_grid_step(xs_a, label=f"dodge {fill_a}")
        step_b = _one_grid_step(xs_b, label=f"dodge {fill_b}")
        assert step_a == pytest.approx(step_b, rel=1e-3), (
            f"dodge: {fill_a} bars step {step_a} px, {fill_b} bars step {step_b} px "
            "-- the groups were binned on two different grids, so neither group's "
            "bars share a band with the other's"
        )

        # 2. Distinct slots WITHIN that shared band: each group sits at one
        #    constant offset inside its band, and the two offsets differ by
        #    less than a whole band (same band, different slot) and by more
        #    than nothing (dodged, not superimposed).
        origin = min(xs_a | xs_b)
        slot_a = _slot_offset(xs_a, step=step_a, origin=origin, label=f"dodge {fill_a}")
        slot_b = _slot_offset(xs_b, step=step_a, origin=origin, label=f"dodge {fill_b}")
        separation = abs(slot_a - slot_b)
        assert 0.0 < separation < step_a, (
            f"dodge: {fill_a} sits at {slot_a:.2f} and {fill_b} at {slot_b:.2f} within "
            f"a {step_a:.2f} px band -- the two groups must take different slots of "
            "the SAME band, not the same slot and not a band apart"
        )

    def test_both_groups_slots_are_the_same_width(self, two_group_df):
        """Two groups splitting one band get one width between them.

        Different widths mean the groups are not commensurable even where
        they do line up -- the taller-looking group might simply be wider.
        The band is common (asserted above), so anything but one width is a
        subdivision that does not divide.
        """
        svg = fm.displot(two_group_df, x="x", hue="g", kind="hist", multiple="dodge").to_svg()
        (fill_a, (_xa, widths_a)), (fill_b, (_xb, widths_b)) = sorted(
            _bar_geometry_by_fill(svg).items()
        )
        assert len(widths_a) == 1 and len(widths_b) == 1, (
            f"dodge: {fill_a} drew widths {sorted(widths_a)} and {fill_b} "
            f"{sorted(widths_b)} -- one group's bars must all be one width"
        )
        assert widths_a == widths_b, (
            f"dodge: {fill_a} bars are {sorted(widths_a)[0]} px wide and {fill_b} bars "
            f"{sorted(widths_b)[0]} px -- two groups splitting one band get one width"
        )

    def test_the_slots_tile_the_padded_band_without_overlapping(self, two_group_df):
        """Adjacent, not overlapping, and between them the whole band.

        `Dodge` eats `DODGE_PADDING` of the band at EACH end and cuts what is
        left into one slot per group, so a slot is ``band * (1 - 2*padding) /
        n_groups`` and the two slots sit edge to edge. Both halves matter and
        fail differently: overlapping slots hide one group behind the other,
        and slots that do not span the padded band leave the bars floating in
        a gap that belongs to no one.
        """
        svg = fm.displot(two_group_df, x="x", hue="g", kind="hist", multiple="dodge").to_svg()
        (fill_a, (xs_a, widths_a)), (fill_b, (xs_b, widths_b)) = sorted(
            _bar_geometry_by_fill(svg).items()
        )
        step = _one_grid_step(xs_a, label=f"dodge {fill_a}")
        width = sorted(widths_a)[0]

        # 1. The slot is what is left of the band after padding, split two ways.
        expected = step * (1.0 - 2.0 * DODGE_PADDING) / 2.0
        assert width == pytest.approx(expected, abs=0.05), (
            f"dodge: slots are {width:.2f} px of a {step:.2f} px band; two slots plus "
            f"{DODGE_PADDING:.0%} padding at each end wants {expected:.2f} px. The bars "
            "do not account for the band they were cut from"
        )

        # 2. Edge to edge, taken in the band. The leftmost pair is one band's
        #    two slots (asserted one slot apart above), so the left group's
        #    right edge is where the right group's left edge starts.
        left, right = sorted((min(xs_a), min(xs_b)))
        assert left + width <= right + 0.05, (
            f"dodge: a slot starting at {left:.2f} px runs {width:.2f} px wide, past "
            f"the next slot's start at {right:.2f} px -- the two groups overlap"
        )
        assert right - (left + width) < 0.05 + step * DODGE_PADDING, (
            f"dodge: {right - (left + width):.2f} px of unclaimed band sits between the "
            "two slots -- the slots must tile the padded band, not float inside it"
        )
        assert sorted(widths_b)[0] == pytest.approx(width, abs=0.05)

    @pytest.mark.parametrize("bins", [6, 10, 17])
    def test_no_group_loses_a_bar_to_an_out_of_domain_slot(self, overlapping_two_group_df, bins):
        """The bins asked for are the bars drawn -- for BOTH groups.

        The leftmost band is where a dodge can silently cost a group a bar:
        the first group's slot offset is negative, so before the Rust fix it
        pushed that bin's start below the resolved x-domain, where the
        unclamped linear scale returns NaN and the row is skipped. The bar
        was ABSENT, not mis-sized, and nothing counted bars -- the render
        simply had 9 where it should have had 10, which is invisible unless
        something asserts the count.

        Parametrized over three bin counts so the assertion tracks the rule
        the caller stated rather than one constant that happens to match; a
        render that always drew ten bars would fail two of the three.

        This fixture's groups both span the full range, so every bin has rows
        from both and every bar has data behind it (see the fixture).
        """
        svg = fm.displot(
            overlapping_two_group_df, x="x", hue="g", kind="hist", multiple="dodge", bins=bins
        ).to_svg()
        (fill_a, (xs_a, _wa)), (fill_b, (xs_b, _wb)) = sorted(_bar_geometry_by_fill(svg).items())
        assert len(xs_a) == bins and len(xs_b) == bins, (
            f"dodge with bins={bins}: {fill_a} drew {len(xs_a)} bars and {fill_b} "
            f"{len(xs_b)} -- a group short a bar lost it, it did not merge it"
        )

        # ...and specifically the leftmost band carries both, which is the
        # band the negative offset used to push out of the domain.
        step = _one_grid_step(xs_a, label=f"dodge {fill_a}")
        origin = min(xs_a | xs_b)
        assert min(xs_a) - origin < step and min(xs_b) - origin < step, (
            f"dodge: {fill_a} starts at {min(xs_a):.2f} px and {fill_b} at "
            f"{min(xs_b):.2f} px, more than one {step:.2f} px band apart -- one group "
            "has no bar in the leftmost band"
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
            ("dodge", {"type": "dodge", "by": "g", "padding": DODGE_PADDING}),
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
#:
#: `"dodge"` was in this table and was REMOVED, not re-measured: the close
#: remediation moved that render on purpose (see `TestDodgeSharesTheBand`),
#: so its digest is no longer a statement about anything. Re-baselining it to
#: the new bytes would turn a record of "this path is untouched" into a
#: record of "whatever it does today", which is what the entry exists to
#: prevent. `kde_kind_stack` still holds the original pre-fix bytes; the two
#: histogram rows are re-measured with the bars grown from the 0 tick, which
#: moves only their `<rect>` elements. Each of those two digests encodes two
#: changes: the shared-edges fix leaving the path untouched, and the bars grown
#: from the 0 tick, which `test_unaffected_histogram_paths_start_on_the_zero_tick`
#: asserts directly.
_PRE_FIX_SVG_DIGESTS = {
    "kde_kind_stack": "c5572e1467e762e9009b92f838db248ee80241534022a6a1fc4cbb841605d9ea",
    "layer": "fe9b741ca7d1d33f80ad65505d0a3f6a0975405631596b4e089a4a24ee941247",
    "no_hue": "8d2607348247a2e2a13d27cd88c83fdf3e5fcafeaa8e4b649c993e2821fdc19d",
}


@pytest.mark.parametrize(
    "call",
    [
        dict(x="x", hue="g", kind="hist", multiple="layer"),
        dict(x="x", kind="hist", multiple="stack"),
    ],
    ids=["layer", "no_hue"],
)
def test_unaffected_histogram_paths_start_on_the_zero_tick(two_group_df, call):
    """The geometry the re-measured digests pin: every bar grows from 0."""
    assert_bars_start_on_the_zero_tick(fm.displot(two_group_df, **call).to_svg())


@pytest.mark.parametrize("name", sorted(_PRE_FIX_SVG_DIGESTS))
def test_unaffected_paths_render_byte_identically(two_group_df, name):
    """`multiple="layer"`, the no-hue path, and the `kind="kde"` branch
    (which does not forward the mode) are unchanged by this fix."""
    calls = {
        "layer": dict(x="x", hue="g", kind="hist", multiple="layer"),
        "no_hue": dict(x="x", kind="hist", multiple="stack"),
        "kde_kind_stack": dict(x="x", hue="g", kind="kde", multiple="stack"),
    }
    svg = fm.displot(two_group_df, **calls[name]).to_svg()
    digest = hashlib.sha256(svg.encode()).hexdigest()
    assert digest == _PRE_FIX_SVG_DIGESTS[name], f"{name} render moved: {digest}"
