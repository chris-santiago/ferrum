"""``displot`` answers "this needs ``x=``" once, in its own voice.

Three defects with one shape, all closed here (wave-4 close remediation):

1. ``displot(y="v", kind="hist")`` raised ``mark_histogram() requires
   .encode(x=...) to specify the histogram field`` and ``kind="kde"`` the
   ``mark_density()`` twin -- the MARK's voice, naming a method the caller
   never called, from a figure function that validates ``kind=``/
   ``multiple=``/``stat=``/``bins=`` as ``displot(...)``.  ``kind="ecdf"``
   already answered correctly; all three now answer from one site.
2. ``displot(y="v", kind="rug", kde=True)`` said nothing at all.  The overlay
   block promoted the rug into layered mode and then appended no overlay
   (its guard was ``x_field is not None``), rendering a plausible chart with
   the requested curve silently absent -- measured 0 ``<polyline>`` against
   1 for the identical call with ``x=``.  That is the REFUSE-MALFORMED /
   WARN-DEGRADED policy's central case; it refuses.
3. ``displot(bins=...)`` was validated for every kind and consumed by none
   under ``kind="kde"``/``"rug"`` -- ``bins=3`` and ``bins=97`` rendered
   byte-identically.  An accepted-and-inert parameter is the WARN-DEGRADED
   half, so it warns.

The matrix test at the bottom is the mechanism-proof form of 1 and 2: every
``(kind, axis, kde, rug)`` cell must either refuse in displot's voice or
render every mark it was asked for.  Neither defect could survive it.
"""

from __future__ import annotations

import hashlib
import warnings

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from ferrum._warn import reset_warnings
from tests._svg_rects import data_bar_rects, polylines, rug_tick_x1_by_stroke

#: Kinds that draw the variable ``x=`` names and compute the other axis.
KINDS_REQUIRING_X = ("hist", "kde", "ecdf")

#: The mark-voiced refusal texts these calls used to raise.  Pinned as
#: NEGATIVE assertions: the fix is not "some ValueError happens" (one always
#: did) but "the caller is told about the API they called".
MARK_VOICED_FRAGMENTS = ("mark_histogram(", "mark_density(", ".encode(x=")


@pytest.fixture
def two_group_df():
    rng = np.random.default_rng(0)
    n = 200
    return pl.DataFrame(
        {
            "v": np.concatenate([rng.normal(0, 1, n), rng.normal(2, 1, n)]),
            "g": ["a"] * n + ["b"] * n,
        }
    )


@pytest.fixture(autouse=True)
def _fresh_warn_registry():
    """`warn_once` dedups per context; each test starts from an empty slate."""
    reset_warnings()
    yield
    reset_warnings()


class TestRefusalsNameDisplot:
    """The three x-requiring kinds refuse a ``y=``-only call by name."""

    @pytest.mark.parametrize("kind", KINDS_REQUIRING_X)
    def test_refusal_names_the_entry_the_caller_invoked(self, two_group_df, kind):
        with pytest.raises(ValueError) as exc:
            fm.displot(two_group_df, y="v", kind=kind)
        message = str(exc.value)
        assert message.startswith(f"displot(kind={kind!r}, y='v')"), message
        for fragment in MARK_VOICED_FRAGMENTS:
            assert fragment not in message, (
                f"the refusal still speaks in the mark's voice ({fragment!r}): {message}"
            )

    @pytest.mark.parametrize("kind", KINDS_REQUIRING_X)
    def test_refusal_states_the_corrective_action_in_the_callers_column(self, two_group_df, kind):
        """Naming the entry is half of it -- the caller also has to be told
        what to pass, spelled with their own column name."""
        with pytest.raises(ValueError) as exc:
            fm.displot(two_group_df, y="v", kind=kind)
        assert "Pass x='v'" in str(exc.value)

    @pytest.mark.parametrize("kind", KINDS_REQUIRING_X)
    def test_a_channel_object_y_reports_its_column_not_its_repr(self, two_group_df, kind):
        """`y=fm.Y("v")` is a documented spelling; the refusal must still name
        the column, so its `Pass x=` is something the caller can paste."""
        with pytest.raises(ValueError) as exc:
            fm.displot(two_group_df, y=fm.Y("v"), kind=kind)
        assert "Pass x='v'" in str(exc.value)

    @pytest.mark.parametrize("kind", KINDS_REQUIRING_X)
    def test_no_variable_at_all_refuses_the_same_way(self, two_group_df, kind):
        with pytest.raises(ValueError, match=rf"^displot\(kind={kind!r}, x=None, y=None\)"):
            fm.displot(two_group_df, kind=kind)

    def test_rug_still_draws_against_y(self, two_group_df):
        """`kind="rug"` is the one kind a `y=` binding is honored on, so it
        must NOT be swept into the refusal."""
        svg = fm.displot(two_group_df, y="v", kind="rug").to_svg()
        assert rug_tick_x1_by_stroke(svg), "the y-bound rug drew no ticks"


class TestOverlayRefusesRatherThanVanishing:
    """`kde=`/`rug=` on an x-less call refuse; they used to render nothing."""

    def test_kde_overlay_on_a_y_bound_rug_refuses_naming_the_overlay(self, two_group_df):
        with pytest.raises(ValueError) as exc:
            fm.displot(two_group_df, y="v", kind="rug", kde=True)
        message = str(exc.value)
        assert message.startswith("displot(kind='rug', y='v', kde=True)"), message
        assert "Pass x='v'" in message
        assert "drop kde=" in message

    def test_the_same_call_with_x_draws_the_curve(self, two_group_df):
        """The positive control: the refusal is about the missing variable,
        not about the combination. With `x=` bound, the overlay is there --
        one curve, stroked, with real vertices."""
        svg = fm.displot(two_group_df, x="v", kind="rug", kde=True).to_svg()
        curves = [pts for _stroke, pts in polylines(svg) if len(pts) > 2]
        assert len(curves) == 1, f"expected exactly one kde overlay curve, got {len(curves)}"
        ys = [y for _x, y in curves[0]]
        assert max(ys) - min(ys) > 10, "the overlay curve is flat"

    def test_rug_overlay_on_an_x_less_call_refuses(self, two_group_df):
        """The `rug=` door, guarded identically, gets the identical answer.

        `kind="hist"` is refused one clause earlier (it needs `x=` for itself),
        so the reachable shape is a kind that does NOT need `x=` -- which,
        with `rug=` excluded by `kind != "rug"`, leaves the no-variable call.
        """
        with pytest.raises(ValueError) as exc:
            fm.displot(two_group_df, kind="rug", kde=True)
        assert str(exc.value).startswith("displot(kind='rug', x=None, y=None, kde=True)")


class TestInertBinsWarns:
    """A binning rule named on a kind that does not bin is reported."""

    @pytest.mark.parametrize("kind", ["kde", "rug"])
    def test_a_stated_rule_the_kind_cannot_act_on_warns(self, two_group_df, kind):
        with pytest.warns(UserWarning) as record:
            fm.displot(two_group_df, x="v", kind=kind, bins=7)
        messages = [str(w.message) for w in record]
        assert any(
            m.startswith(f"displot(kind={kind!r}, bins=7)") and "inert" in m for m in messages
        ), messages

    @pytest.mark.parametrize("kind", ["kde", "rug"])
    def test_the_warning_is_true_the_render_really_is_unmoved(self, two_group_df, kind):
        """The warning's claim -- "the render is identical for every value" --
        is measured here rather than asserted in prose."""
        digests = []
        for bins in (3, 97):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                svg = fm.displot(two_group_df, x="v", kind=kind, bins=bins).to_svg()
            digests.append(hashlib.sha256(svg.encode()).hexdigest())
        assert digests[0] == digests[1], (
            f"kind={kind!r} renders differently for bins=3 vs bins=97 -- bins= is not "
            "inert after all, and the warning is now a lie"
        )

    @pytest.mark.parametrize("bins", [None, "sturges"])
    def test_the_default_rule_is_not_reported(self, two_group_df, bins):
        """Only a rule the caller STATED can be lost. `None` and `"sturges"`
        both mean the default, which is what a non-binning kind would have
        used anyway, so there is nothing to report."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            fm.displot(two_group_df, x="v", kind="kde", bins=bins)

    @pytest.mark.parametrize("kind", ["hist", "ecdf"])
    def test_the_binning_kinds_are_silent_and_actually_bin(self, two_group_df, kind):
        """No warning where the rule acts -- and it does act."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            few = fm.displot(two_group_df, x="v", kind=kind, bins=3).to_svg()
            many = fm.displot(two_group_df, x="v", kind=kind, bins=23).to_svg()
        assert few != many, f"kind={kind!r} ignored bins= -- the silence is unearned"


# ---------------------------------------------------------------------------
# The matrix: every cell refuses in displot's voice, or draws what it was asked
# for.  Neither of the two defects above could survive this test.
# ---------------------------------------------------------------------------

_MATRIX = [
    (kind, axis, kde, rug)
    for kind in ("hist", "kde", "ecdf", "rug")
    for axis in ("x", "y")
    for kde in (False, True)
    for rug in (False, True)
]


def _primary_mark_drawn(svg: str, kind: str) -> bool:
    """Did the primary `kind=` put its own mark in the SVG?"""
    if kind == "hist":
        return bool(data_bar_rects(svg))
    if kind == "rug":
        return bool(rug_tick_x1_by_stroke(svg))
    if kind == "kde":
        # `fill=True` by default, so the density is a filled `<path>`.
        return "<path" in svg
    # ecdf: a step `<polyline>` with real vertices.
    return any(len(pts) > 2 for _stroke, pts in polylines(svg))


@pytest.mark.parametrize("kind,axis,kde,rug", _MATRIX, ids=lambda v: str(v))
def test_every_cell_refuses_or_draws_every_requested_mark(two_group_df, kind, axis, kde, rug):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chart = fm.displot(two_group_df, **{axis: "v"}, kind=kind, kde=kde, rug=rug)
            svg = chart.to_svg()
    except ValueError as exc:
        assert str(exc).startswith("displot("), (
            f"({kind!r}, {axis}=, kde={kde}, rug={rug}) refused, but not in displot's voice: {exc}"
        )
        return

    assert _primary_mark_drawn(svg, kind), (
        f"({kind!r}, {axis}=, kde={kde}, rug={rug}) rendered without its primary mark"
    )
    layer_names = {layer.name for layer in (chart._layers or [])}
    if kde and kind != "kde":
        assert "kde" in layer_names, (
            f"({kind!r}, {axis}=, kde=True, rug={rug}) rendered with no kde layer -- "
            "the overlay was requested and silently dropped"
        )
    if rug and kind != "rug":
        assert "rug" in layer_names, (
            f"({kind!r}, {axis}=, kde={kde}, rug=True) rendered with no rug layer -- "
            "the overlay was requested and silently dropped"
        )
