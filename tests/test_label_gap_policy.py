"""Tick-label gap magnitude and cull permission are independent (`#146`).

Two behaviors, pinned at the Python surface:

1. **The defect.** ``Theme(cull_threshold=N)`` was entirely INERT on a facet
   whose x scale resolves independently. The per-panel x axis was rebuilt with
   the compile-time default gap instead of the theme's, so a 60-category axis
   rendered the *same* label count at ``cull_threshold`` 0, 8 and 40, while the
   un-faceted and shared-x paths tracked the setting. Pre-fix readout, counting
   rendered category labels on a 1200x500 chart::

       cull_threshold | no facet | shared x | independent x
       default (8)    |       30 |       40 |            40
       40             |       20 |       20 |            40   <- flat
       0              |       60 |      120 |            40   <- flat

2. **The split.** ``cull_threshold`` used to mean both "this much gap" and
   "culling enabled", with ``0`` meaning both "no gap requirement" and "never
   drop a label". The new ``cull_labels`` key carries the permission alone, so
   "require a gap of N px AND never drop a label" becomes expressible. No
   pre-existing spelling changes meaning: that is pinned by a hash corpus in
   the task's report, and here by the legacy round-trip assertions below.

This module is net-new feature coverage for the ``cull_labels`` Theme key, so
it lives in its own file rather than in a findings-ID-named module (see
``tests/marks/test_boxen_palette.py``'s docstring for the convention).
"""

import re

import polars as pl
import pytest

import ferrum as fm

N_CATS = 60
CATS = [f"Category {i:02d}" for i in range(N_CATS)]


def _single_panel() -> pl.DataFrame:
    return pl.DataFrame({"cat": CATS, "val": list(range(N_CATS))})


def _two_panel() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "cat": CATS * 2,
            "val": list(range(N_CATS)) * 2,
            "grp": ["A"] * N_CATS + ["B"] * N_CATS,
        }
    )


def _count_labels(svg: str) -> int:
    """Rendered x tick labels. Every category label is unique, so this is a
    visible-label count, not a text-node count."""
    return len(re.findall(r">Category \d\d<", svg))


def _render(mode: str, **theme_kwargs) -> str:
    df = _single_panel() if mode == "none" else _two_panel()
    chart = fm.Chart(df, width=1200, height=500).mark_bar().encode(x="cat", y="val")
    if mode == "shared":
        chart = chart.facet(col="grp")
    elif mode == "independent":
        chart = chart.facet(col="grp").share_scale(x="independent")
    if theme_kwargs:
        chart = chart.theme(fm.Theme(**theme_kwargs))
    return chart.to_svg()


@pytest.mark.parametrize("mode", ["none", "shared", "independent"])
def test_cull_threshold_moves_the_label_count_on_every_facet_resolution(mode):
    """A wider gap drops labels and ``0`` keeps them all -- on all three paths.

    Pre-fix the ``independent`` parametrization failed both assertions: its
    count was pinned at the compile-time default whatever the theme said.
    """
    wide = _count_labels(_render(mode, cull_threshold=40))
    default = _count_labels(_render(mode))
    none = _count_labels(_render(mode, cull_threshold=0))

    assert wide < default, f"{mode}: cull_threshold=40 must drop more than the default"
    assert default < none, f"{mode}: the default must drop labels cull_threshold=0 keeps"


def test_independent_and_shared_facet_x_cull_alike():
    """The independent-x axis reaches the shared-x axis's conclusion.

    Same data, same theme, same visible-label count. Pre-fix this held only at
    the default value (8), which is exactly the signature of a hardcoded
    default.
    """
    for cull_threshold in (0, 8, 40):
        shared = _count_labels(_render("shared", cull_threshold=cull_threshold))
        independent = _count_labels(_render("independent", cull_threshold=cull_threshold))
        assert shared == independent, (
            f"independent-x and shared-x must agree at cull_threshold={cull_threshold}; "
            f"got shared={shared} independent={independent}"
        )


def test_cull_labels_false_keeps_every_label_at_a_positive_gap():
    """ "Gap of N AND never drop a label" -- the state the single integer could
    not spell -- is observably distinct from both legacy spellings."""
    gap_cull = _render("none", cull_threshold=40)
    gap_no_cull = _render("none", cull_threshold=40, cull_labels=False)
    no_gap = _render("none", cull_threshold=0)

    assert _count_labels(gap_cull) < N_CATS, "the legacy gap-40 spelling must still cull"
    assert _count_labels(gap_no_cull) == N_CATS, (
        "cull_labels=False must keep every label at a positive gap"
    )
    assert gap_no_cull != gap_cull, "gap-40-never-cull must differ from gap-40-may-cull"
    # At this density both no-cull spellings land on the same last-resort
    # presentation, so `no_gap` is only a control here: it must keep every
    # label too. The gap's effect on the EARLIER cascade stages, which is what
    # separates the two, is pinned by
    # `test_gap_drives_the_cascade_while_culling_is_forbidden` below.
    assert _count_labels(no_gap) == N_CATS


def test_gap_drives_the_cascade_while_culling_is_forbidden():
    """At a density where the gap decides the PRESENTATION, the three states
    are distinguishable by rendered label count and rotation, not just bytes:
    30 short labels in a wide panel resolve flat with no gap requirement, cull
    to half with a 30px gap, and keep every label rotated when that same gap is
    paired with ``cull_labels=False``."""
    df = pl.DataFrame({"cat": [f"C{i:02d}" for i in range(30)], "val": list(range(30))})

    def render(**theme_kwargs):
        chart = fm.Chart(df, width=1200, height=400).mark_bar().encode(x="cat", y="val")
        svg = chart.theme(fm.Theme(**theme_kwargs)).to_svg()
        elements = re.findall(r"<text[^>]*>C\d\d</text>", svg)
        rotations = {
            re.search(r"rotate\(([-\d.]+)", e).group(1) if "rotate(" in e else "0" for e in elements
        }
        return len(elements), sorted(rotations)

    assert render(cull_threshold=0) == (30, ["0"])
    assert render(cull_threshold=30) == (15, ["0"])
    assert render(cull_threshold=30, cull_labels=False) == (30, ["-90"])


@pytest.mark.parametrize(
    "n_labels,expected",
    [
        pytest.param(4, (4, ["0"]), id="sparse-satisfies-the-gap-flat"),
        pytest.param(30, (30, ["-90"]), id="dense-forces-the-fallthrough"),
    ],
)
def test_cull_labels_false_stage_depends_on_density(n_labels, expected):
    """`{cull_threshold: 30, cull_labels: False}` never drops a label, but
    WHICH stage the survivors land in depends on density, not just on the
    gap being set -- a claim past cycles asserted unconditionally (always
    "-90 degrees") and a probe falsified in 4 of 10 sampled configurations.
    Sparse enough and the 30px gap is satisfiable at S0 (flat, here); dense
    enough and the cascade falls through every earlier stage to S5's
    rotate-intact fallback (``-90``), because `cull_labels=False` forbids
    S4's cull. Both keep every label -- that part IS unconditional, and is
    what the corrected docs/spec/doc-comment prose now actually claims."""
    df = pl.DataFrame({"cat": [f"C{i:02d}" for i in range(n_labels)], "val": list(range(n_labels))})
    chart = fm.Chart(df, width=1200, height=400).mark_bar().encode(x="cat", y="val")
    svg = chart.theme(fm.Theme(cull_threshold=30, cull_labels=False)).to_svg()
    elements = re.findall(r"<text[^>]*>C\d\d</text>", svg)
    rotations = {
        re.search(r"rotate\(([-\d.]+)", e).group(1) if "rotate(" in e else "0" for e in elements
    }
    assert (len(elements), sorted(rotations)) == expected


def test_cull_labels_true_culls_at_a_zero_gap():
    """The mirror corner: no gap requirement, but dropping permitted.

    Also unreachable from the old integer, where ``0`` forbade culling too.
    The panel is narrow enough that even a fully rotated label does not fit
    its slot, so dropping is the only remedy left and the two spellings of
    "gap 0" separate.
    """

    def render(**theme_kwargs) -> int:
        chart = (
            fm.Chart(_single_panel(), width=600, height=400)
            .mark_bar()
            .encode(x="cat", y="val")
            .theme(fm.Theme(**theme_kwargs))
        )
        return _count_labels(chart.to_svg())

    forbidden = render(cull_threshold=0)
    permitted = render(cull_threshold=0, cull_labels=True)
    assert forbidden == N_CATS, "gap 0 alone must still never drop a label"
    assert permitted < N_CATS, (
        f"cull_labels=True at gap 0 must drop genuinely overlapping labels; "
        f"got {permitted} of {N_CATS}"
    )


def test_cull_labels_false_alone_keeps_every_label_at_the_default_gap():
    """The headline user spelling: ``Theme(cull_labels=False)`` with NO
    explicit ``cull_threshold`` at all. It must forbid dropping at the
    *default* gap (8px), not just when paired with an explicit magnitude --
    every other corner of the square is pinned above, but this one, the
    most natural way a user reaches for the key, was not."""
    default = _render("none")
    no_cull = _render("none", cull_labels=False)

    assert _count_labels(default) < N_CATS, "the default gap must still cull at this density"
    assert _count_labels(no_cull) == N_CATS, "cull_labels=False alone must keep every label"
    assert no_cull != default


def test_cull_labels_is_a_known_theme_key():
    """The key rides the Rust manifest, so Python accepts it without a
    hand-maintained parallel list."""
    assert "cull_labels" in fm.Theme._KNOWN_KEYS
    assert fm.Theme(cull_labels=False)._props == {"cull_labels": False}
