"""Net-new coverage for F-L09-03's ``native_scale`` half: catplot's
category channel is now explicitly typed for every one of the 8 kinds
instead of ``native_scale=True`` unconditionally raising with a factually
wrong message.

``native_scale=False`` (the default) types the category channel
``"nominal"`` -- even a numeric column renders as equal-spaced categorical
bands. This is a **disclosed behavior change**: before this fix, the
category channel was never explicitly typed and a numeric column silently
inferred ``"quantitative"``, contradicting the documented default (seaborn
parity is the corrected behavior). ``native_scale=True`` types the channel
``"quantitative"`` and requires the column to actually be numeric -- it
raises, naming the column and its dtype, on a non-numeric one.

This module is scoped to the ``native_scale`` regression per this repo's
Python test-file convention (see ``test_boxen_palette.py``'s module
docstring): a findings-ID-named module stays scoped to pinning the specific
regression it describes.
"""

from __future__ import annotations

import hashlib
import re

import polars as pl
import pytest

import ferrum as fm
from _svg_rects import svg_elements

_ALL_CATPLOT_KINDS = ("strip", "swarm", "box", "violin", "boxen", "point", "bar", "count")


def _circle_cx(svg: str) -> list[float]:
    """Every rendered point mark's x position, in document order.

    Includes legend swatch circles (fixed ``r="4"``) -- fine for the charts
    below with no ``hue=``, which render no legend at all. The one test that
    DOES set ``hue=`` (the dodge test) uses ``_data_circle_cx`` instead.
    """
    return [float(v) for v in re.findall(r'<circle cx="([-\d.]+)"', svg)]


def _circle_cy(svg: str) -> list[float]:
    return [float(v) for v in re.findall(r'<circle cx="[-\d.]+" cy="([-\d.]+)"', svg)]


def _data_circle_cx(svg: str) -> list[float]:
    """Data-point circle x positions, excluding legend swatches.

    Legend swatches render as plain ``<circle>`` elements too, at a fixed
    ``r="4"`` regardless of theme (`tests/_svg_rects.py`'s precedent, mirrored
    from `test_composite_render_repeat_layer.py`), distinct from a data
    point's mark radius.
    """
    circles = svg_elements(svg, "circle", lambda a: a.get("r") != "4")
    return [float(c["cx"]) for c in circles]


def _uneven_numeric_cat_df() -> pl.DataFrame:
    # [1, 2, 10]: a nominal/band axis spaces these three categories EVENLY; a
    # quantitative axis places them at their real numeric positions, so the
    # 2->10 gap must dwarf the 1->2 gap. This is the exact shape the task
    # brief names ("unevenly spaced categories like [1, 2, 10] prove native
    # placement").
    return pl.DataFrame(
        {
            "cat": [1, 1, 2, 2, 10, 10],
            "val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )


def _string_cat_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "cat": ["a", "a", "b", "b"],
            "val": [1.0, 2.0, 3.0, 4.0],
        }
    )


# ---------------------------------------------------------------------------
# native_scale=True places numeric categories at their real numeric positions
# ---------------------------------------------------------------------------


def test_native_scale_true_places_uneven_categories_proportionally():
    df = _uneven_numeric_cat_df()
    svg = fm.catplot(df, x="cat", y="val", kind="strip", jitter=False, native_scale=True).to_svg()
    xs = sorted(set(_circle_cx(svg)))
    assert len(xs) == 3
    gap_1_2 = xs[1] - xs[0]
    gap_2_10 = xs[2] - xs[1]
    # (10-2)=8 is 8x (2-1)=1 in data units; on a linear axis the pixel gaps
    # must reflect that -- a band axis would make them identical instead.
    assert gap_2_10 > gap_1_2 * 3


def test_native_scale_true_orient_h_types_the_y_channel():
    """After the swap for ``orient="h"``, cat is always on x pre-CoordFlip
    (F-L09-03's cross-task note) -- verify the SAME uneven-spacing proof
    holds on the rendered y axis once CoordFlip puts it back there."""
    df = _uneven_numeric_cat_df()
    svg = fm.catplot(
        df, x="val", y="cat", kind="strip", jitter=False, native_scale=True, orient="h"
    ).to_svg()
    ys = sorted(set(_circle_cy(svg)))
    assert len(ys) == 3
    gap_a = abs(ys[1] - ys[0])
    gap_b = abs(ys[2] - ys[1])
    assert max(gap_a, gap_b) > min(gap_a, gap_b) * 3


# ---------------------------------------------------------------------------
# native_scale=False (default) forces nominal bands even for numeric cats --
# the disclosed behavior change
# ---------------------------------------------------------------------------


def test_native_scale_false_forces_equal_spaced_bands_for_numeric_cats():
    df = _uneven_numeric_cat_df()
    svg = fm.catplot(df, x="cat", y="val", kind="strip", jitter=False, native_scale=False).to_svg()
    xs = sorted(set(_circle_cx(svg)))
    assert len(xs) == 3
    gap_1_2 = xs[1] - xs[0]
    gap_2_10 = xs[2] - xs[1]
    assert gap_1_2 == pytest.approx(gap_2_10, abs=0.01)


def test_native_scale_default_matches_explicit_false():
    df = _uneven_numeric_cat_df()
    svg_default = fm.catplot(df, x="cat", y="val", kind="strip", jitter=False).to_svg()
    svg_explicit = fm.catplot(
        df, x="cat", y="val", kind="strip", jitter=False, native_scale=False
    ).to_svg()
    assert svg_default == svg_explicit


def test_native_scale_false_string_cats_byte_identical_to_pre_fix():
    """Byte-identity pin (RED-proven via ``git stash`` against pre-fix
    ``distribution.py``): a string category column was already typed
    Nominal by Rust's own inference, so explicitly typing it Nominal here
    must not move a single byte of rendered SVG. Hashes captured from the
    unpatched module for exactly these three calls; equal pre-fix and
    post-fix (verified directly, not just asserted here).
    """
    df = _string_cat_df()
    pins = {
        "strip": "569a88668f1b70283ea1a7534abe04c4e5e89279c4be5cb88dc9c2247d70525c",
        "box": "8c0e5a97877ca04dcc82b60bb3ad89d320c310fd8c2f6c32e749a1463e977212",
    }
    for kind, expected in pins.items():
        svg = fm.catplot(df, x="cat", y="val", kind=kind).to_svg()
        assert hashlib.sha256(svg.encode()).hexdigest() == expected, kind

    # kind="point" draws a seeded bootstrap; the seed must be pinned too for
    # the hash to be meaningful.
    svg_point = fm.catplot(df, x="cat", y="val", kind="point", ci=95, n_boot=200, seed=0).to_svg()
    assert (
        hashlib.sha256(svg_point.encode()).hexdigest()
        == "a3261abdf0a1db2fdb5454187e7d1636e068df83949356b79a3ee7d21ad39389"
    )


# ---------------------------------------------------------------------------
# native_scale=True on a non-numeric column refuses, naming column + dtype
# ---------------------------------------------------------------------------


def test_native_scale_true_on_string_column_raises_naming_column_and_dtype():
    df = _string_cat_df()
    with pytest.raises(ValueError, match=r"'cat'.*not numeric.*String"):
        fm.catplot(df, x="cat", y="val", kind="strip", native_scale=True)


def test_native_scale_true_on_boolean_column_raises():
    df = pl.DataFrame({"cat": [True, True, False, False], "val": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(ValueError, match=r"'cat'.*not numeric"):
        fm.catplot(df, x="cat", y="val", kind="strip", native_scale=True)


# ---------------------------------------------------------------------------
# Non-finite category values (cycle-2 fix, quality finding 2): NaN/inf must
# not poison the jitter-width computation into a NaN/inf render failure.
# ---------------------------------------------------------------------------


def test_native_scale_jitter_nan_category_values_do_not_crash():
    """Pre-fix (cycle 1): ``drop_nulls()`` alone does not drop NaN, and
    ``sorted()`` over a diff sequence containing NaN silently produced a
    NaN jitter width, which then failed at render time with an opaque
    ``ValueError: position: expected value at line 1 column 42`` naming
    nothing (RED-proven against cycle-1 code). The finite category values
    still render; only the NaN rows themselves drop, same as they would
    with jitter off.
    """
    df = pl.DataFrame(
        {
            "cat": [1.0, float("nan"), float("nan"), 2.0] * 3,
            "val": [float(i) for i in range(12)],
        }
    )
    svg = fm.catplot(
        df, x="cat", y="val", kind="strip", jitter=True, native_scale=True, seed=0
    ).to_svg()
    assert "<svg" in svg


def test_native_scale_jitter_infinite_category_values_do_not_crash():
    df = pl.DataFrame(
        {
            "cat": [1.0, float("inf"), 2.0, float("-inf")] * 3,
            "val": [float(i) for i in range(12)],
        }
    )
    svg = fm.catplot(
        df, x="cat", y="val", kind="strip", jitter=True, native_scale=True, seed=0
    ).to_svg()
    assert "<svg" in svg


# ---------------------------------------------------------------------------
# x=fm.X(...) (the documented `x : str or encoding` shape): native_scale
# must not be inert, order= must not silently drop, and both kind families
# must agree (cycle-2 fix, quality finding 3 / spec-review issues 1-2)
# ---------------------------------------------------------------------------


def test_native_scale_channel_object_form_matches_bare_string_both_values():
    """``x=fm.X("cat")`` (no ``type_=`` of its own) must resolve to the
    SAME positions as the bare-string ``x="cat"`` form, under BOTH
    ``native_scale`` values -- pre-fix (cycle 1) the channel-object form was
    completely inert to ``native_scale`` (always rendered at native
    positions, ignoring ``native_scale=False``'s force-nominal default;
    RED-proven against cycle-1 code).
    """
    df = _uneven_numeric_cat_df()
    for native_scale in (True, False):
        svg_bare = fm.catplot(
            df, x="cat", y="val", kind="strip", jitter=False, native_scale=native_scale
        ).to_svg()
        svg_channel = fm.catplot(
            df, x=fm.X("cat"), y="val", kind="strip", jitter=False, native_scale=native_scale
        ).to_svg()
        assert svg_bare == svg_channel, native_scale


def test_native_scale_channel_object_form_matches_aggregating_kind():
    """The two kind families must agree on the SAME channel-object input
    (spec-review issue 1): pre-fix, ``kind="point"`` banded (typed via the
    bare ``cat_name`` regardless of the caller's channel-object shape)
    while the six non-aggregating kinds stayed inert at native positions --
    the two families disagreed on the identical call.
    """
    df = _uneven_numeric_cat_df()
    for native_scale in (True, False):
        svg_strip = fm.catplot(
            df, x=fm.X("cat"), y="val", kind="strip", jitter=False, native_scale=native_scale
        ).to_svg()
        svg_point = fm.catplot(
            df,
            x=fm.X("cat"),
            y="val",
            kind="point",
            native_scale=native_scale,
            ci=95,
            n_boot=200,
            seed=0,
        ).to_svg()
        xs_strip = sorted(set(_circle_cx(svg_strip)))
        xs_point = sorted(set(_circle_cx(svg_point)))
        assert xs_strip == xs_point, native_scale


def test_native_scale_channel_object_order_is_honored_not_dropped():
    """Pre-fix (cycle 1): ``order=`` on a channel-object ``cat_field``
    silently dropped (rendered in data order, ignoring ``order=``) where
    the ORIGINAL pre-task-6 tree raised a loud ``TypeError`` for the same
    combination -- a documented parameter regressed from a loud failure to
    a silent no-op (RED-proven against cycle-1 code: same render both
    orders). ``order=`` must actually reorder the axis, matching the
    bare-string form's own behavior (`test_catplot_hue_order.py`'s pattern).
    Nominal/ordinal domain values serialize as strings, so ``order=`` needs
    the string spelling of a numeric category (``"1"`` not ``1``) --
    exactly as it already does for a bare-string ``cat_field``.
    """
    df = _uneven_numeric_cat_df()
    svg_a = fm.catplot(
        df,
        x=fm.X("cat"),
        y="val",
        kind="strip",
        jitter=False,
        native_scale=False,
        order=["1", "2", "10"],
    ).to_svg()
    svg_b = fm.catplot(
        df,
        x=fm.X("cat"),
        y="val",
        kind="strip",
        jitter=False,
        native_scale=False,
        order=["10", "2", "1"],
    ).to_svg()
    assert svg_a != svg_b


def test_native_scale_channel_object_explicit_type_honored_verbatim():
    """A channel object that DOES state its own ``type_=`` is honored as-is,
    bypassing ``native_scale`` entirely -- including its numeric-dtype
    check, mirroring ``nominal_color_channel``'s pass-through rule for an
    explicit ``hue=``. A string column explicitly typed ``"nominal"`` must
    not raise under ``native_scale=True`` (it would if the explicit type_
    were ignored and native_scale's own numeric check ran instead).
    """
    df = _string_cat_df()
    svg = fm.catplot(
        df, x=fm.X("cat", type_="nominal"), y="val", kind="strip", native_scale=True
    ).to_svg()
    assert "<svg" in svg


def test_native_scale_caller_explicit_domain_is_not_overridden():
    """Quality finding 1's required guardrail: an explicit caller domain
    must still govern -- the padding fix must never silently widen a
    domain the caller deliberately set. A caller domain narrower than the
    padded one would naturally clip some Jitter-adjusted marks; that
    clipping is the caller's OWN choice, not a bug, and proves the padding
    logic did not override it (a wider auto/padded domain would have kept
    every mark instead).
    """
    df = _three_group_df(1.0)
    svg = fm.catplot(
        df,
        x=fm.X("cat", scale=fm.LinearScale(domain=[0, 2])),
        y="val",
        kind="strip",
        jitter=True,
        native_scale=True,
        seed=0,
    ).to_svg()
    assert len(_circle_cx(svg)) < 30


def test_native_scale_encode_x_door_bypasses_padding_and_typing():
    """Documents (does not endorse or fix) a pre-existing, module-wide
    ``encode=`` override semantic, not a new ``native_scale`` gap (quality
    review cycle 2's one non-blocking follow-up): ``encode={"x": ...}`` is a
    raw ``Chart.encode(**encode)`` override applied AFTER this builder's own
    normalization, so it replaces the padded/typed channel wholesale --
    even with no ``scale=`` of its own, unlike the ``x=fm.X(..., scale=...)``
    door (`test_native_scale_caller_explicit_domain_is_not_overridden`
    above), which carries an explicit domain the padding logic correctly
    detects and defers to. The identical override already erases this
    builder's OTHER normalizations the same way (e.g. `hue=`'s nominal-color
    typing, shipped earlier this batch) -- this is `encode=`'s documented
    contract, not something `native_scale` introduces. Not fixed here: the
    stronger route (routing `encode.get("x")` through
    `_catplot_cat_field_parts`) has an unresolved `orient="h"` key-space
    question (logged as a batch follow-up), so this test pins the CURRENT
    behavior as a known boundary instead.
    """
    df = _three_group_df(1.0)
    # Padding lost: the same 30-row jitter fixture that renders 30/30 through
    # x= drops back to 17/30 through encode=, with no scale=/domain involved
    # at all.
    svg_jitter = fm.catplot(
        df,
        x="cat",
        y="val",
        kind="strip",
        jitter=True,
        native_scale=True,
        seed=0,
        encode={"x": fm.X("cat")},
    ).to_svg()
    assert len(_circle_cx(svg_jitter)) == 17

    # Force-nominal typing lost under native_scale=False: the encode= door
    # still renders the numeric column at native (quantitative) positions,
    # not equal-spaced bands.
    numeric_df = _uneven_numeric_cat_df()
    svg_false = fm.catplot(
        numeric_df,
        x="cat",
        y="val",
        kind="strip",
        jitter=False,
        native_scale=False,
        encode={"x": fm.X("cat")},
    ).to_svg()
    xs = sorted(set(_circle_cx(svg_false)))
    assert len(xs) == 3
    gap_1_2 = xs[1] - xs[0]
    gap_2_10 = xs[2] - xs[1]
    assert gap_2_10 > gap_1_2 * 3  # native spacing, not equal-spaced bands

    # The two kind families disagree on this same door: point/bar rebuild
    # their own x_channel from the bare field name (bypassing encode=
    # entirely for the CATEGORY channel, per `_catplot_summary_layers`'s
    # existing "layered chart resolves from its own layer encodings" design),
    # so they keep the force-nominal bands the chart-level channel lost.
    svg_point = fm.catplot(
        numeric_df,
        x="cat",
        y="val",
        kind="point",
        native_scale=False,
        ci=95,
        n_boot=200,
        seed=0,
        encode={"x": fm.X("cat")},
    ).to_svg()
    xs_point = sorted(set(_circle_cx(svg_point)))
    assert xs_point != xs  # point still bands; strip (via encode=) does not


# ---------------------------------------------------------------------------
# Jitter on a native scale: width rescales with the category column's own
# spacing (data units), not the ordinal 0.4 band-fraction
# ---------------------------------------------------------------------------


def _three_group_df(spacing: float, *, n_per_group: int = 10) -> pl.DataFrame:
    cats = [0.0, spacing, 2 * spacing]
    cat_col: list[float] = []
    val_col: list[float] = []
    for c in cats:
        for j in range(n_per_group):
            cat_col.append(c)
            val_col.append(float(j))
    return pl.DataFrame({"cat": cat_col, "val": val_col})


def test_native_scale_jitter_width_scales_with_category_spacing():
    """The jitter width is computed as 0.4 x the category column's own
    median unique-value spacing, so scaling every category value (and
    therefore the whole axis domain) by a constant factor must scale the
    jitter noise by the exact same factor -- the RENDERED pixel positions
    are then identical regardless of the absolute spacing, because a linear
    scale maps a proportionally-rescaled domain onto the same fixed panel
    width. A FIXED (not spacing-scaled) width would instead collapse to a
    near-invisible sliver at a spacing of 1000 while spreading widely at a
    spacing of 1 -- this discriminates against exactly that regression
    (verified directly against a literal-width Jitter: positions diverge).
    """
    svg_a = fm.catplot(
        _three_group_df(1.0), x="cat", y="val", kind="strip", jitter=True, native_scale=True, seed=0
    ).to_svg()
    svg_b = fm.catplot(
        _three_group_df(1000.0),
        x="cat",
        y="val",
        kind="strip",
        jitter=True,
        native_scale=True,
        seed=0,
    ).to_svg()
    xs_a = sorted(_circle_cx(svg_a))
    xs_b = sorted(_circle_cx(svg_b))
    assert xs_a == xs_b
    # Cycle-2 strengthening (quality finding 5): `len(...) > 0` let a partial
    # mark loss (domain clipping, finding 1) pass silently as long as SOME
    # circles rendered identically at both spacings. All 30 rows (3 groups x
    # 10) must render at both spacings -- the domain-padding fix in
    # `_native_scale_padded_domain` is what makes this hold; pre-fix this
    # was 17 of 30 at both spacings (RED-proven against cycle-1 code).
    assert len(xs_a) == 30


def test_native_scale_true_jitter_default_example_preserves_all_marks():
    """The docstring's own ``native_scale=True`` example (``kind="strip"``,
    default ``jitter=True``) must not lose marks -- this is literally the
    first thing a user of this feature will run (quality review cycle 1
    flagged this exact shape as the lossy one: 17 of 30 circles for a
    3-category, 10-per-group frame at ordinary, non-adversarial spacing).
    """
    df = pl.DataFrame(
        {
            "cat": [2020] * 8 + [2021] * 8 + [2022] * 8,
            "val": [float(i) for i in range(24)],
        }
    )
    svg = fm.catplot(df, x="cat", y="val", kind="strip", native_scale=True, seed=0).to_svg()
    assert len(_circle_cx(svg)) == 24


def test_native_scale_jitter_single_category_falls_back_sanely():
    """Degenerate case: one unique category value has no spacing to derive
    a width from. Must not crash, and must still spread the points a
    little (the documented base_width=0.4 fallback), not collapse them all
    onto one pixel."""
    df = pl.DataFrame({"cat": [5] * 10, "val": [float(i) for i in range(10)]})
    svg = fm.catplot(
        df, x="cat", y="val", kind="strip", jitter=True, native_scale=True, seed=0
    ).to_svg()
    xs = _circle_cx(svg)
    assert len(xs) == 10
    assert max(xs) - min(xs) > 0.0


# ---------------------------------------------------------------------------
# Dodge on a native scale rides the renderer's existing continuous-band path
# ---------------------------------------------------------------------------


def test_native_scale_dodge_renders_distinct_group_positions():
    """Ordinary fixture (cycle-2 fix, quality finding 5): the pre-fix
    version of this test anchored the fixture with extra ``-100``/``110``
    categories specifically chosen to keep the two REAL dodge-tested
    categories away from the renderer's own axis-padding clip -- engineering
    the known mark-loss defect (quality finding 1) out of the pin instead of
    catching it. Now that the category channel's domain is padded to
    absorb the dodge offset (`_native_scale_padded_domain`), the ordinary,
    unpadded ``[1, 2, 10]`` fixture -- no anchors, no adversarial choices --
    renders every one of its 6 marks, at 6 distinct x-positions (dodge
    actually split each category's pair apart; a no-op dodge would collapse
    each pair onto 3 shared positions instead). Pre-fix (cycle 1) this
    fixture rendered only 3 of 6 (RED-proven against cycle-1 code).
    """
    df = pl.DataFrame(
        {
            "cat": [1, 1, 2, 2, 10, 10],
            "val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "grp": ["a", "b", "a", "b", "a", "b"],
        }
    )
    svg = fm.catplot(
        df, x="cat", y="val", kind="strip", hue="grp", dodge=True, native_scale=True
    ).to_svg()
    xs = _data_circle_cx(svg)
    assert len(xs) == 6
    assert len(set(xs)) == 6


# ---------------------------------------------------------------------------
# Aggregating kinds (point/bar): Summary's groupby is dtype-preserving, so
# both native_scale values must work the same way non-aggregating kinds do
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["point", "bar"])
@pytest.mark.parametrize("native_scale", [True, False])
def test_native_scale_aggregating_kinds_render_under_both_values(kind, native_scale):
    df = pl.DataFrame(
        {
            "cat": [1] * 5 + [2] * 5 + [10] * 5,
            "val": [float(i) for i in range(15)],
        }
    )
    svg = fm.catplot(
        df, x="cat", y="val", kind=kind, native_scale=native_scale, ci=95, n_boot=200, seed=0
    ).to_svg()
    assert "<svg" in svg


def test_native_scale_true_aggregating_kind_places_estimates_proportionally():
    df = pl.DataFrame(
        {
            "cat": [1] * 5 + [2] * 5 + [10] * 5,
            "val": [float(i) for i in range(15)],
        }
    )
    svg = fm.catplot(
        df, x="cat", y="val", kind="point", native_scale=True, ci=95, n_boot=200, seed=0
    ).to_svg()
    xs = sorted(set(_circle_cx(svg)))
    assert len(xs) == 3
    gap_1_2 = xs[1] - xs[0]
    gap_2_10 = xs[2] - xs[1]
    assert gap_2_10 > gap_1_2 * 3


# ---------------------------------------------------------------------------
# All 8 kinds accept the flag without error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", _ALL_CATPLOT_KINDS)
def test_native_scale_false_all_kinds_accept_the_flag(kind):
    df = _string_cat_df()
    kwargs = {"ci": 95, "n_boot": 200, "seed": 0} if kind in ("point", "bar") else {}
    svg = fm.catplot(df, x="cat", y="val", kind=kind, native_scale=False, **kwargs).to_svg()
    assert "<svg" in svg


@pytest.mark.parametrize("kind", [k for k in _ALL_CATPLOT_KINDS if k != "boxen"])
def test_native_scale_true_numeric_kinds_accept_the_flag(kind):
    """Every kind except ``boxen`` accepts ``native_scale=True`` on a
    numeric column. ``boxen`` is excluded deliberately, not silently: its
    letter-value transform (``crates/ferrum-core/src/transform/letter_value.rs``)
    hard-requires a Utf8 groupby column TODAY, independent of
    ``native_scale`` entirely -- see
    ``test_native_scale_true_boxen_hits_the_pre_existing_utf8_groupby_limit``
    below, which pins that this is a pre-existing, orthogonal Rust-transform
    gap (bare ``mark_boxen()`` on a numeric ``x=`` already fails on `main`
    before this fix), not something this task introduces or is scoped to
    close (crates/** is out of scope here).
    """
    df = pl.DataFrame(
        {
            "cat": [1] * 5 + [2] * 5 + [10] * 5,
            "val": [float(i) for i in range(15)],
        }
    )
    kwargs = {"ci": 95, "n_boot": 200, "seed": 0} if kind in ("point", "bar") else {}
    svg = fm.catplot(df, x="cat", y="val", kind=kind, native_scale=True, **kwargs).to_svg()
    assert "<svg" in svg


def test_native_scale_true_boxen_hits_the_pre_existing_utf8_groupby_limit():
    """Documents (does not fix) a pre-existing, orthogonal Rust gap:
    ``stat_letter_value`` requires its groupby column to be Utf8 regardless
    of ``native_scale`` -- bare ``mark_boxen()`` on a numeric ``x=`` already
    raises the identical message on unpatched `main`. Not part of
    F-L09-03's scope (crates/** is out of bounds for this task); tracked
    here so the boundary is tested, not silently swallowed.
    """
    df = pl.DataFrame(
        {
            "cat": [1] * 3 + [2] * 3 + [10] * 3,
            "val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        }
    )
    with pytest.raises(ValueError, match=r"must be Utf8"):
        fm.catplot(df, x="cat", y="val", kind="boxen", native_scale=True).to_svg()
    # native_scale=False hits the identical pre-existing limit: force-nominal
    # typing changes the AXIS type, not the raw data fed to the letter-value
    # transform's groupby, so this is untouched by this task's fix either way.
    with pytest.raises(ValueError, match=r"must be Utf8"):
        fm.catplot(df, x="cat", y="val", kind="boxen", native_scale=False).to_svg()
