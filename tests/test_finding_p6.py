"""Regression tests — Finding P6 (2026-08-27 design-review remediation batch).

P6 covers two defects, same root cause (a documented closed vocabulary with
no runtime enforcement):

1. Every ``Literal``-annotated frozen-dataclass field in the census
   (``AUCLabel``/``APLabel``/``BrierLabel.position``, ``CoordPolar.theta``/
   ``direction``, ``CoordGeo.projection``, ``AnnotationText.anchor``/
   ``baseline``/``z``, ``AnnotationSpan.axis``/``label_position``,
   ``AnnotationBracket.direction``, ``AnnotationCallout.arrow``) now rejects
   out-of-vocabulary values at construction via
   ``ferrum._validate.validate_choice``. (``AnnotationImage.anchor`` is
   deliberately excluded — see the module docstring note below and
   ``tests/test_annotation_layer.py::TestAnnotationImage::test_to_dict_all_fields``,
   which pins ``anchor="top-left"`` as a legitimate, already-accepted value
   outside the four functionally-distinct anchors Rust honors.)
2. ``AUCLabel``/``APLabel``/``BrierLabel.position`` was a fully dead public
   field — ``_apply_metric_label`` never read it, so ``"end"`` and
   ``"corner"`` rendered identically. ``position="end"`` (the value that
   already matched every pre-fix render, confirmed by inspection and by the
   byte-identity pin below) stays byte-identical; ``"corner"`` becomes a
   real, distinct placement.

Claim-check correction (post-review): the first validation pass validated
``AnnotationText.anchor``/``baseline`` and ``AnnotationBracket.direction``
against their *docstring* vocabulary only, which is narrower than what
``render/annotation.rs`` (``parse_anchor``, ``emit_bracket``) and
``render/draw.rs`` (``parse_text_baseline``) actually accept and render
distinctly — e.g. ``anchor="left"``/``"right"`` (reachable through
``annotate_text``'s ``_ANCHOR_TO_ALIGN`` alias pass-through),
``baseline="hanging"``/``"central"``/``"text-before-edge"``/
``"text-after-edge"``/``"ideographic"``/``"alphabetic"``, and bracket
``direction="up"``/``"down"``/``"left"``/``"right"``. Previously-working
input would have raised ``ValueError`` post-fix. The validated sets were
widened to the union of every Rust-recognized token (see
``ferrum.annotation.primitives._VALID_TEXT_ANCHORS`` /
``_VALID_TEXT_BASELINES`` / ``_VALID_BRACKET_DIRECTIONS``); the tests below
pin that every alias both constructs and renders identically to its
canonical value. ``AnnotationSpan.axis``/``label_position`` and
``AnnotationCallout.arrow`` were audited too but need no widening: Rust
treats anything outside their documented tokens as a silent binary
fallback (no *named* alias exists to add), so the original docstring
vocabulary is already exactly what Rust recognizes by name.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from ferrum import APLabel, AUCLabel, BrierLabel, Chart
from ferrum.annotation.primitives import (
    AnnotationBracket,
    AnnotationCallout,
    AnnotationSpan,
    AnnotationText,
)
from ferrum.coord import CoordGeo, CoordPolar


def _roc_data(label: str = "c0", scale: float = 1.0, n: int = 50) -> pl.DataFrame:
    """Synthetic ROC curve; ``scale`` controls how far along x the curve reaches,
    so two concatenated series can have different endpoint x values."""
    fpr = np.linspace(0, scale, n)
    tpr = np.sqrt(fpr / scale) if scale else np.zeros(n)
    return pl.DataFrame({"fpr": fpr, "tpr": tpr, "class": [label] * n})


def _text_x_positions(svg: str, prefix: str) -> list[str]:
    return re.findall(rf'<text x="([0-9.]+)"[^>]*>{re.escape(prefix)}', svg)


# ---------------------------------------------------------------------------
# 1. Out-of-vocabulary construction raises — the whole P6 census
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ctor", "match"),
    [
        pytest.param(
            lambda: AUCLabel(position="middle"),
            r"AUCLabel\.position: position must be one of",
            id="AUCLabel.position",
        ),
        pytest.param(
            lambda: APLabel(position="middle"),
            r"APLabel\.position: position must be one of",
            id="APLabel.position",
        ),
        pytest.param(
            lambda: BrierLabel(position="middle"),
            r"BrierLabel\.position: position must be one of",
            id="BrierLabel.position",
        ),
        pytest.param(
            lambda: CoordPolar(theta="z"),
            r"CoordPolar\.theta: theta must be one of",
            id="CoordPolar.theta",
        ),
        pytest.param(
            lambda: CoordPolar(direction=0),
            r"CoordPolar\.direction: direction must be one of",
            id="CoordPolar.direction",
        ),
        pytest.param(
            lambda: CoordGeo(projection="bogus"),
            r"CoordGeo\.projection: projection must be one of",
            id="CoordGeo.projection",
        ),
        pytest.param(
            lambda: AnnotationText(
                x=0,
                y=0,
                text="t",
                font_size=12,
                color="#000",
                anchor="bogus",
                baseline="middle",
                angle=0,
                dx=0,
                dy=0,
                z="above_marks",
            ),
            r"AnnotationText\.anchor: anchor must be one of",
            id="AnnotationText.anchor",
        ),
        pytest.param(
            lambda: AnnotationText(
                x=0,
                y=0,
                text="t",
                font_size=12,
                color="#000",
                anchor="start",
                baseline="bogus",
                angle=0,
                dx=0,
                dy=0,
                z="above_marks",
            ),
            r"AnnotationText\.baseline: baseline must be one of",
            id="AnnotationText.baseline",
        ),
        pytest.param(
            lambda: AnnotationText(
                x=0,
                y=0,
                text="t",
                font_size=12,
                color="#000",
                anchor="start",
                baseline="middle",
                angle=0,
                dx=0,
                dy=0,
                z="bogus",
            ),
            r"AnnotationText\.z: z must be one of",
            id="AnnotationText.z",
        ),
        pytest.param(
            lambda: AnnotationSpan(
                axis="z",
                start=0,
                end=1,
                fill="#eee",
                opacity=0.2,
                label=None,
                label_position="top",
            ),
            r"AnnotationSpan\.axis: axis must be one of",
            id="AnnotationSpan.axis",
        ),
        pytest.param(
            lambda: AnnotationSpan(
                axis="x",
                start=0,
                end=1,
                fill="#eee",
                opacity=0.2,
                label=None,
                label_position="bogus",
            ),
            r"AnnotationSpan\.label_position: label_position must be one of",
            id="AnnotationSpan.label_position",
        ),
        pytest.param(
            lambda: AnnotationBracket(
                x1=0,
                y1=0,
                x2=1,
                y2=0,
                label="A",
                direction="bogus",
                stroke="#333",
                tip_length=6,
            ),
            r"AnnotationBracket\.direction: direction must be one of",
            id="AnnotationBracket.direction",
        ),
        pytest.param(
            lambda: AnnotationCallout(
                x=0,
                y=0,
                text="t",
                text_x=None,
                text_y=None,
                arrow="bogus",
                padding=4,
                background="#fff",
                border_color="#ccc",
                border_radius=3,
            ),
            r"AnnotationCallout\.arrow: arrow must be one of",
            id="AnnotationCallout.arrow",
        ),
    ],
)
def test_out_of_vocab_construction_raises(ctor, match):
    with pytest.raises(ValueError, match=match):
        ctor()


@pytest.mark.parametrize(
    ("cls_name", "ctor"),
    [
        ("AUCLabel", lambda fmt: AUCLabel(format=fmt)),
        ("APLabel", lambda fmt: APLabel(format=fmt)),
        ("BrierLabel", lambda fmt: BrierLabel(format=fmt)),
    ],
)
def test_bad_format_spec_fails_at_construction_with_class_context(cls_name, ctor):
    """A bad ``format`` spec previously only exploded deep inside an f-string
    at render time with no indication of which label constructed it. It now
    fails at construction, naming the constructing class."""
    with pytest.raises(ValueError, match=rf"{cls_name}\.format: 'zzz' is not a valid format spec"):
        ctor("zzz")


def test_valid_constructions_across_census_still_work():
    """Every valid value in each census field's documented vocabulary still
    constructs without error (no over-tightening)."""
    AUCLabel(position="end")
    AUCLabel(position="corner")
    CoordPolar(theta="x", direction=1)
    CoordPolar(theta="y", direction=-1)
    CoordGeo(projection="albers_usa")
    AnnotationText(
        x=0,
        y=0,
        text="t",
        font_size=12,
        color="#000",
        anchor="end",
        baseline="bottom",
        angle=0,
        dx=0,
        dy=0,
        z="below_marks",
    )
    AnnotationSpan(
        axis="y", start=0, end=1, fill="#eee", opacity=0.2, label=None, label_position="bottom"
    )
    AnnotationBracket(
        x1=0, y1=0, x2=1, y2=0, label="A", direction="below", stroke="#333", tip_length=6
    )
    AnnotationCallout(
        x=0,
        y=0,
        text="t",
        text_x=None,
        text_y=None,
        arrow="none",
        padding=4,
        background="#fff",
        border_color="#ccc",
        border_radius=3,
    )


# ---------------------------------------------------------------------------
# 2. position="end" vs "corner" — distinct, documented placements
# ---------------------------------------------------------------------------


def test_end_and_corner_placements_are_visually_distinct():
    """Two series with different curve extents (c0 reaches fpr=1.0, c1 only
    reaches fpr=0.6) discriminate the two placements: "end" anchors each
    series' label at its own curve endpoint (two distinct x positions);
    "corner" anchors every series' label at the same x near the top-right
    of the plot area (one shared x position)."""
    df = pl.concat([_roc_data("c0", scale=1.0), _roc_data("c1", scale=0.6)])
    base = Chart(df).encode(x="fpr", y="tpr", color="class").mark_line()
    svg_end = (base + AUCLabel(position="end")).to_svg()
    svg_corner = (base + AUCLabel(position="corner")).to_svg()

    assert svg_end != svg_corner

    end_xs = _text_x_positions(svg_end, "AUC = ")
    corner_xs = _text_x_positions(svg_corner, "AUC = ")
    assert len(end_xs) == 2
    assert len(corner_xs) == 2
    assert len(set(end_xs)) == 2, "end must place each series at its own endpoint x"
    assert len(set(corner_xs)) == 1, "corner must stack every series at one shared x"


# ---------------------------------------------------------------------------
# 3. position="end" stays byte-identical to today's (pre-fix) dead-field output
# ---------------------------------------------------------------------------
#
# Baselines captured via the git-stash regression protocol: with
# src/ferrum/_metric_labels.py's position-routing changes stashed out
# (`position` completely unread, always end-style placement), these two
# scenarios' rendered SVG hashed to the values below. `position="end"` must
# reproduce them exactly post-fix.


def test_position_end_byte_identical_to_pre_fix_single_series():
    df = _roc_data("c0", scale=1.0)
    base = Chart(df).encode(x="fpr", y="tpr", color="class").mark_line()
    svg = (base + AUCLabel(position="end")).to_svg()
    assert (
        hashlib.sha256(svg.encode()).hexdigest()
        == "36d7d7cd6d53337c91e34c610a0c65480bda8d17f1536e827703f6cf603fb126"
    )


def test_position_end_byte_identical_to_pre_fix_multi_series():
    df = pl.concat([_roc_data("c0", scale=1.0), _roc_data("c1", scale=0.6)])
    base = Chart(df).encode(x="fpr", y="tpr", color="class").mark_line()
    svg = (base + AUCLabel(position="end")).to_svg()
    assert (
        hashlib.sha256(svg.encode()).hexdigest()
        == "5871b7c75e019d8881f25649797fb4b9c8b756c3718138317796441fedb7a017"
    )


# ---------------------------------------------------------------------------
# 4. Rust-recognized aliases construct AND render identically to the
#    canonical value (claim-check fix: the validated set must not be
#    narrower than what Rust's own parser distinguishes)
# ---------------------------------------------------------------------------


def _scatter_base():
    df = pl.DataFrame({"x": [1, 2, 3], "y": [1, 4, 9]})
    return Chart(df).mark_point().encode(x="x", y="y")


def _render_text(**overrides) -> str:
    kwargs = dict(
        x=1.5,
        y=5,
        text="hi",
        font_size=12,
        color="#000",
        anchor="start",
        baseline="middle",
        angle=0,
        dx=0,
        dy=0,
        z="above_marks",
    )
    kwargs.update(overrides)
    return (_scatter_base() + AnnotationText(**kwargs)).to_svg()


def _render_bracket(direction: str) -> str:
    prim = AnnotationBracket(
        x1=0, y1=0, x2=1, y2=0, label="A", direction=direction, stroke="#333", tip_length=6
    )
    return (_scatter_base() + prim).to_svg()


def _render_callout(arrow: str) -> str:
    prim = AnnotationCallout(
        x=1.5,
        y=5,
        text="hi",
        text_x=None,
        text_y=None,
        arrow=arrow,
        padding=4,
        background="#fff",
        border_color="#ccc",
        border_radius=3,
    )
    return (_scatter_base() + prim).to_svg()


def test_annotation_callout_curved_renders_identically_to_straight():
    """Known, out-of-scope Rust limitation (render/annotation.rs::emit_callout
    draws a straight leader line for any ``arrow`` value != "none" — no curve
    is implemented): "curved" and "straight" currently render identically.
    Pinned (not papered over) so a future Rust fix has a test to update
    rather than a silent surprise; see AnnotationCallout.arrow's docstring
    note and the Rust follow-up filed for this and the sibling
    AnnotationBracket.direction defect."""
    svg_curved = _render_callout("curved")
    svg_straight = _render_callout("straight")
    svg_none = _render_callout("none")
    assert svg_curved == svg_straight
    assert svg_curved != svg_none


@pytest.mark.parametrize(("alias", "canonical"), [("left", "start"), ("right", "end")])
def test_annotation_text_anchor_alias_constructs_and_renders_identically(alias, canonical):
    """render/annotation.rs::parse_anchor treats "left"/"right" as aliases
    for "start"/"end" (TextAnchor::Start / TextAnchor::End) — the annotated
    field must accept them and render byte-identically to the canonical
    value, not just avoid raising."""
    AnnotationText(
        x=0,
        y=0,
        text="t",
        font_size=12,
        color="#000",
        anchor=alias,
        baseline="middle",
        angle=0,
        dx=0,
        dy=0,
        z="above_marks",
    )
    assert _render_text(anchor=alias) == _render_text(anchor=canonical)


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("hanging", "top"),
        ("text-before-edge", "top"),
        ("central", "middle"),
        ("text-after-edge", "bottom"),
        ("ideographic", "bottom"),
    ],
)
def test_annotation_text_baseline_alias_constructs_and_renders_identically(alias, canonical):
    """render/draw.rs::parse_text_baseline recognizes several CSS
    dominant-baseline aliases per canonical variant."""
    AnnotationText(
        x=0,
        y=0,
        text="t",
        font_size=12,
        color="#000",
        anchor="start",
        baseline=alias,
        angle=0,
        dx=0,
        dy=0,
        z="above_marks",
    )
    assert _render_text(baseline=alias) == _render_text(baseline=canonical)


def test_annotation_text_baseline_alphabetic_is_a_genuinely_distinct_variant():
    """ "alphabetic" is a fourth, real TextBaseline variant — not an alias of
    "middle" and not present in the field's public docstring vocabulary
    before this fix. It must construct and render distinctly from all
    three documented baselines."""
    AnnotationText(
        x=0,
        y=0,
        text="t",
        font_size=12,
        color="#000",
        anchor="start",
        baseline="alphabetic",
        angle=0,
        dx=0,
        dy=0,
        z="above_marks",
    )
    svg_alpha = _render_text(baseline="alphabetic")
    assert svg_alpha != _render_text(baseline="top")
    assert svg_alpha != _render_text(baseline="middle")
    assert svg_alpha != _render_text(baseline="bottom")


def test_annotate_text_factory_anchor_alias_is_reachable_and_valid():
    """``annotate_text``'s ``_ANCHOR_TO_ALIGN.get(anchor, anchor)`` pass-through
    forwards an unrecognized-by-the-dict ``anchor`` value straight to
    ``AnnotationText`` unchanged, making "left"/"right" user-reachable even
    though they're absent from ``annotate_text``'s own docstring. Must not
    raise, and the resulting chart must carry the alias on the attached
    primitive."""
    chart = fm.annotate_text(1.0, 2.0, "hi", anchor="left")
    assert chart._annotation_primitive.anchor == "left"


@pytest.mark.parametrize("direction", ["up", "down", "left", "right"])
def test_annotation_bracket_direction_alias_constructs(direction):
    """ "up"/"down"/"left"/"right" are the genuinely Rust-recognized tokens
    (render/annotation.rs::emit_bracket match arms); construction must not
    raise for any of them."""
    AnnotationBracket(
        x1=0, y1=0, x2=1, y2=0, label="A", direction=direction, stroke="#333", tip_length=6
    )


def test_annotation_bracket_up_renders_identically_to_above_and_below():
    """Documents a known, out-of-scope Rust limitation this Python-only fix
    does not paper over: "above" (the documented default), "below", and
    "up" all currently render identically in emit_bracket (only "up" has an
    explicit match arm; "above"/"below" fall through to the same tip
    direction as "up"). All three must still be valid to construct."""
    svg_above = _render_bracket("above")
    svg_below = _render_bracket("below")
    svg_up = _render_bracket("up")
    assert svg_above == svg_below == svg_up


def test_annotation_bracket_down_left_right_are_genuinely_distinct():
    svg_up = _render_bracket("up")
    svg_down = _render_bracket("down")
    svg_left = _render_bracket("left")
    svg_right = _render_bracket("right")
    renders = [svg_up, svg_down, svg_left, svg_right]
    assert len(set(renders)) == 4, "up/down/left/right must each render distinctly"


# ---------------------------------------------------------------------------
# 5. annotate_text's anchor/align seam (quality-review fix): each keyword is
#    validated against its own vocabulary, and "center" -- the package's own
#    documented alignment word, not a Rust anchor token -- keeps working via
#    anchor= instead of regressing into AnnotationText's narrower validation.
# ---------------------------------------------------------------------------


def test_annotate_text_anchor_center_renders_identically_to_middle():
    """Pre-fix, ``annotate_text(anchor="center")`` constructed and rendered
    exactly like ``anchor="middle"`` (Rust's ``parse_anchor`` else-arm). The
    claim-check widening must not turn this into a ``ValueError``."""
    svg_center = fm.annotate_text(2, 4, "hi", anchor="center").to_svg()
    svg_middle = fm.annotate_text(2, 4, "hi", anchor="middle").to_svg()
    svg_align_center = fm.annotate_text(2, 4, "hi", align="center").to_svg()
    assert svg_center == svg_middle == svg_align_center


def test_annotate_text_align_bogus_names_align_not_anchor():
    """A bad ``align=`` value must raise naming ``align`` with align's own
    vocabulary -- not ``AnnotationText.anchor``'s vocabulary from two frames
    below, which would name a parameter the caller never wrote."""
    with pytest.raises(ValueError, match=r"annotate_text: align must be one of"):
        fm.annotate_text(2, 4, "hi", align="bogus")


def test_annotate_text_anchor_bogus_names_anchor():
    with pytest.raises(ValueError, match=r"annotate_text: anchor must be one of"):
        fm.annotate_text(2, 4, "hi", anchor="bogus")


def test_annotate_text_anchor_left_still_reachable_and_renders_as_start():
    """Regression guard: fixing the "center" gap must not re-break the
    "left"/"right" alias reachability the prior round's fix established."""
    chart = fm.annotate_text(1.0, 2.0, "hi", anchor="left")
    assert chart._annotation_primitive.anchor == "left"
    svg_left = fm.annotate_text(2, 4, "hi", anchor="left").to_svg()
    svg_start = fm.annotate_text(2, 4, "hi", anchor="start").to_svg()
    assert svg_left == svg_start


# ---------------------------------------------------------------------------
# 6. Null-endpoint handling in _apply_metric_label (quality-review fix):
#    a null x within a color group must not crash construction — it must
#    render the same way the pre-fix code did for that data (a degraded but
#    non-crashing chart), for both "end" and "corner" positions.
# ---------------------------------------------------------------------------


def _roc_data_with_null_endpoint(label: str, n: int = 30) -> pl.DataFrame:
    """Like ``_roc_data`` but nulls out the row that would otherwise be the
    endpoint (max-x) row, reproducing the exact shape polars' default
    descending sort mis-selects (nulls sort first)."""
    fpr = np.linspace(0, 1, n).tolist()
    fpr[-1] = None
    tpr = np.sqrt(np.linspace(0, 1, n)).tolist()
    return pl.DataFrame({"fpr": fpr, "tpr": tpr, "class": [label] * n})


@pytest.mark.parametrize("position", ["end", "corner"])
def test_null_x_endpoint_in_color_group_does_not_crash(position):
    df = pl.concat([_roc_data_with_null_endpoint("c0"), _roc_data("c1", scale=0.6)])
    base = Chart(df).encode(x="fpr", y="tpr", color="class").mark_line()
    svg = (base + AUCLabel(position=position)).to_svg()
    assert "<svg" in svg
    # Both series still emit a label (metric_fn may legitimately propagate
    # NaN for the null-poisoned group -- that's pre-existing, orthogonal
    # behavior; the point is construction/render must not raise).
    assert svg.count("AUC = ") == 2


def test_null_x_endpoint_selects_highest_non_null_x_for_end_position():
    """The endpoint row actually used for "end" placement must be the
    group's highest *non-null* x, not the null row polars' default
    descending sort would place first."""
    df = _roc_data_with_null_endpoint("c0")
    base = Chart(df).encode(x="fpr", y="tpr", color="class").mark_line()
    svg = (base + AUCLabel(position="end")).to_svg()
    xs = _text_x_positions(svg, "AUC = ")
    assert len(xs) == 1
    # The highest non-null fpr is index -2 (index -1 was nulled out); its
    # pixel x must differ from a null-selection placeholder (0.0-ish) and
    # match rendering that same non-null-endpoint chart directly.
    df_trimmed = df.head(df.height - 1)  # drop the null row entirely
    base_trimmed = Chart(df_trimmed).encode(x="fpr", y="tpr", color="class").mark_line()
    svg_trimmed = (base_trimmed + AUCLabel(position="end")).to_svg()
    assert xs == _text_x_positions(svg_trimmed, "AUC = ")


# ---------------------------------------------------------------------------
# 7. Empty-input coverage for _apply_metric_label
# ---------------------------------------------------------------------------


def test_empty_dataframe_does_not_crash():
    """Discovered while adding this coverage: an empty x column makes
    ``arg_max()`` return ``None`` (not an int index) in the no-color branch
    — a pre-existing bug (present at HEAD before any P6 change), not a
    regression this task introduced, but directly triggered by the
    empty-input case this fix was asked to cover. Fixed alongside: label
    emission is skipped (no non-null row exists to anchor one) and the base
    chart still renders, mirroring the grouped branch's own
    ``if group.is_empty(): continue``."""
    df = pl.DataFrame(
        {"fpr": pl.Series([], dtype=pl.Float64), "tpr": pl.Series([], dtype=pl.Float64)}
    )
    base = Chart(df).encode(x="fpr", y="tpr").mark_line()
    svg = (base + AUCLabel()).to_svg()
    assert "<svg" in svg
    assert "AUC = " not in svg


def test_all_null_x_no_color_group_does_not_crash():
    """Same pre-existing ``arg_max() -> None`` bug as the empty-dataframe
    case above, triggered by an all-null (rather than zero-length) x
    column."""
    df = pl.DataFrame({"fpr": [None, None, None], "tpr": [0.1, 0.2, 0.3]})
    base = Chart(df).encode(x="fpr", y="tpr").mark_line()
    svg = (base + AUCLabel()).to_svg()
    assert "<svg" in svg
    assert "AUC = " not in svg


# ---------------------------------------------------------------------------
# 8. Corner routing's only production caller: calibration_chart(compare=...)
#    is the sanctioned output change (P6's "position becomes real" claim);
#    it must be pinned by a test, not just verified by hand.
# ---------------------------------------------------------------------------


def test_calibration_chart_compare_stacks_brier_labels_at_one_x():
    """calibration_chart's Brier annotation calls
    ``_apply_metric_label_explicit(..., position="corner")`` — the only
    production caller of the corner routing this task made real. With
    multiple compared models (multiple color groups), every Brier label
    must land at the same x (the "corner" contract), not at each model's
    own curve endpoint."""
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    X, y = make_classification(n_samples=300, n_classes=2, random_state=0)
    base_model = LogisticRegression().fit(X, y)
    alt_model = LogisticRegression(C=0.1).fit(X, y)
    chart = fm.calibration_chart(base_model, X, y, compare={"alt": alt_model})
    svg = chart.to_svg()
    xs = _text_x_positions(svg, "Brier = ")
    assert len(xs) == 2
    assert len(set(xs)) == 1, "compare= calibration must stack every Brier label at one shared x"


def test_apply_metric_label_explicit_corner_directly():
    """Direct unit-level coverage of the corner branch (not routed through a
    figure function), so the routing contract is pinned independently of
    ``calibration_chart``'s other behavior."""
    from ferrum._metric_labels import _apply_metric_label_explicit

    df = pl.concat(
        [
            _roc_data("m0", scale=1.0).rename({"fpr": "predicted", "tpr": "observed"}),
            _roc_data("m1", scale=0.6).rename({"fpr": "predicted", "tpr": "observed"}),
        ]
    )
    base = Chart(df).encode(x="predicted", y="observed", color="class").mark_line()
    chart = _apply_metric_label_explicit(
        base, "brier", x_col="predicted", y_col="observed", color_col="class", position="corner"
    )
    svg = chart.to_svg()
    xs = _text_x_positions(svg, "Brier = ")
    assert len(xs) == 2
    assert len(set(xs)) == 1
