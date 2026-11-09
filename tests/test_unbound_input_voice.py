"""A statistic resolver's "needs a column bound" refusal speaks as the figure that called in.

Fourteen composite-mark resolvers (``mark_density``, ``mark_histogram``,
``mark_smooth``, ``mark_contour``, ``mark_violin``, ``mark_raster``,
``mark_hex``, ``mark_swarm``, ``mark_boxplot``, ``mark_errorbar``,
``mark_errorband``, ``mark_ribbon``, ``mark_boxen``, ``mark_qq``) raised a
plain ``ValueError`` naming only themselves when a required positional
channel was left with no field -- so a figure built on one of them
(``displot``, ``catplot``) that left that channel unbound spoke in the
MARK's voice, not its own. Each now raises an
:class:`~ferrum._entry.EntryRefusal` built by
:func:`~ferrum._encode_lowering._unbound_input_refusal`: the mark's own
voice keeps its unchanged wording, and any other voice (the figure that
called in) speaks the generic sentence the ECDF resolver pioneered.
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm
from ferrum.encoding import X


def _frame() -> pl.DataFrame:
    return pl.DataFrame({"g": ["a", "a", "b", "b"], "w": [1.0, 2.0, 3.0, 4.0]})


# ---------------------------------------------------------------------------
# The generic builder itself: a re-voiced refusal speaks the generic
# sentence under any voice, own text only under the mark's own.
# ---------------------------------------------------------------------------


def test_the_generic_builder_speaks_its_own_text_under_the_marks_own_voice():
    from ferrum._encode_lowering import _unbound_input_refusal

    refusal = _unbound_input_refusal(
        "mark_smooth", ("x",), "the fit reads", own="mark_smooth() requires .encode(x=...)"
    )
    assert str(refusal) == "mark_smooth() requires .encode(x=...)"


def test_the_generic_builder_speaks_the_generic_sentence_under_another_voice():
    """A re-voiced resolver refusal under an arbitrary voice -- here
    "regplot", standing in for any figure with no fieldless rule of its
    own -- speaks the same template every other re-voiced resolver does."""
    from ferrum._encode_lowering import _unbound_input_refusal

    refusal = _unbound_input_refusal(
        "mark_smooth", ("x",), "the fit reads", own="mark_smooth() requires .encode(x=...)"
    )
    assert refusal.spoken_as("regplot").message_for("regplot") == (
        "regplot needs x bound to a column -- the fit reads the variable x names, "
        "and this x binding names none. Bind x to a column: x='<column>', or "
        "fm.X('<column>', ...) to keep the channel's options."
    )


def test_a_builder_with_no_own_override_speaks_the_generic_sentence_under_its_own_voice_too():
    from ferrum._encode_lowering import _unbound_input_refusal

    refusal = _unbound_input_refusal("displot", ("x",), "the ECDF counts the raw values of")
    assert str(refusal) == (
        "displot needs x bound to a column -- the ECDF counts the raw values of "
        "the variable x names, and this x binding names none. Bind x to a column: "
        "x='<column>', or fm.X('<column>', ...) to keep the channel's options."
    )


@pytest.mark.parametrize("via", ["copy", "pickle"])
def test_a_re_voiced_refusals_text_survives_copy_and_pickle(via):
    """A plain closure has no ``voices`` to declare, so ``copy.copy``/pickle
    used to re-resolve the refusal's text generically, losing a mark's own
    wording (and displot's ECDF ``(kind='ecdf')`` qualifier). Declaring
    ``voices`` freezes each voice's text at construction instead."""
    import copy
    import pickle  # round-tripped on a value this test just built in-process, never untrusted input

    from ferrum._encode_lowering import _unbound_input_refusal

    own_refusal = _unbound_input_refusal(
        "mark_smooth",
        ("x", "y"),
        "the fit reads",
        own="mark_smooth() requires .encode(x=..., y=...)",
    )
    duplicated = (
        copy.copy(own_refusal) if via == "copy" else pickle.loads(pickle.dumps(own_refusal))
    )
    assert str(duplicated) == "mark_smooth() requires .encode(x=..., y=...)"

    reads = "the ECDF counts the raw values of"
    ecdf_refusal = _unbound_input_refusal(
        "displot",
        ("x",),
        reads,
        own=f"displot(kind='ecdf') needs x bound to a column -- {reads} the variable x names, and this x binding names none. Bind x to a column: x='<column>', or fm.X('<column>', ...) to keep the channel's options.",
    )
    duplicated_ecdf = (
        copy.copy(ecdf_refusal) if via == "copy" else pickle.loads(pickle.dumps(ecdf_refusal))
    )
    assert "kind='ecdf'" in str(duplicated_ecdf)


# ---------------------------------------------------------------------------
# Every resolver's own voice is unchanged (a direct mark_* call).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build,message",
    [
        (
            lambda: fm.Chart(_frame()).mark_density().to_svg(),
            "mark_density() requires .encode(x=...) to specify the density field",
        ),
        (
            lambda: fm.Chart(_frame()).mark_histogram().to_svg(),
            "mark_histogram() requires .encode(x=...) to specify the histogram field",
        ),
        (
            lambda: fm.Chart(_frame()).mark_smooth().to_svg(),
            "mark_smooth() requires .encode(x=..., y=...)",
        ),
        (
            lambda: fm.Chart(_frame()).mark_contour().to_svg(),
            "mark_contour() requires .encode(x=..., y=...)",
        ),
        (
            lambda: fm.Chart(_frame()).mark_violin().to_svg(),
            "mark_violin() requires .encode(x=..., y=...)",
        ),
        (
            lambda: fm.Chart(_frame()).mark_raster().to_svg(),
            "mark_raster() requires .encode(x=..., y=...)",
        ),
        (
            lambda: fm.Chart(_frame()).mark_hex().to_svg(),
            "mark_hex() requires .encode(x=..., y=...)",
        ),
        (
            lambda: fm.Chart(_frame()).mark_swarm().to_svg(),
            "mark_swarm() requires .encode(x=..., y=...)",
        ),
        (
            lambda: fm.Chart(_frame()).mark_boxplot().to_svg(),
            "mark_boxplot() requires .encode(x=..., y=...)",
        ),
        (
            lambda: fm.Chart(_frame()).mark_errorbar().to_svg(),
            "mark_errorbar() requires .encode(x=..., y=...)",
        ),
        (
            lambda: fm.Chart(_frame()).mark_errorband().to_svg(),
            "mark_errorband() requires .encode(x=..., y=...)",
        ),
        (
            lambda: fm.Chart(_frame()).mark_ribbon().to_svg(),
            "mark_ribbon() requires .encode(x=..., y=...)",
        ),
        (
            lambda: fm.Chart(_frame()).mark_boxen().to_svg(),
            "mark_boxen() requires .encode(x=..., y=...)",
        ),
        (
            lambda: fm.Chart(_frame()).mark_qq().to_svg(),
            "mark_qq() requires .encode(x=...) to specify the sample field",
        ),
    ],
)
def test_a_marks_own_voice_keeps_its_unchanged_wording(build, message):
    with pytest.raises(ValueError) as err:
        build()
    assert str(err.value) == message


# ---------------------------------------------------------------------------
# A figure's voice speaks the generic sentence instead.
# ---------------------------------------------------------------------------


def test_displot_histogram_speaks_its_own_voice_for_an_unbound_axis():
    with pytest.raises(ValueError) as err:
        fm.displot(_frame(), x="w", kind="hist").encode(x=X(title="T")).to_svg()
    assert str(err.value) == (
        "displot needs x bound to a column -- the histogram bins the variable x "
        "names, and this x binding names none. Bind x to a column: x='<column>', "
        "or fm.X('<column>', ...) to keep the channel's options."
    )


def test_displot_density_speaks_its_own_voice_for_an_unbound_axis():
    with pytest.raises(ValueError) as err:
        fm.displot(_frame(), x="w", kind="kde").encode(x=X(title="T")).to_svg()
    assert str(err.value) == (
        "displot needs x bound to a column -- the kernel density estimate reads "
        "the variable x names, and this x binding names none. Bind x to a column: "
        "x='<column>', or fm.X('<column>', ...) to keep the channel's options."
    )


def test_displot_ecdf_fieldless_x_keeps_its_existing_text():
    """Unchanged: the ECDF resolver already spoke this sentence before this
    fix, now built by the same shared builder every other resolver uses."""
    with pytest.raises(ValueError) as err:
        fm.displot(_frame(), x="w", kind="ecdf").encode(x=X(title="T")).to_svg()
    assert str(err.value) == (
        "displot(kind='ecdf') needs x bound to a column -- the ECDF counts the raw "
        "values of the variable x names, and this x binding names none. Bind x to a "
        "column: x='<column>', or fm.X('<column>', ...) to keep the channel's options."
    )


@pytest.mark.parametrize(
    "kind,reads", [("box", "boxplot"), ("violin", "violin"), ("boxen", "boxen")]
)
def test_catplot_speaks_its_own_voice_for_an_unbound_axis_at_the_call(kind, reads):
    with pytest.raises(ValueError) as err:
        fm.catplot(_frame(), y="w", kind=kind).to_svg()
    assert str(err.value) == (
        f"catplot needs x bound to a column -- the {reads} reads the variable x "
        "names, and this x binding names none. Bind x to a column: x='<column>', "
        "or fm.X('<column>', ...) to keep the channel's options."
    )


def test_catplot_speaks_its_own_voice_for_an_unbound_axis_post_hoc():
    chart = fm.catplot(_frame(), x="g", y="w", kind="box")
    with pytest.raises(ValueError) as err:
        chart.encode(x=X(title="T")).to_svg()
    assert str(err.value) == (
        "catplot needs x bound to a column -- the boxplot reads the variable x "
        "names, and this x binding names none. Bind x to a column: x='<column>', "
        "or fm.X('<column>', ...) to keep the channel's options."
    )


def test_a_plain_chart_with_no_x_keeps_the_marks_own_voice():
    """Unchanged: a plain `Chart(df).mark_smooth()` with no x speaks in its
    own voice, since no figure stamped a different entry on it."""
    with pytest.raises(ValueError) as err:
        fm.Chart(_frame()).mark_smooth().to_svg()
    assert str(err.value) == "mark_smooth() requires .encode(x=..., y=...)"
