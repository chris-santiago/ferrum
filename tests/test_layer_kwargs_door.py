"""The per-layer style door ``layer_kwargs=``, and the one merger it rides.

A composite mark's ``**mark_kwargs`` reach every layer it emits, which is
wrong whenever the caller meant one of them: a bare ``stroke=`` on a
``ci``-bearing ``mark_smooth`` repaints the confidence band as well as the
fit line. ``layer_kwargs={"line": {...}}`` is the narrower spelling, and it
is deliberately NOT a per-mark parameter -- ``mark_smooth(line_kwargs=)``
shipped as one and was reshaped before release, because the same need exists
on every other layered composite (``mark_boxplot``'s median rule,
``mark_violin``'s box, ``mark_pr``'s iso lines, ...) and one named
``*_kwargs`` per mark per layer is how twelve composites drift into thirty
ad hoc parameters.

The vocabulary is not a new one either: it is
``_overrides.LAYER_NAME_CATALOG[kind]``, the same registry the figure-level
``mark={"line": {...}}`` override validates against. One vocabulary, two
doors.

The second class here pins the merger the door depends on. Two functions
used to merge a catch-all style over a layer -- ``_desugar``'s, which
consults ``_Layer.caller_style_keys``, and ``marks/_mark_kwargs``', which was
unconditionally user-wins. They agreed only because no layer reaching the
second one recorded any caller-named key, so the seam was held by two
docstrings pointing at each other rather than by a mechanism.
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from ferrum._desugar import validate_layer_kwargs
from ferrum._layer import _Layer, apply_catch_all_style
from ferrum._overrides import LAYER_NAME_CATALOG
from ferrum.marks._mark_kwargs import apply_user_mark_kwargs

#: A color no theme or palette produces, so finding it in a render means the
#: caller's own instruction reached that element and nothing else did.
AIMED = "#00ff00"


@pytest.fixture
def cat_df() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    return pl.DataFrame({"g": ["a"] * 12 + ["b"] * 12, "v": list(rng.normal(size=24))})


@pytest.fixture
def xy_df() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    x = np.arange(30.0)
    return pl.DataFrame({"x": list(x), "y": list(1.5 * x + rng.normal(0.0, 2.0, 30))})


def elements(svg: str, tag: str) -> list[str]:
    return re.findall(rf"<{tag}[^>]*/>", svg)


class TestTheDoorIsGeneral:
    """Not the smooth mark's private slot: every catalogued composite."""

    def test_boxplot_aims_style_at_one_named_layer(self, cat_df):
        """The median rule takes the color; the box rect it sits inside does not.

        ``mark_boxplot(stroke=...)`` would paint every layer -- that spread
        is what the door exists to narrow, so an assertion that merely finds
        the color somewhere in the SVG would pass on the broad spelling too.
        """
        chart = fm.Chart(cat_df).mark_boxplot(
            layer_kwargs={"median": {"stroke": AIMED}},
        )
        svg = chart.encode(x="g", y="v").to_svg()
        painted_rules = [el for el in elements(svg, "line") if AIMED in el]
        assert len(painted_rules) == 2, (
            f"expected one painted median rule per group; got {painted_rules}"
        )
        assert not [el for el in elements(svg, "rect") if AIMED in el], (
            "the box rects must NOT take a style aimed at the median layer"
        )

    def test_the_broad_kwarg_is_the_control(self, cat_df):
        """Same color, spelled broadly, does reach the box rects.

        Without this the test above could be satisfied by a door that
        silently dropped everything.
        """
        svg = fm.Chart(cat_df).mark_boxplot(stroke=AIMED).encode(x="g", y="v").to_svg()
        assert [el for el in elements(svg, "rect") if AIMED in el], (
            "a bare stroke= is expected to reach every layer, box rects included"
        )

    def test_a_layer_this_call_does_not_emit_is_a_no_op(self, cat_df):
        """``outliers=False`` draws no outlier layer, and aiming at it is
        accepted as a no-op -- the same reading ``_apply_mark_overrides``
        gives a conditionally-absent layer, because both doors are keyed on
        the mark's full catalog rather than on this call's emitted layers."""
        aimed = (
            fm.Chart(cat_df)
            .mark_boxplot(layer_kwargs={"outlier": {"fill": AIMED}}, outliers=False)
            .encode(x="g", y="v")
            .to_svg()
        )
        bare = fm.Chart(cat_df).mark_boxplot(outliers=False).encode(x="g", y="v").to_svg()
        assert aimed == bare

    def test_an_empty_per_layer_dict_changes_nothing(self, xy_df):
        """``lmplot(line_kws=None)`` forwards ``{}``; that must render
        byte-identically, not merely without error."""
        aimed = (
            fm.Chart(xy_df)
            .mark_smooth(method="lm", ci=0.95, layer_kwargs={"line": {}})
            .encode(x="x", y="y")
            .to_svg()
        )
        bare = fm.Chart(xy_df).mark_smooth(method="lm", ci=0.95).encode(x="x", y="y").to_svg()
        assert aimed == bare


class TestTheDoorRefusesInTheCallersVoice:
    def test_mark_smooth_refuses_an_unknown_layer_at_the_call(self, xy_df):
        """Eager, because ``mark_smooth`` declares the parameter: the refusal
        lands on the line that wrote the typo, not at ``.to_svg()``."""
        with pytest.raises(ValueError) as excinfo:
            fm.Chart(xy_df).mark_smooth(layer_kwargs={"lines": {"stroke": AIMED}})
        message = str(excinfo.value)
        assert "mark_smooth(layer_kwargs=)" in message
        assert "'lines'" in message
        assert "['line', 'metrics', 'ribbon']" in message, (
            f"the refusal must name the accepted set; got {message!r}"
        )

    def test_another_mark_refuses_in_its_own_name(self, cat_df):
        """The door reaches every composite through ``**mark_kwargs``, and so
        does the refusal -- named for the mark the caller actually invoked."""
        chart = fm.Chart(cat_df).mark_boxplot(layer_kwargs={"whiskers": {}}).encode(x="g", y="v")
        with pytest.raises(ValueError) as excinfo:
            chart.to_svg()
        message = str(excinfo.value)
        assert "mark_boxplot(layer_kwargs=)" in message
        assert "'whiskers'" in message
        assert "'whisker'" in message, f"the accepted set must be named; got {message!r}"

    def test_a_mark_with_no_named_layers_says_so(self, xy_df):
        """``mark_histogram`` emits one unnamed bar mark, so there is nothing
        to aim at -- and the refusal says that rather than listing an empty
        set with no explanation."""
        chart = fm.Chart(xy_df).mark_histogram(layer_kwargs={"bar": {}}).encode(x="x")
        with pytest.raises(ValueError, match="no named sub-layers"):
            chart.to_svg()

    @pytest.mark.parametrize(
        "bad",
        ["stroke", {"line": "red"}, {"line": ["stroke", "red"]}],
        ids=["not-a-mapping", "value-is-a-string", "value-is-a-list"],
    )
    def test_a_non_dict_is_refused(self, xy_df, bad):
        """Mirrors ``_apply_mark_overrides``' refusal of a non-dict override
        value: same shape of mistake, same exception type."""
        with pytest.raises(TypeError, match="layer_kwargs"):
            fm.Chart(xy_df).mark_smooth(layer_kwargs=bad)

    def test_every_catalogued_name_is_accepted_by_its_own_mark(self):
        """The door's vocabulary IS the catalog, checked wholesale rather than
        by the two marks the tests above happen to use."""
        for kind, names in LAYER_NAME_CATALOG.items():
            for name in names:
                assert validate_layer_kwargs(kind, {name: {"opacity": 0.5}}) == {
                    name: {"opacity": 0.5}
                }, f"{kind}'s own layer {name!r} must be an accepted key"


class TestOneCatchAllMerger:
    """``apply_user_mark_kwargs`` and the dispatcher share one precedence rule.

    The observable consequence: a layer that records a caller-named key keeps
    it, no matter which of the two spreaders reaches it. Before the
    unification the desugar-side spreader was unconditionally user-wins and
    would have overwritten it.
    """

    def test_a_caller_named_key_survives_the_desugar_side_spreader(self):
        layer = _Layer(
            name="line",
            mark="line",
            mark_kwargs={"stroke": "#ff0000"},
            caller_style_keys=frozenset({"stroke"}),
        )
        (merged,) = apply_user_mark_kwargs([layer], {"stroke": "#0000ff", "opacity": 0.5})
        assert merged.mark_kwargs == {"stroke": "#ff0000", "opacity": 0.5}, (
            "the key the caller aimed at this layer must outrank the broad "
            "one, and the broad one must still apply where nothing narrower "
            "was said"
        )

    def test_a_desugar_default_still_loses_to_the_broad_kwarg(self):
        """The rule is specificity, not authorship: an unrecorded key is the
        desugar's own default and the caller's catch-all overrides it (this
        is what lets ``mark_boxplot(opacity=0.8)`` beat a ribbon's 0.3)."""
        layer = _Layer(name="ribbon", mark="ribbon", mark_kwargs={"opacity": 0.3})
        (merged,) = apply_user_mark_kwargs([layer], {"opacity": 0.8})
        assert merged.mark_kwargs == {"opacity": 0.8}

    def test_the_two_spreaders_agree_layer_for_layer(self):
        """Equivalence, stated once: whatever the desugar-side spreader does
        to a layer is what the dispatcher-side merger does to it."""
        layers = [
            _Layer(name="ribbon", mark="ribbon", mark_kwargs={"opacity": 0.3}),
            _Layer(
                name="line",
                mark="line",
                mark_kwargs={"stroke": "#ff0000"},
                caller_style_keys=frozenset({"stroke"}),
            ),
            _Layer(name="metrics", mark="text"),
        ]
        style = {"stroke": "#0000ff", "opacity": 0.5}
        assert apply_user_mark_kwargs(layers, style) == [
            apply_catch_all_style(layer, style) for layer in layers
        ]

    def test_an_empty_catch_all_returns_the_layers_untouched(self):
        layers = [_Layer(name="line", mark="line", mark_kwargs={"stroke": "#ff0000"})]
        assert apply_user_mark_kwargs(layers, {}) is layers
