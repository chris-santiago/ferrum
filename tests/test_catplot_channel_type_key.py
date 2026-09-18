"""catplot's channel rebuilds write the ``type`` key the channel already speaks.

``_catplot_build`` rebuilds the category channel and (under ``hue_order=``)
the color channel from a COPY of the caller's own ``ChannelBase._kwargs`` --
a dict that ``ChannelBase.__init__`` has ALREADY normalized, where the data
type lives under ``"type"``.  Both rebuilds wrote ``"type_"`` onto it, and
``ChannelBase.__init__`` discards ``type_`` whenever ``type`` is present.

The reachable shape is a channel that states ``type_=None`` explicitly:
``fm.X("year", type_=None)`` carries ``{"type": None}``, which reads as
untyped through ``option("type")`` -- so catplot enters the "the caller
stated no type, normalize it" branch -- and then silently drops the typing
that branch exists to apply.  Measured on the channel builder itself:

    Color('grp', **{'type': None, 'type_': 'nominal'})._kwargs
    -> {'type': None}          # the nominal typing, gone
    Color('grp', **{'type': 'nominal'})._kwargs
    -> {'type': 'N'}           # what the branch meant to say

One key, one spelling, in both rebuilds.  Scoped to that defect per this
repo's test-file convention; the hue_order semantics it rides on are pinned
in ``test_catplot_hue_order.py``.
"""

from __future__ import annotations

import json
import re

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from tests._hue_probe import has_colorbar


@pytest.fixture
def numeric_df():
    """Numeric category AND numeric hue columns.

    Both have to be numeric: an untyped numeric column is the only one whose
    inferred type (Quantitative) differs from what catplot's rebuild states
    (Nominal), so it is the only one on which a dropped `type` is visible.
    """
    rng = np.random.default_rng(0)
    return pl.DataFrame(
        {
            "year": [2020, 2021, 2022] * 20,
            "v": rng.normal(size=60),
            "grp_num": [x % 3 for x in range(60)],
        }
    )


def _encoding(chart, channel: str) -> dict:
    return json.loads(chart.to_spec().to_json())["encoding"][channel]


class TestCategoryChannelTyping:
    """``x=fm.X(col, type_=None)`` gets catplot's normalization, not a silent drop."""

    def test_wire_states_the_nominal_typing(self, numeric_df):
        chart = fm.catplot(numeric_df, x=fm.X("year", type_=None), y="v", kind="box")
        assert _encoding(chart, "x")["type"] == "nominal"

    def test_the_axis_renders_three_categories_not_a_number_line(self, numeric_df):
        """The rendered consequence, not just the wire.

        A nominal year axis carries exactly the three levels present; the
        Quantitative reading a dropped typing leaves behind carries a number
        line through them -- measured, the same call renders ``2020.2``,
        ``2020.4``, ... as tick labels, and the marks land at interpolated
        positions rather than one per category.
        """
        svg = fm.catplot(numeric_df, x=fm.X("year", type_=None), y="v", kind="point").to_svg()
        year_labels = {
            label
            for label in re.findall(r"<text[^>]*>([^<]*)</text>", svg)
            if label.startswith("2020") or label.startswith("2021") or label.startswith("2022")
        }
        assert year_labels == {"2020", "2021", "2022"}, (
            f"x tick labels {sorted(year_labels)} -- a fractional label means the "
            "axis resolved continuous, i.e. catplot's nominal typing was dropped"
        )

    def test_a_stated_type_is_still_honored_verbatim(self, numeric_df):
        """The control: this fix must not start overriding an explicit type."""
        chart = fm.catplot(numeric_df, x=fm.X("year", type_="quantitative"), y="v", kind="point")
        assert _encoding(chart, "x")["type"] == "quantitative"

    def test_a_bare_string_category_is_unchanged(self, numeric_df):
        """The spelling with no `_kwargs` to collide with reads the same."""
        chart = fm.catplot(numeric_df, x="year", y="v", kind="box")
        assert _encoding(chart, "x")["type"] == "nominal"


class TestHueChannelTyping:
    """``hue=fm.Color(col, type_=None)`` + ``hue_order=`` types nominal too."""

    def test_wire_states_the_nominal_typing_and_the_order(self, numeric_df):
        chart = fm.catplot(
            numeric_df,
            x="year",
            y="v",
            kind="strip",
            hue=fm.Color("grp_num", type_=None),
            hue_order=[0, 1, 2],
        )
        color = _encoding(chart, "color")
        assert color["type"] == "nominal"
        assert color["sort"] == [0, 1, 2]

    def test_the_render_carries_a_swatch_legend_not_a_colorbar(self, numeric_df):
        """The rendered consequence: a dropped typing left the numeric hue
        inferring Continuous, i.e. a colorbar standing in for the group
        discriminator `hue_order=` was ordering."""
        svg = fm.catplot(
            numeric_df,
            x="year",
            y="v",
            kind="strip",
            hue=fm.Color("grp_num", type_=None),
            hue_order=[0, 1, 2],
        ).to_svg()
        assert not has_colorbar(svg), "a numeric hue still renders a continuous colorbar"

    def test_an_explicitly_continuous_hue_still_refuses_hue_order(self, numeric_df):
        """The control from the other side: an explicit continuous type is
        honored, which is why `hue_order=` on it is refused rather than
        quietly retyped."""
        with pytest.raises(ValueError, match="hue_order= cannot order a quantitative"):
            fm.catplot(
                numeric_df,
                x="year",
                y="v",
                kind="strip",
                hue=fm.Color("grp_num", type_="quantitative"),
                hue_order=[0, 1, 2],
            )
