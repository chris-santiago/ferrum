"""v0.15.1 audit remediation — title=None suppression bug.

Bug: ``Axis(title=None)`` and ``Legend(title=None)`` did not suppress the
axis/legend title because their ``to_dict()`` methods skipped all ``None``
values, emitting ``{}`` (no "title" key).  Rust then fell through to the
field-name default.

Fix: ``Axis`` and ``Legend`` now use a sentinel default (``_UNSET``) for the
``title`` field so ``to_dict()`` can distinguish three states:

  - ``_UNSET`` (omitted)   → key absent; Rust uses the field name (unchanged default)
  - ``None`` (explicit)    → emit ``title: ""``;  Rust treats ``""`` as suppress
  - ``"Foo"`` (string)     → emit ``title: "Foo"``; Rust displays that string

The convention mirrors ``base.py`` (channel-level ``title`` kwarg) and
``prepare.rs`` (three-way resolver for the axis path).

Note: the legend title path in Rust does not have the same three-way
empty-string resolver as the axis path.  After the Python fix, the field name
is correctly suppressed (replaced with ``""``), but a zero-width phantom
``<text></text>`` SVG element may remain.  This is a known Rust-side
limitation tracked separately; it is not a regression introduced here.
"""

from __future__ import annotations

import re

import polars as pl
import pytest

import ferrum as fm
from ferrum._title_sentinel import _UNSET, serialize_title
from ferrum.axis import Axis
from ferrum.legend import Legend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _svg_texts(svg: str) -> list[str]:
    """Return all non-empty text strings from ``<text>`` elements in the SVG."""
    return [t for t in re.findall(r"<text[^>]*>([^<]+)</text>", svg) if t.strip()]


@pytest.fixture()
def simple_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "horsepower": [130.0, 165.0, 150.0],
            "mpg": [18.0, 15.0, 18.0],
            "origin": ["usa", "usa", "europe"],
        }
    )


# ---------------------------------------------------------------------------
# serialize_title helper — unit tests
# ---------------------------------------------------------------------------


class TestSerializeTitle:
    """``serialize_title`` encodes the three-way contract as a pure function."""

    def test_unset_returns_none(self) -> None:
        assert serialize_title(_UNSET) is None

    def test_none_returns_empty_string(self) -> None:
        assert serialize_title(None) == ""

    def test_string_returns_string(self) -> None:
        assert serialize_title("Custom") == "Custom"

    def test_empty_string_returns_empty_string(self) -> None:
        # An explicitly passed "" is already the suppress sentinel.
        assert serialize_title("") == ""


# ---------------------------------------------------------------------------
# Axis.to_dict — unit tests (no render needed)
# ---------------------------------------------------------------------------


class TestAxisToDictTitleContract:
    """``Axis.to_dict()`` implements the three-way title contract."""

    def test_omitted_title_does_not_emit_key(self) -> None:
        """``Axis()`` (title not passed) must not emit 'title' key."""
        assert "title" not in Axis().to_dict()

    def test_none_title_emits_empty_string(self) -> None:
        """``Axis(title=None)`` must emit ``{'title': ''}``.

        An empty string is the Rust-side suppress sentinel (``prepare.rs``
        three-way resolver: ``Some("")`` → ``None`` title, no fallback).
        """
        result = Axis(title=None).to_dict()
        assert result.get("title") == ""

    def test_string_title_emits_verbatim(self) -> None:
        """``Axis(title='Custom')`` must emit ``{'title': 'Custom'}``."""
        result = Axis(title="Custom").to_dict()
        assert result.get("title") == "Custom"

    def test_empty_string_title_emits_empty_string(self) -> None:
        """``Axis(title='')`` already suppressed before this fix — stays correct."""
        result = Axis(title="").to_dict()
        assert result.get("title") == ""

    def test_omitted_preserves_other_fields(self) -> None:
        """Other axis fields still serialize correctly when title is omitted."""
        result = Axis(grid=False, label_angle=-45.0).to_dict()
        assert "title" not in result
        assert result["grid"] is False
        assert result["label_angle"] == -45.0

    def test_none_title_with_other_fields(self) -> None:
        """``title=None`` combines correctly with other fields."""
        result = Axis(title=None, grid=False).to_dict()
        assert result["title"] == ""
        assert result["grid"] is False


# ---------------------------------------------------------------------------
# Legend.to_dict — unit tests (no render needed)
# ---------------------------------------------------------------------------


class TestLegendToDictTitleContract:
    """``Legend.to_dict()`` implements the three-way title contract."""

    def test_omitted_title_does_not_emit_key(self) -> None:
        """``Legend()`` (title not passed) must not emit 'title' key."""
        assert "title" not in Legend().to_dict()

    def test_none_title_emits_empty_string(self) -> None:
        """``Legend(title=None)`` must emit ``{'title': ''}``.

        The empty string propagates to Rust's legend title resolver.
        """
        result = Legend(title=None).to_dict()
        assert result.get("title") == ""

    def test_string_title_emits_verbatim(self) -> None:
        """``Legend(title='Species')`` must emit ``{'title': 'Species'}``."""
        result = Legend(title="Species").to_dict()
        assert result.get("title") == "Species"

    def test_empty_string_title_emits_empty_string(self) -> None:
        """``Legend(title='')`` must emit ``{'title': ''}``."""
        result = Legend(title="").to_dict()
        assert result.get("title") == ""

    def test_omitted_preserves_other_fields(self) -> None:
        """Other legend fields still serialize correctly when title is omitted."""
        result = Legend(orient="bottom", columns=3).to_dict()
        assert "title" not in result
        assert result["orient"] == "bottom"
        assert result["columns"] == 3

    def test_none_title_with_other_fields(self) -> None:
        """``title=None`` combines correctly with other legend fields."""
        result = Legend(title=None, orient="bottom").to_dict()
        assert result["title"] == ""
        assert result["orient"] == "bottom"


# ---------------------------------------------------------------------------
# Channel-level encoding title (fm.X / fm.Y / fm.Color) — unit tests
# ---------------------------------------------------------------------------


class TestChannelLevelTitleContract:
    """Channel-level ``title=None`` already worked; regression guard."""

    def test_x_title_none_emits_empty_string(self) -> None:
        """``fm.X('field', title=None)`` must emit ``title: ''`` in the spec."""
        enc = fm.X("horsepower", title=None)
        spec = enc.to_encoding_spec_dict()
        assert spec.get("title") == ""

    def test_x_title_omitted_does_not_emit_key(self) -> None:
        """``fm.X('field')`` (no title kwarg) must not emit 'title' key."""
        enc = fm.X("horsepower")
        spec = enc.to_encoding_spec_dict()
        assert "title" not in spec

    def test_x_title_custom_emits_verbatim(self) -> None:
        """``fm.X('field', title='HP')`` must emit ``title: 'HP'``."""
        enc = fm.X("horsepower", title="HP")
        spec = enc.to_encoding_spec_dict()
        assert spec.get("title") == "HP"

    def test_color_title_none_emits_empty_string(self) -> None:
        """``fm.Color('field', title=None)`` must emit ``title: ''``."""
        enc = fm.Color("origin", title=None)
        spec = enc.to_encoding_spec_dict()
        assert spec.get("title") == ""

    def test_color_title_omitted_does_not_emit_key(self) -> None:
        """``fm.Color('field')`` (no title kwarg) must not emit 'title' key."""
        enc = fm.Color("origin")
        spec = enc.to_encoding_spec_dict()
        assert "title" not in spec

    def test_color_title_custom_emits_verbatim(self) -> None:
        """``fm.Color('field', title='Region')`` must emit ``title: 'Region'``."""
        enc = fm.Color("origin", title="Region")
        spec = enc.to_encoding_spec_dict()
        assert spec.get("title") == "Region"


# ---------------------------------------------------------------------------
# Axis render tests — SVG-level assertions
# ---------------------------------------------------------------------------


class TestAxisRenderTitleSuppression:
    """``Axis(title=None)`` suppresses the axis title in the rendered SVG."""

    def test_axis_default_shows_field_name(self, simple_df: pl.DataFrame) -> None:
        """``Axis()`` (title omitted) must render the field name as the axis title."""
        svg = (
            fm.Chart(simple_df)
            .mark_point()
            .encode(x=fm.X("horsepower", axis=Axis()), y="mpg")
            .show_svg()
        )
        texts = _svg_texts(svg)
        assert "horsepower" in texts, (
            f"Axis() with no title must show field name as axis title; texts: {texts}"
        )

    def test_axis_title_none_suppresses_field_name(self, simple_df: pl.DataFrame) -> None:
        """``Axis(title=None)`` must suppress the field name from the x-axis."""
        svg = (
            fm.Chart(simple_df)
            .mark_point()
            .encode(x=fm.X("horsepower", axis=Axis(title=None)), y="mpg")
            .show_svg()
        )
        texts = _svg_texts(svg)
        assert "horsepower" not in texts, (
            "Axis(title=None) must suppress the axis title; "
            f"'horsepower' still present in texts: {texts}"
        )

    def test_axis_title_none_no_empty_text_node(self, simple_df: pl.DataFrame) -> None:
        """``Axis(title=None)`` must not emit a phantom empty ``<text>`` node."""
        svg = (
            fm.Chart(simple_df)
            .mark_point()
            .encode(x=fm.X("horsepower", axis=Axis(title=None)), y="mpg")
            .show_svg()
        )
        empty_nodes = re.findall(r"<text[^>]*>\s*</text>", svg)
        assert not empty_nodes, (
            "Axis(title=None) must not emit empty <text> nodes; "
            f"found {len(empty_nodes)} empty text node(s)"
        )

    def test_axis_title_custom_shows_custom(self, simple_df: pl.DataFrame) -> None:
        """``Axis(title='Engine Power')`` must display the custom title."""
        svg = (
            fm.Chart(simple_df)
            .mark_point()
            .encode(x=fm.X("horsepower", axis=Axis(title="Engine Power")), y="mpg")
            .show_svg()
        )
        texts = _svg_texts(svg)
        assert "Engine Power" in texts, (
            f"Axis(title='Engine Power') must show that title; texts: {texts}"
        )
        assert "horsepower" not in texts, "Field name must not appear when Axis title is customized"

    def test_axis_title_none_y_channel_suppresses(self, simple_df: pl.DataFrame) -> None:
        """``Axis(title=None)`` on the Y channel suppresses the y-axis title."""
        svg = (
            fm.Chart(simple_df)
            .mark_point()
            .encode(x="horsepower", y=fm.Y("mpg", axis=Axis(title=None)))
            .show_svg()
        )
        texts = _svg_texts(svg)
        assert "mpg" not in texts, (
            "Axis(title=None) on Y must suppress the y-axis title; "
            f"'mpg' still present in texts: {texts}"
        )


# ---------------------------------------------------------------------------
# Legend render tests — SVG-level assertions
# ---------------------------------------------------------------------------


class TestLegendRenderTitleSuppression:
    """``Legend(title=None)`` suppresses the legend title field name in the SVG.

    The Rust legend title path now implements the same three-way empty-string
    contract as the axis path (v0.15.1 Rust fix): ``Some("")`` → no text node,
    no reserved margin.  Both field-name suppression and phantom-node absence
    are asserted here.
    """

    def test_legend_default_shows_field_name(self, simple_df: pl.DataFrame) -> None:
        """``Legend()`` (title omitted) must render the field name as the legend title."""
        svg = (
            fm.Chart(simple_df)
            .mark_point()
            .encode(x="horsepower", y="mpg", color=fm.Color("origin", legend=Legend()))
            .show_svg()
        )
        texts = _svg_texts(svg)
        assert "origin" in texts, (
            f"Legend() with no title must show field name as legend title; texts: {texts}"
        )

    def test_legend_title_none_suppresses_field_name(self, simple_df: pl.DataFrame) -> None:
        """``Legend(title=None)`` must suppress the field name from the legend."""
        svg = (
            fm.Chart(simple_df)
            .mark_point()
            .encode(x="horsepower", y="mpg", color=fm.Color("origin", legend=Legend(title=None)))
            .show_svg()
        )
        texts = _svg_texts(svg)
        assert "origin" not in texts, (
            "Legend(title=None) must suppress the legend title (field name absent); "
            f"'origin' still present in texts: {texts}"
        )

    def test_legend_title_none_no_empty_text_node(self, simple_df: pl.DataFrame) -> None:
        """``Legend(title=None)`` must not emit a phantom empty ``<text>`` node."""
        svg = (
            fm.Chart(simple_df)
            .mark_point()
            .encode(x="horsepower", y="mpg", color=fm.Color("origin", legend=Legend(title=None)))
            .show_svg()
        )
        empty_nodes = re.findall(r"<text[^>]*>\s*</text>", svg)
        assert not empty_nodes, (
            "Legend(title=None) must not emit empty <text> nodes; "
            f"found {len(empty_nodes)} empty text node(s)"
        )

    def test_legend_title_custom_shows_custom(self, simple_df: pl.DataFrame) -> None:
        """``Legend(title='Region')`` must display the custom title."""
        svg = (
            fm.Chart(simple_df)
            .mark_point()
            .encode(
                x="horsepower",
                y="mpg",
                color=fm.Color("origin", legend=Legend(title="Region")),
            )
            .show_svg()
        )
        texts = _svg_texts(svg)
        assert "Region" in texts, f"Legend(title='Region') must show that title; texts: {texts}"
        assert "origin" not in texts, "Field name must not appear when Legend title is customized"

    def test_channel_title_kwarg_distinct_from_legend_title(self, simple_df: pl.DataFrame) -> None:
        """Channel-level ``title=None`` on a color encoding is NOT the legend title.

        The ``title`` kwarg on a channel encoding is an axis title override;
        it does not control the legend title.  The legend title is a separate
        concept, controlled by passing a ``Legend(title=...)`` instance to the
        ``legend`` kwarg.  This test confirms that ``Color('origin', title=None)``
        does NOT suppress the legend title (the field name still appears in the
        legend because the legend title is unaffected by the channel title kwarg).
        """
        svg = (
            fm.Chart(simple_df)
            .mark_point()
            .encode(x="horsepower", y="mpg", color=fm.Color("origin", title=None))
            .show_svg()
        )
        texts = _svg_texts(svg)
        # The field name appears as the legend title (channel title kwarg does not
        # affect legend title — separate control surface).
        assert "origin" in texts, (
            "Color('origin', title=None): field name must still appear as legend title "
            "because channel-title and legend-title are separate kwargs; "
            f"texts: {texts}"
        )
