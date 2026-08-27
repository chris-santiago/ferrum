"""Regression tests for T3.6 / XNAME-03 — one share_scale validation path.

XNAME-03: ``Chart.share_scale`` had its own inline ``_valid`` tuple and
hand-rolled error raises.  The composition siblings (``_ChartLike.share_scale``
and ``RepeatChart.share_scale``) already routed through the shared validator
``composition._validate_share_modes``.  This drift meant two error-message
formats, two copies of the mode vocabulary, and no guarantee that the three
entry points stayed in sync.

Fix: ``Chart.share_scale`` now builds a ``{ch: mode}`` dict from its x/y
keyword args and passes it to ``_validate_share_modes`` (same path as the
composition siblings).  The inline ``_valid`` tuple and per-channel raises are
deleted.  The x/y-only signature is kept because Rust's ``FacetResolve`` struct
only has ``x`` and ``y`` fields; widening to ``**channels`` would silently
accept channel names the renderer ignores.

These tests assert:
1. All three siblings accept the same mode vocabulary (shared/independent).
2. All three siblings reject the same invalid mode, and the error message shape
   is identical (``_validate_share_modes`` owns the message).
3. ``Chart.share_scale`` still raises before ``facet()`` (Chart-specific guard,
   not drift).
4. The error message from ``Chart.share_scale`` for an invalid mode is produced
   by ``_validate_share_modes`` (same text as the composition siblings).
"""

from __future__ import annotations

import pytest
import polars as pl

import ferrum as fm
from ferrum import Chart
from ferrum.composition import _validate_share_modes, RepeatChart, Resolve


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "g": ["a", "b", "a"]})


@pytest.fixture
def chart(df: pl.DataFrame) -> Chart:
    return Chart(df).mark_point().encode(x="x", y="y")


@pytest.fixture
def faceted(df: pl.DataFrame) -> Chart:
    return Chart(df).mark_point().encode(x="x", y="y").facet(col="g")


@pytest.fixture
def composition(chart: Chart) -> fm.LayerChart:
    return chart | chart


@pytest.fixture
def repeat_chart(df: pl.DataFrame) -> RepeatChart:
    template = Chart(df).encode(x=fm.Repeat.column, y=fm.Repeat.row).mark_point()
    return RepeatChart(template, row=["x", "y"], column=["x", "y"])


# ---------------------------------------------------------------------------
# 1. All three siblings accept "shared" and "independent"
# ---------------------------------------------------------------------------


def test_chart_share_scale_accepts_shared(faceted: Chart) -> None:
    """Chart.share_scale accepts 'shared' without raising."""
    result = faceted.share_scale(x="shared")
    assert isinstance(result, Chart)


def test_chart_share_scale_accepts_independent(faceted: Chart) -> None:
    """Chart.share_scale accepts 'independent' without raising."""
    result = faceted.share_scale(y="independent")
    assert isinstance(result, Chart)


def test_composition_share_scale_accepts_shared(composition: fm.LayerChart) -> None:
    """_ChartLike.share_scale accepts 'shared' without raising."""
    result = composition.share_scale(x="shared")
    assert result is not composition


def test_composition_share_scale_accepts_independent(chart: Chart) -> None:
    """_ChartLike.share_scale accepts 'independent' without raising.

    Post-unification (#45 close), share_scale is sugar for resolve= -- an
    explicit "independent" merges into ``_resolve`` as real metadata rather
    than short-circuiting to an identity no-op, even though it renders the
    same as the (also-independent) default.
    """
    combined = chart | chart
    result = combined.share_scale(x="independent")
    assert result is not combined
    assert result._resolve == Resolve(scale={"x": "independent"})
    assert result.to_svg() == combined.to_svg()


def test_repeat_share_scale_accepts_shared(repeat_chart: RepeatChart) -> None:
    """RepeatChart.share_scale accepts 'shared' without raising."""
    result = repeat_chart.share_scale(x="shared")
    assert result.resolve == Resolve(scale={"x": "shared"})


def test_repeat_share_scale_accepts_independent(repeat_chart: RepeatChart) -> None:
    """RepeatChart.share_scale accepts 'independent' without raising."""
    result = repeat_chart.share_scale(y="independent")
    assert result.resolve == Resolve(scale={"y": "independent"})


# ---------------------------------------------------------------------------
# 2. All three siblings reject the same invalid mode with identical error shape
# ---------------------------------------------------------------------------


def _invalid_mode_message(ch: str, mode: str) -> str:
    """The canonical error message produced by _validate_share_modes."""
    return f"share_scale: {ch} must be one of ['independent', 'shared']; got {mode!r}"


def test_validate_share_modes_produces_canonical_message() -> None:
    """_validate_share_modes is the single source of the error message."""
    with pytest.raises(ValueError, match=r"share_scale: x must be one of .*; got 'bad'"):
        _validate_share_modes({"x": "bad"})


def test_chart_share_scale_invalid_mode_uses_shared_validator(faceted: Chart) -> None:
    """Chart.share_scale invalid mode raises via _validate_share_modes (same message)."""
    with pytest.raises(ValueError, match=r"share_scale: x must be one of .*; got 'together'"):
        faceted.share_scale(x="together")


def test_composition_share_scale_invalid_mode_uses_shared_validator(
    composition: fm.LayerChart,
) -> None:
    """_ChartLike.share_scale invalid mode raises via _validate_share_modes (same message)."""
    with pytest.raises(ValueError, match=r"share_scale: x must be one of .*; got 'together'"):
        composition.share_scale(x="together")


def test_repeat_share_scale_invalid_mode_uses_shared_validator(repeat_chart: RepeatChart) -> None:
    """RepeatChart.share_scale invalid mode raises via _validate_share_modes (same message)."""
    with pytest.raises(ValueError, match=r"share_scale: x must be one of .*; got 'together'"):
        repeat_chart.share_scale(x="together")


def test_all_three_invalid_mode_messages_identical(
    faceted: Chart,
    composition: fm.LayerChart,
    repeat_chart: RepeatChart,
) -> None:
    """All three siblings produce the exact same error message for an invalid mode.

    This pins that validation flows through _validate_share_modes exclusively,
    not through any sibling-local hand-rolled raise.
    """
    invalid_mode = "bogus"
    messages = []
    for obj, kwargs in [
        (faceted, {"x": invalid_mode}),
        (composition, {"x": invalid_mode}),
        (repeat_chart, {"x": invalid_mode}),
    ]:
        try:
            obj.share_scale(**kwargs)
            pytest.fail(f"{type(obj).__name__}.share_scale did not raise for invalid mode")
        except ValueError as exc:
            messages.append(str(exc))
    # All three messages must be identical.
    assert messages[0] == messages[1] == messages[2], (
        f"Error messages diverge across siblings:\n"
        f"  Chart:       {messages[0]!r}\n"
        f"  Composition: {messages[1]!r}\n"
        f"  Repeat:      {messages[2]!r}"
    )


# ---------------------------------------------------------------------------
# 3. Chart.share_scale still raises before facet() (Chart-specific guard)
# ---------------------------------------------------------------------------


def test_chart_share_scale_requires_facet_first(chart: Chart) -> None:
    """Chart.share_scale raises ValueError when called before facet()."""
    with pytest.raises(ValueError, match="facet\\(\\) first"):
        chart.share_scale(x="shared")


def test_chart_share_scale_precondition_before_validation(chart: Chart) -> None:
    """The facet precondition is checked before mode validation.

    Even an invalid mode must not bypass the 'call facet() first' guard.
    """
    # This should raise about missing facet(), not about invalid mode.
    with pytest.raises(ValueError, match="facet\\(\\) first"):
        chart.share_scale(x="bogus")


# ---------------------------------------------------------------------------
# 4. Behavioral parity: Chart.share_scale stores modes in resolve dict correctly
# ---------------------------------------------------------------------------


def test_chart_share_scale_stores_x_and_y(faceted: Chart) -> None:
    """Chart.share_scale correctly stores both x and y in the resolve dict."""
    result = faceted.share_scale(x="shared", y="independent")
    assert result._facet.resolve == {"x": "shared", "y": "independent"}


def test_chart_share_scale_none_args_are_no_op(faceted: Chart) -> None:
    """Chart.share_scale with both args None leaves resolve unchanged."""
    original_resolve = faceted._facet.resolve
    result = faceted.share_scale()
    assert result._facet.resolve == original_resolve


def test_chart_share_scale_merges_with_existing_resolve(df: pl.DataFrame) -> None:
    """Chart.share_scale merges new modes with existing resolve settings."""
    faceted = Chart(df).mark_point().encode(x="x", y="y").facet(col="g").share_scale(x="shared")
    result = faceted.share_scale(y="independent")
    assert result._facet.resolve == {"x": "shared", "y": "independent"}
