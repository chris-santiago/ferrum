"""Cohesion tests for Task R8: narrowed except in ``_apply_overrides``.

Originally this file also covered two other R8 cleanups — a shared offset
key-set used by the Python scene-merge internals, and a ``_FigureChrome``
TypedDict returned by composite chrome resolution. Task 10
(composite-render-unification, Phase B) deleted the scene-merge machinery
(the legacy Python scene-merge module, ``_merge_scene_panels``, ``_inject_figure_chrome``,
``_merge_one_child``, ``_FigureChrome``, and friends) entirely — every
composite now renders through one Rust composite-render entry, so those two
areas of coverage no longer apply and were removed with the module they
tested. The narrowed-except behavior below is unrelated to composition
rendering and remains in force.

(b) Narrowed except — _apply_overrides now uses getattr(..., None) to handle
    missing _rebuild_with_charts (no AttributeError catch at all), and catches
    only NotImplementedError around the rebuild call itself.  An unrelated
    AttributeError raised inside a child's .properties() during _apply now
    propagates rather than being swallowed.  The expected fallback for chart
    types that declare but have not implemented _rebuild_with_charts (the
    abstract _ChartLike slot) is preserved.
"""

from __future__ import annotations

import inspect

import polars as pl
import pytest

import ferrum as fm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df():
    return pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})


@pytest.fixture
def base_chart(df):
    return fm.Chart(df).mark_point().encode(x="x", y="y")


# ---------------------------------------------------------------------------
# (b) Narrowed except: _apply definition is outside the try block
# ---------------------------------------------------------------------------


def test_overrides_apply_definition_outside_try_block():
    """The _apply closure must be defined BEFORE the try in _apply_overrides source.

    The refactor uses getattr(..., None) to handle the missing-method case
    (avoiding an AttributeError catch entirely) and wraps only the rebuild()
    call in a try/except NotImplementedError.  The _apply definition therefore
    appears before any try block in the non-Chart branch, and the except covers
    only NotImplementedError — not AttributeError.

    We verify the intent structurally: in the source the 'def _apply' token
    must appear before the 'try:' token (within the non-Chart branch), and
    'AttributeError' must NOT appear in the except clause.
    """
    import ferrum._overrides as mod

    src = inspect.getsource(mod._apply_overrides)

    # Find the position of 'def _apply' and 'try:' in the source snippet for
    # the non-Chart branch.  Strip the outer function header so both searches
    # are scoped to the body.
    body = src.split("if not isinstance(chart, Chart):", 1)[-1]

    def_pos = body.find("def _apply(")
    try_pos = body.find("try:")

    assert def_pos != -1, "'def _apply' not found in non-Chart branch"
    assert try_pos != -1, "'try:' not found in non-Chart branch"
    assert def_pos < try_pos, (
        "Expected 'def _apply' to appear before 'try:' in the source, "
        "but found them in the wrong order."
    )

    # The except clause must catch ONLY NotImplementedError — not AttributeError.
    # The missing-method case is handled via getattr(..., None), not by catching
    # AttributeError, so child AttributeErrors now propagate.
    except_pos = body.find("except ")
    assert except_pos != -1, "'except' not found in non-Chart branch"
    except_line_end = body.find("\n", except_pos)
    except_clause = body[except_pos:except_line_end]
    assert "AttributeError" not in except_clause, (
        f"'except' clause should not catch AttributeError; got: {except_clause!r}"
    )
    assert "NotImplementedError" in except_clause, (
        f"'except' clause should catch NotImplementedError; got: {except_clause!r}"
    )


# ---------------------------------------------------------------------------
# (b) Narrowed except: behavioral — propagation and fallback
# ---------------------------------------------------------------------------


def test_child_attribute_error_propagates_through_apply_overrides(df):
    """An AttributeError from inside the try-wrapped rebuild(_apply) call must propagate.

    Discriminating design: the sentinel AttributeError is raised ONLY from
    within ``_rebuild_with_charts`` (the try-wrapped call).  The fallback path
    (``chart.properties(**child_properties)`` in the ``except NotImplementedError``
    branch) calls an independent ``.properties()`` method that does NOT
    invoke ``_rebuild_with_charts``, so the fallback would SUCCEED if control
    ever reached it.  The only way the sentinel can surface is via uncaught
    propagation from the try block.

    If the except clause is widened back to ``(NotImplementedError,
    AttributeError)``, the sentinel is caught, the fallback succeeds (returns
    a result object), ``_apply_overrides`` returns normally, and
    ``pytest.raises`` sees ``Failed: DID NOT RAISE`` — the test FAILS.
    Confirmed: see evidence block at end of this function's docstring.

    Evidence of discrimination (widened except in _overrides.py):
        ``except (NotImplementedError, AttributeError):  # widened``
        → test_child_attribute_error_propagates_through_apply_overrides FAILED
          Failed: DID NOT RAISE <class 'AttributeError'>
    """
    from ferrum._overrides import _apply_overrides

    # Build a minimal chart-like whose _rebuild_with_charts is a real callable
    # (so _apply_overrides enters the try block, not the rebuild-is-None branch),
    # but whose .properties() is a simple method that does NOT call
    # _rebuild_with_charts (so the fallback path would succeed independently).
    class _SentinelChart:
        """Chart-like with a real _rebuild_with_charts that raises AttributeError.

        Its .properties() is intentionally independent of _rebuild_with_charts
        so the except-branch fallback would succeed — the sentinel can only
        surface via uncaught propagation from the try block.
        """

        def __init__(self):
            self.width: int | None = None

        def _rebuild_with_charts(self, fn):
            raise AttributeError("SENTINEL_child_bug")

        def properties(self, **kwargs):
            clone = _SentinelChart()
            clone.width = kwargs.get("width", self.width)
            return clone

    chart = _SentinelChart()

    with pytest.raises(AttributeError, match="SENTINEL_child_bug"):
        _apply_overrides(chart, properties={"width": 200})


def test_non_rebuildable_chart_falls_back_via_getattr(df):
    """A chart-like object with no _rebuild_with_charts falls back to .properties().

    The getattr(..., None) path handles this without catching AttributeError.
    The fallback applies child_properties directly on the chart.
    """
    from ferrum._overrides import _apply_overrides

    class _NoRebuildChart:
        """Minimal chart-like that has .properties() but no _rebuild_with_charts."""

        def __init__(self):
            self.width = None

        def properties(self, **kwargs):
            clone = _NoRebuildChart()
            clone.width = kwargs.get("width", self.width)
            return clone

    chart = _NoRebuildChart()
    result = _apply_overrides(chart, properties={"width": 300})
    # Fallback applied properties directly.
    assert result.width == 300


def test_not_implemented_rebuild_falls_back(df):
    """A chart whose _rebuild_with_charts raises NotImplementedError falls back.

    This covers the abstract _ChartLike slot: the method exists on the object
    but raises NotImplementedError.  The except NotImplementedError handler
    applies child_properties directly.
    """
    from ferrum._overrides import _apply_overrides

    class _AbstractishChart:
        """Chart-like with _rebuild_with_charts that raises NotImplementedError."""

        def __init__(self):
            self.width = None

        def _rebuild_with_charts(self, fn):
            raise NotImplementedError("abstract")

        def properties(self, **kwargs):
            clone = _AbstractishChart()
            clone.width = kwargs.get("width", self.width)
            return clone

    chart = _AbstractishChart()
    result = _apply_overrides(chart, properties={"width": 400})
    assert result.width == 400
