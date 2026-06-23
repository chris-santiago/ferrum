"""Cohesion tests for Task R8: shared offset key-set, narrowed except, TypedDict.

Three structural cleanups:

(a) Shared offset key-set — _PANEL_AREA_KEYS / _PANEL_NODE_LIST_KEYS /
    _OUTER_NODE_LIST_KEYS are the single source used by _merge_scene_panels,
    _inject_figure_chrome, and _merge_one_child.  Tests assert the constants
    are referenced by those functions' code and exercise the offset behavior
    for each key type.

(b) Narrowed except — _apply_overrides now uses getattr(..., None) to handle
    missing _rebuild_with_charts (no AttributeError catch at all), and catches
    only NotImplementedError around the rebuild call itself.  An unrelated
    AttributeError raised inside a child's .properties() during _apply now
    propagates rather than being swallowed.  The expected fallback for chart
    types that declare but have not implemented _rebuild_with_charts (the
    abstract _ChartLike slot) is preserved.

(c) TypedDict — _FigureChrome is defined in composition and _figure_chrome_kwargs
    returns an instance of it (its keys are exactly {title, subtitle, caption,
    chrome}).
"""

from __future__ import annotations

import inspect

import polars as pl
import pytest

import ferrum as fm
import json

from ferrum.composition import (
    _EMPTY_SCENE_JSON,
    _OUTER_NODE_LIST_KEYS,
    _PANEL_AREA_KEYS,
    _PANEL_NODE_LIST_KEYS,
    _FigureChrome,
    _empty_scene,
    _inject_figure_chrome,
    _merge_child_scenes,
    _merge_child_scenes_grid,
    _merge_one_child,
    _merge_scene_panels,
)


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
# (a) Shared offset key-set: constant definitions
# ---------------------------------------------------------------------------


def test_panel_area_keys_contains_expected_entries():
    assert "plot_area" in _PANEL_AREA_KEYS
    assert "clip" in _PANEL_AREA_KEYS


def test_panel_node_list_keys_contains_expected_entries():
    assert "axes" in _PANEL_NODE_LIST_KEYS
    assert "grid" in _PANEL_NODE_LIST_KEYS
    assert "annotations" in _PANEL_NODE_LIST_KEYS
    assert "strip_title" in _PANEL_NODE_LIST_KEYS


def test_outer_node_list_keys_contains_expected_entries():
    assert "title" in _OUTER_NODE_LIST_KEYS
    assert "legend" in _OUTER_NODE_LIST_KEYS
    assert "decorations" in _OUTER_NODE_LIST_KEYS


def test_panel_area_keys_referenced_in_merge_scene_panels_source():
    """_merge_scene_panels source must reference the shared constant, not a literal."""
    src = inspect.getsource(_merge_scene_panels)
    assert "_PANEL_AREA_KEYS" in src
    # The old literal tuples must not appear inline.
    assert '"plot_area", "clip"' not in src
    assert "'plot_area', 'clip'" not in src


def test_panel_node_list_keys_referenced_in_merge_scene_panels_source():
    src = inspect.getsource(_merge_scene_panels)
    assert "_PANEL_NODE_LIST_KEYS" in src
    # The old per-key for loops must not be there.
    assert '"axes", "grid"' not in src
    assert '"annotations", "strip_title"' not in src


def test_panel_area_keys_referenced_in_inject_figure_chrome_source():
    src = inspect.getsource(_inject_figure_chrome)
    assert "_PANEL_AREA_KEYS" in src


def test_panel_node_list_keys_referenced_in_inject_figure_chrome_source():
    src = inspect.getsource(_inject_figure_chrome)
    assert "_PANEL_NODE_LIST_KEYS" in src


def test_outer_node_list_keys_referenced_in_inject_figure_chrome_source():
    src = inspect.getsource(_inject_figure_chrome)
    assert "_OUTER_NODE_LIST_KEYS" in src


def test_outer_node_list_keys_referenced_in_merge_one_child_source():
    src = inspect.getsource(_merge_one_child)
    assert "_OUTER_NODE_LIST_KEYS" in src


# ---------------------------------------------------------------------------
# (a) Shared offset key-set: behavioral — each key type IS offset by both paths
# ---------------------------------------------------------------------------


def _make_panel(*, dx: float, dy: float) -> dict:
    """Build a minimal panel dict with one node of each panel key type."""
    return {
        "id": 0,
        "plot_area": {"x": 0.0, "y": 0.0, "width": 100.0, "height": 80.0},
        "clip": {"x": 0.0, "y": 0.0, "width": 100.0, "height": 80.0},
        "marks": [
            {
                "nodes": [{"type": "circle", "cx": 10.0, "cy": 20.0, "r": 3.0}],
            }
        ],
        "axes": [{"type": "line", "x1": 0.0, "y1": 0.0, "x2": 100.0, "y2": 0.0}],
        "grid": [{"type": "line", "x1": 50.0, "y1": 0.0, "x2": 50.0, "y2": 80.0}],
        "annotations": [{"type": "rect", "x": 5.0, "y": 10.0, "width": 10.0, "height": 5.0}],
        "strip_title": [{"type": "text", "x": 50.0, "y": 5.0, "text": "A"}],
    }


def _make_scene(panel: dict) -> dict:
    return {
        "width": 100.0,
        "height": 80.0,
        "panels": [panel],
        "title": [],
        "legend": [],
        "decorations": [],
        "selections": [],
        "interaction": {
            "zoom_enabled": True,
            "pan_enabled": True,
            "conditionals": [],
            "linked_panels": [],
            "tick_levels": [],
            "params": [],
            "param_bindings": [],
        },
        "background": None,
    }


def _empty_merged() -> dict:
    return {
        "width": 0.0,
        "height": 0.0,
        "panels": [],
        "title": [],
        "legend": [],
        "decorations": [],
        "selections": [],
        "interaction": {
            "zoom_enabled": True,
            "pan_enabled": True,
            "conditionals": [],
            "linked_panels": [],
            "tick_levels": [],
            "params": [],
            "param_bindings": [],
        },
        "background": None,
    }


def test_merge_scene_panels_offsets_all_panel_key_types():
    """Every panel node key type is shifted by _merge_scene_panels."""
    panel = _make_panel(dx=30.0, dy=15.0)
    scene = _make_scene(panel)
    merged = _empty_merged()
    _merge_scene_panels(merged, scene, 30.0, 15.0, 0)

    p = merged["panels"][0]
    # Area keys
    assert p["plot_area"]["x"] == pytest.approx(30.0)
    assert p["plot_area"]["y"] == pytest.approx(15.0)
    assert p["clip"]["x"] == pytest.approx(30.0)
    assert p["clip"]["y"] == pytest.approx(15.0)
    # marks nodes
    assert p["marks"][0]["nodes"][0]["cx"] == pytest.approx(40.0)
    assert p["marks"][0]["nodes"][0]["cy"] == pytest.approx(35.0)
    # axes
    assert p["axes"][0]["x1"] == pytest.approx(30.0)
    assert p["axes"][0]["y1"] == pytest.approx(15.0)
    # grid
    assert p["grid"][0]["x1"] == pytest.approx(80.0)
    # annotations
    assert p["annotations"][0]["x"] == pytest.approx(35.0)
    assert p["annotations"][0]["y"] == pytest.approx(25.0)
    # strip_title
    assert p["strip_title"][0]["x"] == pytest.approx(80.0)
    assert p["strip_title"][0]["y"] == pytest.approx(20.0)


def test_inject_figure_chrome_offsets_all_panel_key_types():
    """_inject_figure_chrome shifts panel keys by the header height."""
    panel = _make_panel(dx=0.0, dy=0.0)
    merged = _empty_merged()
    merged["width"] = 100.0
    merged["height"] = 80.0
    merged["panels"] = [panel]

    # Add a title outer-level node to verify outer offsets too.
    merged["title"].append({"type": "text", "x": 10.0, "y": 5.0, "text": "A"})

    _inject_figure_chrome(
        merged,
        title="T",
        subtitle=None,
        caption=None,
        chrome={},
    )

    # The call must have added a header band (header_h > 0 when title is set).
    # We verify that the panel nodes moved down by the same amount as plot_area.
    p = merged["panels"][0]
    header_h = p["plot_area"]["y"]  # was 0, now header_h
    assert header_h > 0, "expected a non-zero header band when title is set"

    # All panel key types shifted by header_h.
    assert p["clip"]["y"] == pytest.approx(header_h)
    assert p["marks"][0]["nodes"][0]["cy"] == pytest.approx(20.0 + header_h)
    assert p["axes"][0]["y1"] == pytest.approx(0.0 + header_h)
    assert p["grid"][0]["y1"] == pytest.approx(0.0 + header_h)
    assert p["annotations"][0]["y"] == pytest.approx(10.0 + header_h)
    assert p["strip_title"][0]["y"] == pytest.approx(5.0 + header_h)

    # Outer title nodes (pre-existing, not chrome) also shifted.
    assert merged["title"][0]["y"] == pytest.approx(5.0 + header_h)


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


# ---------------------------------------------------------------------------
# (c) TypedDict: _FigureChrome is a TypedDict; _figure_chrome_kwargs returns it
# ---------------------------------------------------------------------------


def test_figure_chrome_is_typed_dict():
    """_FigureChrome must be a TypedDict class with the expected keys."""
    import typing

    # TypedDict classes have __annotations__ with the declared keys.
    assert hasattr(_FigureChrome, "__annotations__")
    keys = set(_FigureChrome.__annotations__)
    assert keys == {"title", "subtitle", "caption", "chrome"}


def test_figure_chrome_kwargs_returns_expected_keys(base_chart):
    """_figure_chrome_kwargs must return a dict with exactly the _FigureChrome keys."""
    comp = fm.JointChart(base_chart)
    payload = comp._figure_chrome_kwargs()
    assert set(payload.keys()) == {"title", "subtitle", "caption", "chrome"}
    # Values for an un-titled composite.
    assert payload["title"] is None
    assert payload["subtitle"] is None
    assert payload["caption"] is None
    assert isinstance(payload["chrome"], dict)


def test_figure_chrome_kwargs_carries_title(base_chart):
    """After .properties(title=), _figure_chrome_kwargs reflects the title."""
    comp = fm.JointChart(base_chart).properties(title="MyTitle", subtitle="Sub", caption="Cap")
    payload = comp._figure_chrome_kwargs()
    assert payload["title"] == "MyTitle"
    assert payload["subtitle"] == "Sub"
    assert payload["caption"] == "Cap"


# ---------------------------------------------------------------------------
# COMP-03: _merge_child_scenes* empty-input returns canonical schema
# ---------------------------------------------------------------------------

_FULL_SCENE_KEYS = {
    "panels",
    "width",
    "height",
    "selections",
    "interaction",
    "title",
    "legend",
    "decorations",
    "background",
}


def test_merge_child_scenes_empty_input_returns_full_schema():
    """Regression: COMP-03 — empty-input early-return uses _EMPTY_SCENE_JSON (full schema).

    Before the fix, the four _merge_child_scenes* helpers returned a partial
    literal '{"panels":[],"width":0,"height":0}' on empty input, which carried
    only 3 keys.  After COMP-03 they return _EMPTY_SCENE_JSON, whose parsed
    shape matches _empty_scene() and carries the full key set including
    selections, interaction, legend, decorations, and background.

    This test would KeyError / fail on the old 3-key literal and passes now.
    """
    scene_json, packed = _merge_child_scenes([], spacing=10.0)
    parsed = json.loads(scene_json)
    assert parsed.keys() >= _FULL_SCENE_KEYS
    assert packed == b""


def test_merge_child_scenes_grid_empty_input_returns_full_schema():
    """Regression: COMP-03 — grid variant also uses _EMPTY_SCENE_JSON on empty input."""
    scene_json, packed = _merge_child_scenes_grid([], spacing=10.0, columns=2)
    parsed = json.loads(scene_json)
    assert parsed.keys() >= _FULL_SCENE_KEYS
    assert packed == b""


def test_empty_scene_json_matches_empty_scene_call():
    """_EMPTY_SCENE_JSON is the canonical serialization of _empty_scene()."""
    assert json.loads(_EMPTY_SCENE_JSON) == _empty_scene()
