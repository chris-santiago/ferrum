"""Batch H task 11 (F-L14-01/F-L14-02/F-L06-04): composite ``to_dict``/
``to_json``/``pipe`` + construction-time child validation.

Before this task, no composition view (``HConcatChart``, ``VConcatChart``,
``ConcatChart``, ``LayerChart``, ``JointChart``, ``RepeatChart``,
``ClusterMapChart``) had ``to_dict()``/``to_json()``/``pipe()`` -- calling
any of them raised ``AttributeError``, despite ``ferrum-spec.md`` §3.16
listing all three as part of the shared output surface. Separately, every
composite constructor accepted any object as a child and only failed much
later, at render time, with ``_lower_any``'s generic
``ValueError: unrecognized node kind '...'`` -- e.g.
``fm.HConcatChart([a_dataframe])`` constructed silently and only broke on
``.to_svg()``.

This module pins both fixes:

- ``to_dict()`` is a *recursive* declaration dict on every composition view
  -- a leaf ``Chart`` child embeds as its ``ChartSpec`` JSON dict
  (round-trippable through ``ChartSpec.from_json``), and a nested composite
  child embeds recursively via its own ``to_dict()`` instead of ``None``.
  ``to_json(*, indent=None)`` is ``json.dumps`` of that dict, mirroring
  ``Chart.to_json``'s indent surface. ``pipe(fn, *args, **kwargs)`` returns
  ``fn(self, *args, **kwargs)``.
- Every composite constructor now raises
  ``TypeError: {Class} children must be Chart or composition objects, got
  {TypeName}`` at construction when a child is neither a ``Chart`` nor
  another composition object; ``Chart.__or__``/``__and__`` and every
  composition view's ``__or__``/``__and__`` return ``NotImplemented`` for a
  non-chart, non-composition operand (mirroring ``Chart.__add__``), so
  ``chart | 3`` raises Python's own operator ``TypeError`` instead of
  reaching the constructor.

``to_spec()`` deliberately stays ``Chart``-only (``ChartSpec`` is a
leaf-only Rust dataclass) -- see the ``ferrum-spec.md`` §3.16 dated note
(2026-09-11, batch H task 11).
"""

from __future__ import annotations

import json

import polars as pl
import pytest

import ferrum as fm
from ferrum._core import ChartSpec
from ferrum.composition import (
    _TITLE_STYLE_FIELDS,
    ClusterMapChart,
    ConcatChart,
    HConcatChart,
    JointChart,
    LayerChart,
    RepeatChart,
    VConcatChart,
)


@pytest.fixture
def df():
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [4.0, 3.0, 2.0, 1.0],
            "g": ["a", "b", "a", "b"],
        }
    )


@pytest.fixture
def c1(df):
    return fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")


@pytest.fixture
def c2(df):
    return fm.Chart(df).mark_line().encode(x="x:Q", y="y:Q")


def _leaf_round_trips(embedded: dict) -> None:
    """Assert *embedded* is a leaf ChartSpec dict that round-trips."""
    assert isinstance(embedded, dict)
    ChartSpec.from_json(json.dumps(embedded))  # raises on a malformed dict


# ---------------------------------------------------------------------------
# to_dict() — shape + leaf round-trip, one section per composition kind
# ---------------------------------------------------------------------------


class TestToDictLeafShapes:
    def test_hconcat_to_dict(self, c1, c2):
        d = (c1 | c2).to_dict()
        assert d["kind"] == "hconcat"
        assert len(d["children"]) == 2
        for child in d["children"]:
            _leaf_round_trips(child)

    def test_vconcat_to_dict(self, c1, c2):
        d = (c1 & c2).to_dict()
        assert d["kind"] == "vconcat"
        assert len(d["children"]) == 2
        for child in d["children"]:
            _leaf_round_trips(child)

    def test_concat_to_dict(self, c1, c2):
        cc = ConcatChart(c1, c2, columns=1)
        d = cc.to_dict()
        assert d["kind"] == "wrap"
        assert d["columns"] == 1
        assert len(d["children"]) == 2
        for child in d["children"]:
            _leaf_round_trips(child)

    def test_layer_to_dict(self, c1, c2):
        lc = LayerChart(c1, c2, title="overlay")
        d = lc.to_dict()
        assert d["kind"] == "overlay"
        assert d["title"] == {"text": "overlay"}
        assert len(d["children"]) == 2
        for child in d["children"]:
            _leaf_round_trips(child)

    def test_layer_to_dict_omits_unset_title(self, c1, c2):
        """Unset chrome is omitted, not emitted as ``None`` -- the same rule
        the other six wrappers follow."""
        assert "title" not in LayerChart(c1, c2).to_dict()

    def test_joint_to_dict_equals_spec(self, c1, c2):
        jc = JointChart(c1, top=c2)
        d = jc.to_dict()
        assert d == jc.spec
        assert d["kind"] == "joint"
        _leaf_round_trips(d["center"])
        _leaf_round_trips(d["top"])
        assert d["right"] is None

    def test_repeat_to_dict_equals_spec(self, df):
        template = fm.Chart(df).encode(x=fm.Repeat.column, y="y").mark_point()
        rc = RepeatChart(template, column=["x", "y"])
        d = rc.to_dict()
        assert d == rc.spec
        assert d["kind"] == "repeat"
        _leaf_round_trips(d["template"])
        assert d["diagonal"] is None

    def test_cluster_map_to_dict_equals_spec(self, c1):
        cm = ClusterMapChart(c1, dendrogram_ratio=0.3)
        d = cm.to_dict()
        assert d == cm.spec
        assert d["kind"] == "cluster_map"
        _leaf_round_trips(d["heatmap"])
        assert d["row_dendrogram"] is None
        assert d["col_dendrogram"] is None


# ---------------------------------------------------------------------------
# to_dict() — figure chrome, uniform across all seven wrappers
#
# Batch-H close remediation (2026-09-11): `to_dict()` dropped
# `_figure_title`/`_figure_subtitle`/`_figure_caption` on six of the seven
# wrappers -- `(a | b).properties(title="My title").to_dict()` had no title
# anywhere -- while `LayerChart` emitted `title`. Same declaration, present in
# one member of the family and absent from six.
# ---------------------------------------------------------------------------


def _titled(node):
    return node.properties(title="My title", subtitle="My subtitle", caption="My caption")


class TestToDictFigureChrome:
    def _composites(self, c1, c2, df):
        template = fm.Chart(df).encode(x=fm.Repeat.column, y="y").mark_point()
        return {
            "hconcat": c1 | c2,
            "vconcat": c1 & c2,
            "concat": ConcatChart(c1, c2, columns=1),
            "joint": JointChart(c1, top=c2),
            "repeat": RepeatChart(template, column=["x", "y"]),
            "cluster_map": ClusterMapChart(c1, dendrogram_ratio=0.3),
        }

    def test_every_composite_carries_title_subtitle_caption(self, c1, c2, df):
        """All six ``_CompositeBase`` wrappers emit the figure chrome set via
        ``.properties()`` under the same key names ``LayerChart`` uses."""
        for name, node in self._composites(c1, c2, df).items():
            d = _titled(node).to_dict()
            assert d["title"] == {"text": "My title"}, f"{name} dropped the figure title"
            assert d["subtitle"] == "My subtitle", f"{name} dropped the figure subtitle"
            assert d["caption"] == "My caption", f"{name} dropped the figure caption"

    def test_layer_chart_carries_its_title_through_properties(self, c1, c2):
        """The seventh wrapper reaches the same key by its own ``_title`` slot."""
        assert LayerChart(c1, c2).properties(title="My title").to_dict()["title"] == {
            "text": "My title"
        }

    def test_untitled_composites_are_unchanged(self, c1, c2, df):
        """Chrome keys are omitted when unset, so a chrome-free composite's
        declaration dict is exactly what it was before the rule existed."""
        for name, node in self._composites(c1, c2, df).items():
            d = node.to_dict()
            assert not ({"title", "subtitle", "caption"} & set(d)), (
                f"{name} must omit unset chrome keys, got {sorted(d)}"
            )

    def test_untitled_spec_delegating_composites_still_equal_spec(self, c1, c2, df):
        """Joint/Repeat/ClusterMap keep ``to_dict() == spec`` when untitled;
        ``spec`` itself stays the pure layout-introspection surface and never
        gains chrome."""
        for name in ("joint", "repeat", "cluster_map"):
            node = self._composites(c1, c2, df)[name]
            assert node.to_dict() == node.spec
            titled = _titled(node)
            assert "title" not in titled.spec, f"{name}.spec must not gain chrome"
            assert titled.to_dict() == {
                **titled.spec,
                "title": {"text": "My title"},
                "subtitle": "My subtitle",
                "caption": "My caption",
            }

    def test_chrome_survives_to_json(self, c1, c2, df):
        """The chrome rides ``to_json`` too -- it is part of the declaration,
        not a ``to_dict``-only decoration -- for **both** title spellings, on
        every one of the six ``_CompositeBase`` wrappers.

        The ``fm.Title(...)`` half is the batch-H close re-review fix: the raw
        ``Title`` used to be passed straight into the declaration dict, so
        ``to_dict()`` returned an object ``to_json()`` (one line below it in
        the same class) could not serialize.

        **Cycle-4 correction (H3-4 gating fix, 2026-09-11):** cycle 3
        flattened a ``Title``'s own ``.subtitle`` to nothing -- accepted the
        declaration, silently dropped the subtitle, no warning. That is the
        exact "accepted and inert" shape this batch refused ``shape``
        conditionals for elsewhere in the same cycle. It is now ROUTED into
        the composite's separate ``subtitle`` slot (the one that genuinely
        renders -- see ``test_composite_chrome_render_matches_declaration``
        below), which is why it surfaces as the top-level ``subtitle`` KEY
        (a sibling of ``title``), not nested inside ``title``'s own dict.
        """
        title_obj = fm.Title("My title", subtitle="Title's own subtitle")
        composites = self._composites(c1, c2, df)
        for name, node in composites.items():
            for spelling, title in (("str", "My title"), ("Title", title_obj)):
                d = json.loads(node.properties(title=title).to_json())
                assert d["title"]["text"] == "My title", f"{name}/{spelling} lost its title"
            titled = node.properties(title=title_obj)
            d = titled.to_dict()
            assert "subtitle" not in d["title"], (
                f"{name}: title's own dict must not nest subtitle -- it routes to the "
                "top-level subtitle key"
            )
            assert d["subtitle"] == "Title's own subtitle", (
                f"{name}: a Title's own subtitle must route to the composite's subtitle "
                "slot, not vanish (H3-4 cycle-4 correction)"
            )
        assert json.loads(_titled(c1 | c2).to_json())["caption"] == "My caption"

    def test_layer_chart_title_subtitle_refused_no_subtitle_slot(self, c1, c2):
        """``LayerChart`` has no separate figure-subtitle slot to route a
        ``Title``'s own subtitle into (unlike the six ``_CompositeBase``
        views) -- it is refused typed rather than silently dropped."""
        with pytest.raises(ValueError, match="subtitle"):
            LayerChart(c1, c2).properties(title=fm.Title("My title", subtitle="lost"))

    def test_layer_chart_constructor_richness_refusal_names_itself(self, c1, c2):
        """``LayerChart(..., title=fm.Title(..., subtitle=...))`` -- the
        constructor kwarg -- is named as itself in the refusal, not as a
        ``.properties()`` call the caller never made (2026-09-11, batch-H
        close cycle 5: threads ``declare`` into ``_refuse_title_richness``,
        matching the idiom the alias-collision refusal already uses)."""
        with pytest.raises(ValueError, match=r"LayerChart\(\.\.\., title=fm\.Title\(\.\.\.\)\)"):
            LayerChart(c1, c2, title=fm.Title("My title", subtitle="lost"))
        # The .properties() spelling is unaffected -- it genuinely IS a
        # properties() call.
        with pytest.raises(ValueError, match=r"properties\(title=fm\.Title\(\.\.\.\)\)"):
            LayerChart(c1, c2).properties(title=fm.Title("My title", subtitle="lost"))

    def test_title_subtitle_collides_with_explicit_subtitle(self, c1, c2):
        """``title=fm.Title(..., subtitle=...)`` together with an explicit
        ``subtitle=`` in the same call is refused typed, naming both
        spellings -- the same collision idiom as the conditional-vocabulary
        refusals, not a silent pick-a-winner."""
        node = c1 | c2
        with pytest.raises(ValueError, match="subtitle"):
            node.properties(title=fm.Title("T", subtitle="from title"), subtitle="from subtitle=")

    #: A non-default value per style field, keyed by name so the
    #: parametrize below sweeps every field `_TITLE_STYLE_FIELDS` names --
    #: not a hand-picked subset (2026-09-11, batch-H close cycle 5: the
    #: pre-cycle-5 version of this test hardcoded 4 of the 7 fields,
    #: omitting `font_weight`/`subtitle_font_size`/`subtitle_color`, which
    #: refuse live but were unpinned).
    _NON_DEFAULT_VALUES = {
        "anchor": "end",
        "offset": 12.0,
        "font_size": 14.0,
        "font_weight": "bold",
        "color": "#123456",
        "subtitle_font_size": 10.0,
        "subtitle_color": "#654321",
    }

    @pytest.mark.parametrize("field", [name for name, _ in _TITLE_STYLE_FIELDS])
    def test_composite_chrome_unsupported_title_field_refuses_typed(self, c1, c2, df, field):
        """A ``Title`` style field with no composite-chrome slot at all
        refuses typed, at ``.properties()`` time, for every chrome key on
        every wrapper -- it never reaches ``to_svg()`` as an unmapped crash
        (cycle 4, S3 'one slot of three': widened past ``title=`` alone).
        Parametrized over ``_TITLE_STYLE_FIELDS`` itself (cycle 5) so a
        `Title` field added there is swept automatically instead of needing
        a matching hand-edit here."""
        rich_title = fm.Title("T", **{field: self._NON_DEFAULT_VALUES[field]})
        composites = {**self._composites(c1, c2, df), "layer": LayerChart(c1, c2)}
        for name, node in composites.items():
            for key in ("title", "subtitle", "caption"):
                with pytest.raises(ValueError, match=field):
                    node.properties(**{key: rich_title})

    def test_composite_chrome_render_matches_declaration(self, c1, c2, df):
        """H3-4 acceptance-parity, widened to all three chrome keys x all
        seven wrappers (cycle 4, S3 'one slot of three'): a text-only
        ``fm.Title(...)`` on ``title=``/``subtitle=``/``caption=`` always
        renders -- ``to_dict()``/``to_json()``/``to_svg()`` all succeed --
        for every wrapper, not just the one the H3-4 finding named."""
        composites = {**self._composites(c1, c2, df), "layer": LayerChart(c1, c2)}
        for name, node in composites.items():
            for key in ("title", "subtitle", "caption"):
                titled = node.properties(**{key: fm.Title(f"{key}-text")})
                titled.to_dict()  # must not raise
                titled.to_json()  # must not raise
                svg = titled.to_svg()  # must not raise -- the H3-4 regression
                assert "<svg" in svg, f"{name}/{key}: to_svg() must actually render"

    def test_chrome_title_shape_matches_embedded_children(self, c1, c2):
        """The top-level ``title`` carries the same shape an embedded child's
        does -- one concept, one shape, at every level of the declaration."""
        titled_leaf = c1.properties(title="My title")
        outer = (titled_leaf | c2).properties(title="My title")
        d = outer.to_dict()
        assert d["title"] == d["children"][0]["title"] == {"text": "My title"}

    def test_nested_composite_chrome_rides_the_recursion(self, c1, c2):
        """A titled child composite keeps its own chrome when embedded."""
        inner = _titled(c1 | c2)
        outer = HConcatChart([inner, c1])
        assert outer.to_dict()["children"][0]["title"] == {"text": "My title"}
        assert "title" not in outer.to_dict()


# ---------------------------------------------------------------------------
# to_dict() — nested-composite recursion (the pre-fix bug: nested -> None)
# ---------------------------------------------------------------------------


class TestToDictRecursion:
    def test_hconcat_of_hconcat_recurses(self, c1, c2):
        inner = c1 | c2
        outer = HConcatChart([inner, c1])
        d = outer.to_dict()
        nested = d["children"][0]
        assert isinstance(nested, dict)
        assert nested["kind"] == "hconcat"
        assert len(nested["children"]) == 2
        _leaf_round_trips(d["children"][1])

    def test_layer_of_layer_recurses(self, c1, c2):
        inner = LayerChart(c1, c2)
        outer = HConcatChart([inner, c1])
        d = outer.to_dict()
        nested = d["children"][0]
        assert nested["kind"] == "overlay"
        assert len(nested["children"]) == 2

    def test_joint_center_as_nested_composite_recurses(self, c1, c2):
        layered_center = LayerChart(c1, c2)
        jc = JointChart(layered_center, top=c2)
        d = jc.to_dict()
        assert isinstance(d["center"], dict)
        assert d["center"]["kind"] == "overlay"

    def test_repeat_diagonal_as_nested_composite_recurses(self, df):
        template = fm.Chart(df).encode(x=fm.Repeat.row, y=fm.Repeat.column).mark_point()
        diag_leaf = fm.Chart(df).mark_line().encode(x="x", y="y")
        diag_layered = LayerChart(diag_leaf, diag_leaf)
        rc = RepeatChart(template, row=["x", "y"], column=["x", "y"], diagonal=diag_layered)
        d = rc.to_dict()
        assert isinstance(d["diagonal"], dict)
        assert d["diagonal"]["kind"] == "overlay"

    def test_cluster_map_row_dendrogram_as_nested_composite_recurses(self, c1, c2):
        dendro = LayerChart(c2, c2)
        cm = ClusterMapChart(c1, row_dendrogram=dendro)
        d = cm.to_dict()
        assert isinstance(d["row_dendrogram"], dict)
        assert d["row_dendrogram"]["kind"] == "overlay"

    def test_deeply_nested_three_levels(self, c1, c2):
        level1 = LayerChart(c1, c2)
        level2 = HConcatChart([level1, c1])
        level3 = VConcatChart([level2, c2])
        d = level3.to_dict()
        assert d["kind"] == "vconcat"
        hconcat_dict = d["children"][0]
        assert hconcat_dict["kind"] == "hconcat"
        overlay_dict = hconcat_dict["children"][0]
        assert overlay_dict["kind"] == "overlay"
        _leaf_round_trips(overlay_dict["children"][0])


# ---------------------------------------------------------------------------
# to_json() — indent parity with Chart.to_json / json.dumps(to_dict())
# ---------------------------------------------------------------------------


class TestToJson:
    def test_hconcat_to_json_compact_matches_to_dict(self, c1, c2):
        hc = c1 | c2
        assert json.loads(hc.to_json()) == hc.to_dict()
        assert "\n" not in hc.to_json()

    def test_hconcat_to_json_indent_matches_json_dumps(self, c1, c2):
        hc = c1 | c2
        assert hc.to_json(indent=2) == json.dumps(hc.to_dict(), indent=2)
        assert "\n" in hc.to_json(indent=2)

    def test_layer_to_json_round_trips(self, c1, c2):
        lc = LayerChart(c1, c2)
        assert json.loads(lc.to_json(indent=4)) == lc.to_dict()

    def test_joint_to_json_matches_to_dict(self, c1, c2):
        jc = JointChart(c1, top=c2)
        assert json.loads(jc.to_json()) == jc.to_dict()


# ---------------------------------------------------------------------------
# pipe() — Chart and every composition view
# ---------------------------------------------------------------------------


class TestPipe:
    def test_chart_pipe(self, c1):
        assert c1.pipe(lambda c: type(c).__name__) == "Chart"

    def test_chart_pipe_forwards_args_kwargs(self, c1):
        def add(chart, a, *, b):
            return a + b

        assert c1.pipe(add, 2, b=3) == 5

    @pytest.mark.parametrize(
        "build",
        [
            lambda c1, c2: c1 | c2,
            lambda c1, c2: c1 & c2,
            lambda c1, c2: ConcatChart(c1, c2),
            lambda c1, c2: LayerChart(c1, c2),
            lambda c1, c2: JointChart(c1, top=c2),
        ],
    )
    def test_composite_pipe(self, c1, c2, build):
        composite = build(c1, c2)
        assert composite.pipe(lambda x: type(x).__name__) == type(composite).__name__

    def test_repeat_pipe(self, df):
        template = fm.Chart(df).encode(x=fm.Repeat.column, y="y").mark_point()
        rc = RepeatChart(template, column=["x", "y"])
        assert rc.pipe(lambda x: x.columns) is None

    def test_cluster_map_pipe(self, c1):
        cm = ClusterMapChart(c1)
        assert cm.pipe(lambda x: x.dendrogram_ratio) == 0.2


# ---------------------------------------------------------------------------
# Construction-time child validation (H17, F-L06-04)
# ---------------------------------------------------------------------------


class TestConstructionValidation:
    def test_hconcat_rejects_non_chart_child(self, df, c1):
        with pytest.raises(TypeError, match="HConcatChart children must be Chart or composition"):
            HConcatChart([c1, df])

    def test_vconcat_rejects_non_chart_child(self, df, c1):
        with pytest.raises(TypeError, match="VConcatChart children must be Chart or composition"):
            VConcatChart([c1, 42])

    def test_concat_rejects_non_chart_child(self, c1):
        with pytest.raises(TypeError, match="ConcatChart children must be Chart or composition"):
            ConcatChart(c1, "not a chart")

    def test_layer_rejects_non_chart_child(self, c1):
        with pytest.raises(TypeError, match="LayerChart children must be Chart or composition"):
            LayerChart(c1, object())

    def test_joint_rejects_non_chart_center(self, df):
        with pytest.raises(TypeError, match="JointChart children must be Chart or composition"):
            JointChart(df)

    def test_joint_rejects_non_chart_top(self, c1):
        with pytest.raises(TypeError, match="JointChart children must be Chart or composition"):
            JointChart(c1, top="nope")

    def test_joint_rejects_non_chart_right(self, c1):
        with pytest.raises(TypeError, match="JointChart children must be Chart or composition"):
            JointChart(c1, right=3.14)

    def test_repeat_rejects_non_chart_template(self, df):
        with pytest.raises(TypeError, match="RepeatChart children must be Chart or composition"):
            RepeatChart(df, column=["x"])

    def test_repeat_rejects_non_chart_diagonal(self, df):
        template = fm.Chart(df).encode(x=fm.Repeat.row, y=fm.Repeat.column).mark_point()
        with pytest.raises(TypeError, match="RepeatChart children must be Chart or composition"):
            RepeatChart(template, row=["x"], column=["y"], diagonal=123)

    def test_cluster_map_rejects_non_chart_heatmap(self, df):
        with pytest.raises(
            TypeError, match="ClusterMapChart children must be Chart or composition"
        ):
            ClusterMapChart(df)

    def test_cluster_map_rejects_non_chart_dendrogram(self, c1):
        with pytest.raises(
            TypeError, match="ClusterMapChart children must be Chart or composition"
        ):
            ClusterMapChart(c1, row_dendrogram="bogus")

    def test_message_names_the_actual_type(self, c1):
        with pytest.raises(TypeError, match="got DataFrame"):
            HConcatChart([c1, pl.DataFrame({"a": [1]})])

    def test_construction_fails_before_render_not_after(self, df):
        """Pins the pre-fix/post-fix distinction: TypeError fires at
        construction, so a downstream to_svg() call is never reached (the
        pre-fix bug was a silent construction followed by a late, vague
        ValueError at render time -- F-L06-04)."""
        with pytest.raises(TypeError):
            HConcatChart([df])

    def test_valid_nested_composite_child_still_constructs(self, c1, c2):
        """A nested composition object (not a leaf Chart) is a valid child
        -- the construction-time check accepts Chart OR composition, it does
        not narrow to leaf-only (that stricter rule, where it applies, stays
        at render time -- e.g. JointChart marginals via _composite_tree)."""
        inner = c1 | c2
        outer = HConcatChart([inner, c1])
        assert isinstance(outer, HConcatChart)
        assert len(outer.charts) == 2

    def test_none_optional_slots_still_construct(self, c1):
        """None is a valid 'absent' value for optional slots, not a type error."""
        jc = JointChart(c1, top=None, right=None)
        assert jc.top is None
        assert jc.right is None
        cm = ClusterMapChart(c1, row_dendrogram=None, col_dendrogram=None)
        assert cm.row_dendrogram is None

    def test_joint_rejects_none_center(self):
        """center is required -- unlike top=/right=, None there is not an
        'absent optional slot', it is exactly as invalid as any other
        non-Chart value."""
        with pytest.raises(TypeError, match="JointChart children must be Chart or composition"):
            JointChart(None)

    def test_repeat_rejects_none_template(self):
        with pytest.raises(TypeError, match="RepeatChart children must be Chart or composition"):
            RepeatChart(None, column=["x"])

    def test_cluster_map_rejects_none_heatmap(self):
        with pytest.raises(
            TypeError, match="ClusterMapChart children must be Chart or composition"
        ):
            ClusterMapChart(None)

    def test_hconcat_rejects_none_in_required_list(self, c1):
        """Unlike JointChart's top=/right=, every entry in HConcatChart's
        charts list is required -- None there is invalid too."""
        with pytest.raises(TypeError, match="HConcatChart children must be Chart or composition"):
            HConcatChart([c1, None])

    def test_layer_rejects_none_in_required_list(self, c1):
        with pytest.raises(TypeError, match="LayerChart children must be Chart or composition"):
            LayerChart(c1, None)


# ---------------------------------------------------------------------------
# __or__ / __and__ — NotImplemented for a non-chart operand (H17)
# ---------------------------------------------------------------------------


class TestOperatorNotImplemented:
    def test_chart_or_non_chart_raises_standard_typeerror(self, c1):
        with pytest.raises(TypeError, match="unsupported operand type"):
            c1 | 3

    def test_chart_and_non_chart_raises_standard_typeerror(self, c1):
        with pytest.raises(TypeError, match="unsupported operand type"):
            c1 & 3

    def test_composite_or_non_chart_raises_standard_typeerror(self, c1, c2):
        hc = c1 | c2
        with pytest.raises(TypeError, match="unsupported operand type"):
            hc | "nope"

    def test_composite_and_non_chart_raises_standard_typeerror(self, c1, c2):
        vc = c1 & c2
        with pytest.raises(TypeError, match="unsupported operand type"):
            vc & None

    def test_or_dunder_returns_not_implemented_directly(self, c1):
        """Calling __or__ directly (bypassing the operator protocol's
        TypeError synthesis) confirms the sentinel itself, not just the
        eventual exception message."""
        assert c1.__or__(3) is NotImplemented

    def test_and_dunder_returns_not_implemented_directly(self, c1):
        assert c1.__and__(3) is NotImplemented

    def test_composite_or_dunder_returns_not_implemented_directly(self, c1, c2):
        hc = c1 | c2
        assert hc.__or__(3) is NotImplemented

    def test_valid_chart_or_chart_still_works(self, c1, c2):
        result = c1 | c2
        assert isinstance(result, HConcatChart)

    def test_valid_chart_or_composite_still_works(self, c1, c2):
        composite = c1 & c2
        result = c1 | composite
        assert isinstance(result, HConcatChart)
        assert len(result.charts) == 2
