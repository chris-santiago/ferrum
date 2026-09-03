"""Chart-level axis config takes real effect (batch B, Task 8).

Covers the surfaces spec ``2026-09-02-batch-b-config-plumbing`` §§4.2/4.3/4.9
made honest, each of which was parsed-but-dropped (or wrong-axis) before:

* §4.2 — ``domain_min``/``domain_max``/``nice``/``zero`` shape the positional
  scale domain; an encoding-level ``scale=`` domain wins silently; an ordinal
  axis is refused loudly.
* §4.3 — grid enable AND style are per-axis end to end, through both config
  spellings, with the flat both-axes keys preserved.
* §4.9 — chart-level ``tick_count``/``labels``/``ticks``/``title`` are honored,
  ``label_overlap`` works on y, ``axis_y2``'s per-axis-prep fields reach the
  secondary axis, and ``offset``/``label_flush`` have a typed Python surface.
* §7 — the cascade constraint: a per-axis ``.override(x_axis_…)`` beats a
  general ``configure_axis(...)`` on its own axis, which is what let
  ``_redistribute_general_axis`` retire.

Every assertion here discriminates: it compares a configured render against
the same chart's unconfigured render, or against the *other* axis's render, so
a silently-dropped field fails rather than passing on a bare ``"<svg" in svg``.
"""

from __future__ import annotations

import datetime
import re
import warnings

import polars as pl
import pytest

import ferrum as fm


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [3.0, 11.0, 7.0, 25.0, 14.0],
            "c": [1013.0, 2117.0, 3311.0, 4207.0, 5433.0],
            "g": ["a", "b", "c", "d", "e"],
        }
    )


@pytest.fixture
def chart(df: pl.DataFrame):
    return fm.Chart(df).mark_point().encode(x="a", y="b")


def _texts(svg: str) -> list[str]:
    return re.findall(r">([^<>]+)</text>", svg)


def _grid_line_count(svg: str) -> int:
    """Gridlines in *svg*, identified by the theme's own grid stroke color."""
    return svg.lower().count(_THEME_GRID_COLOR)


#: The default theme's gridline color, used to count gridlines without
#: depending on element ordering.
_THEME_GRID_COLOR = "#d6d3d1"


# ---------------------------------------------------------------------------
# §4.2 — scale-domain config
# ---------------------------------------------------------------------------


class TestScaleDomainConfig:
    def test_domain_max_extends_the_axis_beyond_the_data(self, chart):
        """The y axis must label values the data never reaches."""
        out = chart.configure(axis_y=fm.AxisConfig(domain_max=100.0)).to_svg()
        labels = {t.strip() for t in _texts(out)}
        assert "100" in labels, f"domain_max=100 must extend the y axis; got {sorted(labels)}"
        assert "100" not in {t.strip() for t in _texts(chart.to_svg())}

    def test_domain_min_clamps_the_lower_bound(self, chart):
        out = chart.configure(axis_y=fm.AxisConfig(domain_min=-50.0)).to_svg()
        assert any(t.strip().startswith("-") for t in _texts(out)), (
            "domain_min=-50 must put negative labels on the y axis"
        )

    def test_zero_extends_a_non_zero_domain_to_include_zero(self, df):
        """Data starting at 1000 gets a 0 tick only when zero=True."""
        c = fm.Chart(df).mark_point().encode(x="a", y="c")
        assert "0" not in {t.strip() for t in _texts(c.to_svg())}
        out = c.configure(axis_y=fm.AxisConfig(zero=True)).to_svg()
        assert "0" in {t.strip() for t in _texts(out)}

    def test_nice_rounds_a_ragged_domain(self, df):
        c = fm.Chart(df).mark_point().encode(x="a", y="c")
        assert c.configure(axis_y=fm.AxisConfig(nice=True)).to_svg() != c.to_svg()

    def test_axis_x_and_axis_y_domains_are_independent(self, chart):
        """The x section must not reshape the y scale, or vice versa."""
        x_only = chart.configure(axis_x=fm.AxisConfig(domain_max=99.0)).to_svg()
        y_only = chart.configure(axis_y=fm.AxisConfig(domain_max=99.0)).to_svg()
        assert x_only != y_only
        assert x_only != chart.to_svg()
        assert y_only != chart.to_svg()

    def test_encoding_scale_domain_wins_silently(self, df):
        """Documented cascade: the more specific surface wins, with no warning."""
        scale = fm.LinearScale(domain=[0.0, 50.0])
        pinned = fm.Chart(df).mark_point().encode(x="a", y=fm.Y("b", scale=scale))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            configured = pinned.configure(axis_y=fm.AxisConfig(domain_max=999.0)).to_svg()
        assert configured == pinned.to_svg(), "encoding scale= must win entirely"
        assert not [w for w in caught if "domain_max" in str(w.message)], (
            "losing to a more specific surface is the documented cascade, not a warning"
        )

    def test_ordinal_axis_warns_naming_the_fields(self, df):
        """Wrong surface — loud, unlike the silent cascade loss above."""
        c = fm.Chart(df).mark_bar().encode(x="g", y="b")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            c.configure(axis_x=fm.AxisConfig(domain_min=0.0, nice=True)).to_svg()
        messages = [str(w.message) for w in caught]
        assert any(
            "domain_min" in m and "nice" in m and "ordinal" in m and "x axis" in m for m in messages
        ), messages

    def test_untouched_chart_is_byte_identical(self, chart):
        """A chart exercising none of these fields must not move a byte."""
        assert chart.to_svg() == chart.configure(axis_y=fm.AxisConfig()).to_svg()


# ---------------------------------------------------------------------------
# §4.3 — per-axis grid
# ---------------------------------------------------------------------------


class TestPerAxisGrid:
    def test_disagreeing_x_and_y_are_both_honored(self, chart):
        """Was silently dropped in full: the old guard only wrote on agreement."""
        x_on = chart.configure_grid(x=True, y=False).to_svg()
        y_on = chart.configure_grid(x=False, y=True).to_svg()
        assert x_on != y_on
        assert x_on != chart.to_svg()
        assert y_on != chart.to_svg()

    def test_axis_x_grid_false_leaves_the_y_gridlines_alone(self, chart):
        """The per-axis section must not take the other axis's grid down."""
        x_off = chart.configure(axis_x=fm.AxisConfig(grid=False)).to_svg()
        both_off = chart.configure_axis(grid=False).to_svg()
        assert x_off != both_off, "axis_x=grid False must differ from the shared axis key"
        assert x_off != chart.to_svg()

    def test_both_config_spellings_agree(self, chart):
        """configure_grid(x=False) and configure(axis_x=AxisConfig(grid=False))."""
        via_grid = chart.configure_grid(x=False).to_svg()
        via_axis = chart.configure(axis_x=fm.AxisConfig(grid=False)).to_svg()
        assert via_grid == via_axis

    def test_grid_axis_config_object_styles_only_its_axis(self, chart):
        """The object spelling styles ONE axis, counted against that axis's own
        gridline count rather than inferred from two renders merely differing."""
        red = "#ff0000"
        x_styled = chart.configure_grid(x=fm.GridAxisConfig(color=red)).to_svg()
        y_styled = chart.configure_grid(y=fm.GridAxisConfig(color=red)).to_svg()
        # How many gridlines each axis contributes, measured independently.
        x_count = _grid_line_count(chart.configure_grid(x=True, y=False).to_svg())
        y_count = _grid_line_count(chart.configure_grid(x=False, y=True).to_svg())
        assert x_count and y_count and x_count != y_count, "fixture must distinguish the axes"
        assert x_styled.lower().count(red) == x_count, (
            f"x-styled render must recolor exactly the {x_count} x gridlines"
        )
        assert y_styled.lower().count(red) == y_count, (
            f"y-styled render must recolor exactly the {y_count} y gridlines"
        )

    def test_flat_both_axes_shorthand_still_applies_to_both(self, chart):
        """Back-compat: an axis-unspecified color is the both-axes shorthand.

        Pinned as `x_only + y_only`, not a floor: a regression routing the flat
        shorthand to a single axis still clears any `>= 2` threshold.
        """
        green = "#00ff00"
        both = chart.configure_grid(color=green).to_svg().lower().count(green)
        x_only = (
            chart.configure_grid(x=fm.GridAxisConfig(color=green)).to_svg().lower().count(green)
        )
        y_only = (
            chart.configure_grid(y=fm.GridAxisConfig(color=green)).to_svg().lower().count(green)
        )
        assert x_only and y_only, "each axis alone must recolor some gridlines"
        assert both == x_only + y_only, (
            f"the flat shorthand must reach BOTH axes: {both} != {x_only} + {y_only}"
        )

    def test_bare_bool_spelling_unchanged(self, chart):
        """Every existing caller emits the bool; it must keep meaning enable-only."""
        assert chart.configure_grid(x=True, y=True).to_svg() == chart.to_svg()

    def test_per_channel_grid_false_beats_chart_level_true(self, df):
        """Cascade: per-channel wins (the one pair whose old AND behavior held)."""
        c = fm.Chart(df).mark_point().encode(x=fm.X("a", axis=fm.Axis(grid=False)), y="b")
        assert c.configure_grid(x=True).to_svg() == c.to_svg()


# ---------------------------------------------------------------------------
# §4.9 — residual chart-level fields
# ---------------------------------------------------------------------------


class TestChartLevelResiduals:
    def test_tick_count_thins_the_axis(self, chart):
        few = chart.configure_axis(tick_count=3).to_svg()
        assert few != chart.to_svg()

    def test_per_channel_tick_count_beats_chart_level(self, df):
        c = fm.Chart(df).mark_point().encode(x=fm.X("a", axis=fm.Axis(tick_count=3)), y="b")
        assert c.configure_axis(tick_count=12).to_svg() == c.to_svg()

    def test_label_overlap_applies_on_y(self, chart):
        """Previously x-only: the y axis ignored the policy entirely."""
        out = chart.configure(axis_y=fm.AxisConfig(label_overlap="parity")).to_svg()
        assert out != chart.to_svg()
        assert len(_texts(out)) < len(_texts(chart.to_svg()))

    def test_label_overlap_show_all_on_y_keeps_every_label(self, chart):
        out = chart.configure(axis_y=fm.AxisConfig(label_overlap="true")).to_svg()
        assert len(_texts(out)) == len(_texts(chart.to_svg()))

    def test_labels_false_removes_the_tick_labels(self, chart):
        out = chart.configure(axis_x=fm.AxisConfig(labels=False)).to_svg()
        assert out != chart.to_svg()
        assert len(_texts(out)) < len(_texts(chart.to_svg()))

    def test_ticks_false_removes_the_tick_marks(self, chart):
        assert chart.configure(axis_x=fm.AxisConfig(ticks=False)).to_svg() != chart.to_svg()

    def test_title_sets_the_axis_title(self, chart):
        out = chart.configure(axis_x=fm.AxisConfig(title="Measured A")).to_svg()
        assert "Measured A" in out

    def test_empty_title_suppresses_the_field_name_default(self, chart):
        """`""` removes the title outright — it does not render an empty one."""
        out = chart.configure(axis_x=fm.AxisConfig(title="")).to_svg()
        assert "a" in {t.strip() for t in _texts(chart.to_svg())}
        assert "a" not in {t.strip() for t in _texts(out)}
        # Discriminates suppression from "render the caller's string verbatim":
        # a whitespace-only title is the same suppression, so the two renders
        # are byte-identical. Were the sentinel dropped, `""` and `" "` would
        # emit two different (both pointless) text nodes.
        assert out == chart.configure(axis_x=fm.AxisConfig(title=" ")).to_svg()

    def test_per_channel_title_beats_chart_level(self, df):
        c = fm.Chart(df).mark_point().encode(x=fm.X("a", title="PerChannel"), y="b")
        out = c.configure(axis_x=fm.AxisConfig(title="ChartLevel")).to_svg()
        assert "PerChannel" in out and "ChartLevel" not in out

    def test_chart_level_title_never_resurrects_a_per_channel_suppression(self, df):
        """The tri-state's whole point: title=None is not 'unset'."""
        c = fm.Chart(df).mark_point().encode(x=fm.X("a", title=None), y="b")
        out = c.configure(axis_x=fm.AxisConfig(title="ChartLevel")).to_svg()
        assert "ChartLevel" not in out

    def test_axis_x_title_beats_the_shared_axis_key(self, chart):
        out = chart.configure(
            axis=fm.AxisConfig(title="Shared"), axis_x=fm.AxisConfig(title="XOnly")
        ).to_svg()
        assert "XOnly" in out
        assert "Shared" in out, "the shared key still titles the y axis"


class TestTypedPythonSurface:
    """T2 residual: Rust-honored fields that had no typed Python kwarg."""

    @pytest.mark.parametrize(
        ("kwargs", "label"),
        [({"offset": 12.0}, "offset"), ({"label_flush": True}, "label_flush")],
    )
    def test_field_is_reachable_and_takes_effect(self, chart, kwargs, label):
        assert chart.configure_axis(**kwargs).to_svg() != chart.to_svg(), label

    def test_fields_round_trip_through_axis_config(self):
        cfg = fm.AxisConfig(offset=12.0, label_flush=True, labels=False, ticks=False, title="T")
        d = cfg.to_dict()
        assert d["offset"] == 12.0
        assert d["label_flush"] is True
        assert d["labels"] is False
        assert d["ticks"] is False
        assert d["title"] == "T"


# ---------------------------------------------------------------------------
# §4.9 — axis_y2's per-axis-prep fields
# ---------------------------------------------------------------------------


@pytest.fixture
def dual_axis(df: pl.DataFrame):
    return fm.layer(
        fm.Chart(df).mark_line().encode(x="a", y="b"),
        fm.Chart(df).mark_point().encode(x="a", y="c"),
        resolve={"y": "independent"},
    )


class TestAxisY2:
    @pytest.mark.parametrize(
        ("field", "cfg"),
        [
            ("values", fm.AxisConfig(tick_values=[1000.0, 3000.0, 5000.0])),
            ("label_format", fm.AxisConfig(label_format_raw=",.1f")),
            ("tick_extra", fm.AxisConfig(tick_extra=True)),
            ("tick_min_step", fm.AxisConfig(tick_min_step=2000.0)),
            ("tick_size", fm.AxisConfig(tick_size=14.0)),
            ("domain", fm.AxisConfig(domain=False)),
            ("labels", fm.AxisConfig(labels=False)),
        ],
    )
    def test_field_reaches_the_secondary_axis(self, dual_axis, field, cfg):
        assert dual_axis.configure(axis_y2=cfg).to_svg() != dual_axis.to_svg(), field

    @pytest.mark.parametrize(
        ("field", "cfg"),
        [
            ("domain_max", fm.AxisConfig(domain_max=99999.0)),
            ("domain_min", fm.AxisConfig(domain_min=-5000.0)),
            ("zero", fm.AxisConfig(zero=True)),
            ("nice", fm.AxisConfig(nice=True)),
        ],
    )
    def test_scale_domain_config_reaches_the_secondary_axis(self, dual_axis, field, cfg):
        """§4.2 at the axis_y2 position — silently dropped before.

        The secondary axis resolves its own y scale per independent-y layer, so
        the primary pair's reshaping (over ``provisional_scales.x/.y``) never
        touched it: the config parsed, reached nothing, and warned nothing.
        """
        assert dual_axis.configure(axis_y2=cfg).to_svg() != dual_axis.to_svg(), field

    def test_domain_max_visibly_rescales_the_secondary_axis(self, dual_axis):
        """Discriminating: the secondary axis must LABEL the new bound."""
        out = dual_axis.configure(axis_y2=fm.AxisConfig(domain_max=99999.0)).to_svg()
        labels = {t.strip() for t in _texts(out)}
        assert "90000" in labels, f"axis_y2 domain_max must rescale the right axis; got {labels}"
        assert "90000" not in {t.strip() for t in _texts(dual_axis.to_svg())}

    def test_zero_extends_the_secondary_axis_to_the_origin(self, dual_axis):
        out = dual_axis.configure(axis_y2=fm.AxisConfig(zero=True)).to_svg()
        assert "0" in {t.strip() for t in _texts(out)}
        assert "0" not in {t.strip() for t in _texts(dual_axis.to_svg())}

    def test_secondary_layer_marks_move_with_the_configured_domain(self, dual_axis):
        """The tick labels and the marks must describe ONE domain.

        Ticks are shaped in ``prepare``; the per-panel mark scale is resolved
        again from scratch in ``scene_build``. Only asserting on labels would
        leave that second resolution untested, and a chart whose right axis
        reads to 90000 while its points still sit on the 1013--5433 domain is
        exactly the silent divergence the second application exists to
        prevent.
        """
        circles = re.compile(r'<circle[^>]*\scy="([-\d.]+)"')
        baseline = circles.findall(dual_axis.to_svg())
        widened = circles.findall(
            dual_axis.configure(axis_y2=fm.AxisConfig(domain_max=99999.0)).to_svg()
        )
        assert len(baseline) == len(widened) == 5, "fixture must draw 5 secondary-layer points"
        assert baseline != widened, "marks must be re-placed onto the configured domain"
        # Widening the domain ~18x compresses the points into the bottom of the
        # panel: their total vertical spread must collapse.
        spread = lambda ys: max(map(float, ys)) - min(map(float, ys))  # noqa: E731
        assert spread(widened) < spread(baseline) / 10.0, (
            f"points should compress toward the axis floor; "
            f"spread {spread(baseline):.1f} -> {spread(widened):.1f}"
        )

    def test_scale_domain_config_targets_the_axis_it_names(self, dual_axis):
        """`axis_y2` moves the right axis; `axis_y` moves the left one."""
        baseline_primary = {t.strip() for t in _texts(dual_axis.to_svg())} & {"24", "22", "20"}
        assert baseline_primary, "fixture must put those ticks on the primary y axis"

        y2_only = {
            t.strip()
            for t in _texts(dual_axis.configure(axis_y2=fm.AxisConfig(domain_max=99999.0)).to_svg())
        }
        assert "90000" in y2_only, "the secondary axis takes the new bound"
        assert baseline_primary <= y2_only, "the primary y axis must be untouched"

        y_only = {
            t.strip()
            for t in _texts(dual_axis.configure(axis_y=fm.AxisConfig(domain_max=99999.0)).to_svg())
        }
        assert not (baseline_primary & y_only), "the primary y axis rescaled instead"
        assert "5000" in y_only, "the secondary axis must be untouched"

    def test_ordinal_secondary_axis_warns_naming_y2(self, df):
        """Wrong surface is loud at this position too, and names it `y2`."""
        dual_ordinal = fm.layer(
            fm.Chart(df).mark_line().encode(x="a", y="b"),
            fm.Chart(df).mark_point().encode(x="a", y="g"),
            resolve={"y": "independent"},
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dual_ordinal.configure(axis_y2=fm.AxisConfig(domain_max=9.0)).to_svg()
        assert any(
            "domain_max" in str(w.message) and "y2 axis is ordinal" in str(w.message)
            for w in caught
        ), [str(w.message) for w in caught]

    def test_axis_y2_does_not_disturb_the_primary_axes(self, dual_axis):
        """Scope: the secondary-only override must not leak onto x or y."""
        y2_only = dual_axis.configure(axis_y2=fm.AxisConfig(tick_size=14.0)).to_svg()
        shared = dual_axis.configure_axis(tick_size=14.0).to_svg()
        assert y2_only != shared

    def test_named_but_absent_surface_warns(self, chart):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            chart.configure(axis_y2=fm.AxisConfig(tick_size=9.0)).to_svg()
        assert any("axis_y2" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# §7 — the cascade constraint `_redistribute_general_axis` used to fake
# ---------------------------------------------------------------------------


class TestOverrideBeatsGeneralConfigurePerAxis:
    """``_redistribute_general_axis`` retired with the Rust fix it compensated.

    The helper rewrote the merged config — dropping a leaf from the general
    ``axis`` key and re-pinning it onto the *opposite* axis — to stop the
    general value pre-empting a per-axis override. For every field whose only
    carrier was the shared theme that rewrite inverted the result: the
    synthesized opposite-axis entry ran last into the one global slot, so the
    general value won. ``tick_size`` was the last such field; it now has a
    per-axis slot, and the per-axis config sections no longer write the theme
    at all.
    """

    def test_x_axis_tick_size_override_beats_configure_axis(self, chart):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            overridden = (
                chart.configure_axis(tick_size=12.0).override(x_axis_tick_size=2.0).to_svg()
            )
            general_only = chart.configure_axis(tick_size=12.0).to_svg()
        assert overridden != general_only, (
            "a per-axis override must beat the general configure on its own axis"
        )

    def test_the_other_axis_keeps_the_general_value(self, chart):
        """The override must not silently strip the general value from y."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            overridden = (
                chart.configure_axis(tick_size=12.0).override(x_axis_tick_size=2.0).to_svg()
            )
            x_only = chart.configure(axis_x=fm.AxisConfig(tick_size=2.0)).to_svg()
        assert overridden != x_only, "y must still carry the general tick_size=12"

    def test_construction_order_does_not_matter(self, chart):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            configure_first = (
                chart.configure_axis(tick_size=12.0).override(x_axis_tick_size=2.0).to_svg()
            )
            override_first = (
                chart.override(x_axis_tick_size=2.0).configure_axis(tick_size=12.0).to_svg()
            )
        assert configure_first == override_first


class TestScaleDomainCascadeGate:
    """The cascade gate tests the encoding scale's DOMAIN, not its presence.

    A `scale=` that pins no extent — `LinearScale(clamp=True)`, a `Quantile`
    whose domain is a binning artifact — has no domain for the config to lose
    to, so the config must still apply. Testing `scale.is_some()` instead made
    all four fields silently inert for a common encoding shape.
    """

    def test_scale_without_a_domain_does_not_void_the_config(self, df):
        base = (
            fm.Chart(df).mark_point().encode(x="a", y=fm.Y("b", scale=fm.LinearScale(clamp=True)))
        )
        out = base.configure(axis_y=fm.AxisConfig(domain_max=200.0)).to_svg()
        labels = {t.strip() for t in _texts(out)}
        assert "200" in labels, (
            f"a scale= with no domain pins nothing, so domain_max must apply; got {sorted(labels)}"
        )
        assert out != base.to_svg()

    def test_scale_with_a_domain_still_wins_silently(self, df):
        pinned = (
            fm.Chart(df)
            .mark_point()
            .encode(x="a", y=fm.Y("b", scale=fm.LinearScale(domain=[0.0, 50.0])))
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            configured = pinned.configure(axis_y=fm.AxisConfig(domain_max=999.0)).to_svg()
        assert configured == pinned.to_svg()
        assert not [w for w in caught if "domain_max" in str(w.message)]


class TestScaleDomainRefusals:
    """Typed refusals, matching the scale constructors' own contract."""

    def test_degenerate_domain_refused_with_the_sibling_sentence(self):
        """`LinearScale(domain=[10, 10])`'s words, at the config surface."""
        with pytest.raises(ValueError, match=r"domain endpoints must differ \(lo != hi\)"):
            fm.AxisConfig(domain_min=10.0, domain_max=10.0)
        with pytest.raises(ValueError, match=r"domain endpoints must differ \(lo != hi\)"):
            fm.LinearScale(domain=[10.0, 10.0])

    def test_reversed_domain_is_accepted_like_the_sibling(self):
        """`min > max` is a reversed axis, not degenerate — both surfaces agree."""
        assert fm.AxisConfig(domain_min=50.0, domain_max=0.0).to_dict()["domain_min"] == 50.0
        fm.LinearScale(domain=[50.0, 0.0])

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_bounds_refused_by_the_contract_not_the_serializer(self, bad):
        """Was `chart_config: expected value at line 1 column 27` from json."""
        with pytest.raises(ValueError, match="must be a finite number"):
            fm.AxisConfig(domain_min=bad)
        with pytest.raises(ValueError, match="must be a finite number"):
            fm.AxisConfig(domain_max=bad)

    @pytest.mark.parametrize(
        "bad",
        ["5", datetime.datetime(2020, 1, 1), [1.0], True],
        ids=["str", "datetime", "list", "bool"],
    )
    def test_non_numeric_bounds_refused_in_the_ferrum_voice(self, bad):
        """Was a bare `TypeError: must be real number, not str` out of
        `math.isfinite` — no `AxisConfig:` prefix, no field name. All three bad
        shapes on this refusal now read alike. `bool` is refused too: it is a
        `Real` subclass, and `domain_min=True` is a mistake, not a bound of 1.
        """
        with pytest.raises(ValueError, match=r"AxisConfig: domain_min=.* must be a finite number"):
            fm.AxisConfig(domain_min=bad)
        with pytest.raises(ValueError, match=r"AxisConfig: domain_max=.* must be a finite number"):
            fm.AxisConfig(domain_max=bad)

    def test_degenerate_domain_refused_on_the_bypass_path_too(self, chart):
        """`.override(...)` writes leaves straight onto the wire, bypassing
        `AxisConfig.__post_init__`, so Rust backstops it with the same words."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            bypass = chart.override(y_axis_domain_min=10.0, y_axis_domain_max=10.0)
            with pytest.raises(
                ValueError, match=r"domain endpoints must differ \(lo != hi\)"
            ) as excinfo:
                bypass.to_svg()
        assert "y axis" in str(excinfo.value), "the refusal must name the axis"


class TestLogScaleDomainConfig:
    """A config-written domain is a USER-set domain, so it meets the user-set
    contract each scale kind enforces at construction — not the auto-inferred
    fallback.

    `LogScaleData::sanitize_domain` is the AUTO-domain fallback (a `nice`-
    extended bin edge can legitimately land on 0, GH #49). Routing a domain the
    caller actually wrote through it silently floored an explicit `0` to
    `hi / 1e6` and let a sign-crossing pair render a blank axis — where
    `fm.LogScale(domain=...)` refuses both by name. The refusal now asks the
    SCALE what it would have rejected, so the two surfaces share one vocabulary.
    """

    @pytest.fixture
    def log_chart(self, df: pl.DataFrame):
        return fm.Chart(df).mark_point().encode(x="a", y=fm.Y("c", scale=fm.LogScale(clamp=True)))

    def test_valid_config_domain_applies_to_a_log_axis(self, log_chart):
        out = log_chart.configure(axis_y=fm.AxisConfig(domain_max=100000.0)).to_svg()
        assert out != log_chart.to_svg()

    @pytest.mark.parametrize(
        ("cfg", "expected"),
        [
            (fm.AxisConfig(domain_min=0.0), "log scale domain must not contain 0"),
            (fm.AxisConfig(zero=True), "log scale domain must not contain 0"),
            (
                fm.AxisConfig(domain_min=-500.0),
                "log scale domain endpoints must have the same sign",
            ),
        ],
        ids=["domain_min_0", "zero_true", "domain_min_negative"],
    )
    def test_kind_specific_bounds_are_refused_not_silently_floored(self, log_chart, cfg, expected):
        with pytest.raises(ValueError, match=re.escape(expected)):
            log_chart.configure(axis_y=cfg).to_svg()

    @pytest.mark.parametrize(
        ("domain", "expected"),
        [
            ([0.0, 5433.0], "log scale domain must not contain 0"),
            ([-500.0, 5433.0], "log scale domain endpoints must have the same sign"),
        ],
        ids=["contains_zero", "sign_crossing"],
    )
    def test_the_sibling_constructor_refuses_in_the_same_words(self, domain, expected):
        """Same sentence from `fm.LogScale(domain=...)` — one vocabulary."""
        with pytest.raises(ValueError, match=re.escape(expected)):
            fm.LogScale(domain=domain)

    def test_the_refusal_names_the_axis(self, log_chart):
        with pytest.raises(ValueError, match="y axis scale-domain config"):
            log_chart.configure(axis_y=fm.AxisConfig(domain_min=0.0)).to_svg()

    def test_non_log_kinds_are_unconstrained_beyond_the_shared_rules(self, df):
        """Linear/Time/Pow/Symlog constructors add no kind-specific domain rule,
        so a 0 or negative bound is legal there and must still apply."""
        for scale in (fm.LinearScale(clamp=True), fm.SymlogScale(), fm.PowScale(exponent=1.0)):
            c = fm.Chart(df).mark_point().encode(x="a", y=fm.Y("c", scale=scale))
            out = c.configure(axis_y=fm.AxisConfig(domain_min=-500.0)).to_svg()
            assert out.startswith("<svg"), type(scale).__name__


class TestOneDirectionalFlags:
    """`zero=False` / `nice=False` request the default; nothing forces either.

    Pinned as byte-identical no-ops so the manifest's one-directional
    qualification is enforced rather than merely asserted.
    """

    @pytest.mark.parametrize("cfg", [fm.AxisConfig(zero=False), fm.AxisConfig(nice=False)])
    def test_false_spelling_is_a_byte_identical_no_op(self, chart, cfg):
        assert chart.configure(axis_y=cfg).to_svg() == chart.to_svg()

    def test_zero_false_does_not_erase_a_zero_the_data_reaches(self, df):
        """A stacked bar's y domain reaches 0 because the VALUES do."""
        bar = fm.Chart(df).mark_bar().encode(x="g", y="b")
        assert "0" in {t.strip() for t in _texts(bar.to_svg())}
        out = bar.configure(axis_y=fm.AxisConfig(zero=False)).to_svg()
        assert "0" in {t.strip() for t in _texts(out)}


class TestGridConfigShapeRefusal:
    """The widened per-axis slot refuses a wrong shape at the Python boundary."""

    @pytest.mark.parametrize("bad", ["nonsense", 3, 3.5, ["a"]])
    def test_wrong_shape_names_the_axis_and_the_accepted_spellings(self, bad):
        with pytest.raises(ValueError, match="not a valid grid setting"):
            fm.GridConfig(x=bad)
        with pytest.raises(ValueError, match="not a valid grid setting"):
            fm.GridConfig(y=bad)

    @pytest.mark.parametrize(
        "good",
        [True, False, fm.GridAxisConfig(enabled=True), {"enabled": True, "color": "#eee"}],
        ids=["bool_true", "bool_false", "value_class", "raw_mapping"],
    )
    def test_every_accepted_spelling_constructs(self, good):
        """The raw mapping is accepted too — it is the spelling
        `.override(grid_x={...})` uses, and the wire gate validates its keys."""
        assert fm.GridConfig(x=good).to_dict() is not None


class TestOverrideValueClassLeaves:
    """An override leaf accepts whatever its advertised typed equivalent does.

    The registry names `.configure_grid(x=...)` as `grid_x`'s equivalent, and
    that method now accepts `GridAxisConfig` — so the override spelling of the
    same request must too. It previously raised a bare
    `TypeError: Object of type GridAxisConfig is not JSON serializable` from
    inside `json.dumps`, naming neither the key nor an accepted spelling.
    """

    def test_dataclass_leaf_matches_the_typed_equivalent(self, chart):
        cfg = fm.GridAxisConfig(enabled=True, color="#ff0000")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            overridden = chart.override(grid_x=cfg).to_svg()
        assert overridden == chart.configure_grid(x=cfg).to_svg()
        assert overridden.lower().count("#ff0000") == _grid_line_count(
            chart.configure_grid(x=True, y=False).to_svg()
        )

    def test_dict_and_dataclass_spellings_agree(self, chart):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            as_dict = chart.override(grid_x={"enabled": True, "color": "#ff0000"}).to_svg()
            as_obj = chart.override(
                grid_x=fm.GridAxisConfig(enabled=True, color="#ff0000")
            ).to_svg()
        assert as_dict == as_obj


class TestTitleContractIsOneContract:
    """`AxisConfig(title=...)` means what `fm.Axis(title=...)` means.

    Same kwarg name, two levels of one cascade — so `None` cannot suppress on
    one surface and silently no-op on the other. Both route through
    `_title_sentinel.serialize_title`.
    """

    def test_none_suppresses_on_both_levels(self, chart, df):
        per_channel = fm.Chart(df).mark_point().encode(x=fm.X("a", title=None), y="b").to_svg()
        chart_level = chart.configure(axis_x=fm.AxisConfig(title=None)).to_svg()
        assert "a" not in {t.strip() for t in _texts(chart_level)}
        assert "a" not in {t.strip() for t in _texts(per_channel)}

    def test_none_and_empty_string_agree(self, chart):
        assert (
            chart.configure(axis_x=fm.AxisConfig(title=None)).to_svg()
            == chart.configure(axis_x=fm.AxisConfig(title="")).to_svg()
        )

    def test_omitted_keeps_the_field_name_default(self, chart):
        assert chart.configure(axis_x=fm.AxisConfig()).to_svg() == chart.to_svg()
        assert "a" in {t.strip() for t in _texts(chart.to_svg())}

    def test_to_dict_emits_the_same_wire_token_as_the_per_channel_surface(self):
        assert "title" not in fm.AxisConfig().to_dict()
        assert fm.AxisConfig(title=None).to_dict()["title"] == ""
        assert fm.Axis(title=None).to_dict()["title"] == ""
        assert fm.AxisConfig(title="X").to_dict()["title"] == "X"


class TestOverrideMatchesItsTypedEquivalent:
    """One validation authority per leaf, closing the class rather than symptoms.

    Every chart-config override leaf routes through the dataclass it was
    derived from (`_override_apply._leaf_wire_fragment`), so it accepts what
    that dataclass accepts, refuses what it refuses, and serializes the same
    way. Three separate leaks closed by one seam — each pinned below by its own
    verbatim repro — plus a parity sweep so a FUTURE validator added to any
    config class is covered without anyone remembering this file.
    """

    @staticmethod
    def _override(chart, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return chart.override(**kwargs).to_svg()

    def test_invalid_grid_shape_refuses_in_the_ferrum_voice(self, chart):
        """Was: `data did not match any variant of untagged enum Wire`."""
        with pytest.raises(ValueError, match="not a valid grid setting") as excinfo:
            self._override(chart, grid_x="nonsense")
        assert "untagged enum" not in str(excinfo.value)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_non_finite_domain_bound_refuses_in_the_ferrum_voice(self, chart, bad):
        """Was: `chart_config: expected value at line 1 column 27`.

        Unreachable by the Rust backstop by construction — `json.dumps` emits a
        bare `NaN` token and serde fails before Rust's gate runs — so the
        Python boundary is the only place this can be caught.
        """
        with pytest.raises(ValueError, match="must be a finite number") as excinfo:
            self._override(chart, x_axis_domain_min=bad)
        assert "expected value at line" not in str(excinfo.value)

    def test_title_none_suppresses_here_too(self, chart):
        """Was a no-op: raw `None` reached the wire as JSON null, which Rust
        reads as absent, while `configure_axis(title=None)` suppresses."""
        out = self._override(chart, axis_title=None)
        assert "a" not in {t.strip() for t in _texts(out)}
        assert out == chart.configure_axis(title=None).to_svg()

    @pytest.mark.parametrize(
        ("override_kwargs", "configure"),
        [
            ({"axis_label_angle": -45.0}, lambda c: c.configure_axis(label_angle=-45.0)),
            ({"axis_tick_count": 3}, lambda c: c.configure_axis(tick_count=3)),
            ({"axis_labels": False}, lambda c: c.configure_axis(labels=False)),
            ({"axis_title": "Custom"}, lambda c: c.configure_axis(title="Custom")),
            ({"axis_title": None}, lambda c: c.configure_axis(title=None)),
            ({"axis_domain_max": 100.0}, lambda c: c.configure_axis(domain_max=100.0)),
            ({"axis_offset": 8.0}, lambda c: c.configure_axis(offset=8.0)),
            ({"grid_color": "#00ff00"}, lambda c: c.configure_grid(color="#00ff00")),
            ({"grid_x": False}, lambda c: c.configure_grid(x=False)),
            (
                {"grid_x": fm.GridAxisConfig(enabled=True, color="#ff0000")},
                lambda c: c.configure_grid(x=fm.GridAxisConfig(enabled=True, color="#ff0000")),
            ),
            ({"padding_left": 70.0}, lambda c: c.configure_padding(left=70.0)),
            ({"title_font_size": 22.0}, lambda c: c.configure_title(font_size=22.0)),
        ],
        ids=lambda v: next(iter(v)) if isinstance(v, dict) else "",
    )
    def test_override_renders_identically_to_its_typed_equivalent(
        self, chart, override_kwargs, configure
    ):
        """The registry advertises the typed method as each leaf's equivalent;
        this asserts the advertisement is true, byte for byte."""
        assert self._override(chart, **override_kwargs) == configure(chart).to_svg()

    @pytest.mark.parametrize(
        "order",
        [("padding_auto", "padding_left"), ("padding_left", "padding_auto")],
        ids=["auto_first", "left_first"],
    )
    def test_a_leaf_fragment_carries_no_sibling_defaults(self, order):
        """`to_dict()` emits a class's non-None DEFAULTS too, so a one-leaf
        instance carries keys the caller never named.

        `PaddingConfig` is the one config class with such a default
        (`auto: bool = True`), so `PaddingConfig(left=70).to_dict()` is
        `{"auto": True, "left": 70.0}`. Merging that whole fragment injected an
        unsolicited `auto` AND let it overwrite an explicit sibling leaf in
        kwarg order — the two spellings below disagreed with each other and
        with the typed equivalent. Latent only because `PaddingConfigSpec.auto`
        has no Rust consumer yet (Task 9 gives it one), which is exactly when a
        payload-level divergence becomes a rendering one.
        """
        from ferrum._override_apply import build_payload

        values = {"padding_auto": False, "padding_left": 70.0}
        payload = build_payload({k: values[k] for k in order})
        assert payload.chart_config["padding"] == fm.PaddingConfig(left=70.0, auto=False).to_dict()

    def test_a_single_leaf_fragment_names_only_that_leaf(self):
        """The property, not the sampled case: for every config class, a
        one-leaf override fragment contains exactly the keys that leaf
        produces — never a key a default-constructed instance would emit."""
        from ferrum._override_apply import build_payload

        cases = [
            ("padding_left", 70.0, "padding"),
            ("axis_label_angle", -45.0, "axis"),
            ("grid_color", "#00ff00", "grid"),
            ("title_font_size", 22.0, "title"),
            ("color_scheme", "viridis", "color"),
            ("legend_columns", 2, "legend"),
        ]
        for path, value, section in cases:
            fragment = build_payload({path: value}).chart_config[section]
            leaf = path.split("_", 1)[1] if section != "axis" else path[len("axis_") :]
            assert leaf in fragment, (path, fragment)
            assert set(fragment) <= {leaf, "label_format_type"}, (
                f"{path} leaked a sibling default: {fragment}"
            )

    def test_a_preset_leaf_serializes_its_companion_key_too(self, chart):
        """`label_format` emits `label_format_type` alongside itself, so the
        seam must merge the whole fragment rather than take `[leaf]` — a time
        preset that lost its type would be misparsed as a numeric d3 spec."""
        assert (
            self._override(chart, axis_label_format="date_iso")
            == chart.configure_axis(label_format="date_iso").to_svg()
        )


class TestGridStyleValueRefusals:
    """Both grid classes validate the numeric style pair through one body.

    `GridConfig` gained shape validation last cycle while its `width`/`opacity`
    kept passing silently, and `GridAxisConfig` had neither — leaving the
    family uneven in a way it had not been before. `width` takes the spec §4.7
    pixel contract (the same `validate_pixel_value` `PaddingConfig` uses);
    `opacity` takes its numeric/finite halves plus a `[0, 1]` bound.
    """

    @pytest.mark.parametrize("cls", [fm.GridConfig, fm.GridAxisConfig])
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"width": -4.0}, "must be non-negative"),
            ({"width": "x"}, "must be a numeric pixel value"),
            ({"width": float("nan")}, "finite"),
            ({"opacity": 9.0}, "must be between 0 and 1"),
            ({"opacity": -0.5}, "must be between 0 and 1"),
            ({"opacity": "x"}, "must be a number between 0 and 1"),
        ],
    )
    def test_bad_style_values_refuse_on_both_classes(self, cls, kwargs, match):
        with pytest.raises(ValueError, match=match):
            cls(**kwargs)

    @pytest.mark.parametrize("cls", [fm.GridConfig, fm.GridAxisConfig])
    def test_valid_style_values_construct(self, cls):
        assert cls(width=2.0, opacity=0.5).to_dict()

    @pytest.mark.parametrize(
        ("mapping", "match"),
        [
            ({"width": -5.0}, "must be non-negative"),
            ({"width": float("nan")}, "must be a finite numeric pixel value"),
            ({"opacity": 9.0}, "must be between 0 and 1"),
        ],
    )
    def test_the_mapping_spelling_meets_the_same_value_refusals(self, mapping, match):
        """Validation is a property of the SLOT, not of one spelling of it.

        The mapping spelling was a bypass of the refusals the object spelling
        gained in the same change — `{"width": nan}` landed back on the json
        serializer artifact this batch set out to eliminate. `__post_init__`
        now normalizes a mapping through `GridAxisConfig`, so one authority
        validates every spelling.
        """
        with pytest.raises(ValueError, match=match) as excinfo:
            fm.GridConfig(x=mapping)
        assert "expected value at line" not in str(excinfo.value)
        with pytest.raises(ValueError, match=match):
            fm.GridConfig(y=mapping)

    def test_an_unknown_mapping_key_refuses_by_name(self):
        """Falls out of the dataclass signature; restated in the ferrum voice."""
        with pytest.raises(ValueError, match="unknown grid key") as excinfo:
            fm.GridConfig(x={"enabled": True, "bogus": 1})
        assert "accepted:" in str(excinfo.value)

    def test_a_non_dict_mapping_serializes(self, chart):
        """`MappingProxyType` used to reach `json.dumps` and raise
        `Object of type mappingproxy is not JSON serializable`; normalization
        means exactly one nested type ever reaches `to_dict`."""
        from types import MappingProxyType

        proxy = MappingProxyType({"enabled": True, "color": "#ff0000"})
        assert (
            chart.configure_grid(x=proxy).to_svg()
            == chart.configure_grid(x=fm.GridAxisConfig(enabled=True, color="#ff0000")).to_svg()
        )

    @pytest.mark.parametrize("cls", [fm.GridConfig, fm.GridAxisConfig])
    def test_color_is_deliberately_not_value_validated(self, cls):
        """Config-surface color-VALUE refusal is #107's separate decision and
        an explicit non-goal of this batch's spec (§3), whose gate is KEYS.
        An unparseable color keeps the theme value."""
        assert cls(color="notacolor").to_dict()["color"] == "notacolor"
