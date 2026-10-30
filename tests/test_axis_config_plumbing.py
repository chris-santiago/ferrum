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
import json
import re
import warnings
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import polars as pl
import pytest

import ferrum as fm
from tests._snapshots import legend_texts as _texts


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
        `math.isfinite`, naming neither the door nor the field. All three bad
        shapes on this refusal now read alike. `bool` is refused too: it is a
        `Real` subclass, and `domain_min=True` is a mistake, not a bound of 1.
        """
        with pytest.raises(
            ValueError, match=r"^AxisConfig\(domain_min=\.\.\.\) must be a real number "
        ):
            fm.AxisConfig(domain_min=bad)
        with pytest.raises(
            ValueError, match=r"^AxisConfig\(domain_max=\.\.\.\) must be a real number "
        ):
            fm.AxisConfig(domain_max=bad)

    def test_degenerate_domain_refused_at_the_override_construction_seam(self, chart):
        """Design-review remediation (F1): `.override(...)` no longer
        bypasses `AxisConfig.__post_init__`'s cross-field check by
        constructing each leaf in isolation. `y_axis_domain_min`/
        `y_axis_domain_max` are grouped into one combined
        `AxisConfig(domain_min=..., domain_max=...)` construction
        (`_chart_config_wire_fragment`), which raises the SAME
        degenerate-domain message direct `AxisConfig(domain_min=10.0,
        domain_max=10.0)` construction raises — not the axis-qualified
        Rust-side backstop sentence this path previously fell through to
        (a DIFFERENT sentence than the construction-time one
        `_validate_domain_bounds` quotes verbatim from Rust, reached only
        because the Python check never ran)."""
        with pytest.raises(
            ValueError, match=r"domain endpoints must differ \(lo != hi\)"
        ) as direct:
            fm.AxisConfig(domain_min=10.0, domain_max=10.0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            bypass = chart.override(y_axis_domain_min=10.0, y_axis_domain_max=10.0)
            with pytest.raises(
                ValueError, match=r"domain endpoints must differ \(lo != hi\)"
            ) as via_override:
                bypass.to_svg()

        assert str(via_override.value) == str(direct.value), (
            "the override path must refuse with the SAME message AxisConfig's own "
            "construction does, not a different Rust-side backstop sentence"
        )

    def test_degeneracy_produced_by_zero_plus_max_in_combination(self, chart):
        """`apply_axis_domain_config`'s guard is checked on the COMPUTED
        domain, not on the individually-supplied fields — so `zero=True`
        (forces the low end to 0) combined with `domain_max=0.0` produces a
        zero-width `[0.0, 0.0]` domain that no construction-time check can
        see: `zero` alone is valid, and `domain_max=0.0` alone is valid.
        Every other case in this class refuses at Python construction time
        (`fm.AxisConfig(domain_min=10, domain_max=10)`); this one only
        surfaces once Rust resolves the computed domain during render.
        """
        with pytest.raises(ValueError, match="domain endpoints must differ"):
            chart.configure_axis(zero=True, domain_max=0.0).to_svg()


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
    derived from (`_override_apply._chart_config_wire_fragment`), so it
    accepts what that dataclass accepts, refuses what it refuses, and
    serializes the same way. Three separate leaks closed by one seam — each
    pinned below by its own verbatim repro — plus a parity sweep so a FUTURE
    validator added to any config class is covered without anyone
    remembering this file. See `TestOverrideCrossFieldValidators` below for
    the cross-field coverage `_chart_config_wire_fragment` closes by
    constructing each section from its whole set of named leaves, not one
    leaf at a time.
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
    def test_combined_leaf_construction_is_kwarg_order_independent(self, order):
        """Both override-kwarg orders for one section build ONE combined
        ``PaddingConfig(...)`` call, so they agree with each other and with
        the typed equivalent regardless of which key the caller wrote first.

        ``#135`` retired the baseline-subtraction ``_chart_config_wire_fragment``
        used to need: ``PaddingConfig.auto`` defaulted to ``False`` (the one
        config-class default ``to_dict()`` ever emitted unsolicited --
        see ``tests/test_padding_auto_default.py``'s structural empty-``to_dict()``
        pin), so a one-leaf-at-a-time build of ``{"padding_left": 70.0}``
        would have carried an unsolicited ``"auto": False``, and the two
        kwarg orders below would additionally have disagreed on which
        leaf's ``auto`` won. ``auto`` is now ``bool | None = None`` like
        every other optional leaf, so there is no unsolicited default left
        to subtract -- this pins that the combined-construction order
        independence still holds on its own.
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
        preset that lost its type would be misparsed as a numeric d3 spec.

        `chart`'s x/y are both quantitative, and the shared `axis` config key
        reaches both, so a genuinely time-typed preset now refuses (#133)
        rather than rendering — which is itself the companion-key proof this
        test exists for: `label_format_type` unambiguously declares `"time"`
        only when it actually reached the wire alongside `label_format`. If
        the seam dropped it (the bug this test guards against), the render
        would NOT refuse — it would silently fall through to the
        ambiguous/untyped numeric-format path instead. Both spellings must
        refuse with the identical message, proving they serialize the same
        fragment."""
        with pytest.raises(ValueError, match="requires a temporal scale") as override_exc:
            self._override(chart, axis_label_format="date_iso")
        with pytest.raises(ValueError, match="requires a temporal scale") as configure_exc:
            chart.configure_axis(label_format="date_iso").to_svg()
        assert str(override_exc.value) == str(configure_exc.value)


class TestOverrideCrossFieldValidators:
    """Cross-field `AxisConfig` validators must fire on the override spelling.

    Design-review finding F1: `_chart_config_wire_fragment` (formerly
    `_leaf_wire_fragment`) used to construct the owning dataclass one leaf at
    a time, so a validator that inspects TWO fields together never saw both
    in the same construction and never fired — even though its own docstring
    claimed every validator "today and later" covered the override spelling.
    `build_payload` now groups every leaf named for one section (e.g. every
    `axis_*` key in one `.override(...)` call) before constructing
    `AxisConfig` once from the combined set, so a cross-field validator sees
    exactly what a single `AxisConfig(...)` call with the same kwargs would.
    """

    @staticmethod
    def _override(chart, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return chart.override(**kwargs).to_svg()

    def test_label_format_and_label_format_raw_are_mutually_exclusive(self, chart):
        """Reviewer's exact repro: `AxisConfig(label_format=..., label_format_raw=...)`
        refuses at construction; the override spelling of the same two leaves
        (built one-leaf-at-a-time before this fix) silently combined them and
        reached the wire, where Rust's raw-first precedence picked a winner
        no construction-time check had ever seen."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            fm.AxisConfig(label_format="percent", label_format_raw=",.2f")
        with pytest.raises(ValueError, match="mutually exclusive"):
            self._override(chart, axis_label_format="percent", axis_label_format_raw=",.2f")

    def test_degenerate_domain_bounds_refuse_through_override_too(self, chart):
        """The second cross-field pair `AxisConfig.__init__` validates:
        `domain_min == domain_max` is a degenerate, zero-width domain.
        `_validate_domain_bounds` was written to quote Rust's own message
        verbatim, so a bare substring match on "domain endpoints must
        differ" alone cannot discriminate old from new behavior here (both
        the construction-time AND the Rust render-time backstop contain that
        phrase). Before this fix the override spelling built each bound in
        isolation (each one alone is valid) and only failed later, at
        render, with a DIFFERENT — axis-qualified — Rust-side sentence than
        `AxisConfig`'s own construction-time refusal; asserting exact text
        equality is what actually proves the override path now raises from
        the SAME construction-time check, not a different downstream one."""
        with pytest.raises(ValueError, match="domain endpoints must differ") as direct:
            fm.AxisConfig(domain_min=5, domain_max=5)
        with pytest.raises(ValueError, match="domain endpoints must differ") as via_override:
            self._override(chart, axis_domain_min=5, axis_domain_max=5)
        assert str(via_override.value) == str(direct.value)

    def test_valid_combined_leaves_still_reach_the_wire_together(self, chart):
        """The fix must not turn a legitimate combined section into a
        refusal: two independently-valid, non-conflicting leaves for the
        same section still both reach the wire from one `.override(...)`
        call."""
        out = self._override(chart, axis_label_angle=-45.0, axis_tick_count=3)
        expected = chart.configure_axis(label_angle=-45.0, tick_count=3).to_svg()
        assert out == expected


class TestGridStyleValueRefusals:
    """Both grid classes validate the numeric style pair through one body.

    `GridConfig` gained shape validation last cycle while its `width`/`opacity`
    kept passing silently, and `GridAxisConfig` had neither — leaving the
    family uneven in a way it had not been before. `width` takes the spec §4.7
    pixel contract (the same `coerce_pixel_value` `PaddingConfig` uses);
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
    def test_color_value_now_refuses(self, cls):
        """#107 (wave-1 hardening) closed the carve-out this test used to pin:
        config-surface color-VALUE refusal used to be deferred (an
        unparseable color silently kept the theme value); it now refuses at
        construction like every other config-surface value."""
        with pytest.raises(ValueError, match="not a valid color"):
            cls(color="notacolor")


class TestConfigSurfaceColorRefusal:
    """#107 (wave-1 hardening): the five ``configure.py`` dataclasses that
    accept a literal color string refuse an unparseable one at construction
    instead of silently keeping the theme default (the carve-out this
    replaces lived at ``configure.py``'s ``_validate_grid_style`` docstring).
    ``ferrum.color.to_hex`` is the shared vocabulary authority, so the
    refused message always names it; a value it accepts always constructs.
    """

    @pytest.mark.parametrize(
        ("ctor", "kwargs"),
        [
            (fm.AxisConfig, {"label_color": "notacolor"}),
            (fm.AxisConfig, {"title_color": "notacolor"}),
            (fm.AxisConfig, {"domain_color": "notacolor"}),
            (fm.AxisConfig, {"grid_color": "notacolor"}),
            (fm.LegendConfig, {"label_color": "notacolor"}),
            (fm.TitleConfig, {"color": "notacolor"}),
            (fm.TitleConfig, {"subtitle_color": "notacolor"}),
            (fm.GridAxisConfig, {"color": "notacolor"}),
            (fm.GridConfig, {"color": "notacolor"}),
        ],
        ids=[
            "AxisConfig.label_color",
            "AxisConfig.title_color",
            "AxisConfig.domain_color",
            "AxisConfig.grid_color",
            "LegendConfig.label_color",
            "TitleConfig.color",
            "TitleConfig.subtitle_color",
            "GridAxisConfig.color",
            "GridConfig.color",
        ],
    )
    def test_unparseable_color_refuses_at_construction(self, ctor, kwargs):
        with pytest.raises(ValueError, match="not a valid color"):
            ctor(**kwargs)

    @pytest.mark.parametrize(
        ("ctor", "kwargs"),
        [
            (fm.AxisConfig, {"label_color": "steelblue"}),
            (fm.LegendConfig, {"label_color": "#ff0000"}),
            (fm.TitleConfig, {"color": "rgb(0, 0, 0)"}),
            (fm.GridAxisConfig, {"color": "#eee"}),
            (fm.GridConfig, {"color": "green"}),
        ],
    )
    def test_valid_color_still_constructs(self, ctor, kwargs):
        assert ctor(**kwargs).to_dict()

    @pytest.mark.parametrize(
        ("ctor", "kwargs"),
        [
            (fm.AxisConfig, {"label_color": (1.0, 0.0, 0.0)}),
            (fm.LegendConfig, {"label_color": (255, 0, 0)}),
            (fm.TitleConfig, {"color": (1.0, 0.0, 0.0, 0.5)}),
            (fm.GridAxisConfig, {"color": (0, 0, 0)}),
            (fm.GridConfig, {"color": (0, 128, 0)}),
        ],
        ids=[
            "AxisConfig.label_color",
            "LegendConfig.label_color",
            "TitleConfig.color",
            "GridAxisConfig.color",
            "GridConfig.color",
        ],
    )
    def test_rgb_tuple_refuses_at_construction_not_the_wire(self, ctor, kwargs):
        """Close-remediation (design-review S2): these fields are typed
        ``str | None``, and ``to_hex`` happens to accept RGB(A) *tuples*
        structurally -- so before this fix, a tuple passed the #107 gate,
        serialized onto the wire, and failed there with a serde message
        naming neither the class nor the field
        (``ValueError: chart_config: invalid type: sequence, expected a
        string ...``). It must now refuse at construction, naming the
        class and field, exactly like an unparseable string does."""
        with pytest.raises(ValueError, match="must be a color string"):
            ctor(**kwargs)

    def test_grid_config_nested_x_mapping_color_refuses_too(self):
        """The mapping spelling of `GridConfig(x=...)` normalizes into a
        `GridAxisConfig`, so it must meet the identical color refusal, not
        bypass it the way the pre-normalization mapping spelling used to
        bypass `width`/`opacity` (see `TestGridStyleValueRefusals`)."""
        with pytest.raises(ValueError, match="not a valid color"):
            fm.GridConfig(x={"color": "notacolor"})

    def test_band_colors_transparent_is_a_legal_skip_not_a_color(self):
        """`"transparent"` (any case) means "draw no rect for this band" --
        matching `render/marks/axis.rs`'s own exemption -- so it must not be
        sent through the color parser, which would otherwise refuse it as a
        paint-clearing sentinel."""
        cfg = fm.GridConfig(band_colors=["#f0f0f0", "transparent", "TRANSPARENT"])
        assert cfg.band_colors == ["#f0f0f0", "transparent", "TRANSPARENT"]

    def test_band_colors_other_sentinel_still_refuses(self):
        """Only `"transparent"` is exempted; `"none"` has no special meaning
        for a band entry and is refused like any other unparseable value."""
        with pytest.raises(ValueError, match="not a valid color"):
            fm.GridConfig(band_colors=["#f0f0f0", "none"])

    def test_band_colors_bad_entry_names_its_index(self):
        with pytest.raises(ValueError, match=r"band_colors\[1\]"):
            fm.GridConfig(band_colors=["#f0f0f0", "notacolor"])


# ---------------------------------------------------------------------------
# #141 — secondary-y gridlines owned by the secondary axis
#
# A secondary axis always carries its own ticks (the axis chrome is
# unconditional), so every assertion here reads the secondary's own tick
# y-positions out of the interactive scene JSON. Different axes sharing the
# same plot height commonly land ticks at identical pixel positions by pure
# geometry (e.g. two 11-tick axes both touching the plot's top and bottom
# edge), so "does a gridline exist at this y" is not decidable by looking at
# one render alone. Every assertion below instead takes the exact multiset
# difference between a chart and a baseline that differs only in the thing
# being toggled, so a coincidental shared tick position never produces a
# false positive or false negative.
# ---------------------------------------------------------------------------


def _y2_grid_lines(chart, *, panel_index: int = 0) -> list[dict]:
    """Every node in the interactive scene's flat ``panel.grid`` list."""
    scene = json.loads(chart.interactive().scene_json)
    return scene["panels"][panel_index]["grid"]


def _y2_tick_ys(chart, *, y_slot: str, panel_index: int = 0) -> list[float]:
    """Tick-mark y positions of the secondary axis at *y_slot*.

    The axis group for a given ``y_slot`` always renders its ticks, domain
    line and labels regardless of whether its gridlines are lit, so this
    reads real tick positions straight from the renderer on both lit and
    unlit charts.
    """
    scene = json.loads(chart.interactive().scene_json)
    for node in scene["panels"][panel_index]["axes"]:
        if node["type"] == "group" and node["attrs"] == [["y_slot", y_slot]]:
            ticks = [
                child["y1"]
                for child in node["children"]
                if child["type"] == "line" and child["y1"] == child["y2"]
            ]
            assert ticks, f"y_slot={y_slot!r} axis group has no tick marks"
            return ticks
    raise AssertionError(f"no y_slot={y_slot!r} axis group in panel.axes")


def _y2_grid_entries(chart, *, panel_index: int = 0) -> Counter:
    """Multiset of ``(rounded y1, style json)`` over every grid node."""
    return Counter(
        (round(g["y1"], 6), json.dumps(g["style"], sort_keys=True))
        for g in _y2_grid_lines(chart, panel_index=panel_index)
    )


def _y2_added_entries(chart, baseline, *, panel_index: int = 0) -> Counter:
    """Grid entries present in *chart* beyond what *baseline* already has."""
    return _y2_grid_entries(chart, panel_index=panel_index) - _y2_grid_entries(
        baseline, panel_index=panel_index
    )


def _assert_adds_exactly_the_ticks(
    chart, baseline, tick_ys, *, label: str, panel_index: int = 0
) -> list[dict]:
    """*chart* must add exactly one gridline per *tick_ys* versus *baseline*.

    Returns the added entries' styles, so callers can also check them.
    """
    added = _y2_added_entries(chart, baseline, panel_index=panel_index)
    added_ys = sorted(y for (y, _style) in added.elements())
    expected_ys = sorted(round(y, 6) for y in tick_ys)
    assert added_ys == expected_ys, f"{label}: added y1s {added_ys} != expected ticks {expected_ys}"
    return [json.loads(style) for (_y, style) in added.elements()]


def _assert_adds_nothing(chart, baseline, *, label: str, panel_index: int = 0) -> None:
    added = _y2_added_entries(chart, baseline, panel_index=panel_index)
    assert not added, f"{label}: unexpected added gridlines {list(added.elements())}"


def _assert_adds_exactly_these_entries(chart, baseline, expected: Counter, *, label: str) -> None:
    """*chart* must add exactly *expected* -- position AND style fused into
    one multiset equality, so a lit cell is pinned on both axes at once
    rather than checking position and style as two separable claims."""
    added = _y2_added_entries(chart, baseline)
    assert added == expected, f"{label}: added {dict(added)} != expected {dict(expected)}"


def _assert_matches_theme_major(baseline, baseline_style: dict, *, label: str) -> None:
    """*baseline_style* must equal the primary axis's own gridline style.

    The caller derives *baseline_style* from the secondary's own "no style"
    render, which only proves the secondary is internally consistent with
    itself -- a secondary drawn in the wrong default (e.g. the minor-grid
    style) would still pass that check. Anchoring against *baseline*'s own
    primary gridlines (which carry no secondary styling at all) proves
    "theme major" against an independent source instead.
    """
    primary_styles = {style_json for (_y, style_json) in _y2_grid_entries(baseline)}
    assert primary_styles, f"{label}: baseline has no primary gridlines to anchor theme major"
    assert primary_styles == {json.dumps(baseline_style, sort_keys=True)}, (
        f"{label}: secondary's theme-major default {baseline_style} does not match "
        f"the primary axis's own gridline style {sorted(primary_styles)}"
    )


@pytest.fixture
def y2_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [3.0, 11.0, 7.0, 25.0, 14.0],
            "c": [1013.0, 2117.0, 3311.0, 4207.0, 5433.0],
        }
    )


def _base_chart(df: pl.DataFrame, mark: str, coord):
    """``fm.Chart(df).mark_<mark>()``, under *coord* when one is given."""
    chart = getattr(fm.Chart(df), f"mark_{mark}")()
    return chart if coord is None else chart.coord(coord)


def _secondary_y_route(df: pl.DataFrame, *, axis=None, coord=None):
    """route: ``SecondaryY`` desugar — appends one independent-y layer."""
    return _base_chart(df, "point", coord).encode(x="a", y="b") + fm.SecondaryY("c", axis=axis)


def _independent_layer_route(df: pl.DataFrame, *, axis=None, coord=None):
    """route: ``LayerChart(resolve={"y": "independent"})`` — explicit layers."""
    return fm.layer(
        _base_chart(df, "line", coord).encode(x="a", y="b"),
        _base_chart(df, "point", coord).encode(x="a", y=fm.Y("c", axis=axis)),
        resolve={"y": "independent"},
    )


def _three_stacked_route(df: pl.DataFrame, *, axis=None, coord=None):
    """route: three ``SecondaryY`` layers stacked on one chart.

    ``axis=`` patches only the middle (designated) secondary's slot; the
    other two stay unconfigured. This is what lets the same route exercise
    both a per-channel show/style source (which only ever addresses one
    secondary) and a broadcast ``axis_y2`` source (which reaches all three,
    since it is one chart-level slot, not a per-secondary address).
    """
    stacked = df.with_columns(
        (pl.col("c") * 0.5 + 1000.0).alias("c2"),
        (pl.col("c") * 2.0 - 500.0).alias("c3"),
    )
    return (
        _base_chart(stacked, "point", coord).encode(x="a", y="b")
        + fm.SecondaryY("c")
        + fm.SecondaryY("c2", axis=axis)
        + fm.SecondaryY("c3")
    )


@dataclass(frozen=True)
class _Y2Route:
    """One way a chart acquires one or more secondary y axes.

    ``designated_slot`` is the ``y_slot`` a per-channel ``axis=`` patches;
    ``all_slots`` is every secondary slot on the route, which a broadcast
    ``axis_y2=`` source reaches regardless of which one is "designated".
    """

    build: Callable[..., object]
    designated_slot: str
    all_slots: tuple[str, ...]


#: Every route the design spec names: ``SecondaryY``, an explicit
#: independent-y ``LayerChart``, and three stacked secondaries. Derived
#: once here; every test below draws its routes from this dict rather than
#: hand-listing them again.
_Y2_ROUTES: dict[str, _Y2Route] = {
    "secondary_y": _Y2Route(_secondary_y_route, "1", ("1",)),
    "independent_y_layer": _Y2Route(_independent_layer_route, "1", ("1",)),
    "three_stacked_secondaries": _Y2Route(_three_stacked_route, "2", ("1", "2", "3")),
}


@dataclass(frozen=True)
class _Y2ShowSource:
    """One surface that could plausibly turn a secondary's gridlines on.

    ``scope`` says which axis slot(s) a ``lit`` source reaches: ``"per_channel"``
    patches only a route's designated secondary; ``"axis_y2"`` is one
    chart-level slot that broadcasts to every secondary on the route;
    ``"chart"`` and ``"none"`` light nothing (isolation -- the chart-level
    theme flag and the two shared config surfaces never fall back onto a
    secondary).
    """

    name: str
    lit: bool
    scope: Literal["none", "per_channel", "axis_y2", "chart"]
    grid_value: bool | None = None
    chart_patch: Callable[[object], object] | None = None


#: Every surface named in the design spec (§4.3/§6.9) that could plausibly
#: set a secondary's grid show flag, including the no-op baseline. One
#: declared population -- callers read ``lit``/``scope``/``grid_value``/
#: ``chart_patch`` off each record instead of dispatching on the name.
_Y2_SHOW_SOURCES: tuple[_Y2ShowSource, ...] = (
    _Y2ShowSource("none", lit=False, scope="none"),
    _Y2ShowSource("axis_y2_grid_true", lit=True, scope="axis_y2", grid_value=True),
    _Y2ShowSource("per_channel_grid_true", lit=True, scope="per_channel", grid_value=True),
    _Y2ShowSource(
        "theme_grid_true",
        lit=False,
        scope="chart",
        chart_patch=lambda c: c.theme(fm.Theme(grid=True)),
    ),
    _Y2ShowSource(
        "configure_axis_grid_true",
        lit=False,
        scope="chart",
        chart_patch=lambda c: c.configure_axis(grid=True),
    ),
    _Y2ShowSource(
        "configure_grid_y_true",
        lit=False,
        scope="chart",
        chart_patch=lambda c: c.configure_grid(y=True),
    ),
    _Y2ShowSource("axis_y2_grid_false", lit=False, scope="axis_y2", grid_value=False),
)
_Y2_SHOW_BY_NAME = {s.name: s for s in _Y2_SHOW_SOURCES}


@dataclass(frozen=True)
class _Y2StyleSource:
    """Extra (non-``grid``) stroke fields layered onto a cell, independent
    of which show source (if any) is lighting it. Crossing every style
    source against every (unlit) show source is exactly spec §4.3's "style
    without show draws nothing"; crossing it against every lit show source
    proves the per-channel > axis_y2 > theme-major chain resolves field by
    field, not as one all-or-nothing block.
    """

    name: str
    axis_y2_style: dict
    per_channel_style: dict


_Y2_STYLE_SOURCES: tuple[_Y2StyleSource, ...] = (
    _Y2StyleSource("no_style", {}, {}),
    _Y2StyleSource(
        "axis_y2_style",
        {"grid_color": "#ff00ff", "grid_width": 4.0, "grid_dash": [2.0, 2.0], "grid_opacity": 0.5},
        {},
    ),
    _Y2StyleSource(
        "per_channel_color_wins",
        {"grid_color": "#ff0000", "grid_width": 3.0},
        {"grid_color": "#0000ff"},
    ),
)
_Y2_STYLE_BY_NAME = {s.name: s for s in _Y2_STYLE_SOURCES}
_Y2_NO_STYLE = _Y2_STYLE_BY_NAME["no_style"]


def _hex_to_rgba(color: str) -> dict:
    h = color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return {"r": r, "g": g, "b": b, "a": 255}


def _build_y2_cell(df: pl.DataFrame, route: _Y2Route, show: _Y2ShowSource, style: _Y2StyleSource):
    """Compose one (route, show_source, style_source) population cell."""
    axis_kwargs = dict(style.per_channel_style)
    if show.scope == "per_channel":
        axis_kwargs["grid"] = show.grid_value
    axis_y2_kwargs = dict(style.axis_y2_style)
    if show.scope == "axis_y2":
        axis_y2_kwargs["grid"] = show.grid_value

    axis_obj = fm.Axis(**axis_kwargs) if axis_kwargs else None
    chart = route.build(df, axis=axis_obj)
    if axis_y2_kwargs:
        chart = chart.configure(axis_y2=fm.AxisConfig(**axis_y2_kwargs))
    if show.chart_patch is not None:
        chart = show.chart_patch(chart)
    return chart


def _expected_grid_style(
    baseline_style: dict, per_channel_style: dict, axis_y2_style: dict
) -> dict:
    """The field-by-field precedence chain: per-channel > axis_y2 > theme
    major (*baseline_style*: the resolved style of the same cell rendered
    with no style override, so the theme's own value is never hand-coded
    here)."""

    def pick(field_name):
        if field_name in per_channel_style:
            return per_channel_style[field_name]
        if field_name in axis_y2_style:
            return axis_y2_style[field_name]
        return None

    color = pick("grid_color")
    width = pick("grid_width")
    dash = pick("grid_dash")
    opacity = pick("grid_opacity")
    # Start from the theme-major baseline so any other style field the
    # renderer emits (e.g. stroke_cap/stroke_join) is carried through
    # unexamined rather than hand-listed here field by field.
    expected = dict(baseline_style)
    if color is not None:
        expected["color"] = _hex_to_rgba(color)
    if width is not None:
        expected["width"] = width
    if dash is not None:
        expected["dash"] = dash
    if opacity is not None:
        # An explicit opacity override sets `stroke_opacity` alongside
        # `opacity` (absent entirely when nothing overrides it).
        expected["stroke_opacity"] = opacity
        expected["opacity"] = opacity
    return expected


def _expected_grid_entries(
    tick_ys_by_slot: dict[str, list[float]],
    *,
    designated_slot: str,
    style: _Y2StyleSource,
    baseline_style: dict,
) -> Counter:
    """Every (y, style) a lit cell must add, per secondary slot.

    A slot other than the designated one never had a per-channel ``Axis``
    object at all, so its style resolves from ``axis_y2`` / theme alone --
    only the designated slot's gridlines can carry a per-channel override.
    """
    entries: Counter = Counter()
    for slot, ys in tick_ys_by_slot.items():
        per_channel = style.per_channel_style if slot == designated_slot else {}
        expected_style = _expected_grid_style(baseline_style, per_channel, style.axis_y2_style)
        style_json = json.dumps(expected_style, sort_keys=True)
        for y in ys:
            entries[(round(y, 6), style_json)] += 1
    return entries


class TestAxisY2GridlinesVacuity:
    """#141 -- secondary gridlines are off unless asked."""

    def test_default_is_off(self, y2_df):
        """Vacuity baseline: an unconfigured chart lights no secondary route.

        Compared against a chart that has no secondary layer at all -- the
        primary axis's own gridline count and y-positions are unaffected by
        whether a secondary axis exists (only its reserved plot width is),
        so this baseline isolates "did the unlit secondary add anything".
        """
        primary_only = fm.Chart(y2_df).mark_point().encode(x="a", y="b")
        for label, route in _Y2_ROUTES.items():
            _assert_adds_nothing(route.build(y2_df), primary_only, label=label)

    def test_axis_y2_grid_true_differs_from_default(self, y2_df):
        """Vacuity control: the one surface this issue adds must move the SVG."""
        default = _secondary_y_route(y2_df)
        lit = default.configure(axis_y2=fm.AxisConfig(grid=True))
        assert lit.to_svg() != default.to_svg()


#: Every coordinate system a chart can declare (each public ``fm.Coord*``
#: class), plus ``None`` for no ``.coord()`` call at all.
_Y2_COORDS: tuple[str | None, ...] = (
    *sorted(name for name in dir(fm) if name.startswith("Coord")),
    None,
)


def _has_y_slot_axis(chart, y_slot: str) -> bool:
    scene = json.loads(chart.interactive().scene_json)
    return any(
        node["type"] == "group" and node["attrs"] == [["y_slot", y_slot]]
        for node in scene["panels"][0]["axes"]
    )


#: The coordinate systems that move y off the vertical axis, so a secondary
#: (independent) y axis has no place: a chart declaring one with an
#: independent-y layer is refused.
_Y2_REFUSING_COORDS = frozenset({"CoordFlip", "CoordPolar", "CoordGeo"})


def _independent_y_refusal(coord: str) -> str:
    return (
        f"SecondaryY (a layer with an independent y scale) is not supported under fm.{coord}: "
        "its right y axis has no place once the coord moves y off the vertical axis; draw the "
        "series on a Cartesian panel (put the categories on x and drop the coord), or draw it "
        "on the shared y axis"
    )


class TestAxisY2GridlinesCoord:
    """#141 -- a lit secondary rules its gridlines on every coordinate system that draws it.

    Every public ``fm.Coord*`` class and no ``.coord()`` at all are crossed
    with every route. A chart-level coord that moves y off the vertical axis
    (``_Y2_REFUSING_COORDS``) refuses the independent-y secondary on every
    route, static and interactive; every other one draws the secondary axis
    and adds exactly one gridline per secondary tick when it is lit.
    """

    def test_the_refusing_coords_are_public_coords(self):
        assert _Y2_REFUSING_COORDS <= set(_Y2_COORDS)

    @pytest.mark.parametrize("route_name", sorted(_Y2_ROUTES))
    def test_both_outcomes_occur(self, y2_df, route_name):
        """Some coords refuse and the rest draw the secondary axis: no coord renders without it."""
        route = _Y2_ROUTES[route_name]
        outcomes = set()
        for coord in _Y2_COORDS:
            try:
                drawn = _has_y_slot_axis(_coord_cell(y2_df, route, coord), route.designated_slot)
            except ValueError:
                outcomes.add("refused")
            else:
                outcomes.add("drawn" if drawn else "missing")
        assert outcomes == {"refused", "drawn"}

    @pytest.mark.parametrize("coord", _Y2_COORDS, ids=str)
    @pytest.mark.parametrize("route_name", sorted(_Y2_ROUTES))
    def test_a_lit_secondary_rules_lines_only_where_axes_are_drawn(self, y2_df, route_name, coord):
        route = _Y2_ROUTES[route_name]
        unlit = _coord_cell(y2_df, route, coord)
        lit = unlit.configure(axis_y2=fm.AxisConfig(grid=True))
        label = f"{route_name} x {coord}"
        if coord in _Y2_REFUSING_COORDS:
            for chart in (unlit, lit):
                for render in (chart.to_svg, lambda chart=chart: chart.interactive().scene_json):
                    with pytest.raises(ValueError) as refused:
                        render()
                    assert str(refused.value) == _independent_y_refusal(coord), label
            return
        assert _has_y_slot_axis(lit, route.designated_slot), label
        ticks = [y for slot in route.all_slots for y in _y2_tick_ys(lit, y_slot=slot)]
        _assert_adds_exactly_the_ticks(lit, unlit, ticks, label=label)


def _coord_cell(df: pl.DataFrame, route: _Y2Route, coord: str | None):
    return route.build(df, coord=None if coord is None else getattr(fm, coord)())


class TestAxisY2GridlinesMatrix:
    """#141 -- the full required population: every route x every show
    source x every style source (derived from ``_Y2_ROUTES`` /
    ``_Y2_SHOW_SOURCES`` / ``_Y2_STYLE_SOURCES`` above, never hand-listed
    again here).

    A lit cell is pinned on position AND stroke attributes at once: the
    expected-entries multiset fuses a gridline's y-position with its
    resolved style into one key, so the assertion can't pass by getting
    position right while silently drawing the wrong style (or vice versa).
    An unlit cell must add nothing, *and* must render byte-identically to
    the unconfigured chart -- true for every style source crossed in, which
    is exactly spec §4.3's "style without show draws nothing".
    """

    @pytest.mark.parametrize("style_name", sorted(_Y2_STYLE_BY_NAME))
    @pytest.mark.parametrize("show_name", sorted(_Y2_SHOW_BY_NAME))
    @pytest.mark.parametrize("route_name", sorted(_Y2_ROUTES))
    def test_matrix_cross(self, y2_df, route_name, show_name, style_name):
        route = _Y2_ROUTES[route_name]
        show = _Y2_SHOW_BY_NAME[show_name]
        style = _Y2_STYLE_BY_NAME[style_name]
        label = f"{route_name}/{show_name}/{style_name}"

        baseline = route.build(y2_df)
        chart = _build_y2_cell(y2_df, route, show, style)

        if not show.lit:
            _assert_adds_nothing(chart, baseline, label=label)
            assert chart.to_svg() == baseline.to_svg(), (
                f"{label}: a secondary that isn't shown must render byte-identically "
                "to the unconfigured default, regardless of any style source applied"
            )
            return

        slots = route.all_slots if show.scope == "axis_y2" else (route.designated_slot,)
        tick_ys_by_slot = {slot: _y2_tick_ys(chart, y_slot=slot) for slot in slots}

        no_style_chart = (
            chart if style.name == "no_style" else _build_y2_cell(y2_df, route, show, _Y2_NO_STYLE)
        )
        flat_ys = [y for ys in tick_ys_by_slot.values() for y in ys]
        baseline_style = _assert_adds_exactly_the_ticks(
            no_style_chart, baseline, flat_ys, label=f"{label} (theme-major baseline)"
        )[0]
        _assert_matches_theme_major(baseline, baseline_style, label=label)

        expected = _expected_grid_entries(
            tick_ys_by_slot,
            designated_slot=route.designated_slot,
            style=style,
            baseline_style=baseline_style,
        )
        _assert_adds_exactly_these_entries(chart, baseline, expected, label=label)


class TestAxisY2GridlinesThreeStackedIsolation:
    """A two-source interaction the single-source matrix cross above can't
    reach: a broadcast ``axis_y2`` ``True`` together with one secondary's
    own per-channel override.
    """

    def test_per_channel_false_overrides_a_broadcast_axis_y2_true(self, y2_df):
        route = _Y2_ROUTES["three_stacked_secondaries"]
        baseline = route.build(y2_df)
        chart = route.build(y2_df, axis=fm.Axis(grid=False)).configure(
            axis_y2=fm.AxisConfig(grid=True)
        )
        tick_ys = _y2_tick_ys(chart, y_slot="1") + _y2_tick_ys(chart, y_slot="3")
        _assert_adds_exactly_the_ticks(
            chart, baseline, tick_ys, label="outer two lit, middle overridden off"
        )


class TestAxisY2GridlinesFacet:
    """Spec §4.3: faceted charts draw one set of lit secondary gridlines
    per panel -- not a shared set collapsed onto a single panel, and not
    missing from any panel.
    """

    def test_each_panel_gets_its_own_lit_secondary_gridlines(self, y2_df):
        faceted_df = y2_df.with_columns(g=pl.Series(["p", "p", "p", "q", "q"]))
        baseline = _secondary_y_route(faceted_df).facet(col="g")
        chart = _secondary_y_route(faceted_df, axis=fm.Axis(grid=True)).facet(col="g")

        scene = json.loads(chart.interactive().scene_json)
        panel_count = len(scene["panels"])
        assert panel_count == 2, f"expected 2 facet panels, got {panel_count}"

        for panel_index in range(panel_count):
            tick_ys = _y2_tick_ys(chart, y_slot="1", panel_index=panel_index)
            _assert_adds_exactly_the_ticks(
                chart,
                baseline,
                tick_ys,
                label=f"panel {panel_index}",
                panel_index=panel_index,
            )
