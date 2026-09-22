"""Public-surface coverage for `#147` — one format re-derivation predicate.

The crate used to answer "is this `(format, format_type)` pair a d3 spec, a
strftime pattern, or the ordinal preset?" at six sites with three different
rules, kept in sync by prose. They are now one function, `FormatKind::resolve`,
whose third argument (`TimeAutodetect`) states whether the calling site can
RECOVER from a wrong auto-detection — the only axis the six ever disagreed on.

Every assertion here is on rendered tick-label / mark text — never that an SVG
was produced. The tests before ``TestEverySurfaceAnswersAMalformedPairAlike``
split into two kinds, and the split is load bearing, so do not read them as
pieces of RED evidence one-for-one:

- **5 regression pins**, which FAIL on the pre-fix tree: the three
  independent-facet-axis shapes, the text-mark pattern, and the control-char
  quirk. Each names its pre-fix value in its own docstring, so the
  discrimination is checkable without re-running the probe.
- **6 invariant guards**, which pass on the pre-fix tree too. They exist to
  catch a future change that breaks what this one preserved — the ordinal
  round-trips, the NF-B2 percent cases, the numeric-format-applied-in-place
  control. They are NOT evidence that this task fixed anything, and the
  report does not count them as such.

``TestEverySurfaceAnswersAMalformedPairAlike`` (wave 6A T17, the ruling that
the surfaces must actually agree) is a cross-surface matrix: each case says in
its own docstring which surfaces disagreed pre-fix.

An earlier revision of this docstring claimed all eleven fail pre-fix and that
each names a pre-fix value; both were false, and the file contradicted one of
its own docstrings eighty lines below. Corrected at cycle 3 after quality
review. If you add a test here, say which kind it is.

See `.sdd/task-13-report.md` §12-13 for the probe matrix and the RED proof,
and `.sdd/task-17-report.md` for the cross-surface matrix's RED proof.
"""

from __future__ import annotations

import datetime as dt
import json
import re

import polars as pl
import pytest

import ferrum as fm

TEXT = re.compile(r"<text[^>]*>([^<]*)</text>")

#: Deliberately NOT ``%b %Y``: at 90-day tick spacing the default spacing-keyed
#: temporal formatter emits exactly ``%b %Y``, so a site that silently DROPS the
#: caller's pattern is indistinguishable from one that applies it. ``%Y/%m/%d``
#: differs from the default at every spacing.
PATTERN = "%Y/%m/%d"


@pytest.fixture
def temporal_df() -> pl.DataFrame:
    dates = [dt.date(2020, 1, 1) + dt.timedelta(days=90 * i) for i in range(8)]
    return pl.DataFrame(
        {
            "d": dates,
            "v": [float(i) for i in range(8)],
            "g": ["a", "b"] * 4,
        }
    )


def rendered_text(chart) -> list[str]:
    return [t for t in TEXT.findall(chart.to_svg()) if t.strip()]


def patterned_labels(chart) -> list[str]:
    """Tick labels that actually went through ``%Y/%m/%d`` (two slashes)."""
    return [t for t in rendered_text(chart) if t.count("/") == 2]


class TestIndependentFacetAxisHonorsATemporalFormat:
    """An independent-x facet axis applies the caller's temporal tick format.

    Pre-fix these rendered the scale's own default labels (``"2020"``), because
    ``TickFormatMode::Immediate`` went straight to ``apply_tick_format``, whose
    only input is the already-formatted label STRINGS — which on a temporal
    scale cannot be re-parsed as epoch-ms, so the format was dropped. The
    shared-axis path honored the identical spelling, for the identical chart.
    """

    def _chart(self, df, **axis_kwargs):
        return (
            fm.Chart(df)
            .mark_line()
            .encode(
                x=fm.X("d:T", axis=fm.Axis(label_format=PATTERN, **axis_kwargs)),
                y="v:Q",
            )
        )

    def test_untyped_pattern_is_applied(self, temporal_df):
        """Regression pin. Pre-fix: ``['2020', '2020', ...]`` — pattern dropped."""
        chart = self._chart(temporal_df).facet(col="g").share_scale(x="independent")
        labels = patterned_labels(chart)
        assert labels, "independent facet axis must apply the temporal format"
        assert "2020/01/01" in labels

    def test_explicitly_time_typed_pattern_is_applied(self, temporal_df):
        """Regression pin. Pre-fix: dropped even with ``label_format_type="time"``."""
        chart = (
            self._chart(temporal_df, label_format_type="time")
            .facet(col="g")
            .share_scale(x="independent")
        )
        assert "2020/01/01" in patterned_labels(chart)

    def test_independent_and_shared_axes_agree(self, temporal_df):
        """Regression pin. One format discipline: the two spellings must agree.

        This is the finding itself — pre-fix the shared axis rendered
        ``2020/01/01`` and the independent one ``2020``, so the sets differed.
        """
        base = self._chart(temporal_df)
        shared = set(patterned_labels(base.facet(col="g")))
        independent = set(patterned_labels(base.facet(col="g").share_scale(x="independent")))
        assert shared, "shared facet axis must apply the format (control)"
        assert shared <= independent, (
            "every label the shared axis formats must also be formatted on the "
            f"independent one; shared={sorted(shared)} independent={sorted(independent)}"
        )

    def test_a_numeric_format_is_still_applied_in_place(self):
        """Invariant guard — passes pre-fix too. The other direction: routing
        through the shared resolver must not turn the per-panel surface into
        the DEFERRED one, whose threaded override ``scene_build`` never
        applies. Fails if the delegation ever starts deferring, since no label
        would end in ``%``."""
        df = pl.DataFrame(
            {"q": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], "v": [1.0, 2, 3, 4, 5, 6], "g": ["a", "b"] * 3}
        )
        chart = (
            fm.Chart(df)
            .mark_line()
            .encode(x=fm.X("q:Q", axis=fm.Axis(label_format=".0%")), y="v:Q")
            .facet(col="g")
            .share_scale(x="independent")
        )
        assert any(t.endswith("%") for t in rendered_text(chart))


class TestTextMarkAppliesItsResolvedPattern:
    """A ``text`` channel applies its strftime pattern instead of dropping it.

    Pre-fix ``marks/text.rs`` branched on ``format_type == Some("time")`` and
    called the default day-granularity formatter, discarding the pattern
    entirely — ``2020-01-01`` where the caller asked for ``2020/01/01``.
    """

    def test_time_typed_pattern_reaches_the_rendered_text(self, temporal_df):
        """Regression pin. Pre-fix: ``2020-01-01`` — the pattern was discarded
        and the default day-granularity formatter ran instead."""
        chart = (
            fm.Chart(temporal_df)
            .mark_text()
            .encode(
                x="v:Q",
                y="v:Q",
                text=fm.Text("d", format=PATTERN, format_type="time"),
            )
        )
        marks = [t for t in rendered_text(chart) if t.count("/") == 2]
        assert marks, "the text channel must apply its own strftime pattern"
        assert all(re.fullmatch(r"\d{4}/\d{2}/\d{2}", t) for t in marks), marks
        # The pre-fix spelling must be gone, not merely joined.
        assert not [t for t in rendered_text(chart) if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t)]


class TestOrdinalSentinelHasExactlyOneRepresentation:
    """`__ordinal__` stays the wire token; `\\u0001` stops being a second one."""

    @pytest.fixture
    def ordinal_df(self) -> pl.DataFrame:
        return pl.DataFrame({"n": list(range(1, 12)), "v": [float(i) for i in range(1, 12)]})

    def _chart(self, df, label_format):
        return (
            fm.Chart(df)
            .mark_line()
            .encode(x=fm.X("n:Q", axis=fm.Axis(label_format=label_format)), y="v:Q")
        )

    @pytest.mark.parametrize("spelling", ["ordinal", "__ordinal__"])
    def test_round_trips_through_to_json_and_renders_ordinals(self, ordinal_df, spelling):
        """Invariant guard — passes pre-fix too. Criterion 8 is a PRESERVATION
        criterion: both public spellings resolve to the one documented wire
        token, and must keep doing so."""
        chart = self._chart(ordinal_df, spelling)
        wire = json.loads(chart.to_json())["encoding"]["x"]["axis"]["label_format"]
        assert wire == "__ordinal__"
        labels = rendered_text(chart)
        for expected in ["1st", "2nd", "3rd", "4th", "11th"]:
            assert expected in labels, f"{expected!r} missing from {labels}"

    def test_a_literal_control_char_no_longer_renders_ordinals(self, ordinal_df):
        """Regression pin. `#147`'s probe-confirmed quirk, closed.

        A one-character ``'\\u0001'`` used to be re-encoded into ``D3Spec::ty``
        and dispatched to the ordinal formatter, so this nonsense input
        rendered ``1st…11th`` identically to the real preset. It is now an
        ordinary unrecognized d3 type char.
        """
        quirk = rendered_text(self._chart(ordinal_df, chr(1)))
        assert not any(t.endswith(("st", "nd", "rd", "th")) for t in quirk), quirk
        assert "1" in quirk and "11" in quirk
        # Discriminating against the real preset, not merely self-consistent.
        assert quirk != rendered_text(self._chart(ordinal_df, "__ordinal__"))


class TestPercentSpecIsNeverMisreadAsStrftime:
    """NF-B2's motivating case, now pinned at the public surface.

    A ``%``-bearing NUMERIC spec must never be handed to ``chrono`` just
    because it contains a ``%`` — under either `TimeAutodetect` setting.
    """

    def test_untyped_percent_spec_renders_percentages_on_a_numeric_axis(self):
        """Invariant guard — passes pre-fix too. Fails if ``Allowed`` ever
        leaks to a ``Denied`` surface and hands ``chrono`` a percent spec."""
        df = pl.DataFrame({"q": [0.0, 0.25, 0.5, 0.75, 1.0], "v": [1.0, 2, 3, 4, 5]})
        chart = (
            fm.Chart(df)
            .mark_line()
            .encode(x=fm.X("q:Q", axis=fm.Axis(label_format=".1%")), y="v:Q")
        )
        labels = rendered_text(chart)
        assert "0.0%" in labels and "100.0%" in labels

    def test_preset_resolved_percent_is_unaffected(self):
        """Invariant guard — passes pre-fix too. The ``percent`` preset carries
        ``format_type="number"`` explicitly, so it can never be auto-detected
        as strftime under either setting."""
        df = pl.DataFrame({"q": [0.0, 0.5, 1.0], "v": [1.0, 2, 3]})
        chart = (
            fm.Chart(df)
            .mark_line()
            .encode(x=fm.X("q:Q", axis=fm.Axis(label_format="percent")), y="v:Q")
        )
        assert any(t.endswith("%") for t in rendered_text(chart))


# ---- one outcome per malformed pair, at every surface (wave 6A T17) ----

YEARLY = pl.DataFrame(
    {
        "d": [dt.datetime(2000 + i, 6, 1) for i in range(8)],
        "n": [float(i) for i in range(1, 9)],
        "g": ["a", "b"] * 4,
    }
)


def _axis(field: str, fmt: str | None, fmt_type: str | None):
    kwargs = {}
    if fmt is not None:
        kwargs["label_format"] = fmt
    if fmt_type is not None:
        kwargs["label_format_type"] = fmt_type
    return fm.Chart(YEARLY).mark_point().encode(x=fm.X(field, axis=fm.Axis(**kwargs)), y="n:Q")


def _format_kwargs(fmt: str | None, fmt_type: str | None) -> dict:
    kwargs = {}
    if fmt is not None:
        kwargs["format"] = fmt
    if fmt_type is not None:
        kwargs["format_type"] = fmt_type
    return kwargs


#: Each surface, as ``(field, spec, format_type) -> chart``. The field is the
#: column the format applies to: ``d`` (temporal) or ``n`` (quantitative).
SURFACES = {
    "axis": lambda field, fmt, ft: _axis(field, fmt, ft),
    "independent_facet_axis": lambda field, fmt, ft: (
        _axis(field, fmt, ft).facet(col="g").share_scale(x="independent")
    ),
    "colorbar": lambda field, fmt, ft: (
        fm.Chart(YEARLY)
        .mark_point()
        .encode(
            x="n:Q", y="n:Q", color=fm.Color(field, legend=fm.Legend(**_format_kwargs(fmt, ft)))
        )
    ),
    "text": lambda field, fmt, ft: (
        fm.Chart(YEARLY)
        .mark_text()
        .encode(x="n:Q", y="n:Q", text=fm.Text(field, **_format_kwargs(fmt, ft)))
    ),
}


class TestEverySurfaceAnswersAMalformedPairAlike:
    """One outcome per ``(spec, format_type)`` pair, whichever surface renders it.

    User ruling on ``#147``: the surfaces must actually agree on identical
    input, not document where they differ. Each case compares a surface
    against ITSELF on the well-formed spelling the malformed pair must mean,
    so a surface that answers differently from the others fails on its own row.
    """

    @pytest.mark.parametrize("surface", sorted(SURFACES))
    def test_the_ordinal_preset_wins_over_a_stray_time_type(self, surface):
        """Regression pin for colorbar, text and the independent facet axis.

        Pre-fix: the axis rendered ``1st 2nd …`` while the colorbar and the
        text mark rendered ``1970-01-01`` (their post-resolution
        ``format_type`` guard discarded the ``Ordinal``), and the independent
        facet axis refused the pair as a time format on a non-temporal scale.
        The shared-axis row is the control that passed pre-fix too.
        """
        build = SURFACES[surface]
        stray = rendered_text(build("n:Q", "__ordinal__", "time"))
        declared = rendered_text(build("n:Q", "__ordinal__", None))
        assert stray == declared
        assert "1st" in stray or "2nd" in stray, stray
        assert not any(t.startswith("1970") for t in stray), stray

    @pytest.mark.parametrize("spec", [".2f", ".3e"])
    @pytest.mark.parametrize("surface", sorted(SURFACES))
    def test_a_pattern_free_time_typed_spec_falls_back_to_the_temporal_default(self, surface, spec):
        """Regression pin for both axis rows.

        Pre-fix: on a temporal axis the spec was threaded on, re-seeded
        untyped and applied as d3 to the year labels (``2001.00``,
        ``2.001e+3``), while the colorbar and text mark rendered their
        temporal default for the identical pair. The colorbar and text rows
        are controls that passed pre-fix too.
        """
        build = SURFACES[surface]
        malformed = rendered_text(build("d:T", spec, "time"))
        default = rendered_text(build("d:T", None, "time"))
        assert malformed == default
        assert any(re.match(r"200\d", t) for t in malformed), malformed
        assert not any(t.startswith("1970") for t in malformed), malformed
        assert not any(re.fullmatch(r"\d{4}\.\d{2}|\d\.\d{3}e\+\d", t) for t in malformed)

    @pytest.mark.parametrize("surface", ["axis", "independent_facet_axis"])
    def test_a_pattern_free_time_typed_spec_on_a_quantitative_axis_refuses(self, surface):
        """Regression pin for the shared axis; the independent facet axis is
        its parity partner and refused pre-fix too.

        Pre-fix: the shared axis applied ``.2f`` as d3 and rendered, while
        the independent facet axis refused the identical pair. Both now
        raise the ``#133`` refusal, and the message names the declared
        ``"time"`` type as the cause, since ``.2f`` alone does not say why a
        temporal scale is needed.
        """
        chart = SURFACES[surface]("n:Q", ".2f", "time")
        with pytest.raises(
            ValueError,
            match=r"""format "\.2f" is typed "time", which on the 'x' axis requires a temporal scale""",
        ):
            chart.to_svg()
