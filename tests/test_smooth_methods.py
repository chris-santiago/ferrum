"""Batch-D T9: `Smooth`/`mark_smooth` method + degree routing (spec §4.6).

Covers the "first half" of F-L03-08: `method="glm"` must route to a real
GLM fit (not silently alias to `"lm"`), `method="lm"` with `degree >= 2`
must resolve to a real polynomial fit, and `degree` on a method with no
degree concept ("log"/"sqrt"/"glm") must refuse rather than being
silently ignored (closing D9).

T10 extends this module with the second half of F-L03-08: `method="gam"`
must fit a real generalized additive model — a deterministic penalized
cubic B-spline, not a refusal and not an alias for an existing method.
Those tests live at the bottom, under their own banner.

Round 2 additions: `degree` on `"quadratic"`/`"cubic"` is accepted only as
a no-op when it matches the alias's fixed degree and otherwise refuses
(the alias-degree policy), and a `degree` that leaves the design
rank-deficient for the data at hand refuses on the ungrouped fit path
(the rank-deficiency policy) — both locked in `.sdd/task-9-report.md`'s
round-2 "policy decisions". These are Rust-side policies; the tests below
pin them from the Python entry points that reach them (`Smooth` directly
and `mark_smooth`).

Round 4 additions: a degenerate fit branch (rank-deficient design,
zero-variance x, non-convergent IRLS, too few points) must emit the same
output schema as a well-supported sibling when `show_metrics=True` — the
`_metrics_text`/`_metrics_y` columns present but null, rather than
dropped — so a `groupby`/facet-panel schema never silently diverges with
the data; and a non-`Utf8` `groupby` dtype (`Int64`/`Boolean`) combined
with `show_metrics=True` and `.facet(...)` must render rather than raise
the "only reconcilable for Float64/Utf8" error the rust-quality cycle-3
verdict reproduced through the public API.
"""

from __future__ import annotations

import json

import polars as pl
import pytest

import ferrum as fm
from ferrum import Smooth


def _xy(n: int = 20) -> pl.DataFrame:
    xs = [float(i) for i in range(n)]
    ys = [3.0 + 2.0 * x + (0.4 if i % 2 == 0 else -0.4) for i, x in enumerate(xs)]
    return pl.DataFrame({"x": xs, "y": ys})


def _transform_method(chart: fm.Chart) -> object:
    """The serialized `method` field of a chart's sole transform, read from
    the ChartSpec (not the rendered SVG) — the fastest, least-confounded
    way to observe what `desugar_smooth` actually forwarded to `Smooth`."""
    spec = chart.to_spec()
    transforms = json.loads(spec.to_json())["transforms"]
    return transforms[0]["method"]


# ---------------------------------------------------------------------------
# glm routing
# ---------------------------------------------------------------------------


def test_smooth_glm_method_accepted():
    """RED-proof: before batch-D T9, "glm" hit `Smooth`'s `other` catch-all
    and raised "unknown method". It must now construct successfully."""
    spec = Smooth(x="x", y="y", method="glm")
    cs = fm.ChartSpec(mark="line", x="x", y="y", transforms=[spec])
    rt = fm.ChartSpec.from_json(cs.to_json())
    assert rt == cs


def test_mark_smooth_glm_renders_svg():
    """The glm fit line must actually be drawn. `"<path" in svg` or
    `"d=" in svg` are both vacuous here: every ferrum SVG has a `<clipPath
    id="...">` element (matching `"d="` as a substring of nothing — the
    id, not a path data attribute), and the glm fit renders as a
    `<polyline>`, never a `<path>`. A degenerate glm input (e.g. constant
    x) degrades to zero drawn elements with no error — see the module
    docstring's "Degenerate-input policy" reference in `smooth.rs` — so the
    positive assertion must count real mark elements, not grep for a
    substring both a broken and a working render satisfy."""
    df = _xy()
    svg = fm.Chart(df).mark_smooth(method="glm").encode(x="x", y="y").to_svg()
    assert "<svg" in svg
    assert "NaN" not in svg
    assert svg.count("<polyline") == 1


def test_smooth_glm_ci_band_differs_from_lm_discriminating():
    """RED-proof discriminator against a fake `"glm" -> SmoothMethod::Lm`
    alias (see the Rust-side pin `test_glm_ci_band_differs_from_lm_ci_band_
    proving_distinct_implementation`): the real GLM CI (a Wald band with no
    dispersion scaling) has a different width than `lm`'s t-distribution
    band, even though the two fitted lines coincide (Gaussian/Identity IRLS
    is exactly OLS). `mark_smooth(ci=...)` renders a ribbon (CI band) + line
    layered chart, so an aliased implementation would render byte-identical
    SVGs for both methods; the real implementations must not.

    NaN-proof: a broken/degenerate render (all-NaN grid, zero elements
    drawn) would also make `lm_svg != glm_svg` true without proving
    anything about a real, distinct GLM implementation. The assertions
    below require both renders to actually contain a fitted line and a
    finite CI band before comparing them.
    """
    df = pl.DataFrame(
        {
            "x": [float(i) for i in range(20)],
            "y": [3.0 + 2.0 * i + (1.5 if i % 2 == 0 else -1.5) for i in range(20)],
        }
    )
    lm_svg = fm.Chart(df).mark_smooth(method="lm", ci=0.95, n=10).encode(x="x", y="y").to_svg()
    glm_svg = fm.Chart(df).mark_smooth(method="glm", ci=0.95, n=10).encode(x="x", y="y").to_svg()

    for svg in (lm_svg, glm_svg):
        assert "NaN" not in svg
        assert svg.count("<polyline") == 1  # fitted line
        assert svg.count("<path") == 1  # CI ribbon band

    assert lm_svg != glm_svg


# ---------------------------------------------------------------------------
# lm + degree >= 2 -> polynomial promotion
# ---------------------------------------------------------------------------


def test_smooth_lm_degree_2_equals_method_quadratic_byte_equal(curved_xy_df):
    """`degree=2` == `method="quadratic"` byte-equal (task brief)."""
    df = curved_xy_df
    lm_deg2 = fm.Chart(df).mark_smooth(method="lm", degree=2, ci=None).encode(x="x", y="y")
    quadratic = fm.Chart(df).mark_smooth(method="quadratic", ci=None).encode(x="x", y="y")
    assert lm_deg2.to_svg() == quadratic.to_svg()


def test_smooth_lm_default_degree_not_forwarded_ci_none_path():
    """RED-proof for the `ci=None` forwarding site in `desugar_smooth`
    (`statistical.py`'s single-mark fast path): a bare `method="lm"` call
    must not forward any `degree`, so Rust's own `Lm` default applies. A
    regression that reintroduces an unconditional `degree=2` forward (the
    pre-T9 shape) would serialize `method` as `{"poly": {"degree": 2}}`
    here instead of the bare string `"lm"` — checked directly on the
    ChartSpec, not inferred from SVG rendering, so this cannot be defeated
    by curvature-insensitive test data the way an SVG comparison on
    near-linear data can."""
    chart = fm.Chart(_xy()).mark_smooth(method="lm", ci=None).encode(x="x", y="y")
    assert _transform_method(chart) == "lm"


def test_smooth_lm_default_degree_not_forwarded_layered_ci_path():
    """Same RED-proof as above for the second, layered forwarding site
    (`statistical.py`'s `ci` band path) — the quality-review finding was
    that every existing test used `ci=None`, so a coder who conditionalized
    only one of the two forwarding sites would pass every existing test."""
    chart = fm.Chart(_xy()).mark_smooth(method="lm", ci=0.95).encode(x="x", y="y")
    assert _transform_method(chart) == "lm"


def test_smooth_lm_default_degree_stays_linear_not_promoted_to_quadratic(curved_xy_df):
    """SVG-level companion to the two spec-level pins above, using
    genuinely curved data (`_xy()`'s near-straight-line data makes a
    quadratic fit visually indistinguishable from linear, which is exactly
    why the original version of this test didn't discriminate — see
    `.sdd/task-9-python-quality-reviewer.md` finding 1): a bare `lm` call
    must render identically to an explicit `degree=1` call, and distinctly
    from a real quadratic fit."""
    df = curved_xy_df
    bare = fm.Chart(df).mark_smooth(method="lm", ci=None).encode(x="x", y="y").to_svg()
    degree_1 = (
        fm.Chart(df).mark_smooth(method="lm", degree=1, ci=None).encode(x="x", y="y").to_svg()
    )
    quadratic = fm.Chart(df).mark_smooth(method="quadratic", ci=None).encode(x="x", y="y").to_svg()
    assert bare == degree_1
    assert bare != quadratic


def test_lmplot_order_2_matches_mark_smooth_quadratic(curved_xy_df):
    """Spec §4.7: `lmplot(order=2)` must match `mark_smooth(method="quadratic")`."""
    df = curved_xy_df
    lmplot_svg = fm.lmplot(
        df, x="x", y="y", order=2, ci=None, scatter=False, show_metrics=False
    ).to_svg()
    quadratic_svg = (
        fm.Chart(df).mark_smooth(method="quadratic", ci=None).encode(x="x", y="y").to_svg()
    )
    assert lmplot_svg == quadratic_svg


# ---------------------------------------------------------------------------
# degree refusals (closing D9)
# ---------------------------------------------------------------------------


def test_smooth_lm_degree_zero_refuses():
    with pytest.raises(ValueError, match="degree"):
        Smooth(x="x", y="y", method="lm", degree=0)


@pytest.mark.parametrize("method", ["log", "sqrt", "glm", "gam"])
def test_smooth_degree_refused_for_non_polynomial_non_loess_methods(method):
    with pytest.raises(ValueError, match="degree"):
        Smooth(x="x", y="y", method=method, degree=2)


def test_smooth_loess_degree_still_validated():
    """Regression: the degree<->method refusal must not break loess's own
    existing degree=1|2 validation."""
    with pytest.raises(ValueError, match="degree"):
        Smooth(x="x", y="y", method="loess", degree=3)
    # Valid loess degrees still construct fine.
    Smooth(x="x", y="y", method="loess", degree=1)
    Smooth(x="x", y="y", method="loess", degree=2)


# ---------------------------------------------------------------------------
# round-2 policies: alias-degree conflicts, rank-deficiency refusal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "matching_degree", "conflicting_degree"),
    [("quadratic", 2, 5), ("cubic", 3, 2)],
)
def test_smooth_alias_matching_degree_is_accepted_noop(method, matching_degree, conflicting_degree):
    """A `degree` that agrees with the alias's fixed degree is a harmless
    no-op, the same tolerance `method="lm", degree=1` already gets."""
    Smooth(x="x", y="y", method=method, degree=matching_degree)


@pytest.mark.parametrize(
    ("method", "matching_degree", "conflicting_degree"),
    [("quadratic", 2, 5), ("cubic", 3, 2)],
)
def test_smooth_alias_conflicting_degree_refuses(method, matching_degree, conflicting_degree):
    """A `degree` that conflicts with the alias's fixed degree refuses,
    naming `method="lm"` as the way to actually get that degree — closing
    the residue the spec reviewer flagged (`method="quadratic", degree=5`
    used to silently fit degree 2 with no diagnostic)."""
    with pytest.raises(ValueError, match=f"conflicts with method='{method}'"):
        Smooth(x="x", y="y", method=method, degree=conflicting_degree)


def test_smooth_lm_rank_deficient_degree_refuses_on_ungrouped_fit():
    """A `degree` that leaves the polynomial design rank-deficient for the
    data at hand refuses at fit time on the ungrouped path, rather than
    silently degrading to an all-NaN grid (the same refusal class as
    `degree=0`, reached through a different enforcement site since
    rank-deficiency depends on how many rows survive null/NaN dropping,
    which is only known at `apply()` time)."""
    df = _xy(n=20)
    with pytest.raises(ValueError, match="rank-deficient"):
        fm.Chart(df).mark_smooth(method="lm", degree=20, ci=None).encode(x="x", y="y").to_svg()


def test_smooth_lm_degree_exactly_n_minus_one_is_not_rank_deficient():
    """Boundary: `degree == n_points - 1` is exactly determined (not
    rank-deficient) and still fits, on the same 20-row fixture the refusal
    test above uses at `degree=20`."""
    df = _xy(n=20)
    svg = fm.Chart(df).mark_smooth(method="lm", degree=19, ci=None).encode(x="x", y="y").to_svg()
    assert svg.count("<polyline") == 1


# ---------------------------------------------------------------------------
# round 4: degenerate-branch metrics-schema uniformity + dtype-generic
# facet-panel schema reconciliation (rust-quality cycle-3 findings)
# ---------------------------------------------------------------------------


def _facet_smooth_df() -> pl.DataFrame:
    """Two facet levels sharing one `x`/`y` pair: `"few"` has 3 rows
    (rank-deficient for `method="cubic"`, which needs >= 4), `"many"` has 8
    (sufficient). Mirrors the Rust-level `facet_smooth_batch` fixture."""
    return pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "y": [0.0, 1.0, 4.0, 0.0, 1.0, 8.0, 27.0, 64.0, 125.0, 216.0, 343.0],
            "sp": ["few"] * 3 + ["many"] * 8,
        }
    )


def test_lmplot_facet_sparse_panel_with_show_metrics_renders():
    """RED-proof for the root-cause fix (`build_degenerate_smooth_batch`):
    before batch-D T9 round 4, a degenerate facet panel's fit dropped its
    `_metrics_text`/`_metrics_y` columns while a well-supported sibling
    panel kept them — a schema divergence `concat_batches_by_name` happened
    to already reconcile for these two (`Float64`/`Utf8`) column types, but
    the divergence itself was the defect. `order=3` on the sparse `"few"`
    level (3 rows) is rank-deficient; the well-supported `"many"` level (8
    rows) fits and gets a real metrics annotation."""
    df = _facet_smooth_df()
    svg = fm.lmplot(
        df, x="x", y="y", order=3, ci=None, scatter=False, show_metrics=True, col="sp"
    ).to_svg()
    assert "<svg" in svg
    # One fitted line for "many" (the sparse "few" panel degrades to NaN and
    # draws nothing); one metrics annotation for "many" only.
    assert svg.count("<polyline") == 1
    assert svg.count("R²") == 1


def test_mark_smooth_facet_int64_groupby_with_show_metrics_renders():
    """The live repro the rust-quality cycle-3 verdict reproduced through
    the public API: `groupby=` at a non-`Utf8` dtype (`Int64`), combined
    with `show_metrics=True` and `.facet(col=...)`, used to raise
    "schema divergence across panels is only reconcilable for Float64/Utf8
    columns; column 'gi' is Int64" — because one facet panel's `groupby`
    values are entirely null (`apply_grouped`'s empty-partition fallback
    drops the group column from that panel's own output) while the other
    panel's real groups carry it. `concat_batches_by_name` must now widen
    (null-pad) instead of erroring, for any dtype."""
    df = pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 1.0],
            "y": [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 2.0],
            "sp": ["p1"] * 4 + ["p2"] * 4,
            "gi": [None, None, None, None, 1, 1, 2, 2],
        }
    )
    svg = (
        fm.Chart(df)
        .encode(x="x", y="y")
        .mark_smooth(method="cubic", show_metrics=True, ci=0.95, groupby="gi")
        .facet(col="sp")
        .to_svg()
    )
    assert "<svg" in svg


def test_mark_smooth_facet_boolean_groupby_with_show_metrics_renders():
    """Boolean twin of the Int64 repro above — the reviewer's second named
    dtype."""
    df = pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 1.0],
            "y": [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 2.0],
            "sp": ["p1"] * 4 + ["p2"] * 4,
            "gb": [None, None, None, None, True, True, False, False],
        }
    )
    svg = (
        fm.Chart(df)
        .encode(x="x", y="y")
        .mark_smooth(method="cubic", show_metrics=True, ci=0.95, groupby="gb")
        .facet(col="sp")
        .to_svg()
    )
    assert "<svg" in svg


# ---------------------------------------------------------------------------
# T10: gam (penalized cubic B-splines)
# ---------------------------------------------------------------------------


def _wiggly(n: int = 60) -> pl.DataFrame:
    """Data with two genuine turning points.

    `curved_xy_df` (conftest) is a monotone quadratic — a real polynomial
    fit tracks it exactly, so it cannot distinguish a spline from a
    quadratic. A sine over more than one period can only be followed by a
    method that bends more than once, which is what `gam` claims to do.
    """
    import math

    xs = [i * 0.15 for i in range(n)]
    ys = [math.sin(x * 1.4) * 5.0 + 0.3 * x for x in xs]
    return pl.DataFrame({"x": xs, "y": ys})


def test_smooth_gam_constructs_and_round_trips():
    """`Smooth(method="gam")` is accepted (it raised "unknown method"
    before T10) and survives a spec JSON round-trip."""
    t = Smooth("x", "y", method="gam", ci=None)
    chart = fm.Chart(_wiggly()).transform(t).encode(x="x", y="y").mark_line()
    spec = chart.to_spec()
    assert spec == type(spec).from_json(spec.to_json())
    assert json.loads(spec.to_json())["transforms"][0]["method"] == "gam"


def test_mark_smooth_gam_renders_svg():
    """The gam fit line must actually be drawn — counted, not grepped, for
    the reason spelled out in `test_mark_smooth_glm_renders_svg`."""
    svg = fm.Chart(_wiggly()).mark_smooth(method="gam", ci=None).encode(x="x", y="y").to_svg()
    assert "<svg" in svg
    assert "NaN" not in svg
    assert svg.count("<polyline") == 1


def test_smooth_gam_differs_from_lm_and_loess_discriminating():
    """RED-proof discriminator: `gam` is neither `lm` nor `loess` in
    disguise. All three must render a real fit line (so a broken render
    cannot masquerade as "different"), and gam's must differ from both on
    data with two turning points."""
    df = _wiggly()

    def render(method: str) -> str:
        svg = fm.Chart(df).mark_smooth(method=method, ci=None).encode(x="x", y="y").to_svg()
        assert svg.count("<polyline") == 1, f"{method} must draw exactly one fit line"
        assert "NaN" not in svg, f"{method} render must not contain NaN"
        return svg

    gam, lm, loess = render("gam"), render("lm"), render("loess")
    assert gam != lm
    assert gam != loess


def test_smooth_gam_is_deterministic_and_seed_invariant():
    """gam uses no randomness: repeated renders are byte-identical, and
    unlike `loess` (whose CI band is a seeded bootstrap) its band does not
    move with `seed=`. The loess control keeps the second half of this
    assertion honest."""
    df = _wiggly()

    def render(method: str, seed: int) -> str:
        return (
            fm.Chart(df)
            .mark_smooth(method=method, ci=0.95, seed=seed)
            .encode(x="x", y="y")
            .to_svg()
        )

    assert render("gam", 0) == render("gam", 0)
    assert render("gam", 0) == render("gam", 987654)
    assert render("loess", 0) != render("loess", 987654)


def test_smooth_gam_ci_band_renders_a_ribbon():
    """With `ci` set, gam emits a posterior band that renders as a filled
    ribbon path alongside the fit polyline."""
    svg = fm.Chart(_wiggly()).mark_smooth(method="gam", ci=0.95).encode(x="x", y="y").to_svg()
    assert svg.count("<polyline") == 1
    assert svg.count("<path") == 1
    assert "NaN" not in svg


def test_smooth_gam_degree_refuses():
    """`gam` has no degree concept (fixed cubic spline; smoothness comes
    from the GCV-chosen penalty), so `degree=` must refuse rather than be
    silently ignored — the same disposition `log`/`sqrt`/`glm` get."""
    with pytest.raises(ValueError, match="degree"):
        Smooth("x", "y", method="gam", degree=2)
    with pytest.raises(ValueError, match="degree"):
        fm.Chart(_wiggly()).mark_smooth(method="gam", degree=2).encode(x="x", y="y").to_svg()


def test_smooth_gam_grouped_renders_one_curve_per_group():
    """Grouped gam composes with the shared groupby plumbing: two groups,
    two fit lines."""
    df = _wiggly()
    doubled = pl.concat(
        [
            df.with_columns(pl.lit("a").alias("g")),
            df.with_columns((pl.col("y") + 30.0).alias("y"), pl.lit("b").alias("g")),
        ]
    )
    svg = (
        fm.Chart(doubled)
        .mark_smooth(method="gam", ci=None, groupby="g")
        .encode(x="x", y="y", color="g")
        .to_svg()
    )
    assert svg.count("<polyline") == 2
    assert "NaN" not in svg


def test_smooth_gam_facet_sparse_panel_degrades_instead_of_failing():
    """A facet panel too sparse for a cubic basis degrades to a blank fit
    for that panel while the well-supported panel still renders — the
    partition contract every other method in the family follows."""
    dense = _wiggly(40).with_columns(pl.lit("many").alias("panel"))
    sparse = pl.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 4.0], "panel": ["few"] * 3})
    svg = (
        fm.Chart(pl.concat([dense, sparse]))
        .mark_smooth(method="gam", ci=None)
        .encode(x="x", y="y")
        .facet(col="panel")
        .to_svg()
    )
    assert "<svg" in svg
    assert svg.count("<polyline") == 1, "only the well-supported panel draws a fit"
    assert "NaN" not in svg


def test_smooth_gam_whole_frame_too_sparse_renders_blank_not_raises():
    """gam's whole-frame policy: too little data is DATA degeneracy, not a
    caller's bad parameter, so it degrades like `lm` on a zero-variance x
    rather than raising the way a rank-deficient `degree=` does."""
    df = pl.DataFrame({"x": [1.0, 1.0, 2.0, 2.0], "y": [1.0, 2.0, 3.0, 4.0]})
    svg = fm.Chart(df).mark_smooth(method="gam", ci=None).encode(x="x", y="y").to_svg()
    assert "<svg" in svg
    assert svg.count("<polyline") == 0
    assert "NaN" not in svg


def test_smooth_gam_show_metrics_renders_one_annotation():
    """gam composes with `show_metrics`. The metrics are computed from
    `build_predictor`'s gam arm — a missing arm there would fall back to
    some other method's residuals — so the annotation must be present
    exactly once AND differ from the linear fit's, which is what proves the
    gam predictor (not a sibling's) produced it."""
    df = _wiggly()

    def render(method: str) -> str:
        svg = (
            fm.Chart(df)
            .mark_smooth(method=method, ci=None, show_metrics=True)
            .encode(x="x", y="y")
            .to_svg()
        )
        assert svg.count("R²") == 1
        assert "NaN" not in svg
        return svg

    assert render("gam") != render("lm")


def test_smooth_gam_appears_in_the_registry_methods_vocabulary():
    """The registry's `smooth` vocabulary is derived from the same Rust
    `METHODS` const the parser checks, so `"gam"` must be visible to any
    registry consumer without a second edit."""
    from ferrum import _core

    assert "gam" in _core.transform_registry("smooth")["methods"]
