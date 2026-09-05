"""Batch-D T9: `Smooth`/`mark_smooth` method + degree routing (spec §4.6).

Covers the "first half" of F-L03-08: `method="glm"` must route to a real
GLM fit (not silently alias to `"lm"`), `method="lm"` with `degree >= 2`
must resolve to a real polynomial fit, and `degree` on a method with no
degree concept ("log"/"sqrt"/"glm") must refuse rather than being
silently ignored (closing D9). GAM (`method="gam"`) is a separate task
(T10) on the same file and is intentionally not covered here.

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


@pytest.mark.parametrize("method", ["log", "sqrt", "glm"])
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
