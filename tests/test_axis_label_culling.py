"""Regression coverage for F-L07-04 (spec §4.6): label culling is reachable.

Pre-fix, the x-axis collision cascade's rotation stage (S3) auto-passed at
-90 degrees for any real label width -- ``cos(-90deg)`` is not exactly 0 in
IEEE-754 ``f64``, so a naive ``label_width * cos(angle)`` fit check was
(almost) always satisfied, and the cull (S4) / elide (S5) stages downstream
were unreachable. ``crates/ferrum-core/src/layout/axis.rs``'s
``cascade_collision_recovery`` now judges rotation fit against
``rotated_x_label_footprint``, which adds the label's own line-height
"thickness" to the fit test, so a dense categorical axis actually culls once
rotation genuinely can't resolve the collision.

``cull_threshold`` is a minimum PIXEL-GAP threshold between adjacent visible
tick labels (``Theme.cull_threshold``'s docstring), not a label-count gate;
``0`` disables culling entirely (spec §4.6 / D9) -- dense labels then elide
(truncate with an ellipsis) instead of being dropped.

Spec-review cycle 2 finding: the first cut of this fix let ``cull_threshold``
floor S4's stride *after* culling was already triggered by pure overlap, but
never entered the S0-S3 fit tests themselves -- so on any axis where labels
(or their rotation) fit with room to spare, the threshold was wholly inert
(30 labels in a 600px chart rendered an identical label count at
``cull_threshold`` in {0, 8, 60, 500}). The fix folds ``cull_threshold`` into
every stage's fit budget (``footprint <= effective_slot_w - cull_threshold``,
i.e. the gap left between labels must be at least ``cull_threshold`` px, not
merely non-negative); ``0`` collapses back to the pure-overlap budget.
Spec-review cycle 3 finding (superseded by cycle 4 -- described here in the
past tense for history, NOT current behavior): cycle 2's fix folded
``cull_threshold`` into the S0 (flat) / S1 (wrap) fit budgets too,
correctly -- but the pre-existing S1 (wrap) implementation decided whether
to actually split each label onto multiple lines PER LABEL (keeping a label
combined on one line if its own combined width happened to still fit,
splitting only the ones that didn't). Once the tightened budget made
wrapping reachable on an ordinary axis (not just deliberately-dense ones),
that per-label decision produced a RAGGED axis: sibling tick labels of the
identical format (e.g. every tick reading ``"%b %Y"``) rendered
inconsistently -- some as one combined line, others split onto two --
purely because of real-font width variance across word choices
(``"May"``/``"Jul"``/``"Nov"`` measure wider than ``"Jan"``/``"Mar"``).
Cycle 3's fix made ``wrap_label``'s space rule split UNCONDITIONALLY at
every space (mirroring the underscore rule's pre-existing
unconditional-split precedent), so once the cascade resolved to "wrap" for
an axis, every label in that axis wrapped the SAME way -- spec §4.6's "the
cascade degrades uniformly." Cycle 4 (below) found this over-corrected and
replaced it; ``wrap_label``'s space rule is greedy line-fill again today.

Spec-review cycle 4 finding: cycle 3's unconditional one-word-per-line split
over-corrected -- it degraded EVERY multi-word label to one word per line
wherever wrapping engaged, including at ``cull_threshold=0``, where users
who disabled the density gate entirely still lost the pre-task greedy
multi-word packing (``"United States of America"`` rendered as four stacked
single-word lines instead of the pre-task ``"United States"`` / ``"of
America"``-style pack). Two invariants now hold simultaneously:
``cull_threshold=0`` reproduces the pre-task greedy per-label pack
byte-for-byte (``wrap_label``'s Rule 2 restored to greedy line-fill), and
``cull_threshold>0`` still degrades uniformly across sibling labels but via
a shared per-axis line count (``target_lines = max`` natural line count
across the axis), forcing each shorter label up to that count via balanced
word-group splitting (``wrap_label_to_line_count`` /
``pack_words_to_line_count``) rather than one-word-per-line -- so
``"United States of America"`` still packs multiple words per line even
once the axis-wide degradation kicks in.

Spec-review cycle 5 finding: cycle 4's fix applied the uniform-degradation
treatment at S1 (the ORIGINAL-font wrap stage) only. S2b -- the reduced-font
wrap stage (``cascade_collision_recovery``, reached after S2a's flat-at-
reduced-font check fails) -- calls the same ``wrap_label`` but had no
``cull_threshold`` branch at all, so the exact ragged-sibling axis the S1
fix eliminates was still reachable one cascade stage down, at the DEFAULT
theme, for a realistic label set (mixed single/multi-word categories narrow
enough to need the reduced font). The fix extracts the shared
``uniformize_wrapped_labels`` helper (used by both S1 and S2b) so the same
two invariants -- ``cull_threshold=0`` byte-identical to pre-task,
``cull_threshold>0`` uniform via a shared per-axis line count -- hold at
every wrap-invoking stage of the cascade. ``estimate_x_label_band``'s own S1
wrap call needs no equivalent treatment: its band-height reservation only
ever depends on the maximum line count across labels, which doesn't change
whether or not shorter siblings are later uniformized up to it.

Quality-review cycle 1 finding (S3, headline): S5's elide decision judged
fit with ``rotated_x_label_footprint(-90, w, line_h)``, which at -90 degrees
is ``~line_h`` and effectively WIDTH-INDEPENDENT for any realistic label
width, while the remedy applied (``elide_to_fit``) can only ever shrink the
footprint's WIDTH term -- truncating a vertical label's TEXT cannot shrink
its line-height "thickness" at all. The elide predicate could therefore
never be satisfied by the elision it triggered: a real, dense/narrow chart
(24 ``"Region NN"`` labels at 380px, ``cull_threshold=0``) rendered every
label as a bare, information-free "…", where pre-task (the -90 auto-pass
this whole task fixes) rendered every label legibly. Elision is now gated
on whether truncation could plausibly reduce a label's footprint at all
(``cos_best * width`` must clear a 1px floor); when it's infeasible for
every label, the cascade falls back to ``Rotated`` with labels rendered
INTACT -- the pre-task outcome for a ``cull_threshold`` the user explicitly
disabled -- rather than destroying content for a remedy that could never
work. Quality-review cycle 1 finding (S2, companion): ``greedy_pack_words``'
``Vec<&str>`` accumulator was not byte-faithful to pre-task for labels with
EMPTY split words (a leading/trailing/doubled space) -- pre-task's `String`
accumulator silently absorbed them (``String::push_str("")`` on an empty
string is a no-op), while ``vec![""]`` is not empty, so an empty word
survived into the joined line as a stray space or a phantom extra line.
Fixed with an explicit empty-word filter before packing.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import date

import polars as pl

import ferrum as fm

_SVG_NS = "{http://www.w3.org/2000/svg}"


def _svg_root(svg: str) -> ET.Element:
    return ET.fromstring(svg)


def _text_element_count(svg: str) -> int:
    """Total number of ``<text>`` elements in *svg*."""
    return len(_svg_root(svg).findall(".//" + _SVG_NS + "text"))


def _all_text_content(svg: str) -> list[str]:
    texts: list[str] = []
    for elem in _svg_root(svg).iter():
        if elem.text and elem.text.strip():
            texts.append(elem.text.strip())
        if elem.tail and elem.tail.strip():
            texts.append(elem.tail.strip())
    return texts


def _dense_chart_svg(*, cull_threshold: int | None, n: int = 200, width: int = 150) -> str:
    df = pl.DataFrame(
        {
            "cat": [f"category_label_{i:03d}" for i in range(n)],
            "y": list(range(n)),
        }
    )
    chart = fm.Chart(df).mark_bar().encode(x="cat:N", y="y:Q").properties(width=width, height=300)
    if cull_threshold is not None:
        chart = chart.theme(fm.Theme(cull_threshold=cull_threshold))
    return chart.to_svg()


def test_dense_labels_cull_at_default_threshold() -> None:
    """200 long category labels in a 150px chart: the default cull_threshold
    (8, a pixel gap) must drop some tick labels' text entirely, relative to
    the same chart with culling disabled (``cull_threshold=0``).

    Pre-fix, this comparison would show NO difference at all -- the -90°
    rotation stage auto-passed regardless of ``cull_threshold``, so culling
    never fired under any configuration.
    """
    culled_svg = _dense_chart_svg(cull_threshold=None)  # default theme (8px)
    disabled_svg = _dense_chart_svg(cull_threshold=0)

    culled_count = _text_element_count(culled_svg)
    disabled_count = _text_element_count(disabled_svg)

    assert culled_count < disabled_count, (
        "default cull_threshold should drop some tick-label text nodes "
        f"relative to cull_threshold=0; got {culled_count} vs {disabled_count}"
    )


def test_cull_threshold_zero_never_culls() -> None:
    """``cull_threshold=0`` must never drop a label OR destroy its content.

    Quality-review finding (S3): the previous assertion
    (``_text_element_count(svg) >= n``) counted every ``<text>`` node in the
    WHOLE svg -- y-axis ticks, titles, and the rest give this fixture ~12
    nodes of slack -- and was satisfied even when every one of the 200 tick
    labels rendered as a bare, content-free ellipsis "…" (the exact S3
    regression: a width-independent elide predicate paired with a
    width-only elide remedy that could never satisfy it). Post-fix, elision
    that cannot reduce a label's footprint falls back to ``Rotated`` with
    labels rendered INTACT, so this dense/narrow fixture no longer even
    reaches elision -- every category's FULL original text must be
    recoverable via the per-label ``_present_categories`` helper (the same
    one four sibling pins already use), not merely "some text node exists."
    """
    n = 200
    svg = _dense_chart_svg(cull_threshold=0, n=n)
    present = _present_categories(svg, n, prefix="category_label_", digits=3)
    assert present == n, (
        f"cull_threshold=0 must retain every one of {n} category labels' "
        f"FULL original text (a dropped or content-destroyed label is "
        f"exactly the quality-review S3 regression this pins); only "
        f"{present}/{n} recognizable"
    )
    all_text = _all_text_content(svg)
    assert "…" not in all_text, (
        f"cull_threshold=0 must not collapse any label to a bare, "
        f"content-free ellipsis; got a standalone '…' text node in {all_text!r}"
    )


def test_genuinely_fitting_rotation_wins_over_culling() -> None:
    """A moderate number of short-enough categories that DO fit once rotated
    must render every label, uncullled and unelided -- rotation is tried
    against real fit first, and a genuine fit wins (spec §4.6).
    """
    categories = [f"cat{i:02d}" for i in range(15)]
    df = pl.DataFrame({"cat": categories, "y": list(range(15))})
    svg = (
        fm.Chart(df)
        .mark_bar()
        .encode(x="cat:N", y="y:Q")
        .properties(width=600, height=300)
        .to_svg()
    )
    all_text = _all_text_content(svg)
    joined = " ".join(all_text)
    assert "…" not in joined, f"expected no elision; text={all_text!r}"
    for cat in categories:
        assert any(cat in t for t in all_text), (
            f"category {cat!r} missing -- a genuinely-fitting rotation must "
            f"not cull any label. text={all_text!r}"
        )


def _present_categories(svg: str, n: int, prefix: str = "category_", digits: int = 2) -> int:
    """Count how many of the ``n`` category labels appear (in full, possibly
    as one word of a wrapped label) anywhere in *svg*'s text content --
    dropped by culling if absent, present (whole or truncated by elision, but
    S1/S2/S3 never truncate here) otherwise. ``digits`` matches the caller's
    zero-padding width (e.g. ``category_label_005`` needs ``digits=3``).
    """
    all_text = _all_text_content(svg)
    return sum(1 for i in range(n) if any(f"{prefix}{i:0{digits}d}" in t for t in all_text))


def _thirty_labels_600px_chart() -> fm.Chart:
    n = 30
    df = pl.DataFrame({"cat": [f"category_{i:02d}" for i in range(n)], "y": list(range(n))})
    return fm.Chart(df).mark_bar().encode(x="cat:N", y="y:Q").properties(width=600, height=300)


def test_cull_threshold_zero_vs_sixty_render_different_label_counts() -> None:
    """Spec-review cycle 2 RED proof, reproduced verbatim from the reviewer's
    live repro: 30 labels in a 600px chart (a slot width where -90° rotation
    fits with room to spare under a pure-overlap check) must render a
    DIFFERENT visible-label count at ``cull_threshold=0`` vs ``=60`` --
    before the fix, `cull_threshold` never entered the fit test unless
    rotation had already failed on pure overlap, so this comparison showed
    NO difference at any threshold.
    """
    chart = _thirty_labels_600px_chart()
    svg_zero = chart.theme(fm.Theme(cull_threshold=0)).to_svg()
    svg_sixty = chart.theme(fm.Theme(cull_threshold=60)).to_svg()

    present_zero = _present_categories(svg_zero, 30)
    present_sixty = _present_categories(svg_sixty, 30)

    assert present_zero == 30, f"cull_threshold=0 should show all 30 labels; got {present_zero}"
    assert present_sixty < present_zero, (
        f"cull_threshold=60 must render FEWER visible labels than cull_threshold=0 "
        f"for the identical chart; got {present_sixty} vs {present_zero}"
    )


def test_cull_threshold_larger_than_any_gap_culls_to_spaced_subset() -> None:
    """A `cull_threshold` far larger than the chart's own tick spacing must
    still cull down to a small, spaced subset of labels -- not leave every
    label visible (which pure-overlap-only fit tests, ignorant of
    `cull_threshold`, would do since none of these labels visually overlap
    at any rotation) and not drop every label to zero (the first tick always
    survives the cull stride).
    """
    chart = _thirty_labels_600px_chart()
    svg = chart.theme(fm.Theme(cull_threshold=500)).to_svg()
    present = _present_categories(svg, 30)
    assert 0 < present < 30, (
        f"cull_threshold=500 (>> the chart's own gaps) must cull down to a "
        f"small spaced subset, not zero and not all 30; got {present}"
    )
    assert present <= 6, f"cull_threshold=500 should leave very few labels visible; got {present}"


def test_cull_threshold_zero_matches_pure_overlap_boundary() -> None:
    """`cull_threshold=0` must be identical to the pure-overlap-only fit test
    (no gap requirement beyond non-overlap) right AT the boundary where a
    small positive threshold already tips into culling -- proving `0` isn't
    silently treated as some small implicit gap of its own. For this
    30-labels/600px chart, `cull_threshold=5` is the first value that culls
    (empirically the tightest boundary for this fixture); `0` must still
    show every label.
    """
    chart = _thirty_labels_600px_chart()
    svg_zero = chart.theme(fm.Theme(cull_threshold=0)).to_svg()
    svg_five = chart.theme(fm.Theme(cull_threshold=5)).to_svg()

    present_zero = _present_categories(svg_zero, 30)
    present_five = _present_categories(svg_five, 30)

    assert present_zero == 30, (
        f"cull_threshold=0 must show every label (pure-overlap fit only); got {present_zero}"
    )
    assert present_five < present_zero, (
        "cull_threshold=5 must already cull relative to cull_threshold=0 at this "
        f"boundary fixture, discriminating 0 from a nearby positive value; "
        f"got {present_five} vs {present_zero}"
    )


def test_cull_threshold_zero_not_silently_replaced_by_rust_default() -> None:
    """`Theme(cull_threshold=0)` is a valid, explicit "disable" value -- it
    must render DIFFERENTLY from the default theme (Rust-side default of 8),
    not be silently dropped in favor of that default. Uses the same
    30-labels/600px fixture, where the default (8) already culls (15 of 30
    visible, per `test_dense_labels_cull_at_default_threshold`'s sibling
    scenario) while 0 must not.
    """
    chart = _thirty_labels_600px_chart()
    svg_default = chart.to_svg()  # no .theme() call -- Rust-side default (8)
    svg_zero = chart.theme(fm.Theme(cull_threshold=0)).to_svg()

    present_default = _present_categories(svg_default, 30)
    present_zero = _present_categories(svg_zero, 30)

    assert present_zero == 30, f"cull_threshold=0 should show all 30 labels; got {present_zero}"
    assert present_default < present_zero, (
        "the default theme (cull_threshold=8) must cull relative to an explicit "
        f"cull_threshold=0 on the identical chart; got {present_default} vs {present_zero} "
        "-- if these matched, 0 would be indistinguishable from being silently "
        "replaced by the Rust-side default"
    )


# Extracted to tests/_temporal_ticks.py (shared with test_flexibility_campaign.py)
# -- quality-review cycle 1 finding: this module's own copy had drifted from
# the sibling's (matched a month token via a looser regex that also matched
# non-month three-letter capitalized words like "Val"/"Sum", instead of set
# membership). See the module docstring's quality-review cycle 1 section.
from tests._temporal_ticks import month_year_tick_shapes as _month_year_tick_shapes  # noqa: E402


def test_eighteen_month_axis_wraps_uniformly_not_raggedly() -> None:
    """Spec-review cycle 3 RED proof: an 18-month ``"%b %Y"`` axis at the
    DEFAULT theme must resolve every date tick the SAME way -- all one line,
    or all wrapped onto two -- never a ragged mix.

    This exact fixture's real per-tick slot width is narrow enough that the
    spec §4.6 gap requirement genuinely isn't satisfiable flat for every
    tick at the default `cull_threshold` (empirically: it starts wrapping at
    `cull_threshold=4`, well below the default of 8), so wrapping is the
    legitimately-resolving stage -- what must not happen is SOME ticks
    staying one-line while others split. Pre-fix (cycle 2's S0/S1 budget
    change without a uniform S1), this rendered a mix like
    ``["Jan 2020", "Mar 2020", "May", "2020", "Jul 2020", ...]``.
    """
    df = pl.DataFrame(
        {
            "date": pl.date_range(date(2020, 1, 1), date(2021, 6, 1), "1mo", eager=True),
            "val": list(range(18)),
        }
    )
    svg = (
        fm.Chart(df)
        .mark_line()
        .encode(x=fm.X("date:T", axis=fm.Axis(label_format="%b %Y")), y="val:Q")
        .to_svg()
    )
    tick_labels = re.findall(r"<text[^>]*>([^<]+)</text>", svg)
    shapes = _month_year_tick_shapes(tick_labels)
    assert shapes, f"expected at least one 'MMM YYYY' tick; got {tick_labels}"
    assert len(set(shapes)) == 1, (
        "every 'MMM YYYY' tick in one axis must wrap the SAME way (spec §4.6: "
        f"the cascade degrades uniformly); got per-tick shapes {shapes} from "
        f"raw ticks {tick_labels}"
    )


# ---------------------------------------------------------------------------
# Spec-review cycle 4: cull_threshold=0 must preserve pre-task greedy
# multi-word packing; cull_threshold>0 must degrade uniformly WITHOUT ever
# forcing worst-case one-word-per-line packing (see module docstring).
# ---------------------------------------------------------------------------

_MULTI_WORD_COUNTRIES = [
    "United States of America",
    "Costa Rica",
    "United Kingdom",
    "South Korea",
    "New Zealand",
    "Saudi Arabia",
]


def _multi_word_countries_chart() -> fm.Chart:
    df = pl.DataFrame(
        {"country": _MULTI_WORD_COUNTRIES, "val": list(range(1, len(_MULTI_WORD_COUNTRIES) + 1))}
    )
    return (
        fm.Chart(df).mark_bar().encode(x="country:N", y="val:Q").properties(width=900, height=300)
    )


def _label_lines_for_words(all_text: list[str], words: list[str]) -> list[str]:
    """Find the contiguous run of *all_text* entries whose combined
    whitespace-split words match *words* in order, and return that run --
    the physical lines/text-nodes the label actually rendered as (a label
    wrapped onto N lines produces N entries; a label kept on one line, or
    with 2+ words packed onto some lines, produces fewer than ``len(words)``
    entries).
    """
    target = list(words)
    n = len(all_text)
    for start in range(n):
        collected: list[str] = []
        end = start
        while end < n and len(collected) < len(target):
            collected.extend(all_text[end].split())
            end += 1
        if collected == target:
            return all_text[start:end]
    raise AssertionError(f"could not find word sequence {target!r} in {all_text!r}")


def test_cull_threshold_zero_preserves_pretask_greedy_packing() -> None:
    """Spec-review cycle 4 RED proof (invariant 1): ``cull_threshold=0``
    must reproduce the pre-task greedy multi-word pack byte-for-byte --
    ``"United States of America"`` (4 words) must still combine 2+ words
    onto at least one line where they fit, never degrade to one word per
    line just because a sibling label in the axis needed to wrap. Disabling
    the density gate entirely must not lose the pre-task packing behavior.
    """
    svg = _multi_word_countries_chart().theme(fm.Theme(cull_threshold=0)).to_svg()
    all_text = _all_text_content(svg)
    lines = _label_lines_for_words(all_text, "United States of America".split())
    assert len(lines) < 4, (
        "cull_threshold=0 must preserve pre-task greedy multi-word packing for "
        f"'United States of America' (4 words); got it split across {len(lines)} "
        f"physical lines {lines!r} -- one-word-per-line indicates the wrap rule "
        f"regressed to the cycle-3 unconditional split. full text: {all_text!r}"
    )


def test_cull_threshold_positive_wraps_uniformly_without_worst_case_packing() -> None:
    """Spec-review cycle 4 RED proof (invariant 2): once ``cull_threshold>0``
    forces this mixed-length multi-word axis to wrap, every country label
    must resolve to the SAME line count (uniform degradation, spec §4.6) --
    AND at least one label (``"United States of America"``, the one with
    the most words) must still pack 2+ words per line, proving uniformity
    came from a shared per-axis line count rather than collapsing every
    label to one word per line.
    """
    svg = _multi_word_countries_chart().theme(fm.Theme(cull_threshold=8)).to_svg()
    all_text = _all_text_content(svg)

    line_counts = {
        country: len(_label_lines_for_words(all_text, country.split()))
        for country in _MULTI_WORD_COUNTRIES
    }
    assert len(set(line_counts.values())) == 1, (
        "every country label must wrap to the SAME line count once "
        f"cull_threshold>0 (spec §4.6 uniform degradation); got {line_counts}"
    )

    usa_lines = _label_lines_for_words(all_text, "United States of America".split())
    assert any(" " in line for line in usa_lines), (
        "'United States of America' must still pack 2+ words on at least one "
        "line even once cull_threshold>0 forces a uniform line count -- "
        f"worst-case one-word-per-line degradation is exactly what this fix "
        f"must avoid; got lines {usa_lines!r}"
    )


# ---------------------------------------------------------------------------
# Spec-review cycle 5: the reviewer's exact S2b (reduced-font wrap) repro --
# see module docstring.
# ---------------------------------------------------------------------------

_S2B_MIXED_LENGTH_CATEGORIES = [
    "Massachusetts Institute",
    "Costa Rica",
    "New Zealand",
    "South Africa",
    "United Kingdom",
]


def _s2b_categories_chart() -> fm.Chart:
    df = pl.DataFrame(
        {
            "cat": _S2B_MIXED_LENGTH_CATEGORIES,
            "val": list(range(1, len(_S2B_MIXED_LENGTH_CATEGORIES) + 1)),
        }
    )
    return fm.Chart(df).mark_bar().encode(x="cat:N", y="val:Q").properties(width=500, height=300)


def test_s2b_reduced_font_wrap_ragged_at_zero_uniform_at_default_threshold() -> None:
    """Spec-review cycle 5 RED proof: the reviewer's exact repro -- 5 mixed
    single/multi-word categories at 500px, no other chart properties set --
    resolves via S2b (the reduced-font wrap stage: verified below at
    font-size 9.02, the ``FONT_SHRINK_FACTOR``-reduced size), not S1. The
    same uniform-degradation guarantee S1 already carries must hold there
    too.

    At ``cull_threshold=0``: RAGGED is the CORRECT, pre-task-identical
    result -- ``"Massachusetts Institute"`` doesn't fit combined even at the
    reduced font and wraps to 2 lines, while the other four (shorter)
    2-word categories each fit combined and stay on 1 line -- a genuinely
    non-uniform set of natural line counts, which is what pre-task
    (never-uniformized) code would also produce. At the DEFAULT theme
    (``cull_threshold=8``): every category must resolve to the SAME line
    count -- before this cycle's fix, S2b had no `cull_threshold` branch at
    all, so this rendered identically (ragged) at both 0 and 8.
    """
    chart = _s2b_categories_chart()

    svg_zero = chart.theme(fm.Theme(cull_threshold=0)).to_svg()
    assert 'font-size="9.02"' in svg_zero, (
        "fixture must reach S2b (reduced font 9.02) for this pin to be "
        f"meaningful; got svg head: {svg_zero[:800]!r}"
    )
    all_text_zero = _all_text_content(svg_zero)
    line_counts_zero = {
        cat: len(_label_lines_for_words(all_text_zero, cat.split()))
        for cat in _S2B_MIXED_LENGTH_CATEGORIES
    }
    assert len(set(line_counts_zero.values())) > 1, (
        "cull_threshold=0 must preserve the pre-task RAGGED per-label result "
        f"at S2b (byte-identity, not forced uniformity); got uniform line "
        f"counts {line_counts_zero} -- if every category now wraps the same "
        "way even at 0, byte-identity to pre-task has regressed"
    )
    assert line_counts_zero["Massachusetts Institute"] == 2, (
        "'Massachusetts Institute' must still wrap to 2 lines at the reduced "
        f"font; got {line_counts_zero}"
    )

    svg_default = chart.to_svg()  # default theme -- cull_threshold=8
    assert 'font-size="9.02"' in svg_default, (
        f"fixture must still reach S2b at the default theme; got svg head: {svg_default[:800]!r}"
    )
    all_text_default = _all_text_content(svg_default)
    line_counts_default = {
        cat: len(_label_lines_for_words(all_text_default, cat.split()))
        for cat in _S2B_MIXED_LENGTH_CATEGORIES
    }
    assert len(set(line_counts_default.values())) == 1, (
        "every category must wrap to the SAME line count once cull_threshold>0 "
        f"reaches the S2b (reduced-font) wrap stage; got {line_counts_default}"
    )
