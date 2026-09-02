"""Completeness guard for the color-channel typing invariant.

**The invariant.** In ``src/ferrum/marks`` and ``src/ferrum/plots``, every
binding to the ``color`` *encoding* channel whose value is a data-driven
field must declare its scale type: it routes through
``ferrum.marks._desugar_helpers.nominal_color_channel`` (for a group/class
discriminator) or through an explicit ``Color(...)`` construction (for a
field that is continuous by design). A bare name or a bare field-name
string is a defect, because an untyped field infers its scale type from the
column's *runtime dtype* — so an integer-valued discriminator (a class id, a
fold index, a model id, an ``Int64`` ``hue=``) silently resolves to a
Continuous color scale. On a line or ribbon mark that collapses the groups
into one merged shape; on a bar, rect, polygon, rule, segment, or tick mark
it renders a fabricated colorbar over a categorical field with **no warning
at all**. Both are wrong charts, and the silent half is the dangerous one.

**Why this is a test and not a review checklist.** Three consecutive design
review cycles found the same class unswept. Cycle 1 named one call site and
it was fixed; cycle 2 found four more siblings; cycle 3 found the audit
itself had been scoped to the literal token ``color_field`` while the class
reached ``plots/*.py``'s ``hue=`` bindings, where five charts were rendering
wrong. Each remediation was correct and each was incomplete, because the
rule lived in a docstring and an ``.sdd`` report rather than in anything
that fails. This module is the artifact that converts the rule into an
enforced invariant: a new raw binding fails here immediately, and a
deliberate exception must be written into :data:`ALLOWED` with a rationale,
which puts the disposition in code review instead of a report no future
reader will consult.

**No second color vocabulary.** Classifying a string constant as "a literal
color" rather than "a field name" is delegated entirely to ferrum's single
parser (``ferrum.color.to_hex``) and its single sentinel predicate
(``ferrum.marks.base._is_paint_sentinel``). This module defines no color
names, no hex patterns, and no sentinel spellings of its own.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Iterator, NamedTuple

import pytest

from ferrum.color import to_hex
from ferrum.marks.base import _is_paint_sentinel

#: Repository-relative roots the guard walks.
SCANNED_ROOTS = ("src/ferrum/marks", "src/ferrum/plots")

#: The one encoding channel this invariant governs. ``fill``/``stroke`` are
#: deliberately excluded: in these modules they are mark *paint* kwargs
#: (literal colors and booleans), not data-field encodings, and they are
#: already gated at construction time by ``MarkBase.__init__``'s literal-color
#: validation. Widening this set would conflate two different contracts.
CHANNEL = "color"

#: Call targets that constitute an explicit type declaration. A binding whose
#: value is a call to one of these has stated its scale type (or delegated
#: that statement to a helper that does), so it satisfies the invariant.
TYPED_CONSTRUCTORS = frozenset(
    {
        "nominal_color_channel",
        "Color",
        "_Color",
        "shap_beeswarm_color_channel",
    }
)


class Binding(NamedTuple):
    """One ``color``-channel binding found by the scan."""

    path: str
    lineno: int
    expr: str

    @property
    def key(self) -> tuple[str, str]:
        """Allowlist key: file plus source text, deliberately not line number.

        Keying on the expression rather than the line keeps the allowlist
        stable when unrelated edits move code, so a rebased branch does not
        produce spurious failures — while still failing when the *expression*
        changes, which is when the disposition genuinely needs re-checking.
        """
        return (self.path, self.expr)


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent


def _color_bindings(tree: ast.AST) -> Iterator[tuple[ast.expr, int]]:
    """Yield ``(value_node, lineno)`` for every ``color`` binding in *tree*.

    Covers the three shapes the codebase actually uses to build an encoding:
    a dict literal (``{"color": v}``), a subscript assignment
    (``enc["color"] = v``), and a call keyword (``.encode(color=v)``).
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == CHANNEL:
                    yield value, key.lineno
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == CHANNEL
                ):
                    yield node.value, node.lineno
        elif isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg == CHANNEL:
                    yield keyword.value, node.lineno


def _is_typed_expr(node: ast.expr) -> bool:
    """True when *node* declares a scale type on every path it can take.

    A call to one of :data:`TYPED_CONSTRUCTORS` declares a type. A conditional
    declares one only when *both* branches do — the shape several desugars use
    to pick a colormap (``Color(f, scheme=cmap) if cmap else Color(f)``). A
    conditional with one raw branch is deliberately not typed, which is what
    keeps ``relplot``'s scatter carve-out visible in :data:`ALLOWED` instead of
    passing silently.
    """
    if isinstance(node, ast.IfExp):
        return _is_typed_expr(node.body) and _is_typed_expr(node.orelse)
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
    return name in TYPED_CONSTRUCTORS


def _is_literal_paint(node: ast.expr) -> bool:
    """True when *node* is a constant that names a color, not a data field.

    Delegates the whole judgment to ferrum's single parser and single
    sentinel predicate — see this module's docstring.
    """
    if not isinstance(node, ast.Constant):
        return False
    if node.value is None:
        return True
    if not isinstance(node.value, str):
        return False
    if _is_paint_sentinel(node.value):
        return True
    try:
        to_hex(node.value)
    except ValueError:
        return False
    return True


def _typed_local_names(tree: ast.AST) -> set[str]:
    """Names assigned anywhere from a :data:`TYPED_CONSTRUCTORS` call.

    Figure builders that bind one hue onto several layers type it once
    (``hue_ch = nominal_color_channel(hue)``) and reuse the channel object, so
    that both layers cannot resolve different scale types. Treating those
    reuses as typed keeps the allowlist to genuine judgment calls instead of
    filling it with local variable names.

    Module-wide rather than per-scope on purpose: a name that is *ever* bound
    to a raw field elsewhere is still caught, because this only whitelists
    names that have a typed assignment, and shadowing a typed channel name
    with a raw field would be caught by review as an obvious misnaming.
    """
    typed: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _is_typed_expr(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    typed.add(target.id)
    return typed


def _scan() -> list[Binding]:
    """Return every ``color`` binding that does not declare its type."""
    root = _repo_root()
    found: list[Binding] = []
    for scanned in SCANNED_ROOTS:
        for path in sorted((root / scanned).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            typed_names = _typed_local_names(tree)
            for value, lineno in _color_bindings(tree):
                if _is_typed_expr(value) or _is_literal_paint(value):
                    continue
                if isinstance(value, ast.Name) and value.id in typed_names:
                    continue
                found.append(
                    Binding(
                        path=str(path.relative_to(root)),
                        lineno=lineno,
                        expr=ast.unparse(value),
                    )
                )
    return found


#: Adjudicated exceptions: ``(path, expression) -> rationale``.
#:
#: Every entry is a binding a reviewer has judged correct as written. Adding
#: an entry is the deliberate act this guard exists to force; the rationale is
#: the review record. Entries fall into four kinds, and the rationale says
#: which: a field that is *continuous by design*, a field whose dtype is
#: *guaranteed Utf8 by construction*, the *point-mark carve-out*, or a value
#: that is *not an encoding binding at all*.
ALLOWED: dict[tuple[str, str], str] = {
    # -- Continuous by design: the field is a measured quantity, not a
    #    discriminator, so a Continuous color scale is the correct reading.
    (
        "src/ferrum/marks/diagnostic/_clustering.py",
        "color_field",
    ): "desugar_decision_boundary's color_field defaults to 'z', the grid cell's "
    "prediction value (class index or probability) — genuinely continuous. The "
    "same expression also covers desugar_intercluster_distance's point layer, "
    "which is the point-mark carve-out (sole color consumer; its sibling layer "
    "is an unencoded text label).",
    (
        "src/ferrum/marks/heavy_stat.py",
        "'level_value'",
    ): "Filled-contour isoband density level — a continuous quantity ramped by "
    "the sequential colormap, which is the intended reading.",
    (
        "src/ferrum/marks/heavy_stat.py",
        "'value'",
    ): "Heatmap cell value — the measured quantity the heatmap exists to show.",
    (
        "src/ferrum/plots/matrix.py",
        "'value'",
    ): "Heatmap cell value (figure-function side of the same binding).",
    (
        "src/ferrum/plots/matrix.py",
        "'count'",
    ): "jointplot(kind='hist') 2-D bin count — a continuous aggregate.",
    (
        "src/ferrum/plots/classification.py",
        "'value'",
    ): "Confusion-matrix / report-heatmap cell value — a continuous count or rate.",
    (
        "src/ferrum/plots/ranking.py",
        "'scatter_z'",
    ): "rank2d correlation coefficient — continuous on [-1, 1].",
    (
        "src/ferrum/marks/diagnostic/_explanation.py",
        "'shap_sign'",
    ): "SHAP signed contribution driving a diverging ramp — continuous by design.",
    # -- Utf8 by construction: the producing code guarantees a string dtype,
    #    so inference cannot reach Continuous. Kept as literal field names
    #    because the guarantee is local and visible at the producing site.
    (
        "src/ferrum/marks/diagnostic/_selection.py",
        "'split'",
    ): "cv_scores 'split' holds the literal strings 'train'/'test' (relabelled to "
    "'Training Score'/'Cross-Validation Score' by the figure builders), never a "
    "fold index.",
    (
        "src/ferrum/plots/model_selection.py",
        "'split'",
    ): "Same 'split' column as above, Utf8 by construction "
    "(plots/model_selection.py replaces it with display strings before encoding).",
    (
        "src/ferrum/marks/diagnostic/_classification.py",
        "'metric'",
    ): "discrimination_threshold's 'metric' holds metric names "
    "('precision'/'recall'/'f1'/'queue_rate') — Utf8 by construction.",
    (
        "src/ferrum/plots/classification.py",
        "'y'",
    ): "class_balance builds its own frame and casts the class column to Utf8 "
    "(`series.cast(pl.Utf8, strict=False)`) before encoding it.",
    (
        "src/ferrum/plots/clustering.py",
        "'label:N'",
    ): "Already typed Nominal, via the ':N' shorthand suffix rather than a Color(...) call.",
    # -- Point-mark carve-out: a `point` mark that is the sole consumer of the
    #    discriminator in its chart may legitimately read a numeric field as a
    #    gradient. See nominal_color_channel's docstring for why "sole
    #    consumer" is load-bearing.
    (
        "src/ferrum/plots/distribution.py",
        "hue if kind == 'scatter' else nominal_color_channel(hue)",
    ): "relplot(kind='scatter') is a single unpaired point mark, so a numeric "
    "hue keeps rendering a gradient (seaborn parity); kind='line' groups rows "
    "into one polyline per level and is typed.",
    (
        "src/ferrum/marks/diagnostic/_regression.py",
        "color_field",
    ): "desugar_residuals and desugar_prediction_error both bind color_field on a "
    "`point` layer only; their sibling layers (rule / ribbon / identity line) "
    "carry no color encoding, so the point mark is the sole consumer.",
    # -- Not an encoding binding: the scan's three syntactic shapes also match
    #    a few dicts and kwargs that are not encodings at all.
    (
        "src/ferrum/plots/matrix.py",
        "'shared'",
    ): "Not an encoding: a composite scale-resolve directive "
    "(`resolve={'color': 'shared'}`) unifying the color domain across panels.",
    (
        "src/ferrum/plots/matrix.py",
        "encode_kwargs.pop('color')",
    ): "Not a ferrum-authored binding: a user-supplied color channel forwarded "
    "verbatim from heatmap(**encode_kwargs). Typing it would overrule the caller.",
    (
        "src/ferrum/plots/regression.py",
        "color",
    ): "residplot forwards its own already-resolved color argument; typing it "
    "here would overrule a caller-supplied channel object.",
    (
        "src/ferrum/plots/regression.py",
        "'_label'",
    ): "Computed two-level annotation label column, written by the builder "
    "immediately above as Utf8 literals.",
    (
        "src/ferrum/marks/diagnostic/_classification.py",
        "'_iso_f'",
    ): "Iso-F level column on mark_pr's iso_line layer. Not a caller-named "
    "discriminator, and the layer sets an explicit literal `stroke` in its own "
    "mark_kwargs, so this channel does not drive the curves' paint. The "
    "separate observation that all iso-F curves render one identical colour is "
    "that hardcoded stroke, not this binding — tracked as its own follow-up.",
}


def test_every_color_binding_declares_its_scale_type() -> None:
    """Fail when a ``color`` binding neither declares a type nor is allowlisted.

    This is the guard described in the module docstring. If it fails, the fix
    is almost always to wrap the field in ``nominal_color_channel(...)``; wrap
    it in an explicit ``Color(..., type_=...)`` when the field is genuinely
    continuous, or add an :data:`ALLOWED` entry with a rationale when the
    binding is correct as written for some other reason.
    """
    undeclared = [b for b in _scan() if b.key not in ALLOWED]
    assert not undeclared, (
        "color bindings that neither declare a scale type nor "
        "are allowlisted:\n"
        + "\n".join(f"  {b.path}:{b.lineno}: color = {b.expr}" for b in undeclared)
    )


def test_allowlist_has_no_stale_entries() -> None:
    """Fail when an :data:`ALLOWED` entry no longer matches any binding.

    A stale exception is how an allowlist rots into a rubber stamp: the entry
    outlives the code it excused and then silently covers some future binding
    that happens to reuse the expression. Deleting the code must delete the
    exception.
    """
    live = {b.key for b in _scan()}
    stale = sorted(key for key in ALLOWED if key not in live)
    assert not stale, "ALLOWED entries matching no binding — delete them:\n" + "\n".join(
        f"  {path}: {expr}" for path, expr in stale
    )


@pytest.mark.parametrize(
    "rationale_key",
    sorted(ALLOWED, key=lambda k: (k[0], k[1])),
    ids=lambda k: f"{pathlib.Path(k[0]).name}::{k[1]}",
)
def test_every_allowlist_entry_carries_a_rationale(rationale_key: tuple[str, str]) -> None:
    """Every exception states why, in prose, at a length that is actually an argument.

    The threshold is deliberately crude — it cannot judge whether a rationale
    is *good*. What it prevents is the failure mode that makes allowlists
    useless: an entry added under deadline with ``""`` or ``"ok"`` as its
    justification, which reads as adjudicated without ever having been.
    """
    rationale = ALLOWED[rationale_key]
    assert len(rationale.split()) >= 8, (
        f"allowlist entry {rationale_key} needs a real rationale, got {rationale!r}"
    )
