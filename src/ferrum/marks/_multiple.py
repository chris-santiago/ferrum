"""One owner for what a ``multiple=`` mode means.

A grouped distribution mark answers two questions from one word.  Where are
the groups drawn relative to each other (the position adjustment), and are
they binned/gridded against one global extent or each against its own
(``shared_extent``)?  The answers are not independent: bars can only be
stacked when every group's bars land on the same edges, so stacking without
a shared extent is not a weaker stack, it is a side-by-side render wearing a
stack's name.  The same is true of dodging, one word over -- a slot is a
subdivision of a band, and without shared edges there is no band the groups
both belong to -- so ``"layer"``, which asks nothing of its neighbours, is
the only mode that may bin per group.

``multiple_semantics`` decides both together and is the only place either is
decided.  GH #194 is what happens otherwise: ``desugar_histogram`` derived
both halves from ``multiple=``, ``displot`` re-derived only the position and
never forwarded the mode, so its ``Bin`` kept ``shared_extent=False`` and
``multiple="stack"`` drew two groups' bars side by side, at different widths,
sharing no bin edge.

The mode vocabulary itself (:data:`MULTIPLE_MODES`) lives here too, so the
set of accepted words and the meaning of each word cannot drift apart.
Public entries (``mark_histogram``, ``mark_density``, ``displot``) validate
``multiple=`` in their own voice before reaching this function; the
``ValueError`` below is the unreachable-today guard that keeps a newly added
mode from silently mapping to "layer".
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional

if TYPE_CHECKING:  # pragma: no cover - import kept out of the runtime path
    # Deferred: `ferrum.position` is imported inside `multiple_semantics` so
    # this leaf module stays importable from anywhere in the package.
    from ferrum.position import Dodge, Identity, Stack

MULTIPLE_MODES = ("dodge", "fill", "layer", "stack")


class MultipleSemantics(NamedTuple):
    """The pair of decisions a ``multiple=`` mode makes.

    Attributes
    ----------
    position : Dodge, Stack, Identity or None
        The position adjustment the mode requires, or ``None`` for "no
        adjustment declared" (``multiple="layer"``).  A caller that wants an
        explicit no-op in the spec substitutes its own ``Identity()``, which
        is why that type is in the annotation even though this module never
        builds one.
    shared_extent : bool
        Whether the mark's binning/gridding transform must compute one
        global extent across all groups before splitting them.
    """

    position: "Dodge | Stack | Identity | None"
    shared_extent: bool


def multiple_semantics(
    multiple: str,
    *,
    by: Optional[str] = None,
    value_axis: Optional[str] = None,
) -> MultipleSemantics:
    """Return the position adjustment and extent mode ``multiple`` implies.

    Parameters
    ----------
    multiple : {"layer", "stack", "fill", "dodge"}
        The mode.
    by : str or None, default None
        Column the position adjustment groups by.  ``None`` leaves the
        adjustment to resolve its own default (the color channel) at render
        time.
    value_axis : {"x", "y"} or None, default None
        Explicit value axis for the ``Stack`` adjustments, needed when a
        mark remaps its count/density column onto ``x`` without setting
        ``CoordFlip`` (GH #77, ``orientation="horizontal"``).

    Returns
    -------
    MultipleSemantics
        ``(position, shared_extent)``.

    Raises
    ------
    ValueError
        If ``multiple`` is not one of :data:`MULTIPLE_MODES`.

    Examples
    --------
    >>> multiple_semantics("layer")
    MultipleSemantics(position=None, shared_extent=False)
    >>> multiple_semantics("stack", by="g").shared_extent
    True
    """
    from ferrum.position import Dodge, Stack

    if multiple == "layer":
        return MultipleSemantics(None, False)
    if multiple == "dodge":
        # The band a dodged bar takes its slot in IS the bin, so the bins
        # have to be the same ones for every group before a slot means
        # anything. Without a shared extent each group bins over its own
        # range: measured on two 300-row groups, `multiple="dodge"` drew one
        # group as 9 bars of width 35.37 px and the other as 10 of width
        # 21.05 px, sharing not one x position -- #194's "two histograms
        # standing side by side wearing a stack's name", under a different
        # word. Shared edges are what make the slots adjacent instead of
        # merely overlapping.
        return MultipleSemantics(Dodge(by=by), True)
    if multiple == "stack":
        return MultipleSemantics(Stack(by=by, offset="zero", value_axis=value_axis), True)
    if multiple == "fill":
        return MultipleSemantics(Stack(by=by, offset="normalize", value_axis=value_axis), True)
    raise ValueError(f"unknown multiple {multiple!r}; expected one of {MULTIPLE_MODES}")
