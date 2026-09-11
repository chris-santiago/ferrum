"""Two-line title value class — spec §3.19, Schwabish SB1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Title:
    """Chart title with optional subtitle.

    Accepted on a leaf ``Chart``'s ``title=`` -- ``Chart(title=...)`` /
    ``Chart.properties(title=...)`` -- with every field here (subtitle,
    anchor, offset, font, color) honored: a leaf's ``ChartSpec.title`` is a
    full ``TitleSpec``.

    On a composition's figure-level ``title=``/``subtitle=``/``caption=``
    (``HConcatChart``/``VConcatChart``/``ConcatChart``/``JointChart``/
    ``RepeatChart``/``ClusterMapChart``/``LayerChart``), the composite-tree
    render entry's wire chrome fields are bare strings with no
    subtitle/styling slot of their own, so a ``Title`` passed to any of the
    three is normalized at ``.properties()``: a text-only ``Title`` (no
    field but ``.text`` set away from default) reduces to its ``.text``;
    ``title=fm.Title(text=..., subtitle=...)`` on the six views with a
    separate figure-subtitle slot ROUTES the ``Title``'s own ``subtitle``
    into that slot -- unless an explicit ``subtitle=`` is *also* given in
    the same call, which raises ``ValueError`` naming both spellings; and a
    ``Title`` carrying any other field this composite genuinely cannot
    express (``anchor``, ``offset``, ``font_size``, ``font_weight``,
    ``color``, ``subtitle_font_size``, ``subtitle_color`` on any of the
    three keys; ``subtitle`` itself on ``subtitle=``/``caption=``, or on
    ``LayerChart``'s ``title=``, which has no separate subtitle slot) raises
    ``ValueError`` at ``.properties()`` time rather than silently dropping
    it (2026-09-11, batch-H close cycle 3, made lossless and widened to all
    three chrome keys in cycle 4; see ``ferrum-spec.md`` §3.16's dated
    notes). Passing a bare string is equivalent to ``Title(text=string)``
    everywhere.

    Parameters
    ----------
    text : str
        Primary title text.
    subtitle : str, optional
        Secondary line below the title; ``None`` suppresses the second line.
    anchor : {"start", "middle", "end"}, default "start"
        Horizontal anchor.
    offset : float, optional
        Pixel offset from plot area; defaults to the theme's ``title_offset``.
    font_size : float, optional
        Title font size in points.
    font_weight : str, optional
        Title font weight (e.g. ``"600"``, ``"bold"``).
    color : str, optional
        Title color as a CSS color string.
    subtitle_font_size : float, optional
        Subtitle font size; defaults to ``font_size * 0.85``.
    subtitle_color : str, optional
        Subtitle color; defaults to the theme's ``label_color``.

    Examples
    --------
    >>> import ferrum as fm
    >>> title = fm.Title("My Chart", subtitle="subtitle text", anchor="start")
    >>> fm.Chart(df).mark_point().encode(x="x", y="y").properties(title=title)
    """

    text: str
    subtitle: Optional[str] = None
    anchor: str = "start"
    offset: Optional[float] = None
    font_size: Optional[float] = None
    font_weight: Optional[str] = None
    color: Optional[str] = None
    subtitle_font_size: Optional[float] = None
    subtitle_color: Optional[str] = None

    def to_spec_dict(self) -> dict:
        """Serialize to the dict shape the Rust binding expects.

        Only emits non-default fields so the JSON sent to Rust stays
        minimal and forward-compatible.
        """
        out: dict = {"text": self.text}
        if self.subtitle is not None:
            out["subtitle"] = self.subtitle
        if self.anchor != "start":
            out["anchor"] = self.anchor
        for field_name in (
            "offset",
            "font_size",
            "font_weight",
            "color",
            "subtitle_font_size",
            "subtitle_color",
        ):
            value = getattr(self, field_name)
            if value is not None:
                out[field_name] = value
        return out
