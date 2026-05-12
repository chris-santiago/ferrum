"""Two-line title value class — spec §3.19, Schwabish SB1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Title:
    """Chart title with optional subtitle.

    Accepted everywhere a title string is accepted: ``Chart(title=...)``,
    ``Chart.properties(title=...)``, ``HConcat/VConcat/Layer(title=...)``.
    Passing a bare string is equivalent to ``Title(text=string)``.

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
