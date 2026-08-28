"""Mechanical oracle queries over the committed golden SVG corpus.

Provenance: adopted from ``archive/api-contract-audit`` 67098541 under full
review, 2026-08-28. The archive branch is untrusted; every line below was read
against the SVG 1.1 specification and the geometry claims were re-derived by
independent tests in ``tests/test_svg_corpus.py``. Sections rewritten during
adoption are marked ``REVIEW 2026-08-28`` at the site of the change.

The goldens are a queryable dataset, not only a set of byte-equality baselines.
A golden diff can tell you that output moved; it cannot tell you that output is
*wrong*. These queries name the FILE and the DEFECT.

Each query has a mechanical oracle, i.e. a property of well-formed chart output
that no correct renderer should ever violate:

``off_viewport``
    A drawable element whose geometry lies entirely outside the root viewBox.
    It is in the DOM, it costs bytes, and nobody can see it.
``negative_size``
    ``width``/``height``/``r`` below zero. Per SVG 1.1 a negative value is an
    error; renderers disagree on whether to drop the element or clamp it.
``nan_inf_attr``
    A non-finite number in a *numeric attribute value*. Anchored on parsed
    attribute values, never on a substring scan of the file: the embedded
    base64 font contains the letters ``nan`` and ``inf`` many times over, and a
    text scan reports a false positive on every golden in the repo.
``zero_size_filled``
    A filled ``rect``/``circle`` with zero extent: a mark that was computed,
    emitted, and renders as nothing.
``duplicate_id``
    Two elements sharing one ``id``. ``url(#id)`` resolves to the FIRST match,
    so a duplicate silently cross-wires gradients and clip paths. The output
    stays well-formed, renders, and is wrong.
``outside_clip``
    An element whose geometry falls entirely outside the clip rectangle it
    references. A mark clipped out of existence is present in the DOM and
    invisible, which reads as "the feature does nothing" one layer up.
``duplicate_drawable``
    Two chrome elements (``line``/``text``) painted at identical resolved
    coordinates with identical paint and text. Coincident chrome is the
    signature of a composition that drew one panel's axes twice; it renders
    identically to the correct output, so byte-equality can never catch it.
    Data marks are excluded by design: coincident data is legal.

Modelling limits, stated so a clean run is not over-read:

* ``<text>`` is reduced to its anchor point. Glyph advance and ``dx``/``dy``
  (which the corpus writes in ``em`` units) are outside the model, so the
  ``off_viewport`` bound for text is the anchor, not the inked box.
* A ``clipPath`` is modelled only when it is exactly one ``rect``, carrying no
  transform of its own and none between it and the ``clipPath``, under the
  default ``clipPathUnits="userSpaceOnUse"``. Every other clip is skipped
  rather than guessed at: a wrong clip window manufactures ``outside_clip``
  findings indistinguishable from real ones.
* A clipped element is judged against its clip's *rectangle*, so a clip whose
  effective region is a rotated rectangle is over-approximated by that
  rectangle's axis-aligned bounds — conservative, never over-reporting.
* Presentation attributes are read from the element itself. ``fill``/``stroke``
  inherited from an ancestor ``<g>`` or set via ``style=`` are not resolved,
  so ``zero_size_filled`` and ``duplicate_drawable`` under-report rather than
  over-report. (No golden uses either form today.)
* Path control points bound the curve, so a path entirely outside a rectangle
  has all its control points outside it too. The converse does not hold, which
  makes ``off_viewport`` conservative in the right direction for paths.

Anything this module cannot model raises `SvgCorpusError`. A query that
silently degrades to identity or to a truncated parse reports a clean zero for
a broken corpus, which is the exact failure the harness exists to prevent.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence

from tests import _snapshots

_REPO_ROOT = Path(__file__).resolve().parent.parent


class SvgCorpusError(ValueError):
    """The corpus model met SVG it cannot represent faithfully."""


def golden_paths() -> list[Path]:
    """Every committed golden SVG, deduplicated, in a stable order.

    REVIEW 2026-08-28: the archive redefined the two golden roots and its own
    ``find_goldens``. ``tests/_snapshots.py`` already owns both, so this
    delegates rather than forking a second copy that can drift.
    """
    return sorted(set(_snapshots.find_goldens(*_snapshots.iter_default_roots())))


# --------------------------------------------------------------------------
# lexical
# --------------------------------------------------------------------------

#: Attributes whose values are numeric (scalars, number lists, path data and
#: transform lists alike). REVIEW 2026-08-28: the archive kept two frozensets
#: plus a special case for ``d``; there is one consumer, so there is one set.
#: ``transform`` is added — a non-finite there is as fatal as one in ``x``.
_NUMERIC_VALUE_ATTRS = frozenset(
    "x y x1 y1 x2 y2 cx cy r rx ry width height stroke-width font-size offset "
    "opacity fill-opacity stroke-opacity dx dy "
    "points viewBox stroke-dasharray d transform".split()
)

_DRAWABLE = frozenset({"rect", "circle", "ellipse", "line", "polyline", "polygon", "path", "text"})

#: Elements whose text content is part of their identity. Everything else keeps
#: an empty ``Node.text``: a ``<g>``'s subtree text is not that element's
#: identity, and ``<style>`` holds a multi-hundred-kilobyte base64 font body.
_TEXT_CONTENT_TAGS = frozenset({"text", "tspan", "title", "desc"})

#: Tags the ``duplicate_drawable`` oracle policies. Chrome only: coincident
#: data marks (circle/rect/path/polyline) are legal output.
_CHROME_TAGS = frozenset({"line", "text"})

#: `render/scene_build.rs::BREAK_HIDDEN` (Rust layer). A break-axis chart hides a
#: mark that falls inside the gap by moving it to this coordinate rather than
#: dropping it, because dropping it would desynchronise `data_indices`. Every
#: legitimate off-viewport coordinate in the corpus is exactly this value, so the
#: sentinel is a PROPERTY the queries can recognise — not a file allow-list that
#: silently absorbs the next real defect.
BREAK_HIDDEN = -99999.0

_NUM_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
_NONFINITE_RE = re.compile(r"(?i)(?<![0-9A-Za-z])(nan|[-+]?inf(?:inity)?)(?![0-9A-Za-z])")
_SEPARATOR_RE = re.compile(r"[\s,]*")
_TRANSFORM_RE = re.compile(r"([A-Za-z]\w*)\s*\(([^)]*)\)")

_PATH_COMMANDS = "MmZzLlHhVvCcSsQqTtAa"
#: Coordinate pairs consumed by one instance of each pair-taking command.
_PATH_PAIR_ARITY = {"M": 1, "L": 1, "T": 1, "S": 2, "Q": 2, "C": 3}


def _numbers(text: str) -> list[float]:
    return [float(v) for v in _NUM_RE.findall(text or "")]


def _as_float(text: str | None, default: float = 0.0) -> float:
    if text is None:
        return default
    m = _NUM_RE.match(text.strip())
    if not m:
        return default
    return float(m.group())


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Matrix:
    """A 2-D affine transform ``(a c e / b d f)``, SVG's column order."""

    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    d: float = 1.0
    e: float = 0.0
    f: float = 0.0

    def then(self, other: Matrix) -> Matrix:
        """Return ``self`` followed by ``other`` applied to the result."""
        # other * self, in SVG's parent-to-child accumulation order.
        return Matrix(
            a=self.a * other.a + self.b * other.c,
            b=self.a * other.b + self.b * other.d,
            c=self.c * other.a + self.d * other.c,
            d=self.c * other.b + self.d * other.d,
            e=self.e * other.a + self.f * other.c + other.e,
            f=self.e * other.b + self.f * other.d + other.f,
        )

    def apply(self, x: float, y: float) -> tuple[float, float]:
        """Map a point from this matrix's source space to its target space."""
        return (self.a * x + self.c * y + self.e, self.b * x + self.d * y + self.f)


IDENTITY = Matrix()


def parse_transform(text: str) -> Matrix:
    """Parse an SVG ``transform`` attribute into a single matrix.

    REVIEW 2026-08-28: the archive silently skipped any function it did not
    know (``skewX``/``skewY`` among them), degrading an unmodelled transform to
    identity and turning a real off-viewport finding into a clean zero.
    ``skew*`` is implemented; anything else raises.
    """
    m = IDENTITY
    for name, args in _TRANSFORM_RE.findall(text or ""):
        vals = _numbers(args)
        if name == "translate":
            tx = vals[0] if vals else 0.0
            ty = vals[1] if len(vals) > 1 else 0.0
            step = Matrix(e=tx, f=ty)
        elif name == "scale":
            sx = vals[0] if vals else 1.0
            sy = vals[1] if len(vals) > 1 else sx
            step = Matrix(a=sx, d=sy)
        elif name == "rotate":
            ang = math.radians(vals[0] if vals else 0.0)
            cos, sin = math.cos(ang), math.sin(ang)
            step = Matrix(a=cos, b=sin, c=-sin, d=cos)
            if len(vals) >= 3:
                cx, cy = vals[1], vals[2]
                step = Matrix(e=-cx, f=-cy).then(step).then(Matrix(e=cx, f=cy))
        elif name == "skewX":
            step = Matrix(c=math.tan(math.radians(vals[0] if vals else 0.0)))
        elif name == "skewY":
            step = Matrix(b=math.tan(math.radians(vals[0] if vals else 0.0)))
        elif name == "matrix":
            if len(vals) != 6:
                raise SvgCorpusError(f"matrix() takes 6 values, got {len(vals)}: {text!r}")
            step = Matrix(*vals)
        else:
            raise SvgCorpusError(f"unmodelled transform function {name!r} in {text!r}")
        # The first function listed applies LAST to a point, so each new step
        # composes underneath what has accumulated so far.
        m = step.then(m)
    return m


def _nested_viewport_matrix(attrib: dict[str, str]) -> Matrix:
    """Transform established by a nested ``<svg>`` element.

    A nested ``<svg x y width height viewBox>`` translates to ``(x, y)`` and
    then scales its ``viewBox`` to fit ``width``/``height``. Ignoring the scale
    reports every inset mark as off-viewport, which is a false positive with
    exactly the shape of a real finding.
    """
    x, y = _as_float(attrib.get("x")), _as_float(attrib.get("y"))
    m = Matrix(e=x, f=y)
    vb = _numbers(attrib.get("viewBox", ""))
    w, h = _as_float(attrib.get("width"), 0.0), _as_float(attrib.get("height"), 0.0)
    if len(vb) == 4 and vb[2] > 0 and vb[3] > 0 and w > 0 and h > 0:
        # preserveAspectRatio defaults to xMidYMid meet: uniform scale, centred.
        s = min(w / vb[2], h / vb[3])
        tx = (w - vb[2] * s) / 2.0
        ty = (h - vb[3] * s) / 2.0
        m = Matrix(e=-vb[0], f=-vb[1]).then(Matrix(a=s, d=s)).then(Matrix(e=tx, f=ty)).then(m)
    return m


class _PathCursor:
    """A character cursor over SVG path data.

    REVIEW 2026-08-28: the archive pre-tokenised ``d`` with one number regex,
    which merges an elliptical arc's two single-digit flags into the number
    that follows them (``a1 1 0 011 1`` lexes ``011`` as 11). Arc flags are
    only recognisable from position, so the scan is position-driven.
    """

    __slots__ = ("_d", "_pos")

    def __init__(self, d: str) -> None:
        self._d = d
        self._pos = 0

    @property
    def position(self) -> int:
        """Offset of the next unconsumed character."""
        return self._pos

    def _skip_separators(self) -> None:
        self._pos = _SEPARATOR_RE.match(self._d, self._pos).end()

    def at_end(self) -> bool:
        """True once only separators remain."""
        self._skip_separators()
        return self._pos >= len(self._d)

    def take_command(self) -> str | None:
        """Consume and return the next command letter, or None if a number is next."""
        self._skip_separators()
        if self._pos < len(self._d) and self._d[self._pos] in _PATH_COMMANDS:
            ch = self._d[self._pos]
            self._pos += 1
            return ch
        return None

    def take_number(self) -> float:
        """Consume and return the next number."""
        self._skip_separators()
        m = _NUM_RE.match(self._d, self._pos)
        if m is None:
            raise SvgCorpusError(f"expected a number at offset {self._pos} in {self._d[:80]!r}")
        self._pos = m.end()
        return float(m.group())

    def take_flag(self) -> float:
        """Consume and return the next arc flag (a bare ``0`` or ``1``)."""
        self._skip_separators()
        if self._pos < len(self._d) and self._d[self._pos] in "01":
            ch = self._d[self._pos]
            self._pos += 1
            return float(ch)
        raise SvgCorpusError(f"expected an arc flag at offset {self._pos} in {self._d[:80]!r}")


def path_points(d: str) -> list[tuple[float, float]]:
    """Control/endpoint coordinates of a path ``d``, in the path's own space.

    Control points bound the curve, so a path entirely outside a rectangle has
    all of its control points outside it too. The converse is not true, which
    makes this conservative in the right direction for "is it off-screen".
    """
    cursor = _PathCursor(d or "")
    pts: list[tuple[float, float]] = []
    cur = (0.0, 0.0)
    start = (0.0, 0.0)
    cmd: str | None = None
    consumed_through = -1
    while not cursor.at_end():
        # Termination invariant, and deliberately unreachable today: every
        # branch below either consumes a command letter, consumes a number, or
        # raises — `Z` is the one command that takes no coordinates, which is
        # why it rejects a trailing number rather than looping back. The guard
        # stays because the failure it converts has no diagnostic: inside
        # `pytest -n auto` a spin produces no message, no traceback and no
        # failing test name, so the NEXT non-consuming branch would be found by
        # a timeout, not by a test. Mutating it away is an equivalent mutant.
        if cursor.position == consumed_through:
            raise SvgCorpusError(f"path parser made no progress at offset {cursor.position}")
        consumed_through = cursor.position
        taken = cursor.take_command()
        if taken is not None:
            cmd = taken
        elif cmd is None:
            raise SvgCorpusError(f"path data begins with a number: {d[:80]!r}")
        elif cmd in "Mm":
            # A moveto's extra coordinate pairs are an implicit lineto.
            cmd = "L" if cmd == "M" else "l"
        upper = cmd.upper()
        rel = cmd.islower()
        if upper == "Z":
            if taken is None:
                raise SvgCorpusError(
                    f"closepath takes no arguments; a number follows Z in {d[:80]!r}"
                )
            cur = start
            continue
        if upper in ("H", "V"):
            v = cursor.take_number()
            if upper == "H":
                cur = (cur[0] + v if rel else v, cur[1])
            else:
                cur = (cur[0], cur[1] + v if rel else v)
            pts.append(cur)
        elif upper == "A":
            cursor.take_number()  # rx
            cursor.take_number()  # ry
            cursor.take_number()  # x-axis-rotation
            cursor.take_flag()  # large-arc-flag
            cursor.take_flag()  # sweep-flag
            x, y = cursor.take_number(), cursor.take_number()
            cur = (cur[0] + x, cur[1] + y) if rel else (x, y)
            pts.append(cur)
        else:
            # Every relative coordinate of one command instance is measured
            # from the current point as it stood BEFORE the command.
            origin = cur
            for _ in range(_PATH_PAIR_ARITY[upper]):
                x, y = cursor.take_number(), cursor.take_number()
                cur = (origin[0] + x, origin[1] + y) if rel else (x, y)
                pts.append(cur)
            if upper == "M":
                start = cur
    return pts


def local_points(tag: str, attrib: dict[str, str]) -> list[tuple[float, float]]:
    """Bounding reference points of one element, in its own coordinate space."""

    def g(key: str, default: float = 0.0) -> float:
        return _as_float(attrib.get(key), default)

    if tag == "rect":
        x, y, w, h = g("x"), g("y"), g("width"), g("height")
        return [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
    if tag in ("circle", "ellipse"):
        cx, cy = g("cx"), g("cy")
        rx = g("r", g("rx"))
        ry = g("r", g("ry"))
        return [(cx - rx, cy - ry), (cx + rx, cy + ry)]
    if tag == "line":
        return [(g("x1"), g("y1")), (g("x2"), g("y2"))]
    if tag in ("polyline", "polygon"):
        nums = _numbers(attrib.get("points", ""))
        return list(zip(nums[0::2], nums[1::2]))
    if tag == "path":
        return path_points(attrib.get("d", ""))
    if tag == "text":
        return [(g("x"), g("y"))]
    return []


# --------------------------------------------------------------------------
# document model
# --------------------------------------------------------------------------


class Rect(NamedTuple):
    """An axis-aligned rectangle in resolved user space, ``(x0, y0)-(x1, y1)``."""

    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def from_origin_size(cls, x: float, y: float, width: float, height: float) -> Rect:
        """Build from an SVG ``x``/``y``/``width``/``height`` quadruple."""
        return cls(x, y, x + width, y + height)

    def encloses_none_of(self, points: Sequence[tuple[float, float]], *, tol: float) -> bool:
        """True when every point lies on one side of this rectangle.

        The shared "wholly outside" test behind ``off_viewport`` and
        ``outside_clip``: both ask whether a bounding point set misses a
        rectangle entirely, and a copy of the comparison in each was one
        rectangle's worth of drift waiting to happen.
        """
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return (
            max(xs) < self.x0 - tol
            or min(xs) > self.x1 + tol
            or max(ys) < self.y0 - tol
            or min(ys) > self.y1 + tol
        )

    def __str__(self) -> str:
        """Render as ``x0,y0..x1,y1`` for finding messages."""
        return f"{self.x0:g},{self.y0:g}..{self.x1:g},{self.y1:g}"


@dataclass(frozen=True)
class Node:
    """One element of a golden, with its accumulated transform resolved."""

    #: Position in document order. REVIEW 2026-08-28: the archive recovered
    #: this with ``doc.nodes.index(node)``, which compares Nodes by VALUE and
    #: so resolves two identical siblings to the first — the precise shape the
    #: ``duplicate_drawable`` oracle exists to find.
    index: int
    tag: str
    attrib: dict[str, str]
    ctm: Matrix
    ancestors: tuple[str, ...]
    in_defs: bool
    #: Text content, for text-bearing tags only (see ``_TEXT_CONTENT_TAGS``).
    text: str = ""

    @property
    def points(self) -> list[tuple[float, float]]:
        """The element's reference points, resolved into root user space."""
        return [self.ctm.apply(x, y) for x, y in local_points(self.tag, self.attrib)]

    @property
    def at_break_sentinel(self) -> bool:
        """True when any of the element's own coordinates is `BREAK_HIDDEN`."""
        coords: list[float] = []
        for key in ("x", "y", "x1", "y1", "x2", "y2", "cx", "cy"):
            if key in self.attrib:
                coords.append(_as_float(self.attrib[key]))
        for key in ("points", "d"):
            if key in self.attrib:
                coords.extend(_numbers(self.attrib[key]))
        return any(math.isclose(v, BREAK_HIDDEN) for v in coords)

    @property
    def clip_ref(self) -> str | None:
        """The id this element's ``clip-path`` references, if any."""
        ref = self.attrib.get("clip-path")
        if not ref:
            return None
        m = re.match(r"url\(#([^)]+)\)", ref.strip())
        return m.group(1) if m else None


@dataclass(frozen=True)
class Document:
    """A golden SVG flattened into transform-resolved nodes."""

    path: Path
    #: The visible window. REVIEW 2026-08-28: the archive kept only the
    #: viewBox's width/height and compared geometry against ``0..width``, so a
    #: document with a non-zero viewBox ORIGIN both flagged visible elements and
    #: cleared invisible ones. ``_nested_viewport_matrix`` already subtracted
    #: the origin for inset viewports; the root viewport now agrees with it.
    viewport: Rect
    nodes: tuple[Node, ...]
    #: ``clipPath`` id -> clip rectangle, for modelled clip paths only
    #: (see ``parse_golden``; unmodelled shapes are absent, not approximated).
    clip_rects: dict[str, Rect] = field(default_factory=dict)
    #: Every ``clipPath`` id the document defines, whatever shape it holds.
    clip_path_ids: frozenset[str] = frozenset()
    id_counts: dict[str, int] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """Repo-relative path, for use in finding messages."""
        try:
            return str(self.path.relative_to(_REPO_ROOT))
        except ValueError:
            return str(self.path)


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _root_viewport(root: ET.Element) -> Rect:
    """The visible window of a document, honouring the viewBox origin."""
    vb = _numbers(root.get("viewBox", ""))
    if len(vb) == 4:
        return Rect.from_origin_size(vb[0], vb[1], vb[2], vb[3])
    return Rect.from_origin_size(
        0.0, 0.0, _as_float(root.get("width"), 0.0), _as_float(root.get("height"), 0.0)
    )


@dataclass(frozen=True)
class _ClipScope:
    """The ``clipPath`` currently being descended into.

    ``ctm`` is the transform in effect where the ``clipPath`` is *defined*, so
    comparing a shape's own ctm against it detects any transform introduced
    between the two — on the ``clipPath``, on an intervening ``<g>``, or on the
    shape itself.
    """

    clip_id: str
    ctm: Matrix


@dataclass(frozen=True)
class _Frame:
    """One entry of the depth-first traversal."""

    element: ET.Element
    ctm: Matrix
    ancestors: tuple[str, ...]
    in_defs: bool
    clip_scope: _ClipScope | None


def parse_golden(path: Path) -> Document:
    """Parse one golden SVG into a flat, transform-resolved node list.

    A ``clipPath`` enters ``Document.clip_rects`` only when it is one
    untransformed ``rect`` under the default ``clipPathUnits``. Anything else
    is left unmodelled rather than approximated (a guessed clip window
    manufactures ``outside_clip`` findings that look exactly like real ones).
    """
    root = ET.fromstring(path.read_text())
    nodes: list[Node] = []
    id_counts: dict[str, int] = {}
    clip_path_ids: set[str] = set()
    clip_rect_by_owner: dict[str, Rect] = {}
    clip_shape_counts: Counter[str] = Counter()
    stack = [_Frame(root, IDENTITY, (), False, None)]
    while stack:
        frame = stack.pop()
        el = frame.element
        tag = _strip_ns(el.tag)
        attrib = {_strip_ns(k): v for k, v in el.attrib.items()}
        node_ctm = frame.ctm
        if "transform" in attrib:
            node_ctm = parse_transform(attrib["transform"]).then(node_ctm)
        if tag == "svg" and frame.ancestors:
            # REVIEW 2026-08-28: composed under any own transform rather than
            # replacing it; the viewport transform is the innermost one.
            node_ctm = _nested_viewport_matrix(attrib).then(node_ctm)
        if "id" in attrib:
            id_counts[attrib["id"]] = id_counts.get(attrib["id"], 0) + 1
        child_defs = frame.in_defs or tag == "defs"
        if tag == "clipPath":
            child_scope = _open_clip_scope(attrib, frame.ctm, clip_path_ids)
        else:
            child_scope = frame.clip_scope
        scope = frame.clip_scope
        if scope is not None and tag in _DRAWABLE:
            clip_shape_counts[scope.clip_id] += 1
            if tag == "rect" and node_ctm == scope.ctm:
                clip_rect_by_owner[scope.clip_id] = Rect.from_origin_size(
                    _as_float(attrib.get("x")),
                    _as_float(attrib.get("y")),
                    _as_float(attrib.get("width")),
                    _as_float(attrib.get("height")),
                )
        nodes.append(
            Node(
                index=len(nodes),
                tag=tag,
                attrib=attrib,
                ctm=node_ctm,
                ancestors=frame.ancestors,
                in_defs=child_defs and tag != "defs",
                text="".join(el.itertext()) if tag in _TEXT_CONTENT_TAGS else "",
            )
        )
        for child in reversed(list(el)):
            stack.append(_Frame(child, node_ctm, frame.ancestors + (tag,), child_defs, child_scope))
    clip_rects = {cid: r for cid, r in clip_rect_by_owner.items() if clip_shape_counts[cid] == 1}
    return Document(
        path=path,
        viewport=_root_viewport(root),
        nodes=tuple(nodes),
        clip_rects=clip_rects,
        clip_path_ids=frozenset(clip_path_ids),
        id_counts=id_counts,
    )


def _open_clip_scope(
    attrib: dict[str, str], parent_ctm: Matrix, seen_ids: set[str]
) -> _ClipScope | None:
    """Start tracking a ``clipPath``'s shapes, or decline to model it.

    Declines when the element has no id (unreferenceable), when its id is
    already taken (``url(#id)`` reaches the FIRST definition, so a later
    duplicate is unreachable and its shapes must not be attributed to the id),
    or when ``clipPathUnits`` is not the default ``userSpaceOnUse`` (the shape
    coordinates would be object-bounding-box fractions, not user units).
    """
    clip_id = attrib.get("id")
    if clip_id is None:
        return None
    already_defined = clip_id in seen_ids
    seen_ids.add(clip_id)
    if already_defined or attrib.get("clipPathUnits", "userSpaceOnUse") != "userSpaceOnUse":
        return None
    return _ClipScope(clip_id=clip_id, ctm=parent_ctm)


# --------------------------------------------------------------------------
# findings
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """One oracle hit: the file, the element, and what is wrong with it."""

    query: str
    file: str
    tag: str
    detail: str
    #: True when the element sits at the documented `BREAK_HIDDEN` sentinel.
    sentinel: bool = False

    def __str__(self) -> str:
        """Render as ``query: file <tag> detail`` for assertion output."""
        mark = " [BREAK_HIDDEN]" if self.sentinel else ""
        return f"{self.query}: {self.file} <{self.tag}> {self.detail}{mark}"


def unexplained(findings: Iterable[Finding]) -> list[Finding]:
    """Drop findings explained by the `BREAK_HIDDEN` sentinel."""
    return [f for f in findings if not f.sentinel]


def _extent(points: Sequence[tuple[float, float]]) -> str:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return f"x=[{min(xs):.1f},{max(xs):.1f}] y=[{min(ys):.1f},{max(ys):.1f}]"


def query_off_viewport(doc: Document, *, tol: float = 0.5) -> list[Finding]:
    """Drawable elements lying wholly outside the root viewBox."""
    out = []
    for node in doc.nodes:
        if node.tag not in _DRAWABLE or node.in_defs:
            continue
        pts = node.points
        if pts and doc.viewport.encloses_none_of(pts, tol=tol):
            out.append(
                Finding(
                    "off_viewport",
                    doc.name,
                    node.tag,
                    f"{_extent(pts)} outside {doc.viewport}",
                    sentinel=node.at_break_sentinel,
                )
            )
    return out


def query_negative_size(doc: Document) -> list[Finding]:
    """Negative ``width``/``height``/``r``/``rx``/``ry``, defs included."""
    out = []
    for node in doc.nodes:
        for attr in ("width", "height", "r", "rx", "ry"):
            if attr not in node.attrib:
                continue
            if _as_float(node.attrib[attr], 0.0) < 0:
                out.append(
                    Finding("negative_size", doc.name, node.tag, f"{attr}={node.attrib[attr]}")
                )
    return out


def query_nan_inf(doc: Document) -> list[Finding]:
    """Non-finite numbers in numeric ATTRIBUTE VALUES.

    Anchored on attribute values because the embedded base64 font body
    contains ``nan``/``inf`` as substrings; a file-level text scan reports a
    false positive on every golden in the corpus.
    """
    out = []
    for node in doc.nodes:
        for key, raw in node.attrib.items():
            if key in _NUMERIC_VALUE_ATTRS and _NONFINITE_RE.search(raw):
                out.append(Finding("nan_inf_attr", doc.name, node.tag, f"{key}={raw[:60]!r}"))
    return out


def query_zero_size_filled(doc: Document) -> list[Finding]:
    """Filled ``rect``/``circle`` elements with zero extent."""
    out = []
    for node in doc.nodes:
        if node.in_defs:
            continue
        fill = node.attrib.get("fill", "").strip().lower()
        if fill in ("", "none", "transparent"):
            continue
        if node.tag == "rect":
            w = _as_float(node.attrib.get("width"), 0.0)
            h = _as_float(node.attrib.get("height"), 0.0)
            if w == 0 or h == 0:
                out.append(
                    Finding("zero_size_filled", doc.name, "rect", f"width={w:g} height={h:g}")
                )
        elif node.tag == "circle" and _as_float(node.attrib.get("r"), 0.0) == 0:
            out.append(Finding("zero_size_filled", doc.name, "circle", "r=0"))
    return out


def query_duplicate_ids(doc: Document) -> list[Finding]:
    """Repeated ``id`` values. ``url(#id)`` resolves to the first match."""
    return [
        Finding("duplicate_id", doc.name, "*", f"id={ident!r} appears {n} times")
        for ident, n in sorted(doc.id_counts.items())
        if n > 1
    ]


def _resolved_clip_rect(rect: Rect, ctm: Matrix) -> Rect:
    """The clip rectangle in root space, mapped through the referencing ctm.

    ``clipPathUnits`` defaults to ``userSpaceOnUse``: the rect is written in
    the user space of the element that REFERENCES it. REVIEW 2026-08-28: all
    four corners, not the two the archive mapped — under a rotate or skew ctm
    two corners do not bound the mapped rectangle.
    """
    corners = [
        ctm.apply(px, py)
        for px, py in (
            (rect.x0, rect.y0),
            (rect.x1, rect.y0),
            (rect.x0, rect.y1),
            (rect.x1, rect.y1),
        )
    ]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return Rect(min(xs), min(ys), max(xs), max(ys))


def query_outside_clip(doc: Document, *, tol: float = 0.5) -> list[Finding]:
    """Elements lying wholly outside the clip rectangle they reference.

    Two scopes, deliberately different and stated here rather than inherited:
    an *undefined* reference is a document-level defect and is reported
    wherever it appears, ``<defs>`` included; the *geometry* judgment is made
    only for painted content, because a clipped template inside ``<defs>``
    is judged where it is used, not where it is written (see
    ``_subtree_drawables``).

    A clip this module declines to model (see ``parse_golden``) is skipped,
    never guessed at.
    """
    out = []
    for node in doc.nodes:
        ref = node.clip_ref
        if ref is None:
            continue
        rect = doc.clip_rects.get(ref)
        if rect is None:
            if ref in doc.clip_path_ids:
                continue  # defined but unmodelled — not a defect
            out.append(
                Finding("outside_clip", doc.name, node.tag, f"clip-path url(#{ref}) undefined")
            )
            continue
        window = _resolved_clip_rect(rect, node.ctm)
        for child in _subtree_drawables(doc, node):
            pts = child.points
            if pts and window.encloses_none_of(pts, tol=tol):
                out.append(
                    Finding(
                        "outside_clip",
                        doc.name,
                        child.tag,
                        f"{_extent(pts)} outside clip #{ref} {window}",
                        sentinel=child.at_break_sentinel,
                    )
                )
    return out


class _DrawableKey(NamedTuple):
    """What makes two painted chrome elements the same element (spec §4.5)."""

    tag: str
    points: tuple[tuple[float, float], ...]
    text: str
    fill: str
    stroke: str

    @classmethod
    def of(cls, node: Node, *, ndigits: int) -> _DrawableKey:
        """Build the identity key for one node."""
        return cls(
            tag=node.tag,
            points=tuple((round(x, ndigits), round(y, ndigits)) for x, y in node.points),
            text=node.text,
            fill=node.attrib.get("fill", ""),
            stroke=node.attrib.get("stroke", ""),
        )


def query_duplicate_drawable(doc: Document, *, ndigits: int = 3) -> list[Finding]:
    """Chrome elements painted twice at the same place with the same paint.

    Two ``line``/``text`` nodes outside ``<defs>`` that agree on tag, resolved
    coordinates, text content, ``fill`` and ``stroke`` are one element drawn
    twice. This is the permanent oracle for the shared-axis chrome dedup: a
    composition that lays two panels over one rect and paints both panels'
    axes produces output that is pixel-identical to the correct render, so no
    byte-equality golden can catch it.

    Data marks are excluded on purpose. Two circles at one point are a legal
    overplot, not a defect.
    """
    counts: Counter[_DrawableKey] = Counter()
    for node in doc.nodes:
        if node.tag not in _CHROME_TAGS or node.in_defs or node.at_break_sentinel:
            continue
        counts[_DrawableKey.of(node, ndigits=ndigits)] += 1
    # Counter preserves first-insertion order, i.e. document order.
    return [
        Finding(
            "duplicate_drawable",
            doc.name,
            key.tag,
            f"{n} coincident <{key.tag}> at {key.points} text={key.text!r} "
            f"fill={key.fill!r} stroke={key.stroke!r}",
        )
        for key, n in counts.items()
        if n > 1
    ]


def _subtree_drawables(doc: Document, node: Node) -> list[Node]:
    """Painted drawables governed by ``node``'s clip: itself, or its descendants.

    ``<defs>`` content is excluded on both paths: a template is judged where it
    is used, not where it is written, which is the scope ``query_off_viewport``
    and ``query_zero_size_filled`` also use.
    """
    if node.in_defs:
        return []
    if node.tag in _DRAWABLE:
        return [node]
    depth = len(node.ancestors)
    out = []
    for other in doc.nodes[node.index + 1 :]:
        if len(other.ancestors) <= depth:
            break
        if other.tag in _DRAWABLE and not other.in_defs:
            out.append(other)
    return out


QUERIES = {
    "off_viewport": query_off_viewport,
    "negative_size": query_negative_size,
    "nan_inf_attr": query_nan_inf,
    "zero_size_filled": query_zero_size_filled,
    "duplicate_id": query_duplicate_ids,
    "outside_clip": query_outside_clip,
    "duplicate_drawable": query_duplicate_drawable,
}


def run_all(paths: Sequence[Path] | None = None) -> dict[str, list[Finding]]:
    """Run every query over every golden (or the given subset), keyed by query.

    The keys are ``QUERIES``' keys unconditionally, so an empty result for a
    query is distinguishable from a query that never ran.
    """
    out: dict[str, list[Finding]] = {name: [] for name in QUERIES}
    for path in golden_paths() if paths is None else paths:
        doc = parse_golden(path)
        for name, fn in QUERIES.items():
            out[name].extend(fn(doc))
    return out
