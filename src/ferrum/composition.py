"""Multi-chart composition primitives (HConcat, VConcat, Layer, Concat, Joint, Repeat, ClusterMap)."""

from __future__ import annotations

import copy
import json as _json
import struct
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from typing import TypedDict

from ferrum._chrome import chrome_kwargs, merge_configure_layers
from ferrum._configure_mixin import ConfigureMixin
from ferrum._overrides import _FIGURE_CHROME_KEYS

# ---------------------------------------------------------------------------
# Shared offset key-set constants
# ---------------------------------------------------------------------------

# Keys for the two area sub-dicts in a panel node (x/y are offset directly).
_PANEL_AREA_KEYS: tuple[str, ...] = ("plot_area", "clip")

# Keys for per-node lists inside a panel (each element passed to _offset_node).
_PANEL_NODE_LIST_KEYS: tuple[str, ...] = ("axes", "grid", "annotations", "strip_title")

# Keys for per-node lists in the outer scene (title / legend / decorations).
_OUTER_NODE_LIST_KEYS: tuple[str, ...] = ("title", "legend", "decorations")


class _FigureChrome(TypedDict):
    """Figure-level chrome payload threaded through scene-merge helpers.

    Produced by :meth:`_CompositeBase._figure_chrome_kwargs` and consumed by
    :func:`_inject_figure_chrome` (via every ``_merge_child_scenes*`` function
    and :func:`_render_single_with_figure_chrome`).  Using a ``TypedDict`` makes
    the key contract explicit and prevents the dict from silently drifting.
    """

    title: Optional[str]
    subtitle: Optional[str]
    caption: Optional[str]
    chrome: dict


def _embed_chart_spec(c) -> Optional[dict]:
    """Convert a Chart's ``.to_spec()`` output to an embedded JSON dict."""
    if c is None or not hasattr(c, "to_spec"):
        return None
    return _json.loads(c.to_spec().to_json())


def _copy_configure_layers(src: "_ChartLike", dst: "_ChartLike") -> None:
    """Copy ``_configure_layers`` from *src* to *dst* if present.

    Used by ``theme()``, ``properties()``, and ``share_scale()`` after
    ``_rebuild_with_charts`` to ensure composition-level configure
    settings survive rebuild operations.
    """
    config = getattr(src, "_configure_layers", None)
    if config:
        dst._configure_layers = list(config)


def _shallow_copy_composite(src) -> object:
    """Shallow copy for ``_ChartLike`` subclasses that mix ``__slots__`` and ``__dict__``.

    Shared by ``_CompositeBase.__copy__`` and ``LayerChart.__copy__`` so the
    ``__dict__`` + MRO-slot copy logic lives in one place.  Returns a new
    instance with all ``__dict__`` keys and all ``__slots__`` attributes from
    the full MRO copied to the new object.  The caller is responsible for
    making any mutable slot attributes (e.g. ``charts``, ``_charts``) into
    fresh copies after this returns.
    """
    new = object.__new__(type(src))
    if hasattr(src, "__dict__"):
        new.__dict__.update(src.__dict__)
    for cls in type(src).__mro__:
        for slot in getattr(cls, "__slots__", ()):
            if slot == "__dict__":
                continue
            try:
                setattr(new, slot, getattr(src, slot))
            except AttributeError:
                pass
    return new


def _validate_resolve(resolve: Optional[Dict[str, str]], label: str) -> None:
    """Raise ``ValueError`` when *resolve* is not a valid channel-mode dict.

    Parameters
    ----------
    resolve : dict or None
        Must be a dict mapping channel names to ``"shared"`` or
        ``"independent"`` when not ``None``.
    label : str
        Class or function name used in the error message.
    """
    if resolve is None:
        return
    if not isinstance(resolve, dict):
        raise ValueError(
            f"{label}: resolve must be a dict mapping channel names "
            f"to 'shared' or 'independent'; got {type(resolve).__name__}"
        )
    for ch, mode in resolve.items():
        if mode not in ("shared", "independent"):
            raise ValueError(
                f"{label}: resolve[{ch!r}]={mode!r}; expected 'shared' or 'independent'"
            )


def _validate_share_modes(channels: Dict[str, str]) -> None:
    """Raise ``ValueError`` when any ``share_scale`` value is not a valid mode.

    Shared by :meth:`_ChartLike.share_scale` and
    :meth:`RepeatChart.share_scale` so the ``"shared"``/``"independent"``
    vocabulary is validated in one place.

    Parameters
    ----------
    channels : dict
        Channel name → mode mapping from a ``share_scale`` call.
    """
    for ch, mode in channels.items():
        if mode not in ("shared", "independent"):
            raise ValueError(f"share_scale: {ch}={mode!r}; expected 'shared' or 'independent'")


def _apply_resolve(charts: list, resolve: Optional[Dict[str, str]]) -> list:
    """Return charts with shared-scale injection applied per *resolve*.

    For each channel whose mode is ``"shared"``, compute the union domain
    across all charts and inject an explicit scale on every chart that
    binds that channel.  Charts are returned unchanged when *resolve* is
    ``None`` or empty, or when no channel qualifies.

    Parameters
    ----------
    charts : list of Chart
        Source charts.
    resolve : dict or None
        Per-channel scale-sharing spec, e.g. ``{"color": "shared"}``.

    Returns
    -------
    list of Chart
        Original list (same object) when nothing changes, otherwise a
        new list with scale-injected clones.
    """
    if not resolve:
        return charts
    from ferrum._scale_share import compute_union_domain, inject_scale

    shared = [ch for ch, mode in resolve.items() if mode == "shared"]
    if not shared:
        return charts
    result = list(charts)
    for channel in shared:
        sd = compute_union_domain(result, channel)
        if sd is None:
            continue
        result = [inject_scale(c, channel, sd) for c in result]
    return result


class _ChartLike(ConfigureMixin):
    """Common rendering plumbing shared by every composition wrapper.

    Concrete subclasses must implement :meth:`to_svg`, :attr:`charts`,
    :meth:`theme`, :meth:`properties`, and :meth:`__repr__`.  This base
    centralizes the save / show / Jupyter-display / PNG-stub boilerplate
    that previously drifted across five copies (K2 / K3 / K11 / K15).

    **Composition-level configure** (``configure_axis``, ``configure_grid``,
    etc.) is accumulated in ``_configure_layers`` and injected into each
    child chart at render time.  Child-level config always wins over
    composition-level config because the composition layers are prepended
    (earlier entries in ``_configure`` are overridden by later ones).
    """

    def to_svg(self) -> str:  # pragma: no cover - abstract
        raise NotImplementedError(f"{type(self).__name__} must implement to_svg")

    def interactive(self, *, toolbar: bool = True):
        """Return an interactive rendering of this composition.

        Parameters
        ----------
        toolbar : bool, default True
            Whether to show the interactive toolbar (zoom/pan controls, export
            button). Set to ``False`` to render without the toolbar.

        Returns
        -------
        InteractiveChart
            An interactive widget/container backed by the WASM renderer.
        """
        from ferrum._interactive import InteractiveChart

        return InteractiveChart(self, toolbar=toolbar)

    # Subclasses provide ``charts`` as either an instance attribute
    # (symmetric containers — HConcat / VConcat) or as a ``@property``
    # (asymmetric containers — Joint / Repeat / ClusterMap, where the
    # shape is fixed and a derived list is the natural accessor).  We
    # do not declare it on the base because Python's data-descriptor
    # rules would block the attribute form on ``_CompositeBase``.

    def show(self) -> None:
        """Print the SVG markup to stdout."""
        print(self.to_svg())

    def _repr_svg_(self) -> str:
        """Return SVG for Jupyter inline display."""
        return self.to_svg()

    def _repr_mimebundle_(self, include=None, exclude=None) -> dict:
        """Return a Jupyter MIME bundle for rich display.

        Jupyter prefers ``_repr_mimebundle_`` over per-type ``_repr_*_``
        methods when both exist, so providing it lets front-ends negotiate
        formats without falling back to text repr.
        """
        return {"image/svg+xml": self.to_svg()}

    def to_png(self, *, scale: float = 2.0) -> bytes:
        """Return the composition rendered as PNG bytes.

        This **returns** the PNG-encoded image data; it does not display the
        composition.  Rasterises the SVG produced by :meth:`to_svg` through
        the Rust resvg pipeline — the same rasteriser ``Chart.to_png()`` uses.

        Parameters
        ----------
        scale : float, default 2.0
            Pixel-density multiplier applied to the SVG's intrinsic dimensions.
            ``2.0`` (the default) produces a retina-quality image.  ``1.0``
            renders at 1:1 resolution.

        Returns
        -------
        bytes
            PNG image as raw bytes suitable for ``IPython.display.Image``
            or writing directly to disk.
        """
        from ferrum._core import rasterize_svg

        return bytes(rasterize_svg(self.to_svg(), scale=scale))

    def _figure_title_text(self) -> str:
        """Return the resolved figure title text for the document ``<title>``.

        This is the canonical title accessor shared by every chart-like.
        The base implementation resolves a single chart's ``_title`` (a
        ``Title`` dataclass or plain string); :class:`_CompositeBase`
        overrides it to resolve the composite's ``_figure_title``.  Both
        fall back to ``"Ferrum chart"`` when no title is set.
        """
        from ferrum.display import _extract_title_text

        return _extract_title_text(getattr(self, "_title", None))

    def to_html(
        self,
        *,
        embed_wasm: bool = True,
        toolbar: bool = True,
        csp_nonce: str | None = None,
    ) -> str:
        """Return the composition as a self-contained interactive HTML document.

        This **returns** the HTML markup; it does not display the composition
        or write it to disk.  The returned string is byte-identical to what
        ``save(path)`` writes for an ``.html`` destination — it embeds the
        WASM-backed interactive renderer rather than a static SVG snapshot.
        Because it bundles that renderer, the document is substantially larger
        than a static export; for a lightweight static image use
        :meth:`to_svg` / :meth:`to_png`.

        Routes through the shared :func:`ferrum.display.html_string` helper, so
        the HTML assembly and tab-title resolution are identical to a plain
        ``Chart``.  It does **not** construct a live ``InteractiveChart``
        widget, so headless HTML export works without ``anywidget`` installed.

        Parameters
        ----------
        embed_wasm : bool, default True
            When True, the WASM binary is base64-inlined for single-file
            distribution.  When False, the document references an adjacent
            ``ferrum_wasm_bg.wasm`` sidecar that must be served alongside it.
        toolbar : bool, default True
            When False, the interactive toolbar (zoom / pan controls, export
            button) is hidden in the rendered HTML.
        csp_nonce : str, optional
            When provided, both the ``<style>`` and ``<script type="module">``
            tags receive a ``nonce="..."`` attribute so they pass strict
            Content-Security-Policy headers.

        Returns
        -------
        str
            A complete, self-contained interactive HTML document.
        """
        from ferrum.display import html_string

        return html_string(
            self,
            embed_wasm=embed_wasm,
            toolbar=toolbar,
            csp_nonce=csp_nonce,
        )

    def show_svg(self) -> str:
        """Render the composition to an SVG string.

        .. deprecated:: 0.16.0
            Use :meth:`to_svg` instead.  ``show_svg`` will be removed in a
            future release.  It now forwards to :meth:`to_svg`.

        Returns
        -------
        str
            SVG markup for the composition.
        """
        warnings.warn(
            f"{type(self).__name__}.show_svg() is deprecated; use .to_svg() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.to_svg()

    def show_png(self, *, scale: float = 2.0) -> bytes:
        """Render the composition to PNG bytes.

        .. deprecated:: 0.16.0
            Use :meth:`to_png` instead.  ``show_png`` will be removed in a
            future release.  It now forwards to :meth:`to_png`.

        Parameters
        ----------
        scale : float, default 2.0
            Pixel-density multiplier applied to the SVG's intrinsic dimensions.

        Returns
        -------
        bytes
            PNG image as raw bytes.
        """
        warnings.warn(
            f"{type(self).__name__}.show_png() is deprecated; use .to_png() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.to_png(scale=scale)

    def save(
        self,
        path: str,
        *,
        format=None,
        scale: float = 2.0,
        toolbar: bool = True,
        embed_wasm: bool = True,
        csp_nonce: str | None = None,
    ) -> None:
        """Save the composition to a file.

        Routes through :func:`ferrum.display.save_chart` — the single
        save-format router shared with ``Chart.save`` — so the supported
        format table (``svg`` / ``png`` / ``html`` / ``json`` / ``pdf``) and
        the HTML / title resolution are identical across chart and composite.

        Parameters
        ----------
        path : str
            Destination file path.  The extension determines the format when
            *format* is omitted.
        format : {"svg", "png", "html", "json", "pdf"}, optional
            Explicit format override.  Other formats raise ``ValueError``.
        scale : float, default 2.0
            Pixel-density multiplier for PNG and PDF output.  Has no effect
            on SVG, HTML, or JSON exports.
        toolbar : bool, default True
            Whether to include the interactive toolbar (zoom/pan controls,
            export button) when saving as HTML.  Has no effect on SVG, PNG,
            JSON, or PDF exports.
        embed_wasm : bool, default True
            For ``"html"`` format only.  When True, the WASM binary is
            base64-inlined for single-file distribution.  When False, an
            adjacent ``ferrum_wasm_bg.wasm`` sidecar is written alongside.
        csp_nonce : str, optional
            For ``"html"`` format only.  When provided, both the ``<style>``
            and ``<script type="module">`` tags receive a ``nonce="..."``
            attribute so they pass strict Content-Security-Policy headers.

        Raises
        ------
        ValueError
            If *format* (or the path extension) is not a recognised export
            format.
        """
        from ferrum.display import save_chart

        save_chart(
            self,
            path,
            format=format,
            scale=scale,
            toolbar=toolbar,
            embed_wasm=embed_wasm,
            csp_nonce=csp_nonce,
        )

    def share_scale(self, **channels):
        """Share scales across this composition's member charts.

        Computes the union domain for each channel marked ``"shared"``
        and re-emits every member chart with an explicit ``scale=`` dict
        on that channel, so the participating axes lock to the same
        ticks.  Channels marked ``"independent"`` (the default for any
        channel not listed) keep their per-chart domains.

        Parameters
        ----------
        **channels : str
            Channel name → ``"shared"`` | ``"independent"``.  Common
            channels: ``x``, ``y``, ``color``, ``size``.

        Returns
        -------
        _ChartLike
            A new composition of the same type with the shared scales
            injected.  No-op (returns ``self``) when no channel is
            ``"shared"`` or none of the requested channels are bound on
            any member chart.

        Raises
        ------
        ValueError
            If any value is not ``"shared"`` or ``"independent"``.

        Examples
        --------
        >>> import ferrum as fm
        >>> combined = (chart_a | chart_b).share_scale(x="shared")
        >>> grid = fm.JointChart(center, top=hist_x, right=hist_y).share_scale(y="shared")
        """
        _validate_share_modes(channels)
        shared = [ch for ch, mode in channels.items() if mode == "shared"]
        if not shared:
            return self
        from ferrum._scale_share import compute_union_domain, inject_scale

        member_charts = self.charts
        scale_dicts = {}
        for channel in shared:
            sd = compute_union_domain(member_charts, channel)
            if sd is not None:
                scale_dicts[channel] = sd
        if not scale_dicts:
            return self

        def _apply(chart):
            out = chart
            for ch, sd in scale_dicts.items():
                out = inject_scale(out, ch, sd)
            return out

        result = self._rebuild_with_charts(_apply)
        _copy_configure_layers(self, result)
        return result

    def theme(self, t):
        """Apply a theme to every sub-chart and return a new composition.

        Parameters
        ----------
        t : Theme
            Theme value to apply.

        Returns
        -------
        _ChartLike
            A new instance of the same composition class with *t* applied
            to each sub-chart.
        """
        result = self._rebuild_with_charts(lambda c: c.theme(t))
        _copy_configure_layers(self, result)
        return result

    def properties(self, **kwargs):
        """Forward ``properties(**kwargs)`` to every sub-chart.

        This base implementation is used by ``LayerChart`` and by plain
        single-chart wrappers that do not override it.
        ``_CompositeBase`` overrides this with a version that intercepts
        figure-level chrome (``title``, ``subtitle``, ``caption``) and stores
        it at the composition level rather than fanning it to every child.

        Parameters
        ----------
        **kwargs
            Keyword arguments accepted by ``Chart.properties`` (e.g.
            ``width``, ``height``, ``title``).

        Returns
        -------
        _ChartLike
            A new instance of the same composition class with updated
            sub-chart properties.
        """
        result = self._rebuild_with_charts(lambda c: c.properties(**kwargs))
        _copy_configure_layers(self, result)
        return result

    # ---- Declarative configuration surface ----

    def _inject_parent_config(self, chart):
        """Prepend composition-level configure layers onto a child chart.

        For ``Chart`` children, composition-level layers are prepended to the
        chart's ``_configure`` list so that per-chart layers (which appear
        later) override them — ``_resolve_chart_config`` processes
        ``_configure`` in order with later entries winning.

        For nested composition children (``_ChartLike`` subclasses), the
        layers are merged into the child's ``_configure_layers`` so they
        propagate further down at render time.
        """
        config = getattr(self, "_configure_layers", None)
        if not config:
            return chart
        if isinstance(chart, _ChartLike):
            # Nested composition: merge into child's own _configure_layers.
            new = copy.copy(chart)
            existing = getattr(chart, "_configure_layers", [])
            new._configure_layers = list(config) + list(existing)
            return new
        # Plain Chart: prepend to _configure list.
        new = chart._clone()
        new._configure = list(config) + list(new._configure)
        return new

    def _append_configure(self, config) -> "_ChartLike":
        """Clone self, append *config* to ``_configure_layers``, return new instance."""
        new = copy.copy(self)
        new._configure_layers = list(getattr(self, "_configure_layers", [])) + [config]
        return new

    # ---- Declarative configuration surface (provided by ConfigureMixin) ----

    def _rebuild_with_charts(self, fn):  # pragma: no cover - abstract
        """Return a new composition with each member chart transformed by *fn*.

        Subclasses must implement this — it's the seam between the
        generic ``share_scale`` / ``theme`` / ``properties`` plumbing on
        the base and each composition's constructor signature.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement _rebuild_with_charts")


class _CompositeBase(_ChartLike):
    """Shared base for every composite chart with figure-level chrome.

    This is the single home for figure-level title / subtitle / caption
    across all composites — the symmetric list containers (HConcat /
    VConcat / Concat) *and* the asymmetric panel layouts (Joint / Repeat /
    ClusterMap).  Figure chrome is stored at the composition level
    (``_figure_title`` / ``_figure_subtitle`` / ``_figure_caption``),
    intercepted in :meth:`properties` so it never reaches an inner panel,
    and surfaced for the HTML document title via :meth:`_figure_title_text`.

    The symmetric containers also use this class's ``__init__`` to hold an
    ordered ``charts`` list and a pixel ``spacing`` between panels, plus
    ``__or__`` / ``__and__`` to chain further compositions.  The asymmetric
    layouts keep their own slot-based ``__init__`` and ``charts`` property;
    they call :meth:`_init_figure_chrome` to wire the chrome fields.

    **Symmetric-concat layout strategy.**  ``HConcatChart`` and
    ``VConcatChart`` differ only in their layout axis, so their
    ``_resolved_charts`` / ``_rebuild_with_charts`` / ``_render_interactive``
    / ``to_svg`` / ``__repr__`` bodies live here once, parameterized by three
    class attributes a subclass overrides: :attr:`_layout` (the
    ``_merge_child_scenes`` layout key), :attr:`_svg_compose_name` (the
    ``ferrum._core`` SVG compositor to import), and :attr:`_svg_align` (the
    cross-axis alignment passed to that compositor).  These default to
    ``None`` on the base; the asymmetric layouts (Joint / Repeat /
    ClusterMap) and the wrapping-grid ``ConcatChart`` override the symmetric
    methods wholesale, so the ``None`` defaults are never reached for them.
    """

    # Symmetric-concat strategy hooks (overridden by HConcat / VConcat).
    _layout: Optional[str] = None
    _svg_compose_name: Optional[str] = None
    _svg_align: Optional[str] = None

    def __init__(
        self,
        charts: List,
        *,
        spacing: float = 10.0,
        resolve: Optional[Dict[str, str]] = None,
    ) -> None:
        _validate_resolve(resolve, type(self).__name__)
        self.charts = list(charts)
        self.spacing = spacing
        self._resolve = resolve
        self._init_figure_chrome()

    def _init_figure_chrome(self) -> None:
        """Initialize the figure-chrome fields to their empty state.

        Called from every composite's ``__init__`` (directly for the
        asymmetric layouts, via :meth:`__init__` for the symmetric ones)
        so the chrome attributes always exist before :meth:`properties`
        runs.
        """
        self._figure_title: Optional[str] = None
        self._figure_subtitle: Optional[str] = None
        self._figure_caption: Optional[str] = None

    def _carry_figure_chrome(self, dst: "_CompositeBase") -> None:
        """Copy this composite's figure chrome onto *dst* (a new instance)."""
        dst._figure_title = self._figure_title
        dst._figure_subtitle = self._figure_subtitle
        dst._figure_caption = self._figure_caption

    def _figure_title_text(self) -> str:
        """Resolve the composite's figure title text for the document ``<title>``."""
        from ferrum.display import _extract_title_text

        return _extract_title_text(self._figure_title)

    def _figure_chrome_kwargs(self) -> "_FigureChrome":
        """Bundle figure chrome for the interactive scene-merge functions.

        Returns the ``figure_chrome`` payload consumed by every
        ``_merge_child_scenes*`` helper: the title / subtitle / caption text
        plus the positioning ``chrome`` sub-dict resolved from this
        composite's configure layers (the same kwargs the SVG path passes to
        ``compose_svg_*``).  This keeps the interactive on-canvas band in step
        with the SVG band from a single source of chrome values.
        """
        return _FigureChrome(
            title=self._figure_title,
            subtitle=self._figure_subtitle,
            caption=self._figure_caption,
            chrome=chrome_kwargs(merge_configure_layers(getattr(self, "_configure_layers", None))),
        )

    def __copy__(self):
        """Shallow copy that duplicates mutable list attributes."""
        new = _shallow_copy_composite(self)
        # Ensure the mutable charts list is a fresh copy.  Asymmetric
        # layouts expose ``charts`` as a read-only property (derived from
        # their panels), so only refresh it when it is a writable attribute.
        if not isinstance(getattr(type(self), "charts", None), property):
            new.charts = list(self.charts)
        return new

    def __or__(self, other):
        return HConcatChart([self, other])

    def __and__(self, other):
        return VConcatChart([self, other])

    def properties(self, **kwargs):
        """Set figure-level or per-child chart properties.

        The keyword arguments ``title``, ``subtitle``, and ``caption`` are
        intercepted and stored at the figure level — they render once around
        the whole composed figure and are never fanned to individual child
        panels.  All other keyword arguments (e.g. ``width``, ``height``)
        are forwarded to each child via ``Chart.properties``.

        Parameters
        ----------
        title : str, optional
            Figure-level title rendered above all panels.
        subtitle : str, optional
            Figure-level subtitle rendered below the title, above all panels.
        caption : str, optional
            Figure-level caption rendered below all panels.
        **kwargs
            Additional keyword arguments forwarded to ``Chart.properties``
            for every child chart (e.g. ``width``, ``height``).

        Returns
        -------
        _CompositeBase
            A new instance of the same composition class with the figure
            chrome stored and / or child properties updated.
        """
        # Separate figure-level chrome from per-child kwargs.
        # Use the shared _FIGURE_CHROME_KEYS constant so the key set stays
        # in sync with the factory-dict split in _overrides._apply_overrides.
        chrome_vals = {k: kwargs.pop(k, None) for k in _FIGURE_CHROME_KEYS}
        figure_title = chrome_vals["title"]
        figure_subtitle = chrome_vals["subtitle"]
        figure_caption = chrome_vals["caption"]

        if kwargs:
            # Forward remaining (non-chrome) kwargs to the appropriate panel(s).
            result = self._forward_child_properties(kwargs)
        else:
            # Nothing to fan — rebuild preserving charts unchanged.
            result = self._rebuild_with_charts(lambda c: c)

        _copy_configure_layers(self, result)

        # ``_rebuild_with_charts`` / ``_forward_child_properties`` already
        # carried this composite's existing chrome onto ``result``; only
        # override the fields a value was given for.
        if figure_title is not None:
            result._figure_title = figure_title
        if figure_subtitle is not None:
            result._figure_subtitle = figure_subtitle
        if figure_caption is not None:
            result._figure_caption = figure_caption

        return result

    def _forward_child_properties(self, kwargs: dict) -> "_CompositeBase":
        """Apply non-chrome ``properties`` kwargs to the relevant child panel(s).

        The default fans the kwargs to every member chart, which is correct
        for the symmetric containers (HConcat / VConcat / Concat) and for the
        repeat-grid template.  Asymmetric layouts whose marginal panels derive
        their size from a primary panel (Joint → center, ClusterMap → heatmap)
        override this to route the kwargs to that primary panel only.
        """
        return self._rebuild_with_charts(lambda c: c.properties(**kwargs))

    # ------------------------------------------------------------------
    # Symmetric-concat layout strategy.
    #
    # These five methods are the shared bodies of HConcat / VConcat,
    # parameterized by ``_layout`` / ``_svg_compose_name`` / ``_svg_align``.
    # The asymmetric layouts (Joint / Repeat / ClusterMap) and the
    # wrapping-grid ConcatChart override all five, so the ``None`` hook
    # defaults are never reached for them.
    # ------------------------------------------------------------------

    def _resolved_charts(self) -> list:
        """Return charts with shared scales injected per ``resolve``.

        Uses ``getattr`` for ``_resolve`` so this shared base method is safe on
        any ``_CompositeBase`` subclass: the symmetric concat classes set
        ``_resolve`` in ``__init__``, while asymmetric subclasses
        (Joint/Repeat/ClusterMap) never set it and override the render path, so
        they would otherwise hit an ``AttributeError`` if this base method were
        ever called on them.
        """
        return _apply_resolve(self.charts, getattr(self, "_resolve", None))

    def _rebuild_with_charts(self, fn):
        new = type(self)(
            [fn(c) for c in self.charts],
            spacing=self.spacing,
            resolve=getattr(self, "_resolve", None),
        )
        self._carry_figure_chrome(new)
        return new

    def _render_interactive(self) -> tuple[str, bytes]:
        """Render to (scene_json, packed_data) by merging child scenes along ``_layout``."""
        charts = [self._inject_parent_config(c) for c in self._resolved_charts()]
        return _merge_child_scenes(
            charts,
            self.spacing,
            layout=self._layout,
            figure_chrome=self._figure_chrome_kwargs(),
        )

    def to_svg(self) -> str:
        """Render the concatenated charts to an SVG string along ``_layout``."""
        from ferrum import _core

        compose = getattr(_core, self._svg_compose_name)
        charts = [self._inject_parent_config(c) for c in self._resolved_charts()]
        svgs = [c.to_svg() for c in charts]
        chrome = chrome_kwargs(merge_configure_layers(getattr(self, "_configure_layers", None)))
        return compose(
            svgs,
            spacing=self.spacing,
            align=self._svg_align,
            title=self._figure_title,
            subtitle=self._figure_subtitle,
            caption=self._figure_caption,
            **chrome,
        )

    def __repr__(self) -> str:
        """Return a short developer-readable description."""
        return f"{type(self).__name__}([{', '.join(repr(c) for c in self.charts)}])"


class HConcatChart(_CompositeBase):
    """Horizontal concatenation of two or more charts.

    Each sub-chart retains its own scales, axes, and legend by default.
    Pass ``resolve=`` to unify one or more channels across panels.
    Construct via the ``|`` operator on ``Chart`` instances or directly
    with a list.

    Parameters
    ----------
    charts : list of Chart
        Sub-charts to concatenate left-to-right.
    spacing : float, default 10.0
        Horizontal pixel gap between adjacent charts.
    resolve : dict, optional
        Per-channel scale-sharing overrides, e.g.
        ``{"color": "shared"}``.  Accepts the same keys and values as
        ``ConcatChart(resolve=...)``.

    Examples
    --------
    >>> import ferrum as fm
    >>> combined = fm.Chart(df).encode(x="hp", y="mpg").mark_point() | fm.Chart(df).encode(x="hp").mark_histogram()
    >>> combined.save("side_by_side.svg")
    """

    # Layout-strategy hooks consumed by _CompositeBase's symmetric-concat
    # methods (_render_interactive / to_svg).  Construction, resolve, rebuild,
    # and __repr__ are all inherited unchanged.
    _layout = "horizontal"
    _svg_compose_name = "compose_svg_horizontal"
    _svg_align = "top"


class VConcatChart(_CompositeBase):
    """Vertical concatenation of two or more charts.

    Each sub-chart retains its own scales, axes, and legend by default.
    Pass ``resolve=`` to unify one or more channels across panels.
    Construct via the ``&`` operator on ``Chart`` instances or directly
    with a list.

    Parameters
    ----------
    charts : list of Chart
        Sub-charts to stack top-to-bottom.
    spacing : float, default 10.0
        Vertical pixel gap between adjacent charts.
    resolve : dict, optional
        Per-channel scale-sharing overrides, e.g.
        ``{"color": "shared"}``.  Accepts the same keys and values as
        ``ConcatChart(resolve=...)``.

    Examples
    --------
    >>> import ferrum as fm
    >>> stacked = fm.Chart(df).encode(x="hp", y="mpg").mark_point() & fm.Chart(df).encode(x="hp").mark_histogram()
    >>> stacked.save("stacked.svg")
    """

    # Layout-strategy hooks consumed by _CompositeBase's symmetric-concat
    # methods (_render_interactive / to_svg).  Construction, resolve, rebuild,
    # and __repr__ are all inherited unchanged.
    _layout = "vertical"
    _svg_compose_name = "compose_svg_vertical"
    _svg_align = "left"


# --------------------------------------------------------------------------
# Phase 9 compound views: JointChart, RepeatChart, ClusterMapChart
# --------------------------------------------------------------------------


class JointChart(_CompositeBase):
    """Joint distribution view: center chart plus optional top and right marginals.

    Lays out a 2 × 2 grid: center chart occupies the bottom-left panel,
    *top* marginal goes top-left, *right* marginal goes bottom-right, and the
    top-right corner is empty.  The x-axis is shared between the center and
    top charts; the y-axis is shared between the center and right charts.

    The panel size ratio between the center and each marginal is controlled by
    ``ratio``.  A ratio of 5 gives the center 5/(5+1) of each dimension and
    each marginal 1/(5+1).

    Most users obtain a ``JointChart`` from `ferrum.jointplot` rather than
    constructing one directly.

    Parameters
    ----------
    center : Chart
        Primary scatter / distribution chart occupying the main panel.
    top : Chart, optional
        Marginal chart drawn above the center (e.g. a histogram of the x
        variable).
    right : Chart, optional
        Marginal chart drawn to the right of the center (e.g. a histogram
        of the y variable).
    ratio : int, default 5
        Size ratio of the center panel to each marginal panel.  Must be > 0.
    spacing : float, default 10.0
        Pixel gap between adjacent panels.

    Raises
    ------
    ValueError
        If *ratio* is not > 0.

    Examples
    --------
    >>> import ferrum as fm
    >>> joint = fm.jointplot(df, x="hp", y="mpg")
    >>> joint.save("joint.svg")
    """

    __slots__ = ("center", "top", "right", "ratio", "spacing")

    def __init__(
        self,
        center,
        *,
        top=None,
        right=None,
        ratio: int = 5,
        spacing: float = 10.0,
    ) -> None:
        if ratio <= 0:
            raise ValueError(f"ratio must be > 0; got {ratio}")
        self.center = center
        self.top = top
        self.right = right
        self.ratio = ratio
        self.spacing = spacing
        self._init_figure_chrome()

    @property
    def charts(self) -> list:
        """List of Chart : All non-None sub-charts (center, top, right)."""
        return [c for c in (self.center, self.top, self.right) if c is not None]

    @property
    def spec(self) -> dict:
        """Dict : Serializable layout spec consumed by the SVG compositor."""
        share_x = ["center"]
        if self.top is not None:
            share_x.append("top")
        share_y = ["center"]
        if self.right is not None:
            share_y.append("right")
        return {
            "kind": "joint",
            "center": _embed_chart_spec(self.center),
            "top": _embed_chart_spec(self.top),
            "right": _embed_chart_spec(self.right),
            "ratio": self.ratio,
            "spacing": self.spacing,
            "share": {"x": share_x, "y": share_y},
        }

    def _forward_child_properties(self, kwargs: dict) -> "JointChart":
        """Route non-chrome ``properties`` kwargs to the center chart only.

        The marginals (top, right) are kept unchanged because their width /
        height is derived from the center plus ``ratio`` at render time.
        Figure-level chrome (``title`` / ``subtitle`` / ``caption``) is
        intercepted by :meth:`_CompositeBase.properties` before this hook
        runs, so it never reaches the center panel.
        """
        result = JointChart(
            self.center.properties(**kwargs),
            top=self.top,
            right=self.right,
            ratio=self.ratio,
            spacing=self.spacing,
        )
        self._carry_figure_chrome(result)
        return result

    def _rebuild_with_charts(self, fn):
        new = JointChart(
            fn(self.center),
            top=(fn(self.top) if self.top is not None else None),
            right=(fn(self.right) if self.right is not None else None),
            ratio=self.ratio,
            spacing=self.spacing,
        )
        self._carry_figure_chrome(new)
        return new

    def _render_interactive(self) -> tuple[str, bytes]:
        """Render to (scene_json, packed_data) by merging child scenes in a 2x2 grid.

        Grid layout mirrors the SVG path (``to_svg``):
          - top marginal  at (row=0, col=0)
          - center        at (row=1, col=0)
          - right marginal at (row=1, col=1)
        """
        center = self._inject_parent_config(self.center)
        top = self._inject_parent_config(self.top) if self.top is not None else None
        right = self._inject_parent_config(self.right) if self.right is not None else None

        has_top = top is not None
        has_right = right is not None

        # No marginals: render center directly (still wrap the figure band).
        if not has_top and not has_right:
            return _render_single_with_figure_chrome(center, self._figure_chrome_kwargs())

        # Build grid panels matching the SVG path layout.
        panels: list[tuple[int, int, object]] = []
        if has_top and has_right:
            # Full 2x2 grid: top at (0,0), center at (1,0), right at (1,1).
            panels = [(0, 0, top), (1, 0, center), (1, 1, right)]
        elif has_top:
            # Vertical stack: top at (0,0), center at (1,0).
            panels = [(0, 0, top), (1, 0, center)]
        else:
            # Horizontal stack: center at (0,0), right at (0,1).
            panels = [(0, 0, center), (0, 1, right)]

        return _merge_child_scenes_nonuniform_grid(
            panels,
            self.spacing,
            figure_chrome=self._figure_chrome_kwargs(),
        )

    def to_svg(self) -> str:
        """Render the joint chart to an SVG string.

        Returns
        -------
        str
            SVG markup with the 2 × 2 grid layout.
        """
        from ferrum._core import compose_svg_grid

        # F20: the Rust grid compositor now honors row_ratios/col_ratios via
        # viewBox-scaled per-panel wrappers, so marginals can be passed at
        # their native size and the compositor handles proportional sizing.
        # The marginals still suppress their own axis decoration — the
        # data axis is redundant against the centre panel and the marginal-
        # only axis (count/density on a thin strip) is illegible at marginal
        # size.
        center = self._inject_parent_config(self.center)
        top = self._inject_parent_config(self.top) if self.top is not None else None
        right = self._inject_parent_config(self.right) if self.right is not None else None
        top_chart = top.axis(show=False) if top is not None else None
        right_chart = right.axis(show=False) if right is not None else None
        top_svg = top_chart.to_svg() if top_chart is not None else None
        right_svg = right_chart.to_svg() if right_chart is not None else None
        panels = [top_svg, None, center.to_svg(), right_svg]
        marginal_share = 1.0 / (self.ratio + 1)
        center_share = self.ratio / (self.ratio + 1)
        chrome = chrome_kwargs(merge_configure_layers(getattr(self, "_configure_layers", None)))
        return compose_svg_grid(
            panels,
            rows=2,
            cols=2,
            row_ratios=[marginal_share, center_share],
            col_ratios=[center_share, marginal_share],
            spacing=self.spacing,
            title=self._figure_title,
            subtitle=self._figure_subtitle,
            caption=self._figure_caption,
            **chrome,
        )

    def __repr__(self) -> str:
        """Return a short developer-readable description."""
        return (
            f"JointChart(center={self.center!r}, top={self.top!r}, "
            f"right={self.right!r}, ratio={self.ratio})"
        )


class RepeatChart(_CompositeBase):
    """Repeat a template chart over a grid of row / column field combinations.

    Use ``Repeat.column``, ``Repeat.row``, or ``Repeat.layer`` typed sentinels
    in the template's ``.encode(...)`` call to mark which encoding channel
    receives the per-panel field substitution.  ``RepeatChart.expand()``
    materializes the grid into fully-resolved ``(row_field, col_field, Chart)``
    tuples.

    ``diagonal=`` provides an alternate template for panels where
    ``row_field == col_field`` (symmetric n × n repeat).  ``corner=True``
    filters the expanded grid to the lower triangle including the diagonal.

    Most users obtain a ``RepeatChart`` through ``Chart.repeat()`` or
    ``ferrum.pairplot``.

    Parameters
    ----------
    template : Chart
        Template chart whose ``Repeat.*`` placeholders are substituted per
        panel.
    row : list of str, optional
        Field names assigned to the row axis.
    column : list of str, optional
        Field names assigned to the column axis.
    layer : list of str, optional
        Field names assigned to the layer axis (for non-grid repeat layouts).
    diagonal : Chart, optional
        Alternate template used when ``row_field == col_field``.  Requires
        both *row* and *column* to be set.
    corner : bool, default False
        When ``True``, only the lower-triangle panels (``ri >= ci``) are
        rendered, giving a half-matrix layout.
    spacing : float, default 10.0
        Pixel gap between adjacent panels.
    columns : int, optional
        Maximum number of columns for a wrapped 1-D repeat layout (no-op
        for 2-D row/column repeat).
    resolve : dict, optional
        Per-channel scale-sharing overrides — e.g.
        ``resolve={"x": "shared", "y": "independent"}``.  ``"shared"``
        computes the union domain across all panels (and across every
        layer of layered panels) and injects an explicit scale on every
        participating chart so the axis ticks match.  ``"independent"``
        (the default for unlisted channels) keeps per-panel domains.

    Raises
    ------
    ValueError
        If *diagonal* is set but *row* or *column* is not.

    Examples
    --------
    >>> import ferrum as fm
    >>> base = fm.Chart(df).encode(x=fm.Repeat.column, y=fm.Repeat.row).mark_point()
    >>> grid = fm.RepeatChart(base, row=["mpg", "hp"], column=["mpg", "hp"])
    >>> grid.save("pair_grid.svg")
    """

    __slots__ = (
        "template",
        "row",
        "column",
        "layer",
        "diagonal",
        "corner",
        "spacing",
        "columns",
        "resolve",
    )

    def __init__(
        self,
        template,
        *,
        row=None,
        column=None,
        layer=None,
        diagonal=None,
        corner: bool = False,
        spacing: float = 10.0,
        columns: Optional[int] = None,
        resolve=None,
    ) -> None:
        if diagonal is not None and (row is None or column is None):
            raise ValueError("RepeatChart: diagonal= requires both row= and column= to be set")
        if corner and (row is None or column is None):
            raise ValueError("RepeatChart: corner=True requires both row= and column= to be set")
        if row is None and column is None and layer is None:
            raise ValueError("RepeatChart: at least one of row=, column=, or layer= must be set")
        if columns is not None and columns <= 0:
            raise ValueError(f"RepeatChart: columns must be > 0; got {columns}")
        _validate_resolve(resolve, "RepeatChart")
        self.template = template
        self.row = list(row) if row is not None else None
        self.column = list(column) if column is not None else None
        self.layer = list(layer) if layer is not None else None
        self.diagonal = diagonal
        self.corner = corner
        self.spacing = spacing
        self.columns = columns
        self.resolve = resolve
        self._init_figure_chrome()

    @property
    def charts(self) -> list:
        """List of Chart : Template plus diagonal (when set), in init order."""
        return [c for c in (self.template, self.diagonal) if c is not None]

    @property
    def spec(self) -> dict:
        """Dict : Serializable layout spec consumed by the SVG compositor."""
        return {
            "kind": "repeat",
            "template": _embed_chart_spec(self.template),
            "row": self.row,
            "column": self.column,
            "layer": self.layer,
            "diagonal": _embed_chart_spec(self.diagonal),
            "corner": self.corner,
            "columns": self.columns,
            "resolve": self.resolve,
            "spacing": self.spacing,
        }

    def expand(self) -> list:
        """Materialize the template into fully-resolved chart panels.

        Panel iteration shape:

        - 2-D grid (both *row* and *column* set): ``len(row) × len(column)``
          panels, optionally filtered by *corner*; *diagonal* substitutes
          the template on ``row_field == col_field`` panels.
        - 1-D wrap (only one of *row* or *column* set): the populated
          field list, paired with ``None`` on the missing axis.  Geometry
          is applied by :meth:`to_svg` driven by ``columns``.
        - Layer-only (``layer=`` set, *row* and *column* both ``None``):
          a single panel containing all layers.

        When ``layer=`` is set, each panel becomes a layered ``Chart``
        with one layer per element in ``self.layer`` (substituted into
        every ``Repeat.layer`` placeholder).  Diagonal panels skip
        layering — the diagonal template already defines that panel.

        Returns
        -------
        list of tuple
            Each element is ``(row_field, col_field, Chart)`` with all
            ``Repeat.*`` placeholders replaced.  For 1-D and layer-only
            layouts the unused axis is ``None``.

        Raises
        ------
        ValueError
            If *diagonal* is set but ``row != column`` (asymmetric
            repeat), or if the template references a ``Repeat.*``
            placeholder for an axis that was not populated.
        """
        panels = [
            (row_field, col_field, self._make_panel(row_field, col_field))
            for row_field, col_field in self._panel_coordinates()
        ]
        return self._apply_resolve(panels)

    def _apply_resolve(self, panels: list) -> list:
        """Inject shared scales onto every panel per ``self.resolve``.

        For each channel marked ``"shared"``, walks every panel (and every
        layer of layered panels), computes the union domain, and re-emits
        each panel with an explicit ``scale=`` dict on that channel.
        ``"independent"`` channels are no-ops.  When no panel binds a
        shared channel the channel is silently skipped — sharing a
        channel that nothing uses is harmless.
        """
        if not self.resolve:
            return panels
        from ferrum._scale_share import compute_union_domain, inject_scale

        shared = [ch for ch, mode in self.resolve.items() if mode == "shared"]
        if not shared:
            return panels
        result = list(panels)
        for channel in shared:
            charts = [chart for _, _, chart in result]
            scale_dict = compute_union_domain(charts, channel)
            if scale_dict is None:
                continue
            result = [
                (row_field, col_field, inject_scale(chart, channel, scale_dict))
                for row_field, col_field, chart in result
            ]
        return result

    def _panel_coordinates(self) -> list:
        """Compute ``(row_field, col_field)`` pairs for every panel.

        Either entry is ``None`` when the corresponding axis is unset
        (1-D wrap) or both are ``None`` (layer-only).
        """
        if self.row is not None and self.column is not None:
            if self.diagonal is not None and self.row != self.column:
                raise ValueError(
                    "RepeatChart: diagonal= requires row == column "
                    "(diagonal panels only exist on a symmetric grid); "
                    f"got row={self.row!r}, column={self.column!r}"
                )
            coords = []
            for ri, row_field in enumerate(self.row):
                for ci, col_field in enumerate(self.column):
                    if self.corner and ri < ci:
                        continue
                    coords.append((row_field, col_field))
            return coords
        if self.column is not None:
            return [(None, f) for f in self.column]
        if self.row is not None:
            return [(f, None) for f in self.row]
        # layer-only: __init__ already ruled out the all-None axes case.
        return [(None, None)]

    def _make_panel(self, row_field: Optional[str], col_field: Optional[str]):
        """Build the chart for one panel, layering across ``self.layer`` if set."""
        use_diagonal = (
            self.diagonal is not None
            and self.row is not None
            and self.column is not None
            and row_field == col_field
        )
        if use_diagonal:
            # Diagonal panels are intentional overrides; skip layering.
            return self._resolve_template(self.diagonal, row_field, col_field)
        if self.layer is not None:
            layers = [
                self._resolve_template(self.template, row_field, col_field, layer_field=lf)
                for lf in self.layer
            ]
            result = layers[0]
            for nxt in layers[1:]:
                result = result + nxt
            return result
        return self._resolve_template(self.template, row_field, col_field)

    def _resolve_template(
        self,
        source,
        row_field: Optional[str],
        col_field: Optional[str],
        layer_field: Optional[str] = None,
    ):
        """Clone source (a Chart) and substitute Repeat placeholders in encoding.

        Any of the field arguments may be ``None`` when the corresponding
        axis is unset; ``_concrete_field`` raises if the template
        actually references the missing axis.
        """
        from ferrum.repeat import _RepeatPlaceholder
        from ferrum.encoding.base import ChannelBase

        new = source._clone()
        for axis, ch in list(new._encoding.items()):
            if isinstance(ch, _RepeatPlaceholder):
                concrete = self._concrete_field(ch.field, row_field, col_field, layer_field)
                from ferrum.chart import _channel_class_for

                cls = _channel_class_for(axis) or _channel_class_for("x")
                new._encoding[axis] = cls(concrete)
            elif isinstance(ch, ChannelBase) and isinstance(ch.field, _RepeatPlaceholder):
                concrete = self._concrete_field(ch.field.field, row_field, col_field, layer_field)
                new._encoding[axis] = ch.__class__(concrete)
        return new

    @staticmethod
    def _concrete_field(
        placeholder_axis: str,
        row_field: Optional[str],
        col_field: Optional[str],
        layer_field: Optional[str] = None,
    ) -> str:
        """Map a Repeat placeholder axis name to the concrete field string.

        Raises ``ValueError`` if the template references a placeholder for
        an axis that was not populated on the ``RepeatChart`` (e.g.
        ``Repeat.row`` in a column-only 1-D repeat, or ``Repeat.layer``
        without ``layer=``).
        """
        if placeholder_axis == "column":
            if col_field is None:
                raise ValueError("RepeatChart: template uses Repeat.column but column= was not set")
            return col_field
        if placeholder_axis == "row":
            if row_field is None:
                raise ValueError("RepeatChart: template uses Repeat.row but row= was not set")
            return row_field
        if placeholder_axis == "layer":
            if layer_field is None:
                raise ValueError("RepeatChart: template uses Repeat.layer but layer= was not set")
            return layer_field
        raise ValueError(f"unknown Repeat placeholder axis '{placeholder_axis}'")

    def _rebuild_with_charts(self, fn):
        new = RepeatChart(
            fn(self.template),
            row=self.row,
            column=self.column,
            layer=self.layer,
            diagonal=(fn(self.diagonal) if self.diagonal is not None else None),
            corner=self.corner,
            spacing=self.spacing,
            columns=self.columns,
            resolve=self.resolve,
        )
        self._carry_figure_chrome(new)
        return new

    def share_scale(self, **channels):
        """Share scales across this repeat's panels by merging into ``resolve=``.

        Equivalent to constructing the chart with ``resolve={...}`` set
        — both paths run through :meth:`_apply_resolve` at ``expand()``
        time, so the union-domain computation sees every panel (including
        each layer of layered panels) exactly once.  Passing the same
        channel twice with different modes takes the call's value.

        Parameters
        ----------
        **channels : str
            Channel name → ``"shared"`` | ``"independent"``.

        Returns
        -------
        RepeatChart
            A new ``RepeatChart`` with the merged ``resolve=`` config.
        """
        _validate_share_modes(channels)
        merged = dict(self.resolve or {})
        merged.update(channels)
        result = RepeatChart(
            self.template,
            row=self.row,
            column=self.column,
            layer=self.layer,
            diagonal=self.diagonal,
            corner=self.corner,
            spacing=self.spacing,
            columns=self.columns,
            resolve=merged or None,
        )
        _copy_configure_layers(self, result)
        self._carry_figure_chrome(result)
        return result

    def _render_interactive(self) -> tuple[str, bytes]:
        """Render to (scene_json, packed_data) by expanding panels and merging scenes."""
        panels = [(r, c, self._inject_parent_config(chart)) for r, c, chart in self.expand()]

        if self.corner and self.row is not None and self.column is not None:
            # Corner mode: panels must be placed at their true (row, col) grid
            # coordinates with gaps for the upper triangle.  Map field names
            # back to integer indices for the sparse grid merge.
            row_index = {v: i for i, v in enumerate(self.row)}
            col_index = {v: i for i, v in enumerate(self.column)}
            indexed = [(row_index[r], col_index[c], chart) for r, c, chart in panels]
            return _merge_child_scenes_sparse_grid(
                indexed,
                self.spacing,
                figure_chrome=self._figure_chrome_kwargs(),
            )

        expanded_charts = [chart for _, _, chart in panels]
        if self.row is not None and self.column is not None:
            n_cols = len(self.column)
        else:
            n_cols, _ = self._wrap_dimensions(len(expanded_charts))
        return _merge_child_scenes_grid(
            expanded_charts,
            self.spacing,
            columns=n_cols,
            figure_chrome=self._figure_chrome_kwargs(),
        )

    def to_svg(self) -> str:
        """Render the repeated grid to an SVG string.

        Returns
        -------
        str
            SVG markup containing all materialized panel charts in a grid.

        Notes
        -----
        2-D grids (both ``row`` and ``column`` set) lay out as
        ``len(row) × len(column)``.  1-D layouts (only one axis set) wrap
        by ``columns`` — column-only spreads left-to-right and wraps
        downward; row-only spreads top-to-bottom in a single column unless
        ``columns`` opens additional columns.  When ``columns`` is unset
        the 1-D layout is a single row (column-only) or column (row-only).
        """
        from ferrum._core import compose_svg_grid

        panels = [(r, c, self._inject_parent_config(chart)) for r, c, chart in self.expand()]
        if self.row is not None and self.column is not None:
            n_rows = len(self.row)
            n_cols = len(self.column)
            grid: list = [None] * (n_rows * n_cols)
            for row_field, col_field, chart in panels:
                ri = self.row.index(row_field)
                ci = self.column.index(col_field)
                grid[ri * n_cols + ci] = chart.to_svg()
        else:
            n_panels = len(panels)
            n_cols, n_rows = self._wrap_dimensions(n_panels)
            grid = [None] * (n_rows * n_cols)
            for idx, (_, _, chart) in enumerate(panels):
                grid[idx] = chart.to_svg()
        chrome = chrome_kwargs(merge_configure_layers(getattr(self, "_configure_layers", None)))
        return compose_svg_grid(
            grid,
            rows=n_rows,
            cols=n_cols,
            row_ratios=[1.0] * n_rows,
            col_ratios=[1.0] * n_cols,
            spacing=self.spacing,
            title=self._figure_title,
            subtitle=self._figure_subtitle,
            caption=self._figure_caption,
            **chrome,
        )

    def _wrap_dimensions(self, n_panels: int) -> tuple:
        """Compute ``(n_cols, n_rows)`` for a 1-D wrapped layout.

        ``columns=`` is honored when set; otherwise column-only repeats
        produce a single row and row-only repeats produce a single column.
        """
        if self.columns is not None:
            n_cols = min(self.columns, n_panels)
        elif self.column is not None:
            n_cols = n_panels  # horizontal default: one row
        else:
            n_cols = 1  # vertical default: one column
        n_cols = max(1, n_cols)
        n_rows = (n_panels + n_cols - 1) // n_cols
        return n_cols, n_rows

    def __repr__(self) -> str:
        """Return a short developer-readable description."""
        return (
            f"RepeatChart(row={self.row}, column={self.column}, "
            f"diagonal={'set' if self.diagonal is not None else 'None'}, corner={self.corner})"
        )


class ClusterMapChart(_CompositeBase):
    """Clustered heatmap with optional row and column dendrograms.

    Lays out a 2 × 2 grid: the heatmap occupies the bottom-right panel,
    the column dendrogram goes top-right, the row dendrogram (rotated 90°)
    goes bottom-left, and the top-left corner is empty.  Dendrogram value
    axes are hidden; categorical axes align with the heatmap row/column labels.

    Panel size is split by ``dendrogram_ratio``: dendrograms receive that
    fraction of the total width/height, the heatmap receives the remainder.

    Most users obtain a ``ClusterMapChart`` from `ferrum.clustermap` rather
    than constructing one directly.

    Parameters
    ----------
    heatmap : Chart
        The central heatmap chart.
    row_dendrogram : Chart, optional
        Dendrogram chart for the row axis.  Displayed to the left of the
        heatmap, rotated 90°.
    col_dendrogram : Chart, optional
        Dendrogram chart for the column axis.  Displayed above the heatmap.
    dendrogram_ratio : float, default 0.2
        Fraction of the total width/height allocated to each dendrogram panel.
        Must be in the open interval (0, 1).
    spacing : float, default 10.0
        Pixel gap between adjacent panels.

    Raises
    ------
    ValueError
        If *dendrogram_ratio* is not in the open interval (0, 1).

    Examples
    --------
    >>> import ferrum as fm
    >>> cm = fm.clustermap(df, method="ward", cmap="rdbu")
    >>> cm.save("clustermap.svg")
    """

    __slots__ = (
        "heatmap",
        "row_dendrogram",
        "col_dendrogram",
        "dendrogram_ratio",
        "spacing",
    )

    def __init__(
        self,
        heatmap,
        *,
        row_dendrogram=None,
        col_dendrogram=None,
        dendrogram_ratio: float = 0.2,
        spacing: float = 10.0,
    ) -> None:
        if not (0.0 < dendrogram_ratio < 1.0):
            raise ValueError(f"dendrogram_ratio must be in (0, 1); got {dendrogram_ratio}")
        self.heatmap = heatmap
        self.row_dendrogram = row_dendrogram
        self.col_dendrogram = col_dendrogram
        self.dendrogram_ratio = dendrogram_ratio
        self.spacing = spacing
        self._init_figure_chrome()

    @property
    def charts(self) -> list:
        """List of Chart : All non-None sub-charts in ``__init__`` order
        (heatmap, row_dendrogram, col_dendrogram).
        """
        return [
            c for c in (self.heatmap, self.row_dendrogram, self.col_dendrogram) if c is not None
        ]

    @property
    def spec(self) -> dict:
        """Dict : Serializable layout spec consumed by the SVG compositor."""
        return {
            "kind": "cluster_map",
            "heatmap": _embed_chart_spec(self.heatmap),
            "row_dendrogram": _embed_chart_spec(self.row_dendrogram),
            "col_dendrogram": _embed_chart_spec(self.col_dendrogram),
            "dendrogram_ratio": self.dendrogram_ratio,
            "spacing": self.spacing,
        }

    def _forward_child_properties(self, kwargs: dict) -> "ClusterMapChart":
        """Route non-chrome ``properties`` kwargs to the heatmap chart only.

        The dendrogram panels are kept unchanged because their width / height
        is derived from the heatmap plus ``dendrogram_ratio`` at render time.
        Figure-level chrome (``title`` / ``subtitle`` / ``caption``) is
        intercepted by :meth:`_CompositeBase.properties` before this hook
        runs, so it never reaches the heatmap panel.
        """
        result = ClusterMapChart(
            self.heatmap.properties(**kwargs),
            row_dendrogram=self.row_dendrogram,
            col_dendrogram=self.col_dendrogram,
            dendrogram_ratio=self.dendrogram_ratio,
            spacing=self.spacing,
        )
        self._carry_figure_chrome(result)
        return result

    def _rebuild_with_charts(self, fn):
        new = ClusterMapChart(
            fn(self.heatmap),
            row_dendrogram=(fn(self.row_dendrogram) if self.row_dendrogram is not None else None),
            col_dendrogram=(fn(self.col_dendrogram) if self.col_dendrogram is not None else None),
            dendrogram_ratio=self.dendrogram_ratio,
            spacing=self.spacing,
        )
        self._carry_figure_chrome(new)
        return new

    def _render_interactive(self) -> tuple[str, bytes]:
        """Render to (scene_json, packed_data) by merging child scenes in a 2x2 grid.

        Grid layout mirrors the SVG path (``to_svg``):
          - col_dendrogram at (row=0, col=1) -- above heatmap
          - row_dendrogram at (row=1, col=0) -- left of heatmap
          - heatmap        at (row=1, col=1) -- main content
        """
        heatmap = self._inject_parent_config(self.heatmap)
        row_dendro = (
            self._inject_parent_config(self.row_dendrogram)
            if self.row_dendrogram is not None
            else None
        )
        col_dendro = (
            self._inject_parent_config(self.col_dendrogram)
            if self.col_dendrogram is not None
            else None
        )
        has_row = row_dendro is not None
        has_col = col_dendro is not None

        # No dendrograms: render heatmap directly (still wrap the figure band).
        if not has_row and not has_col:
            return _render_single_with_figure_chrome(heatmap, self._figure_chrome_kwargs())

        # Build grid panels matching the SVG path layout.
        panels: list[tuple[int, int, object]] = []
        if has_row and has_col:
            # Full 2x2: col_dendro at (0,1), row_dendro at (1,0), heatmap at (1,1).
            panels = [
                (0, 1, col_dendro),
                (1, 0, row_dendro),
                (1, 1, heatmap),
            ]
        elif has_row:
            # Horizontal: row_dendro at (0,0), heatmap at (0,1).
            panels = [(0, 0, row_dendro), (0, 1, heatmap)]
        else:
            # Vertical: col_dendro at (0,0), heatmap at (1,0).
            panels = [(0, 0, col_dendro), (1, 0, heatmap)]

        return _merge_child_scenes_nonuniform_grid(
            panels,
            self.spacing,
            figure_chrome=self._figure_chrome_kwargs(),
        )

    def to_svg(self) -> str:
        """Render the cluster map to an SVG string.

        Returns
        -------
        str
            SVG markup with the 2 × 2 grid layout.
        """
        from ferrum._core import compose_svg_grid

        heatmap = self._inject_parent_config(self.heatmap)
        d = self.dendrogram_ratio
        h = 1.0 - d
        # Pre-resize each component so the heatmap fills (h × h) of the grid
        # and dendrograms occupy the remaining (d) on the row/col axis they
        # sit beside. Post-F20 the compositor honors row_ratios/col_ratios,
        # but we still pre-resize because the dendrogram tree topology depends
        # on the panel viewport at SVG-emit time — letting the compositor
        # rescale after-the-fact would distort branch positions.
        hm_w = heatmap._width or 600.0
        hm_h = heatmap._height or 400.0
        dendro_w = hm_w * d / h
        dendro_h = hm_h * d / h
        # Dendrograms have no meaningful axes (only the tree structure
        # matters). clustermap() already calls .axis(show=False) on each
        # dendrogram chart at construction time, so spec-level suppression
        # is in effect here — no post-render SVG mangling needed.
        col_dendro = (
            self._inject_parent_config(self.col_dendrogram).properties(width=hm_w, height=dendro_h)
            if self.col_dendrogram is not None
            else None
        )
        row_dendro = (
            self._inject_parent_config(self.row_dendrogram).properties(width=dendro_w, height=hm_h)
            if self.row_dendrogram is not None
            else None
        )
        col_svg = col_dendro.to_svg() if col_dendro is not None else None
        row_svg = row_dendro.to_svg() if row_dendro is not None else None
        panels = [None, col_svg, row_svg, heatmap.to_svg()]
        chrome = chrome_kwargs(merge_configure_layers(getattr(self, "_configure_layers", None)))
        return compose_svg_grid(
            panels,
            rows=2,
            cols=2,
            row_ratios=[d, h],
            col_ratios=[d, h],
            spacing=self.spacing,
            title=self._figure_title,
            subtitle=self._figure_subtitle,
            caption=self._figure_caption,
            **chrome,
        )

    def __repr__(self) -> str:
        """Return a short developer-readable description."""
        return (
            f"ClusterMapChart(heatmap=set, row_dendrogram={'set' if self.row_dendrogram else 'None'}, "
            f"col_dendrogram={'set' if self.col_dendrogram else 'None'}, "
            f"ratio={self.dendrogram_ratio})"
        )


# ---------------------------------------------------------------------------
# Phase 12: LayerChart and ConcatChart
# ---------------------------------------------------------------------------


class LayerChart(_ChartLike):
    """Overlay multiple charts on shared axes (same coordinate space).

    All layers share x/y scales by default (union domain).  The charts
    are merged using the same ``Chart + Chart`` layer-merge logic that
    the ``+`` operator provides — domain union, null-padded diagonal
    concat for heterogeneous data, named-transform routing for per-layer
    transforms.

    Use ``LayerChart`` when you have pre-built ``Chart`` objects and want
    a composition-level overlay without constructing the ``+`` chain
    inline.  The resulting SVG is rendered as a single plot area with
    all layers stacked.

    Parameters
    ----------
    *charts : Chart
        Two or more charts to overlay.  At least one chart is required.
    resolve : dict, optional
        Per-channel scale-sharing overrides — e.g.
        ``resolve={"color": "independent"}``.  By default all positional
        channels (x, y) are shared (union domain); non-positional
        channels follow the same inheritance rules as ``Chart + Chart``.
    title : str, optional
        Title applied to the combined chart via ``.properties(title=...)``.

    Raises
    ------
    ValueError
        If fewer than one chart is provided, or if ``resolve`` contains
        invalid values.

    Examples
    --------
    >>> import ferrum as fm
    >>> scatter = fm.Chart(df).mark_point().encode(x="x", y="y")
    >>> line = fm.Chart(df).mark_line().encode(x="x", y="y")
    >>> fm.LayerChart(scatter, line).save("overlay.svg")
    """

    __slots__ = ("_charts", "_resolve", "_title")

    def __init__(
        self,
        *charts,
        resolve: Optional[Dict[str, str]] = None,
        title: Optional[str] = None,
    ) -> None:
        if len(charts) < 1:
            raise ValueError("LayerChart requires at least one chart")
        _validate_resolve(resolve, "LayerChart")
        self._charts = list(charts)
        self._resolve = resolve
        self._title = title

    def __copy__(self):
        """Shallow copy that duplicates the mutable _charts list."""
        new = _shallow_copy_composite(self)
        # _shallow_copy_composite copies _charts as the same list reference;
        # make it a fresh copy so mutations to the original don't affect the copy.
        new._charts = list(self._charts)
        return new

    @property
    def charts(self) -> list:
        """List of Chart : All member charts in layer order (bottom to top)."""
        return list(self._charts)

    def _render_interactive(self) -> tuple[str, bytes]:
        """Render to (scene_json, packed_data) via the merged multi-layer Chart."""
        from ferrum._interactive import _render_scene

        merged = self._build_merged()
        return _render_scene(merged)

    def to_svg(self) -> str:
        """Render the layered charts to an SVG string.

        Merges all layers using the ``Chart + Chart`` operator which
        handles domain union, data merging, and transform routing, then
        renders the resulting multi-layer chart to SVG.

        Returns
        -------
        str
            SVG markup with all layers rendered in a single plot area.
        """
        merged = self._build_merged()
        return merged.to_svg()

    def _build_merged(self):
        """Merge member charts into a single multi-layer Chart via ``+``.

        Applies ``resolve=`` scale sharing, ``title=``, and composition-level
        configure layers when set.
        """
        result = self._charts[0]
        for chart in self._charts[1:]:
            result = result + chart
        if self._resolve:
            shared = [ch for ch, mode in self._resolve.items() if mode == "shared"]
            if shared:
                from ferrum._scale_share import compute_union_domain, inject_scale

                for channel in shared:
                    sd = compute_union_domain(self._charts, channel)
                    if sd is not None:
                        result = inject_scale(result, channel, sd)
        if self._title is not None:
            result = result.properties(title=self._title)
        # Prepend composition-level configure layers so per-chart config wins.
        result = self._inject_parent_config(result)
        return result

    def properties(self, **kwargs):
        """Forward non-chrome ``properties(**kwargs)`` to every layer; store ``title`` locally.

        ``LayerChart`` is a single-plot overlay: it merges its layers into one
        ``Chart`` at render time via :meth:`_build_merged`, which already applies
        ``self._title`` to the merged chart.  Because of that, ``title`` must be
        stored on the ``LayerChart`` itself (not fanned to the inner charts), so that:

        - :meth:`_figure_title_text` (→ ``_title``) returns the correct text for
          the HTML document ``<title>``.
        - :meth:`_build_merged` applies the title to the merged chart's on-plot
          chrome exactly once — inner layers carry no stray title.

        Non-chrome kwargs (``width``, ``height``, ...) are fanned to every layer as
        usual via the base :meth:`_ChartLike.properties` implementation.

        Parameters
        ----------
        **kwargs
            Same keyword arguments accepted by ``Chart.properties``.  ``title``
            is intercepted here; all other kwargs are forwarded to each layer.

        Returns
        -------
        LayerChart
            A new ``LayerChart`` instance with ``_title`` updated and / or
            per-layer properties applied.
        """
        title = kwargs.pop("title", None)

        if kwargs:
            # Fan non-chrome kwargs to every layer.
            result = self._rebuild_with_charts(lambda c: c.properties(**kwargs))
            _copy_configure_layers(self, result)
        else:
            # Nothing to fan — preserve layers unchanged.
            result = copy.copy(self)
            _copy_configure_layers(self, result)

        if title is not None:
            result._title = title

        return result

    def _rebuild_with_charts(self, fn):
        return LayerChart(
            *[fn(c) for c in self._charts],
            resolve=self._resolve,
            title=self._title,
        )

    def __repr__(self) -> str:
        """Return a short developer-readable description."""
        n = len(self._charts)
        return f"LayerChart({n} layer{'s' if n != 1 else ''})"


class ConcatChart(_CompositeBase):
    """General wrapping concatenation of charts in a grid.

    Arranges charts left-to-right, wrapping to the next row after
    ``columns`` charts.  When ``columns`` is ``None``, all charts are
    placed in a single row.

    Parameters
    ----------
    *charts : Chart
        Two or more charts to arrange.
    columns : int, optional
        Maximum number of columns before wrapping.  Defaults to
        ``len(charts)`` (single row, no wrapping).
    spacing : float, default 10.0
        Pixel gap between adjacent panels.
    resolve : dict, optional
        Per-channel scale-sharing overrides — e.g.
        ``resolve={"x": "shared", "y": "shared"}``.

    Raises
    ------
    ValueError
        If fewer than one chart is provided, if ``columns`` is not > 0,
        or if ``resolve`` contains invalid values.

    Examples
    --------
    >>> import ferrum as fm
    >>> charts = [fm.Chart(df).mark_point().encode(x=col, y="y") for col in cols]
    >>> fm.ConcatChart(*charts, columns=2).save("grid.svg")
    """

    __slots__ = ("_columns", "_resolve")

    def __init__(
        self,
        *charts,
        columns: Optional[int] = None,
        spacing: float = 10.0,
        resolve: Optional[Dict[str, str]] = None,
    ) -> None:
        if len(charts) < 1:
            raise ValueError("ConcatChart requires at least one chart")
        if columns is not None and columns <= 0:
            raise ValueError(f"ConcatChart: columns must be > 0; got {columns}")
        _validate_resolve(resolve, "ConcatChart")
        super().__init__(list(charts), spacing=spacing)
        self._columns = columns
        self._resolve = resolve

    @property
    def columns(self) -> Optional[int]:
        """Number of columns in the wrapping grid, or None for single-row."""
        return self._columns

    def _render_interactive(self) -> tuple[str, bytes]:
        """Render to (scene_json, packed_data) by merging child scenes in a grid."""
        render_charts = [self._inject_parent_config(c) for c in self._resolved_charts()]
        n_cols = self._columns if self._columns is not None else len(render_charts)
        n_cols = min(n_cols, len(render_charts))
        return _merge_child_scenes_grid(
            render_charts,
            self.spacing,
            columns=n_cols,
            figure_chrome=self._figure_chrome_kwargs(),
        )

    def to_svg(self) -> str:
        """Render the concatenated charts to an SVG string.

        Returns
        -------
        str
            SVG markup with charts arranged in a wrapping grid.
        """
        from ferrum._core import compose_svg_grid

        n_panels = len(self.charts)
        n_cols = self._columns if self._columns is not None else n_panels
        n_cols = min(n_cols, n_panels)
        n_rows = (n_panels + n_cols - 1) // n_cols

        # Apply resolve (shared scales) and composition-level config before rendering
        render_charts = [self._inject_parent_config(c) for c in self._resolved_charts()]

        grid: list = [None] * (n_rows * n_cols)
        for idx, chart in enumerate(render_charts):
            grid[idx] = chart.to_svg()

        chrome = chrome_kwargs(merge_configure_layers(getattr(self, "_configure_layers", None)))
        return compose_svg_grid(
            grid,
            rows=n_rows,
            cols=n_cols,
            row_ratios=[1.0] * n_rows,
            col_ratios=[1.0] * n_cols,
            spacing=self.spacing,
            title=self._figure_title,
            subtitle=self._figure_subtitle,
            caption=self._figure_caption,
            **chrome,
        )

    def _resolved_charts(self) -> list:
        """Return charts with shared scales injected per ``resolve``."""
        return _apply_resolve(self.charts, self._resolve)

    def _rebuild_with_charts(self, fn):
        new = ConcatChart(
            *[fn(c) for c in self.charts],
            columns=self._columns,
            spacing=self.spacing,
            resolve=self._resolve,
        )
        # Carry figure-level chrome through rebuilds.
        self._carry_figure_chrome(new)
        return new

    def __repr__(self) -> str:
        """Return a short developer-readable description."""
        n = len(self.charts)
        return f"ConcatChart({n} chart{'s' if n != 1 else ''}, columns={self._columns})"


# ---------------------------------------------------------------------------
# Interactive scene-merging helpers (composition → WASM renderer)
# ---------------------------------------------------------------------------


@dataclass
class _PlacedChild:
    """One child's rendered scene + packed bytes at its computed placement.

    Carries the per-child ``(panel_id_offset, dx, dy, scene, packed)`` placement
    as a single record so the packed-instance offset cannot drift from the
    scene-node offset (both read ``dx``/``dy`` from the same field).  Built by
    each ``_merge_child_scenes*`` variant's placement loop and consumed by
    :func:`_assemble_placed_children`.
    """

    scene: dict  # parsed child scene JSON
    packed: bytes  # child packed GPU-instance bytes
    dx: float  # lateral placement offset
    dy: float  # vertical placement offset
    panel_id_offset: int  # cumulative panel-id base for this child


def _assemble_placed_children(
    placed: list["_PlacedChild"],
    width: float,
    height: float,
    figure_chrome: Optional["_FigureChrome"],
) -> tuple[str, bytes]:
    """Merge placed children into one scene + packed buffer.

    Builds a fresh :func:`_empty_scene`, sets *width*/*height*, runs
    :func:`_merge_one_child` for each placement in order, injects figure chrome
    (only when *figure_chrome* is not ``None``, after the merge loop), then
    merges packed bytes with ``y_offset = chrome_top_h``.  Returns
    ``(scene_json, packed)``.

    Only ever called on the non-empty path; each variant keeps its own
    empty-children early-return guard.
    """
    merged = _empty_scene()
    merged["width"] = width
    merged["height"] = height

    for pc in placed:
        _merge_one_child(merged, pc.scene, pc.dx, pc.dy, pc.panel_id_offset)

    # chrome_top_h = title+subtitle band height; chrome_bottom_h = caption band height;
    # together they are the figure chrome.
    chrome_top_h = 0.0
    if figure_chrome is not None:
        chrome_top_h = _inject_figure_chrome(merged, **figure_chrome)

    merged_packed = _merge_packed_data(
        [pc.packed for pc in placed],
        [pc.panel_id_offset for pc in placed],
        [(pc.dx, pc.dy) for pc in placed],
        y_offset=chrome_top_h,
    )
    return _json.dumps(merged), merged_packed


def _render_charts(charts: list) -> list[tuple[dict, bytes]]:
    """Render each chart and return a list of ``(scene_dict, packed)`` pairs.

    Shared by the four ``_merge_child_scenes*`` helpers so the
    ``_render_scene`` + ``_json.loads`` prelude is not copy-pasted.
    The caller is responsible for its own empty-input guard (COMP-03).
    """
    from ferrum._interactive import _render_scene

    return [
        (_json.loads(scene_json), packed)
        for scene_json, packed in (_render_scene(c) for c in charts)
    ]


def _merge_child_scenes(
    charts: list,
    spacing: float,
    layout: str = "horizontal",
    *,
    figure_chrome: Optional["_FigureChrome"] = None,
) -> tuple[str, bytes]:
    """Render each child chart and merge their scene JSONs.

    Parameters
    ----------
    charts : list
        Child charts to render.
    spacing : float
        Pixel gap between charts.
    layout : ``"horizontal"`` or ``"vertical"``
    figure_chrome : dict, optional
        Figure-level chrome to inject as an on-canvas title band, with keys
        ``title`` / ``subtitle`` / ``caption`` and a ``chrome`` positioning
        sub-dict (see :func:`_inject_figure_chrome`).  ``None`` injects no band.

    Returns
    -------
    tuple[str, bytes]
        ``(merged_scene_json, merged_packed_data)``
    """
    rendered = _render_charts(charts)

    if not rendered:
        return _EMPTY_SCENE_JSON, b""

    x_offset = 0.0
    y_offset = 0.0
    panel_id_offset = 0
    width = 0
    height = 0
    placed: list[_PlacedChild] = []

    for scene, packed in rendered:
        dx = x_offset if layout == "horizontal" else 0.0
        dy = y_offset if layout == "vertical" else 0.0
        placed.append(_PlacedChild(scene, packed, dx, dy, panel_id_offset))
        panel_id_offset += len(scene.get("panels", []))

        w = scene.get("width", 0)
        h = scene.get("height", 0)
        if layout == "horizontal":
            x_offset += w + spacing
            width = x_offset - spacing
            height = max(height, h)
        else:
            y_offset += h + spacing
            height = y_offset - spacing
            width = max(width, w)

    return _assemble_placed_children(placed, width, height, figure_chrome)


def _merge_child_scenes_grid(
    charts: list,
    spacing: float,
    columns: int,
    *,
    figure_chrome: Optional["_FigureChrome"] = None,
) -> tuple[str, bytes]:
    """Render child charts in a wrapping grid layout.

    Arranges charts left-to-right, wrapping to the next row after
    *columns* charts.  Each row is merged horizontally, then rows
    are merged vertically.

    Parameters
    ----------
    charts : list
        Child charts to render.
    spacing : float
        Pixel gap between charts.
    columns : int
        Number of columns before wrapping.
    figure_chrome : dict, optional
        Figure-level chrome band to inject (see :func:`_inject_figure_chrome`).

    Returns
    -------
    tuple[str, bytes]
        ``(merged_scene_json, merged_packed_data)``
    """
    if not charts:
        return _EMPTY_SCENE_JSON, b""

    columns = max(1, columns)

    # Render all children up front.
    rendered: list[tuple[dict, bytes]] = _render_charts(charts)

    # Partition into rows.
    rows: list[list[tuple[dict, bytes]]] = []
    for i in range(0, len(rendered), columns):
        rows.append(rendered[i : i + columns])

    # Place each row horizontally, then stack rows vertically.
    y_offset = 0.0
    panel_id_offset = 0
    width = 0
    placed: list[_PlacedChild] = []

    for row in rows:
        row_width = 0.0
        row_height = 0.0
        x_offset = 0.0

        for scene, packed in row:
            placed.append(_PlacedChild(scene, packed, x_offset, y_offset, panel_id_offset))
            panel_id_offset += len(scene.get("panels", []))

            w = scene.get("width", 0)
            h = scene.get("height", 0)
            x_offset += w + spacing
            row_width = x_offset - spacing
            row_height = max(row_height, h)

        width = max(width, row_width)
        y_offset += row_height + spacing

    height = y_offset - spacing

    return _assemble_placed_children(placed, width, height, figure_chrome)


def _merge_child_scenes_sparse_grid(
    panels: list[tuple[int, int, object]],
    spacing: float,
    *,
    figure_chrome: Optional["_FigureChrome"] = None,
) -> tuple[str, bytes]:
    """Render child charts in a sparse grid layout (for corner-mode repeat).

    Each panel carries explicit ``(row, col)`` grid coordinates.  Panels are
    positioned at ``(col * panel_w + col * spacing, row * panel_h + row * spacing)``
    using uniform panel dimensions (the max width/height across all children).
    Grid positions without a panel (upper triangle in corner mode) are left empty.

    Parameters
    ----------
    panels : list of (row, col, chart)
        Each element is a ``(row_index, col_index, chart)`` triple.
    spacing : float
        Pixel gap between adjacent panels.
    figure_chrome : dict, optional
        Figure-level chrome band to inject (see :func:`_inject_figure_chrome`).

    Returns
    -------
    tuple[str, bytes]
        ``(merged_scene_json, merged_packed_data)``
    """
    if not panels:
        return _EMPTY_SCENE_JSON, b""

    # Render all children up front, preserving their (row, col) coordinates.
    row_cols = [(r, c) for r, c, _ in panels]
    charts = [chart for _, _, chart in panels]
    scenes_packed = _render_charts(charts)
    rendered: list[tuple[int, int, dict, bytes]] = [
        (r, c, scene, packed) for (r, c), (scene, packed) in zip(row_cols, scenes_packed)
    ]

    # Uniform panel dimensions (max across all children).
    panel_w = max(s.get("width", 0) for _, _, s, _ in rendered)
    panel_h = max(s.get("height", 0) for _, _, s, _ in rendered)

    # Grid extents.
    max_row = max(r for r, _, _, _ in rendered)
    max_col = max(c for _, c, _, _ in rendered)
    n_rows = max_row + 1
    n_cols = max_col + 1

    # Place each panel at its (row, col) position.
    panel_id_offset = 0
    placed: list[_PlacedChild] = []

    for row_idx, col_idx, scene, packed in rendered:
        dx = col_idx * (panel_w + spacing)
        dy = row_idx * (panel_h + spacing)
        placed.append(_PlacedChild(scene, packed, dx, dy, panel_id_offset))
        panel_id_offset += len(scene.get("panels", []))

    width = n_cols * panel_w + (n_cols - 1) * spacing
    height = n_rows * panel_h + (n_rows - 1) * spacing

    return _assemble_placed_children(placed, width, height, figure_chrome)


def _merge_child_scenes_nonuniform_grid(
    panels: list[tuple[int, int, object]],
    spacing: float,
    *,
    figure_chrome: Optional["_FigureChrome"] = None,
) -> tuple[str, bytes]:
    """Render child charts in a sparse grid with per-row/per-column sizing.

    Unlike ``_merge_child_scenes_sparse_grid`` which uses uniform panel
    dimensions, this variant computes the maximum width per column and
    maximum height per row, so that differently-sized children (e.g.
    marginal plots next to a center plot) occupy only as much space as
    they need.

    Parameters
    ----------
    panels : list of (row, col, chart)
        Each element is a ``(row_index, col_index, chart)`` triple.
    spacing : float
        Pixel gap between adjacent panels.
    figure_chrome : dict, optional
        Figure-level chrome band to inject (see :func:`_inject_figure_chrome`).

    Returns
    -------
    tuple[str, bytes]
        ``(merged_scene_json, merged_packed_data)``
    """
    if not panels:
        return _EMPTY_SCENE_JSON, b""

    # Render all children up front, preserving their (row, col) coordinates.
    row_cols = [(r, c) for r, c, _ in panels]
    charts = [chart for _, _, chart in panels]
    scenes_packed = _render_charts(charts)
    rendered: list[tuple[int, int, dict, bytes]] = [
        (r, c, scene, packed) for (r, c), (scene, packed) in zip(row_cols, scenes_packed)
    ]

    # Compute per-row heights and per-column widths.
    row_heights: dict[int, float] = {}
    col_widths: dict[int, float] = {}
    for row_idx, col_idx, scene, _packed in rendered:
        h = scene.get("height", 0)
        w = scene.get("width", 0)
        row_heights[row_idx] = max(row_heights.get(row_idx, 0), h)
        col_widths[col_idx] = max(col_widths.get(col_idx, 0), w)

    # Compute cumulative offsets for each row/col.
    sorted_rows = sorted(row_heights)
    sorted_cols = sorted(col_widths)
    row_y: dict[int, float] = {}
    y = 0.0
    for i, r in enumerate(sorted_rows):
        row_y[r] = y
        y += row_heights[r] + (spacing if i < len(sorted_rows) - 1 else 0)
    total_height = y

    col_x: dict[int, float] = {}
    x = 0.0
    for i, c in enumerate(sorted_cols):
        col_x[c] = x
        x += col_widths[c] + (spacing if i < len(sorted_cols) - 1 else 0)
    total_width = x

    # Place each panel at its computed position.
    panel_id_offset = 0
    placed: list[_PlacedChild] = []

    for row_idx, col_idx, scene, packed in rendered:
        dx = col_x[col_idx]
        dy = row_y[row_idx]
        placed.append(_PlacedChild(scene, packed, dx, dy, panel_id_offset))
        panel_id_offset += len(scene.get("panels", []))

    return _assemble_placed_children(placed, total_width, total_height, figure_chrome)


def _merge_one_child(
    merged: dict,
    scene: dict,
    dx: float,
    dy: float,
    panel_id_offset: int,
) -> int:
    """Merge a single child scene into *merged* at the given offset.

    Handles panels, selections, interaction conditionals, tick_levels,
    linked_panels, zoom_enabled/pan_enabled propagation,
    background, and title/legend/decoration nodes.

    Returns
    -------
    int
        The number of panels merged (so the caller can update
        ``panel_id_offset``).
    """
    _merge_scene_panels(merged, scene, dx, dy, panel_id_offset)
    n_panels = len(scene.get("panels", []))

    merged["selections"].extend(scene.get("selections", []))
    child_interaction = scene.get("interaction", {})
    merged["interaction"]["conditionals"].extend(child_interaction.get("conditionals", []))
    for tl in child_interaction.get("tick_levels", []):
        tl_copy = dict(tl)
        tl_copy["panel_id"] = tl_copy.get("panel_id", 0) + panel_id_offset
        merged["interaction"]["tick_levels"].append(tl_copy)
    for lp in child_interaction.get("linked_panels", []):
        merged["interaction"]["linked_panels"].append([p + panel_id_offset for p in lp])
    if not child_interaction.get("zoom_enabled", True):
        merged["interaction"]["zoom_enabled"] = False
    if not child_interaction.get("pan_enabled", True):
        merged["interaction"]["pan_enabled"] = False
    existing_param_names = {p["name"] for p in merged["interaction"]["params"]}
    for param in child_interaction.get("params", []):
        if param["name"] not in existing_param_names:
            merged["interaction"]["params"].append(param)
            existing_param_names.add(param["name"])
    existing_binding_keys = {
        (b["param"], b["role"], b.get("panel"), b.get("channel"))
        for b in merged["interaction"]["param_bindings"]
    }
    for binding in child_interaction.get("param_bindings", []):
        b = dict(binding)
        if b.get("panel") is not None:
            b["panel"] = b["panel"] + panel_id_offset
        key = (b["param"], b["role"], b.get("panel"), b.get("channel"))
        if key not in existing_binding_keys:
            merged["interaction"]["param_bindings"].append(b)
            existing_binding_keys.add(key)
    if merged["background"] is None and scene.get("background"):
        merged["background"] = scene["background"]

    # Offset and merge outer-level nodes (title, legend, decorations).
    for key in _OUTER_NODE_LIST_KEYS:
        for node in scene.get(key, []):
            n = copy.deepcopy(node)
            _offset_node(n, dx, dy)
            merged[key].append(n)

    return n_panels


def _inject_figure_chrome(
    merged: dict,
    *,
    title: Optional[str],
    subtitle: Optional[str],
    caption: Optional[str],
    chrome: dict,
) -> float:  # called via **figure_chrome (_FigureChrome unpacked)
    """Inject a figure-level title / subtitle / caption band into *merged*.

    This is the **single** shared implementation of the interactive on-canvas
    figure-chrome band, called by every composite scene-merge function so the
    band renders identically for HConcat / VConcat / Concat / Joint /
    ClusterMap / Repeat.  It is the interactive counterpart of the SVG band the
    Rust ``compose_svg_*`` compositors emit: the layout math (band heights,
    node x / y, anchor) lives in the Rust ``figure_title_nodes`` helper; this
    function only injects the returned nodes and offsets / grows the merged
    scene by the returned band heights.

    When no chrome text is present this is a no-op, so a composite with no
    figure title produces a byte-identical merged scene to before.

    The ``width`` / ``height`` passed to ``figure_title_nodes`` are the merged
    panels' pre-chrome bounding box (``merged["width"]`` / ``merged["height"]``
    as set by the caller before this runs).  The title/subtitle chrome-top band is
    positioned identically to SVG for all composites.  The caption (chrome-bottom)
    absolute y matches SVG for the concat family (HConcat / VConcat / Concat /
    Repeat), where the interactive body height equals the SVG body height.  For
    JointChart and ClusterMapChart the interactive body is native panel size
    rather than the ratio-scaled viewBox the SVG path uses, so the caption y
    will differ from the SVG (pre-existing W5 limitation — interactive
    nonuniform-grid layout is flat horizontal, not a 2×2 grid).

    Parameters
    ----------
    merged : dict
        The merged scene dict, with ``width`` / ``height`` already set to the
        composed panels' pre-chrome bounding box.  Mutated in place.
    title, subtitle, caption : str or None
        Figure-level chrome text.  All ``None`` -> no-op.
    chrome : dict
        Positioning kwargs (``left_inset`` / ``right_inset`` / ``anchor``)
        resolved from the composite's configure layers, matching the SVG path.

    Returns
    -------
    float
        The chrome-top band height (``chrome_top_h``, title+subtitle) applied to
        shift every panel's scene nodes down.  Returned so the caller can apply
        the **same** downward shift to the panels' packed GPU-instance bytes
        (which live in a separate binary sidecar, not in *merged*), keeping packed
        marks aligned with the shifted plot_area / axes.  ``0.0`` when no chrome
        text is present or the band has no top (caption-only), in which case no
        offset is applied anywhere.
    """
    if title is None and subtitle is None and caption is None:
        return 0.0

    from ferrum._core import figure_title_nodes

    panel_w = merged.get("width", 0) or 0.0
    panel_h = merged.get("height", 0) or 0.0
    # chrome_top_h = title+subtitle band height; chrome_bottom_h = caption band height;
    # together they are the figure chrome.
    nodes_json, chrome_top_h, chrome_bottom_h = figure_title_nodes(
        width=float(panel_w),
        height=float(panel_h),
        title=title,
        subtitle=subtitle,
        caption=caption,
        **chrome,
    )

    # Offset every child scene node DOWN by the chrome-top band height so the
    # panels sit below the title/subtitle.  The chrome nodes themselves are
    # already in outer-canvas space (caption y already includes chrome_top_h +
    # panel_h) and must NOT be offset.
    if chrome_top_h:
        for panel in merged.get("panels", []):
            for area_key in _PANEL_AREA_KEYS:
                area = panel.get(area_key)
                if area is not None:
                    area["y"] = area.get("y", 0) + chrome_top_h
            for batch in panel.get("marks", []):
                for node in batch.get("nodes", []):
                    _offset_node(node, 0.0, chrome_top_h)
            for key in _PANEL_NODE_LIST_KEYS:
                for node in panel.get(key, []):
                    _offset_node(node, 0.0, chrome_top_h)
        for key in _OUTER_NODE_LIST_KEYS:
            for node in merged.get(key, []):
                _offset_node(node, 0.0, chrome_top_h)

    # Inject the chrome nodes (already absolute) into the merged title list.
    # Note: for JointChart/ClusterMapChart the caption y is relative to the
    # interactive body height, which differs from the SVG ratio-viewBox body
    # (W5 limitation — interactive nonuniform-grid layout).
    merged.setdefault("title", []).extend(_json.loads(nodes_json))

    # Grow the merged canvas to fit the chrome-top + chrome-bottom bands
    # (width unchanged).
    merged["height"] = panel_h + chrome_top_h + chrome_bottom_h

    # Report the chrome-top shift so the caller can apply the identical downward
    # offset to the packed GPU-instance bytes (see _merge_packed_data).
    return float(chrome_top_h)


def _render_single_with_figure_chrome(chart, figure_chrome: "_FigureChrome") -> tuple[str, bytes]:
    """Render one chart's scene and wrap it in the figure-chrome band.

    Used by the asymmetric composites (Joint / ClusterMap) on their
    single-panel fast path (no marginals / dendrograms), where the body is a
    lone child scene rather than a merged grid.  When *figure_chrome* carries
    no title text this returns the child scene unchanged (byte-identical to
    ``_render_scene``), preserving backward compatibility.
    """
    from ferrum._interactive import _render_scene

    scene_json, packed = _render_scene(chart)
    if (
        figure_chrome.get("title") is None
        and figure_chrome.get("subtitle") is None
        and figure_chrome.get("caption") is None
    ):
        return scene_json, packed

    scene = _json.loads(scene_json)
    chrome_top_h = _inject_figure_chrome(scene, **figure_chrome)
    # Shift the packed GPU instances down by the chrome-top band so they stay
    # aligned with the scene nodes _inject_figure_chrome just shifted.  No
    # panel-id remap here (single child) and no lateral placement offset,
    # so child_xy is [(0.0, 0.0)].
    if chrome_top_h:
        packed = _merge_packed_data([packed], [0], [(0.0, 0.0)], y_offset=chrome_top_h)
    return _json.dumps(scene), packed


def _empty_scene() -> dict:
    """Return a skeleton scene dict for merging."""
    return {
        "width": 0,
        "height": 0,
        "background": None,
        "title": [],
        "panels": [],
        "legend": [],
        "decorations": [],
        "selections": [],
        "interaction": {
            "zoom_enabled": True,
            "pan_enabled": True,
            "conditionals": [],
            "linked_panels": [],
            "tick_levels": [],
            "params": [],
            "param_bindings": [],
        },
    }


# Canonical empty-merge return value: (scene_json, packed_bytes).
# Used by all four _merge_child_scenes* helpers on their empty-input early-return
# so the empty case shares the full scene schema with _empty_scene().
_EMPTY_SCENE_JSON: str = _json.dumps(_empty_scene())


def _merge_scene_panels(
    merged: dict,
    scene: dict,
    dx: float,
    dy: float,
    panel_id_offset: int,
) -> None:
    """Offset and append panels from *scene* into *merged*.

    Each panel is deep-copied before mutation so the original *scene*
    dict is not modified in place — callers may re-read it (e.g.
    ``_merge_one_child`` counts ``scene.get("panels", [])`` after this
    call returns).
    """
    for panel in scene.get("panels", []):
        panel = copy.deepcopy(panel)
        panel["id"] = panel.get("id", 0) + panel_id_offset

        for area_key in _PANEL_AREA_KEYS:
            area = panel.get(area_key, {})
            area["x"] = area.get("x", 0) + dx
            area["y"] = area.get("y", 0) + dy

        for batch in panel.get("marks", []):
            for node in batch.get("nodes", []):
                _offset_node(node, dx, dy)
        for key in _PANEL_NODE_LIST_KEYS:
            for node in panel.get(key, []):
                _offset_node(node, dx, dy)

        merged["panels"].append(panel)


def _offset_node(node: dict, dx: float, dy: float) -> None:
    """Offset a scene node's position by ``(dx, dy)``."""
    if dx == 0.0 and dy == 0.0:
        return
    t = node.get("type")
    if t == "circle":
        node["cx"] = node.get("cx", 0) + dx
        node["cy"] = node.get("cy", 0) + dy
    elif t == "rect":
        node["x"] = node.get("x", 0) + dx
        node["y"] = node.get("y", 0) + dy
    elif t == "line":
        node["x1"] = node.get("x1", 0) + dx
        node["y1"] = node.get("y1", 0) + dy
        node["x2"] = node.get("x2", 0) + dx
        node["y2"] = node.get("y2", 0) + dy
    elif t == "text":
        node["x"] = node.get("x", 0) + dx
        node["y"] = node.get("y", 0) + dy
    elif t == "path":
        for cmd in node.get("commands", []):
            for xkey in ("x", "cx", "c1x", "c2x"):
                if xkey in cmd:
                    cmd[xkey] = cmd[xkey] + dx
            for ykey in ("y", "cy", "c1y", "c2y"):
                if ykey in cmd:
                    cmd[ykey] = cmd[ykey] + dy
    elif t == "image":
        node["x"] = node.get("x", 0) + dx
        node["y"] = node.get("y", 0) + dy
    elif t == "polygon":
        for ring in node.get("rings", []):
            for pt in ring:
                pt[0] += dx
                pt[1] += dy
    elif t == "polyline":
        for pt in node.get("points", []):
            pt[0] += dx
            pt[1] += dy
    elif t == "group":
        for child in node.get("children", []):
            _offset_node(child, dx, dy)


# Packed GPU-instance layout (mirrors ferrum-core pack_instances.rs and the
# CircleInstance / RectInstance structs in ferrum-wasm scene_load.rs).
#   kind 0 (CircleInstance): 16 f32 = 64-byte stride; cx is f32[0] (byte 0),
#                             cy is f32[1] (byte 4).
#   kind 1 (RectInstance):   18 f32 = 72-byte stride; position_x is f32[0]
#                             (byte 0), position_y is f32[1] (byte 4).
# X is the first f32 (byte 0) and Y is the second f32 (byte 4) in both
# records.  Composition translates both fields by the per-panel (dx, dy)
# placement offset so packed instances land in the same absolute frame as
# their plot_area and scene-graph nodes.
_PACKED_INSTANCE_SIZES = {0: 64, 1: 72}  # kind -> sizeof(Instance)
_PACKED_X_FIELD_OFFSET = 0  # byte offset of the X f32 within each instance
_PACKED_Y_FIELD_OFFSET = 4  # byte offset of the Y f32 within each instance


def _offset_packed_batch_xy(
    buf: bytearray, instance_start: int, kind: int, count: int, dx: float, dy: float
) -> None:
    """Add *(dx, dy)* to the X and Y f32 of every instance in one packed batch.

    *buf* is the assembled output buffer; *instance_start* is the byte index in
    *buf* where this batch's instance data begins (just past its 20-byte
    header).  For both ``CircleInstance`` and ``RectInstance``, X is the first
    f32 (``_PACKED_X_FIELD_OFFSET`` = 0) and Y is the second f32
    (``_PACKED_Y_FIELD_OFFSET`` = 4).  When *dx* or *dy* is zero, that axis is
    skipped so callers that only shift one axis incur no extra writes.
    """
    stride = _PACKED_INSTANCE_SIZES[kind]
    for i in range(count):
        base = instance_start + i * stride
        if dx != 0.0:
            x_pos = base + _PACKED_X_FIELD_OFFSET
            (x,) = struct.unpack_from("<f", buf, x_pos)
            struct.pack_into("<f", buf, x_pos, x + dx)
        if dy != 0.0:
            y_pos = base + _PACKED_Y_FIELD_OFFSET
            (y,) = struct.unpack_from("<f", buf, y_pos)
            struct.pack_into("<f", buf, y_pos, y + dy)


def _merge_packed_data(
    packed_list: list[bytes],
    panel_id_offsets: list[int],
    child_xy_offsets: list[tuple[float, float]],
    *,
    y_offset: float = 0.0,
) -> bytes:
    """Merge packed binary data from multiple child scenes.

    Rewrites the ``panel_idx`` field in each batch's 20-byte header to
    account for the panel-id offset of each child in the composed layout.
    Also translates every packed GPU instance by the per-child composition
    offset *(child_dx, child_dy)* plus the global figure-title band
    *y_offset*, so packed (>1000-mark) circle/rect batches stay aligned with
    the scene-graph nodes and ``plot_area`` rects that
    :func:`_merge_scene_panels` and :func:`_inject_figure_chrome` place in
    the same absolute composite frame.

    When both the per-child dx/dy and the global *y_offset* are zero the
    instance bytes are left byte-identical to the unmerged child (only
    ``panel_idx`` in the header changes).

    Binary format per batch::

        [header 20B][instance_data][data_indices?][tooltips?]

    Header layout (20 bytes, all u32 little-endian)::

        [panel_idx][batch_idx][kind][count][flags]

    After the header:
    - Instance data: ``count * 64`` bytes for kind=0 (CircleInstance),
      ``count * 72`` bytes for kind=1 (RectInstance).
    - If ``flags & 0x2``: ``count * 4`` bytes of u32 data indices.
    - If ``flags & 0x1``: tooltip string table in the format produced by
      ``pack_instances.rs::pack_tooltips``:

      .. code-block:: text

          [num_fields: u32]
          num_fields   × [name_len: u32][name UTF-8 bytes]
          count×num_fields × [value_len: u32][value UTF-8 bytes]

      There is NO overall byte-length prefix.  ``count`` is the instance
      count from the batch header (field index 3).

    Parameters
    ----------
    packed_list : list[bytes]
        Packed binary data from each child scene.
    panel_id_offsets : list[int]
        Cumulative panel-id offset for each child (same length as
        *packed_list*).
    child_xy_offsets : list[tuple[float, float]]
        Per-child ``(dx, dy)`` panel-grid placement offsets (same length as
        *packed_list*).  These are the same offsets passed to
        :func:`_merge_scene_panels` for the corresponding child.  A child at
        ``(0.0, 0.0)`` with *y_offset* == 0.0 stays byte-identical.
    y_offset : float, default 0.0
        Global pixels to add to every packed instance's Y coordinate.  This
        is the ``chrome_top_h`` returned by :func:`_inject_figure_chrome` for a
        titled composite; ``0.0`` for a non-titled composite.
    """
    result = bytearray()
    for packed, offset, (child_dx, child_dy) in zip(
        packed_list, panel_id_offsets, child_xy_offsets
    ):
        if not packed:
            continue
        pos = 0
        while pos + 20 <= len(packed):
            panel_idx, batch_idx, kind, count, flags = struct.unpack_from("<5I", packed, pos)
            if kind not in _PACKED_INSTANCE_SIZES:
                break  # unknown kind — stop parsing this child

            # Rewrite panel_idx with the composition offset
            header = struct.pack("<5I", panel_idx + offset, batch_idx, kind, count, flags)

            batch_end = pos + 20 + count * _PACKED_INSTANCE_SIZES[kind]

            if flags & 0x2:
                batch_end += count * 4  # u32 data indices

            if flags & 0x1:
                # Scan the tooltip string table field-by-field, mirroring
                # scene_load.rs::unpack_binary_instances (lines ~708-735).
                # Format: [num_fields:u32] + num_fields×[len:u32][name] +
                #         count×num_fields×[len:u32][value].  No overall
                # length prefix — must walk the table to find its end.
                if batch_end + 4 > len(packed):
                    break  # truncated — stop parsing this child
                num_fields = struct.unpack_from("<I", packed, batch_end)[0]
                batch_end += 4
                # Skip field names.
                for _ in range(num_fields):
                    if batch_end + 4 > len(packed):
                        break
                    slen = struct.unpack_from("<I", packed, batch_end)[0]
                    batch_end += 4 + slen
                    if batch_end > len(packed):
                        break
                # Skip per-row values: count rows × num_fields entries.
                for _ in range(count * num_fields):
                    if batch_end + 4 > len(packed):
                        break
                    slen = struct.unpack_from("<I", packed, batch_end)[0]
                    batch_end += 4 + slen
                    if batch_end > len(packed):
                        break

            if batch_end > len(packed):
                break  # truncated data — stop parsing this child

            # Append the rewritten header + the batch body (instances + any
            # trailing data_indices / tooltips), then translate the X and Y
            # fields of every instance by the combined per-panel placement
            # offset and the global figure-title band.
            instance_start = len(result) + 20
            result.extend(header)
            result.extend(packed[pos + 20 : batch_end])
            total_dx = child_dx
            total_dy = child_dy + y_offset
            if total_dx != 0.0 or total_dy != 0.0:
                _offset_packed_batch_xy(result, instance_start, kind, count, total_dx, total_dy)

            pos = batch_end

    return bytes(result)


# ---------------------------------------------------------------------------
# Layer-composition helpers (extracted from chart.py)
# ---------------------------------------------------------------------------


def _expand_layers(c) -> tuple[list, list]:
    """Return ``(layers, top_level_transforms)`` for one side of ``Chart + Chart``.

    Composite-mark charts arrive pre-layered (``_layers`` is set, ``_mark`` is
    ``None``) -- splat their layers as-is and carry their top-level transforms
    across.  Plain single-mark charts wrap into a one-element ``_Layer`` list.

    Transforms are returned as plain PyO3 objects.  The named-transform path
    (routing a layer's output to a specific ``data_source``) is handled in
    ``__add__`` when the LHS chart has no transforms and the RHS does.

    Encoding-implicit ``_PendingAggregate`` sentinels (added to ``c._transforms``
    by ``encode()`` for channels like ``Y("v", aggregate="mean")``) are excluded
    from the returned top-level transforms.  In a layered chart each layer
    aggregates its own data independently; ``Chart.to_spec`` rebuilds these
    aggregates per-layer from each layer's encoding via
    ``_resolve_layer_aggregates``.  Leaving them at the chart top level would
    aggregate the merged batch once with the wrong (single-layer) groupby.
    """
    from ferrum._layer import _Layer
    from ferrum.encoding.base import _PendingAggregate, _PendingBin

    def _top_transforms(chart) -> list:
        return [
            t
            for t in (chart._transforms or [])
            if not isinstance(t, (_PendingAggregate, _PendingBin))
        ]

    if c._layers is not None:
        return list(c._layers), _top_transforms(c)
    return [
        _Layer(
            mark=c._mark,
            encoding=dict(c._encoding),
            transforms=[],
            mark_kwargs=dict(c._mark_kwargs) if c._mark_kwargs else None,
            position=c._position,
        )
    ], _top_transforms(c)


def _merge_top_transforms(new, rhs_top_xforms: list) -> None:
    """Merge RHS top-level transforms into the combined chart's pipeline.

    Deduplicates by identity first (fast), then by value equality
    (PyO3 transform classes implement ``__eq__`` via ``#[pyclass(eq, ...)]``;
    ``_NamedTransform`` defers to its inner transform for equality checks).
    Value deduplication prevents the same logical transform from running
    twice when both sides of ``+`` use an identical transform object.
    """
    from ferrum._layer_transforms import _NamedTransform

    existing = list(new._transforms or [])
    existing_ids = {id(t) for t in existing}
    for t in rhs_top_xforms:
        if id(t) in existing_ids:
            continue
        # Value dedup: unwrap _NamedTransform for the equality check.
        inner_t = t.transform if isinstance(t, _NamedTransform) else t
        if any(inner_t == (e.transform if isinstance(e, _NamedTransform) else e) for e in existing):
            continue
        existing.append(t)
        existing_ids.add(id(t))
    new._transforms = existing


def _warn_on_layer_conflicts(lhs, rhs) -> None:
    """Warn when layered chart ``+`` would silently discard RHS theme/facet/coord."""
    if (
        (rhs._theme is not None and rhs._theme != lhs._theme)
        or rhs._facet != lhs._facet
        or rhs._coord != lhs._coord
    ):
        import warnings

        warnings.warn(
            "Layered chart `+`: secondary layer's theme/facet/coord is ignored; "
            "primary layer wins.",
            UserWarning,
            stacklevel=3,
        )


def _promote_layer_color(new) -> None:
    """Promote the first layer's ``ChannelBase`` color encoding to chart level when absent.

    When the LHS of ``Chart + Chart`` has no color encoding, the merged chart
    inherits ``_encoding["color"] = None`` from the LHS clone.  The Rust
    renderer builds the chart-level color scale from ``spec.encoding.color``
    only, so a ``None`` chart-level color means no color scale is created —
    every layer that carries a layer-level color encoding then falls back to
    the theme default color, collapsing all categories to one.

    This function scans the merged layers in order and promotes the first
    layer that carries a ``ChannelBase`` color encoding (not a plain string
    shorthand) to the chart level, so ``build_color_scale`` can see the
    scheme and build the correct domain.

    Plain string-valued color encodings (e.g. ``"class"`` from composite-mark
    desugars like ``mark_roc``) are intentionally skipped — they are
    layer-internal shorthands and must not be promoted to chart level because
    ``_build_encoding_specs`` expects ``ChannelBase`` objects there.

    This is a no-op when:
    - the chart-level color encoding is already set (first encoding-bearing
      layer already won via the LHS clone), or
    - no layer carries a ``ChannelBase`` color encoding.
    """
    from ferrum.encoding.base import ChannelBase

    if new._encoding.get("color") is not None:
        return
    for layer in new._layers or []:
        color_ch = layer.encoding.get("color")
        if isinstance(color_ch, ChannelBase):
            new._encoding["color"] = copy.copy(color_ch)
            return
