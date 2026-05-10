"""Custom HTML shell that pairs a Plotly figure with a sidebar of links.

Used by the ``embedding_map_with_keyword_search`` plugins to render
an interactive page where hovering a label in the sidebar grows the
corresponding marker on the plot.  The sidebar's per-item
``data-trace``/``data-point`` attributes drive a small JS handler
that calls ``Plotly.restyle`` on the configured highlight trace.

The whole page is a self-contained HTML string with Plotly loaded
from a CDN, so the artifact slots into the existing ``text/html``
display pipeline (rendered in a sandboxed iframe) with no frontend
changes.
"""

from dataclasses import dataclass, field
from html import escape

import plotly.graph_objects as go


@dataclass(frozen=True)
class SidebarItem:
    """One clickable label in a sidebar.

    Attributes:
        label: Display text shown in the sidebar.
        similarity: Cosine similarity to the keyword; rendered as a
            small numeric badge.
        trace_index: Plotly trace index this item points to (used by
            JS to call ``Plotly.restyle`` on the right trace).
        point_index: Index of this item within ``trace_index``'s
            marker array.
        badge: Optional short tag (e.g. ``"current"`` / ``"past"``)
            rendered as a coloured pill before the label.
    """

    label: str
    similarity: float
    trace_index: int
    point_index: int
    badge: str | None = None


@dataclass(frozen=True)
class Sidebar:
    """One sidebar section.

    Attributes:
        title: Section heading rendered above the item list.
        items: Items in display order.
    """

    title: str
    items: list[SidebarItem] = field(default_factory=list)


_DEFAULT_HIGHLIGHT_SIZE = 14
_HOVER_HIGHLIGHT_SIZE = 26


def render_keyword_search_html(
    figure: go.Figure,
    sidebars: list[Sidebar],
    keyword: str,
    *,
    page_title: str = "Embedding map — keyword search",
    default_highlight_size: int = _DEFAULT_HIGHLIGHT_SIZE,
    hover_highlight_size: int = _HOVER_HIGHLIGHT_SIZE,
) -> str:
    """Render a self-contained HTML page wrapping *figure* with sidebars.

    Args:
        figure: Plotly figure containing at least one trace whose
            ``marker.size`` is paged through by the sidebar
            interactions.  The plot div in the rendered HTML is
            given ``id="plot"`` so JS can reference it directly.
        sidebars: One or more sidebar sections to render on the
            left.  Each item carries the ``trace_index`` /
            ``point_index`` of the marker it controls.
        keyword: The user-entered keyword; rendered in the sidebar
            heading.
        page_title: ``<title>`` element value.
        default_highlight_size: Marker size used as the "rest" size
            for highlight traces.  Must match the size each
            highlight trace was created with so resetting the
            marker after a hover keeps the original look.
        hover_highlight_size: Marker size assumed when the user
            hovers a sidebar item.

    Returns:
        str: A complete HTML page ready to write to disk.
    """
    # Sizing per trace_index lets the JS rebuild the correct-length
    # size array for each restyle call without having to ask Plotly.
    trace_sizes: dict[int, int] = {}
    for sb in sidebars:
        for item in sb.items:
            trace_sizes.setdefault(item.trace_index, 0)
            trace_sizes[item.trace_index] = max(trace_sizes[item.trace_index], item.point_index + 1)

    sidebar_html = _render_sidebars_html(sidebars, keyword)
    plot_html = figure.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        div_id="plot",
        default_width="100%",
        default_height="100%",
    )

    handler_script = _render_handler_script(
        trace_sizes=trace_sizes,
        default_size=default_highlight_size,
        hover_size=hover_highlight_size,
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(page_title)}</title>
  <style>{_CSS}</style>
</head>
<body>
  <div class="kw-container">
    <aside class="kw-sidebar">{sidebar_html}</aside>
    <main class="kw-plot-pane">{plot_html}</main>
  </div>
  <script>{handler_script}</script>
</body>
</html>
"""


_CSS = """
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; height: 100%; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #1f2937; }
.kw-container { display: flex; height: 100vh; }
.kw-sidebar { width: 32%; max-width: 380px; min-width: 220px; overflow-y: auto; padding: 12px 14px; border-right: 1px solid #e5e7eb; background: #f9fafb; }
.kw-plot-pane { flex: 1; min-width: 0; }
.kw-keyword-heading { font-size: 13px; font-weight: 500; color: #6b7280; margin: 0 0 12px; }
.kw-keyword-heading em { color: #111827; font-style: normal; font-weight: 600; }
.kw-section-title { font-size: 11px; font-weight: 600; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; margin: 14px 0 4px; }
.kw-section-title:first-of-type { margin-top: 4px; }
.kw-list { list-style: none; padding: 0; margin: 0; }
.kw-item { display: flex; align-items: baseline; gap: 6px; padding: 5px 8px; cursor: pointer; border-radius: 4px; font-size: 13px; line-height: 1.3; transition: background-color 80ms ease; }
.kw-item:hover { background: #dbeafe; }
.kw-badge { display: inline-block; padding: 1px 6px; border-radius: 999px; font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.4px; flex-shrink: 0; }
.kw-badge-current { background: #dbeafe; color: #1e40af; }
.kw-badge-past { background: #fee2e2; color: #991b1b; }
.kw-label { flex: 1; word-break: break-word; }
.kw-similarity { font-variant-numeric: tabular-nums; font-size: 11px; color: #9ca3af; flex-shrink: 0; }
"""


def _render_sidebars_html(sidebars: list[Sidebar], keyword: str) -> str:
    """Render the sidebar markup including per-section titles.

    Args:
        sidebars: Sections to render in order.
        keyword: User-entered keyword displayed in the leading
            heading.

    Returns:
        str: HTML fragment for the entire sidebar.
    """
    parts: list[str] = [
        f'<p class="kw-keyword-heading">Closest to <em>"{escape(keyword)}"</em></p>'
    ]
    for sb in sidebars:
        parts.append(f'<h3 class="kw-section-title">{escape(sb.title)}</h3>')
        parts.append('<ul class="kw-list">')
        for item in sb.items:
            badge_html = _render_badge_html(item.badge)
            parts.append(
                f'<li class="kw-item" '
                f'data-trace="{item.trace_index}" '
                f'data-point="{item.point_index}">'
                f"{badge_html}"
                f'<span class="kw-label">{escape(item.label)}</span>'
                f'<span class="kw-similarity">{item.similarity:.3f}</span>'
                f"</li>"
            )
        parts.append("</ul>")
    return "\n".join(parts)


def _render_badge_html(badge: str | None) -> str:
    """Render an optional badge pill as HTML.

    Args:
        badge: Badge text, or ``None`` to render nothing.  ``"current"``
            / ``"past"`` get a colour class; any other value renders
            as a neutral grey.

    Returns:
        str: HTML for the badge or an empty string.
    """
    if badge is None:
        return ""
    klass = {
        "current": "kw-badge kw-badge-current",
        "past": "kw-badge kw-badge-past",
    }.get(badge, "kw-badge")
    return f'<span class="{klass}">{escape(badge)}</span>'


def _render_handler_script(
    *,
    trace_sizes: dict[int, int],
    default_size: int,
    hover_size: int,
) -> str:
    """Render the inline JS that wires sidebar hovers to ``Plotly.restyle``.

    Args:
        trace_sizes: Map from highlight-trace index to its marker
            count, used to rebuild the size array of the right
            length on each restyle call.
        default_size: Marker size used as the "rest" size on
            highlight traces.
        hover_size: Marker size used while a sidebar item is
            hovered.

    Returns:
        str: JS source ready to embed in a ``<script>`` tag.
    """
    sizes_js_obj = "{" + ",".join(f"{k}:{v}" for k, v in trace_sizes.items()) + "}"
    return f"""
const TRACE_SIZES = {sizes_js_obj};
const DEFAULT_SIZE = {default_size};
const HOVER_SIZE = {hover_size};

function buildSizes(traceIdx, hoveredPointIdx) {{
  const total = TRACE_SIZES[traceIdx];
  const out = new Array(total).fill(DEFAULT_SIZE);
  if (hoveredPointIdx !== null) out[hoveredPointIdx] = HOVER_SIZE;
  return out;
}}

document.querySelectorAll('.kw-item').forEach((li) => {{
  const traceIdx = parseInt(li.dataset.trace, 10);
  const pointIdx = parseInt(li.dataset.point, 10);
  li.addEventListener('mouseenter', () => {{
    Plotly.restyle('plot', {{ 'marker.size': [buildSizes(traceIdx, pointIdx)] }}, [traceIdx]);
  }});
  li.addEventListener('mouseleave', () => {{
    Plotly.restyle('plot', {{ 'marker.size': [buildSizes(traceIdx, null)] }}, [traceIdx]);
  }});
}});
"""
