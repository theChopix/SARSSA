"""Hierarchical clustering of deduplicated neuron labels as a dendrogram.

Neurons sharing a label collapse into one leaf (``label ×N``); unlabeled
ones are dropped. Outputs an interactive Plotly HTML (hover a merge for
its subtree's labels, plus a neuron→label index for Ctrl+F) and a static
PDF that spells out every neuron id.
"""

import io
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

from plugins.labeling_evaluation._embedding_cache import embed_labels
from plugins.plugin_interface import (
    ArtifactDisplaySpec,
    ArtifactFileSpec,
    ArtifactSpec,
    BasePlugin,
    DependentDropdownHint,
    OutputArtifactSpec,
    OutputParamSpec,
    ParamGroup,
    PluginIOSpec,
    StaticDropdownHint,
)
from utils.embedder.registry import known_providers
from utils.plugin_logger import get_logger

logger = get_logger(__name__)

_ABOVE_THRESHOLD_COLOR = "#1f77b4"
_CLUSTER_PALETTE = [
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
_MAX_IDS_IN_LEAF = 8
_MAX_HOVER_LABELS = 8


def _depth_linkage(Z: np.ndarray) -> np.ndarray:
    """Copy of *Z* with heights replaced by inverted depth from root.

    Args:
        Z: SciPy linkage matrix.

    Returns:
        np.ndarray: Linkage matrix with depth-based heights.
    """
    n = Z.shape[0] + 1
    depth = {2 * n - 2: 0}
    # parents have a larger node id, so a descending sweep sees them first.
    for i in range(n - 2, -1, -1):
        d = depth[n + i]
        depth[int(Z[i, 0])] = d + 1
        depth[int(Z[i, 1])] = d + 1
    max_depth = max(depth[n + i] for i in range(n - 1))
    Zd = Z.copy()
    for i in range(n - 1):
        Zd[i, 2] = max_depth - depth[n + i] + 1
    return Zd


def _tree_geometry(Z: np.ndarray, leaf_order: list[int]) -> list[tuple]:
    """U-link coordinates for every merge of *Z*.

    Args:
        Z: Linkage matrix whose heights define the x axis.
        leaf_order: Leaves in display order.

    Returns:
        list[tuple]: Per merge ``(xs, ys)`` — the four heights and four
            leaf-space positions of the U.
    """
    n = Z.shape[0] + 1
    pos = {leaf: 5.0 + 10.0 * rank for rank, leaf in enumerate(leaf_order)}
    top = dict.fromkeys(range(n), 0.0)
    links = []
    for i in range(n - 1):
        a, b = int(Z[i, 0]), int(Z[i, 1])
        h = float(Z[i, 2])
        links.append(((top[a], h, h, top[b]), (pos[a], pos[a], pos[b], pos[b])))
        pos[n + i] = (pos[a] + pos[b]) / 2.0
        top[n + i] = h
    return links


def _merge_colors(Z: np.ndarray, leaf_order: list[int]) -> list[str]:
    """Colour per merge, mimicking SciPy's threshold rule.

    Components below ``0.7 × max_height`` take palette colours by leftmost
    leaf; the rest get the default colour.

    Args:
        Z: Linkage matrix used for plotting.
        leaf_order: Leaves in display order.

    Returns:
        list[str]: Hex colour per merge row.
    """
    n = Z.shape[0] + 1
    threshold = 0.7 * float(Z[:, 2].max()) if Z.shape[0] else 0.0
    parent = {}
    for i in range(n - 1):
        parent[int(Z[i, 0])] = n + i
        parent[int(Z[i, 1])] = n + i

    below = [float(Z[i, 2]) <= threshold for i in range(n - 1)]

    def component_root(node: int) -> int:
        while node in parent and below[parent[node] - n]:
            node = parent[node]
        return node

    rank = {leaf: r for r, leaf in enumerate(leaf_order)}

    def first_leaf_rank(node: int) -> int:
        while node >= n:
            node = int(Z[node - n, 0])
        return rank[node]

    colors = []
    component_color: dict[int, str] = {}
    order = sorted(
        {component_root(n + i) for i in range(n - 1) if below[i]},
        key=first_leaf_rank,
    )
    for k, comp in enumerate(order):
        component_color[comp] = _CLUSTER_PALETTE[k % len(_CLUSTER_PALETTE)]
    for i in range(n - 1):
        if below[i]:
            colors.append(component_color[component_root(n + i)])
        else:
            colors.append(_ABOVE_THRESHOLD_COLOR)
    return colors


def _longest_label_width_in(texts: list[str], font_size: int) -> float:
    """Width in inches of the widest label at *font_size* points.

    Args:
        texts: Leaf label strings.
        font_size: Label font size in points.

    Returns:
        float: Widest label width in inches (0.0 if *texts* is empty).
    """
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    if not texts:
        return 0.0
    prop = FontProperties()
    widest = 0.0
    for text in sorted(texts, key=len, reverse=True)[:10]:
        width = TextPath((0, 0), text, size=font_size, prop=prop).get_extents().width
        widest = max(widest, width)
    return widest / 72.0


class Plugin(BasePlugin):
    description = (
        "Embeds every distinct neuron label and clusters the labels into a "
        "hierarchical tree by semantic similarity (cosine distance). Neurons sharing a "
        "label collapse into one leaf ('label ×N'), so the tree shows the structure of "
        "the label space without duplicate noise. The interactive variant reveals each "
        "subtree's labels on hover and keeps every neuron id text-searchable."
    )
    io_spec = PluginIOSpec(
        required_steps=["neuron_labeling"],
        input_artifacts=[
            ArtifactSpec(
                "neuron_labeling",
                "neuron_labels.json",
                "neuron_labels",
                "json",
            ),
        ],
        output_artifacts=[
            OutputArtifactSpec("linkage_matrix", "linkage_matrix.npy", "npy"),
            OutputArtifactSpec("dendrogram_html", "dendrogram.html", "text"),
            OutputArtifactSpec("dendrogram_pdf", "dendrogram.pdf", "bytes"),
        ],
        output_params=[
            OutputParamSpec("num_neurons", "num_neurons"),
            OutputParamSpec("num_unique_labels", "num_unique_labels"),
        ],
        display=ArtifactDisplaySpec(
            files=[
                ArtifactFileSpec(
                    "dendrogram.html",
                    "Dendrogram (interactive)",
                    "text/html",
                ),
            ],
        ),
        param_ui_hints=[
            StaticDropdownHint("embedding_provider", choices=known_providers()),
            DependentDropdownHint(
                "embedding_model",
                depends_on_param="embedding_provider",
                resolver="embedder_models",
            ),
            StaticDropdownHint(
                "linkage_method",
                choices=[
                    "single",
                    "complete",
                    "average",
                    "weighted",
                    "centroid",
                    "median",
                    "ward",
                ],
            ),
            StaticDropdownHint("axis_mode", choices=["distance", "depth"]),
        ],
        param_groups=[
            ParamGroup("Embedding", ["embedding_provider", "embedding_model"]),
            ParamGroup("Clustering", ["linkage_method"]),
            ParamGroup(
                "Visualization",
                ["axis_mode", "figure_width", "base_height", "label_font_size"],
            ),
        ],
    )

    def load_context(self, context: dict) -> None:
        """Load neuron labels and group neurons by identical label text."""
        super().load_context(context)
        self.neuron_ids = sorted(self.neuron_labels.keys(), key=lambda x: int(x))
        groups: dict[str, list[str]] = {}
        unlabeled = 0
        for nid in self.neuron_ids:
            label = self.neuron_labels[nid]["label"]
            if label is None:
                unlabeled += 1
                continue
            groups.setdefault(str(label), []).append(nid)
        self.unique_labels = list(groups.keys())
        self.label_id_groups = list(groups.values())
        self.num_unlabeled = unlabeled
        logger.info(
            f"Loaded {len(self.neuron_ids)} neuron labels "
            f"({len(self.unique_labels)} unique, {unlabeled} unlabeled)"
        )

    def _leaf_texts(self, max_ids: int | None = _MAX_IDS_IN_LEAF) -> list[str]:
        """Build display labels: ``label ×N [n1, n2, …]``.

        Args:
            max_ids: Cap on ids shown per group before an ``+K more``
                suffix; ``None`` expands every id (used for the static
                PDF/SVG, whose width auto-grows to fit).
        """
        texts = []
        for label, ids in zip(self.unique_labels, self.label_id_groups, strict=True):
            if len(ids) == 1:
                texts.append(f"{label} [n{ids[0]}]")
                continue
            if max_ids is None or len(ids) <= max_ids:
                shown = ", ".join(f"n{i}" for i in ids)
                texts.append(f"{label} ×{len(ids)} [{shown}]")
            else:
                shown = ", ".join(f"n{i}" for i in ids[:max_ids])
                texts.append(f"{label} ×{len(ids)} [{shown}, +{len(ids) - max_ids} more]")
        return texts

    def _hover_texts(self, Z: np.ndarray) -> list[str]:
        """Build per-merge hover text listing the subtree's labels."""
        n = Z.shape[0] + 1
        members: dict[int, list[int]] = {i: [i] for i in range(n)}
        hovers = []
        for i in range(n - 1):
            merged = members[int(Z[i, 0])] + members[int(Z[i, 1])]
            members[n + i] = merged
            neuron_count = sum(len(self.label_id_groups[j]) for j in merged)
            items = sorted(merged, key=lambda j: -len(self.label_id_groups[j]))
            lines = [
                f"{self.unique_labels[j]} ×{len(self.label_id_groups[j])}"
                for j in items[:_MAX_HOVER_LABELS]
            ]
            if len(items) > _MAX_HOVER_LABELS:
                lines.append(f"… +{len(items) - _MAX_HOVER_LABELS} more labels")
            hovers.append(
                f"<b>{len(merged)} labels · {neuron_count} neurons</b><br>" + "<br>".join(lines)
            )
        return hovers

    def _build_html(
        self,
        Z_plot: np.ndarray,
        leaf_order: list[int],
        leaf_texts: list[str],
        axis_title: str,
        label_font_size: int,
    ) -> str:
        """Render the interactive Plotly figure plus the neuron index."""
        links = _tree_geometry(Z_plot, leaf_order)
        colors = _merge_colors(Z_plot, leaf_order)
        hovers = self._hover_texts(Z_plot)

        # One trace per colour (None-separated segments) keeps it fast; the
        # lines carry no hover — the junction markers below do.
        grouped: dict[str, tuple[list, list]] = {}
        for (xs, ys), color in zip(links, colors, strict=True):
            gx, gy = grouped.setdefault(color, ([], []))
            gx.extend([*xs, None])
            gy.extend([*ys, None])

        fig = go.Figure()
        for color, (gx, gy) in grouped.items():
            fig.add_trace(
                go.Scatter(
                    x=gx,
                    y=gy,
                    mode="lines",
                    line={"color": color, "width": 1},
                    hoverinfo="skip",
                )
            )
        # Hover sits on invisible markers along each merge's vertical
        # connector (x=height, y between the two child positions)
        hover_x: list[float] = []
        hover_y: list[float] = []
        hover_t: list[str] = []
        for (xs, ys), hover in zip(links, hovers, strict=True):
            lo, hi = sorted((ys[1], ys[2]))
            steps = max(1, round((hi - lo) / 10.0))
            for k in range(steps + 1):
                hover_x.append(xs[1])
                hover_y.append(lo + (hi - lo) * k / steps)
                hover_t.append(hover)
        fig.add_trace(
            go.Scatter(
                x=hover_x,
                y=hover_y,
                mode="markers",
                marker={"size": 8, "color": "rgba(0,0,0,0)"},
                text=hover_t,
                hoverinfo="text",
            )
        )
        # leaf labels as a text trace
        positions = [5.0 + 10.0 * r for r in range(len(leaf_order))]
        fig.add_trace(
            go.Scatter(
                x=[0.0] * len(positions),
                y=positions,
                mode="text",
                text=[leaf_texts[leaf] for leaf in leaf_order],
                textposition="middle left",
                textfont={"size": label_font_size + 1},
                cliponaxis=False,
                hoverinfo="skip",
            )
        )

        max_label_len = max((len(t) for t in leaf_texts), default=10)
        margin_left = min(600, 30 + int(max_label_len * (label_font_size + 1) * 0.62))
        row_px = 2 * (label_font_size + 1)
        height_px = max(700, row_px * len(leaf_order) + 160)
        fig.update_layout(
            title="Neuron label dendrogram (cosine similarity of embeddings)",
            autosize=True,
            showlegend=False,
            margin={"l": margin_left},
            xaxis={"title": axis_title, "showticklabels": False},
            yaxis={"showticklabels": False, "range": [-10, 10.0 * len(leaf_order) + 10]},
            template="plotly_white",
        )

        index_entries = "".join(
            f"[neuron {nid}] {label}<br>"
            for label, ids in zip(self.unique_labels, self.label_id_groups, strict=True)
            for nid in ids
        )
        index_html = (
            "<div style='font-family:monospace;font-size:11px;margin:2em'>"
            "<h3>Neuron → label index</h3>"
            f"{index_entries}</div>"
        )

        html = fig.to_html(
            full_html=True,
            include_plotlyjs=True,
            default_width="100%",
            default_height=f"{height_px}px",
            config={"responsive": True},
        )
        return html.replace("</body>", index_html + "</body>")

    def run(
        self,
        embedding_provider: Annotated[
            str,
            "Embedding backend used to turn the neuron label texts into "
            "vectors (e.g. 'openai'); must be a provider known to the "
            "embedder registry.",
        ] = "openai",
        embedding_model: Annotated[
            str,
            "Provider-specific embedding model identifier (e.g. "
            "'text-embedding-3-small' for OpenAI). Determines the semantic "
            "space the labels are embedded into.",
        ] = "text-embedding-3-small",
        linkage_method: Annotated[
            str,
            "Hierarchical-clustering linkage method passed to SciPy (e.g. "
            "'average', 'single', 'complete', 'ward'). Controls how cluster "
            "distances are aggregated and therefore the tree's shape.",
        ] = "average",
        axis_mode: Annotated[
            str,
            "What the merge axis encodes: 'distance' places merges at their "
            "cosine merge distance, 'depth' spreads them by depth below the "
            "root so structure stays readable when distances are skewed.",
        ] = "depth",
        figure_width: Annotated[
            int,
            "Width of the rendered dendrogram figure, in inches. Larger "
            "values give the merge axis more room.",
        ] = 20,
        base_height: Annotated[
            int,
            "Minimum height of the static SVG/PDF figure, in inches. Actual "
            "height grows with the label count (~0.25 in per label); this "
            "sets the floor for small label sets.",
        ] = 10,
        label_font_size: Annotated[
            int,
            "Font size of the per-leaf label text. Lower values keep many "
            "labels legible without overlap; the row spacing scales with it.",
        ] = 12,
    ) -> None:
        """Cluster deduplicated labels and render the dendrogram figures.

        Args:
            embedding_provider: Embedder provider name.
            embedding_model: Provider-specific model identifier.
            linkage_method: SciPy linkage method.
            axis_mode: ``"distance"`` or ``"depth"`` merge-axis encoding.
            figure_width: Figure width in inches.
            base_height: Minimum static-figure height in inches.
            label_font_size: Leaf label font size.

        Raises:
            ValueError: If *axis_mode* is not a known mode.
        """
        if axis_mode not in ("distance", "depth"):
            raise ValueError(f"Unknown axis_mode '{axis_mode}'.")

        num_labels = len(self.unique_labels)
        self.notifier.info(
            f"Deduplicated {len(self.neuron_ids):,} neurons into "
            f"{num_labels:,} unique labels ({self.num_unlabeled:,} unlabeled excluded)"
        )
        logger.info(
            f"Embedding {num_labels} unique labels with {embedding_provider}:{embedding_model}"
        )
        embeddings = embed_labels(
            self.unique_labels, embedding_provider, embedding_model, self.notifier
        )

        self.notifier.info(f"Computing pairwise distances for {num_labels:,} labels...")
        distances = pdist(embeddings, metric="cosine")
        self.notifier.info(f"Clustering ({linkage_method} linkage)...")
        self.linkage_matrix = linkage(distances, method=linkage_method)

        Z_plot = (
            _depth_linkage(self.linkage_matrix) if axis_mode == "depth" else self.linkage_matrix
        )
        axis_title = (
            "Depth below root (root at right)"
            if axis_mode == "depth"
            else f"Distance ({linkage_method} linkage)"
        )
        leaf_order = dendrogram(Z_plot, no_plot=True)["leaves"]

        self.notifier.info(f"Building the interactive figure ({num_labels:,} leaves)...")
        # HTML keeps capped ids (the index below stays fully searchable).
        self.dendrogram_html = self._build_html(
            Z_plot,
            leaf_order,
            self._leaf_texts(),
            axis_title,
            label_font_size=label_font_size,
        )

        self.notifier.info(f"Drawing the static dendrogram ({num_labels:,} leaves)...")
        # PDF expands every id (fully Ctrl+F-able); the figure widens by
        # the longest label so tight_layout can't squeeze the tree
        pdf_leaf_texts = self._leaf_texts(max_ids=None)
        label_margin = _longest_label_width_in(pdf_leaf_texts, label_font_size)
        total_width = figure_width + label_margin + 1.0
        dynamic_height = max(base_height, num_labels * 0.25)
        self._fig, ax = plt.subplots(figsize=(total_width, dynamic_height))
        dendrogram(
            Z_plot,
            labels=pdf_leaf_texts,
            ax=ax,
            orientation="right",
            leaf_font_size=label_font_size,
        )
        ax.set_title("Neuron label dendrogram (cosine similarity of embeddings)")
        ax.set_xlabel(axis_title)
        ax.set_ylabel("Neuron label")
        plt.tight_layout()

        self.num_neurons = len(self.neuron_ids)
        self.num_unique_labels = num_labels

        self.notifier.info("Exporting the figure (PDF)...")
        pdf_buf = io.BytesIO()
        self._fig.savefig(pdf_buf, format="pdf")
        self.dendrogram_pdf = pdf_buf.getvalue()
        plt.close(self._fig)
