"""Single variant of the keyword-search embedding-map labeling plugin.

Renders an interactive UMAP scatter plot of neuron labels and adds:

- A keyword marker (star symbol) at the keyword's UMAP-projected
  position so the viewer can see where it sits in label space.
- A separate "highlight" trace containing the top-k labels closest
  to the keyword by cosine similarity over the high-dimensional
  embeddings, drawn with larger gold markers.
- A sidebar listing those top-k labels.  Hovering a sidebar item
  grows the matching marker on the plot.
"""

import json
import tempfile
from typing import Annotated, Any

import mlflow
import plotly.graph_objects as go

from plugins.labeling_evaluation._embedding_cache import embed_labels
from plugins.labeling_evaluation._embedding_map import compute_label_embedding_coords
from plugins.labeling_evaluation._keyword_search import find_top_k_nearest
from plugins.labeling_evaluation._keyword_search_html import (
    Sidebar,
    SidebarItem,
    render_keyword_search_html,
)
from plugins.plugin_interface import (
    ArtifactDisplaySpec,
    ArtifactFileSpec,
    ArtifactSpec,
    BasePlugin,
    OutputArtifactSpec,
    OutputParamSpec,
    PluginIOSpec,
)
from utils.plugin_logger import get_logger

logger = get_logger(__name__)

PLAIN_COLOR = "#9ca3af"
HIGHLIGHT_COLOR = "#f59e0b"
KEYWORD_COLOR = "#10b981"
HIGHLIGHT_DEFAULT_SIZE = 14
KEYWORD_MARKER_SIZE = 18


class Plugin(BasePlugin):
    """Embedding-map plugin with a keyword-search overlay.

    Embeds the keyword alongside the labels so the keyword has UMAP
    coordinates, then ranks the labels by cosine similarity to the
    keyword (high-dimensional embeddings) and highlights the top-k
    on the plot + in a sidebar.
    """

    name = "Embedding Map with Keyword Search"

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
            OutputArtifactSpec("umap_coords", "umap_coords.npy", "npy"),
            OutputArtifactSpec(
                "top_k_matches",
                "top_k_matches.json",
                "json",
            ),
        ],
        output_params=[
            OutputParamSpec("embedding_provider", "embedding_provider_param"),
            OutputParamSpec("embedding_model", "embedding_model_param"),
            OutputParamSpec("umap_n_neighbors", "umap_n_neighbors_param"),
            OutputParamSpec("umap_min_dist", "umap_min_dist_param"),
            OutputParamSpec("umap_metric", "umap_metric_param"),
            OutputParamSpec("umap_random_state", "umap_random_state_param"),
            OutputParamSpec("keyword", "keyword_param"),
            OutputParamSpec("k", "k_param"),
            OutputParamSpec("num_top_k_matches", "num_top_k_matches_param"),
            OutputParamSpec("num_neurons", "num_neurons"),
        ],
        display=ArtifactDisplaySpec(
            files=[
                ArtifactFileSpec(
                    "keyword_search_map.html",
                    "Embedding Map (keyword search)",
                    "text/html",
                ),
            ],
        ),
    )

    def load_context(self, context: dict[str, Any]) -> None:
        """Load neuron labels and derive sorted ids and texts.

        Args:
            context: Pipeline context mapping step names to per-step
                run-id dicts.
        """
        super().load_context(context)
        self.neuron_ids = sorted(self.neuron_labels.keys(), key=lambda x: int(x))
        self.label_texts = [str(self.neuron_labels[nid]) for nid in self.neuron_ids]
        logger.info(f"Loaded {len(self.neuron_ids)} neuron labels")

    def run(
        self,
        keyword: Annotated[
            str,
            "Search term embedded with the same model as the labels; "
            "labels are ranked by cosine similarity to it on the raw "
            "high-dimensional embeddings. Must be non-empty.",
        ],
        k: Annotated[
            int,
            "Maximum number of closest labels to highlight, ranked by "
            "cosine similarity to the keyword. Clamped to the number of "
            "available labels.",
        ] = 10,
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
        umap_n_neighbors: Annotated[
            int,
            "UMAP n_neighbors: how many neighbours define local structure. "
            "Low values emphasise fine local clusters; high values preserve "
            "more global layout.",
        ] = 15,
        umap_min_dist: Annotated[
            float,
            "UMAP min_dist: minimum spacing between points in the 2-D "
            "projection. Low values pack similar labels into tight clumps; "
            "high values spread them out more evenly.",
        ] = 0.1,
        umap_metric: Annotated[
            str,
            "Distance metric UMAP uses to compare the high-dimensional "
            "label embeddings (e.g. 'cosine', 'euclidean').",
        ] = "cosine",
        umap_random_state: Annotated[
            int,
            "Seed for UMAP. Fix it for a reproducible layout across runs; "
            "changing it yields a different but equivalent projection.",
        ] = 42,
        point_size: Annotated[
            int,
            "Diameter of the scatter markers in the projection. Purely "
            "cosmetic; does not affect the embeddings, projection, or any "
            "ranking.",
        ] = 8,
    ) -> None:
        """Compute coords + top-k matches and build the interactive figure.

        Args:
            keyword: Search term (single or multi-word).  Embedded
                with the same provider/model as the labels.
            k: Maximum number of nearest labels to highlight.
            embedding_provider: Embedder provider name resolved by
                the registry.
            embedding_model: Provider-specific model identifier.
            umap_n_neighbors: ``n_neighbors`` knob forwarded to UMAP.
            umap_min_dist: ``min_dist`` knob forwarded to UMAP.
            umap_metric: Distance metric forwarded to UMAP.
            umap_random_state: Seed forwarded to UMAP.
            point_size: Marker size for non-highlighted labels.
        """
        if not keyword.strip():
            raise ValueError("keyword must not be empty")

        # Embed keyword + labels in one batch so they share the same
        # UMAP coordinate space.  embed_labels caches the inner call,
        # so the second call (inside compute_label_embedding_coords)
        # is a free lookup.
        combined_texts = self.label_texts + [keyword]
        embeddings = embed_labels(combined_texts, embedding_provider, embedding_model)
        keyword_embedding = embeddings[-1]
        label_embeddings = embeddings[:-1]

        logger.info(
            "Computing UMAP for %d labels + 1 keyword with %s:%s "
            "(n_neighbors=%d, min_dist=%s, metric=%s)",
            len(self.label_texts),
            embedding_provider,
            embedding_model,
            umap_n_neighbors,
            umap_min_dist,
            umap_metric,
        )

        combined_coords = compute_label_embedding_coords(
            label_texts=combined_texts,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            umap_n_neighbors=umap_n_neighbors,
            umap_min_dist=umap_min_dist,
            umap_metric=umap_metric,
            umap_random_state=umap_random_state,
        )
        self.umap_coords = combined_coords[:-1]
        keyword_coords = combined_coords[-1]

        top_k_indices, top_k_similarities = find_top_k_nearest(
            keyword_embedding, label_embeddings, k
        )
        top_k_indices_set = {int(i) for i in top_k_indices}

        plain_indices = [i for i in range(len(self.label_texts)) if i not in top_k_indices_set]
        plain_coords = self.umap_coords[plain_indices] if plain_indices else None
        plain_hover = [
            f"<b>Neuron {self.neuron_ids[i]}</b><br>{self.label_texts[i]}" for i in plain_indices
        ]

        highlight_coords = self.umap_coords[top_k_indices]
        highlight_hover = [
            (
                f"<b>Neuron {self.neuron_ids[int(idx)]}</b><br>"
                f"{self.label_texts[int(idx)]}<br>"
                f"<b>Similarity:</b> {sim:.4f}"
            )
            for idx, sim in zip(top_k_indices, top_k_similarities, strict=True)
        ]

        top_k_records = [
            {
                "neuron_id": self.neuron_ids[int(idx)],
                "label": self.label_texts[int(idx)],
                "similarity": float(sim),
            }
            for idx, sim in zip(top_k_indices, top_k_similarities, strict=True)
        ]
        self.top_k_matches = json.dumps(top_k_records, indent=2)

        self._fig = go.Figure()
        if plain_coords is not None:
            self._fig.add_trace(
                go.Scatter(
                    x=plain_coords[:, 0],
                    y=plain_coords[:, 1],
                    mode="markers",
                    name="Other labels",
                    marker={
                        "size": point_size,
                        "opacity": 0.6,
                        "color": PLAIN_COLOR,
                    },
                    text=plain_hover,
                    hovertemplate="%{text}<extra></extra>",
                )
            )
        # Highlight trace must always be present (and its trace index
        # known to the JS) so sidebar hovers can target it.  When
        # there is no plain trace, the highlight trace has index 0.
        highlight_trace_index = 1 if plain_coords is not None else 0
        self._fig.add_trace(
            go.Scatter(
                x=highlight_coords[:, 0],
                y=highlight_coords[:, 1],
                mode="markers",
                name=f"Top {len(top_k_records)} matches",
                marker={
                    "size": HIGHLIGHT_DEFAULT_SIZE,
                    "opacity": 0.95,
                    "color": HIGHLIGHT_COLOR,
                    "line": {"width": 1, "color": "#92400e"},
                },
                text=highlight_hover,
                hovertemplate="%{text}<extra></extra>",
            )
        )
        self._fig.add_trace(
            go.Scatter(
                x=[keyword_coords[0]],
                y=[keyword_coords[1]],
                mode="markers",
                name=f"Keyword: {keyword}",
                marker={
                    "size": KEYWORD_MARKER_SIZE,
                    "symbol": "star",
                    "color": KEYWORD_COLOR,
                    "line": {"width": 1, "color": "#065f46"},
                },
                text=[f"<b>Keyword</b><br>{keyword}"],
                hovertemplate="%{text}<extra></extra>",
            )
        )
        self._fig.update_layout(
            title={
                "text": (f'Embedding map — top-{len(top_k_records)} closest to "{keyword}"'),
                "font": {"size": 18},
            },
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            hovermode="closest",
            template="plotly_white",
            margin={"t": 60, "r": 20, "b": 40, "l": 50},
        )

        sidebar_items = [
            SidebarItem(
                label=f"[neuron {rec['neuron_id']}] {rec['label']}",
                similarity=rec["similarity"],
                trace_index=highlight_trace_index,
                point_index=i,
            )
            for i, rec in enumerate(top_k_records)
        ]
        self._sidebars = [Sidebar(title=f"Top {len(top_k_records)} matches", items=sidebar_items)]
        self._keyword = keyword

        self.embedding_provider_param = embedding_provider
        self.embedding_model_param = embedding_model
        self.umap_n_neighbors_param = umap_n_neighbors
        self.umap_min_dist_param = umap_min_dist
        self.umap_metric_param = umap_metric
        self.umap_random_state_param = umap_random_state
        self.keyword_param = keyword
        self.k_param = k
        self.num_top_k_matches_param = len(top_k_records)
        self.num_neurons = len(self.neuron_ids)

    def update_context(self) -> None:
        """Log standard artifacts via base class, then save the custom HTML page."""
        super().update_context()
        html = render_keyword_search_html(self._fig, self._sidebars, self._keyword)
        with tempfile.TemporaryDirectory() as tmp:
            with open(f"{tmp}/keyword_search_map.html", "w", encoding="utf-8") as f:
                f.write(html)
            mlflow.log_artifacts(tmp)
        logger.info("Keyword-search embedding map saved to mlflow artifacts as HTML")
