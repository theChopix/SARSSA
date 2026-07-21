from typing import Annotated

import plotly.graph_objects as go

from plugins.labeling_evaluation._embedding_map import compute_label_embedding_coords
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


class Plugin(BasePlugin):
    description = (
        "Projects all neuron-label embeddings into an interactive 2-D map with UMAP. "
        "Each point is one neuron's label and proximity reflects semantic similarity — "
        "hover to read labels and visually inspect the clusters and outliers in what the "
        "autoencoder's neurons collectively represent."
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
            OutputArtifactSpec("umap_coords", "umap_coords.npy", "npy"),
            OutputArtifactSpec("embedding_map_html", "embedding_map.html", "text"),
        ],
        output_params=[
            OutputParamSpec("num_neurons", "num_neurons"),
        ],
        display=ArtifactDisplaySpec(
            files=[
                ArtifactFileSpec(
                    "embedding_map.html",
                    "Embedding Map",
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
                "umap_metric",
                choices=["cosine", "euclidean", "manhattan", "correlation", "chebyshev"],
            ),
        ],
        param_groups=[
            ParamGroup("Embedding", ["embedding_provider", "embedding_model"]),
            ParamGroup(
                "UMAP",
                ["umap_n_neighbors", "umap_min_dist", "umap_metric", "umap_random_state"],
            ),
            ParamGroup("Visualization", ["point_size"]),
        ],
    )

    def load_context(self, context: dict) -> None:
        """Load neuron labels and derive sorted IDs and label texts."""
        super().load_context(context)
        self.neuron_ids = sorted(self.neuron_labels.keys(), key=lambda x: int(x))
        self.label_texts = [str(self.neuron_labels[nid]["label"]) for nid in self.neuron_ids]
        logger.info(f"Loaded {len(self.neuron_ids)} neuron labels")

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
        logger.info(
            f"Embedding {len(self.label_texts)} neuron labels with "
            f"{embedding_provider}:{embedding_model}; "
            f"UMAP (n_neighbors={umap_n_neighbors}, min_dist={umap_min_dist}, "
            f"metric={umap_metric})"
        )

        self.umap_coords = compute_label_embedding_coords(
            label_texts=self.label_texts,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            umap_n_neighbors=umap_n_neighbors,
            umap_min_dist=umap_min_dist,
            umap_metric=umap_metric,
            umap_random_state=umap_random_state,
            notifier=self.notifier,
        )

        hover_texts = [
            f"<b>Neuron {nid}</b><br>{self.neuron_labels[nid]['label']}" for nid in self.neuron_ids
        ]

        self._fig = go.Figure(
            go.Scatter(
                x=self.umap_coords[:, 0],
                y=self.umap_coords[:, 1],
                mode="markers",
                marker={"size": point_size, "opacity": 0.8, "colorscale": "Viridis"},
                text=hover_texts,
                hovertemplate="%{text}<extra></extra>",
                customdata=self.neuron_ids,
            )
        )

        self._fig.update_layout(
            title={
                "text": "Neuron label embedding map (UMAP 2-D projection)",
                "font": {"size": 18},
            },
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            hovermode="closest",
            template="plotly_white",
        )

        # output params
        self.num_neurons = len(self.neuron_ids)

        # Render the interactive figure.
        self.embedding_map_html = self._fig.to_html()
