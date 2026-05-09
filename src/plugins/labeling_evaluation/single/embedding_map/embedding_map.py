import tempfile

import mlflow
import plotly.graph_objects as go

from plugins.labeling_evaluation._embedding_map import compute_label_embedding_coords
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


class Plugin(BasePlugin):
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
        ],
        output_params=[
            OutputParamSpec("embedding_model", "embedding_model_param"),
            OutputParamSpec("umap_n_neighbors", "umap_n_neighbors_param"),
            OutputParamSpec("umap_min_dist", "umap_min_dist_param"),
            OutputParamSpec("umap_metric", "umap_metric_param"),
            OutputParamSpec("umap_random_state", "umap_random_state_param"),
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
    )

    def load_context(self, context: dict) -> None:
        """Load neuron labels and derive sorted IDs and label texts."""
        super().load_context(context)
        self.neuron_ids = sorted(self.neuron_labels.keys(), key=lambda x: int(x))
        self.label_texts = [str(self.neuron_labels[nid]) for nid in self.neuron_ids]
        logger.info(f"Loaded {len(self.neuron_ids)} neuron labels")

    def run(
        self,
        embedding_model: str = "text-embedding-3-small",
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        umap_metric: str = "cosine",
        umap_random_state: int = 42,
        point_size: int = 8,
    ) -> None:
        logger.info(
            f"Embedding {len(self.label_texts)} neuron labels with {embedding_model}; "
            f"UMAP (n_neighbors={umap_n_neighbors}, min_dist={umap_min_dist}, "
            f"metric={umap_metric})"
        )

        self.umap_coords = compute_label_embedding_coords(
            label_texts=self.label_texts,
            embedding_model=embedding_model,
            umap_n_neighbors=umap_n_neighbors,
            umap_min_dist=umap_min_dist,
            umap_metric=umap_metric,
            umap_random_state=umap_random_state,
        )

        hover_texts = [
            f"<b>Neuron {nid}</b><br>{self.neuron_labels[nid]}" for nid in self.neuron_ids
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
        self.embedding_model_param = embedding_model
        self.umap_n_neighbors_param = umap_n_neighbors
        self.umap_min_dist_param = umap_min_dist
        self.umap_metric_param = umap_metric
        self.umap_random_state_param = umap_random_state
        self.num_neurons = len(self.neuron_ids)

    def update_context(self) -> None:
        """Log standard artifacts via base class, then save interactive HTML."""
        super().update_context()
        with tempfile.TemporaryDirectory() as tmp:
            self._fig.write_html(f"{tmp}/embedding_map.html")
            mlflow.log_artifacts(tmp)
        logger.info("Embedding map saved to mlflow artifacts as interactive HTML")
