import tempfile

import mlflow
import numpy as np
import plotly.graph_objects as go
import umap

from plugins.plugin_interface import BasePlugin
from utils.embedder.openai_embedder import OpenAIEmbeddingLLM
from utils.mlflow_manager import MLflowRunLoader
from utils.plugin_logger import get_logger

logger = get_logger(__name__)


class Plugin(BasePlugin):
    def _load_artifacts(self, context):
        """Load neuron labeling artifacts from the neuron_labeling pipeline step."""
        neuron_labeling_run_id = context["neuron_labeling"]["run_id"]
        neuron_labeling_loader = MLflowRunLoader(neuron_labeling_run_id)

        logger.info(f"Loading neuron labeling artifacts from run {neuron_labeling_run_id}")

        self.neuron_labels = neuron_labeling_loader.get_json_artifact("neuron_labels.json")
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
    ) -> dict:
        self._load_artifacts(self._context)

        logger.info(f"Embedding {len(self.label_texts)} neuron labels with {embedding_model}")

        embedder = OpenAIEmbeddingLLM(model=embedding_model)
        embeddings = np.array([embedder.generate_embedding(t) for t in self.label_texts])

        logger.info(
            f"Embeddings shape: {embeddings.shape}. "
            f"Running UMAP (n_neighbors={umap_n_neighbors}, min_dist={umap_min_dist}, "
            f"metric={umap_metric})"
        )

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric=umap_metric,
            random_state=umap_random_state,
        )
        coords = reducer.fit_transform(embeddings)

        hover_texts = [
            f"<b>Neuron {nid}</b><br>{self.neuron_labels[nid]}" for nid in self.neuron_ids
        ]

        fig = go.Figure(
            go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers",
                marker={"size": point_size, "opacity": 0.8, "colorscale": "Viridis"},
                text=hover_texts,
                hovertemplate="%{text}<extra></extra>",
                customdata=self.neuron_ids,
            )
        )

        fig.update_layout(
            title={
                "text": "Neuron label embedding map (UMAP 2-D projection)",
                "font": {"size": 18},
            },
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            hovermode="closest",
            template="plotly_white",
        )

        # 6. Log artifacts to MLflow
        with tempfile.TemporaryDirectory() as tmp:
            html_path = f"{tmp}/embedding_map.html"
            fig.write_html(html_path)
            mlflow.log_artifact(html_path)

            coords_path = f"{tmp}/umap_coords.npy"
            np.save(coords_path, coords)
            mlflow.log_artifact(coords_path)

        mlflow.log_params(
            {
                "embedding_model": embedding_model,
                "umap_n_neighbors": umap_n_neighbors,
                "umap_min_dist": umap_min_dist,
                "umap_metric": umap_metric,
                "umap_random_state": umap_random_state,
                "num_neurons": len(self.neuron_ids),
            }
        )

        logger.info("Embedding map saved to mlflow artifacts as interactive HTML")
