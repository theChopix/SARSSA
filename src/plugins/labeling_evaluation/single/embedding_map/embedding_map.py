import json
import tempfile

import mlflow
import numpy as np
import plotly.graph_objects as go
import umap

from plugins.plugin_interface import BasePlugin
from utils.embedder.openai_embedder import OpenAIEmbeddingLLM
from utils.plugin_logger import get_logger

logger = get_logger(__name__)


class Plugin(BasePlugin):
    def run(
        self,
        context: dict,
        embedding_model: str = "text-embedding-3-small",
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        umap_metric: str = "cosine",
        umap_random_state: int = 42,
        point_size: int = 8,
    ) -> dict:

        
        # 1. Resolve neuron labeling run and its artifact path                
        labeling_run_id = context["last_plugin_run_id"]
        logger.info(f"Loading neuron labeling run: {labeling_run_id}")

        labeling_run = mlflow.get_run(labeling_run_id)
        artifact_uri = labeling_run.info.artifact_uri
        artifact_dir = _uri_to_path(artifact_uri)

        # 2. Load neuron labelss
        labels_path = f"{artifact_dir}/neuron_labeling/neuron_labels.json"
        logger.info(f"Loading neuron labels from {labels_path}")

        with open(labels_path, "r") as f:
            neuron_labels: dict = json.load(f)

        neuron_ids = sorted(neuron_labels.keys(), key=lambda x: int(x))
        label_texts = [str(neuron_labels[nid]) for nid in neuron_ids]

        logger.info(f"Embedding {len(label_texts)} neuron labels with {embedding_model}")

        # 3. Embed labels
        embedder = OpenAIEmbeddingLLM(model=embedding_model)
        embeddings = np.array(
            [embedder.generate_embedding(t) for t in label_texts]
        )

        logger.info(
            f"Embeddings shape: {embeddings.shape}. "
            f"Running UMAP (n_neighbors={umap_n_neighbors}, min_dist={umap_min_dist}, "
            f"metric={umap_metric})"
        )

        # 4. UMAP dimensionality reduction
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric=umap_metric,
            random_state=umap_random_state,
        )
        coords = reducer.fit_transform(embeddings)
        # 5. Build interactive Plotly scatter                                 #
        hover_texts = [
            f"<b>Neuron {nid}</b><br>{neuron_labels[nid]}"
            for nid in neuron_ids
        ]

        fig = go.Figure(
            go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers",
                marker=dict(size=point_size, opacity=0.8, colorscale="Viridis"),
                text=hover_texts,
                hovertemplate="%{text}<extra></extra>",
                customdata=neuron_ids,
            )
        )

        fig.update_layout(
            title=dict(
                text="Neuron label embedding map (UMAP 2-D projection)",
                font=dict(size=18),
            ),
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            hovermode="closest",
            template="plotly_white",
        )

        # 6. Log artifacts to MLflow
        with tempfile.TemporaryDirectory() as tmp:
            html_path = f"{tmp}/embedding_map.html"
            fig.write_html(html_path)
            mlflow.log_artifact(html_path, artifact_path="labeling_evaluation")

            coords_path = f"{tmp}/umap_coords.npy"
            np.save(coords_path, coords)
            mlflow.log_artifact(coords_path, artifact_path="labeling_evaluation")

        mlflow.log_params(
            {
                "embedding_model": embedding_model,
                "umap_n_neighbors": umap_n_neighbors,
                "umap_min_dist": umap_min_dist,
                "umap_metric": umap_metric,
                "umap_random_state": umap_random_state,
                "num_neurons": len(neuron_ids),
            }
        )

        # 7. Update context
        context["labeling_evaluation"] = {
            "status": "completed",
            "artifact_path": "labeling_evaluation",
        }

        logger.info(
            "Embedding map saved to mlflow artifacts as interactive HTML under 'labeling_evaluation/'"
        )

        return context



# helpers                                                                      #

def _uri_to_path(uri: str) -> str:
    if uri.startswith("file://"):
        return uri[len("file://"):]
    if uri.startswith("mlruns"):
        return "./" + uri
    return uri
