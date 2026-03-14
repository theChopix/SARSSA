import json
import tempfile
import numpy as np
import mlflow
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

from utils.plugin_logger import get_logger
from plugins.plugin_interface import BasePlugin
from utils.embedder.openai_embedder import OpenAIEmbeddingLLM

logger = get_logger(__name__)


class Plugin(BasePlugin):
    def run(self,
            context: dict,

            embedding_model: str = "text-embedding-3-small",
            linkage_method: str = "average",
            figure_width: int = 20,
            figure_height: int = 10,
            label_rotation: int = 90,
        ):

        # resolve previous run ID (neuron_labeling step)
        base_run_id = context["last_plugin_run_id"]
        nl_run = mlflow.get_run(base_run_id)

        artifact_uri = nl_run.info.artifact_uri
        artifact_path = './' + artifact_uri[artifact_uri.find('mlruns'):]

        # load neuron_labels.json produced by neuron_labeling
        labels_path = f"{artifact_path}/neuron_labeling/neuron_labels.json"
        logger.info(f"Loading neuron labels from {labels_path}")
        with open(labels_path, "r") as f:
            neuron_labels: dict = json.load(f)

        # neuron_labels: {neuron_id (str) -> tag_id}  – values may be ints or strings
        neuron_ids = sorted(neuron_labels.keys(), key=lambda x: int(x))
        label_texts = [str(neuron_labels[nid]) for nid in neuron_ids]

        logger.info(f"Embedding {len(label_texts)} neuron labels with {embedding_model}")
        embedder = OpenAIEmbeddingLLM(model=embedding_model)
        embeddings = np.array([embedder.generate_embedding(t) for t in label_texts])

        # pairwise cosine distance then hierarchical clustering
        distances = pdist(embeddings, metric="cosine")
        Z = linkage(distances, method=linkage_method)

        # draw dendrogram
        fig, ax = plt.subplots(figsize=(figure_width, figure_height))
        dendrogram(
            Z,
            labels=label_texts,
            ax=ax,
            leaf_rotation=label_rotation,
            leaf_font_size=8,
        )
        ax.set_title("Neuron label dendrogram (cosine similarity of embeddings)")
        ax.set_xlabel("Neuron label")
        ax.set_ylabel(f"Distance ({linkage_method} linkage)")
        plt.tight_layout()

        with tempfile.TemporaryDirectory() as tmp:
            fig_path = f"{tmp}/dendrogram.png"
            fig.savefig(fig_path, dpi=150)
            mlflow.log_artifact(fig_path, artifact_path="labeling_evaluation")

            linkage_path = f"{tmp}/linkage_matrix.npy"
            np.save(linkage_path, Z)
            mlflow.log_artifact(linkage_path, artifact_path="labeling_evaluation")

        plt.close(fig)

        mlflow.log_params({
            "embedding_model": embedding_model,
            "linkage_method": linkage_method,
            "num_neurons": len(neuron_ids),
        })

        context["labeling_evaluation"] = {
            "status": "completed",
            "artifact_path": "labeling_evaluation",
        }

        logger.info("Dendrogram saved to mlflow artifacts under 'labeling_evaluation/'")
        return context
