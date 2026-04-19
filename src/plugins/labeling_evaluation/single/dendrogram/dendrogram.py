import tempfile

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

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
        linkage_method: str = "average",
        figure_width: int = 20,
        base_height: int = 10,
        label_font_size: int = 6,
    ):
        self._load_artifacts(self._context)

        logger.info(f"Embedding {len(self.label_texts)} neuron labels with {embedding_model}")

        embedder = OpenAIEmbeddingLLM(model=embedding_model)
        embeddings = np.array([embedder.generate_embedding(t) for t in self.label_texts])

        # pairwise cosine distance then hierarchical clustering
        distances = pdist(embeddings, metric="cosine")
        Z = linkage(distances, method=linkage_method)

        # dynamic height so labels remain readable
        num_labels = len(self.label_texts)
        dynamic_height = max(base_height, num_labels * 0.25)

        logger.info(
            f"Creating dendrogram with {num_labels} labels (figure height={dynamic_height})"
        )
        label_texts = [f"[neuron {nid}] {self.neuron_labels[nid]}" for nid in self.neuron_ids]

        fig, ax = plt.subplots(figsize=(figure_width, dynamic_height))

        dendrogram(
            Z,
            labels=label_texts,
            ax=ax,
            orientation="right",
            leaf_font_size=label_font_size,
        )

        ax.set_title("Neuron label dendrogram (cosine similarity of embeddings)")
        ax.set_xlabel(f"Distance ({linkage_method} linkage)")
        ax.set_ylabel("Neuron label")

        plt.tight_layout()

        with tempfile.TemporaryDirectory() as tmp:
            # SVG (zoomable)
            svg_path = f"{tmp}/dendrogram.svg"
            fig.savefig(svg_path)

            # PDF (searchable text)
            pdf_path = f"{tmp}/dendrogram.pdf"
            fig.savefig(pdf_path)

            mlflow.log_artifact(svg_path)
            mlflow.log_artifact(pdf_path)

            linkage_path = f"{tmp}/linkage_matrix.npy"
            np.save(linkage_path, Z)
            mlflow.log_artifact(linkage_path)

        plt.close(fig)

        mlflow.log_params(
            {
                "embedding_model": embedding_model,
                "linkage_method": linkage_method,
                "num_neurons": len(self.neuron_ids),
            }
        )

        logger.info("Dendrogram saved to mlflow artifacts as SVG and searchable PDF'")
