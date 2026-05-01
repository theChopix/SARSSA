import tempfile

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

from plugins.plugin_interface import (
    ArtifactDisplaySpec,
    ArtifactFileSpec,
    ArtifactSpec,
    BasePlugin,
    OutputArtifactSpec,
    OutputParamSpec,
    PluginIOSpec,
)
from utils.embedder.openai_embedder import OpenAIEmbeddingLLM
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
            OutputArtifactSpec("linkage_matrix", "linkage_matrix.npy", "npy"),
        ],
        output_params=[
            OutputParamSpec("embedding_model", "embedding_model_param"),
            OutputParamSpec("linkage_method", "linkage_method_param"),
            OutputParamSpec("num_neurons", "num_neurons"),
        ],
        display=ArtifactDisplaySpec(
            files=[
                ArtifactFileSpec(
                    "dendrogram.svg",
                    "Dendrogram",
                    "image/svg+xml",
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
        linkage_method: str = "average",
        figure_width: int = 20,
        base_height: int = 10,
        label_font_size: int = 6,
    ) -> None:
        logger.info(f"Embedding {len(self.label_texts)} neuron labels with {embedding_model}")

        embedder = OpenAIEmbeddingLLM(model=embedding_model)
        embeddings = np.array([embedder.generate_embedding(t) for t in self.label_texts])

        # pairwise cosine distance then hierarchical clustering
        distances = pdist(embeddings, metric="cosine")
        self.linkage_matrix = linkage(distances, method=linkage_method)

        # dynamic height so labels remain readable
        num_labels = len(self.label_texts)
        dynamic_height = max(base_height, num_labels * 0.25)

        logger.info(
            f"Creating dendrogram with {num_labels} labels (figure height={dynamic_height})"
        )
        label_texts = [f"[neuron {nid}] {self.neuron_labels[nid]}" for nid in self.neuron_ids]

        self._fig, ax = plt.subplots(figsize=(figure_width, dynamic_height))

        dendrogram(
            self.linkage_matrix,
            labels=label_texts,
            ax=ax,
            orientation="right",
            leaf_font_size=label_font_size,
        )

        ax.set_title("Neuron label dendrogram (cosine similarity of embeddings)")
        ax.set_xlabel(f"Distance ({linkage_method} linkage)")
        ax.set_ylabel("Neuron label")

        plt.tight_layout()

        # output params
        self.embedding_model_param = embedding_model
        self.linkage_method_param = linkage_method
        self.num_neurons = len(self.neuron_ids)

    def update_context(self) -> None:
        """Log standard artifacts via base class, then save figure as SVG/PDF."""
        super().update_context()
        with tempfile.TemporaryDirectory() as tmp:
            self._fig.savefig(f"{tmp}/dendrogram.svg")
            self._fig.savefig(f"{tmp}/dendrogram.pdf")
            mlflow.log_artifacts(tmp)
        plt.close(self._fig)
        logger.info("Dendrogram saved to mlflow artifacts as SVG and searchable PDF")
