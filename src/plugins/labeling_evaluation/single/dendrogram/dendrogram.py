import tempfile
from typing import Annotated

import matplotlib.pyplot as plt
import mlflow
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

from plugins.labeling_evaluation._embedding_cache import embed_labels
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
            OutputArtifactSpec("linkage_matrix", "linkage_matrix.npy", "npy"),
        ],
        output_params=[
            OutputParamSpec("embedding_provider", "embedding_provider_param"),
            OutputParamSpec("embedding_model", "embedding_model_param"),
            OutputParamSpec("linkage_method", "linkage_method_param"),
            OutputParamSpec("num_neurons", "num_neurons"),
        ],
        display=ArtifactDisplaySpec(
            files=[
                ArtifactFileSpec(
                    "dendrogram.pdf",
                    "Dendrogram",
                    "application/pdf",
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
        figure_width: Annotated[
            int,
            "Width of the rendered dendrogram figure, in inches. Larger "
            "values give the horizontal distance axis more room.",
        ] = 20,
        base_height: Annotated[
            int,
            "Minimum height of the dendrogram figure, in inches. Actual "
            "height grows with the label count (~0.25 in per label); this "
            "sets the floor for small label sets.",
        ] = 10,
        label_font_size: Annotated[
            int,
            "Font size of the per-leaf neuron-label text. Lower values keep "
            "many labels legible without overlap.",
        ] = 6,
    ) -> None:
        logger.info(
            f"Embedding {len(self.label_texts)} neuron labels with "
            f"{embedding_provider}:{embedding_model}"
        )

        embeddings = embed_labels(self.label_texts, embedding_provider, embedding_model)

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
        self.embedding_provider_param = embedding_provider
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
