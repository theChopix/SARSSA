import io
from typing import Annotated

import matplotlib.pyplot as plt
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


class Plugin(BasePlugin):
    description = (
        "Embeds every neuron label and clusters them into a hierarchical tree by "
        "semantic similarity (cosine distance). The dendrogram reveals how labels group "
        "into themes and exposes near-duplicate neurons that merge close to the leaves — "
        "a quick read on the label space's structure and redundancy."
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
            OutputArtifactSpec("dendrogram_svg", "dendrogram.svg", "text"),
            OutputArtifactSpec("dendrogram_pdf", "dendrogram.pdf", "bytes"),
        ],
        output_params=[
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
        ],
        param_groups=[
            ParamGroup("Embedding", ["embedding_provider", "embedding_model"]),
            ParamGroup("Clustering", ["linkage_method"]),
            ParamGroup(
                "Visualization",
                ["figure_width", "base_height", "label_font_size"],
            ),
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

        embeddings = embed_labels(
            self.label_texts, embedding_provider, embedding_model, self.notifier
        )

        # pairwise cosine distance then hierarchical clustering
        self.notifier.info(f"Computing pairwise distances for {len(embeddings):,} labels...")
        distances = pdist(embeddings, metric="cosine")
        self.notifier.info(f"Clustering ({linkage_method} linkage)...")
        self.linkage_matrix = linkage(distances, method=linkage_method)

        # dynamic height so labels remain readable
        num_labels = len(self.label_texts)
        dynamic_height = max(base_height, num_labels * 0.25)

        logger.info(
            f"Creating dendrogram with {num_labels} labels (figure height={dynamic_height})"
        )
        label_texts = [
            f"[neuron {nid}] {self.neuron_labels[nid]['label']}" for nid in self.neuron_ids
        ]

        self.notifier.info(f"Drawing the dendrogram ({num_labels:,} leaves)...")
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
        self.num_neurons = len(self.neuron_ids)

        # Render the figure to memory and free it.
        self.notifier.info("Exporting the figure (SVG + PDF)...")
        svg_buf = io.StringIO()
        self._fig.savefig(svg_buf, format="svg")
        self.dendrogram_svg = svg_buf.getvalue()
        pdf_buf = io.BytesIO()
        self._fig.savefig(pdf_buf, format="pdf")
        self.dendrogram_pdf = pdf_buf.getvalue()
        plt.close(self._fig)
