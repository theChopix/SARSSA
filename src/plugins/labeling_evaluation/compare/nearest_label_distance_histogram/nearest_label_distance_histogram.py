"""Compare plugin: frequency histogram of nearest-neighbour label distances.

For each label in the current run, finds the cosine distance to the
nearest label in a user-selected past run, then bins those distances
into a Plotly histogram so the *distribution* of label drift is
visible at a glance.

Use this together with the ``nearest_label_distance_bars`` plugin:
the bars show *which* labels drifted, this histogram shows *how
many* drifted by *how much*.
"""

from typing import Annotated, Any

import numpy as np
import plotly.graph_objects as go

from plugins.compare_plugin_interface import BaseComparePlugin
from plugins.labeling_evaluation._nearest_label_distance import (
    compute_nearest_distances,
)
from plugins.plugin_interface import (
    ArtifactDisplaySpec,
    ArtifactFileSpec,
    ArtifactSpec,
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

BAR_COLOR = "#1f77b4"


class Plugin(BaseComparePlugin):
    """Frequency histogram of per-current-label nearest-neighbour distances.

    Loads ``neuron_labels.json`` for both the current pipeline run
    and a user-selected past run, embeds both label sets, computes
    each current label's cosine distance to its nearest past-label
    embedding, and renders a Plotly histogram with a configurable
    bin count.
    """

    name = "Nearest Label Distance — Histogram (compare)"
    description = (
        "Bins each current-run label's distance to its nearest past-run label into a "
        "histogram. Shows the distribution of label drift — how many concepts stayed "
        "well-aligned versus drifted far — the companion overview to the per-label bar "
        "chart."
    )

    past_run_required_steps = ["neuron_labeling"]

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
            OutputArtifactSpec(
                "nearest_distances",
                "nearest_distances.json",
                "json",
            ),
            OutputArtifactSpec(
                "distance_histogram_html",
                "nearest_label_distance_histogram.html",
                "text",
            ),
        ],
        output_params=[
            OutputParamSpec("num_neurons_current", "num_neurons_current_param"),
            OutputParamSpec("num_neurons_past", "num_neurons_past_param"),
            OutputParamSpec("mean_distance", "mean_distance_param"),
            OutputParamSpec("median_distance", "median_distance_param"),
        ],
        display=ArtifactDisplaySpec(
            files=[
                ArtifactFileSpec(
                    "nearest_label_distance_histogram.html",
                    "Nearest Label Distance — Histogram",
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
        ],
        param_groups=[
            ParamGroup("Comparison", ["past_run_id"]),
            ParamGroup("Embedding", ["embedding_provider", "embedding_model"]),
            ParamGroup("Visualization", ["histogram_bins"]),
        ],
    )

    def load_context(self, context: dict[str, Any]) -> None:
        """Load current-run neuron labels and derive sorted ids and texts.

        Args:
            context: Pipeline context mapping step names to per-step
                run-id dicts.
        """
        super().load_context(context)
        self.current_neuron_ids = sorted(self.neuron_labels.keys(), key=lambda x: int(x))
        self.current_label_texts = [
            str(self.neuron_labels[nid]["label"]) for nid in self.current_neuron_ids
        ]
        logger.info(f"Loaded {len(self.current_neuron_ids)} current-run neuron labels")

    def run(
        self,
        # Consumed by BaseComparePlugin (loads the past context); kept in
        # the signature so the registry exposes it as a UI parameter.
        past_run_id: Annotated[  # noqa: ARG002
            str,
            "A previously completed pipeline run to compare against; its "
            "neuron labels are embedded so each current-run label can be "
            "matched to its nearest past-run label.",
        ],
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
        histogram_bins: Annotated[
            int,
            "Number of bins for the distance-frequency histogram. More bins "
            "reveal finer structure in the drift distribution but become "
            "noisier when there are few labels.",
        ] = 30,
    ) -> None:
        """Compute nearest distances and render the histogram figure.

        Args:
            past_run_id: MLflow run id of the past pipeline run to
                compare against.  Loaded automatically by
                :class:`BaseComparePlugin` into ``self.past_context``.
            embedding_provider: Embedder provider name resolved by
                the registry (e.g. ``"openai"``).
            embedding_model: Provider-specific model identifier
                (e.g. ``"text-embedding-3-small"`` for OpenAI).
            histogram_bins: Number of bins for the frequency
                histogram.
        """
        past_neuron_labels: dict[str, str] = self.load_past_artifact(
            "neuron_labeling",
            "neuron_labels.json",
            "json",
        )
        past_neuron_ids = sorted(past_neuron_labels.keys(), key=lambda x: int(x))
        past_label_texts = [str(past_neuron_labels[nid]["label"]) for nid in past_neuron_ids]

        logger.info(
            "Computing nearest-neighbour distances for %d current vs %d past labels with %s:%s",
            len(self.current_label_texts),
            len(past_label_texts),
            embedding_provider,
            embedding_model,
        )

        result = compute_nearest_distances(
            current_label_texts=self.current_label_texts,
            past_label_texts=past_label_texts,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            notifier=self.notifier,
        )

        # Persist the same per-label JSON the bars plugin emits so
        # the two artifacts can be compared / cross-referenced.
        per_label: list[dict[str, Any]] = []
        for i, nid in enumerate(self.current_neuron_ids):
            past_idx = int(result.nearest_past_indices[i])
            per_label.append(
                {
                    "neuron_id": nid,
                    "label": self.current_label_texts[i],
                    "distance": float(result.distances[i]),
                    "nearest_past_neuron_id": past_neuron_ids[past_idx],
                    "nearest_past_label": past_label_texts[past_idx],
                }
            )
        self.nearest_distances = per_label

        self._fig = go.Figure()
        self._fig.add_trace(
            go.Histogram(
                x=result.distances,
                nbinsx=histogram_bins,
                marker={"color": BAR_COLOR},
                hovertemplate=("Distance bin: %{x}<br>Labels: %{y}<extra></extra>"),
            )
        )
        self._fig.update_layout(
            title={
                "text": ("Distribution of nearest-neighbour distances (current vs past)"),
                "font": {"size": 18},
            },
            xaxis_title="Cosine distance to nearest past label",
            yaxis_title="Number of current-run labels",
            template="plotly_white",
            bargap=0.05,
        )
        self.num_neurons_current_param = len(self.current_neuron_ids)
        self.num_neurons_past_param = len(past_neuron_ids)
        self.mean_distance_param = float(result.distances.mean())
        self.median_distance_param = float(np.median(result.distances))

        # Render the interactive figure.
        self.distance_histogram_html = self._fig.to_html()
