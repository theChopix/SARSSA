"""Compare plugin: per-current-label nearest-neighbour distance bar chart.

For each label in the current run, finds the cosine distance to the
nearest label in a user-selected past run and renders an interactive
Plotly bar chart with one bar per current-run neuron, sorted in
descending order of distance — labels that drifted furthest from any
reference label appear on the left.

Hover on a bar reveals the current-run neuron id, its label text,
and the past-run label that was the nearest neighbour.
"""

import json
import tempfile
from typing import Any

import mlflow
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
    OutputArtifactSpec,
    OutputParamSpec,
    PluginIOSpec,
)
from utils.plugin_logger import get_logger

logger = get_logger(__name__)

BAR_COLOR = "#1f77b4"


class Plugin(BaseComparePlugin):
    """Per-label nearest-neighbour distance bar chart.

    Loads ``neuron_labels.json`` for both the current pipeline run
    and a user-selected past run, embeds both label sets, computes
    each current label's cosine distance to its nearest past-label
    embedding, and renders a sorted-descending Plotly bar chart.
    """

    name = "Nearest Label Distance — Bars (compare)"

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
        ],
        output_params=[
            OutputParamSpec("embedding_provider", "embedding_provider_param"),
            OutputParamSpec("embedding_model", "embedding_model_param"),
            OutputParamSpec("past_run_id", "past_run_id_param"),
            OutputParamSpec("num_neurons_current", "num_neurons_current_param"),
            OutputParamSpec("num_neurons_past", "num_neurons_past_param"),
            OutputParamSpec("mean_distance", "mean_distance_param"),
            OutputParamSpec("median_distance", "median_distance_param"),
        ],
        display=ArtifactDisplaySpec(
            files=[
                ArtifactFileSpec(
                    "nearest_label_distance_bars.html",
                    "Nearest Label Distance — Bars",
                    "text/html",
                ),
            ],
        ),
    )

    def load_context(self, context: dict[str, Any]) -> None:
        """Load current-run neuron labels and derive sorted ids and texts.

        Args:
            context: Pipeline context mapping step names to per-step
                run-id dicts.
        """
        super().load_context(context)
        self.current_neuron_ids = sorted(self.neuron_labels.keys(), key=lambda x: int(x))
        self.current_label_texts = [str(self.neuron_labels[nid]) for nid in self.current_neuron_ids]
        logger.info(f"Loaded {len(self.current_neuron_ids)} current-run neuron labels")

    def run(
        self,
        past_run_id: str,
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        """Compute nearest distances and render the bar-chart figure.

        Args:
            past_run_id: MLflow run id of the past pipeline run to
                compare against.  Loaded automatically by
                :class:`BaseComparePlugin` into ``self.past_context``.
            embedding_provider: Embedder provider name resolved by
                the registry (e.g. ``"openai"``).
            embedding_model: Provider-specific model identifier
                (e.g. ``"text-embedding-3-small"`` for OpenAI).
        """
        past_neuron_labels: dict[str, str] = self.load_past_artifact(
            "neuron_labeling",
            "neuron_labels.json",
            "json",
        )
        past_neuron_ids = sorted(past_neuron_labels.keys(), key=lambda x: int(x))
        past_label_texts = [str(past_neuron_labels[nid]) for nid in past_neuron_ids]

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
        )

        # Pair each current neuron with its distance + nearest past label
        # so the JSON artifact is human-readable downstream.
        per_label: list[dict[str, Any]] = []
        for i, nid in enumerate(self.current_neuron_ids):
            past_idx = int(result.nearest_past_indices[i])
            nearest_pid = past_neuron_ids[past_idx]
            per_label.append(
                {
                    "neuron_id": nid,
                    "label": self.current_label_texts[i],
                    "distance": float(result.distances[i]),
                    "nearest_past_neuron_id": nearest_pid,
                    "nearest_past_label": past_label_texts[past_idx],
                }
            )

        # Sort descending so the most-drifted labels are leftmost.
        per_label_sorted = sorted(per_label, key=lambda r: r["distance"], reverse=True)
        self.nearest_distances = json.dumps(per_label_sorted, indent=2)

        bar_x = [r["neuron_id"] for r in per_label_sorted]
        bar_y = [r["distance"] for r in per_label_sorted]
        hover_text = [
            (
                f"<b>Current Neuron {r['neuron_id']}</b><br>"
                f"{r['label']}<br><br>"
                f"<b>Nearest past label</b> (Neuron {r['nearest_past_neuron_id']})<br>"
                f"{r['nearest_past_label']}<br><br>"
                f"Cosine distance: {r['distance']:.4f}"
            )
            for r in per_label_sorted
        ]

        self._fig = go.Figure()
        self._fig.add_trace(
            go.Bar(
                x=bar_x,
                y=bar_y,
                marker={"color": BAR_COLOR},
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
            )
        )
        self._fig.update_layout(
            title={
                "text": ("Per-label nearest-neighbour distance (current vs past, sorted desc)"),
                "font": {"size": 18},
            },
            xaxis={
                "title": "Current-run labels (sorted by distance)",
                "type": "category",
                "showticklabels": False,
            },
            yaxis_title="Cosine distance to nearest past label",
            template="plotly_white",
            bargap=0.05,
        )

        self.embedding_provider_param = embedding_provider
        self.embedding_model_param = embedding_model
        self.past_run_id_param = past_run_id
        self.num_neurons_current_param = len(self.current_neuron_ids)
        self.num_neurons_past_param = len(past_neuron_ids)
        self.mean_distance_param = float(result.distances.mean())
        self.median_distance_param = float(np.median(result.distances))

    def update_context(self) -> None:
        """Log standard artifacts via base class, then save interactive HTML."""
        super().update_context()
        with tempfile.TemporaryDirectory() as tmp:
            self._fig.write_html(f"{tmp}/nearest_label_distance_bars.html")
            mlflow.log_artifacts(tmp)
        logger.info("Nearest-label distance bar chart saved to mlflow artifacts as HTML")
