"""Compare variant of the embedding-map labeling-evaluation plugin.

Renders a single UMAP scatter plot containing two color-distinguished
sets of points — the current run's neuron labels and a user-selected
past run's labels.  Both sides go through the same embedding + UMAP
pass (concatenated input) so their 2-D coordinates live in a shared
space, making the visual comparison meaningful.
"""

import tempfile
from typing import Any

import mlflow
import plotly.graph_objects as go

from plugins.compare_plugin_interface import BaseComparePlugin
from plugins.labeling_evaluation._embedding_map import compute_label_embedding_coords
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

CURRENT_RUN_COLOR = "#1f77b4"
PAST_RUN_COLOR = "#d62728"


class Plugin(BaseComparePlugin):
    """Embedding-map compare plugin.

    Loads ``neuron_labels.json`` for both the current pipeline run and
    a user-selected past run, concatenates the two label texts,
    embeds and UMAP-projects them in a single pass, and renders a
    Plotly figure with two scatter traces (one per run) coloured to
    distinguish their origin.
    """

    name = "Embedding Map (compare)"

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
                "current_umap_coords",
                "current_umap_coords.npy",
                "npy",
            ),
            OutputArtifactSpec(
                "past_umap_coords",
                "past_umap_coords.npy",
                "npy",
            ),
        ],
        output_params=[
            OutputParamSpec("embedding_model", "embedding_model_param"),
            OutputParamSpec("umap_n_neighbors", "umap_n_neighbors_param"),
            OutputParamSpec("umap_min_dist", "umap_min_dist_param"),
            OutputParamSpec("umap_metric", "umap_metric_param"),
            OutputParamSpec("umap_random_state", "umap_random_state_param"),
            OutputParamSpec("past_run_id", "past_run_id_param"),
            OutputParamSpec("num_neurons_current", "num_neurons_current_param"),
            OutputParamSpec("num_neurons_past", "num_neurons_past_param"),
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
        embedding_model: str = "text-embedding-3-small",
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        umap_metric: str = "cosine",
        umap_random_state: int = 42,
        point_size: int = 8,
    ) -> None:
        """Embed and UMAP-project current + past labels in one shared space.

        Args:
            past_run_id: MLflow run id of the past pipeline run to
                compare against.  Loaded automatically by
                :class:`BaseComparePlugin` into ``self.past_context``.
            embedding_model: OpenAI embedding model identifier
                (e.g. ``"text-embedding-3-small"``).
            umap_n_neighbors: ``n_neighbors`` knob forwarded to UMAP.
            umap_min_dist: ``min_dist`` knob forwarded to UMAP.
            umap_metric: Distance metric forwarded to UMAP.
            umap_random_state: Seed forwarded to UMAP for reproducibility.
            point_size: Marker size for both traces.
        """
        past_neuron_labels: dict[str, str] = self.load_past_artifact(
            "neuron_labeling",
            "neuron_labels.json",
            "json",
        )
        past_neuron_ids = sorted(past_neuron_labels.keys(), key=lambda x: int(x))
        past_label_texts = [str(past_neuron_labels[nid]) for nid in past_neuron_ids]

        logger.info(
            "Embedding %d current + %d past labels with %s; "
            "UMAP (n_neighbors=%d, min_dist=%s, metric=%s)",
            len(self.current_label_texts),
            len(past_label_texts),
            embedding_model,
            umap_n_neighbors,
            umap_min_dist,
            umap_metric,
        )

        combined_texts = self.current_label_texts + past_label_texts
        combined_coords = compute_label_embedding_coords(
            label_texts=combined_texts,
            embedding_model=embedding_model,
            umap_n_neighbors=umap_n_neighbors,
            umap_min_dist=umap_min_dist,
            umap_metric=umap_metric,
            umap_random_state=umap_random_state,
        )
        split = len(self.current_label_texts)
        self.current_umap_coords = combined_coords[:split]
        self.past_umap_coords = combined_coords[split:]

        current_hover = [
            f"<b>Current Run · Neuron {nid}</b><br>{self.neuron_labels[nid]}"
            for nid in self.current_neuron_ids
        ]
        past_hover = [
            f"<b>Past Run · Neuron {nid}</b><br>{past_neuron_labels[nid]}"
            for nid in past_neuron_ids
        ]

        self._fig = go.Figure()
        self._fig.add_trace(
            go.Scatter(
                x=self.current_umap_coords[:, 0],
                y=self.current_umap_coords[:, 1],
                mode="markers",
                name="Current Run",
                marker={"size": point_size, "opacity": 0.8, "color": CURRENT_RUN_COLOR},
                text=current_hover,
                hovertemplate="%{text}<extra></extra>",
                customdata=self.current_neuron_ids,
            )
        )
        self._fig.add_trace(
            go.Scatter(
                x=self.past_umap_coords[:, 0],
                y=self.past_umap_coords[:, 1],
                mode="markers",
                name="Past Run",
                marker={"size": point_size, "opacity": 0.8, "color": PAST_RUN_COLOR},
                text=past_hover,
                hovertemplate="%{text}<extra></extra>",
                customdata=past_neuron_ids,
            )
        )

        self._fig.update_layout(
            title={
                "text": "Neuron label embedding map (UMAP 2-D projection) — current vs past",
                "font": {"size": 18},
            },
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            hovermode="closest",
            template="plotly_white",
            legend={"title": {"text": "Run"}},
        )

        self.embedding_model_param = embedding_model
        self.umap_n_neighbors_param = umap_n_neighbors
        self.umap_min_dist_param = umap_min_dist
        self.umap_metric_param = umap_metric
        self.umap_random_state_param = umap_random_state
        self.past_run_id_param = past_run_id
        self.num_neurons_current_param = len(self.current_neuron_ids)
        self.num_neurons_past_param = len(past_neuron_ids)

    def update_context(self) -> None:
        """Log standard artifacts via base class, then save interactive HTML."""
        super().update_context()
        with tempfile.TemporaryDirectory() as tmp:
            self._fig.write_html(f"{tmp}/embedding_map.html")
            mlflow.log_artifacts(tmp)
        logger.info("Compare embedding map saved to mlflow artifacts as interactive HTML")
