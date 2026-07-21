"""Compare variant of the embedding-map labeling-evaluation plugin.

Renders a single UMAP scatter plot containing two color-distinguished
sets of points — the current run's neuron labels and a user-selected
past run's labels.  Both sides go through the same embedding + UMAP
pass (concatenated input) so their 2-D coordinates live in a shared
space, making the visual comparison meaningful.
"""

from typing import Annotated, Any

import plotly.graph_objects as go

from plugins.compare_plugin_interface import BaseComparePlugin
from plugins.labeling_evaluation._embedding_map import compute_label_embedding_coords
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
    description = (
        "Projects the neuron labels of the current run and a chosen past run into one "
        "shared 2-D UMAP space, colored by run. Overlapping regions mean the two runs "
        "captured similar concepts; separate clusters reveal where their labeled "
        "semantic coverage diverged."
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
                "current_umap_coords",
                "current_umap_coords.npy",
                "npy",
            ),
            OutputArtifactSpec(
                "past_umap_coords",
                "past_umap_coords.npy",
                "npy",
            ),
            OutputArtifactSpec("embedding_map_html", "embedding_map.html", "text"),
        ],
        output_params=[
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
        param_ui_hints=[
            StaticDropdownHint("embedding_provider", choices=known_providers()),
            DependentDropdownHint(
                "embedding_model",
                depends_on_param="embedding_provider",
                resolver="embedder_models",
            ),
            StaticDropdownHint(
                "umap_metric",
                choices=["cosine", "euclidean", "manhattan", "correlation", "chebyshev"],
            ),
        ],
        param_groups=[
            ParamGroup("Comparison", ["past_run_id"]),
            ParamGroup("Embedding", ["embedding_provider", "embedding_model"]),
            ParamGroup(
                "UMAP",
                ["umap_n_neighbors", "umap_min_dist", "umap_metric", "umap_random_state"],
            ),
            ParamGroup("Visualization", ["point_size"]),
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
            "neuron labels are embedded and projected into the same space "
            "as the current run's.",
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
        umap_n_neighbors: Annotated[
            int,
            "UMAP n_neighbors: how many neighbours define local structure. "
            "Low values emphasise fine local clusters; high values preserve "
            "more global layout.",
        ] = 15,
        umap_min_dist: Annotated[
            float,
            "UMAP min_dist: minimum spacing between points in the 2-D "
            "projection. Low values pack similar labels into tight clumps; "
            "high values spread them out more evenly.",
        ] = 0.1,
        umap_metric: Annotated[
            str,
            "Distance metric UMAP uses to compare the high-dimensional "
            "label embeddings (e.g. 'cosine', 'euclidean').",
        ] = "cosine",
        umap_random_state: Annotated[
            int,
            "Seed for UMAP. Fix it for a reproducible layout across runs; "
            "changing it yields a different but equivalent projection.",
        ] = 42,
        point_size: Annotated[
            int,
            "Diameter of the scatter markers in the projection. Purely "
            "cosmetic; does not affect the embeddings, projection, or any "
            "ranking.",
        ] = 8,
    ) -> None:
        """Embed and UMAP-project current + past labels in one shared space.

        Args:
            past_run_id: MLflow run id of the past pipeline run to
                compare against.  Loaded automatically by
                :class:`BaseComparePlugin` into ``self.past_context``.
            embedding_provider: Embedder provider name resolved by
                the registry (e.g. ``"openai"``).
            embedding_model: Provider-specific model identifier
                (e.g. ``"text-embedding-3-small"`` for OpenAI).
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
        past_label_texts = [str(past_neuron_labels[nid]["label"]) for nid in past_neuron_ids]

        logger.info(
            "Embedding %d current + %d past labels with %s:%s; "
            "UMAP (n_neighbors=%d, min_dist=%s, metric=%s)",
            len(self.current_label_texts),
            len(past_label_texts),
            embedding_provider,
            embedding_model,
            umap_n_neighbors,
            umap_min_dist,
            umap_metric,
        )

        combined_texts = self.current_label_texts + past_label_texts
        combined_coords = compute_label_embedding_coords(
            label_texts=combined_texts,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            umap_n_neighbors=umap_n_neighbors,
            umap_min_dist=umap_min_dist,
            umap_metric=umap_metric,
            umap_random_state=umap_random_state,
            notifier=self.notifier,
        )
        split = len(self.current_label_texts)
        self.current_umap_coords = combined_coords[:split]
        self.past_umap_coords = combined_coords[split:]

        current_hover = [
            f"<b>Current Run · Neuron {nid}</b><br>{self.neuron_labels[nid]['label']}"
            for nid in self.current_neuron_ids
        ]
        past_hover = [
            f"<b>Past Run · Neuron {nid}</b><br>{past_neuron_labels[nid]['label']}"
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
        self.num_neurons_current_param = len(self.current_neuron_ids)
        self.num_neurons_past_param = len(past_neuron_ids)

        # Render the interactive figure.
        self.embedding_map_html = self._fig.to_html()
