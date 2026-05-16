"""Compare variant of the keyword-search embedding-map plugin.

Shares the layout pass with ``compare/embedding_map`` (one combined
UMAP fit so the current/past sides live in the same coordinate
space), then overlays:

- A keyword marker (star) at the keyword's UMAP position.
- One or two highlight traces of the labels closest to the keyword
  by cosine similarity, depending on ``search_scope``:
  - ``"separate"``: top-k computed per side, two highlight traces
    (one per origin), two sidebar sections.
  - ``"combined"``: top-k computed across both label sets pooled
    together, one sidebar with current/past badges per item.  Each
    matched item is still drawn in its origin's colour, so the
    visual distinction between current and past is preserved.
"""

import json
import tempfile
from dataclasses import dataclass
from typing import Annotated, Any, Literal

import mlflow
import numpy as np
import plotly.graph_objects as go

from plugins.compare_plugin_interface import BaseComparePlugin
from plugins.labeling_evaluation._embedding_cache import embed_labels
from plugins.labeling_evaluation._embedding_map import compute_label_embedding_coords
from plugins.labeling_evaluation._keyword_search import find_top_k_nearest
from plugins.labeling_evaluation._keyword_search_html import (
    Sidebar,
    SidebarItem,
    render_keyword_search_html,
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

CURRENT_PLAIN_COLOR = "#bfdbfe"
CURRENT_HIGHLIGHT_COLOR = "#1d4ed8"
PAST_PLAIN_COLOR = "#fecaca"
PAST_HIGHLIGHT_COLOR = "#b91c1c"
KEYWORD_COLOR = "#10b981"
HIGHLIGHT_DEFAULT_SIZE = 14
KEYWORD_MARKER_SIZE = 18

SearchScope = Literal["separate", "combined"]


class Plugin(BaseComparePlugin):
    """Compare embedding-map plugin with keyword-search highlights.

    Inherits past-run plumbing from :class:`BaseComparePlugin`.
    """

    name = "Embedding Map with Keyword Search (compare)"

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
            OutputArtifactSpec(
                "top_k_matches",
                "top_k_matches.json",
                "json",
            ),
        ],
        output_params=[
            OutputParamSpec("embedding_provider", "embedding_provider_param"),
            OutputParamSpec("embedding_model", "embedding_model_param"),
            OutputParamSpec("umap_n_neighbors", "umap_n_neighbors_param"),
            OutputParamSpec("umap_min_dist", "umap_min_dist_param"),
            OutputParamSpec("umap_metric", "umap_metric_param"),
            OutputParamSpec("umap_random_state", "umap_random_state_param"),
            OutputParamSpec("keyword", "keyword_param"),
            OutputParamSpec("k", "k_param"),
            OutputParamSpec("search_scope", "search_scope_param"),
            OutputParamSpec("past_run_id", "past_run_id_param"),
            OutputParamSpec("num_neurons_current", "num_neurons_current_param"),
            OutputParamSpec("num_neurons_past", "num_neurons_past_param"),
        ],
        display=ArtifactDisplaySpec(
            files=[
                ArtifactFileSpec(
                    "keyword_search_map.html",
                    "Embedding Map (keyword search) — current vs past",
                    "text/html",
                ),
            ],
        ),
    )

    def load_context(self, context: dict[str, Any]) -> None:
        """Load current-run neuron labels and derive sorted ids and texts."""
        super().load_context(context)
        self.current_neuron_ids = sorted(self.neuron_labels.keys(), key=lambda x: int(x))
        self.current_label_texts = [str(self.neuron_labels[nid]) for nid in self.current_neuron_ids]
        logger.info(f"Loaded {len(self.current_neuron_ids)} current-run neuron labels")

    def run(
        self,
        past_run_id: Annotated[
            str,
            "A previously completed pipeline run to compare against; its "
            "neuron labels are embedded and projected into the same space "
            "as the current run's.",
        ],
        keyword: Annotated[
            str,
            "Search term embedded with the same model as the labels; "
            "labels are ranked by cosine similarity to it on the raw "
            "high-dimensional embeddings. Must be non-empty.",
        ],
        k: Annotated[
            int,
            "Maximum number of closest labels to highlight, ranked by "
            "cosine similarity to the keyword. Clamped to the number of "
            "available labels.",
        ] = 10,
        search_scope: Annotated[
            SearchScope,
            "How top-k is computed across the two runs: 'separate' ranks "
            "the k closest labels within each run independently; 'combined' "
            "pools both runs' labels and takes the k closest overall.",
        ] = "separate",
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
        """Embed both sides + keyword and build the compare keyword-search figure.

        Args:
            past_run_id: MLflow run id of the past pipeline run to
                compare against.
            keyword: Search term (single or multi-word).
            k: Maximum number of nearest labels to highlight (per
                side in ``"separate"``, in total in ``"combined"``).
            search_scope: ``"separate"`` for per-side top-k +
                two sidebars, ``"combined"`` for one pooled
                ranking + one badged sidebar.
            embedding_provider: Embedder provider name.
            embedding_model: Embedding model identifier.
            umap_n_neighbors: ``n_neighbors`` knob forwarded to UMAP.
            umap_min_dist: ``min_dist`` knob forwarded to UMAP.
            umap_metric: Distance metric forwarded to UMAP.
            umap_random_state: Seed forwarded to UMAP.
            point_size: Marker size for non-highlighted labels.
        """
        if not keyword.strip():
            raise ValueError("keyword must not be empty")
        if search_scope not in ("separate", "combined"):
            raise ValueError(f"search_scope must be 'separate' or 'combined', got {search_scope!r}")

        past_neuron_labels: dict[str, str] = self.load_past_artifact(
            "neuron_labeling",
            "neuron_labels.json",
            "json",
        )
        past_neuron_ids = sorted(past_neuron_labels.keys(), key=lambda x: int(x))
        past_label_texts = [str(past_neuron_labels[nid]) for nid in past_neuron_ids]

        n_current = len(self.current_label_texts)
        n_past = len(past_label_texts)
        combined_texts = self.current_label_texts + past_label_texts + [keyword]

        embeddings = embed_labels(combined_texts, embedding_provider, embedding_model)
        current_embeddings = embeddings[:n_current]
        past_embeddings = embeddings[n_current : n_current + n_past]
        keyword_embedding = embeddings[-1]

        logger.info(
            "Computing UMAP for %d current + %d past + 1 keyword with %s:%s; scope=%s, k=%d",
            n_current,
            n_past,
            embedding_provider,
            embedding_model,
            search_scope,
            k,
        )

        combined_coords = compute_label_embedding_coords(
            label_texts=combined_texts,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            umap_n_neighbors=umap_n_neighbors,
            umap_min_dist=umap_min_dist,
            umap_metric=umap_metric,
            umap_random_state=umap_random_state,
        )
        self.current_umap_coords = combined_coords[:n_current]
        self.past_umap_coords = combined_coords[n_current : n_current + n_past]
        keyword_coords = combined_coords[-1]

        if search_scope == "separate":
            current_top_k = _build_per_side_top_k(
                neuron_ids=self.current_neuron_ids,
                label_texts=self.current_label_texts,
                embeddings=current_embeddings,
                keyword_embedding=keyword_embedding,
                k=k,
            )
            past_top_k = _build_per_side_top_k(
                neuron_ids=past_neuron_ids,
                label_texts=past_label_texts,
                embeddings=past_embeddings,
                keyword_embedding=keyword_embedding,
                k=k,
            )
        else:  # "combined"
            current_top_k, past_top_k = _build_pooled_top_k(
                current_neuron_ids=self.current_neuron_ids,
                current_label_texts=self.current_label_texts,
                past_neuron_ids=past_neuron_ids,
                past_label_texts=past_label_texts,
                current_embeddings=current_embeddings,
                past_embeddings=past_embeddings,
                keyword_embedding=keyword_embedding,
                k=k,
            )

        # Build figure: plain-current, plain-past, highlight-current,
        # highlight-past, keyword.  Trace indices are deterministic so
        # the JS handlers built below can reference them by number.
        current_highlight_indices = {m.local_index for m in current_top_k}
        past_highlight_indices = {m.local_index for m in past_top_k}

        plain_current_idx = [i for i in range(n_current) if i not in current_highlight_indices]
        plain_past_idx = [i for i in range(n_past) if i not in past_highlight_indices]

        self._fig = go.Figure()
        self._fig.add_trace(
            _make_plain_trace(
                name="Current Run",
                coords=self.current_umap_coords[plain_current_idx],
                hover=[
                    f"<b>Current Neuron {self.current_neuron_ids[i]}</b><br>"
                    f"{self.current_label_texts[i]}"
                    for i in plain_current_idx
                ],
                color=CURRENT_PLAIN_COLOR,
                size=point_size,
            )
        )
        self._fig.add_trace(
            _make_plain_trace(
                name="Past Run",
                coords=self.past_umap_coords[plain_past_idx],
                hover=[
                    f"<b>Past Neuron {past_neuron_ids[i]}</b><br>{past_label_texts[i]}"
                    for i in plain_past_idx
                ],
                color=PAST_PLAIN_COLOR,
                size=point_size,
            )
        )

        self._fig.add_trace(
            _make_highlight_trace(
                name="Top current",
                matches=current_top_k,
                coords=self.current_umap_coords,
                color=CURRENT_HIGHLIGHT_COLOR,
                origin_label="Current",
            )
        )
        current_highlight_trace_index = 2
        self._fig.add_trace(
            _make_highlight_trace(
                name="Top past",
                matches=past_top_k,
                coords=self.past_umap_coords,
                color=PAST_HIGHLIGHT_COLOR,
                origin_label="Past",
            )
        )
        past_highlight_trace_index = 3

        self._fig.add_trace(
            go.Scatter(
                x=[keyword_coords[0]],
                y=[keyword_coords[1]],
                mode="markers",
                name=f"Keyword: {keyword}",
                marker={
                    "size": KEYWORD_MARKER_SIZE,
                    "symbol": "star",
                    "color": KEYWORD_COLOR,
                    "line": {"width": 1, "color": "#065f46"},
                },
                text=[f"<b>Keyword</b><br>{keyword}"],
                hovertemplate="%{text}<extra></extra>",
            )
        )

        scope_label = "per side" if search_scope == "separate" else "pooled"
        self._fig.update_layout(
            title={
                "text": (
                    f'Embedding map (compare) — top-{k} closest to "{keyword}" ({scope_label})'
                ),
                "font": {"size": 18},
            },
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            hovermode="closest",
            template="plotly_white",
            legend={"title": {"text": "Trace"}},
            margin={"t": 60, "r": 20, "b": 40, "l": 50},
        )

        self._sidebars = _build_sidebars(
            current_top_k=current_top_k,
            past_top_k=past_top_k,
            current_trace_idx=current_highlight_trace_index,
            past_trace_idx=past_highlight_trace_index,
            search_scope=search_scope,
        )
        self._keyword = keyword

        self.top_k_matches = json.dumps(
            {
                "current": [m.to_dict() for m in current_top_k],
                "past": [m.to_dict() for m in past_top_k],
            },
            indent=2,
        )

        self.embedding_provider_param = embedding_provider
        self.embedding_model_param = embedding_model
        self.umap_n_neighbors_param = umap_n_neighbors
        self.umap_min_dist_param = umap_min_dist
        self.umap_metric_param = umap_metric
        self.umap_random_state_param = umap_random_state
        self.keyword_param = keyword
        self.k_param = k
        self.search_scope_param = search_scope
        self.past_run_id_param = past_run_id
        self.num_neurons_current_param = n_current
        self.num_neurons_past_param = n_past

    def update_context(self) -> None:
        """Log standard artifacts via base class, then save the custom HTML page."""
        super().update_context()
        html = render_keyword_search_html(self._fig, self._sidebars, self._keyword)
        with tempfile.TemporaryDirectory() as tmp:
            with open(f"{tmp}/keyword_search_map.html", "w", encoding="utf-8") as f:
                f.write(html)
            mlflow.log_artifacts(tmp)
        logger.info("Compare keyword-search embedding map saved to mlflow artifacts as HTML")


# ── Helpers ─────────────────────────────────────────────────


@dataclass(frozen=True)
class _Match:
    """One top-k match record paired with its origin-side index.

    Attributes:
        neuron_id: Neuron id from the originating side.
        label: Label text for that neuron.
        similarity: Cosine similarity to the keyword.
        local_index: Index of the neuron within its origin's
            coords/labels arrays — not the global pooled index.
    """

    neuron_id: str
    label: str
    similarity: float
    local_index: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict (drops local_index)."""
        return {
            "neuron_id": self.neuron_id,
            "label": self.label,
            "similarity": self.similarity,
        }


def _build_per_side_top_k(
    *,
    neuron_ids: list[str],
    label_texts: list[str],
    embeddings: np.ndarray,
    keyword_embedding: np.ndarray,
    k: int,
) -> list[_Match]:
    """Compute top-k matches against a single side's embeddings."""
    indices, similarities = find_top_k_nearest(keyword_embedding, embeddings, k)
    return [
        _Match(
            neuron_id=neuron_ids[int(idx)],
            label=label_texts[int(idx)],
            similarity=float(sim),
            local_index=int(idx),
        )
        for idx, sim in zip(indices, similarities, strict=True)
    ]


def _build_pooled_top_k(
    *,
    current_neuron_ids: list[str],
    current_label_texts: list[str],
    past_neuron_ids: list[str],
    past_label_texts: list[str],
    current_embeddings: np.ndarray,
    past_embeddings: np.ndarray,
    keyword_embedding: np.ndarray,
    k: int,
) -> tuple[list[_Match], list[_Match]]:
    """Compute one pooled ranking and split it back into per-origin matches.

    The pooled rank order is preserved within each side's list so
    sidebar items in "combined" mode read in global rank order when
    interleaved by similarity.
    """
    pooled_embeddings = np.concatenate([current_embeddings, past_embeddings], axis=0)
    indices, similarities = find_top_k_nearest(keyword_embedding, pooled_embeddings, k)

    n_current = len(current_label_texts)
    current_matches: list[_Match] = []
    past_matches: list[_Match] = []
    for idx, sim in zip(indices, similarities, strict=True):
        idx_int = int(idx)
        if idx_int < n_current:
            current_matches.append(
                _Match(
                    neuron_id=current_neuron_ids[idx_int],
                    label=current_label_texts[idx_int],
                    similarity=float(sim),
                    local_index=idx_int,
                )
            )
        else:
            past_idx = idx_int - n_current
            past_matches.append(
                _Match(
                    neuron_id=past_neuron_ids[past_idx],
                    label=past_label_texts[past_idx],
                    similarity=float(sim),
                    local_index=past_idx,
                )
            )
    return current_matches, past_matches


def _make_plain_trace(
    *,
    name: str,
    coords: np.ndarray,
    hover: list[str],
    color: str,
    size: int,
) -> go.Scatter:
    """Build the plain (non-highlighted) scatter trace for one side."""
    return go.Scatter(
        x=coords[:, 0] if len(coords) > 0 else [],
        y=coords[:, 1] if len(coords) > 0 else [],
        mode="markers",
        name=name,
        marker={"size": size, "opacity": 0.55, "color": color},
        text=hover,
        hovertemplate="%{text}<extra></extra>",
    )


def _make_highlight_trace(
    *,
    name: str,
    matches: list[_Match],
    coords: np.ndarray,
    color: str,
    origin_label: str,
) -> go.Scatter:
    """Build a highlight-trace scatter for one side's top-k matches.

    The trace's points are indexed in the same order as *matches* so
    sidebar JS can target ``matches[i]`` by passing point index ``i``
    to ``Plotly.restyle``.
    """
    if not matches:
        # An empty highlight trace still needs to occupy a stable
        # trace index so the keyword trace sits at the same spot
        # regardless of whether matches were found.
        return go.Scatter(
            x=[],
            y=[],
            mode="markers",
            name=name,
            marker={
                "size": HIGHLIGHT_DEFAULT_SIZE,
                "color": color,
                "line": {"width": 1, "color": "#1f2937"},
            },
        )
    pts = coords[[m.local_index for m in matches]]
    hover = [
        (
            f"<b>{origin_label} Neuron {m.neuron_id}</b><br>{m.label}<br>"
            f"<b>Similarity:</b> {m.similarity:.4f}"
        )
        for m in matches
    ]
    return go.Scatter(
        x=pts[:, 0],
        y=pts[:, 1],
        mode="markers",
        name=name,
        marker={
            "size": HIGHLIGHT_DEFAULT_SIZE,
            "opacity": 0.95,
            "color": color,
            "line": {"width": 1, "color": "#1f2937"},
        },
        text=hover,
        hovertemplate="%{text}<extra></extra>",
    )


def _build_sidebars(
    *,
    current_top_k: list[_Match],
    past_top_k: list[_Match],
    current_trace_idx: int,
    past_trace_idx: int,
    search_scope: SearchScope,
) -> list[Sidebar]:
    """Build the sidebar(s) per the configured search scope."""
    current_items = [
        SidebarItem(
            label=f"[neuron {m.neuron_id}] {m.label}",
            similarity=m.similarity,
            trace_index=current_trace_idx,
            point_index=i,
            badge="current" if search_scope == "combined" else None,
        )
        for i, m in enumerate(current_top_k)
    ]
    past_items = [
        SidebarItem(
            label=f"[neuron {m.neuron_id}] {m.label}",
            similarity=m.similarity,
            trace_index=past_trace_idx,
            point_index=i,
            badge="past" if search_scope == "combined" else None,
        )
        for i, m in enumerate(past_top_k)
    ]

    if search_scope == "separate":
        return [
            Sidebar(title=f"Current — top {len(current_items)}", items=current_items),
            Sidebar(title=f"Past — top {len(past_items)}", items=past_items),
        ]
    pooled = sorted(current_items + past_items, key=lambda i: i.similarity, reverse=True)
    return [Sidebar(title=f"Top {len(pooled)} matches (pooled)", items=pooled)]
