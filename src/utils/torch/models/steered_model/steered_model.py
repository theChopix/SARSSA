"""Steered recommendation model that modifies SAE neuron activations.

Implements the steering mechanism from the LLM4MORS research: given a user's
interaction vector, the model encodes it through the base model and SAE, then
redistributes the SAE activation budget so that a fraction ``alpha`` is
allocated to the target neuron(s) while the remaining ``(1 - alpha)`` is
spread across the original activations.  The modified SAE embedding is then
decoded back through the base model to produce steered recommendation scores.

Three steering modes are supported:

* **single neuron** – boost one specific neuron id.
* **multiple neurons** – boost several neuron ids (activation budget is split
  equally among them).
* **concept** – resolve a concept (tag) to a neuron id via a concept→neuron
  mapping tensor and then steer on that neuron.
"""

import torch
import torch.nn as nn

from utils.torch.models.base_model import BaseModel
from utils.torch.models.sae_model import SAE


class SteeredModel(nn.Module):
    """Wraps a base CF model + SAE and applies neuron-level steering.

    The steering formula (per user embedding ``e`` with total activation
    ``s = e.sum()``):

    1. Rescale original activations: ``e *= (1 - alpha) / s``
    2. Inject steering signal: ``e[target_neurons] += alpha / n_targets``
    3. Restore original total activation: ``e *= s``

    This preserves the overall activation magnitude while shifting the
    representation toward the target concept.

    Args:
        base_model: Pre-trained collaborative filtering model (frozen).
        sae: Trained sparse autoencoder (frozen).
        alpha: Steering strength in [0, 1].  0 = no steering, 1 = full override.
    """

    def __init__(self, base_model: BaseModel, sae: SAE, alpha: float = 0.0):
        super().__init__()
        self.base_model = base_model
        self.sae = sae
        self.alpha = alpha

    def eval(self):
        self.base_model.eval()
        self.sae.eval()
        return self

    # ------------------------------------------------------------------
    # Core encode / decode through base-model + SAE
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_to_sae(
        self, interaction_batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode interactions → base model embedding → SAE sparse embedding.

        Returns:
            sae_embeddings, input_mean, input_std  (needed for SAE decode).
        """
        user_embeddings = self.base_model.encode(interaction_batch)
        sae_embeddings, _, input_mean, input_std, _ = self.sae.encode(user_embeddings)
        return sae_embeddings, input_mean, input_std

    @torch.no_grad()
    def _decode_from_sae(
        self,
        sae_embeddings: torch.Tensor,
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
    ) -> torch.Tensor:
        """Decode SAE embedding → base model embedding → item scores."""
        return self.base_model.decode(self.sae.decode(sae_embeddings, input_mean, input_std))

    # ------------------------------------------------------------------
    # Steering helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _steer_embeddings(
        sae_embeddings: torch.Tensor,
        neuron_indices: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """Apply the activation-redistribution steering to SAE embeddings.

        Args:
            sae_embeddings: (batch, num_neurons) sparse activation vectors.
            neuron_indices: (batch, n_targets) neuron indices to boost per user.
                For single-neuron steering this has shape (batch, 1).
            alpha: Steering strength in [0, 1].

        Returns:
            Modified sae_embeddings (same shape).
        """
        if alpha == 0.0:
            return sae_embeddings

        e = sae_embeddings.clone()
        s = e.sum(dim=-1, keepdim=True)  # total activation per user
        # Guard against zero-activation users
        s_safe = s.clamp(min=1e-8)
        e = e * ((1 - alpha) / s_safe)

        n_targets = neuron_indices.shape[1]
        per_target_alpha = alpha / n_targets

        # Scatter the steering signal into the target neurons
        batch_size = e.shape[0]
        for t in range(n_targets):
            target_col = neuron_indices[:, t]  # (batch,)
            e[torch.arange(batch_size, device=e.device), target_col] += per_target_alpha

        e = e * s  # restore original total activation magnitude
        return e

    # ------------------------------------------------------------------
    # Public steering + recommendation API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def steer_single_neuron(
        self,
        interaction_batch: torch.Tensor,
        neuron_id: int,
        alpha: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Steer toward a single neuron.

        Args:
            interaction_batch: (batch, num_items) user interaction vectors.
            neuron_id: Index of the neuron to boost.
            alpha: Override instance alpha if provided.

        Returns:
            steered_sae_embeddings, input_mean, input_std
        """
        a = alpha if alpha is not None else self.alpha
        sae_emb, mean, std = self._encode_to_sae(interaction_batch)
        neuron_indices = torch.full(
            (sae_emb.shape[0], 1), neuron_id, dtype=torch.long, device=sae_emb.device
        )
        steered = self._steer_embeddings(sae_emb, neuron_indices, a)
        return steered, mean, std

    @torch.no_grad()
    def steer_multiple_neurons(
        self,
        interaction_batch: torch.Tensor,
        neuron_ids: list[int],
        alpha: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Steer toward multiple neurons simultaneously.

        The steering budget ``alpha`` is split equally among the given neurons.

        Args:
            interaction_batch: (batch, num_items) user interaction vectors.
            neuron_ids: List of neuron indices to boost.
            alpha: Override instance alpha if provided.

        Returns:
            steered_sae_embeddings, input_mean, input_std
        """
        a = alpha if alpha is not None else self.alpha
        sae_emb, mean, std = self._encode_to_sae(interaction_batch)
        neuron_indices = (
            torch.tensor(neuron_ids, dtype=torch.long, device=sae_emb.device)
            .unsqueeze(0)
            .expand(sae_emb.shape[0], -1)
        )
        steered = self._steer_embeddings(sae_emb, neuron_indices, a)
        return steered, mean, std

    @torch.no_grad()
    def steer_by_concept(
        self,
        interaction_batch: torch.Tensor,
        concept_ids: torch.Tensor,
        concept_neuron_mapping: torch.Tensor,
        alpha: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Steer toward concepts resolved to neurons via a mapping tensor.

        This is the mode used in the batch-evaluation setting where each user
        in the batch may have a different target concept.

        Args:
            interaction_batch: (batch, num_items) user interaction vectors.
            concept_ids: (batch,) integer concept indices (e.g. tag indices).
            concept_neuron_mapping: (num_concepts,) tensor mapping each concept
                to a neuron index.
            alpha: Override instance alpha if provided.

        Returns:
            steered_sae_embeddings, input_mean, input_std
        """
        a = alpha if alpha is not None else self.alpha
        sae_emb, mean, std = self._encode_to_sae(interaction_batch)
        neuron_indices = concept_neuron_mapping[concept_ids].unsqueeze(-1)  # (batch, 1)
        steered = self._steer_embeddings(sae_emb, neuron_indices, a)
        return steered, mean, std

    @torch.no_grad()
    def recommend(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        concept_neuron_mapping: torch.Tensor,
        k: int = 20,
        mask_interactions: bool = True,
        alpha: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce steered top-K recommendations (concept-based, per-user).

        This mirrors the notebook's ``SAESteeredModel.recommend`` interface
        where each user in the batch has an associated concept id.

        Args:
            batch: Tuple of (interaction_batch, concept_id_batch).
            concept_neuron_mapping: (num_concepts,) tensor.
            k: Number of recommendations.
            mask_interactions: Whether to mask already-interacted items.
            alpha: Override instance alpha if provided.

        Returns:
            (topk_scores, topk_indices) both as numpy arrays.
        """
        interaction_batch, concept_id_batch = batch

        steered_emb, mean, std = self.steer_by_concept(
            interaction_batch, concept_id_batch, concept_neuron_mapping, alpha
        )

        scores = self._decode_from_sae(steered_emb, mean, std)
        scores = scores - interaction_batch
        scores = self.base_model.normalize_relevance_scores(scores)

        if mask_interactions:
            scores = torch.where(interaction_batch != 0, 0, scores)

        topk_scores, topk_indices = torch.topk(scores, k)
        return topk_scores.cpu().numpy(), topk_indices.cpu().numpy()

    @torch.no_grad()
    def recommend_single_neuron(
        self,
        interaction_batch: torch.Tensor,
        neuron_id: int,
        k: int = 20,
        mask_interactions: bool = True,
        alpha: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce steered top-K recommendations for a single neuron.

        Args:
            interaction_batch: (batch, num_items) user interaction vectors.
            neuron_id: Neuron index to steer toward.
            k: Number of recommendations.
            mask_interactions: Whether to mask already-interacted items.
            alpha: Override instance alpha if provided.

        Returns:
            (topk_scores, topk_indices) both as numpy arrays.
        """
        steered_emb, mean, std = self.steer_single_neuron(interaction_batch, neuron_id, alpha)

        scores = self._decode_from_sae(steered_emb, mean, std)
        scores = scores - interaction_batch
        scores = self.base_model.normalize_relevance_scores(scores)

        if mask_interactions:
            scores = torch.where(interaction_batch != 0, 0, scores)

        topk_scores, topk_indices = torch.topk(scores, k)
        return topk_scores.cpu().numpy(), topk_indices.cpu().numpy()

    @torch.no_grad()
    def recommend_multiple_neurons(
        self,
        interaction_batch: torch.Tensor,
        neuron_ids: list[int],
        k: int = 20,
        mask_interactions: bool = True,
        alpha: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce steered top-K recommendations for multiple neurons.

        Args:
            interaction_batch: (batch, num_items) user interaction vectors.
            neuron_ids: List of neuron indices to steer toward.
            k: Number of recommendations.
            mask_interactions: Whether to mask already-interacted items.
            alpha: Override instance alpha if provided.

        Returns:
            (topk_scores, topk_indices) both as numpy arrays.
        """
        steered_emb, mean, std = self.steer_multiple_neurons(interaction_batch, neuron_ids, alpha)

        scores = self._decode_from_sae(steered_emb, mean, std)
        scores = scores - interaction_batch
        scores = self.base_model.normalize_relevance_scores(scores)

        if mask_interactions:
            scores = torch.where(interaction_batch != 0, 0, scores)

        topk_scores, topk_indices = torch.topk(scores, k)
        return topk_scores.cpu().numpy(), topk_indices.cpu().numpy()
