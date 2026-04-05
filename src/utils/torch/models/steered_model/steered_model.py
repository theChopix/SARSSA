import torch
import torch.nn as nn

from utils.torch.models.base_model import BaseModel
from utils.torch.models.sae_model import SAE


class SteeredModel(nn.Module):
    """Wraps a base CF model + SAE and applies neuron-level steering.

    The steering formula (per user embedding ``e`` with total activation
    ``s = e.sum()``):

    1. Rescale original activations: ``e *= (1 - alpha) / s``
    2. Inject steering signal: ``e[n] += alpha / len(neuron_ids)`` for each target neuron
    3. Restore original total activation: ``e *= s``

    This preserves the overall activation magnitude while shifting the
    representation toward the target neurons.  For a single target neuron the
    formula reduces exactly to the original LLM4MORS notebook implementation.

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

    @staticmethod
    def _steer_embeddings(
        sae_embeddings: torch.Tensor,
        neuron_indices: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """Apply activation-redistribution steering to SAE embeddings.

        Args:
            sae_embeddings: (batch, num_neurons) sparse activation vectors.
            neuron_indices: (batch, n_targets) neuron indices to boost — the
                same set of neurons is applied to every user in the batch.
            alpha: Steering strength in [0, 1].

        Returns:
            Modified sae_embeddings (same shape, same total activation per user).
        """
        if alpha == 0.0:
            return sae_embeddings

        e = sae_embeddings.clone()
        s = e.sum(dim=-1, keepdim=True)  # total activation per user
        s_safe = s.clamp(min=1e-8)  # guard against zero-activation users
        e = e * ((1 - alpha) / s_safe)

        n_targets = neuron_indices.shape[1]
        per_target_alpha = alpha / n_targets

        batch_size = e.shape[0]
        for t in range(n_targets):
            e[torch.arange(batch_size, device=e.device), neuron_indices[:, t]] += per_target_alpha

        e = e * s  # restore original total activation magnitude
        return e

    @torch.no_grad()
    def steer(
        self,
        interaction_batch: torch.Tensor,
        neuron_ids: list[int],
        alpha: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Steer toward one or more neurons and return modified SAE embeddings.

        When ``neuron_ids`` contains a single element this is equivalent to the
        single-neuron steering in the LLM4MORS notebook.  For multiple neurons
        the steering budget ``alpha`` is split equally among them.

        Args:
            interaction_batch: (batch, num_items) user interaction vectors.
            neuron_ids: Neuron indices to boost (same for all users in batch).
            alpha: Override instance alpha if provided.

        Returns:
            (steered_sae_embeddings, input_mean, input_std)
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
    def recommend(
        self,
        interaction_batch: torch.Tensor,
        neuron_ids: list[int],
        k: int = 20,
        mask_interactions: bool = True,
        alpha: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce steered top-K recommendations toward the given neuron(s).

        Args:
            interaction_batch: (batch, num_items) user interaction vectors.
            neuron_ids: Neuron indices to boost (same for all users in batch).
                Pass a single-element list for single-neuron steering.
            k: Number of recommendations to return.
            mask_interactions: Whether to mask already-interacted items.
            alpha: Override instance alpha if provided.

        Returns:
            (topk_scores, topk_indices) — both numpy arrays of shape (batch, k).
        """
        steered_emb, mean, std = self.steer(interaction_batch, neuron_ids, alpha)

        scores = self._decode_from_sae(steered_emb, mean, std)
        scores = scores - interaction_batch
        scores = self.base_model.normalize_relevance_scores(scores)

        if mask_interactions:
            scores = torch.where(interaction_batch != 0, 0, scores)

        topk_scores, topk_indices = torch.topk(scores, k)
        return topk_scores.cpu().numpy(), topk_indices.cpu().numpy()
