import torch
import torch.nn as nn

from utils.torch.models.base_model import BaseModel
from utils.torch.models.sae_model import SAE


class FusedModel(nn.Module):
    """Fuses a base model and a SAE via composition.

    Uses the base model's encode/decode interface rather than
    accessing internal attributes directly.
    """

    def __init__(self, base_model: BaseModel, sae: SAE):
        super().__init__()
        self.base_model = base_model
        self.sae = sae

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.base_model.decode(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        z_sae = self.sae(z)[0]
        return nn.ReLU()(self.decode(z_sae) - x)

    @torch.no_grad()
    def recommend(
        self,
        interaction_batch: torch.Tensor,
        k: int | None,
        mask_interactions: bool = True,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(interaction_batch)
        z_sae = self.sae(z)[0]
        scores = self.decode(z_sae) - interaction_batch
        scores = self.base_model.normalize_relevance_scores(scores)
        if k is None:
            k = scores.shape[-1]
        if mask_interactions:
            if mask is None:
                mask = interaction_batch != 0
            scores = torch.where(mask, 0, scores)
        topk_scores, topk_indices = torch.topk(scores, k)
        return topk_scores.cpu().numpy(), topk_indices.cpu().numpy()
