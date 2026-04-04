import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """Base interface for all recommendation models.

    All models in the registry must implement encode, decode, and recommend.
    """

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, e: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def recommend(
        self,
        interaction_batch: torch.Tensor,
        k: int | None = None,
        mask_interactions: bool = True,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def get_config(self) -> dict:
        """Return architecture config needed to reconstruct this model."""
        raise NotImplementedError

    @staticmethod
    def normalize_relevance_scores(relevance_scores: torch.Tensor) -> torch.Tensor:
        """Normalize relevance scores to [0, 1] range."""
        maxs = torch.max(relevance_scores, dim=-1, keepdim=True)[0]
        mins = torch.min(relevance_scores, dim=-1, keepdim=True)[0]
        return (relevance_scores - mins) / (maxs - mins + 1e-8)
