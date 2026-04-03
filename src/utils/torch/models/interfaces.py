import torch
import torch.nn as nn
from typing import Optional, Tuple


class BaseModel(nn.Module):
    """Base interface for all recommendation models.
    
    All models in the registry must implement encode, decode, and recommend.
    """

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def recommend(
        self,
        interaction_batch: torch.Tensor,
        k: Optional[int] = None,
        mask_interactions: bool = True,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def get_config(self) -> dict:
        """Return architecture config needed to reconstruct this model."""
        raise NotImplementedError
