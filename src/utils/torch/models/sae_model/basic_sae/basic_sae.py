import torch

from utils.torch.models.sae_model import SAE
from utils.torch.models.model_registry import register_sae_model


@register_sae_model("BasicSAE")
class BasicSAE(SAE):
    def __init__(self, input_dim: int, embedding_dim: int, **kwargs):
        super().__init__(input_dim, embedding_dim, **kwargs)
        self.reconstruction_coef = self.cfg.get("reconstruction_coef", 1.0)
        self.auxiliary_coef = self.cfg.get("auxiliary_coef", 0)
        self.contrastive_coef = self.cfg.get("contrastive_coef", 0)

    def post_process_embedding(self, e: torch.Tensor) -> torch.Tensor:
        return e

    def total_loss(self, partial_losses: dict) -> torch.Tensor:
        reconstruction_loss = partial_losses[self.reconstruction_loss] + self.cfg["l1_coef"] * partial_losses["L1"]
        auxiliary_loss = partial_losses["Auxiliary"]
        return reconstruction_loss + self.cfg["auxiliary_coef"] * auxiliary_loss

    def get_config(self) -> dict:
        return {
            "model_type": "BasicSAE",
            "architecture": {
                "input_dim": self.input_dim,
                "embedding_dim": self.embedding_dim,
                "reconstruction_loss": self.reconstruction_loss,
                "normalize": self.normalize,
                "auxiliary_coef": self.cfg.get("auxiliary_coef", 0),
                "contrastive_coef": self.cfg.get("contrastive_coef", 0),
                "l1_coef": self.cfg.get("l1_coef", 0),
                "reconstruction_coef": self.reconstruction_coef,
                "n_batches_to_dead": self.n_batches_to_dead,
                "topk_aux": self.topk_aux,
            }
        }
