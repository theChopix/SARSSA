import torch
import torch.nn as nn

from utils.torch.models.sae_model import SAE
from utils.torch.models.model_registry import register_sae_model


@register_sae_model("BatchTopKSAE")
class BatchTopKSAE(SAE):
    def __init__(self, input_dim: int, embedding_dim: int, **kwargs):
        super().__init__(input_dim, embedding_dim, **kwargs)
        self.threshold = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.processed_batches_count = nn.Parameter(torch.zeros(1), requires_grad=False)
        
        self.k = self.cfg["k"]
        self.reconstruction_coef = self.cfg.get("reconstruction_coef", 1.0)
        self.auxiliary_coef = self.cfg.get("auxiliary_coef", 0)
        self.contrastive_coef = self.cfg.get("contrastive_coef", 0)
        self.l1_coef = self.cfg.get("l1_coef", 0)
        
    def _update_threshold(self, min_batch_value: float) -> None:
        self.processed_batches_count += 1
        self.threshold += (min_batch_value - self.threshold) / self.processed_batches_count

    def post_process_embedding(self, e: torch.Tensor) -> torch.Tensor:
        if self.training:
            batch_topk = torch.topk(e.flatten(), self.k * e.shape[0], dim=-1)
            e_topk = torch.zeros_like(e.flatten()).scatter(-1, batch_topk.indices, batch_topk.values).reshape(e.shape)
            min_nonzero_value = batch_topk.values[batch_topk.values > 0].min().cpu().item()
            self._update_threshold(min_nonzero_value)
        else:
            e_topk = torch.where(e > self.threshold, e, torch.zeros_like(e))
        return e_topk
    
    def total_loss(self, partial_losses: dict) -> torch.Tensor:
        rec_coef, aux_coef, con_coef = self.cfg["reconstruction_coef"], self.cfg["auxiliary_coef"], self.cfg["contrastive_coef"]
        
        reconstruction_loss = rec_coef * partial_losses[self.reconstruction_loss] + self.l1_coef * partial_losses["L1"]
        auxiliary_loss = aux_coef * partial_losses["Auxiliary"]
        contrastive_loss = con_coef * partial_losses["Contrastive"]
        
        return reconstruction_loss + auxiliary_loss + contrastive_loss

    def get_config(self) -> dict:
        return {
            "model_type": "BatchTopKSAE",
            "architecture": {
                "input_dim": self.input_dim,
                "embedding_dim": self.embedding_dim,
                "k": self.k,
                "reconstruction_loss": self.reconstruction_loss,
                "normalize": self.normalize,
                "auxiliary_coef": self.cfg.get("auxiliary_coef", 0),
                "contrastive_coef": self.cfg.get("contrastive_coef", 0),
                "l1_coef": self.l1_coef,
                "reconstruction_coef": self.reconstruction_coef,
                "n_batches_to_dead": self.n_batches_to_dead,
                "topk_aux": self.topk_aux,
            }
        }
