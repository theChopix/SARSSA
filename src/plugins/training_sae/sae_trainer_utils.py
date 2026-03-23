import torch
import numpy as np
import random
import scipy.sparse as sp
import os

from utils.datasets.data_loader import DataLoader
from utils.models.elsa import ELSA
from utils.models.sae import SAE
from utils.models.elsa_with_sae import ELSAWithSAE


class Utils:
    """Utility class for SAE model training and evaluation.
    
    This class provides static methods for:
    - Device and seed management for reproducibility
    - Data splitting for evaluation
    - Recommendation metrics computation (Recall@K, NDCG@K, Precision@K, Hit Rate@K)
    - SAE model evaluation (sparsity, reconstruction quality, recommendation degradation)
    - Model checkpoint saving and loading
    """

    @staticmethod
    def set_seed(seed: int) -> None:
        """Set random seeds for reproducibility across all libraries.
        
        Sets seeds for PyTorch (CPU, CUDA, MPS), NumPy, and Python's random module
        to ensure deterministic behavior across different runs.
        
        Args:
            seed: Random seed value to use across all libraries.
        """
        torch.manual_seed(seed)
        torch.mps.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    @staticmethod
    def set_device():
        """Automatically detect and return the best available device.
        
        Checks for device availability in order of preference:
        1. CUDA (NVIDIA GPU)
        2. MPS (Apple Silicon GPU)
        3. CPU (fallback)
        
        Returns:
            torch.device: The best available device for computation.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device('mps') if torch.backends.mps.is_available() else device
        return device

    @staticmethod
    def split_input_target_interactions(user_item_csr: sp.csr_matrix, target_ratio: float, seed: int = 42) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """Split user-item interactions into input and target sets for evaluation.
        
        For each user, randomly selects a fraction of their interactions as targets
        (held-out for testing) and uses the remaining as inputs. This simulates a
        real recommendation scenario where we try to predict held-out interactions.
        
        Args:
            user_item_csr: Sparse CSR matrix of user-item interactions (users x items).
            target_ratio: Fraction of interactions to use as targets (e.g., 0.2 for 20%).
            seed: Random seed for reproducible splitting.
        
        Returns:
            tuple: (inputs, targets) where both are sparse CSR matrices with the same
                   shape as user_item_csr. inputs contains (1-target_ratio) of interactions,
                   targets contains target_ratio of interactions.
        """
        np.random.seed(seed)
        
        # Create random mask for each user's interactions
        target_mask = np.concatenate(
            [
                np.random.permutation(
                    np.array([True] * int(np.ceil(row_nnz * target_ratio)) + 
                            [False] * int((row_nnz - np.ceil(row_nnz * target_ratio))))
                )
                for row_nnz in np.diff(user_item_csr.indptr)
            ]
        )
        
        inputs: sp.csr_matrix = user_item_csr.copy()
        targets: sp.csr_matrix = user_item_csr.copy()

        inputs.data *= ~target_mask
        targets.data *= target_mask
        inputs.eliminate_zeros()
        targets.eliminate_zeros()
        
        return inputs, targets

    @staticmethod
    def _recall_at_k_batch(batch_topk_indices: torch.Tensor, batch_target: torch.Tensor, k: int) -> torch.Tensor:
        """Compute Recall@K for a batch of predictions.
        
        Recall@K measures the proportion of relevant items that appear in the top-K
        recommendations. Formula from https://arxiv.org/pdf/1802.05814
        
        Args:
            batch_topk_indices: Tensor of shape (batch_size, k) containing indices of top-K predictions.
            batch_target: Tensor of shape (batch_size, num_items) with 1s for relevant items.
            k: Number of top recommendations to consider.
        
        Returns:
            Tensor of shape (batch_size,) containing Recall@K for each sample.
        """
        target = batch_target.bool()
        predicted_batch = torch.zeros_like(target).scatter(1, batch_topk_indices, torch.ones_like(batch_topk_indices, dtype=bool))
        r = (predicted_batch & target).sum(axis=1) / torch.minimum(target.sum(axis=1), torch.ones_like(target.sum(axis=1)) * k)
        return r

    @staticmethod
    def ndcg_at_k(topk_batch: torch.Tensor, target_batch: torch.Tensor, k: int) -> torch.Tensor:
        """Compute Normalized Discounted Cumulative Gain (NDCG@K) for a batch.
        
        NDCG@K measures ranking quality by considering both relevance and position.
        Items at higher positions contribute more to the score.
        
        Args:
            topk_batch: Tensor of shape (batch_size, k) containing indices of top-K predictions.
            target_batch: Tensor of shape (batch_size, num_items) with 1s for relevant items.
            k: Number of top recommendations to consider.
        
        Returns:
            tuple: (ndcg, dcg, idcg) where each is a tensor of shape (batch_size,).
                   ndcg is the normalized score, dcg is the actual gain, idcg is the ideal gain.
        """
        target_batch = target_batch.bool()
        relevance = target_batch.gather(1, topk_batch).float()
        
        # DCG@k - Discounted Cumulative Gain
        gains = 2**relevance - 1
        discounts = torch.log2(torch.arange(2, k + 2, device=relevance.device, dtype=torch.float))
        dcg = (gains / discounts).sum(dim=1)
        
        # IDCG@k - Ideal DCG (perfect ranking)
        sorted_relevance, _ = torch.sort(target_batch.float(), dim=1, descending=True)
        ideal_gains = 2 ** sorted_relevance[:, :k] - 1
        ideal_discounts = torch.log2(torch.arange(2, k + 2, device=relevance.device, dtype=torch.float))
        idcg = (ideal_gains / ideal_discounts).sum(dim=1)
        idcg[idcg == 0] = 1
        
        return dcg / idcg, dcg, idcg

    @staticmethod
    def evaluate_sparse_encoder(base_model: ELSA, sae_model: SAE, split_csr: sp.csr_matrix, 
                               target_ratio: float, batch_size: int, device, seed: int = 42) -> dict[str, float]:
        """Evaluate a sparse autoencoder model on recommendation quality and sparsity.
        
        This comprehensive evaluation measures:
        - Reconstruction quality (cosine similarity)
        - Sparsity metrics (L0, dead neurons)
        - Recommendation performance (Recall@20, NDCG@20)
        - Performance degradation compared to base ELSA model
        
        Args:
            base_model: Pre-trained ELSA model (frozen).
            sae_model: Trained SAE model to evaluate.
            split_csr: Sparse CSR matrix of user-item interactions for this split.
            target_ratio: Fraction of interactions to use as targets (e.g., 0.2).
            batch_size: Batch size for processing.
            device: Device to run evaluation on.
            seed: Random seed for reproducible splitting.
        
        Returns:
            dict: Dictionary containing evaluation metrics:
                - CosineSim: Average cosine similarity between original and reconstructed embeddings
                - L0: Average number of non-zero features in sparse embeddings
                - DeadNeurons: Fraction of neurons that never activate
                - R20: Recall@20 with SAE
                - R20_Degradation: Difference in Recall@20 (SAE - ELSA)
                - NDCG20_base: NDCG@20 with base ELSA model
                - NDCG20: NDCG@20 with SAE
                - NDCG20_Degradation: Difference in NDCG@20 (SAE - ELSA)
        """
        # Split data into inputs and targets for evaluation
        inputs, targets = Utils.split_input_target_interactions(split_csr, target_ratio, seed)
        inputs = DataLoader(inputs, batch_size, device, shuffle=False)
        targets = DataLoader(targets, batch_size, device, shuffle=False)
        full = DataLoader(split_csr, batch_size, device, shuffle=False)

        # Create fused model combining ELSA and SAE
        fused_model = ELSAWithSAE(base_model, sae_model)
        
        base_model.eval()
        sae_model.eval()
        fused_model.eval()
        
        # Compute embeddings and their reconstructions
        embeddings = np.vstack([base_model.encode(batch).detach().cpu().numpy() for batch in full])
        embeddings_dataloader = DataLoader(embeddings, batch_size, device, shuffle=False)
        
        embeddings = torch.tensor(embeddings, device=device)
        reconstructed_embeddings = torch.tensor(
            np.vstack([sae_model(batch)[0].detach().cpu().numpy() for batch in embeddings_dataloader]), 
            device=device
        )
        sparse_embeddings = torch.tensor(
            np.vstack([sae_model.encode(batch)[0].detach().cpu().numpy() for batch in embeddings_dataloader]), 
            device=device
        )
        
        # Generate recommendations from both models
        elsa_recommendations = np.vstack([base_model.recommend(batch, 20, mask_interactions=True)[1] for batch in inputs])
        elsa_rec_dataloader = DataLoader(elsa_recommendations, batch_size, device, shuffle=False)
        sae_recommendations = np.vstack([fused_model.recommend(batch, 20, mask_interactions=True)[1] for batch in inputs])
        sae_rec_dataloader = DataLoader(sae_recommendations, batch_size, device, shuffle=False)
        
        # Compute metrics
        cosines = Utils.evaluate_cosine_similarity(embeddings, reconstructed_embeddings)
        l0s = Utils.evaluate_l0(sparse_embeddings)
        dead_neurons = Utils.evaluate_dead_neurons(sparse_embeddings)
        
        # Compute recommendation metrics for both models
        recalls = np.mean(np.concatenate(
            [Utils._recall_at_k_batch(recs, targs, 20).cpu().numpy() for recs, targs in zip(elsa_rec_dataloader, targets)]
        ))
        recalls_with_sae = np.mean(np.concatenate(
            [Utils._recall_at_k_batch(recs, targs, 20).cpu().numpy() for recs, targs in zip(sae_rec_dataloader, targets)]
        ))
        recall_degradations = recalls_with_sae - recalls
        
        ndcgs = np.mean(np.concatenate(
            [Utils.ndcg_at_k(recs, targs, 20)[0].cpu().numpy() for recs, targs in zip(elsa_rec_dataloader, targets)]
        ))
        ndcgs_with_sae = np.mean(np.concatenate(
            [Utils.ndcg_at_k(recs, targs, 20)[0].cpu().numpy() for recs, targs in zip(sae_rec_dataloader, targets)]
        ))
        ndcg_degradations = ndcgs_with_sae - ndcgs

        return {
            'CosineSim': float(np.mean(cosines)),
            'L0': float(np.mean(l0s)),
            'DeadNeurons': dead_neurons / sae_model.encoder_w.shape[1],
            'R20': float(recalls_with_sae),
            'R20_Degradation': float(recall_degradations),
            'NDCG20_base': float(ndcgs),
            'NDCG20': float(ndcgs_with_sae),
            'NDCG20_Degradation': float(ndcg_degradations)
        }
        
    @staticmethod
    def evaluate_cosine_similarity(embeddings: torch.Tensor, reconstructed_embeddings: torch.Tensor) -> np.ndarray:
        """Compute cosine similarity between original and reconstructed embeddings.
        
        Measures reconstruction quality by computing the cosine similarity
        (direction similarity, ignoring magnitude) between embeddings.
        
        Args:
            embeddings: Original embeddings tensor of shape (num_samples, embedding_dim).
            reconstructed_embeddings: Reconstructed embeddings of same shape.
        
        Returns:
            np.ndarray: Cosine similarities of shape (num_samples,), values in [-1, 1].
        """
        return torch.nn.functional.cosine_similarity(embeddings, reconstructed_embeddings, dim=1).detach().cpu().numpy()

    @staticmethod
    def evaluate_l0(sparse_embeddings: torch.Tensor) -> np.ndarray:
        """Compute L0 norm (number of non-zero elements) for sparse embeddings.
        
        Measures achieved sparsity by counting active features per embedding.
        
        Args:
            sparse_embeddings: Sparse embeddings tensor of shape (num_samples, embedding_dim).
        
        Returns:
            np.ndarray: L0 norms of shape (num_samples,), indicating number of active features.
        """
        return (sparse_embeddings > 0).float().sum(-1).detach().cpu().numpy()

    @staticmethod
    def evaluate_dead_neurons(sparse_embeddings: torch.Tensor) -> int:
        """Count the number of dead neurons (never activate across all samples).
        
        Dead neurons indicate wasted model capacity and potential training issues.
        
        Args:
            sparse_embeddings: Sparse embeddings tensor of shape (num_samples, embedding_dim).
        
        Returns:
            int: Number of neurons that never activate across all samples.
        """
        return int((sparse_embeddings.sum(0) == 0).sum().detach().cpu().numpy())
        
    @staticmethod
    def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, filepath: str) -> None:
        """Save model and optimizer state to a checkpoint file.
        
        Saves both model parameters and optimizer state for potential training resumption.
        Creates parent directories if they don't exist.
        
        Args:
            model: PyTorch model to save.
            optimizer: Optimizer to save (contains momentum and other state).
            filepath: Path where checkpoint should be saved.
        """
        checkpoint = {
            "model_state_dict": model.to('cpu').state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }
        if '/' in filepath:
            os.makedirs("/".join(filepath.split("/")[:-1]), exist_ok=True)
        torch.save(checkpoint, filepath)
        
    @staticmethod
    def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, filepath: str, device) -> None:
        """Load model and optimizer state from a checkpoint file.
        
        Restores both model parameters and optimizer state from a saved checkpoint.
        
        Args:
            model: PyTorch model to load state into.
            optimizer: Optimizer to load state into.
            filepath: Path to the checkpoint file.
            device: Device to map the checkpoint to.
        """
        checkpoint = torch.load(filepath, weights_only=True, map_location=str(device))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])