import torch
import numpy as np
import random
import scipy.sparse as sp
import os

from utils.datasets.data_loader import DataLoader
from utils.models.elsa import ELSA


class Utils:
    """Utility class for ELSA model training and evaluation.
    
    This class provides static methods for:
    - Device and seed management for reproducibility
    - Data splitting for evaluation
    - Recommendation metrics computation (Recall@K, NDCG@K)
    - Model evaluation on validation/test sets
    - Model checkpoint saving
    """
    @staticmethod
    def set_seed(seed: int) -> None:
        """Set random seeds for reproducibility across all libraries.
        
        Args:
            seed: Random seed value.
        """
        torch.manual_seed(seed)  # CPU seed
        torch.mps.manual_seed(seed)  # Metal (Apple Silicon) seed
        torch.cuda.manual_seed(seed)  # Single GPU seed
        torch.cuda.manual_seed_all(seed)  # Multi-GPU seed
        np.random.seed(seed)  # NumPy seed
        random.seed(seed)  # Python seed
        
    @staticmethod
    def set_device():
        """Detect and return the best available device for training.
        
        Priority: CUDA > MPS (Apple Silicon) > CPU
        
        Returns:
            torch.device: The selected device.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device('mps') if torch.backends.mps.is_available() else device
        return device
    
    @staticmethod
    def split_input_target_interactions(user_item_csr: sp.csr_matrix, target_ratio: float, seed: int = 42) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """Split user-item interactions into input and target sets for evaluation.
        
        For each user, randomly selects a portion of their interactions as targets
        and uses the rest as inputs. This simulates a recommendation scenario where
        we predict held-out interactions.
        
        Args:
            user_item_csr: Sparse CSR matrix of user-item interactions (users x items).
            target_ratio: Ratio of interactions to use as targets (0.0-1.0).
            seed: Random seed for reproducibility.
        
        Returns:
            tuple: (inputs, targets) - Two sparse CSR matrices with the same shape.
        """
        np.random.seed(seed)
        
        # Create a random mask for each user's interactions
        target_mask = np.concatenate(
            [
                np.random.permutation(np.array([True] * int(np.ceil(row_nnz * target_ratio)) + [False] * int((row_nnz - np.ceil(row_nnz * target_ratio)))))
                for row_nnz in np.diff(user_item_csr.indptr)
            ]
        )
        
        # Create input and target matrices
        inputs: sp.csr_matrix = user_item_csr.copy()
        targets: sp.csr_matrix = user_item_csr.copy()
        
        # Apply masks: inputs get non-target interactions, targets get target interactions
        inputs.data *= ~target_mask
        targets.data *= target_mask
        
        # Remove zero entries to maintain sparsity
        inputs.eliminate_zeros()
        targets.eliminate_zeros()
        
        return inputs, targets
    
    @staticmethod
    def _recall_at_k_batch(batch_topk_indices: torch.Tensor, batch_target: torch.Tensor, k: int) -> torch.Tensor:
        """Compute Recall@K for a batch of predictions.
        
        Recall@K measures the proportion of relevant items that appear in the top-K
        recommendations. Formula from https://arxiv.org/pdf/1802.05814
        
        Args:
            batch_topk_indices: Indices of top-K predicted items (batch_size x K).
            batch_target: Binary target matrix (batch_size x num_items).
            k: Number of top recommendations to consider.
        
        Returns:
            torch.Tensor: Recall@K scores for each user in the batch (batch_size,).
        """
        target = batch_target.bool()
        # Create binary prediction matrix from top-K indices
        predicted_batch = torch.zeros_like(target).scatter(1, batch_topk_indices, torch.ones_like(batch_topk_indices, dtype=bool))
        # Recall = (# relevant items in top-K) / min(# relevant items, K)
        r = (predicted_batch & target).sum(axis=1) / torch.minimum(target.sum(axis=1), torch.ones_like(target.sum(axis=1)) * k)
        return r
    
    @staticmethod
    def evaluate_recall_at_k_from_elsa(model: ELSA, inputs: DataLoader, targets: DataLoader, k: int) -> np.ndarray:
        """Evaluate Recall@K for an ELSA model on a dataset.
        
        Args:
            model: Trained ELSA model.
            inputs: DataLoader with input interactions.
            targets: DataLoader with target interactions.
            k: Number of top recommendations to consider.
        
        Returns:
            np.ndarray: Recall@K scores for all users.
        """
        recall = []
        for input_batch, target_batch in zip(inputs, targets):
            # Generate top-K recommendations (masking already-seen items)
            _, topk_indices = model.recommend(input_batch, k, mask_interactions=True)
            # Compute recall for this batch
            recall.append(Utils._recall_at_k_batch(torch.tensor(topk_indices).to(target_batch.device), target_batch, k))
        return torch.cat(recall).detach().cpu().numpy()
    
    @staticmethod
    def ndcg_at_k(topk_batch: torch.Tensor, target_batch: torch.Tensor, k: int) -> torch.Tensor:
        """Compute Normalized Discounted Cumulative Gain (NDCG@K) for a batch.
        
        NDCG@K measures ranking quality by considering both relevance and position.
        Higher-ranked relevant items contribute more to the score.
        
        Args:
            topk_batch: Indices of top-K predicted items (batch_size x K).
            target_batch: Binary target matrix (batch_size x num_items).
            k: Number of top recommendations to consider.
        
        Returns:
            tuple: (ndcg, dcg, idcg) - NDCG scores, DCG scores, and ideal DCG scores.
        """
        target_batch = target_batch.bool()
        # Get relevance of predicted items
        relevance = target_batch.gather(1, topk_batch).float()
        
        # Compute DCG@k (Discounted Cumulative Gain)
        gains = 2**relevance - 1
        discounts = torch.log2(torch.arange(2, k + 2, device=relevance.device, dtype=torch.float))
        dcg = (gains / discounts).sum(dim=1)
        
        # Compute IDCG@k (Ideal DCG - best possible ranking)
        sorted_relevance, _ = torch.sort(target_batch.float(), dim=1, descending=True)
        ideal_gains = 2 ** sorted_relevance[:, :k] - 1
        ideal_discounts = torch.log2(torch.arange(2, k + 2, device=relevance.device, dtype=torch.float))
        idcg = (ideal_gains / ideal_discounts).sum(dim=1)
        idcg[idcg == 0] = 1  # Avoid division by zero
        
        # Compute nDCG@k (normalized)
        return dcg / idcg, dcg, idcg

    @staticmethod
    def evaluate_ndcg_at_k_from_elsa(model: ELSA, inputs: DataLoader, targets: DataLoader, k: int) -> np.ndarray:
        """Evaluate NDCG@K for an ELSA model on a dataset.
        
        Args:
            model: Trained ELSA model.
            inputs: DataLoader with input interactions.
            targets: DataLoader with target interactions.
            k: Number of top recommendations to consider.
        
        Returns:
            np.ndarray: NDCG@K scores for all users.
        """
        ndcg = []
        for input_batch, target_batch in zip(inputs, targets):
            # Generate top-K recommendations (masking already-seen items)
            _, topk_indices = model.recommend(input_batch, k, mask_interactions=True)
            # Compute NDCG for this batch (take first element of tuple)
            ndcg.append(Utils.ndcg_at_k(torch.tensor(topk_indices).to(target_batch.device), target_batch, k)[0])
        return torch.cat(ndcg).detach().cpu().numpy()
    
    @staticmethod
    def evaluate_dense_encoder(model: ELSA, split_csr: sp.csr_matrix, target_ratio: float, batch_size: int, device, seed: int = 42) -> dict[str, float]:
        """Evaluate an ELSA model using Recall@20 and NDCG@20 metrics.
        
        This is the main evaluation function that:
        1. Splits interactions into inputs and targets
        2. Computes Recall@20 and NDCG@20
        3. Returns average scores across all users
        
        Args:
            model: Trained ELSA model to evaluate.
            split_csr: Sparse CSR matrix of user-item interactions.
            target_ratio: Ratio of interactions to use as targets.
            batch_size: Batch size for evaluation.
            device: PyTorch device for computation.
            seed: Random seed for reproducibility.
        
        Returns:
            dict: Dictionary with 'R20' (Recall@20) and 'NDCG20' (NDCG@20) scores.
        """
        # Split data into inputs and targets for evaluation
        inputs, targets = Utils.split_input_target_interactions(split_csr, target_ratio, seed)
        inputs = DataLoader(inputs, batch_size, device, shuffle=False)
        targets = DataLoader(targets, batch_size, device, shuffle=False)
        
        # Compute metrics
        recalls = Utils.evaluate_recall_at_k_from_elsa(model, inputs, targets, k=20)
        ndcgs = Utils.evaluate_ndcg_at_k_from_elsa(model, inputs, targets, k=20)
        
        return {
            'R20': float(np.mean(recalls)),
            'NDCG20': float(np.mean(ndcgs))
        }

    @staticmethod
    def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, filepath: str) -> None:
        """Save model and optimizer state to a checkpoint file.
        
        Args:
            model: PyTorch model to save.
            optimizer: Optimizer to save.
            filepath: Path where checkpoint will be saved.
        """
        # Move model to CPU before saving to ensure compatibility
        checkpoint = {
            "model_state_dict": model.to('cpu').state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }
        
        # Create directory if it doesn't exist
        if '/' in filepath:
            os.makedirs("/".join(filepath.split("/")[:-1]), exist_ok=True)
        
        torch.save(checkpoint, filepath)
        
    @staticmethod
    def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, filepath: str, device) -> None:
        """Load model and optimizer state from a checkpoint file.
        
        Args:
            model: PyTorch model to load state into.
            optimizer: Optimizer to load state into.
            filepath: Path to checkpoint file.
            device: Device to map tensors to.
        """
        checkpoint = torch.load(filepath, weights_only=True, map_location=str(device))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])