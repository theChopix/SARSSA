import torch
import numpy as np
import scipy.sparse as sp

from utils.datasets.data_loader import DataLoader
from utils.torch.models.base_model import BaseModel
from utils.torch.models.sae_model import SAE
from utils.torch.models.fused_model import FusedModel


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
    
    target_mask = np.concatenate(
        [
            np.random.permutation(np.array([True] * int(np.ceil(row_nnz * target_ratio)) + [False] * int((row_nnz - np.ceil(row_nnz * target_ratio)))))
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
    predicted_batch = torch.zeros_like(target).scatter(1, batch_topk_indices, torch.ones_like(batch_topk_indices, dtype=bool))
    r = (predicted_batch & target).sum(axis=1) / torch.minimum(target.sum(axis=1), torch.ones_like(target.sum(axis=1)) * k)
    return r


def evaluate_recall_at_k_from_elsa(model: BaseModel, inputs: DataLoader, targets: DataLoader, k: int) -> np.ndarray:
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
        _, topk_indices = model.recommend(input_batch, k, mask_interactions=True)
        recall.append(_recall_at_k_batch(torch.tensor(topk_indices).to(target_batch.device), target_batch, k))
    return torch.cat(recall).detach().cpu().numpy()


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
    relevance = target_batch.gather(1, topk_batch).float()
    
    gains = 2**relevance - 1
    discounts = torch.log2(torch.arange(2, k + 2, device=relevance.device, dtype=torch.float))
    dcg = (gains / discounts).sum(dim=1)
    
    sorted_relevance, _ = torch.sort(target_batch.float(), dim=1, descending=True)
    ideal_gains = 2 ** sorted_relevance[:, :k] - 1
    ideal_discounts = torch.log2(torch.arange(2, k + 2, device=relevance.device, dtype=torch.float))
    idcg = (ideal_gains / ideal_discounts).sum(dim=1)
    idcg[idcg == 0] = 1
    
    return dcg / idcg, dcg, idcg


def evaluate_ndcg_at_k_from_elsa(model: BaseModel, inputs: DataLoader, targets: DataLoader, k: int) -> np.ndarray:
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
        _, topk_indices = model.recommend(input_batch, k, mask_interactions=True)
        ndcg.append(ndcg_at_k(torch.tensor(topk_indices).to(target_batch.device), target_batch, k)[0])
    return torch.cat(ndcg).detach().cpu().numpy()


def evaluate_dense_encoder(model: BaseModel, split_csr: sp.csr_matrix, target_ratio: float, batch_size: int, device, seed: int = 42) -> dict[str, float]:
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
    inputs, targets = split_input_target_interactions(split_csr, target_ratio, seed)
    inputs = DataLoader(inputs, batch_size, device, shuffle=False)
    targets = DataLoader(targets, batch_size, device, shuffle=False)
    
    recalls = evaluate_recall_at_k_from_elsa(model, inputs, targets, k=20)
    ndcgs = evaluate_ndcg_at_k_from_elsa(model, inputs, targets, k=20)
    
    return {
        'R20': float(np.mean(recalls)),
        'NDCG20': float(np.mean(ndcgs))
    }


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


def evaluate_l0(sparse_embeddings: torch.Tensor) -> np.ndarray:
    """Compute L0 norm (number of non-zero elements) for sparse embeddings.
    
    Measures achieved sparsity by counting active features per embedding.
    
    Args:
        sparse_embeddings: Sparse embeddings tensor of shape (num_samples, embedding_dim).
    
    Returns:
        np.ndarray: L0 norms of shape (num_samples,), indicating number of active features.
    """
    return (sparse_embeddings > 0).float().sum(-1).detach().cpu().numpy()


def evaluate_dead_neurons(sparse_embeddings: torch.Tensor) -> int:
    """Count the number of dead neurons (never activate across all samples).
    
    Dead neurons indicate wasted model capacity and potential training issues.
    
    Args:
        sparse_embeddings: Sparse embeddings tensor of shape (num_samples, embedding_dim).
    
    Returns:
        int: Number of neurons that never activate across all samples.
    """
    return int((sparse_embeddings.sum(0) == 0).sum().detach().cpu().numpy())


def evaluate_sparse_encoder(base_model: BaseModel, sae_model: SAE, split_csr: sp.csr_matrix, 
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
    inputs, targets = split_input_target_interactions(split_csr, target_ratio, seed)
    inputs = DataLoader(inputs, batch_size, device, shuffle=False)
    targets = DataLoader(targets, batch_size, device, shuffle=False)
    full = DataLoader(split_csr, batch_size, device, shuffle=False)

    fused_model = FusedModel(base_model, sae_model)
    
    base_model.eval()
    sae_model.eval()
    fused_model.eval()
    
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
    
    elsa_recommendations = np.vstack([base_model.recommend(batch, 20, mask_interactions=True)[1] for batch in inputs])
    elsa_rec_dataloader = DataLoader(elsa_recommendations, batch_size, device, shuffle=False)
    sae_recommendations = np.vstack([fused_model.recommend(batch, 20, mask_interactions=True)[1] for batch in inputs])
    sae_rec_dataloader = DataLoader(sae_recommendations, batch_size, device, shuffle=False)
    
    cosines = evaluate_cosine_similarity(embeddings, reconstructed_embeddings)
    l0s = evaluate_l0(sparse_embeddings)
    dead_neurons = evaluate_dead_neurons(sparse_embeddings)
    
    recalls = np.mean(np.concatenate(
        [_recall_at_k_batch(recs, targs, 20).cpu().numpy() for recs, targs in zip(elsa_rec_dataloader, targets)]
    ))
    recalls_with_sae = np.mean(np.concatenate(
        [_recall_at_k_batch(recs, targs, 20).cpu().numpy() for recs, targs in zip(sae_rec_dataloader, targets)]
    ))
    recall_degradations = recalls_with_sae - recalls
    
    ndcgs = np.mean(np.concatenate(
        [ndcg_at_k(recs, targs, 20)[0].cpu().numpy() for recs, targs in zip(elsa_rec_dataloader, targets)]
    ))
    ndcgs_with_sae = np.mean(np.concatenate(
        [ndcg_at_k(recs, targs, 20)[0].cpu().numpy() for recs, targs in zip(sae_rec_dataloader, targets)]
    ))
    ndcg_degradations = ndcgs_with_sae - ndcgs

    return {
        'CosineSim': float(np.mean(cosines)),
        'L0': float(np.mean(l0s)),
        'DeadNeurons': dead_neurons / sae_model.embedding_dim,
        'R20': float(recalls_with_sae),
        'R20_Degradation': float(recall_degradations),
        'NDCG20_base': float(ndcgs),
        'NDCG20': float(ndcgs_with_sae),
        'NDCG20_Degradation': float(ndcg_degradations)
    }
