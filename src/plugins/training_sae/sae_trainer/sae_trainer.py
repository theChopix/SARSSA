import json
import os
from copy import deepcopy

import mlflow
import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm

from plugins.plugin_interface import BasePlugin
from utils.data_loading.data_loader import DataLoader
from utils.mlflow_manager import MLflowRunLoader
from utils.plugin_logger import get_logger
from utils.torch.evalution import evaluate_sparse_encoder
from utils.torch.models.base_model import BaseModel
from utils.torch.models.model_loader import load_base_model
from utils.torch.models.model_registry import get_sae_model_class
from utils.torch.models.sae_model import SAE
from utils.torch.runtime import set_device, set_seed

logger = get_logger(__name__)


def train(
    model: SAE,
    base_model: BaseModel,
    optimizer,
    train_csr,
    valid_csr,
    test_csr,
    device,
    epochs: int,
    batch_size: int,
    early_stop: int,
    evaluate_every: int,
    contrastive_coef: float,
    sample_users: bool,
    target_ratio: float,
    seed: int,
):
    """Train a Sparse Autoencoder (SAE) model with early stopping and MLflow tracking.

    The training process learns sparse representations of dense base model embeddings while
    maintaining recommendation quality. Supports multiple training modes (from interactions
    or pre-computed embeddings) and advanced techniques like contrastive learning and
    auxiliary loss for dead neuron reactivation.

    Args:
        model: SAE model to train (BasicSAE, TopKSAE, or BatchTopKSAE).
        base_model: Pre-trained base model (frozen) used to encode interactions.
        optimizer: Optimizer for training (typically Adam).
        train_csr: Training data as sparse CSR matrix (users x items).
        valid_csr: Validation data as sparse CSR matrix.
        test_csr: Test data as sparse CSR matrix.
        device: Device to train on (cuda/mps/cpu).
        dataset: Dataset name for logging.
        epochs: Maximum number of training epochs.
        batch_size: Batch size for training.
        early_stop: Number of epochs without improvement before stopping (0 to disable).
        evaluate_every: Evaluate every N epochs.
        model_name: Name of SAE variant (BasicSAE/TopKSAE/BatchTopKSAE).
        embedding_dim: Dimension of sparse embeddings.
        top_k: Number of active features (for TopKSAE/BatchTopKSAE).
        contrastive_coef: Coefficient for contrastive loss.
        sample_users: Whether to randomly sample user interactions during training.
        target_ratio: Fraction of interactions to use as targets for evaluation.
        seed: Random seed for reproducibility.
    """

    def sampled_interactions(batch, ratio=0.8):
        """Randomly sample a fraction of interactions from a batch.

        Used for data augmentation during training.

        Args:
            batch: Batch of interactions.
            ratio: Fraction of interactions to keep (default: 0.8).

        Returns:
            Batch with randomly masked interactions.
        """
        mask = torch.rand_like(batch) < ratio
        return batch.clone() * mask

    # Create data loaders
    train_interaction_dataloader = DataLoader(train_csr, batch_size, device, shuffle=False)
    valid_interaction_dataloader = DataLoader(valid_csr, batch_size, device, shuffle=False)

    # Pre-compute embeddings for faster training (when not using augmentation)
    train_user_embeddings = np.vstack(
        [
            base_model.encode(batch).detach().cpu().numpy()
            for batch in tqdm(train_interaction_dataloader, desc="Computing training embeddings")
        ]
    )
    train_embeddings_dataloader = DataLoader(
        train_user_embeddings, batch_size, device, shuffle=True
    )

    # Pre-compute validation embeddings with sampling
    val_user_embeddings = np.vstack(
        [
            base_model.encode(sampled_interactions(batch)).detach().cpu().numpy()
            for batch in tqdm(valid_interaction_dataloader, desc="Computing validation embeddings")
        ]
    )
    valid_embeddings_dataloader = DataLoader(val_user_embeddings, batch_size, device, shuffle=False)

    # Pre-compute augmented validation embeddings for contrastive loss
    val_positive_user_embeddings = np.vstack(
        [
            base_model.encode(sampled_interactions(batch)).detach().cpu().numpy()
            for batch in tqdm(
                valid_interaction_dataloader,
                desc="Computing augmented validation embeddings",
            )
        ]
    )
    val_positive_embeddings_dataloader = DataLoader(
        val_positive_user_embeddings, batch_size, device, shuffle=False
    )

    val_user_embeddings = torch.tensor(val_user_embeddings, device=device)

    def train_epoch_from_interactions():
        """Train one epoch using on-the-fly interaction encoding.

        This mode supports data augmentation and contrastive learning but is slower
        as it encodes interactions through base model every epoch.

        Returns:
            dict: Training losses for this epoch.
        """
        train_losses = {"Loss": [], "L2": [], "L1": [], "L0": [], "Cosine": []}
        model.train()

        pbar = tqdm(train_interaction_dataloader, desc=f"Epoch {epoch}/{epochs}")
        for batched_interactions in pbar:
            # Generate positive samples for contrastive learning
            if contrastive_coef > 0:
                positive_batch = sampled_interactions(batched_interactions, ratio=0.5)
                positive_batch += (torch.rand_like(positive_batch) < 0.1).float()  # Add noise
                positive_batch = base_model.encode(positive_batch).detach()
            else:
                positive_batch = None

            # Optionally sample user interactions
            if sample_users:
                batched_interactions = sampled_interactions(batched_interactions)

            # Encode interactions through base model
            embedding_batch = base_model.encode(batched_interactions).detach()

            # Train SAE
            losses = model.train_step(optimizer, embedding_batch, positive_batch)
            pbar.set_postfix({"train_loss": losses["Loss"].cpu().item()})
            for key, val in train_losses.items():
                val.append(losses[key].item())
        return train_losses

    def train_epoch_from_embeddings():
        """Train one epoch using pre-computed embeddings.

        This mode is faster but doesn't support data augmentation or contrastive learning.

        Returns:
            dict: Training losses for this epoch.
        """
        train_losses = {"Loss": [], "L2": [], "L1": [], "L0": [], "Cosine": []}
        model.train()
        pbar = tqdm(train_embeddings_dataloader, desc=f"Epoch {epoch}/{epochs}")
        for batched_embeddings in pbar:
            losses = model.train_step(optimizer, batched_embeddings, None)
            pbar.set_postfix({"train_loss": losses["Loss"].cpu().item()})
            for key, val in train_losses.items():
                val.append(losses[key].item())

        return train_losses

    # Initialize early stopping
    if early_stop > 0:
        best_epoch = 0
        epochs_without_improvement = 0
        best_loss = np.inf
        best_optimizer = deepcopy(optimizer)
        best_model = deepcopy(model)

    # Determine training mode based on whether we need on-the-fly encoding
    are_interactions_needed = sample_users or contrastive_coef > 0

    # Training loop
    for epoch in range(1, epochs + 1):
        if are_interactions_needed:
            train_losses = train_epoch_from_interactions()
        else:
            train_losses = train_epoch_from_embeddings()

        # Log training losses
        for key, val in train_losses.items():
            mlflow.log_metric(f"loss/{key}/train", float(np.mean(val)), step=epoch)

        # Periodic evaluation
        if epoch % evaluate_every == 0:
            model.eval()
            # Compute validation losses
            valid_losses = {
                "Loss": [],
                "L2": [],
                "L1": [],
                "L0": [],
                "Cosine": [],
                "Auxiliary": [],
                "Contrastive": [],
            }
            for embedding, positive_embedding in zip(
                valid_embeddings_dataloader, val_positive_embeddings_dataloader
            ):
                losses = model.compute_loss_dict(embedding, positive_embedding)
                for key, val in losses.items():
                    valid_losses[key].append(val.item())

            # Log validation losses
            for key, val in valid_losses.items():
                mlflow.log_metric(f"loss/{key}/valid", float(np.mean(val)), step=epoch)

            # Compute validation metrics (sparsity, reconstruction quality, recommendation performance)
            valid_metrics = evaluate_sparse_encoder(
                base_model,
                model,
                valid_csr,
                target_ratio,
                batch_size,
                device,
                seed=seed,
            )
            for key, val in valid_metrics.items():
                mlflow.log_metric(f"{key}/valid", val, step=epoch)

            valid_loss = float(np.mean(valid_losses["Loss"]))
            logger.info(
                f"Epoch {epoch}/{epochs} - "
                f"Loss: {valid_loss:.4f} - "
                f"Cosine: {valid_metrics['CosineSim']:.4f} - "
                f"NDCG20 Degradation: {valid_metrics['NDCG20_Degradation']:.4f}"
            )

            # Early stopping check
            if early_stop > 0:
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_optimizer = deepcopy(optimizer)
                    best_model = deepcopy(model)
                    best_epoch = epoch
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= early_stop:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
    # Restore best model if early stopping was used
    if early_stop > 0:
        logger.info(f"Loading best model from epoch {best_epoch}")
        model = best_model
        optimizer = best_optimizer

    # Final evaluation on test set
    test_metrics = evaluate_sparse_encoder(
        base_model, model, test_csr, target_ratio, batch_size, device, seed=seed
    )
    for key, val in test_metrics.items():
        mlflow.log_metric(f"{key}/test", val)
    logger.info(
        f"Test metrics - "
        f"Cosine: {test_metrics['CosineSim']:.4f} - "
        f"L0: {test_metrics['L0']:.1f} - "
        f"NDCG20 Degradation: {test_metrics['NDCG20_Degradation']:.4f}"
    )

    # Export model artifact (config.json + model.pt)
    artifact_dir = "model_artifact"
    os.makedirs(artifact_dir, exist_ok=True)

    config = model.get_config()
    with open(os.path.join(artifact_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    torch.save({"state_dict": model.state_dict()}, os.path.join(artifact_dir, "model.pt"))

    mlflow.log_artifacts(artifact_dir)

    # Cleanup
    for fname in os.listdir(artifact_dir):
        os.remove(os.path.join(artifact_dir, fname))
    os.rmdir(artifact_dir)
    logger.info("Model artifact successfully saved")


class Plugin(BasePlugin):
    """SAE (Sparse Autoencoder) training plugin.

    Trains a sparse autoencoder on top of a pre-trained base model to learn
    interpretable, sparse representations of user embeddings while maintaining
    recommendation quality.

    Expects prior dataset_loading and training_cfm steps in the pipeline context.
    """

    def _load_artifacts(self, context, device):
        """Load dataset and base model artifacts from previous pipeline steps."""
        # Load dataset artifacts
        dataset_run_id = context["dataset_loading"]["run_id"]
        dataset_loader = MLflowRunLoader(dataset_run_id)

        logger.info(f"Loading dataset artifacts from run {dataset_run_id}")

        train_npz = dataset_loader.get_npz_artifact("train_csr.npz")
        valid_npz = dataset_loader.get_npz_artifact("valid_csr.npz")
        test_npz = dataset_loader.get_npz_artifact("test_csr.npz")
        self.train_csr: sp.csr_matrix = (
            sp.csr_matrix(train_npz) if not isinstance(train_npz, sp.csr_matrix) else train_npz
        )
        self.valid_csr: sp.csr_matrix = (
            sp.csr_matrix(valid_npz) if not isinstance(valid_npz, sp.csr_matrix) else valid_npz
        )
        self.test_csr: sp.csr_matrix = (
            sp.csr_matrix(test_npz) if not isinstance(test_npz, sp.csr_matrix) else test_npz
        )

        dataset_params = dataset_loader.get_parameters()
        self.dataset = dataset_params["dataset_name"]
        self.num_users = int(dataset_params["num_users"])
        self.num_items = int(dataset_params["num_items"])
        self.min_user_interactions = int(dataset_params["min_user_interactions"])
        self.min_item_interactions = int(dataset_params["min_item_interactions"])
        self.val_ratio = float(dataset_params["val_ratio"])
        self.test_ratio = float(dataset_params["test_ratio"])

        logger.info(
            f"Training data: {self.train_csr.shape}, Validation data: {self.valid_csr.shape}, Test data: {self.test_csr.shape}"
        )

        # Load base model via registry-based loader
        base_run_id = context["training_cfm"]["run_id"]
        base_loader = MLflowRunLoader(base_run_id)

        base_params = base_loader.get_parameters()
        self.base_model_name = base_params["model"]
        self.base_factors = int(base_params["factors"])
        self.base_min_user_interactions = int(base_params["min_user_interactions"])
        self.base_min_item_interactions = int(base_params["min_item_interactions"])
        self.base_users = int(base_params["users"])
        self.base_items = int(base_params["items"])

        logger.info(f"Base model: {self.base_model_name} with {self.base_factors} factors")

        assert self.num_items == self.base_items, (
            "Number of items in dataset does not match base model"
        )
        assert self.num_users == self.base_users, (
            "Number of users in dataset does not match base model"
        )

        # Load pre-trained base model from artifact (config.json + model.pt)
        self.base_model = load_base_model(base_loader.get_artifact_path(), device)

        logger.info("Base model loaded successfully")

    def run(
        self,
        context: dict,
        epochs: int = 4000,
        early_stop: int = 250,
        batch_size: int = 64,
        embedding_dim: int = 2048,
        top_k: int = 128,
        sample_users: bool = False,
        model: str = "TopKSAE",
        note: str = "",
        target_ratio: float = 0.2,
        normalize: bool = False,
        seed: int = 42,
        reconstruction_loss: str = "Cosine",
        auxiliary_coef: float = 1 / 32,
        contrastive_coef: float = 0.3,
        lr: float = 1e-5,
        beta1: float = 0.9,
        beta2: float = 0.99,
        l1_coef: float = 3e-4,
        evaluate_every: int = 10,
        n_batches_to_dead: int = 5,
        topk_aux: int = 512,
    ):
        """Execute the SAE training pipeline.

        Args:
            context: Pipeline context containing information from previous steps.
                     Must contain context['dataset_loading']['run_id'] and
                     context['training_cfm']['run_id'].
            epochs: Maximum number of training epochs.
            early_stop: Number of epochs without improvement before stopping.
            batch_size: Batch size for training.
            embedding_dim: Dimension of sparse embeddings.
            top_k: Number of active features (for TopKSAE/BatchTopKSAE).
            sample_users: Whether to randomly sample user interactions.
            model: SAE variant ('BasicSAE', 'TopKSAE', or 'BatchTopKSAE').
            note: Optional note for the experiment.
            target_ratio: Fraction of interactions to use as targets for evaluation.
            normalize: Whether to L2-normalize sparse embeddings.
            seed: Random seed for reproducibility.
            reconstruction_loss: Loss function ('L2' or 'Cosine').
            auxiliary_coef: Coefficient for auxiliary loss (dead neuron reactivation).
            contrastive_coef: Coefficient for contrastive loss.
            lr: Learning rate.
            beta1: Adam optimizer beta1 parameter.
            beta2: Adam optimizer beta2 parameter.
            l1_coef: L1 sparsity penalty coefficient.
            evaluate_every: Evaluate every N epochs.
            n_batches_to_dead: Batches before neuron considered dead.
            topk_aux: Top-K for auxiliary loss.

        Returns:
            dict: Updated context with training status.
        """
        # Initialize device and set random seed
        device = set_device()
        logger.info(f"Using device: {device}")

        set_seed(seed)

        self._load_artifacts(context, device)

        expansion_ratio = embedding_dim / self.base_factors
        reconstruction_coef = 1 - (auxiliary_coef + contrastive_coef + l1_coef)

        logger.info(
            f"Expansion ratio: {expansion_ratio:.1f}x ({self.base_factors} → {embedding_dim})"
        )

        # Initialize SAE model via registry
        sae_cls = get_sae_model_class(model)
        sae = sae_cls(
            input_dim=self.base_factors,
            embedding_dim=embedding_dim,
            reconstruction_loss=reconstruction_loss,
            topk_aux=topk_aux,
            n_batches_to_dead=n_batches_to_dead,
            l1_coef=l1_coef,
            k=top_k,
            device=device,
            normalize=normalize,
            auxiliary_coef=auxiliary_coef,
            contrastive_coef=contrastive_coef,
            reconstruction_coef=reconstruction_coef,
        ).to(device)

        logger.info(f"Initialized {model} with {embedding_dim} dimensions")

        # Initialize optimizer
        optimizer = torch.optim.Adam(sae.parameters(), lr=lr, betas=(beta1, beta2))

        # Log all parameters to MLflow
        mlflow.log_params(
            {
                "expansion_ratio": expansion_ratio,
                "min_item_interactions": self.min_item_interactions,
                "target_ratio": target_ratio,
                "beta2": beta2,
                "beta1": beta1,
                "top_k": top_k,
                "seed": seed,
                "l1_coef": l1_coef,
                "min_user_interactions": self.min_user_interactions,
                "epochs": epochs,
                "note": note,
                "reconstruction_loss": reconstruction_loss,
                "val_ratio": self.val_ratio,
                "dataset": self.dataset,
                "batch_size": batch_size,
                "lr": lr,
                "base_factors": self.base_factors,
                "normalize": normalize,
                "base_model": self.base_model_name,
                "base_min_item_interactions": self.base_min_item_interactions,
                "users": self.num_users,
                "topk_aux": topk_aux,
                "items": self.num_items,
                "base_users": self.base_users,
                "early_stop": early_stop,
                "evaluate_every": evaluate_every,
                "reconstruction_coef": reconstruction_coef,
                "embedding_dim": embedding_dim,
                "base_items": self.base_items,
                "auxiliary_coef": auxiliary_coef,
                "contrastive_coef": contrastive_coef,
                "n_batches_to_dead": n_batches_to_dead,
                "test_ratio": self.test_ratio,
                "model": model,
                "base_min_user_interactions": self.base_min_user_interactions,
                "sample_users": sample_users,
            }
        )

        # Train the model
        train(
            model=sae,
            base_model=self.base_model,
            optimizer=optimizer,
            train_csr=self.train_csr,
            valid_csr=self.valid_csr,
            test_csr=self.test_csr,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            early_stop=early_stop,
            evaluate_every=evaluate_every,
            contrastive_coef=contrastive_coef,
            sample_users=sample_users,
            target_ratio=target_ratio,
            seed=seed,
        )
