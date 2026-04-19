from copy import deepcopy

import mlflow
import numpy as np
import torch
from tqdm import tqdm

from plugins.plugin_interface import (
    ArtifactSpec,
    BasePlugin,
    OutputArtifactSpec,
    ParamSpec,
    PluginIOSpec,
)
from utils.data_loading.data_loader import DataLoader
from utils.plugin_logger import get_logger
from utils.torch.evalution import evaluate_dense_encoder
from utils.torch.models.base_model.elsa import ELSA
from utils.torch.runtime import set_device, set_seed

logger = get_logger(__name__)


def train(
    model: ELSA,
    optimizer,
    train_csr,
    valid_csr,
    test_csr,
    device,
    epochs: int,
    batch_size: int,
    early_stop: int,
    target_ratio: float,
    seed: int,
    dataset: str,
    factors: int,
    lr: float,
    val_ratio: float,
    test_ratio: float,
    beta1: float,
    beta2: float,
    min_user_interactions: int,
    min_item_interactions: int,
    num_users: int,
    num_items: int,
):
    """Train an ELSA model with early stopping and MLflow tracking.

    This function performs the complete training loop including:
    - Training on batches with progress tracking
    - Validation after each epoch
    - Early stopping based on NDCG@20
    - Test evaluation on the best model
    - Model checkpointing and MLflow artifact logging

    Args:
        model: ELSA model instance to train.
        optimizer: PyTorch optimizer (typically Adam).
        train_csr: Training data as sparse CSR matrix (users x items).
        valid_csr: Validation data as sparse CSR matrix (users x items).
        test_csr: Test data as sparse CSR matrix (users x items).
        device: PyTorch device (cuda, mps, or cpu).
        epochs: Maximum number of training epochs.
        batch_size: Number of samples per batch.
        early_stop: Number of epochs without improvement before stopping (0 to disable).
        target_ratio: Ratio of interactions to use as targets in evaluation (0.0-1.0).
        seed: Random seed for reproducibility.
        dataset: Name of the dataset being used.
        factors: Number of latent factors (embedding dimension).
        lr: Learning rate for optimizer.
        val_ratio: Validation set ratio used in data splitting.
        test_ratio: Test set ratio used in data splitting.
        beta1: Adam optimizer beta1 parameter.
        beta2: Adam optimizer beta2 parameter.
        min_user_interactions: Minimum interactions per user (dataset filtering threshold).
        min_item_interactions: Minimum interactions per item (dataset filtering threshold).
        num_users: Total number of users in the dataset.
        num_items: Total number of items in the dataset.

    Returns:
        None. The function logs metrics to MLflow and saves the model checkpoint.
    """
    # Log all hyperparameters and dataset info to MLflow
    mlflow.log_params(
        {
            "dataset": dataset,
            "epochs": epochs,
            "batch_size": batch_size,
            "factors": factors,
            "lr": lr,
            "early_stop": early_stop,
            "seed": seed,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "target_ratio": target_ratio,
            "beta1": beta1,
            "beta2": beta2,
            "model": "ELSA",
            "min_user_interactions": min_user_interactions,
            "min_item_interactions": min_item_interactions,
            "users": num_users,
            "items": num_items,
        }
    )

    # Create data loaders for training and validation
    train_dataloader = DataLoader(train_csr, batch_size, device, shuffle=True)
    valid_dataloader = DataLoader(valid_csr, batch_size, device, shuffle=False)

    # Initialize early stopping variables
    best_epoch = 0
    epochs_without_improvement = 0
    best_ndcg = 0
    best_optimizer = None
    best_model = None

    # Save initial state if early stopping is enabled
    if early_stop > 0:
        best_optimizer = deepcopy(optimizer)
        best_model = deepcopy(model)

    # Main training loop
    for epoch in range(1, epochs + 1):
        train_losses = []
        model.train()

        # Train on all batches with progress bar
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            losses = model.train_step(optimizer, batch)
            train_losses.append(losses["Loss"].item())
            pbar.set_postfix({"train_loss": losses["Loss"].cpu().item()})
        # Log average training loss for this epoch
        mlflow.log_metric("loss/train", float(np.mean(train_losses)), step=epoch)

        # Evaluate on validation set
        model.eval()
        valid_metrics = evaluate_dense_encoder(
            model, valid_csr, target_ratio, batch_size, device, seed=seed + epoch
        )
        valid_metrics["loss"] = float(
            np.mean([model.compute_loss_dict(batch)["Loss"].item() for batch in valid_dataloader])
        )

        # Log validation metrics to MLflow
        for key, val in valid_metrics.items():
            mlflow.log_metric(f"{key}/valid", val, step=epoch)

        logger.info(
            f"Epoch {epoch}/{epochs} - Loss: {valid_metrics['loss']:.4f} - R@20: {valid_metrics['R20']:.4f} - NDCG20: {valid_metrics['NDCG20']:.4f}"
        )

        # Early stopping logic
        if early_stop > 0:
            if valid_metrics["NDCG20"] > best_ndcg:
                best_ndcg = valid_metrics["NDCG20"]
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
    if early_stop > 0 and best_model is not None:
        logger.info(f"Loading best model from epoch {best_epoch}")
        model = best_model
        if best_optimizer is not None:
            optimizer = best_optimizer

    # Final evaluation on test set
    test_metrics = evaluate_dense_encoder(
        model, test_csr, target_ratio, batch_size, device, seed=seed
    )
    for key, val in test_metrics.items():
        mlflow.log_metric(f"{key}/test", val)
    logger.info(
        f"Test metrics - R@20: {test_metrics['R20']:.4f} - NDCG20: {test_metrics['NDCG20']:.4f}"
    )

    return model


class Plugin(BasePlugin):
    """ELSA (Scalable Linear Shallow Autoencoder) training plugin.

    This plugin implements the training pipeline for collaborative filtering using
    the ELSA model. It handles model initialization, training with
    early stopping, and MLflow experiment tracking.

    Expects a prior dataset_loading step in the pipeline context.
    """

    name = "ELSA Trainer"

    io_spec = PluginIOSpec(
        required_steps=["dataset_loading"],
        input_artifacts=[
            ArtifactSpec("dataset_loading", "train_csr.npz", "train_csr", "npz"),
            ArtifactSpec("dataset_loading", "valid_csr.npz", "valid_csr", "npz"),
            ArtifactSpec("dataset_loading", "test_csr.npz", "test_csr", "npz"),
        ],
        input_params=[
            ParamSpec("dataset_loading", "num_users", "num_users", int),
            ParamSpec("dataset_loading", "num_items", "num_items", int),
            ParamSpec("dataset_loading", "min_user_interactions", "min_user_interactions", int),
            ParamSpec("dataset_loading", "min_item_interactions", "min_item_interactions", int),
            ParamSpec("dataset_loading", "dataset_name", "dataset", str),
            ParamSpec("dataset_loading", "val_ratio", "val_ratio", float),
            ParamSpec("dataset_loading", "test_ratio", "test_ratio", float),
        ],
        output_artifacts=[
            OutputArtifactSpec("trained_model", "", "model"),
        ],
    )

    def run(
        self,
        epochs: int = 100,
        batch_size: int = 64,
        factors: int = 256,
        lr: float = 0.0001,
        early_stop: int = 10,
        seed: int = 43,
        target_ratio: float = 0.2,
        beta1: float = 0.9,
        beta2: float = 0.99,
    ):
        """Execute the ELSA training pipeline.

        Args:
            epochs: Maximum number of training epochs.
            batch_size: Number of samples per batch.
            factors: Number of latent factors (embedding dimension).
            lr: Learning rate for Adam optimizer.
            early_stop: Epochs without improvement before stopping (0 to disable).
            seed: Random seed for reproducibility.
            target_ratio: Ratio of interactions used as targets in evaluation.
            beta1: Adam optimizer beta1 parameter (momentum).
            beta2: Adam optimizer beta2 parameter (RMSprop).

        Returns:
            dict: Updated context with training status.
        """
        # Initialize device and set random seed for reproducibility
        device = set_device()
        logger.info(f"Using device: {device}")

        set_seed(seed)

        # Initialize ELSA model and optimizer
        model = ELSA(input_dim=self.num_items, embedding_dim=factors).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))

        self.trained_model = train(
            model=model,
            optimizer=optimizer,
            train_csr=self.train_csr,
            valid_csr=self.valid_csr,
            test_csr=self.test_csr,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            early_stop=early_stop,
            target_ratio=target_ratio,
            seed=seed,
            dataset=self.dataset,
            factors=factors,
            lr=lr,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            beta1=beta1,
            beta2=beta2,
            min_user_interactions=self.min_user_interactions,
            min_item_interactions=self.min_item_interactions,
            num_users=self.num_users,
            num_items=self.num_items,
        )
