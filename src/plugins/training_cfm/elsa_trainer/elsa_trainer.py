from copy import deepcopy
from typing import Annotated

import mlflow
import numpy as np
import torch
from tqdm import tqdm

from plugins.plugin_interface import (
    ArtifactSpec,
    BasePlugin,
    OutputArtifactSpec,
    OutputParamSpec,
    ParamGroup,
    ParamSpec,
    PluginIOSpec,
)
from utils.cancellation import CancellationToken
from utils.data_loading.data_loader import DataLoader
from utils.plugin_logger import get_logger
from utils.plugin_notifier import PluginNotifier
from utils.torch.evaluation import evaluate_dense_encoder
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
    notifier: PluginNotifier | None = None,
    cancellation: CancellationToken | None = None,
):
    """Train an ELSA model with early stopping and MLflow tracking.

    This function performs the complete training loop including:
    - Training on batches with progress tracking
    - Validation after each epoch
    - Early stopping based on the lowest validation reconstruction loss
      (ELSA.normalized_mse_loss on the full val interactions, sample-weighted)
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

    Returns:
        None. The function logs metrics to MLflow and saves the model checkpoint.
    """
    # Create data loaders for training and validation
    train_dataloader = DataLoader(train_csr, batch_size, device, shuffle=True, seed=seed)
    valid_dataloader = DataLoader(valid_csr, batch_size, device, shuffle=False)

    # Initialize early stopping variables
    best_epoch = 0
    epochs_without_improvement = 0
    best_loss = float("inf")
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
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            losses = model.train_step(optimizer, batch)
            train_losses.append(losses["Loss"].item())
            pbar.set_postfix({"train_loss": losses["Loss"].cpu().item()})
        # Log average training loss for this epoch
        mlflow.log_metric("loss/train", float(np.mean(train_losses)), step=epoch)

        # Evaluate on validation set. NDCG@20/R@20 are logged but do NOT drive
        # model selection. The deterministic ranking pass uses a fixed seed (no
        # per-epoch reseed) so the input/target split is identical every epoch.
        model.eval()
        valid_metrics = evaluate_dense_encoder(
            model, valid_csr, target_ratio, batch_size, device, seed=seed
        )
        # Validation reconstruction loss: ELSA.normalized_mse_loss on the FULL
        # val interactions (val_csr is both input and reconstruction target),
        # aggregated as a sample-weighted mean (sum(loss * batch_rows) / n_val_users).
        val_loss_sum = 0.0
        for batch in valid_dataloader:
            batch_loss = model.compute_loss_dict(batch)["Loss"].item()
            val_loss_sum += batch_loss * batch.shape[0] / valid_dataloader.dataset_size
        valid_metrics["loss"] = float(val_loss_sum)

        # Log validation metrics to MLflow
        for key, val in valid_metrics.items():
            mlflow.log_metric(f"{key}/valid", val, step=epoch)

        logger.info(
            f"Epoch {epoch}/{epochs} - Loss: {valid_metrics['loss']:.4f} - R@20: {valid_metrics['R20']:.4f} - NDCG20: {valid_metrics['NDCG20']:.4f}"
        )
        if notifier is not None:
            notifier.info(
                f"Epoch {epoch}/{epochs} finished — valid loss: {valid_metrics['loss']:.4f}"
                f" — R@20: {valid_metrics['R20']:.4f}"
                f" — NDCG@20: {valid_metrics['NDCG20']:.4f}"
            )

        # Early stopping logic: select the model with the LOWEST validation
        # reconstruction loss (not NDCG@20).
        if early_stop > 0:
            if valid_metrics["loss"] < best_loss:
                best_loss = valid_metrics["loss"]
                best_optimizer = deepcopy(optimizer)
                best_model = deepcopy(model)
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stop:
                    logger.info(f"Early stopping at epoch {epoch}")
                    if notifier is not None:
                        notifier.warning(
                            f"Early stopping at epoch {epoch} (best was epoch {best_epoch})"
                        )
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
    if notifier is not None:
        notifier.success(
            f"ELSA training done — R@20: {test_metrics['R20']:.4f}"
            f" — NDCG@20: {test_metrics['NDCG20']:.4f}"
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
    description = (
        "Trains the base collaborative-filtering recommender — an ELSA shallow "
        "autoencoder — on the interaction matrix. It learns normalized item embeddings "
        "and dense user embeddings, minimizing reconstruction loss with early "
        "stopping. This embedding space is what the sparse autoencoder later decomposes."
    )

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
        output_params=[
            OutputParamSpec("model", "model_name"),
            OutputParamSpec("dataset", "dataset"),
            OutputParamSpec("users", "num_users"),
            OutputParamSpec("items", "num_items"),
            OutputParamSpec("val_ratio", "val_ratio"),
            OutputParamSpec("test_ratio", "test_ratio"),
            OutputParamSpec("min_user_interactions", "min_user_interactions"),
            OutputParamSpec("min_item_interactions", "min_item_interactions"),
        ],
        param_groups=[
            ParamGroup("Architecture", ["factors"]),
            ParamGroup("Training loop", ["epochs", "batch_size", "early_stop", "seed"]),
            ParamGroup("Optimizer", ["lr", "beta1", "beta2"]),
            ParamGroup("Evaluation", ["target_ratio"]),
        ],
    )

    def run(
        self,
        epochs: Annotated[
            int,
            "Maximum passes over the training set (early stopping may end "
            "training sooner). Higher allows fuller convergence at the cost "
            "of runtime. Default 25.",
        ] = 25,
        batch_size: Annotated[
            int,
            "Users per gradient update. Larger batches are faster per epoch "
            "with smoother gradients but use more memory and may need a "
            "higher learning rate. Default 1024.",
        ] = 512,
        factors: Annotated[
            int,
            "Dimensionality of the latent user/item embedding. Higher adds "
            "model capacity to fit more patterns but risks overfitting and "
            "costs more memory/compute. Default 1024.",
        ] = 1024,
        lr: Annotated[
            float,
            "Adam optimizer step size. Too high diverges or oscillates; too "
            "low trains slowly. Typical range 1e-4 to 1e-3. Default 3e-4.",
        ] = 0.0003,
        early_stop: Annotated[
            int,
            "Stop after this many consecutive epochs without a lower "
            "validation reconstruction loss. 0 disables early stopping (run "
            "all epochs). Default 10.",
        ] = 10,
        seed: Annotated[
            int,
            "Random seed for weight initialization and training. Fix for "
            "reproducible models across runs. Default 42.",
        ] = 42,
        target_ratio: Annotated[
            float,
            "Fraction of each user's interactions hidden as prediction "
            "targets during validation/test ranking scoring; the rest form "
            "the input. E.g. 0.2 predicts 20% from the other 80%. Does not "
            "affect model selection (which uses reconstruction loss on the "
            "full val interactions). Default 0.2.",
        ] = 0.2,
        beta1: Annotated[
            float,
            "Adam first-moment (momentum) decay: how strongly past "
            "gradients smooth each update. Standard value 0.9.",
        ] = 0.9,
        beta2: Annotated[
            float,
            "Adam second-moment decay: controls per-parameter adaptive "
            "step scaling. Standard value 0.99.",
        ] = 0.99,
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
        self.model_name = "ELSA"

        # Initialize device and set random seed for reproducibility
        device = set_device()
        logger.info(f"Using device: {device}")
        self.notifier.info(
            f"ELSA training starting — {epochs} epochs, factors={factors}, device={device}"
        )

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
            notifier=self.notifier,
            cancellation=self.cancellation,
        )
