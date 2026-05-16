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
    ParamSpec,
    PluginIOSpec,
)
from utils.data_loading.data_loader import DataLoader
from utils.plugin_logger import get_logger
from utils.plugin_notifier import PluginNotifier
from utils.torch.evalution import evaluate_sparse_encoder
from utils.torch.models.base_model import BaseModel
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
    notifier: PluginNotifier | None = None,
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
            if notifier is not None:
                notifier.info(
                    f"Epoch {epoch}/{epochs} — loss: {valid_loss:.4f}"
                    f" — cosine: {valid_metrics['CosineSim']:.4f}"
                    f" — NDCG@20 degradation: {valid_metrics['NDCG20_Degradation']:.4f}"
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
                        if notifier is not None:
                            notifier.warning(
                                f"Early stopping at epoch {epoch} (best was epoch {best_epoch})"
                            )
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
    if notifier is not None:
        notifier.success(
            f"SAE training done — cosine: {test_metrics['CosineSim']:.4f}"
            f" — L0: {test_metrics['L0']:.1f}"
            f" — NDCG@20 degradation: {test_metrics['NDCG20_Degradation']:.4f}"
        )

    return model


class Plugin(BasePlugin):
    """SAE (Sparse Autoencoder) training plugin.

    Trains a sparse autoencoder on top of a pre-trained base model to learn
    interpretable, sparse representations of user embeddings while maintaining
    recommendation quality.

    Expects prior dataset_loading and training_cfm steps in the pipeline context.
    """

    name = "SAE Trainer"

    io_spec = PluginIOSpec(
        required_steps=["dataset_loading", "training_cfm"],
        input_artifacts=[
            ArtifactSpec("dataset_loading", "train_csr.npz", "train_csr", "npz"),
            ArtifactSpec("dataset_loading", "valid_csr.npz", "valid_csr", "npz"),
            ArtifactSpec("dataset_loading", "test_csr.npz", "test_csr", "npz"),
            ArtifactSpec("training_cfm", "", "base_model", "base_model"),
        ],
        input_params=[
            ParamSpec("dataset_loading", "num_users", "num_users", int),
            ParamSpec("dataset_loading", "num_items", "num_items", int),
            ParamSpec(
                "dataset_loading",
                "min_user_interactions",
                "min_user_interactions",
                int,
            ),
            ParamSpec(
                "dataset_loading",
                "min_item_interactions",
                "min_item_interactions",
                int,
            ),
            ParamSpec("dataset_loading", "dataset_name", "dataset", str),
            ParamSpec("dataset_loading", "val_ratio", "val_ratio", float),
            ParamSpec("dataset_loading", "test_ratio", "test_ratio", float),
            ParamSpec("training_cfm", "model", "base_model_name", str),
            ParamSpec("training_cfm", "factors", "base_factors", int),
            ParamSpec(
                "training_cfm",
                "min_user_interactions",
                "base_min_user_interactions",
                int,
            ),
            ParamSpec(
                "training_cfm",
                "min_item_interactions",
                "base_min_item_interactions",
                int,
            ),
            ParamSpec("training_cfm", "users", "base_users", int),
            ParamSpec("training_cfm", "items", "base_items", int),
        ],
        output_artifacts=[
            OutputArtifactSpec("trained_model", "", "model"),
        ],
    )

    def load_context(self, context: dict) -> None:
        """Load upstream artifacts and validate cross-step consistency.

        Calls the base ``load_context()`` to hydrate all declared
        inputs, then asserts that the dataset dimensions match the
        base model dimensions.

        Args:
            context: Pipeline context dict.

        Raises:
            AssertionError: If dataset and base model user/item
                counts do not match.
        """
        super().load_context(context)

        assert self.num_items == self.base_items, (
            "Number of items in dataset does not match base model"
        )
        assert self.num_users == self.base_users, (
            "Number of users in dataset does not match base model"
        )
        logger.info(f"Base model: {self.base_model_name} with {self.base_factors} factors")

    def run(
        self,
        epochs: Annotated[
            int,
            "Maximum SAE training epochs (early stopping may end training "
            "sooner). Higher allows fuller convergence at the cost of "
            "runtime.",
        ] = 4000,
        early_stop: Annotated[
            int,
            "Stop after this many consecutive evaluations without "
            "validation-loss improvement. 0 disables early stopping. "
            "Counted in evaluation steps, not raw epochs.",
        ] = 250,
        batch_size: Annotated[
            int,
            "User embeddings per gradient update. Larger batches are faster "
            "per epoch and give more stable sparsity statistics but use "
            "more memory.",
        ] = 64,
        embedding_dim: Annotated[
            int,
            "Width of the SAE's sparse hidden layer (the dictionary size). "
            "Larger yields more, finer-grained interpretable features but "
            "is slower; usually a multiple of the base model's factor count "
            "(the expansion ratio).",
        ] = 2048,
        top_k: Annotated[
            int,
            "Active features allowed per input for TopKSAE/BatchTopKSAE "
            "(the sparsity level). Lower is sparser and more interpretable "
            "but reconstructs worse; unused by BasicSAE, which relies on "
            "l1_coef instead.",
        ] = 128,
        sample_users: Annotated[
            bool,
            "If true, randomly mask part of each user's interactions every "
            "epoch as data augmentation. Improves robustness but slows "
            "training (forces on-the-fly encoding).",
        ] = False,
        model: Annotated[
            str,
            "Sparse-autoencoder variant. 'TopKSAE'/'BatchTopKSAE' enforce a "
            "hard active-feature budget (top_k); 'BasicSAE' uses a soft L1 "
            "penalty (l1_coef) instead.",
        ] = "TopKSAE",
        note: Annotated[
            str,
            "Free-text note logged to MLflow for your own bookkeeping. Has no effect on training.",
        ] = "",
        target_ratio: Annotated[
            float,
            "Fraction of each user's interactions hidden as targets when "
            "scoring recommendation quality in evaluation. E.g. 0.2 "
            "predicts 20% from the other 80%.",
        ] = 0.2,
        normalize: Annotated[
            bool,
            "If true, L2-normalize the sparse embeddings: can stabilize "
            "training and make feature magnitudes comparable, but discards "
            "activation-scale information.",
        ] = False,
        seed: Annotated[
            int,
            "Random seed for weight init and data ordering. Fix for "
            "reproducible SAE features across runs.",
        ] = 42,
        reconstruction_loss: Annotated[
            str,
            "How reconstruction error against the base embedding is "
            "measured: 'Cosine' (direction only, scale-invariant) or 'L2' "
            "(squared Euclidean, scale-sensitive).",
        ] = "Cosine",
        auxiliary_coef: Annotated[
            float,
            "Weight of the auxiliary loss that revives dead neurons by "
            "making them reconstruct the residual. Higher reduces dead "
            "features but can disturb the main reconstruction.",
        ] = 1 / 32,
        contrastive_coef: Annotated[
            float,
            "Weight of the contrastive loss pulling together augmented "
            "views of the same user. >0 enables it (and on-the-fly "
            "encoding); higher favors view-invariant features.",
        ] = 0.3,
        lr: Annotated[
            float,
            "Adam step size for the SAE. SAEs are sensitive: too high "
            "causes feature collapse, too low stalls learning. Typical "
            "range ~1e-5 to 1e-4.",
        ] = 1e-5,
        beta1: Annotated[
            float,
            "Adam first-moment (momentum) decay. Standard value 0.9.",
        ] = 0.9,
        beta2: Annotated[
            float,
            "Adam second-moment decay (per-parameter adaptive step scaling). Standard value 0.99.",
        ] = 0.99,
        l1_coef: Annotated[
            float,
            "Strength of the L1 sparsity penalty on activations. Higher is "
            "sparser and more interpretable but reconstructs worse. Primary "
            "sparsity control for BasicSAE.",
        ] = 3e-4,
        evaluate_every: Annotated[
            int,
            "Run validation (losses + recommendation metrics) every N "
            "epochs. Smaller gives finer monitoring and earlier early-stop "
            "decisions but more overhead.",
        ] = 10,
        n_batches_to_dead: Annotated[
            int,
            "A feature is flagged dead after this many consecutive batches "
            "with zero activation; dead features are what the auxiliary "
            "loss tries to revive.",
        ] = 5,
        topk_aux: Annotated[
            int,
            "How many dead features the auxiliary loss reactivates per step "
            "(its top-k budget). Higher revives more aggressively but adds "
            "reconstruction noise.",
        ] = 512,
    ):
        """Execute the SAE training pipeline.

        Args:
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
        self.notifier.info(
            f"SAE training starting — {model} {embedding_dim}d, {epochs} epochs"
            f", top_k={top_k}, device={device}"
        )

        set_seed(seed)

        self.base_model.to(device)

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
        self.trained_model = train(
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
            notifier=self.notifier,
        )
