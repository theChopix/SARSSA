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
    StaticDropdownHint,
    ToggleHint,
)
from utils.cancellation import CancellationToken
from utils.data_loading.data_loader import DataLoader
from utils.plugin_logger import get_logger
from utils.plugin_notifier import PluginNotifier
from utils.torch.evaluation import evaluate_sparse_encoder
from utils.torch.models.base_model import BaseModel
from utils.torch.models.sae_model import SAE
from utils.torch.models.sae_model.topk_sae import TopKSAE
from utils.torch.runtime import set_device, set_seed

logger = get_logger(__name__)

# contrastive positive view - interaction-dropout keep probability (train + valid)
_POSITIVE_KEEP_PROB = 0.8

# anchor augmentation (sample_interactions) - interaction-dropout keep probability
_ANCHOR_KEEP_PROB = 0.8


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
    sample_interactions: bool,
    target_ratio: float,
    seed: int,
    notifier: PluginNotifier | None = None,
    cancellation: CancellationToken | None = None,
):
    """Train a Sparse Autoencoder (SAE) model with early stopping and MLflow tracking.

    The training process learns sparse representations of dense base model embeddings
    while maintaining recommendation quality. Supports multiple training modes (from
    interactions or pre-computed embeddings) and advanced techniques like contrastive
    learning and auxiliary loss for dead neuron reactivation.

    Args:
        model: SAE model to train (BasicSAE, TopKSAE, or BatchTopKSAE).
        base_model: Pre-trained base model (frozen) used to encode interactions.
        optimizer: Optimizer for training (typically Adam).
        train_csr: Training data as sparse CSR matrix (users x items).
        valid_csr: Validation data as sparse CSR matrix.
        test_csr: Test data as sparse CSR matrix.
        device: Device to train on (cuda/mps/cpu).
        epochs: Maximum number of training epochs.
        batch_size: Batch size for training.
        early_stop: Number of evaluations without improvement before stopping (0 disables).
        evaluate_every: Evaluate every N epochs.
        contrastive_coef: Coefficient for contrastive loss.
        sample_interactions: Whether to randomly drop part of each user's interactions
            each epoch (input-dropout augmentation).
        target_ratio: Fraction of interactions to use as targets for evaluation.
        seed: Random seed for reproducibility.

    Returns:
        The trained (best, if early stopping is enabled) SAE model.
    """

    def prepare():
        """Build interaction loaders and precompute the base embeddings.

        Only precomputes what the current config consumes, storing each loader
        into a dict as it is built. Keys for skipped precomputes are simply
        absent (their consumers are guarded by the same condition):

        - ``train_interaction_dataloader`` — always;
        - ``train_embeddings_dataloader`` — only on the no-augmentation fast path;
        - ``valid_embeddings_dataloader`` — always;
        - ``val_positive_embeddings_dataloader`` — only when contrastive is enabled.
        """
        loaders = {}

        train_interaction_dataloader = DataLoader(
            train_csr, batch_size, device, shuffle=True, seed=seed
        )
        loaders["train_interaction_dataloader"] = train_interaction_dataloader

        # Train embeddings feed ONLY the no-augmentation fast path, so skip this
        # expensive full-train encode when augmentation/contrastive is enabled.
        if not (sample_interactions or contrastive_coef > 0):
            train_user_embeddings = np.vstack(
                [
                    base_model.encode(batch).detach().cpu().numpy()
                    for batch in tqdm(
                        train_interaction_dataloader, desc="Computing training embeddings"
                    )
                ]
            )
            loaders["train_embeddings_dataloader"] = DataLoader(
                train_user_embeddings, batch_size, device, shuffle=True, seed=seed
            )

        valid_interaction_dataloader = DataLoader(valid_csr, batch_size, device, shuffle=False)
        val_user_embeddings = np.vstack(
            [
                base_model.encode(batch).detach().cpu().numpy()
                for batch in tqdm(
                    valid_interaction_dataloader, desc="Computing validation embeddings"
                )
            ]
        )
        loaders["valid_embeddings_dataloader"] = DataLoader(
            val_user_embeddings, batch_size, device, shuffle=False
        )

        # Augmented validation view feeds ONLY the contrastive term, so skip it
        #  unless contrastive is enabled.
        if contrastive_coef > 0:
            val_positive_user_embeddings = np.vstack(
                [
                    base_model.encode(batch * (torch.rand_like(batch) < _POSITIVE_KEEP_PROB))
                    .detach()
                    .cpu()
                    .numpy()
                    for batch in tqdm(
                        valid_interaction_dataloader,
                        desc="Computing augmented validation embeddings",
                    )
                ]
            )
            loaders["val_positive_embeddings_dataloader"] = DataLoader(
                val_positive_user_embeddings, batch_size, device, shuffle=False
            )

        return loaders

    def train_epoch(epoch, d):
        """Run one training epoch; return the per-batch training losses (lists).

        Two strategies share one per-batch step:

        - augmentation/contrastive off → iterate the precomputed embeddings;
        - otherwise → iterate interactions, building the (optional) contrastive
          positive and the (optionally augmented) anchor on the fly.
        """
        model.train()
        train_losses = {"Loss": [], "L2": [], "L1": [], "L0": [], "Cosine": []}
        are_interactions_needed = sample_interactions or contrastive_coef > 0

        def step(anchor, positive, pbar):
            """Shared per-batch update + loss accumulation."""
            losses = model.train_step(optimizer, anchor, positive)
            pbar.set_postfix({"train_loss": losses["Loss"].cpu().item()})
            for key in train_losses:
                train_losses[key].append(losses[key].item())

        if are_interactions_needed:
            pbar = tqdm(d["train_interaction_dataloader"], desc=f"Epoch {epoch}/{epochs}")
            for batched_interactions in pbar:
                if cancellation is not None:
                    cancellation.raise_if_cancelled()

                # Contrastive positive view (freshly sampled each epoch):
                # keep ~80% of interactions, then encode.
                if contrastive_coef > 0:
                    positive_batch = batched_interactions * (
                        torch.rand_like(batched_interactions) < _POSITIVE_KEEP_PROB
                    )
                    positive_batch = base_model.encode(positive_batch).detach()
                else:
                    positive_batch = None

                # Optional anchor augmentation: keep ~80% of interactions.
                if sample_interactions:
                    batched_interactions = batched_interactions * (
                        torch.rand_like(batched_interactions) < _ANCHOR_KEEP_PROB
                    )

                anchor_batch = base_model.encode(batched_interactions).detach()
                step(anchor_batch, positive_batch, pbar)
        else:
            pbar = tqdm(d["train_embeddings_dataloader"], desc=f"Epoch {epoch}/{epochs}")
            for batched_embeddings in pbar:
                if cancellation is not None:
                    cancellation.raise_if_cancelled()
                step(batched_embeddings, None, pbar)

        return train_losses

    def evaluate(epoch, d):
        """Run one validation pass; log losses + metrics and return (loss, metrics).

        Validation losses use a SAMPLE-WEIGHTED mean over batches
        (sum(loss*batch_rows)/n_val_users) so early stopping sees the true
        per-user mean regardless of an uneven final batch.
        """
        model.eval()
        valid_losses = {
            "Loss": [],
            "L2": [],
            "L1": [],
            "L0": [],
            "Cosine": [],
            "Auxiliary": [],
            "Contrastive": [],
        }
        # The contrastive positive view is only built when contrastive is enabled;
        # otherwise pass None (compute_loss_dict then reports Contrastive as 0).
        if contrastive_coef > 0:
            pairs = zip(d["valid_embeddings_dataloader"], d["val_positive_embeddings_dataloader"])
        else:
            pairs = ((embedding, None) for embedding in d["valid_embeddings_dataloader"])

        valid_batch_rows = []
        for embedding, positive_embedding in pairs:
            losses = model.compute_loss_dict(embedding, positive_embedding)
            valid_batch_rows.append(embedding.shape[0])
            for key, val in losses.items():
                valid_losses[key].append(val.item())

        valid_batch_rows = np.asarray(valid_batch_rows, dtype=np.float64)
        n_val_users = valid_batch_rows.sum()

        def _weighted_mean(values):
            return float(np.dot(values, valid_batch_rows) / n_val_users)

        for key, val in valid_losses.items():
            mlflow.log_metric(f"loss/{key}/valid", _weighted_mean(val), step=epoch)

        valid_metrics = evaluate_sparse_encoder(
            base_model, model, valid_csr, target_ratio, batch_size, device, seed=seed
        )
        for key, val in valid_metrics.items():
            mlflow.log_metric(f"{key}/valid", val, step=epoch)

        return _weighted_mean(valid_losses["Loss"]), valid_metrics

    # ── orchestration ────────────────────────────────────────────────────────

    d = prepare()

    # Initialize early stopping
    if early_stop > 0:
        best_epoch = 0
        epochs_without_improvement = 0
        best_loss = np.inf
        best_optimizer = deepcopy(optimizer)
        best_model = deepcopy(model)

    for epoch in range(1, epochs + 1):
        train_losses = train_epoch(epoch, d)
        for key, val in train_losses.items():
            mlflow.log_metric(f"loss/{key}/train", float(np.mean(val)), step=epoch)

        if epoch % evaluate_every == 0:
            valid_loss, valid_metrics = evaluate(epoch, d)
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
    """Top-K SAE training plugin.

    Trains a Top-K sparse autoencoder on top of a pre-trained base model. A hard
    budget of ``top_k`` active features per user enforces sparsity, learning
    interpretable, sparse representations of user embeddings while maintaining
    recommendation quality.

    Expects prior dataset_loading and training_cfm steps in the pipeline context.
    """

    name = "TopK SAE Trainer"
    description = (
        "Trains a Top-K sparse autoencoder on the recommender's dense embeddings, "
        "re-encoding each into a wider but sparse vector where exactly top_k 'neurons' "
        "fire. Each neuron ideally captures one interpretable concept while reconstruction "
        "preserves recommendation quality."
    )

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
        output_params=[
            OutputParamSpec("model", "model_name"),
            OutputParamSpec("dataset", "dataset"),
            OutputParamSpec("users", "num_users"),
            OutputParamSpec("items", "num_items"),
            OutputParamSpec("val_ratio", "val_ratio"),
            OutputParamSpec("test_ratio", "test_ratio"),
            OutputParamSpec("min_user_interactions", "min_user_interactions"),
            OutputParamSpec("min_item_interactions", "min_item_interactions"),
            OutputParamSpec("expansion_ratio", "expansion_ratio"),
            OutputParamSpec("reconstruction_coef", "reconstruction_coef"),
            OutputParamSpec("base_model", "base_model_name"),
            OutputParamSpec("base_factors", "base_factors"),
            OutputParamSpec("base_users", "base_users"),
            OutputParamSpec("base_items", "base_items"),
            OutputParamSpec("base_min_user_interactions", "base_min_user_interactions"),
            OutputParamSpec("base_min_item_interactions", "base_min_item_interactions"),
        ],
        param_ui_hints=[
            StaticDropdownHint("reconstruction_loss", choices=["Cosine", "L2"]),
            ToggleHint("sample_interactions"),
            ToggleHint("normalize"),
        ],
        param_groups=[
            ParamGroup("Architecture", ["embedding_dim", "top_k", "normalize"]),
            ParamGroup(
                "Training Loop",
                ["epochs", "batch_size", "early_stop", "sample_interactions", "seed"],
                subgroups=[
                    ParamGroup(
                        "Loss",
                        ["reconstruction_loss", "l1_coef"],
                        subgroups=[
                            ParamGroup(
                                "Dead Neurons Auxiliary",
                                ["auxiliary_coef", "topk_aux", "n_batches_to_dead"],
                            ),
                            ParamGroup("Contrastive", ["contrastive_coef", "temperature"]),
                        ],
                    ),
                ],
            ),
            ParamGroup("Optimizer", ["lr", "beta1", "beta2"]),
            ParamGroup("Evaluation", ["evaluate_every", "target_ratio"]),
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
            "runtime. Default 250.",
        ] = 250,
        early_stop: Annotated[
            int,
            "Stop after this many consecutive evaluations without "
            "validation-loss improvement. 0 disables early stopping. "
            "Counted in evaluation steps, not raw epochs. Default 50.",
        ] = 50,
        batch_size: Annotated[
            int,
            "User embeddings per gradient update. Larger batches are faster "
            "per epoch and give more stable sparsity statistics but use "
            "more memory. Default 1024.",
        ] = 512,
        embedding_dim: Annotated[
            int,
            "Width of the SAE's sparse hidden layer (the dictionary size). "
            "Larger yields more, finer-grained interpretable features but "
            "is slower; usually a multiple of the base model's factor count "
            "(the expansion ratio). Default 8192 (= 8 x ELSA factors 1024).",
        ] = 8192,
        top_k: Annotated[
            int,
            "Active features allowed per input (the sparsity level). Exactly "
            "top_k features stay active per user. Lower is sparser and more "
            "interpretable but reconstructs worse. Default 32.",
        ] = 32,
        sample_interactions: Annotated[
            bool,
            "If true, randomly mask part of each user's interactions every "
            "epoch as data augmentation. Improves robustness but slows "
            "training (forces on-the-fly encoding).",
        ] = False,
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
            "features but can disturb the main reconstruction. Disabled by "
            "default (0.0).",
        ] = 0.0,
        contrastive_coef: Annotated[
            float,
            "Weight of the contrastive loss pulling together augmented "
            "views of the same user. >0 enables it (and on-the-fly "
            "encoding); higher favors view-invariant features. Disabled by "
            "default (0.0).",
        ] = 0.0,
        temperature: Annotated[
            float,
            "Softmax temperature of the contrastive loss. Sparse codes are "
            "non-negative, so their cosine similarities are squeezed into "
            "[0, 1]; dividing by a low temperature restores the contrast "
            "between the true pair and the in-batch negatives. Typical "
            "range 0.07-0.2. Default 0.2.",
        ] = 0.2,
        lr: Annotated[
            float,
            "Adam step size for the SAE. SAEs are sensitive: too high "
            "causes feature collapse, too low stalls learning. Default "
            "3e-4.",
        ] = 3e-4,
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
            "Strength of the L1 sparsity penalty on activations. Secondary "
            "to top_k here (shrinks the surviving activations); set 0 to "
            "rely on top_k alone. Default 3e-4.",
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
        """Execute the Top-K SAE training pipeline.

        Args:
            epochs: Maximum number of training epochs.
            early_stop: Number of epochs without improvement before stopping.
            batch_size: Batch size for training.
            embedding_dim: Dimension of sparse embeddings.
            top_k: Number of active features per input.
            sample_interactions: Whether to randomly drop part of each user's interactions
                (input-dropout augmentation).
            target_ratio: Fraction of interactions to use as targets for evaluation.
            normalize: Whether to L2-normalize sparse embeddings.
            seed: Random seed for reproducibility.
            reconstruction_loss: Loss function ('L2' or 'Cosine').
            auxiliary_coef: Coefficient for auxiliary loss (dead neuron reactivation).
            contrastive_coef: Coefficient for contrastive loss.
            temperature: Softmax temperature for the contrastive loss.
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
            f"Top-K SAE training starting — {embedding_dim}d, {epochs} epochs"
            f", top_k={top_k}, device={device}"
        )

        set_seed(seed)

        self.base_model.to(device)

        self.expansion_ratio = embedding_dim / self.base_factors
        self.model_name = "TopKSAE"
        # Total loss = reconstruction_coef*recon + l1_coef*L1 (+ aux*Aux + con*Con).
        # The L1 penalty is added on TOP of the reconstruction term, so it must not be
        # subtracted here. With aux=0 and con=0 reconstruction_coef == 1.0.
        if auxiliary_coef + contrastive_coef >= 1:
            raise ValueError(
                "auxiliary_coef + contrastive_coef must be < 1 "
                f"(got {auxiliary_coef + contrastive_coef})"
            )
        reconstruction_coef = 1 - (auxiliary_coef + contrastive_coef)
        self.reconstruction_coef = reconstruction_coef

        logger.info(
            f"Expansion ratio: {self.expansion_ratio:.1f}x ({self.base_factors} → {embedding_dim})"
        )

        # Initialize the Top-K SAE model
        sae = TopKSAE(
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
            temperature=temperature,
            reconstruction_coef=reconstruction_coef,
        ).to(device)

        logger.info(f"Initialized TopKSAE with {embedding_dim} dimensions")

        # Initialize optimizer
        optimizer = torch.optim.Adam(sae.parameters(), lr=lr, betas=(beta1, beta2))

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
            sample_interactions=sample_interactions,
            target_ratio=target_ratio,
            seed=seed,
            notifier=self.notifier,
            cancellation=self.cancellation,
        )
