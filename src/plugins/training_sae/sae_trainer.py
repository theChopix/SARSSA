import os
import argparse
import datetime
import mlflow
import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from utils.datasets.lastFm1k_loader import LastFm1kLoader
from utils.datasets.movieLens_loader import MovieLensLoader
from utils.datasets.data_loader import DataLoader
from utils.torch.models.elsa import ELSA
from utils.torch.models.sae import SAE, BasicSAE, TopKSAE, BatchTopKSAE
from utils.plugin_logger import get_logger
from utils.torch.runtime import set_device, set_seed
from utils.torch.checkpointing import save_checkpoint, load_checkpoint
from utils.torch.evalution import evaluate_sparse_encoder

from plugins.plugin_interface import BasePlugin


logger = get_logger(__name__)


def parse_arguments():
    """Parse command-line arguments for SAE training.
    
    Returns:
        argparse.Namespace: Parsed arguments containing all hyperparameters.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='LastFM1k', help='Dataset to use. For now, only "LastFM1k" and "MovieLens" are supported')
    parser.add_argument('--epochs', type=int, default=4_000, help='Number of epochs to train the model')
    parser.add_argument('--early_stop', type=int, default=250, help='Number of epochs to wait for improvement before stopping')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--embedding_dim', type=int, default=2048, help='Number of factors for the model')
    parser.add_argument('--top_k', type=int, default=128, help='Top k parameter for TopKSAE')
    parser.add_argument("--base_run_id", type=str, default='4a43996d7eec489183ad0d6b0c00d935', help="Run ID of the base model")
    parser.add_argument("--sample_users", action='store_true', default=False, help="Choose randomly 0.5 - 1.0 of the users interactions")
    parser.add_argument('--model', type=str, default='TopKSAE', help='Model to use (BasicSAE, TopKSAE, BatchTopKSAE)')
    parser.add_argument('--note', type=str, default='', help='Note for the experiment')
    parser.add_argument('--target_ratio', type=float, default=0.2, help='Ratio of target interactions')
    parser.add_argument("--normalize", action='store_true', help="Normalize the sparse embedding (BasicSAE, TopKSAE)")
    # stable parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test ratio')
    # training parameters
    parser.add_argument("--reconstruction_loss", type=str, default="Cosine", help="Reconstruction loss (L2 or Cosine)")
    parser.add_argument("--auxiliary_coef", type=float, default=1/32, help="Auxiliary loss coefficient (BasicSAE, TopKSAE)")
    parser.add_argument("--contrastive_coef", type=float, default=0.3, help="Contrastive loss coefficient (BasicSAE, TopKSAE)")
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for training')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.99, help='Beta2 for Adam optimizer')
    parser.add_argument("--l1_coef", type=float, default=3e-4, help="L1 loss coefficient (BasicSAE, TopKSAE)")
    parser.add_argument('--evaluate_every', type=int, default=10, help='Evaluate every n epochs')
    # auxiliary parameters
    parser.add_argument("--n_batches_to_dead", type=int, default=5, help="Number of batches to wait before optimizing the dead neurons (BasicSAE, TopKSAE)")
    parser.add_argument("--topk_aux", type=int, default=512, help="Top k for auxiliary loss (BasicSAE, TopKSAE)")
    return parser.parse_args()


def train(
    model: SAE,
    base_model: ELSA,
    optimizer,
    train_csr,
    valid_csr,
    test_csr,
    device,
    dataset: str,
    epochs: int,
    batch_size: int,
    early_stop: int,
    evaluate_every: int,
    model_name: str,
    embedding_dim: int,
    top_k: int,
    contrastive_coef: float,
    sample_users: bool,
    target_ratio: float,
    seed: int,
):
    """Train a Sparse Autoencoder (SAE) model with early stopping and MLflow tracking.
    
    The training process learns sparse representations of dense ELSA embeddings while
    maintaining recommendation quality. Supports multiple training modes (from interactions
    or pre-computed embeddings) and advanced techniques like contrastive learning and
    auxiliary loss for dead neuron reactivation.
    
    Args:
        model: SAE model to train (BasicSAE, TopKSAE, or BatchTopKSAE).
        base_model: Pre-trained ELSA model (frozen) used to encode interactions.
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
    # Generate run name based on model configuration
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if model_name in ['TopKSAE', 'BatchTopKSAE']:
        run_name = f'{model_name}_{embedding_dim}_{top_k}_{timestamp}'
    else:
        run_name = f'{model_name}_{embedding_dim}_{timestamp}'

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
    train_embeddings_dataloader = DataLoader(train_user_embeddings, batch_size, device, shuffle=True)
    
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
            for batch in tqdm(valid_interaction_dataloader, desc="Computing augmented validation embeddings")
        ]
    )
    val_positive_embeddings_dataloader = DataLoader(val_positive_user_embeddings, batch_size, device, shuffle=False)
    
    val_user_embeddings = torch.tensor(val_user_embeddings, device=device)
    
    def train_epoch_from_interactions():
        """Train one epoch using on-the-fly interaction encoding.
        
        This mode supports data augmentation and contrastive learning but is slower
        as it encodes interactions through ELSA every epoch.
        
        Returns:
            dict: Training losses for this epoch.
        """
        train_losses = {"Loss": [], "L2": [], "L1": [], "L0": [], "Cosine": []}
        model.train()
        
        pbar = tqdm(train_interaction_dataloader, desc=f'Epoch {epoch}/{epochs}')
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
                
            # Encode interactions through ELSA
            embedding_batch = base_model.encode(batched_interactions).detach()
            
            # Train SAE
            losses = model.train_step(optimizer, embedding_batch, positive_batch)
            pbar.set_postfix({'train_loss': losses['Loss'].cpu().item()})
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
        pbar = tqdm(train_embeddings_dataloader, desc=f'Epoch {epoch}/{epochs}')
        for batched_embeddings in pbar:
            losses = model.train_step(optimizer, batched_embeddings, None)
            pbar.set_postfix({'train_loss': losses['Loss'].cpu().item()})
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
            mlflow.log_metric(f'loss/{key}/train', float(np.mean(val)), step=epoch)
                
        # Periodic evaluation
        if epoch % evaluate_every == 0:
            model.eval()
            # Compute validation losses
            valid_losses = {"Loss": [], "L2": [], "L1": [], "L0": [], "Cosine": [], "Auxiliary": [], "Contrastive": []}
            for embedding, positive_embedding in zip(valid_embeddings_dataloader, val_positive_embeddings_dataloader):
                losses = model.compute_loss_dict(embedding, positive_embedding)
                for key, val in losses.items():
                    valid_losses[key].append(val.item())
            
            # Log validation losses
            for key, val in valid_losses.items():
                mlflow.log_metric(f'loss/{key}/valid', float(np.mean(val)), step=epoch)
                
            # Compute validation metrics (sparsity, reconstruction quality, recommendation performance)
            valid_metrics = evaluate_sparse_encoder(base_model, model, valid_csr, target_ratio, batch_size, device, seed=seed)
            for key, val in valid_metrics.items():
                mlflow.log_metric(f'{key}/valid', val, step=epoch)
            
            valid_loss = float(np.mean(valid_losses["Loss"]))
            logger.info(
                f'Epoch {epoch}/{epochs} - '
                f'Loss: {valid_loss:.4f} - '
                f'Cosine: {valid_metrics["CosineSim"]:.4f} - '
                f'NDCG20 Degradation: {valid_metrics["NDCG20_Degradation"]:.4f}'
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
                        logger.info(f'Early stopping at epoch {epoch}')
                        break
    # Restore best model if early stopping was used
    if early_stop > 0:
        logger.info(f'Loading best model from epoch {best_epoch}')
        model = best_model
        optimizer = best_optimizer
    
    # Final evaluation on test set
    test_metrics = evaluate_sparse_encoder(base_model, model, test_csr, target_ratio, batch_size, device, seed=seed)
    for key, val in test_metrics.items():
        mlflow.log_metric(f'{key}/test', val)
    logger.info(
        f'Test metrics - '
        f'Cosine: {test_metrics["CosineSim"]:.4f} - '
        f'L0: {test_metrics["L0"]:.1f} - '
        f'NDCG20 Degradation: {test_metrics["NDCG20_Degradation"]:.4f}'
    )
    
    # Save model checkpoint and log artifacts to MLflow
    temp_path = './checkpoint.ckpt'
    save_checkpoint(model, optimizer, temp_path)
    mlflow.log_artifact(temp_path)
    mlflow.log_artifact('utils/torch/models/sae.py')
    os.remove(temp_path)
    logger.info('Model successfully saved')


class Plugin(BasePlugin):
    """SAE (Sparse Autoencoder) training plugin.
    
    Trains a sparse autoencoder on top of a pre-trained ELSA model to learn
    interpretable, sparse representations of user embeddings while maintaining
    recommendation quality.
    """
    
    def run(
        self,
        context: dict,
        dataset: str = 'LastFM1k',
        epochs: int = 4000,
        early_stop: int = 250,
        batch_size: int = 64,
        embedding_dim: int = 2048,
        top_k: int = 128,
        sample_users: bool = False,
        model: str = 'TopKSAE',
        note: str = '',
        target_ratio: float = 0.2,
        normalize: bool = False,

        seed: int = 42,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,

        reconstruction_loss: str = "Cosine",
        auxiliary_coef: float = 1/32,
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
            dataset: Dataset to use ('LastFM1k' or 'MovieLens').
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
            val_ratio: Fraction of data to use for validation.
            test_ratio: Fraction of data to use for testing.
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
        logger.info(f'Using device: {device}')
        
        set_seed(seed)
        
        # Create args namespace for compatibility
        args = argparse.Namespace(
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            early_stop=early_stop,
            embedding_dim=embedding_dim,
            top_k=top_k,
            base_run_id=context['training_cfm']['run_id'],  
            sample_users=sample_users,
            model=model,
            note=note,
            target_ratio=target_ratio,
            normalize=normalize,
            seed=seed,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            reconstruction_loss=reconstruction_loss,
            auxiliary_coef=auxiliary_coef,
            contrastive_coef=contrastive_coef,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            l1_coef=l1_coef,
            evaluate_every=evaluate_every,
            n_batches_to_dead=n_batches_to_dead,
            topk_aux=topk_aux,
        )

        # Load base model information from previous pipeline step
        base_model_run = mlflow.get_run(args.base_run_id)
        
        base_params = base_model_run.data.params
        artifact_path = base_model_run.info.artifact_uri
        # remove all before mlruns
        artifact_path = './' + artifact_path[artifact_path.find('mlruns'):]
        
        assert base_params['dataset'] == args.dataset, 'Base model dataset does not match current dataset'
        
        logger.info(f'Params: {vars(args)}')
        
        args.base_model = base_params['model']
        args.base_factors = int(base_params['factors'])
        args.base_min_user_interactions = int(base_params['min_user_interactions'])
        args.base_min_item_interactions = int(base_params['min_item_interactions'])
        args.base_users = int(base_params['users'])
        args.base_items = int(base_params['items'])
        args.expansion_ratio = args.embedding_dim / args.base_factors
        args.reconstruction_coef = 1 - (args.auxiliary_coef + args.contrastive_coef + args.l1_coef)
        
        logger.info(f'Base model: {args.base_model} with {args.base_factors} factors')
        logger.info(f'Expansion ratio: {args.expansion_ratio:.1f}x ({args.base_factors} → {args.embedding_dim})')
        
        # Load and prepare dataset
        logger.info(f'Loading {args.dataset}')
        if args.dataset == 'LastFM1k':
            dataset_loader = LastFm1kLoader()
        elif args.dataset == 'MovieLens':
            dataset_loader = MovieLensLoader()
        else:
            raise ValueError(f'Dataset {args.dataset} not supported. Check typos.')
        dataset_loader.prepare(args)
        
        args.min_user_interactions = dataset_loader.MIN_USER_INTERACTIONS
        args.min_item_interactions = dataset_loader.MIN_ITEM_INTERACTIONS
        args.users = len(dataset_loader.users)
        args.items = len(dataset_loader.items)
        
        assert args.items == args.base_items, 'Number of items in dataset does not match base model'
        assert args.users == args.base_users, 'Number of users in dataset does not match base model'
        
        train_csr = dataset_loader.train_csr
        valid_csr = dataset_loader.valid_csr
        test_csr = dataset_loader.test_csr
        
        logger.info(f'Training data: {train_csr.shape}, Validation data: {valid_csr.shape}, Test data: {test_csr.shape}')
        
        # Load pre-trained ELSA model from artifacts
        base_model = ELSA(args.base_items, args.base_factors)
        base_optimizer = torch.optim.Adam(base_model.parameters())
        load_checkpoint(base_model, base_optimizer, f'{artifact_path}/checkpoint.ckpt', device)
        base_model.to(device)
        base_model.eval()
        
        logger.info('Base ELSA model loaded successfully')
        
        # Configure SAE model
        cfg = {
            'reconstruction_loss': args.reconstruction_loss,
            "topk_aux": args.topk_aux,
            "n_batches_to_dead": args.n_batches_to_dead,
            "l1_coef": args.l1_coef,
            "k": args.top_k,
            "device": device,
            "normalize": args.normalize,
            "auxiliary_coef": args.auxiliary_coef,
            "contrastive_coef": args.contrastive_coef,
            "reconstruction_coef": args.reconstruction_coef,
        }
        
        # Initialize SAE model based on variant
        if args.model == 'BasicSAE':
            model = BasicSAE(args.base_factors, args.embedding_dim, cfg).to(device)
        elif args.model == 'TopKSAE':
            model = TopKSAE(args.base_factors, args.embedding_dim, cfg).to(device)
        elif args.model == 'BatchTopKSAE':
            model = BatchTopKSAE(args.base_factors, args.embedding_dim, cfg).to(device)
        else:
            raise ValueError(f'Model {args.model} not supported. Check typos.')
        
        logger.info(f'Initialized {args.model} with {args.embedding_dim} dimensions')

        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        
        # Log all parameters to MLflow
        mlflow.log_params({
            'expansion_ratio': args.expansion_ratio,
            'min_item_interactions': args.min_item_interactions,
            'target_ratio': args.target_ratio,
            'beta2': args.beta2,
            'beta1': args.beta1,
            'top_k': args.top_k,
            'seed': args.seed,
            'l1_coef': args.l1_coef,
            'min_user_interactions': args.min_user_interactions,
            'epochs': args.epochs,
            'note': args.note,
            'reconstruction_loss': args.reconstruction_loss,
            'val_ratio': args.val_ratio,
            'dataset': args.dataset,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'base_factors': args.base_factors,
            'normalize': args.normalize,
            'base_model': args.base_model,
            'base_min_item_interactions': args.base_min_item_interactions,
            'users': args.users,
            'topk_aux': args.topk_aux,
            'items': args.items,
            'base_users': args.base_users,
            'early_stop': args.early_stop,
            'evaluate_every': args.evaluate_every,
            'reconstruction_coef': args.reconstruction_coef,
            'embedding_dim': args.embedding_dim,
            'base_items': args.base_items,
            'auxiliary_coef': args.auxiliary_coef,
            'contrastive_coef': args.contrastive_coef,
            'n_batches_to_dead': args.n_batches_to_dead,
            'test_ratio': args.test_ratio,
            'model': args.model,
            'base_min_user_interactions': args.base_min_user_interactions,
            'sample_users': args.sample_users,
        })
        
        # Train the model
        train(
            model=model,
            base_model=base_model,
            optimizer=optimizer,
            train_csr=train_csr,
            valid_csr=valid_csr,
            test_csr=test_csr,
            device=device,
            dataset=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            early_stop=args.early_stop,
            evaluate_every=args.evaluate_every,
            model_name=args.model,
            embedding_dim=args.embedding_dim,
            top_k=args.top_k,
            contrastive_coef=args.contrastive_coef,
            sample_users=args.sample_users,
            target_ratio=args.target_ratio,
            seed=args.seed,
        )

        # Update context for next pipeline step
        context["training_sae"] = {
            "status": "trained",
            "model_name": args.model,
            "run_id": mlflow.active_run().info.run_id,
            "embedding_dim": args.embedding_dim,
            "top_k": args.top_k if args.model in ['TopKSAE', 'BatchTopKSAE'] else None,
        }
        return context


if __name__ == '__main__':
    args = parse_arguments()
    plugin = Plugin()
    plugin.run(context={}, **vars(args))