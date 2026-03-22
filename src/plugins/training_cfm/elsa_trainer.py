import argparse
import torch
import mlflow
import numpy as np
import os
from tqdm import tqdm
from copy import deepcopy

from utils.datasets.lastFm1k_loader import LastFm1kLoader
from utils.datasets.movieLens_loader import MovieLensLoader
from utils.datasets.data_loader import DataLoader
from utils.models.elsa import ELSA
from utils.plugin_logger import get_logger
from .elsa_trainer_utils import Utils

from plugins.plugin_interface import BasePlugin


logger = get_logger(__name__)


def parse_arguments():
    """Parse command-line arguments for ELSA training.
    
    Returns:
        argparse.Namespace: Parsed arguments containing all training configuration parameters.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='LastFM1k', help='Dataset to use. For now, only "LastFM1k" and "MovieLens" are supported')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--factors', type=int, default=256, help='Number of factors for the model')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for training')
    parser.add_argument('--early_stop', type=int, default=10, help='Number of epochs to wait for improvement before stopping')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test ratio')
    parser.add_argument('--target_ratio', type=float, default=0.2, help='Ratio of target interactions')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.99, help='Beta2 for Adam optimizer')
    return parser.parse_args()


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
    mlflow.log_params({
        'dataset': dataset,
        'epochs': epochs,
        'batch_size': batch_size,
        'factors': factors,
        'lr': lr,
        'early_stop': early_stop,
        'seed': seed,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'target_ratio': target_ratio,
        'beta1': beta1,
        'beta2': beta2,
        'model': 'ELSA',
        'min_user_interactions': min_user_interactions,
        'min_item_interactions': min_item_interactions,
        'users': num_users,
        'items': num_items,
    })
    
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
        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch}/{epochs}')
        for batch in pbar:
            losses = model.train_step(optimizer, batch)
            train_losses.append(losses['Loss'].item())
            pbar.set_postfix({'train_loss': losses['Loss'].cpu().item()})
        # Log average training loss for this epoch
        mlflow.log_metric('loss/train', float(np.mean(train_losses)), step=epoch)

        # Evaluate on validation set
        model.eval()
        valid_metrics = Utils.evaluate_dense_encoder(model, valid_csr, target_ratio, batch_size, device, seed=seed + epoch)
        valid_metrics['loss'] = float(np.mean([model.compute_loss_dict(batch)['Loss'].item() for batch in valid_dataloader]))
        
        # Log validation metrics to MLflow
        for key, val in valid_metrics.items():
            mlflow.log_metric(f'{key}/valid', val, step=epoch)
        
        logger.info(f'Epoch {epoch}/{epochs} - Loss: {valid_metrics["loss"]:.4f} - R@20: {valid_metrics["R20"]:.4f} - NDCG20: {valid_metrics["NDCG20"]:.4f}')
        
        # Early stopping logic
        if early_stop > 0:
            if valid_metrics['NDCG20'] > best_ndcg:
                best_ndcg = valid_metrics['NDCG20']
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
    test_metrics = Utils.evaluate_dense_encoder(model, test_csr, target_ratio, batch_size, device, seed=seed)
    for key, val in test_metrics.items():
        mlflow.log_metric(f'{key}/test', val)
    logger.info(f'Test metrics - R@20: {test_metrics["R20"]:.4f} - NDCG20: {test_metrics["NDCG20"]:.4f}')
    
    # Save model checkpoint and log artifacts to MLflow
    temp_path = 'checkpoint.ckpt'
    Utils.save_checkpoint(model, optimizer, temp_path)
    mlflow.log_artifact(temp_path)
    mlflow.log_artifact('utils/models/elsa.py')
    os.remove(temp_path)
    logger.info('Model successfully saved')


class Plugin(BasePlugin):
    """ELSA (Scalable Linear Shallow Autoencoder) training plugin.
    
    This plugin implements the training pipeline for collaborative filtering using
    the ELSA model. It handles dataset loading, model initialization, training with
    early stopping, and MLflow experiment tracking.
    """
    
    def run(
        self,
        context: dict,
        dataset: str = 'LastFM1k',
        epochs: int = 100,
        batch_size: int = 64,
        factors: int = 256,
        lr: float = 0.0001,
        early_stop: int = 10,
        seed: int = 43,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        target_ratio: float = 0.2,
        beta1: float = 0.9,
        beta2: float = 0.99,
    ):
        """Execute the ELSA training pipeline.
        
        Args:
            context: Shared context dictionary for pipeline communication.
            dataset: Dataset name ('LastFM1k' or 'MovieLens').
            epochs: Maximum number of training epochs.
            batch_size: Number of samples per batch.
            factors: Number of latent factors (embedding dimension).
            lr: Learning rate for Adam optimizer.
            early_stop: Epochs without improvement before stopping (0 to disable).
            seed: Random seed for reproducibility.
            val_ratio: Validation set ratio (0.0-1.0).
            test_ratio: Test set ratio (0.0-1.0).
            target_ratio: Ratio of interactions used as targets in evaluation.
            beta1: Adam optimizer beta1 parameter (momentum).
            beta2: Adam optimizer beta2 parameter (RMSprop).
        
        Returns:
            dict: Updated context with training status.
        """
        # Initialize device and set random seed for reproducibility
        device = Utils.set_device()
        logger.info(f'Using device: {device}')
        
        Utils.set_seed(seed)
        
        # Load and prepare dataset
        dataset_loader = self._load_dataset(dataset, seed, val_ratio, test_ratio)
        train_csr, valid_csr, test_csr = dataset_loader.train_csr, dataset_loader.valid_csr, dataset_loader.test_csr
        
        # Extract dataset metadata for logging
        min_user_interactions = dataset_loader.MIN_USER_INTERACTIONS
        min_item_interactions = dataset_loader.MIN_ITEM_INTERACTIONS
        num_users = len(dataset_loader.users)
        num_items = len(dataset_loader.items)
        
        logger.info(f'Training data: {train_csr.shape}, Validation data: {valid_csr.shape}, Test data: {test_csr.shape}')
        
        # Initialize ELSA model and optimizer
        model = ELSA(input_dim=num_items, embedding_dim=factors).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
        
        train(
            model=model,
            optimizer=optimizer,
            train_csr=train_csr,
            valid_csr=valid_csr,
            test_csr=test_csr,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            early_stop=early_stop,
            target_ratio=target_ratio,
            seed=seed,
            dataset=dataset,
            factors=factors,
            lr=lr,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            beta1=beta1,
            beta2=beta2,
            min_user_interactions=min_user_interactions,
            min_item_interactions=min_item_interactions,
            num_users=num_users,
            num_items=num_items,
        )
        
        context["model"] = {"status": "trained", "model_name": "ELSA"}
        return context
    
    def _load_dataset(self, dataset: str, seed: int, val_ratio: float, test_ratio: float):
        """Load and prepare the specified dataset.
        
        Args:
            dataset: Dataset name ('LastFM1k' or 'MovieLens').
            seed: Random seed for data splitting.
            val_ratio: Validation set ratio.
            test_ratio: Test set ratio.
        
        Returns:
            DatasetLoader: Prepared dataset loader with train/val/test splits.
        
        Raises:
            ValueError: If dataset name is not supported.
        """
        logger.info(f'Loading dataset: {dataset}')
        
        if dataset == 'LastFM1k':
            dataset_loader = LastFm1kLoader()
        elif dataset == 'MovieLens':
            dataset_loader = MovieLensLoader()
        else:
            raise ValueError(f'Dataset "{dataset}" not supported. Available: LastFM1k, MovieLens')
        
        # Prepare dataset with train/val/test splits
        config = argparse.Namespace(seed=seed, val_ratio=val_ratio, test_ratio=test_ratio)
        dataset_loader.prepare(config)
        return dataset_loader


if __name__ == '__main__':
    args = parse_arguments()
    plugin = Plugin()
    plugin.run(context={}, **vars(args))