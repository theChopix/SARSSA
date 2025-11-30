import argparse
import torch
import mlflow
import numpy as np
import os
import datetime
from tqdm import tqdm
from copy import deepcopy

from utils.datasets.lastFm1k_loader import LastFm1kLoader
from utils.datasets.data_loader import DataLoader
from utils.models.elsa import ELSA
from utils.plugin_logger import get_logger
from .elsa_trainer_utils import Utils

from plugins.plugin_interface import BasePlugin


logger = get_logger(__name__)

device = Utils.set_device()
logger.info(f'Device: {device}')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='LastFM1k', help='Dataset to use. For now, only "LastFM1k" and "MovieLens" are supported')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--factors', type=int, default=256, help='Number of factors for the model')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for training')
    parser.add_argument('--early_stop', type=int, default=10, help='Number of epochs to wait for improvement before stopping')
    # stable parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test ratio')
    parser.add_argument('--target_ratio', type=float, default=0.2, help='Ratio of target interactions')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.99, help='Beta2 for Adam optimizer')
    return parser.parse_args()


def train(args, model: ELSA, optimizer, train_csr, valid_csr, test_csr):
    dataset = args.dataset
    nr_epochs = args.epochs
    batch_size = args.batch_size
    early_stop = args.early_stop
    
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    mlflow.log_params(vars(args))
    
    train_dataloader = DataLoader(train_csr, batch_size, device, shuffle=True)
    valid_dataloader = DataLoader(valid_csr, batch_size, device, shuffle=False)
    
    if early_stop > 0:
        best_epoch = 0
        epochs_without_improvement = 0
        best_ndcg = 0
        best_optimizer = deepcopy(optimizer)
        best_model = deepcopy(model)
    
    for epoch in range(1, nr_epochs+1):
        train_losses = []
        model.train()
        
        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch}/{nr_epochs}')
        for batch in pbar: # train one batch
            losses = model.train_step(optimizer, batch)
            train_losses.append(losses['Loss'].item())
            pbar.set_postfix({'train_loss': losses['Loss'].cpu().item()})
        mlflow.log_metric('loss/train', float(np.mean(train_losses)), step=epoch)
        # Evaluate
        model.eval()
        valid_metrics = Utils.evaluate_dense_encoder(model, valid_csr, args.target_ratio, batch_size, device, seed=args.seed+epoch)
        valid_metrics['loss'] = float(np.mean([model.compute_loss_dict(batch)['Loss'].item() for batch in valid_dataloader]))
        for key, val in valid_metrics.items():
            mlflow.log_metric(f'{key}/valid', val, step=epoch)
        logger.info(f'Epoch {epoch}/{nr_epochs} - Loss: {valid_metrics["loss"]:.4f} - R@20: {valid_metrics["R20"]:.4f} - NDCG20: {valid_metrics["NDCG20"]:.4f}')
        
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
    if early_stop > 0:
        logger.info(f'Loading best model from epoch {best_epoch}')
        model = best_model
        optimizer = best_optimizer
        
    test_metrics = Utils.evaluate_dense_encoder(model, test_csr, args.target_ratio, batch_size, device, seed=args.seed)
    for key, val in test_metrics.items():
        mlflow.log_metric(f'{key}/test', val)
    logger.info(f'Test metrics - R@20: {test_metrics["R20"]:.4f} - NDCG20: {test_metrics["NDCG20"]:.4f}')
    
    # Save model
    temp_path = 'checkpoint.ckpt'
    Utils.save_checkpoint(model, optimizer, temp_path)
    mlflow.log_artifact(temp_path)
    mlflow.log_artifact('utils/models/elsa.py')
    os.remove(temp_path)
    logger.info('Model successfully saved')
                

class Plugin(BasePlugin):
    def run(self, 
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

        args = argparse.Namespace(
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            factors=factors,
            lr=lr,
            early_stop=early_stop,
            seed=seed,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            target_ratio=target_ratio,
            beta1=beta1,
            beta2=beta2,
            model='ELSA',
        )
    
        Utils.set_seed(args.seed)
        
        # Load dataset
        logger.info(f'Loading {args.dataset}')
        if args.dataset == 'LastFM1k':
            dataset_loader = LastFm1kLoader()
        else: 
            raise ValueError(f'Dataset {args.dataset} not supported. Check typos.')
        dataset_loader.prepare(args)
        
        args.min_user_interactions = dataset_loader.MIN_USER_INTERACTIONS
        args.min_item_interactions = dataset_loader.MIN_ITEM_INTERACTIONS
        args.users = len(dataset_loader.users)
        args.items = len(dataset_loader.items)
        
        train_csr = dataset_loader.train_csr
        valid_csr = dataset_loader.valid_csr
        test_csr = dataset_loader.test_csr
        
        item_cnt = len(dataset_loader.items)
        model = ELSA(input_dim=item_cnt, embedding_dim=args.factors).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        
        train(args, model, optimizer, train_csr, valid_csr, test_csr)

        context["model"] = {"status": "trained", "model_name": "ELSA"} 
        # TODO: add needed parts into the context
        return context

if __name__ == '__main__':
    args = parse_arguments()
    plg = Plugin()
    plg.run(context={}, **vars(args))