import torch
import numpy as np
import random

class Utils: 
    @staticmethod
    def set_seed(seed: int) -> None:
        torch.manual_seed(seed)  # CPU seed
        torch.mps.manual_seed(seed)  # Metal seed
        torch.cuda.manual_seed(seed)  # GPU seed
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)  # NumPy seed
        random.seed(seed)  # Python seed

    @staticmethod
    def set_device():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device('mps') if torch.backends.mps.is_available() else device
        return device
    

    @staticmethod
    def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, filepath: str, device) -> None:
        checkpoint = torch.load(filepath, weights_only=True, map_location=str(device))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])