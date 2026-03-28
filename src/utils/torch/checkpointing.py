import torch
import os


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, filepath: str) -> None:
    """Save model and optimizer state to a checkpoint file.
    
    Args:
        model: PyTorch model to save.
        optimizer: Optimizer to save.
        filepath: Path where checkpoint will be saved.
    """
    checkpoint = {
        "model_state_dict": model.to('cpu').state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    
    if '/' in filepath:
        os.makedirs("/".join(filepath.split("/")[:-1]), exist_ok=True)
    
    torch.save(checkpoint, filepath)
    

def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, filepath: str, device) -> None:
    """Load model and optimizer state from a checkpoint file.
    
    Args:
        model: PyTorch model to load state into.
        optimizer: Optimizer to load state into.
        filepath: Path to checkpoint file.
        device: Device to map tensors to.
    """
    checkpoint = torch.load(filepath, weights_only=True, map_location=str(device))
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
