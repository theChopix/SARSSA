import json
import torch
from pathlib import Path

from utils.torch.model_registry import get_base_model_class, get_sae_model_class

# Import model modules to trigger @register_base_model / @register_sae_model decorators
import utils.torch.models.elsa  # noqa: F401
import utils.torch.models.sae  # noqa: F401


def load_base_model(artifact_path: str, device: str = "cpu"):
    """Load a base model from a config.json + model.pt artifact directory.
    
    Args:
        artifact_path: Path to directory containing config.json and model.pt.
        device: Device to load the model onto.
    
    Returns:
        Loaded model in eval mode on the specified device.
    """
    artifact_path = Path(artifact_path)

    with open(artifact_path / "config.json", "r") as f:
        config = json.load(f)

    model_cls = get_base_model_class(config["model_type"])
    model = model_cls(**config["architecture"])

    state_dict = torch.load(artifact_path / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state_dict["state_dict"])

    model.to(device)
    model.eval()

    return model


def load_sae_model(artifact_path: str, device: str = "cpu"):
    """Load a SAE model from a config.json + model.pt artifact directory.
    
    Args:
        artifact_path: Path to directory containing config.json and model.pt.
        device: Device to load the model onto.
    
    Returns:
        Loaded SAE model in eval mode on the specified device.
    """
    artifact_path = Path(artifact_path)

    with open(artifact_path / "config.json", "r") as f:
        config = json.load(f)

    model_cls = get_sae_model_class(config["model_type"])
    model = model_cls(**config["architecture"])

    state_dict = torch.load(artifact_path / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state_dict["state_dict"])

    model.to(device)
    model.eval()

    return model
