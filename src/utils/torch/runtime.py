import logging
import os
import random
from contextlib import nullcontext

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value.
    """
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def set_device():
    """Detect and return the best available device for training.

    Priority: CUDA > MPS (Apple Silicon) > CPU

    Returns:
        torch.device: The selected device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps") if torch.backends.mps.is_available() else device
    return device


def autocast_context(device: torch.device):
    """Return a bf16 mixed-precision autocast context on CUDA, else a no-op.

    On CUDA, matmuls run on Tensor Cores in bfloat16 (much faster on
    Ampere/Hopper/Blackwell) while autocast keeps reductions and losses in
    fp32 for numerical stability. bf16 needs no GradScaler. On CPU/MPS this
    is a no-op so training behaves exactly as before.

    Args:
        device: Device the batch/model live on.

    Returns:
        A context manager to wrap the forward pass / loss computation.
    """
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def maybe_compile(model, device):
    """Best-effort ``torch.compile`` of the model's forward on CUDA.

    Compiling lowers Python/kernel-launch overhead. It is skipped on
    non-CUDA devices and when the ``SARSSA_DISABLE_COMPILE`` environment
    variable is set, and silently falls back to eager execution if
    compilation fails, so training never breaks because of it.

    Args:
        model: The model whose ``forward`` should be compiled.
        device: Device the model runs on.

    Returns:
        The same model instance (with a compiled ``forward`` when enabled).
    """
    if device.type != "cuda" or os.environ.get("SARSSA_DISABLE_COMPILE"):
        return model
    try:
        model.forward = torch.compile(model.forward)
        logger.info("torch.compile enabled for %s", type(model).__name__)
    except Exception as exc:  # pragma: no cover - depends on runtime/GPU
        logger.warning("torch.compile failed, falling back to eager: %s", exc)
    return model
