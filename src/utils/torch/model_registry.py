BASE_MODEL_REGISTRY = {}
SAE_REGISTRY = {}


def register_base_model(name: str):
    """Decorator to register a base model class (e.g. ELSA)."""
    def decorator(cls):
        BASE_MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def register_sae_model(name: str):
    """Decorator to register a SAE model class (e.g. TopKSAE)."""
    def decorator(cls):
        SAE_REGISTRY[name] = cls
        return cls
    return decorator


def get_base_model_class(name: str):
    if name not in BASE_MODEL_REGISTRY:
        raise ValueError(f"Unknown base model type: {name}. Available: {list(BASE_MODEL_REGISTRY.keys())}")
    return BASE_MODEL_REGISTRY[name]


def get_sae_model_class(name: str):
    if name not in SAE_REGISTRY:
        raise ValueError(f"Unknown SAE model type: {name}. Available: {list(SAE_REGISTRY.keys())}")
    return SAE_REGISTRY[name]
