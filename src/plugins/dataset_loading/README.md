# Dataset Loading Plugin

This plugin system provides efficient dataset loading and preparation for the SARSSA pipeline. It loads datasets once, stores all necessary artifacts in MLflow, and allows downstream plugins to reuse the prepared data without recomputation.

## Architecture

The dataset loading system consists of:

1. **Dataset Loaders** (`movieLens_loader.py`, `lastFm1k_loader.py`) - Handle dataset-specific loading logic
2. **Plugin Interfaces** (`plugin.py`) - MLflow integration and artifact storage
3. **Artifact Loader** (`artifact_loader.py`) - Helper utilities for downstream plugins to load artifacts

## Supported Datasets

- **MovieLens** - Movie ratings dataset with tag metadata
- **LastFM1k** - Music listening dataset

## How It Works

### Approach: Store Multiple Configurations

The plugin uses **Approach 1** from the design:

- Stores **both** the full dataset and train/val/test splits in a single run
- Training plugins (`training_cfm`, `training_sae`) use the split data
- Neuron labeling plugin uses the full dataset
- All from a single `dataset_loading` execution

### Artifacts Stored

Each dataset_loading run stores the following to MLflow:

**Core Data:**
- `users.npy` - User ID array
- `items.npy` - Item ID array  
- `full_csr.npz` - Full interaction matrix (all users, for neuron_labeling)

**Split Data (for training):**
- `train_csr.npz` - Training interactions
- `valid_csr.npz` - Validation interactions
- `test_csr.npz` - Test interactions
- `train_idx.npy`, `valid_idx.npy`, `test_idx.npy` - Split indices
- `train_users.npy`, `valid_users.npy`, `test_users.npy` - Split user arrays

**Metadata:**
- `metadata.json` - Complete dataset information

**Tags (MovieLens only):**
- `tag_ids.json` - List of tag names
- `tag_item_matrix.npz` - Sparse tag-item matrix

## Usage

### 1. Running the Dataset Loading Plugin

```python
# Example: Load MovieLens dataset
from plugins.dataset_loading.movieLens.plugin import Plugin

context = {}
plugin = Plugin()
context = plugin.run(
    context=context,
    seed=42,
    val_ratio=0.1,
    test_ratio=0.1,
)

# The context now contains:
# context["dataset"] = {
#     "run_id": "...",
#     "artifact_path": "dataset",
#     "dataset_name": "MovieLens",
#     "num_users": 1234,
#     "num_items": 5678,
#     ...
# }
```

### 2. Loading Artifacts in Downstream Plugins

#### Option A: Load All Artifacts

```python
from plugins.dataset_loading.artifact_loader import DatasetArtifactLoader

# From context (recommended)
loader = DatasetArtifactLoader.from_context(context)
artifacts = loader.load()

# Access data
users = artifacts.users
items = artifacts.items
train_csr = artifacts.train_csr
full_csr = artifacts.full_csr
metadata = artifacts.metadata

# Check if tags are available
if artifacts.has_tags:
    tag_ids = artifacts.tag_ids
    tag_item_matrix = artifacts.tag_item_matrix
```

#### Option B: Load Specific Artifacts

```python
# For training plugins
loader = DatasetArtifactLoader.from_context(context)
train_csr, valid_csr, test_csr, items, users = loader.load_for_training()

# For neuron labeling
full_csr, items, tag_ids, tag_item_matrix = loader.load_for_neuron_labeling()

# Or load individual artifacts
train_csr = loader.load_train_csr()
metadata = loader.load_metadata()
```

#### Option C: Load from Run ID

```python
# If you have the run_id directly
loader = DatasetArtifactLoader(run_id="abc123def456")
artifacts = loader.load()
```

## Complete Pipeline Example

### Workflow 1: Training Pipeline (ELSA → SAE → Neuron Labeling)

```python
import mlflow

# Step 1: Load dataset
from plugins.dataset_loading.movieLens.plugin import Plugin as DatasetPlugin

context = {}
with mlflow.start_run(run_name="dataset_loading"):
    dataset_plugin = DatasetPlugin()
    context = dataset_plugin.run(
        context=context,
        seed=42,
        val_ratio=0.1,
        test_ratio=0.1,
    )

# Step 2: Train ELSA model
from plugins.training_cfm.elsa_trainer import Plugin as ElsaPlugin
from plugins.dataset_loading.artifact_loader import DatasetArtifactLoader

with mlflow.start_run(run_name="elsa_training"):
    # Load dataset artifacts
    loader = DatasetArtifactLoader.from_context(context)
    train_csr, valid_csr, test_csr, items, users = loader.load_for_training()
    
    # Train ELSA
    elsa_plugin = ElsaPlugin()
    context = elsa_plugin.run(
        context=context,
        train_csr=train_csr,
        valid_csr=valid_csr,
        test_csr=test_csr,
        items=items,
        # ... other params
    )

# Step 3: Train SAE
from plugins.training_sae.sae_trainer import Plugin as SaePlugin

with mlflow.start_run(run_name="sae_training"):
    # Reuse same dataset artifacts
    loader = DatasetArtifactLoader.from_context(context)
    train_csr, valid_csr, test_csr, items, users = loader.load_for_training()
    
    sae_plugin = SaePlugin()
    context = sae_plugin.run(
        context=context,
        train_csr=train_csr,
        valid_csr=valid_csr,
        test_csr=test_csr,
        # ... other params
    )

# Step 4: Neuron Labeling
from plugins.neuron_labeling.tf_idf import Plugin as LabelingPlugin

with mlflow.start_run(run_name="neuron_labeling"):
    # Load full dataset (no splits)
    loader = DatasetArtifactLoader.from_context(context)
    full_csr, items, tag_ids, tag_item_matrix = loader.load_for_neuron_labeling()
    
    labeling_plugin = LabelingPlugin()
    context = labeling_plugin.run(
        context=context,
        full_csr=full_csr,
        items=items,
        tag_ids=tag_ids,
        tag_item_matrix=tag_item_matrix,
        # ... other params
    )
```

## Benefits

### 1. **Performance**
- Dataset preparation (`prepare()`) runs **once** instead of 3+ times
- Significant time savings for large datasets
- Reduced I/O operations

### 2. **Consistency**
- All plugins use **identical** dataset splits
- Same seed ensures reproducibility
- No risk of different splits across experiments

### 3. **Flexibility**
- Training plugins use split data (train/val/test)
- Neuron labeling uses full dataset
- Both needs satisfied from single run

### 4. **Modularity**
- Clear separation of concerns
- Dataset loading is a distinct pipeline stage
- Easy to swap datasets or modify splits

## Parameters

### Dataset Loading Plugin Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | int | 42 | Random seed for reproducibility |
| `val_ratio` | float | 0.1 | Validation set ratio (0.0 - 1.0) |
| `test_ratio` | float | 0.1 | Test set ratio (0.0 - 1.0) |

### Context Structure

After running dataset_loading, the context contains:

```python
context["dataset"] = {
    "run_id": str,              # MLflow run ID
    "artifact_path": str,       # "dataset"
    "dataset_name": str,        # "MovieLens" or "LastFM1k"
    "num_users": int,           # Total number of users
    "num_items": int,           # Total number of items
    "num_interactions": int,    # Total interactions
    "has_tags": bool,           # Whether tags are available
    "seed": int,                # Random seed used
    "val_ratio": float,         # Validation ratio used
    "test_ratio": float,        # Test ratio used
}
```

## Dataset-Specific Details

### MovieLens

- **Minimum user interactions:** 50
- **Minimum item interactions:** 20
- **Rating threshold:** ≥ 4.0 (only high ratings)
- **Tags:** Available (tag_ids.json, tag_item_matrix.npz)
- **Default paths:**
  - Ratings: `../data/movieLens/MovieLensRatings.csv`
  - Tags: `../data/movieLens/MovieLensTags.csv`

### LastFM1k

- **Minimum user interactions:** 5
- **Minimum item interactions:** 10
- **Tags:** Not available
- **Default path:** `../data/LastFm1k.tsv`
- **Special handling:** Skips corrupted rows

## Troubleshooting

### Error: "Context does not contain 'dataset' key"

**Cause:** Downstream plugin trying to load artifacts before dataset_loading ran.

**Solution:** Ensure dataset_loading plugin runs first and context is passed through.

### Error: "FileNotFoundError: tag_ids.json"

**Cause:** Trying to load tags from LastFM1k dataset (which doesn't have tags).

**Solution:** Check `artifacts.has_tags` before accessing tag data, or use the safe loading methods:
```python
tag_ids = loader.load_tag_ids()  # Returns None if not available
if tag_ids is not None:
    # Use tags
```

### Different Split Requirements

**Q:** What if I need different val/test ratios for different experiments?

**A:** Run dataset_loading multiple times with different parameters. Each run stores its own artifacts with its own run_id. Reference the appropriate run_id in downstream plugins.

## Migration Guide

### Before (Old Approach)

```python
# Each plugin loaded dataset independently
dataset_loader = MovieLensLoader()
dataset_loader.prepare(argparse.Namespace(
    seed=seed,
    val_ratio=0.1,
    test_ratio=0.1,
))
# Use dataset_loader.train_csr, etc.
```

### After (New Approach)

```python
# Step 1: Run dataset_loading once
context = dataset_plugin.run(context, seed=42, val_ratio=0.1, test_ratio=0.1)

# Step 2: Load in downstream plugins
loader = DatasetArtifactLoader.from_context(context)
train_csr, valid_csr, test_csr, items, users = loader.load_for_training()
```

## Future Enhancements

Potential improvements:
- In-memory caching for repeated loads within same session
- Support for additional datasets
- Automatic dataset validation between plugins
- Dataset versioning and lineage tracking
