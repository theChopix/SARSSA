from typing import Annotated

import numpy as np
import scipy.sparse as sp

from plugins.neuron_labeling._confidence import (
    labels_with_confidence,
    point_biserial_matrix,
)
from plugins.plugin_interface import (
    ArtifactSpec,
    BasePlugin,
    OutputArtifactSpec,
    OutputParamSpec,
    PluginIOSpec,
)
from utils.plugin_logger import get_logger
from utils.torch.evaluation import compute_sae_item_activations
from utils.torch.runtime import set_seed

logger = get_logger(__name__)


class Plugin(BasePlugin):
    name = "Tag Correlation Labeling"
    description = (
        "Assigns a human-readable label to every autoencoder neuron. It runs the "
        "autoencoder over all items to measure each neuron's activations, then "
        "labels each neuron with the item attribute (tag) whose presence its "
        "activation correlates with most strongly, using the point-biserial "
        "correlation between the binary attribute and the continuous activation."
    )

    io_spec = PluginIOSpec(
        required_steps=["dataset_loading", "training_cfm", "training_sae"],
        input_artifacts=[
            ArtifactSpec(
                "dataset_loading",
                "items.npy",
                "items",
                "npy",
                loader_kwargs={"allow_pickle": True},
            ),
            ArtifactSpec("dataset_loading", "tag_ids.json", "tag_ids", "json"),
            ArtifactSpec(
                "dataset_loading",
                "tag_item_matrix.npz",
                "tag_item_counts",
                "npz",
            ),
            ArtifactSpec("training_cfm", "", "base_model", "base_model"),
            ArtifactSpec("training_sae", "", "sae", "sae_model"),
        ],
        output_artifacts=[
            OutputArtifactSpec("item_acts", "item_acts.npz", "npz"),
            OutputArtifactSpec("neuron_labels", "neuron_labels.json", "json"),
            OutputArtifactSpec(
                "neuron_labels_with_confidence",
                "neuron_labels_with_confidence.json",
                "json",
            ),
            OutputArtifactSpec(
                "top_tag_per_neuron",
                "top_tag_per_neuron.json",
                "json",
            ),
            OutputArtifactSpec(
                "top_neuron_per_tag",
                "top_neuron_per_tag.json",
                "json",
            ),
        ],
        output_params=[
            OutputParamSpec("num_tags", "num_tags"),
            OutputParamSpec("num_neurons", "num_neurons"),
            OutputParamSpec("mean_top_correlation", "mean_top_correlation"),
            OutputParamSpec("mean_confidence", "mean_confidence"),
        ],
    )

    def run(
        self,
        batch_size: Annotated[
            int,
            "Items encoded per forward pass when computing SAE activations. "
            "Larger is faster but uses more memory; does not change the "
            "resulting neuron labels.",
        ] = 1024,
        seed: Annotated[
            int,
            "Random seed for the activation computation. Fix for "
            "reproducible neuron labels across runs.",
        ] = 42,
        min_support: Annotated[
            int,
            "Minimum number of items a tag must apply to before it can label a "
            "neuron. Point-biserial correlation is unstable for very rare tags, "
            "so tags below this threshold are ignored during labelling.",
        ] = 5,
    ) -> None:
        """Compute neuron labels from SAE activation - attribute correlations.

        Args:
            batch_size: Batch size for computing SAE activations.
            seed: Random seed for reproducibility.
            min_support: Minimum items per tag for the tag to be considered.
        """
        # CPU: wide one-hot pass OOMs small GPUs
        device = "cpu"
        set_seed(seed)

        self.base_model.to(device)
        self.sae.to(device)

        # SAE activations: (items x neurons)
        item_acts = compute_sae_item_activations(
            self.base_model,
            self.sae,
            len(self.items),
            batch_size=batch_size,
            device=device,
        )
        item_acts_np = item_acts.numpy()
        num_items, num_neurons = item_acts_np.shape

        # binary item x tag matrix (tag_item_counts is tags x items, un-deduplicated)
        attr = (self.tag_item_counts > 0).astype(np.float64).T.tocsr()

        # drop tags on too few items
        items_per_tag = np.asarray(attr.sum(axis=0)).ravel()
        valid_tag_mask = (items_per_tag >= min_support) & (items_per_tag < num_items)
        if not valid_tag_mask.any():
            logger.warning("No tag meets min_support=%d; neurons stay unlabelled.", min_support)

        corr = point_biserial_matrix(item_acts_np, attr)

        # dead neurons never activate (all-zero column) -> no variance -> masked
        #   out and left unlabelled
        dead_neuron_mask = item_acts_np.sum(axis=0) == 0

        # label each neuron with its most-correlated valid tag; dead neurons
        #   (and the no-valid-tag case) get None. The tag index is kept so its
        #   correlation can be reused as the label's confidence.
        corr_by_neuron = corr.copy()
        corr_by_neuron[~valid_tag_mask, :] = -np.inf
        labelled = valid_tag_mask.any()
        label_tag_index = {
            int(n): None
            if dead_neuron_mask[n] or not labelled
            else int(corr_by_neuron[:, n].argmax())
            for n in range(num_neurons)
        }
        self.top_tag_per_neuron = {
            n: None if idx is None else self.tag_ids[idx] for n, idx in label_tag_index.items()
        }

        # neuron_labels is a copy of top_tag_per_neuron
        self.neuron_labels = dict(self.top_tag_per_neuron)

        # attach each label's point-biserial correlation as its confidence
        self.neuron_labels_with_confidence, self.mean_confidence = labels_with_confidence(
            self.neuron_labels, label_tag_index, corr
        )

        # best neuron per valid tag; dead neurons masked so never selected.
        corr_by_tag = corr.copy()
        corr_by_tag[:, dead_neuron_mask] = -np.inf
        self.top_neuron_per_tag = {
            self.tag_ids[int(t)]: int(corr_by_tag[t].argmax())
            for t in range(corr.shape[0])
            if valid_tag_mask[t]
        }

        # overall SAE interpretability - mean best correlation per valid tag
        top_corr = corr_by_tag[valid_tag_mask].max(axis=1) if labelled else np.array([])
        self.mean_top_correlation = float(top_corr.mean()) if top_corr.size else 0.0

        self.item_acts = sp.csr_matrix(item_acts_np)

        self.num_tags = len(self.tag_ids)
        self.num_neurons = num_neurons
