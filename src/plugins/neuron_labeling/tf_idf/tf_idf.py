from typing import Annotated

import numpy as np
import scipy.sparse as sp

from plugins.plugin_interface import (
    ArtifactSpec,
    BasePlugin,
    OutputArtifactSpec,
    OutputParamSpec,
    PluginIOSpec,
)
from utils.plugin_logger import get_logger
from utils.torch.evalution import compute_sae_item_activations
from utils.torch.runtime import set_device, set_seed

logger = get_logger(__name__)


class Plugin(BasePlugin):
    name = "TF-IDF Labeling"

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
            OutputArtifactSpec("item_acts", "item_acts.pt", "pt"),
            OutputArtifactSpec("tag_item_prob", "tag_item_prob.npz", "npz"),
            OutputArtifactSpec("tag_neuron", "tag_neuron.npy", "npy"),
            OutputArtifactSpec("neuron_labels", "neuron_labels.json", "json"),
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
            OutputParamSpec("neuron_labeling", "neuron_labeling"),
            OutputParamSpec("num_tags", "num_tags"),
            OutputParamSpec("num_neurons", "num_neurons"),
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
    ) -> None:
        """Compute TF-IDF neuron labels from SAE activations.

        Args:
            batch_size: Batch size for computing SAE activations.
            seed: Random seed for reproducibility.
        """
        device = set_device()
        set_seed(seed)

        self.base_model.to(device)
        self.sae.to(device)

        # compute SAE activations
        self.item_acts = compute_sae_item_activations(
            self.base_model,
            self.sae,
            len(self.items),
            batch_size=batch_size,
            device=device,
        )

        # build tag–item probability matrix (global normalization)
        self.tag_item_prob: sp.csr_matrix = self.tag_item_counts.copy()
        self.tag_item_prob.data /= self.tag_item_prob.data.sum()

        # aggregate tag → neuron
        self.tag_neuron = self.tag_item_prob @ self.item_acts.numpy()

        tag_neuron_dense = (
            self.tag_neuron.toarray()
            if sp.issparse(self.tag_neuron)
            else np.asarray(self.tag_neuron)
        )

        # top_tag_per_neuron: for each neuron, the tag with highest
        #   tfidf score (term=neuron, document=tag)
        tfidf_nt = self._compute_tfidf(tag_neuron_dense.T)
        self.top_tag_per_neuron = {
            int(n): self.tag_ids[int(tfidf_nt[n].argmax())] for n in range(tfidf_nt.shape[0])
        }

        # neuron_labels is a copy of top_tag_per_neuron
        self.neuron_labels = dict(self.top_tag_per_neuron)

        # top_neuron_per_tag: for each tag, the neuron that best
        #   characterises it (term=tag, document=neuron)
        tfidf_tn = self._compute_tfidf(tag_neuron_dense)
        self.top_neuron_per_tag = {
            self.tag_ids[int(t)]: int(tfidf_tn[t].argmax()) for t in range(tfidf_tn.shape[0])
        }

        # output params
        self.neuron_labeling = True
        self.num_tags = len(self.tag_ids)
        self.num_neurons = self.item_acts.shape[1]

    @staticmethod
    def _compute_tfidf(x: np.ndarray) -> np.ndarray:
        """Compute TF-IDF for a term-document value matrix.

        Rows are treated as terms and columns as documents.
        TF is normalized by column sum (document length), IDF is
        computed per term (row) with smoothing.

        Args:
            x: Dense matrix of shape ``(num_terms, num_documents)``.

        Returns:
            np.ndarray: TF-IDF matrix of the same shape.
        """
        col_sums = np.sum(x, axis=0, keepdims=True)
        tf = np.divide(
            x,
            col_sums,
            out=np.zeros_like(x, dtype=float),
            where=col_sums != 0,
        )
        df = np.count_nonzero(x, axis=1)
        num_documents = x.shape[1]
        idf = np.log((num_documents + 1) / (df + 1)) + 1
        return tf * idf[:, np.newaxis]
