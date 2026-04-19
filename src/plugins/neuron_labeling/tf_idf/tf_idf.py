import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer

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
            OutputArtifactSpec(
                "tfidf_tag_to_neuron",
                "tfidf_tag_to_neuron.npz",
                "npz",
            ),
            OutputArtifactSpec(
                "tfidf_neuron_to_tag",
                "tfidf_neuron_to_tag.npz",
                "npz",
            ),
            OutputArtifactSpec("neuron_labels", "neuron_labels.json", "json"),
            OutputArtifactSpec(
                "characteristic_neuron_per_tag",
                "characteristic_neuron_per_tag.json",
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
        batch_size: int = 1024,
        seed: int = 42,
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

        # build tag–item probability matrix
        self.tag_item_prob: sp.csr_matrix = self.tag_item_counts.multiply(
            1.0 / self.tag_item_counts.sum(axis=1)
        )

        # aggregate tag → neuron
        self.tag_neuron = self.tag_item_prob @ self.item_acts.numpy()

        # TF-IDF
        tfidf = TfidfTransformer(norm=None)
        self.tfidf_tag_to_neuron = tfidf.fit_transform(self.tag_neuron)
        self.tfidf_neuron_to_tag = tfidf.fit_transform(self.tag_neuron.T).T

        self.neuron_labels = {
            int(n): self.tag_ids[int(self.tfidf_neuron_to_tag[:, n].argmax())]
            for n in range(self.tfidf_neuron_to_tag.shape[1])
        }

        # tag → neuron mappings (used by the steering plugin)
        # characteristic_neuron_per_tag: for each tag, the neuron that best characterises it
        #   tfidf with term=tag, document=neuron → argmax over neurons for each tag
        tfidf_tn_dense = (
            self.tfidf_tag_to_neuron.toarray()
            if sp.issparse(self.tfidf_tag_to_neuron)
            else np.asarray(self.tfidf_tag_to_neuron)
        )
        self.characteristic_neuron_per_tag = {
            self.tag_ids[int(t)]: int(tfidf_tn_dense[t].argmax())
            for t in range(tfidf_tn_dense.shape[0])
        }

        # top_neuron_per_tag: for each tag, the neuron whose firing is most unique to it
        #   tfidf with term=neuron, document=tag → argmax over neurons for each tag
        tfidf_nt_dense = (
            self.tfidf_neuron_to_tag.toarray()
            if sp.issparse(self.tfidf_neuron_to_tag)
            else np.asarray(self.tfidf_neuron_to_tag)
        )
        self.top_neuron_per_tag = {
            self.tag_ids[int(t)]: int(tfidf_nt_dense[t].argmax())
            for t in range(min(tfidf_nt_dense.shape[0], len(self.tag_ids)))
        }

        # output params
        self.neuron_labeling = True
        self.num_tags = len(self.tag_ids)
        self.num_neurons = self.item_acts.shape[1]
