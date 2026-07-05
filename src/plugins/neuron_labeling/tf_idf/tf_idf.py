from typing import Annotated

import mlflow
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
    StaticDropdownHint,
)
from utils.plugin_logger import get_logger
from utils.torch.evaluation import compute_sae_item_activations
from utils.torch.runtime import set_seed

logger = get_logger(__name__)

# TF-IDF orientations — which entity plays the "document" role.
_ORIENTATION_NEURON_DOC = "neuron_as_document"
_ORIENTATION_TAG_DOC = "tag_as_document"


class Plugin(BasePlugin):
    name = "Tag TF-IDF Labeling"
    description = (
        "Assigns a human-readable label to every autoencoder neuron. It runs the "
        "autoencoder over all items to measure each neuron's activations, then uses "
        "TF-IDF over the dataset's tags to pick the tag that most distinctively "
        "characterizes each neuron."
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
            OutputArtifactSpec("tag_item_prob", "tag_item_prob.npz", "npz"),
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
            OutputParamSpec("num_tags", "num_tags"),
            OutputParamSpec("num_neurons", "num_neurons"),
        ],
        param_ui_hints=[
            StaticDropdownHint(
                param_name="orientation",
                choices=[_ORIENTATION_TAG_DOC, _ORIENTATION_NEURON_DOC],
            ),
        ],
    )

    def run(
        self,
        orientation: Annotated[
            str,
            "TF-IDF orientation — which entity is the document. 'tag_as_document' "
            "normalises the term frequency per tag; 'neuron_as_document' "
            "normalises it per neuron.",
        ] = _ORIENTATION_TAG_DOC,
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
            orientation: Which entity is the TF-IDF document; see the
                parameter annotation.
        """
        if orientation not in (_ORIENTATION_NEURON_DOC, _ORIENTATION_TAG_DOC):
            raise ValueError(
                f"orientation must be one of ({_ORIENTATION_NEURON_DOC!r}, "
                f"{_ORIENTATION_TAG_DOC!r}), got {orientation!r}"
            )

        # CPU: wide one-hot pass OOMs small GPUs
        device = "cpu"
        set_seed(seed)

        self.base_model.to(device)
        self.sae.to(device)

        # compute SAE activations
        item_acts = compute_sae_item_activations(
            self.base_model,
            self.sae,
            len(self.items),
            batch_size=batch_size,
            device=device,
        )

        # build tag–item probability matrix (global normalization)
        self.tag_item_prob: sp.csr_matrix = self.tag_item_counts.copy()
        self.tag_item_prob.data /= self.tag_item_prob.data.sum()

        item_acts_np = item_acts.numpy()

        # DEAD neurons never activate on any item (all-zero column in
        #   item_acts). They carry no signal, so they are masked out below
        #   and left unlabelled rather than assigned a spurious tag.
        dead_neuron_mask = item_acts_np.sum(axis=0) == 0

        # aggregate tag → neuron (kept in-memory only; not persisted)
        tag_neuron = self.tag_item_prob @ item_acts_np

        tag_neuron_dense = (
            tag_neuron.toarray() if sp.issparse(tag_neuron) else np.asarray(tag_neuron)
        )

        # TF-IDF in the requested orientation, as a (num_neurons, num_tags)
        #   score matrix so the downstream argmaxes are identical either way.
        tfidf_nt = self._oriented_tfidf(tag_neuron_dense, orientation)

        # top_tag_per_neuron: for each neuron, the tag with highest tfidf
        #   score; dead neurons get no label (None). The chosen tag index is
        #   kept so its activation-presence correlation can score the label.
        label_tag_index = {
            int(n): None if dead_neuron_mask[n] else int(tfidf_nt[n].argmax())
            for n in range(tfidf_nt.shape[0])
        }
        self.top_tag_per_neuron = {
            n: None if idx is None else self.tag_ids[idx] for n, idx in label_tag_index.items()
        }

        # neuron_labels pairs each label with its confidence: the point-biserial
        #   correlation between the neuron's activation and the binary presence
        #   of its TF-IDF label tag. TF-IDF picks distinctive tags, so this
        #   exposes how well activation actually tracks the chosen tag (can be
        #   weak or negative).
        attr = (self.tag_item_counts > 0).astype(np.float64).T.tocsr()
        corr = point_biserial_matrix(item_acts_np, attr)
        self.neuron_labels, mean_confidence = labels_with_confidence(
            self.top_tag_per_neuron, label_tag_index, corr
        )
        mlflow.log_metric("mean_confidence", mean_confidence)

        # top_neuron_per_tag: for each tag, the neuron that best characterises
        #   it — argmax over the neuron axis of the same oriented tfidf. Dead
        #   neurons are masked to -inf so they are never selected.
        tfidf_nt_masked = tfidf_nt.copy()
        tfidf_nt_masked[dead_neuron_mask, :] = -np.inf
        self.top_neuron_per_tag = {
            self.tag_ids[int(t)]: int(tfidf_nt_masked[:, t].argmax())
            for t in range(tfidf_nt_masked.shape[1])
        }

        # persist activations sparsely — TopK SAE outputs are ~94% zeros,
        # so CSR shrinks this artifact ~8x vs. a dense tensor
        self.item_acts = sp.csr_matrix(item_acts_np)

        # output params
        self.num_tags = len(self.tag_ids)
        self.num_neurons = item_acts.shape[1]

        self.notifier.success(
            f"TF-IDF labeling DONE. Mean label confidence: {mean_confidence:.3f} — "
            "the average correlation between a neuron's activation and its assigned tag."
        )

    @staticmethod
    def _oriented_tfidf(tag_neuron: np.ndarray, orientation: str) -> np.ndarray:
        """TF-IDF scores as ``(num_neurons, num_tags)`` for the orientation.

        Args:
            tag_neuron: Dense ``(num_tags, num_neurons)`` association matrix.
            orientation: Which entity is the TF-IDF document — one of the
                module-level orientation constants.

        Returns:
            np.ndarray: ``(num_neurons, num_tags)`` TF-IDF score matrix.
        """
        if orientation == _ORIENTATION_TAG_DOC:
            # document=tag, term=neuron (original): TF normalised per tag.
            return Plugin._compute_tfidf(tag_neuron.T)
        # document=neuron, term=tag: TF normalised per neuron.
        return Plugin._compute_tfidf(tag_neuron).T

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
