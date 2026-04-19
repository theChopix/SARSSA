import json
import tempfile

import mlflow
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import TfidfTransformer

from plugins.plugin_interface import BasePlugin
from utils.mlflow_manager import MLflowRunLoader
from utils.plugin_logger import get_logger
from utils.torch.evalution import compute_sae_item_activations
from utils.torch.models.model_loader import load_base_model, load_sae_model
from utils.torch.runtime import set_device, set_seed

logger = get_logger(__name__)
device = set_device()


class Plugin(BasePlugin):
    name = "TF-IDF Labeling"

    def _load_artifacts(self, context, device):
        """Load dataset, ELSA, and SAE artifacts from previous pipeline steps."""
        # Load dataset artifacts
        dataset_run_id = context["dataset_loading"]["run_id"]
        dataset_loader = MLflowRunLoader(dataset_run_id)

        logger.info(f"Loading dataset artifacts from run {dataset_run_id}")

        self.items = dataset_loader.get_npy_artifact("items.npy", allow_pickle=True)
        self.num_items = len(self.items)
        tag_ids_data = dataset_loader.get_json_artifact("tag_ids.json")
        self.tag_ids = list(tag_ids_data) if isinstance(tag_ids_data, dict) else tag_ids_data
        tag_item_matrix = dataset_loader.get_npz_artifact("tag_item_matrix.npz")
        self.tag_item_counts: sp.csr_matrix = (
            sp.csr_matrix(tag_item_matrix)
            if not isinstance(tag_item_matrix, sp.csr_matrix)
            else tag_item_matrix
        )

        if self.tag_ids is None or self.tag_item_counts is None:
            raise RuntimeError("Dataset does not support neuron labeling (no tag data available)")

        # Load base model via registry-based loader
        logger.info("Loading base model")
        base_run_id = context["training_cfm"]["run_id"]
        base_loader = MLflowRunLoader(base_run_id)
        self.base_model = load_base_model(base_loader.download_artifact_dir(), device)
        logger.info("Base model loaded successfully")

        # Load SAE model via registry-based loader
        logger.info("Loading SAE model")
        sae_run_id = context["training_sae"]["run_id"]
        sae_loader = MLflowRunLoader(sae_run_id)
        self.sae = load_sae_model(sae_loader.download_artifact_dir(), device)
        logger.info("SAE model loaded successfully")

    def run(
        self,
        batch_size: int = 1024,
        seed: int = 42,
    ):
        set_seed(seed)

        self._load_artifacts(self._context, device)

        # compute SAE activations
        item_acts = compute_sae_item_activations(
            self.base_model,
            self.sae,
            self.num_items,
            batch_size=batch_size,
            device=device,
        )

        # build tag–item probability matrix
        tag_item_prob: sp.csr_matrix = self.tag_item_counts.multiply(
            1.0 / self.tag_item_counts.sum(axis=1)
        )

        # aggregate tag → neuron
        tag_neuron = tag_item_prob @ item_acts.numpy()

        # TF-IDF
        tfidf = TfidfTransformer(norm=None)
        tfidf_tn = tfidf.fit_transform(tag_neuron)
        tfidf_nt = tfidf.fit_transform(tag_neuron.T).T

        neuron_labels = {
            int(n): self.tag_ids[int(tfidf_nt[:, n].argmax())] for n in range(tfidf_nt.shape[1])
        }

        # tag → neuron mappings (used by the steering plugin)
        # characteristic_neuron_per_tag: for each tag, the neuron that best characterises it
        #   tfidf with term=tag, document=neuron → argmax over neurons for each tag
        tfidf_tn_dense = tfidf_tn.toarray() if sp.issparse(tfidf_tn) else np.asarray(tfidf_tn)
        characteristic_neuron_per_tag = {
            self.tag_ids[int(t)]: int(tfidf_tn_dense[t].argmax())
            for t in range(tfidf_tn_dense.shape[0])
        }

        # top_neuron_per_tag: for each tag, the neuron whose firing is most unique to it
        #   tfidf with term=neuron, document=tag → argmax over neurons for each tag
        tfidf_nt_dense = tfidf_nt.toarray() if sp.issparse(tfidf_nt) else np.asarray(tfidf_nt)
        top_neuron_per_tag = {
            self.tag_ids[int(t)]: int(tfidf_nt_dense[t].argmax())
            for t in range(min(tfidf_nt_dense.shape[0], len(self.tag_ids)))
        }

        # log artifacts
        with tempfile.TemporaryDirectory() as tmp:
            torch.save(item_acts, f"{tmp}/item_acts.pt")
            sp.save_npz(f"{tmp}/tag_item_prob.npz", tag_item_prob)
            np.save(f"{tmp}/tag_neuron.npy", tag_neuron)
            sp.save_npz(f"{tmp}/tfidf_tag_to_neuron.npz", tfidf_tn)
            sp.save_npz(f"{tmp}/tfidf_neuron_to_tag.npz", tfidf_nt)

            with open(f"{tmp}/neuron_labels.json", "w") as f:
                json.dump(neuron_labels, f, indent=2)

            with open(f"{tmp}/characteristic_neuron_per_tag.json", "w") as f:
                json.dump(characteristic_neuron_per_tag, f, indent=2)

            with open(f"{tmp}/top_neuron_per_tag.json", "w") as f:
                json.dump(top_neuron_per_tag, f, indent=2)

            mlflow.log_artifacts(tmp)

        mlflow.log_params(
            {
                "neuron_labeling": True,
                "num_tags": len(self.tag_ids),
                "num_neurons": item_acts.shape[1],
            }
        )
