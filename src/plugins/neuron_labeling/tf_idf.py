import json
import tempfile
import numpy as np
import scipy.sparse as sp
import torch
import mlflow
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfTransformer

from utils.torch.models.elsa import ELSA
from utils.torch.models.sae import BasicSAE, TopKSAE, BatchTopKSAE
from utils.plugin_logger import get_logger
from plugins.plugin_interface import BasePlugin
from utils.torch.runtime import set_device, set_seed
from utils.torch.checkpointing import load_checkpoint
from utils.mlflow_manager import MLflowRunLoader

logger = get_logger(__name__)
device = set_device()


@torch.no_grad()
def compute_sae_item_activations(
    elsa,
    sae,
    num_items,
    batch_size=1024,
    device="cpu",
):
    elsa.eval().to(device)
    sae.eval().to(device)

    eye = torch.eye(num_items, device=device)
    activations = []

    for i in tqdm(range(0, num_items, batch_size), desc="Computing SAE item activations"):
        batch = eye[i : i + batch_size]
        dense = elsa.encode(batch)
        e, *_ = sae.encode(dense)
        activations.append(e.cpu())

    return torch.cat(activations)  # (items × neurons)


class Plugin(BasePlugin):
    def _load_artifacts(self, context, device, sae_model):
        """Load dataset, ELSA, and SAE artifacts from previous pipeline steps."""
        # Load dataset artifacts
        dataset_run_id = context['dataset_loading']['run_id']
        dataset_loader = MLflowRunLoader(dataset_run_id)

        logger.info(f'Loading dataset artifacts from run {dataset_run_id}')

        self.items = dataset_loader.get_npy_artifact('items.npy', allow_pickle=True)
        self.num_items = len(self.items)
        self.tag_ids = dataset_loader.get_json_artifact('tag_ids.json')
        self.tag_item_counts = dataset_loader.get_npz_artifact('tag_item_matrix.npz')

        if self.tag_ids is None or self.tag_item_counts is None:
            raise RuntimeError("Dataset does not support neuron labeling (no tag data available)")

        # Load ELSA model
        logger.info("Loading ELSA model")
        elsa_run_id = context['training_cfm']['run_id']
        elsa_loader = MLflowRunLoader(elsa_run_id)
        elsa_params = elsa_loader.get_parameters()

        self.elsa = ELSA(
            input_dim=int(elsa_params["items"]),
            embedding_dim=int(elsa_params["factors"]),
        )
        elsa_opt = torch.optim.Adam(self.elsa.parameters())
        load_checkpoint(
            self.elsa,
            elsa_opt,
            elsa_loader.get_artifact_path("checkpoint.ckpt"),
            device,
        )
        self.elsa.to(device).eval()

        # Load SAE model
        logger.info("Loading SAE model")
        sae_run_id = context["training_sae"]['run_id']
        sae_loader = MLflowRunLoader(sae_run_id)
        sae_params = sae_loader.get_parameters()

        cfg = {
            "reconstruction_loss": sae_params["reconstruction_loss"],
            "k": int(sae_params["top_k"]),
            "device": device,
            "normalize": sae_params["normalize"] == "True",
            "auxiliary_coef": float(sae_params["auxiliary_coef"]),
            "contrastive_coef": float(sae_params["contrastive_coef"]),
            "l1_coef": float(sae_params["l1_coef"]),
            "reconstruction_coef": float(sae_params["reconstruction_coef"]),
        }

        if sae_model == "BasicSAE":
            self.sae = BasicSAE(
                int(elsa_params["factors"]),
                int(sae_params["embedding_dim"]),
                cfg,
            )
        elif sae_model == "TopKSAE":
            self.sae = TopKSAE(
                int(elsa_params["factors"]),
                int(sae_params["embedding_dim"]),
                cfg,
            )
        elif sae_model == "BatchTopKSAE":
            self.sae = BatchTopKSAE(
                int(elsa_params["factors"]),
                int(sae_params["embedding_dim"]),
                cfg,
            )
        else:
            raise ValueError(f"SAE model {sae_model} not supported")

        sae_opt = torch.optim.Adam(self.sae.parameters())
        load_checkpoint(
            self.sae,
            sae_opt,
            sae_loader.get_artifact_path("checkpoint.ckpt"),
            device,
        )
        self.sae.to(device).eval()

    def run(self,
            context: dict,

            batch_size: int = 1024,
            seed: int = 42,
            sae_model: str = "TopKSAE",   # BasicSAE / TopKSAE / BatchTopKSAE
    ):
        set_seed(seed)

        self._load_artifacts(context, device, sae_model)

        # compute SAE activations
        item_acts = compute_sae_item_activations(
            self.elsa,
            self.sae,
            self.num_items,
            batch_size=batch_size,
            device=device,
        )

        # build tag–item probability matrix
        tag_item_prob = self.tag_item_counts.multiply(
            1.0 / self.tag_item_counts.sum(axis=1)
        )

        # aggregate tag → neuron
        tag_neuron = tag_item_prob @ item_acts.numpy()

        # TF-IDF
        tfidf = TfidfTransformer(norm=None)
        tfidf_tn = tfidf.fit_transform(tag_neuron)
        tfidf_nt = tfidf.fit_transform(tag_neuron.T).T

        neuron_labels = {
            int(n): self.tag_ids[int(tfidf_nt[:, n].argmax())]
            for n in range(tfidf_nt.shape[1])
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

            mlflow.log_artifacts(tmp)

        mlflow.log_params({
            "neuron_labeling": True,
            "num_tags": len(self.tag_ids),
            "num_neurons": item_acts.shape[1],
        })

        # context update
        context["neuron_labeling"] = {
            "status": "completed",
        }

        return context
