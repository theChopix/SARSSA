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
    def run(self,
            context: dict,

            batch_size: int = 1024,
            seed: int = 42,
            sae_model: str = "TopKSAE",   # BasicSAE / TopKSAE / BatchTopKSAE
    ):
        set_seed(seed)

        # resolve previous run IDs
        base_run_id = context["training_sae"]['run_id']  # SAE run
        sae_run = mlflow.get_run(base_run_id)
        sae_params = sae_run.data.params

        base_model_run_id = context['training_cfm']['run_id']  # ELSA run
        elsa_run = mlflow.get_run(base_model_run_id)

        # load dataset artifacts from the dataset_loading pipeline step
        dataset_run_id = context['dataset_loading']['run_id']
        dataset_run = mlflow.get_run(dataset_run_id)
        dataset_artifact_uri = dataset_run.info.artifact_uri
        if 'mlruns' in dataset_artifact_uri:
            dataset_artifact_uri = './' + dataset_artifact_uri[dataset_artifact_uri.find('mlruns'):]

        logger.info(f'Loading dataset artifacts from run {dataset_run_id}')

        items = np.load(f'{dataset_artifact_uri}/items.npy', allow_pickle=True)
        num_items = len(items)

        with open(f'{dataset_artifact_uri}/tag_ids.json', 'r') as f:
            tag_ids = json.load(f)

        tag_item_counts = sp.load_npz(f'{dataset_artifact_uri}/tag_item_matrix.npz')

        if tag_ids is None or tag_item_counts is None:
            raise RuntimeError("Dataset does not support neuron labeling (no tag data available)")

        # load ELSA model
        logger.info("Loading ELSA model")
        elsa = ELSA(
            input_dim=int(elsa_run.data.params["items"]),
            embedding_dim=int(elsa_run.data.params["factors"]),
        )
        elsa_opt = torch.optim.Adam(elsa.parameters())
        elsa_artifact_path = elsa_run.info.artifact_uri
        elsa_artifact_path = './' + elsa_artifact_path[elsa_artifact_path.find('mlruns'):]
        load_checkpoint(
            elsa,
            elsa_opt,
            f"{elsa_artifact_path}/checkpoint.ckpt",
            device,
        )
        elsa.to(device).eval()

        # load SAE model
        logger.info("Loading SAE model")

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
            sae = BasicSAE(
                int(elsa_run.data.params["factors"]),
                int(sae_params["embedding_dim"]),
                cfg,
            )
        elif sae_model == "TopKSAE":
            sae = TopKSAE(
                int(elsa_run.data.params["factors"]),
                int(sae_params["embedding_dim"]),
                cfg,
            )
        elif sae_model == "BatchTopKSAE":
            sae = BatchTopKSAE(
                int(elsa_run.data.params["factors"]),
                int(sae_params["embedding_dim"]),
                cfg,
            )
        else:
            raise ValueError(f"SAE model {sae_model} not supported")

        sae_opt = torch.optim.Adam(sae.parameters())
        sae_artifact_path = sae_run.info.artifact_uri
        sae_artifact_path = './' + sae_artifact_path[sae_artifact_path.find('mlruns'):]
        load_checkpoint(
            sae,
            sae_opt,
            f"{sae_artifact_path}/checkpoint.ckpt",
            device,
        )
        sae.to(device).eval()

        # compute SAE activations
        item_acts = compute_sae_item_activations(
            elsa,
            sae,
            num_items,
            batch_size=batch_size,
            device=device,
        )

        # build tag–item probability matrix
        tag_item_prob = tag_item_counts.multiply(
            1.0 / tag_item_counts.sum(axis=1)
        )

        # aggregate tag → neuron
        tag_neuron = tag_item_prob @ item_acts.numpy()

        # TF-IDF
        tfidf = TfidfTransformer(norm=None)
        tfidf_tn = tfidf.fit_transform(tag_neuron)
        tfidf_nt = tfidf.fit_transform(tag_neuron.T).T

        neuron_labels = {
            int(n): tag_ids[int(tfidf_nt[:, n].argmax())]
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

            mlflow.log_artifacts(tmp, artifact_path="neuron_labeling")

        mlflow.log_params({
            "neuron_labeling": True,
            "num_tags": len(tag_ids),
            "num_neurons": item_acts.shape[1],
        })

        # context update
        context["neuron_labeling"] = {
            "status": "completed",
            "artifact_path": "neuron_labeling",
        }

        return context
