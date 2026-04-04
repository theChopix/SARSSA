"""SAE-based steering plugin for recommender systems.

This plugin evaluates how steering individual SAE neurons (or concepts mapped
to neurons) influences recommendation behaviour.  It supports three modes:

* **single neuron** – steer all test users toward a single specified neuron.
* **multiple neurons** – steer all test users toward several neurons at once.
* **concept (batch evaluation)** – for each test user, automatically select
  the most-desired concept and steer toward the corresponding neuron.
  This mode is evaluated using *both* concept→neuron mapping strategies
  produced by the neuron-labeling step:
    - *characteristic*: neuron that best characterises the concept.
    - *top*: neuron whose firing is most unique to the concept.

The plugin loads artifacts from previous pipeline steps (dataset, base model,
SAE, neuron labeling) and sweeps over a configurable range of steering
strengths ``alpha ∈ [0, 1]``, logging Recall@K and NDCG@K for both
*personal* (held-out) and *segment* (concept-related) targets.
"""

import json
import tempfile

import mlflow
import numpy as np
import scipy.sparse as sp
import torch

from plugins.plugin_interface import BasePlugin
from utils.data_loading.data_loader import DataLoader
from utils.mlflow_manager import MLflowRunLoader
from utils.plugin_logger import get_logger
from utils.torch.evalution import (
    evaluate_steering,
    split_input_target_interactions,
)
from utils.torch.models.model_loader import load_base_model, load_sae_model
from utils.torch.models.steered_model import SteeredModel
from utils.torch.runtime import set_device, set_seed

logger = get_logger(__name__)


class Plugin(BasePlugin):
    """SAE steering evaluation plugin.

    Expects prior ``dataset_loading``, ``training_cfm``, ``training_sae``,
    and ``neuron_labeling`` steps in the pipeline context.
    """

    # Artifact loading

    def _load_artifacts(self, context: dict, device):
        """Load all required artifacts from previous pipeline steps."""

        #  dataset artifacts
        dataset_run_id = context["dataset_loading"]["run_id"]
        dataset_loader = MLflowRunLoader(dataset_run_id)
        logger.info(f"Loading dataset artifacts from run {dataset_run_id}")

        train_npz = dataset_loader.get_npz_artifact("train_csr.npz")
        test_npz = dataset_loader.get_npz_artifact("test_csr.npz")
        self.train_csr: sp.csr_matrix = (
            sp.csr_matrix(train_npz) if not isinstance(train_npz, sp.csr_matrix) else train_npz
        )
        self.test_csr: sp.csr_matrix = (
            sp.csr_matrix(test_npz) if not isinstance(test_npz, sp.csr_matrix) else test_npz
        )

        dataset_params = dataset_loader.get_parameters()
        self.dataset = dataset_params["dataset_name"]
        self.num_items = int(dataset_params["num_items"])

        # tag data (needed for concept-based steering evaluation)
        tag_ids_data = dataset_loader.get_json_artifact("tag_ids.json")
        self.tag_ids = list(tag_ids_data) if isinstance(tag_ids_data, dict) else tag_ids_data
        tag_item_matrix = dataset_loader.get_npz_artifact("tag_item_matrix.npz")
        self.tag_item_counts: sp.csr_matrix = (
            sp.csr_matrix(tag_item_matrix)
            if not isinstance(tag_item_matrix, sp.csr_matrix)
            else tag_item_matrix
        )

        logger.info(
            f"Dataset: {self.dataset} | "
            f"Train: {self.train_csr.shape}, Test: {self.test_csr.shape} | "
            f"Tags: {len(self.tag_ids)}"
        )

        #  base model
        base_run_id = context["training_cfm"]["run_id"]
        base_loader = MLflowRunLoader(base_run_id)
        logger.info("Loading base model")
        self.base_model = load_base_model(base_loader.get_artifact_path(), device)
        logger.info("Base model loaded successfully")

        #  SAE model
        sae_run_id = context["training_sae"]["run_id"]
        sae_loader = MLflowRunLoader(sae_run_id)
        logger.info("Loading SAE model")
        self.sae = load_sae_model(sae_loader.get_artifact_path(), device)
        logger.info("SAE model loaded successfully")

        #  neuron labeling artifacts
        labeling_run_id = context["neuron_labeling"]["run_id"]
        labeling_loader = MLflowRunLoader(labeling_run_id)
        logger.info(f"Loading neuron labeling artifacts from run {labeling_run_id}")

        self.neuron_labels: dict = labeling_loader.get_json_artifact("neuron_labels.json")

        # concept → neuron mappings (both strategies)
        self.characteristic_neuron_per_tag: dict = labeling_loader.get_json_artifact(
            "characteristic_neuron_per_tag.json"
        )
        self.top_neuron_per_tag: dict = labeling_loader.get_json_artifact("top_neuron_per_tag.json")

        logger.info(
            f"Loaded {len(self.neuron_labels)} neuron labels, "
            f"{len(self.characteristic_neuron_per_tag)} characteristic mappings, "
            f"{len(self.top_neuron_per_tag)} top-neuron mappings"
        )

    # Helpers

    def _build_concept_neuron_tensor(self, mapping: dict[str, int], device) -> torch.Tensor:
        """Convert a tag-name → neuron-id dict to an index-aligned tensor.

        ``self.tag_ids`` defines the canonical tag ordering.  The returned
        tensor has shape ``(num_tags,)`` and maps each tag index to a neuron.
        """
        result = torch.zeros(len(self.tag_ids), dtype=torch.long, device=device)
        for idx, tag in enumerate(self.tag_ids):
            result[idx] = mapping.get(tag, 0)
        return result

    # Steering evaluation modes

    def _evaluate_concept_steering(
        self,
        alphas: np.ndarray,
        test_inputs: sp.csr_matrix,
        test_targets: sp.csr_matrix,
        device,
        k: int,
        batch_size: int,
    ) -> dict:
        """Run concept-based steering evaluation for both mapping strategies.

        Sweeps over ``alphas`` and logs metrics for each mapping strategy.

        Returns:
            dict keyed by ``"characteristic"`` and ``"top"`` with per-alpha results.
        """
        char_mapping = self._build_concept_neuron_tensor(self.characteristic_neuron_per_tag, device)
        top_mapping = self._build_concept_neuron_tensor(self.top_neuron_per_tag, device)

        results: dict = {"characteristic": {}, "top": {}}

        for alpha in alphas:
            a = float(alpha)
            logger.info(f"Evaluating concept steering | alpha={a:.3f} | map=characteristic")
            model = SteeredModel(self.base_model, self.sae, alpha=a)
            model.eval()
            res = evaluate_steering(
                model,
                char_mapping,
                test_inputs,
                test_targets,
                self.tag_item_counts,
                device,
                k=k,
                batch_size=batch_size,
            )
            results["characteristic"][a] = res
            self._log_metrics(res, a, "characteristic")

            logger.info(f"Evaluating concept steering | alpha={a:.3f} | map=top")
            model_top = SteeredModel(self.base_model, self.sae, alpha=a)
            model_top.eval()
            res_top = evaluate_steering(
                model_top,
                top_mapping,
                test_inputs,
                test_targets,
                self.tag_item_counts,
                device,
                k=k,
                batch_size=batch_size,
            )
            results["top"][a] = res_top
            self._log_metrics(res_top, a, "top")

        return results

    def _evaluate_single_neuron_steering(
        self,
        neuron_id: int,
        alphas: np.ndarray,
        test_inputs: sp.csr_matrix,
        test_targets: sp.csr_matrix,
        device,
        k: int,
        batch_size: int,
    ) -> dict:
        """Evaluate steering toward a single specified neuron across alphas.

        Returns:
            dict keyed by alpha with personal recall/ndcg metrics.
        """
        results: dict = {}

        for alpha in alphas:
            a = float(alpha)
            logger.info(f"Evaluating single-neuron steering | neuron={neuron_id} | alpha={a:.3f}")
            model = SteeredModel(self.base_model, self.sae, alpha=a)
            model.eval()

            all_recalls = []
            all_ndcgs = []

            input_dl_iter = DataLoader(test_inputs, batch_size, device)
            target_dl_iter = DataLoader(test_targets, batch_size, device)
            for input_batch, target_batch in zip(input_dl_iter, target_dl_iter):
                _, topk_indices = model.recommend_single_neuron(
                    input_batch,
                    neuron_id,
                    k=k,
                    mask_interactions=True,
                )
                topk_t = torch.tensor(topk_indices, device=target_batch.device)
                # Recall
                target_bool = target_batch.bool()
                predicted = torch.zeros_like(target_bool).scatter(
                    1, topk_t, torch.ones_like(topk_t, dtype=torch.bool)
                )
                r = (predicted & target_bool).sum(dim=1) / torch.minimum(
                    target_bool.sum(dim=1),
                    torch.ones_like(target_bool.sum(dim=1)) * k,
                )
                all_recalls.append(r.cpu().numpy())
                # NDCG
                relevance = target_bool.gather(1, topk_t).float()
                gains = 2**relevance - 1
                discounts = torch.log2(
                    torch.arange(2, k + 2, device=target_batch.device, dtype=torch.float)
                )
                dcg = (gains / discounts).sum(dim=1)
                sorted_rel, _ = torch.sort(target_batch.float(), dim=1, descending=True)
                ideal_gains = 2 ** sorted_rel[:, :k] - 1
                idcg = (ideal_gains / discounts).sum(dim=1)
                idcg[idcg == 0] = 1
                all_ndcgs.append((dcg / idcg).cpu().numpy())

            recalls = np.concatenate(all_recalls)
            ndcgs = np.concatenate(all_ndcgs)

            results[a] = {
                f"recall@{k}(personal)": {
                    "mean": float(np.mean(recalls)),
                    "se": float(np.std(recalls) / np.sqrt(len(recalls))),
                },
                f"ndcg@{k}(personal)": {
                    "mean": float(np.mean(ndcgs)),
                    "se": float(np.std(ndcgs) / np.sqrt(len(ndcgs))),
                },
            }
            self._log_metrics(results[a], a, f"single_n{neuron_id}")

        return results

    def _evaluate_multi_neuron_steering(
        self,
        neuron_ids: list[int],
        alphas: np.ndarray,
        test_inputs: sp.csr_matrix,
        test_targets: sp.csr_matrix,
        device,
        k: int,
        batch_size: int,
    ) -> dict:
        """Evaluate steering toward multiple specified neurons across alphas.

        Returns:
            dict keyed by alpha with personal recall/ndcg metrics.
        """
        results: dict = {}
        neuron_label = "_".join(str(n) for n in neuron_ids)

        for alpha in alphas:
            a = float(alpha)
            logger.info(f"Evaluating multi-neuron steering | neurons={neuron_ids} | alpha={a:.3f}")
            model = SteeredModel(self.base_model, self.sae, alpha=a)
            model.eval()

            all_recalls = []
            all_ndcgs = []

            input_dl_iter = DataLoader(test_inputs, batch_size, device)
            target_dl_iter = DataLoader(test_targets, batch_size, device)
            for input_batch, target_batch in zip(input_dl_iter, target_dl_iter):
                _, topk_indices = model.recommend_multiple_neurons(
                    input_batch,
                    neuron_ids,
                    k=k,
                    mask_interactions=True,
                )
                topk_t = torch.tensor(topk_indices, device=target_batch.device)
                # Recall
                target_bool = target_batch.bool()
                predicted = torch.zeros_like(target_bool).scatter(
                    1, topk_t, torch.ones_like(topk_t, dtype=torch.bool)
                )
                r = (predicted & target_bool).sum(dim=1) / torch.minimum(
                    target_bool.sum(dim=1),
                    torch.ones_like(target_bool.sum(dim=1)) * k,
                )
                all_recalls.append(r.cpu().numpy())
                # NDCG
                relevance = target_bool.gather(1, topk_t).float()
                gains = 2**relevance - 1
                discounts = torch.log2(
                    torch.arange(2, k + 2, device=target_batch.device, dtype=torch.float)
                )
                dcg = (gains / discounts).sum(dim=1)
                sorted_rel, _ = torch.sort(target_batch.float(), dim=1, descending=True)
                ideal_gains = 2 ** sorted_rel[:, :k] - 1
                idcg = (ideal_gains / discounts).sum(dim=1)
                idcg[idcg == 0] = 1
                all_ndcgs.append((dcg / idcg).cpu().numpy())

            recalls = np.concatenate(all_recalls)
            ndcgs = np.concatenate(all_ndcgs)

            results[a] = {
                f"recall@{k}(personal)": {
                    "mean": float(np.mean(recalls)),
                    "se": float(np.std(recalls) / np.sqrt(len(recalls))),
                },
                f"ndcg@{k}(personal)": {
                    "mean": float(np.mean(ndcgs)),
                    "se": float(np.std(ndcgs) / np.sqrt(len(ndcgs))),
                },
            }
            self._log_metrics(results[a], a, f"multi_{neuron_label}")

        return results

    # MLflow logging helpers

    @staticmethod
    def _log_metrics(metrics: dict, alpha: float, prefix: str):
        """Log a flat dict of steering metrics to MLflow."""
        step = int(round(alpha * 1000))  # use millionths as step for nice ordering
        for metric_name, value in metrics.items():
            if isinstance(value, dict):
                mlflow.log_metric(f"{prefix}/{metric_name}/mean", value["mean"], step=step)
                mlflow.log_metric(f"{prefix}/{metric_name}/se", value["se"], step=step)
            else:
                mlflow.log_metric(f"{prefix}/{metric_name}", value, step=step)

    # Plugin entry point

    def run(
        self,
        context: dict,
        target_ratio: float = 0.2,
        k: int = 20,
        batch_size: int = 1024,
        alpha_steps: int = 6,
        alpha_max: float = 0.5,
        neuron_id: int | None = None,
        neuron_ids: list[int] | None = None,
        seed: int = 42,
    ):
        """Execute the steering evaluation pipeline.

        The plugin always runs the **concept-based** evaluation (both mapping
        strategies, across the alpha sweep).  Additionally, if ``neuron_id``
        or ``neuron_ids`` are provided, single-neuron and/or multi-neuron
        evaluations are also performed.

        Args:
            context: Pipeline context with run IDs from previous steps.
                Required keys: ``dataset_loading``, ``training_cfm``,
                ``training_sae``, ``neuron_labeling``.
            target_ratio: Fraction of test interactions held out as targets.
            k: Top-K for evaluation metrics.
            batch_size: Batch size for inference.
            alpha_steps: Number of alpha values to evaluate (linearly spaced
                from 0 to ``alpha_max``).
            alpha_max: Maximum steering strength.
            neuron_id: Optional single neuron index for single-neuron steering.
            neuron_ids: Optional list of neuron indices for multi-neuron steering.
            seed: Random seed for reproducibility.
        """
        device = set_device()
        logger.info(f"Using device: {device}")
        set_seed(seed)

        self._load_artifacts(context, device)

        # Split test interactions into inputs and targets
        test_inputs, test_targets = split_input_target_interactions(
            self.test_csr, target_ratio, seed
        )
        logger.info(f"Test split: {test_inputs.shape[0]} users, target_ratio={target_ratio}")

        alphas = np.linspace(0, alpha_max, alpha_steps)
        logger.info(f"Alpha sweep: {alphas}")

        # Log parameters
        mlflow.log_params(
            {
                "dataset": self.dataset,
                "target_ratio": target_ratio,
                "k": k,
                "batch_size": batch_size,
                "alpha_steps": alpha_steps,
                "alpha_max": alpha_max,
                "seed": seed,
                "neuron_id": neuron_id,
                "neuron_ids": str(neuron_ids) if neuron_ids else None,
            }
        )

        all_results: dict = {}

        #  Concept-based steering (always runs)
        logger.info("Concept-based steering evaluation")
        concept_results = self._evaluate_concept_steering(
            alphas,
            test_inputs,
            test_targets,
            device,
            k,
            batch_size,
        )
        all_results["concept"] = concept_results

        #  Single neuron steering (optional)
        if neuron_id is not None:
            logger.info(f"Single-neuron steering evaluation (neuron {neuron_id})")
            single_results = self._evaluate_single_neuron_steering(
                neuron_id,
                alphas,
                test_inputs,
                test_targets,
                device,
                k,
                batch_size,
            )
            all_results["single_neuron"] = {neuron_id: single_results}

        #  Multi-neuron steering (optional)
        if neuron_ids is not None and len(neuron_ids) > 0:
            logger.info(f"Multi-neuron steering evaluation (neurons {neuron_ids})")
            multi_results = self._evaluate_multi_neuron_steering(
                neuron_ids,
                alphas,
                test_inputs,
                test_targets,
                device,
                k,
                batch_size,
            )
            all_results["multi_neuron"] = {str(neuron_ids): multi_results}

        #  Save full results as JSON artifact
        with tempfile.TemporaryDirectory() as tmp:
            results_path = f"{tmp}/steering_results.json"
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            mlflow.log_artifacts(tmp)

        logger.info("Steering evaluation complete — results logged to MLflow")
