import time
import random
from datetime import datetime

from plugins.plugin_interface import BasePlugin

class Plugin(BasePlugin):
    """
    Mock Training plugin — simulates running ELSA and SAE training without real data.
    Intended for testing the pipeline and API interfaces.
    """

    def run(self, context, dataset="LastFM1k", epochs=5, embedding_dim=512, top_k=128, **kwargs):
        start_time = datetime.now()
        print(f"[MockTrainer] Starting training pipeline with dataset={dataset}, epochs={epochs}")

        # simulating step 1: Loading dataset
        time.sleep(1)
        context["dataset"] = {
            "name": dataset,
            "users": 1000,
            "items": 5000,
            "split": {"train": 0.8, "valid": 0.1, "test": 0.1}
        }
        print(f"[MockTrainer] Dataset {dataset} loaded successfully")

        # simulating step 2: Training ELSA model
        print(f"[MockTrainer] Training ELSA model...")
        time.sleep(1.5)
        context["elsa_model"] = {
            "type": "ELSA",
            "embedding_dim": embedding_dim,
            "trained_epochs": epochs,
            "metrics": {
                "loss": round(random.uniform(0.2, 0.4), 4),
                "R20": round(random.uniform(0.3, 0.6), 3),
                "NDCG20": round(random.uniform(0.25, 0.55), 3)
            }
        }

        # simulating step 3: Training SAE model
        print(f"[MockTrainer] Training SAE model (TopK={top_k})...")
        time.sleep(1.5)
        context["sae_model"] = {
            "type": "TopKSAE",
            "embedding_dim": embedding_dim * 2,
            "top_k": top_k,
            "trained_epochs": epochs * 2,
            "metrics": {
                "CosineSim": round(random.uniform(0.7, 0.95), 3),
                "NDCG20_Degradation": round(random.uniform(0.01, 0.05), 3)
            }
        }

        # simulating result
        context["training_summary"] = {
            "dataset": dataset,
            "duration_sec": (datetime.now() - start_time).total_seconds(),
            "timestamp": datetime.now().isoformat(),
            "notes": "Mock training completed successfully"
        }

        print(f"[MockTrainer] Training completed in {context['training_summary']['duration_sec']:.2f}s")
        return context
