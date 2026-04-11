"""FastAPI application entry point."""

import mlflow
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_pipelines import router as pipelines_router
from app.api.routes_plugins import router as plugins_router
from app.config.config import ARTIFACT_ROOT, EXPERIMENT_NAME, TRACKING_URI

mlflow.set_tracking_uri(TRACKING_URI)

# Ensure the experiment exists with the correct artifact storage path.
# Without this, mlflow.set_experiment() (called later in pipeline_engine)
# would auto-create the experiment with the default artifact location
# (./mlruns/<id>), ignoring our configured ARTIFACT_ROOT.
if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
    mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=ARTIFACT_ROOT)

app = FastAPI(title="Application for Recommender Systems with SAEs Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pipelines_router, prefix="/pipelines", tags=["Pipelines"])
app.include_router(plugins_router, prefix="/plugins", tags=["Plugins"])
