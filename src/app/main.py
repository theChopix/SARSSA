"""FastAPI application entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mlflow.exceptions import MlflowException

from app.api.routes_items import router as items_router
from app.api.routes_pipelines import router as pipelines_router
from app.api.routes_plugins import router as plugins_router
from app.config.config import ARTIFACT_ROOT, EXPERIMENT_NAME, TRACKING_URI
from app.core.run_recovery import fail_orphaned_runs

mlflow.set_tracking_uri(TRACKING_URI)

# SQLite needs the DB's parent directory to exist (fresh clones).
if TRACKING_URI.startswith("sqlite:///"):
    Path(TRACKING_URI.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)

# ensuring the experiment exists before any run starts.
#  Via the MLflow server we omit artifact_location so
#   the experiment gets a portable mlflow-artifacts:/ root
#    (proxied by the server); the direct-SQLite
#  fallback keeps the configured local path.
try:
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        if TRACKING_URI.startswith("http"):
            mlflow.create_experiment(EXPERIMENT_NAME)
        else:
            mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=ARTIFACT_ROOT)
except MlflowException as exc:
    raise RuntimeError(
        f"MLflow at {TRACKING_URI} is unreachable ({exc}). "
        "If running locally, start it first with `just mlflow`."
    ) from exc


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Sweep zombie MLflow runs around the server lifecycle: at startup
    (runs a previous hard kill stranded as RUNNING) and at graceful
    shutdown (the run this process is abandoning)."""
    fail_orphaned_runs("orphaned_at_startup")
    yield
    fail_orphaned_runs("terminated_at_shutdown")


app = FastAPI(
    title="Application for Recommender Systems with SAEs Service",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(items_router, prefix="/items", tags=["Items"])
app.include_router(pipelines_router, prefix="/pipelines", tags=["Pipelines"])
app.include_router(plugins_router, prefix="/plugins", tags=["Plugins"])
