from fastapi import FastAPI
from app.api.routes_pipelines import router as pipelines_router

app = FastAPI(title="Application for Recommender Systems with SAEs Service")
app.include_router(pipelines_router, prefix="/pipelines", tags=["Pipelines"])

