from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_pipelines import router as pipelines_router
from app.api.routes_plugins import router as plugins_router

app = FastAPI(title="Application for Recommender Systems with SAEs Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pipelines_router, prefix="/pipelines", tags=["Pipelines"])
app.include_router(plugins_router, prefix="/plugins", tags=["Plugins"])
