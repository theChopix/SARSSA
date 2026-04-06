from fastapi import APIRouter, HTTPException

from app.core.pipeline_engine import PipelineEngine
from app.core.pipeline_runs import get_pipeline_runs
from app.models.pipeline import PipelineRequest

router = APIRouter()


@router.get("/runs")
def list_pipeline_runs():
    try:
        return get_pipeline_runs()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/run")
def run_pipeline(context: dict, pipeline_request: PipelineRequest):
    steps = [step.model_dump() for step in pipeline_request.steps]

    engine = PipelineEngine(steps)
    result = engine.run(context)

    return {"message": "Pipeline finished", "result": result}
