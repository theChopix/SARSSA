from fastapi import APIRouter

from app.core.pipeline_engine import PipelineEngine
from app.models.pipeline import PipelineRequest

router = APIRouter()


@router.post("/run")
def run_pipeline(context: dict, pipeline_request: PipelineRequest):
    steps = [step.model_dump() for step in pipeline_request.steps]

    engine = PipelineEngine(steps)
    result = engine.run(context)

    return {"message": "Pipeline finished", "result": result}
