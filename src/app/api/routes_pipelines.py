from fastapi import APIRouter
from app.models.pipeline import PipelineRequest
from app.core.pipeline_engine import PipelineEngine

router = APIRouter()

@router.post("/run")
def run_pipeline(pipeline_request: PipelineRequest):
    steps = [step.model_dump() for step in pipeline_request.steps]

    context = {}
    engine = PipelineEngine(steps)
    result = engine.run(context)
    
    return {"message": "Pipeline finished", "result": result}
