from fastapi import APIRouter
from app.core.pipeline_engine import PipelineEngine

router = APIRouter()

@router.post("/run")
def run_pipeline():
    pipeline_definition = [
        {"plugin": "training.dummy_trainer"},
        {"plugin": "evaluation.dummy_evaluator"},
    ]
    context = {}
    engine = PipelineEngine(pipeline_definition)
    result = engine.run(context)
    return {"message": "Pipeline finished", "result": result}
