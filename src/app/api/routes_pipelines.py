from fastapi import APIRouter, HTTPException

from app.core.pipeline_engine import PipelineEngine
from app.core.pipeline_runs import get_pipeline_runs
from app.models.pipeline import PipelineRequest, StepDefinition

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


@router.post("/{run_id}/execute-step")
def execute_step(run_id: str, step: StepDefinition):
    try:
        result = PipelineEngine.execute_step(run_id, step.model_dump())
        return {"message": "Step executed", "result": result}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Run or context not found: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
