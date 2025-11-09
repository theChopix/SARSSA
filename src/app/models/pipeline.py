from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class StepDefinition(BaseModel):
    plugin: str
    params: Optional[Dict[str, Any]] = {}

class PipelineRequest(BaseModel):
    steps: List[StepDefinition]