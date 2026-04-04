from typing import Any

from pydantic import BaseModel


class StepDefinition(BaseModel):
    plugin: str
    params: dict[str, Any] | None = {}


class PipelineRequest(BaseModel):
    steps: list[StepDefinition]
