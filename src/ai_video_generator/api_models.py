from pydantic import BaseModel, ConfigDict

from ai_video_generator.domain import ChainSpec, ConditioningStack


class DryRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chain: ChainSpec
    conditioning_stack: ConditioningStack
