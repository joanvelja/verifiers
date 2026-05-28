from asyncio import Future
from enum import Enum
from typing import Annotated, Literal, TypeAlias, TypeVar, cast

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    PlainValidator,
)

from verifiers.types import (
    ClientConfig,
    GenerationPlan,
    MemberGenerationPlan,
    RolloutInput,
    RolloutOutput,
    SamplingArgs,
)
from verifiers.utils.message_utils import normalize_messages

CoercedRolloutOutput = Annotated[
    RolloutOutput, BeforeValidator(lambda v: RolloutOutput(v))
]


def _coerce_rollout_input(value: object) -> RolloutInput:
    if not isinstance(value, dict):
        raise TypeError(f"RolloutInput must be a dict, got {type(value).__name__}")
    if "prompt" not in value:
        raise ValueError("RolloutInput.prompt is required")
    if "example_id" not in value:
        raise ValueError("RolloutInput.example_id is required")

    input_value = dict(value)
    input_value["prompt"] = normalize_messages(
        input_value["prompt"], field_name="input.prompt"
    )
    return cast(RolloutInput, input_value)


def _coerce_group_input(value: object) -> list[RolloutInput]:
    if not isinstance(value, list):
        raise TypeError(f"group_inputs must be a list, got {type(value).__name__}")
    return [_coerce_rollout_input(item) for item in value]


RunInput: TypeAlias = Annotated[RolloutInput, PlainValidator(_coerce_rollout_input)]
GroupInput: TypeAlias = Annotated[
    list[RolloutInput], PlainValidator(_coerce_group_input)
]


class BaseRequest(BaseModel):
    # needed for RolloutInput to work
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_type: str


class BaseResponse(BaseModel):
    # needed for RolloutOutput to work
    model_config = ConfigDict(arbitrary_types_allowed=True)

    success: bool = True
    error: str | None = None  # TODO: type errors later


class HealthRequest(BaseRequest):
    request_type: Literal["health"] = "health"


class HealthResponse(BaseResponse): ...


class RunRolloutRequest(BaseRequest):
    request_type: Literal["run_rollout"] = "run_rollout"

    # Pydantic rejects some provider message shapes at this protocol boundary.
    input: RunInput
    client_config: ClientConfig
    model: str
    sampling_args: SamplingArgs
    max_retries: int
    state_columns: list[str] | None
    generation: MemberGenerationPlan | None = None


class RunRolloutResponse(BaseResponse):
    output: CoercedRolloutOutput | None = None


class RunGroupRequest(BaseRequest):
    request_type: Literal["run_group"] = "run_group"

    # Pydantic rejects some provider message shapes at this protocol boundary.
    group_inputs: GroupInput
    client_config: ClientConfig
    model: str
    sampling_args: SamplingArgs
    max_retries: int
    state_columns: list[str] | None
    generation: GenerationPlan | None = None


class RunGroupResponse(BaseResponse):
    outputs: list[CoercedRolloutOutput] | None = None


BaseRequestT = TypeVar("BaseRequestT", bound=BaseRequest)
BaseResponseT = TypeVar("BaseResponseT", bound=BaseResponse)


class ServerState(str, Enum):
    STARTUP = "startup"  # Initial state, before first successful health check
    HEALTHY = "healthy"  # Server is responsive and working normally
    UNHEALTHY = "unhealthy"  # Server failed health checks


class ServerError(RuntimeError): ...


class PendingRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_id: str
    request: BaseRequest
    future: Future[dict]
    timeout: float | None = None
    submitted_at: float  # timestamp
