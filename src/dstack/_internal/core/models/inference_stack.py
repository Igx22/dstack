from typing import Annotated

from pydantic import Field

from dstack._internal.core.models.common import FrozenCoreModel

DYNAMO_NATIVE_PROCESSOR = "dynamo"

StackFact = Annotated[str | None, Field(exclude_if=lambda value: value is None)]


class InferenceEngine(FrozenCoreModel):
    name: StackFact = None
    version: StackFact = None


class InferenceRouter(FrozenCoreModel):
    name: StackFact = None
    version: StackFact = None
    processor: StackFact = None
    """Value of the Dynamo frontend's `--dyn-chat-processor` flag, e.g. `dynamo` or `vllm`."""
    transport: StackFact = None
    """Transport from the router to the engine, e.g. `http` for SMG."""


class InferenceStack(FrozenCoreModel):
    """Private metadata sent by the server when it registers a Gateway service."""

    engine: InferenceEngine | None = Field(default=None, exclude_if=lambda value: value is None)
    router: InferenceRouter | None = Field(
        default_factory=InferenceRouter, exclude_if=lambda value: value == InferenceRouter()
    )
    """Omission leaves the routing topology unknown. Explicit null declares a direct connection."""

    def parser_facts(self) -> tuple[bool, str | None, str | None, str | None, str | None]:
        """Parser inputs that must stay stable while a registered run keeps its UUID."""
        return (
            self.router is None,
            self.engine.name if self.engine else None,
            self.router.name if self.router else None,
            self.router.processor if self.router and self.router.name == "dynamo" else None,
            self.router.transport if self.router and self.router.name == "smg" else None,
        )
