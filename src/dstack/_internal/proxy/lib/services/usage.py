"""Usage observation is optional for shared proxy consumers."""

from collections.abc import Mapping
from enum import Enum, auto
from typing import Any

from pydantic import BaseModel, ValidationError

from dstack._internal.proxy.lib.models import Service
from dstack._internal.proxy.lib.schemas.usage import TokenUsage, UsageKey

MAX_JSON_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_SSE_EVENT_BYTES = 1024 * 1024


class UsageRule(Enum):
    """How an OpenAI-compatible endpoint reports token counts for a stream."""

    FINAL_ONLY = auto()
    PER_CHOICE = auto()
    REQUEST_WIDE = auto()
    REQUEST_WIDE_INLINE_FINAL = auto()


def normalize_usage(usage: Any) -> TokenUsage | None:
    if isinstance(usage, BaseModel):
        usage = usage.model_dump()
    if not isinstance(usage, Mapping):
        return None
    details = usage.get("prompt_tokens_details")
    if details is not None and not isinstance(details, Mapping):
        return None
    cached = details.get("cached_tokens") if details is not None else None
    try:
        return TokenUsage.model_validate(
            {
                "prompt_tokens": usage.get("prompt_tokens"),
                "cached_prompt_tokens": 0 if cached is None else cached,
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }
        )
    except ValidationError:
        return None


class UsageAttempt:
    """No-op default; Gateway supplies a concrete per-attempt observer."""

    rule: UsageRule = UsageRule.FINAL_ONLY

    def observe_usage(self, usage: Any) -> None:
        pass

    def observe_partial_usage(self, usage: TokenUsage | None) -> None:
        pass

    def mark_dispatched(self) -> None:
        pass

    def mark_upstream_done(self) -> None:
        pass

    async def finish(self) -> None:
        pass


class UsageObserver:
    def start_service_attempt(self, service: Service, key: UsageKey) -> UsageAttempt:
        return UsageAttempt()
