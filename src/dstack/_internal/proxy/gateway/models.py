"""Things stored in GatewayProxyRepo in addition to those from BaseProxyRepo."""

from functools import cached_property
from hashlib import sha256
from typing import Any, Optional
from uuid import uuid4

from pydantic import AnyHttpUrl, Field, PrivateAttr

from dstack._internal.core.models.inference_stack import DYNAMO_NATIVE_PROCESSOR, InferenceStack
from dstack._internal.proxy.lib.const import DEFAULT_SERVICE_READ_TIMEOUT
from dstack._internal.proxy.lib.models import ImmutableModel, Service
from dstack._internal.proxy.lib.services.usage import UsageRule


class GatewayService(Service):
    inference_stack: InferenceStack | None = Field(default=None, exclude_if=lambda v: v is None)
    """Stack facts received at registration and restored from Gateway state."""
    _usage_rule: UsageRule = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        self._usage_rule = _select_usage_rule(self.inference_stack)

    @property
    def usage_rule(self) -> UsageRule:
        return self._usage_rule

    @cached_property
    def nginx_route_key(self) -> str:
        """Match loopback dispatch to the selected registration across Nginx reloads.

        A digest also bounds and escapes opaque legacy IDs for the Nginx configuration.
        For missing IDs, preserve the provisional key across replica updates and ID recovery.
        """
        return sha256((self.id or uuid4().hex).encode()).hexdigest()


def _select_usage_rule(stack: InferenceStack | None) -> UsageRule:
    if stack is None or stack.engine is None or stack.router is None:
        return UsageRule.FINAL_ONLY
    engine = stack.engine.name
    router = stack.router
    if router.name == "smg" and router.transport == "http" and engine == "vllm":
        return UsageRule.PER_CHOICE
    if router.name == "dynamo" and engine in ("vllm", "sglang"):
        if router.processor == DYNAMO_NATIVE_PROCESSOR:
            return UsageRule.REQUEST_WIDE
        if router.processor == "vllm":
            return UsageRule.REQUEST_WIDE_INLINE_FINAL
    return UsageRule.FINAL_ONLY


class ModelEntrypoint(ImmutableModel):
    project_name: str
    domain: str
    https: bool
    read_timeout: int = DEFAULT_SERVICE_READ_TIMEOUT


class ACMESettings(ImmutableModel):
    server: Optional[AnyHttpUrl] = None
    eab_kid: Optional[str] = None
    eab_hmac_key: Optional[str] = None


class GlobalProxyConfig(ImmutableModel):
    acme_settings: ACMESettings = ACMESettings()
