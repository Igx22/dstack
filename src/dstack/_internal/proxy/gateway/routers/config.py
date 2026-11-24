from typing import Annotated

from fastapi import APIRouter, Depends

from dstack._internal.proxy.gateway.deps import (
    GatewayDependencyInjector,
    get_gateway_injector,
    get_gateway_proxy_repo,
)
from dstack._internal.proxy.gateway.repo.repo import GatewayProxyRepo
from dstack._internal.proxy.gateway.schemas.common import OkResponse
from dstack._internal.proxy.gateway.schemas.config import ConfigRequest

router = APIRouter()


@router.post("")
async def update_global_config(
    body: ConfigRequest,
    repo: Annotated[GatewayProxyRepo, Depends(get_gateway_proxy_repo)],
    injector: Annotated[GatewayDependencyInjector, Depends(get_gateway_injector)],
) -> OkResponse:
    async with injector.config_lock:
        config = await repo.get_config()
        acme_changes = {
            name.removeprefix("acme_"): getattr(body, name)
            for name in body.model_fields_set
            if name.startswith("acme_")
        }
        await repo.set_config(
            config.model_copy(
                update={"acme_settings": config.acme_settings.model_copy(update=acme_changes)}
            )
        )
        return OkResponse()
