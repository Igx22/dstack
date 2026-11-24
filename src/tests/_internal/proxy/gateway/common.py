import uuid
from typing import Optional

from dstack._internal.core.models.inference_stack import InferenceStack
from dstack._internal.proxy.gateway.models import GatewayService
from dstack._internal.proxy.lib.models import Replica


def make_service(
    project_name: str,
    run_name: str,
    domain: Optional[str] = None,
    https: Optional[bool] = None,
    auth: bool = False,
    strip_prefix: bool = True,
    run_id: Optional[str] = None,
    inference_stack: InferenceStack | None = None,
) -> GatewayService:
    if run_id is None:
        run_id = uuid.uuid4().hex
    return GatewayService(
        id=run_id,
        inference_stack=inference_stack,
        project_name=project_name,
        run_name=run_name,
        domain=domain,
        https=https,
        auth=auth,
        client_max_body_size=2**20,
        strip_prefix=strip_prefix,
        replicas=(
            Replica(
                id="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                app_port=80,
                ssh_destination="ubuntu@server",
                ssh_port=22,
                ssh_proxy=None,
            ),
        ),
    )
