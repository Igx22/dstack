import asyncio
import json

import httpx
import pytest

from dstack._internal.proxy.gateway.app import make_app
from dstack._internal.proxy.gateway.repo.repo import GatewayProxyRepo
from dstack._internal.proxy.gateway.services.metering_store import MeteringStore


@pytest.mark.asyncio
async def test_legacy_profile_map_is_discarded_on_normal_config_save(tmp_path):
    state = tmp_path / "state-v2.json"
    state.write_text(json.dumps({"config": {"usage_metering_profiles": {"run": "retired"}}}))
    repo = GatewayProxyRepo.load(state)
    app = make_app(repo=repo, metering_store=MeteringStore(tmp_path / "usage.json"))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://control"
    ) as client:
        (await client.post("/api/config", json={"acme_eab_kid": "kid"})).raise_for_status()
    assert (await repo.get_config()).acme_settings.eab_kid == "kid"
    assert "usage_metering_profiles" not in json.loads(state.read_text())["config"]


@pytest.mark.asyncio
async def test_concurrent_acme_updates_preserve_both_fields(tmp_path):
    repo = GatewayProxyRepo()
    app = make_app(repo=repo, metering_store=MeteringStore(tmp_path / "usage.json"))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://control"
    ) as client:
        replies = await asyncio.gather(
            client.post("/api/config", json={"acme_eab_kid": "new-kid"}),
            client.post("/api/config", json={"acme_eab_hmac_key": "new-key"}),
        )
        for reply in replies:
            reply.raise_for_status()
    acme = (await repo.get_config()).acme_settings
    assert acme.eab_kid == "new-kid"
    assert acme.eab_hmac_key == "new-key"
