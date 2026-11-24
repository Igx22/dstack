import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import pytest

from dstack._internal.proxy.gateway.app import lifespan, make_app
from dstack._internal.proxy.gateway.deps import (
    GatewayDependencyInjector,
    get_gateway_injector_from_app,
)
from dstack._internal.proxy.gateway.repo.repo import GatewayProxyRepo
from dstack._internal.proxy.gateway.services.metering_store import MeteringStore
from dstack._internal.proxy.gateway.services.nginx import Nginx
from dstack._internal.proxy.lib.deps import get_proxy_auth_provider
from tests._internal.proxy.gateway.routers.test_registry import (
    register_replica_payload,
    register_service_payload,
    sample_model_options,
)
from tests._internal.proxy.gateway.services.test_metering import FakeClock
from tests._internal.proxy.lib.services.test_openai_usage import (
    IdentityProvider,
    public_events,
    sse_choice,
    sse_reply,
    token_counts,
)


@asynccontextmanager
async def gateway(tmp_path, clock):
    app = make_app(
        repo=GatewayProxyRepo.load(tmp_path / "state-v2.json"),
        nginx=Nginx(nginx_dir=tmp_path),
        metering_store=MeteringStore(tmp_path / "usage-counters.json"),
        clock=clock,
    )
    app.dependency_overrides[get_proxy_auth_provider] = IdentityProvider
    async with (
        lifespan(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://gateway",
            headers={"Authorization": "Bearer token"},
        ) as client,
    ):
        yield client, get_gateway_injector_from_app(app).get_metering()


async def register(client, name, stack=None):
    payload = register_service_payload(
        run_name=name, domain=f"{name}.test", options=sample_model_options(name)
    )
    if stack is not None:
        payload["inference_stack"] = stack
    resp = await client.post("/api/registry/p/services/register", json=payload)
    resp.raise_for_status()
    resp = await client.post(
        f"/api/registry/p/services/{name}/replicas/register",
        json=register_replica_payload(payload["id"]),
    )
    resp.raise_for_status()
    return payload["id"]


async def chat(client, name):
    reply = await client.post(
        "/api/models/p/chat/completions",
        json={
            "model": name,
            "messages": [],
            "stream": True,
            "n": 2,
            "stream_options": {"include_usage": True},
        },
    )
    events = public_events(reply)
    return [event["usage"]["completion_tokens"] for event in events if event.get("usage")]


async def batches(client):
    resp = await client.get("/api/usage/batches")
    resp.raise_for_status()
    return resp.json()["batches"]


@pytest.mark.asyncio
async def test_routes_and_queued_usage_survive_upgrade_and_run_name_reuse(
    tmp_path, system_mocks, monkeypatch
):
    (tmp_path / "sites-enabled").mkdir()
    clock = FakeClock()
    options = []

    def upstream(request):
        body = json.loads(request.content)
        options.append((body["model"], body["stream_options"]))
        return sse_reply(
            [
                sse_choice(0, usage=token_counts(12, 3)),
                sse_choice(1, usage=token_counts(12, 5)),
                sse_choice(0, finish_reason="stop", usage=token_counts(12, 6)),
                sse_choice(1, finish_reason="stop", usage=token_counts(12, 10)),
            ]
        )

    smg = {"engine": {"name": "vllm"}, "router": {"name": "smg", "transport": "http"}}
    dynamo = {"engine": {"name": "vllm"}, "router": {"name": "dynamo", "processor": "vllm"}}
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(upstream), base_url="http://model"
    ) as model:

        async def connection(*args, **kwargs):
            return model

        monkeypatch.setattr(GatewayDependencyInjector, "get_service_replica_client", connection)
        async with gateway(tmp_path, clock) as (client, collector):
            old_id = await register(client, "model-a", smg)
            legacy_id = await register(client, "model-b")
            assert await chat(client, "model-a") == [16]
            clock.advance()
            await collector.rotate_if_due()
            (pending,) = await batches(client)
            assert pending["entries"][0]["service_id"] == old_id
            assert pending["entries"][0]["completion_tokens"] == 16

        path = tmp_path / "state-v2.json"
        state = json.loads(path.read_text())
        assert state["services"]["p"]["model-a"]["inference_stack"] == smg
        state["services"]["p"]["model-b"].pop("inference_stack", None)
        state["config"]["usage_metering_profiles"] = {legacy_id: "vllm-0.19.1-per-choice"}
        path.write_text(json.dumps(state))

        async with gateway(tmp_path, clock) as (client, collector):
            assert await batches(client) == [pending]
            options.clear()
            assert await asyncio.gather(chat(client, "model-a"), chat(client, "model-b")) == (
                [[16], []]
            )
            assert dict(options) == {
                "model-a": {"include_usage": True, "continuous_usage_stats": True},
                "model-b": {"include_usage": True},
            }
            resp = await client.post("/api/registry/p/services/model-a/unregister")
            resp.raise_for_status()
            new_id = await register(client, "model-a", dynamo)
            assert new_id != old_id
            assert await chat(client, "model-a") == [10]
            clock.advance()
            await collector.rotate_if_due()
            original, latest = await batches(client)
            assert original == pending
            assert {
                entry["service_id"]: (entry["completion_tokens"], entry["unknown_usage_attempts"])
                for entry in latest["entries"]
            } == {old_id: (16, 0), legacy_id: (0, 1), new_id: (10, 0)}
        assert "usage_metering_profiles" not in json.loads(path.read_text())["config"]
