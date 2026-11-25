import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock
from uuid import UUID

import httpx
import httpx2
import pytest
import pytest_asyncio

from dstack._internal.proxy.gateway.app import make_app
from dstack._internal.proxy.gateway.deps import get_gateway_injector_from_app
from dstack._internal.proxy.gateway.repo.repo import GatewayProxyRepo
from dstack._internal.proxy.gateway.services.metering_store import MeteringStore
from dstack._internal.proxy.lib.schemas.model_proxy import ChatCompletionsRequest
from dstack._internal.proxy.lib.schemas.usage import TokenUsage
from dstack._internal.proxy.lib.services.model_proxy.clients.tgi import TGIChatCompletions
from dstack._internal.proxy.lib.services.usage import UsageAttempt
from tests._internal.proxy.gateway.routers.test_registry import (
    register_replica_payload,
    register_service_payload,
)
from tests._internal.proxy.gateway.services.test_metering import FakeClock

pytestmark = pytest.mark.asyncio


class NativeStream(httpx.AsyncByteStream):
    def __init__(self, events, *, read_error=False):
        self.data = "".join(
            "data: " + json.dumps(event, ensure_ascii=False) + "\r\n\r\n" for event in events
        ).encode()
        self.read_error = read_error

    async def __aiter__(self):
        # Split two-byte é, JSON, and CRLF across HTTP reads.
        for byte in self.data:
            yield bytes([byte])
        if self.read_error:
            raise httpx.ReadError("upstream reset")


def text_event(text):
    return {"token": {"text": text, "special": False}, "details": None}


def final_event(text="!", *, reason="length", special=False, counts=None):
    return {
        "token": {"text": text, "special": special},
        "generated_text": "café!" if reason == "length" else "café",
        "details": {
            "finish_reason": reason,
            **({"input_length": 3, "generated_tokens": 2} if counts is None else counts),
        },
    }


def sse(*events, read_error=False):
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        stream=NativeStream(events, read_error=read_error),
    )


async def test_final_text_without_metering():
    def upstream(request):
        assert request.url.path == "/generate_stream"
        return sse(text_event("café"), final_event())

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(upstream), base_url="http://tgi"
    ) as http:
        adapter = TGIChatCompletions(http, "prompt", "</s>")
        request = ChatCompletionsRequest(model="test", messages=[], stream=True)
        chunks = [chunk.model_dump() async for chunk in adapter.stream(request)]

    choices = [choice for chunk in chunks for choice in chunk["choices"]]
    assert "".join(choice["delta"].get("content", "") for choice in choices) == "café!"
    assert [choice["finish_reason"] for choice in choices] == [None, "length"]
    assert choices[-1]["delta"]["content"] == "!"


USER_ID = UUID(int=2).hex


async def register_tgi(client):
    model = {
        "type": "chat",
        "name": "test",
        "format": "tgi",
        "chat_template": "prompt",
        "eos_token": "</s>",
    }
    registration = register_service_payload(run_name="run", options={"openai": {"model": model}})
    registered = await client.post("/api/registry/p/services/register", json=registration)
    registered.raise_for_status()
    replica = await client.post(
        "/api/registry/p/services/run/replicas/register", json=register_replica_payload()
    )
    replica.raise_for_status()
    return registration["id"]


@pytest_asyncio.fixture
async def tgi_gateway(tmp_path, monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(
        "dstack._internal.proxy.gateway.services.registry.apply_service",
        AsyncMock(return_value={}),
    )

    def authorize(request):
        assert request.headers["authorization"] == "Bearer token"
        if request.url.path == "/api/projects/p/get":
            return httpx2.Response(200, json={})
        assert request.url.path == "/api/projects/p/authorize_proxy"
        return httpx2.Response(200, json={"consumer_subject_id": USER_ID})

    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(authorize), base_url="http://server"
    ) as server:
        monkeypatch.setattr(
            "dstack._internal.proxy.gateway.app.HTTPMultiClient", lambda *_: server
        )
        app = make_app(
            repo=GatewayProxyRepo(),
            metering_store=MeteringStore(tmp_path / "usage.json"),
            clock=clock,
        )
        injector = get_gateway_injector_from_app(app)
        collector = injector.get_metering()
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://gateway",
                headers={"Authorization": "Bearer token"},
            ) as client:
                run_id = await register_tgi(client)
                yield client, injector, run_id, clock
        finally:
            await collector.close()


@pytest.fixture
def gateway_request(tgi_gateway, monkeypatch):
    """Send the first request on a freshly registered route and verify its minute entry."""
    client, injector, run_id, clock = tgi_gateway

    @asynccontextmanager
    async def send(response, *, batch, stream=True, include=True, stop=None):
        calls = []

        def upstream(request):
            calls.append((request.method, request.url.path))
            body = json.loads(request.content)
            assert "stream_options" not in body
            assert body["parameters"]["decoder_input_details"] is (not stream)
            if stop is not None:
                assert stop in body["parameters"]["stop"]
            return response

        payload = {"model": "test", "messages": [], "stream": stream}
        if stream and include is not None:
            payload["stream_options"] = {"include_usage": include}
        if not stream:
            payload["n"] = 2
        if stop is not None:
            payload["stop"] = stop
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(upstream), base_url="http://tgi"
        ) as backend:
            monkeypatch.setattr(
                injector, "get_service_replica_client", AsyncMock(return_value=backend)
            )
            reply = await client.post("/api/models/p/chat/completions", json=payload)
        assert reply.status_code == 200
        yield reply
        assert calls == [("POST", "/generate_stream" if stream else "/generate")]
        clock.advance()
        await injector.get_metering().rotate_if_due()
        result = await client.get("/api/usage/batches")
        result.raise_for_status()
        (minute,) = result.json()["batches"]
        prompt, completion, unknown = batch
        assert minute["entries"] == [
            {
                "consumer_subject_id": USER_ID,
                "service_id": run_id,
                "prompt_tokens": prompt,
                "cached_prompt_tokens": 0,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
                "unknown_usage_attempts": unknown,
            }
        ]

    return send


def usage(prompt, completion):
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": None,
    }


def text_choice(text):
    return {
        "delta": {"content": text, "role": "assistant"},
        "index": 0,
        "finish_reason": None,
        "logprobs": None,
    }


def final_choice(reason="length", text=None):
    return {
        "delta": {} if text is None else {"content": text},
        "index": 0,
        "finish_reason": reason,
        "logprobs": None,
    }


def assert_stream(
    response, choices, text, *, include=True, final_usage=None, terminal="data: [DONE]"
):
    assert response.headers["content-type"].startswith("text/event-stream")
    *frames, last, blank = response.text.split("\n\n")
    assert (last, blank) == (terminal, "")
    assert all(frame.startswith("data:") for frame in frames)
    events = [json.loads(frame.removeprefix("data:")) for frame in frames]
    observed_choices = [c for event in events for c in event["choices"]]
    assert "".join(c["delta"].get("content", "") for c in observed_choices) == text
    assert observed_choices == choices

    first = events[0]
    assert UUID(first["id"]).version == 4
    assert type(first["created"]) is int
    common = {
        "id": first["id"],
        "created": first["created"],
        "model": "test",
        "object": "chat.completion.chunk",
        "system_fingerprint": "",
    }
    visible = {"usage": None} if include else {}
    expected = [{**common, "choices": [choice], **visible} for choice in choices]
    if include and final_usage is not None:
        expected.append({**common, "choices": [], "usage": final_usage})
    assert events == expected


@pytest.mark.parametrize(
    "include", [True, False, None], ids=["include-usage", "hide-usage", "default"]
)
async def test_fragmented_stream(gateway_request, include):
    final = {"details": {"finish_reason": "length", "input_length": 2, "generated_tokens": 3}}
    upstream = sse(text_event("A café"), text_event("!"), final)
    async with gateway_request(upstream, batch=(2, 3, 0), include=include) as reply:
        assert_stream(
            reply,
            [text_choice("A café"), text_choice("!"), final_choice()],
            "A café!",
            include=include,
            final_usage=usage(2, 3),
        )


@pytest.mark.parametrize(
    "include", [True, False, None], ids=["include-usage", "hide-usage", "default"]
)
async def test_final_length_token(gateway_request, include):
    async with gateway_request(
        sse(text_event("café"), final_event()), batch=(3, 2, 0), include=include
    ) as reply:
        assert_stream(
            reply,
            [text_choice("café"), final_choice(text="!")],
            "café!",
            include=include,
            final_usage=usage(3, 2),
        )


@pytest.mark.parametrize(
    "reason,token,special,stop",
    [("eos_token", "</s>", True, None), ("stop_sequence", "###", False, "###")],
)
async def test_final_stop_token(gateway_request, reason, token, special, stop):
    upstream = sse(text_event("café"), final_event(token, reason=reason, special=special))
    async with gateway_request(upstream, batch=(3, 2, 0), stop=stop) as reply:
        assert_stream(
            reply, [text_choice("café"), final_choice("stop")], "café", final_usage=usage(3, 2)
        )


async def test_final_text_with_partial_usage(gateway_request):
    upstream = sse(text_event("café"), final_event(counts={"generated_tokens": 2}))
    async with gateway_request(upstream, batch=(0, 2, 1)) as reply:
        assert_stream(reply, [text_choice("café"), final_choice(text="!")], "café!")


@pytest.mark.parametrize(
    "counts,batch",
    [
        pytest.param({"generated_tokens": 3}, (0, 3, 1), id="missing-input"),
        pytest.param(
            {"input_length": 2, "generated_tokens": True}, (0, 0, 1), id="boolean-output"
        ),
    ],
)
async def test_incomplete_final_counts(gateway_request, counts, batch):
    upstream = sse(text_event("A café"), {"details": {"finish_reason": "length", **counts}})
    async with gateway_request(upstream, batch=batch) as reply:
        assert_stream(reply, [text_choice("A café"), final_choice()], "A café")


async def test_zero_output(gateway_request):
    upstream = sse(
        {"details": {"finish_reason": "length", "input_length": 2, "generated_tokens": 0}}
    )
    async with gateway_request(upstream, batch=(2, 0, 0)) as reply:
        assert_stream(reply, [final_choice()], "", final_usage=usage(2, 0))


@pytest.mark.parametrize(
    "ending,terminal",
    [
        ("eof", "data: [DONE]"),
        ("read-error", ": \"Error requesting model: ReadError('upstream reset')\""),
        ("tgi-error", ": 'generation failed'"),
    ],
)
async def test_interrupted_stream(gateway_request, ending, terminal):
    events = [text_event("A café"), text_event("!")]
    if ending == "tgi-error":
        events.append({"error": "generation failed"})
    async with gateway_request(
        sse(*events, read_error=ending == "read-error"), batch=(0, 0, 1)
    ) as reply:
        assert_stream(
            reply, [text_choice("A café"), text_choice("!")], "A café!", terminal=terminal
        )


@pytest.mark.parametrize(
    "counts,partial",
    [
        pytest.param({"generated_tokens": 3, "input_length": None}, True, id="null-input"),
        pytest.param({"generated_tokens": 3, "input_length": "2"}, True, id="string-input"),
        pytest.param({"generated_tokens": 3, "input_length": -1}, True, id="negative-input"),
        pytest.param({"input_length": 2}, False, id="missing-output"),
        pytest.param({"input_length": 2, "generated_tokens": None}, False, id="null-output"),
        pytest.param({"input_length": 2, "generated_tokens": -1}, False, id="negative-output"),
        pytest.param({"input_length": 2, "generated_tokens": "3"}, False, id="string-output"),
        pytest.param({"input_length": 2, "generated_tokens": 1.5}, False, id="fractional-output"),
    ],
)
async def test_adapter_rejects_invalid_counts(counts, partial):
    def respond(request):
        return sse(text_event("A café"), {"details": {"finish_reason": "length", **counts}})

    attempt = Mock(spec=UsageAttempt)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), base_url="http://tgi"
    ) as http:
        adapter = TGIChatCompletions(http, "prompt", "</s>")
        request = ChatCompletionsRequest.model_validate(
            {
                "model": "test",
                "messages": [],
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        )
        chunks = [c async for c in adapter.stream(request, attempt=attempt)]
    assert [[choice.model_dump() for choice in chunk.choices] for chunk in chunks] == [
        [text_choice("A café")],
        [final_choice()],
    ]
    assert all(chunk.usage is None for chunk in chunks)
    attempt.mark_dispatched.assert_called_once_with()
    attempt.mark_upstream_done.assert_called_once_with()
    attempt.observe_usage.assert_not_called()
    if partial:
        attempt.observe_partial_usage.assert_called_once_with(
            TokenUsage(completion_tokens=3, total_tokens=3)
        )
    else:
        attempt.observe_partial_usage.assert_not_called()


@pytest.mark.parametrize(
    "texts,prefill,outputs,batch,public_usage",
    [
        pytest.param(
            ("first", "second"), [{}, {}], (3, 5), (2, 8, 0), usage(2, 8), id="shared-prompt"
        ),
        pytest.param(("A", "B"), None, (4, 2), (0, 6, 1), None, id="null-prefill"),
        pytest.param(("A", "B"), [{}, {}, {}], ("4", 2), (0, 0, 1), None, id="string-output"),
    ],
)
async def test_json_sequences(gateway_request, texts, prefill, outputs, batch, public_usage):
    upstream = httpx.Response(
        200,
        json={
            "generated_text": texts[0],
            "details": {
                "seed": 1,
                "finish_reason": "length",
                "prefill": prefill,
                "generated_tokens": outputs[0],
                "best_of_sequences": [
                    {
                        "generated_text": texts[1],
                        "finish_reason": "eos_token",
                        "generated_tokens": outputs[1],
                    }
                ],
            },
        },
    )
    async with gateway_request(upstream, stream=False, batch=batch) as reply:
        body = reply.json()
        assert UUID(body.pop("id")).version == 4
        assert type(body.pop("created")) is int
        assert body == {
            "model": "test",
            "object": "chat.completion",
            "system_fingerprint": "fp_1",
            "choices": [
                {
                    "index": i,
                    "message": {"role": "assistant", "content": t},
                    "finish_reason": reason,
                }
                for i, (t, reason) in enumerate(zip(texts, ("length", "stop")))
            ],
            "usage": public_usage,
        }
