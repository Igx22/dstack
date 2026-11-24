from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from dstack._internal.core.models.common import validate_json_extra_ignore
from dstack._internal.proxy.lib.schemas.model_proxy import (
    ChatCompletionsChunk,
    ChatCompletionsRequest,
    ChatCompletionsUsage,
)
from dstack._internal.proxy.lib.schemas.usage import UsageBatch, UsageCheckpoint, UsageEntry
from dstack._internal.proxy.lib.services.usage import normalize_usage


class TestUsagePreservation:
    def test_usage_round_trip(self):
        usage = validate_json_extra_ignore(
            ChatCompletionsUsage,
            '{"prompt_tokens":20,"completion_tokens":3,"total_tokens":23,'
            '"prompt_tokens_details":{"cached_tokens":5}}',
        )
        assert usage.model_dump()["prompt_tokens_details"] == {"cached_tokens": 5}

    def test_stream_options_and_usage_only_event(self):
        request = ChatCompletionsRequest.model_validate(
            {
                "model": "test",
                "messages": [],
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        )
        assert request.model_dump(exclude_unset=True)["stream_options"] == {"include_usage": True}
        chunk = validate_json_extra_ignore(
            ChatCompletionsChunk,
            '{"model":"test","choices":[],"usage":{"prompt_tokens":20,'
            '"completion_tokens":3,"total_tokens":23}}',
        )
        assert chunk.usage is not None
        assert chunk.usage.prompt_tokens == 20
        assert "stream_options" not in ChatCompletionsRequest(
            model="test", messages=[]
        ).model_dump(exclude_unset=True)


class TestUsageNormalization:
    @pytest.mark.parametrize(
        "details,cached",
        [(None, 0), ({}, 0), ({"cached_tokens": None}, 0), ({"cached_tokens": 5}, 5)],
    )
    def test_unavailable_cache_details_preserve_final_counts(self, details, cached):
        raw = dict(
            prompt_tokens=20, completion_tokens=3, total_tokens=23, prompt_tokens_details=details
        )
        usage = normalize_usage(raw)
        assert usage is not None
        assert usage.cached_prompt_tokens == cached
        assert usage.prompt_tokens == 20

    @pytest.mark.parametrize(
        "patch",
        [
            {"prompt_tokens": -1},
            {"completion_tokens": True},
            {"total_tokens": 2},
            {"prompt_tokens": 20.0},
            {"prompt_tokens": "20"},
            {"prompt_tokens_details": {"cached_tokens": 21}},
            {"prompt_tokens_details": {"cached_tokens": -1}},
            {"prompt_tokens_details": {"cached_tokens": False}},
            {"prompt_tokens_details": {"cached_tokens": "5"}},
            {"prompt_tokens_details": []},
        ],
    )
    def test_invalid_usage_is_unknown(self, patch):
        raw = dict(prompt_tokens=20, completion_tokens=3, total_tokens=23)
        assert normalize_usage(raw | patch) is None
        assert normalize_usage(None) is None


class TestUsageContracts:
    def test_checkpoint_rejects_duplicate_keys_and_ids(self):
        entry = UsageEntry(
            consumer_subject_id="u",
            service_id="s",
            prompt_tokens=20,
            cached_prompt_tokens=5,
            completion_tokens=3,
            total_tokens=23,
        )
        start = datetime(2026, 11, 16, 10, tzinfo=timezone.utc)
        batch = UsageBatch(
            batch_id="a",
            interval_start=start,
            interval_end=start + timedelta(seconds=60),
            entries=[entry],
        )
        with pytest.raises(ValidationError):
            UsageBatch.model_validate(batch.model_dump() | {"entries": [entry, entry]})
        with pytest.raises(ValidationError):
            UsageCheckpoint(active_batch=batch, pending_batches=[batch])
        with pytest.raises(ValidationError):
            UsageBatch.model_validate(
                batch.model_dump() | {"interval_end": start + timedelta(seconds=30)}
            )

    @pytest.mark.parametrize("value", [-1, True, 1.0, "1"])
    def test_rejects_invalid_persisted_counter(self, value):
        with pytest.raises(ValidationError):
            UsageEntry(
                consumer_subject_id="u", service_id="s", prompt_tokens=value, total_tokens=value
            )
