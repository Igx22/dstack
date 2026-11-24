"""Fetch durable minute batches, commit usage rows, then ACK over SSH."""

import asyncio
import json
import sqlite3
import uuid
from collections import Counter
from contextlib import asynccontextmanager, nullcontext

import httpx
from pydantic import JsonValue, TypeAdapter, ValidationError
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from dstack._internal.proxy.lib.schemas.usage import Identifier, UsageBatch
from dstack._internal.server.db import get_db, get_session_ctx
from dstack._internal.server.models import GatewayModel, GatewayReplicaModel
from dstack._internal.server.services.gateways.connection import GatewayConnection
from dstack._internal.server.services.usage import InvalidUsageBatch, ingest_usage_batch
from dstack._internal.utils.logging import get_logger

logger = get_logger(__name__)
PAGE_SIZE = 100
MAX_PAGES = 10
COLLECTION_TIMEOUT = 12
_sqlite_ingestion_lock = asyncio.Lock()
_batch_identifier = TypeAdapter(Identifier)


async def collect_gateway_usage(
    connection: GatewayConnection, replica_id: uuid.UUID, lock_token: uuid.UUID
) -> None:
    try:
        async with asyncio.timeout(COLLECTION_TIMEOUT):
            async with get_session_ctx() as session:
                source = await _get_source(session, connection.ip_address, replica_id, lock_token)
            if source is not None:
                await _collect_pages(connection, source, lock_token)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code not in (404, 405):
            logger.warning("Usage collection HTTP failure from %s: %s", replica_id, exc)
    except Exception:
        logger.exception(
            "Usage collection failed from %s; unacknowledged batches retained", replica_id
        )


async def drain_gateway_usage(
    connection: GatewayConnection,
    replica_id: uuid.UUID,
    gateway_id: uuid.UUID,
    lock_token: uuid.UUID,
) -> bool:
    """Only a sealed, empty queue from the retiring replica permits termination."""
    source = replica_id, gateway_id
    try:
        async with asyncio.timeout(COLLECTION_TIMEOUT):
            async with get_session_ctx() as session:
                if (
                    await _get_source(session, connection.ip_address, replica_id, lock_token)
                    != source
                ):
                    return False
            try:
                async with connection.client() as client:
                    state = await client.drain_usage()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in (404, 405, 503):
                    # An old or interrupted drain can still have published, retryable minutes.
                    await _collect_pages(connection, source, lock_token)
                raise
            if state.empty:
                return True
            # An open stream may outlive many minutes. Keep delivering closed pages.
            caught_up = await _collect_pages(connection, source, lock_token)
            if not state.sealed or not caught_up:
                return False
            async with connection.client() as client:
                state = await client.drain_usage()
            return state.sealed and state.empty
    except Exception as exc:
        logger.warning("Gateway replica %s drain incomplete, will retry: %r", replica_id, exc)
        return False


async def _collect_pages(
    connection: GatewayConnection,
    source: tuple[uuid.UUID, uuid.UUID | None],
    lock_token: uuid.UUID,
) -> bool:
    replica_id, gateway_id = source
    for _ in range(MAX_PAGES):
        async with connection.client() as client:
            page = await client.fetch_usage_batches(PAGE_SIZE)
        if not page.batches:
            return not page.has_more
        batch_ids = [_read_batch_id(payload) for payload in page.batches]
        counts = Counter(batch_ids)
        acknowledged: list[str] = []
        rejected: list[tuple[str | None, JsonValue, str]] = []
        async with _ingestion_session() as session:
            # The lease or source may have changed while GET was in flight.
            current = await _get_source(
                session, connection.ip_address, replica_id, lock_token, for_update=True
            )
            if current != source:
                return False
            for batch_id, payload in zip(batch_ids, page.batches):
                if batch_id is not None and counts[batch_id] > 1:
                    reason = "Duplicate batch ID in the same page"
                else:
                    reason = await _ingest_received_batch(session, replica_id, gateway_id, payload)
                if reason is not None:
                    rejected.append((batch_id, payload, reason))
                    # An unbound replica may only ACK an exact replay.
                    if gateway_id is None:
                        continue
                if batch_id is not None and counts[batch_id] == 1:
                    acknowledged.append(batch_id)
        # The page has committed. Log every rejected payload before allowing its removal.
        for batch_id, payload, reason in rejected:
            logger.error(
                "Could not insert usage batch %r from Gateway %s replica %s: %s. Batch: %s",
                batch_id,
                gateway_id,
                replica_id,
                reason,
                json.dumps(payload, separators=(",", ":")),
            )
        if not acknowledged:
            return False
        async with connection.client() as client:
            await client.acknowledge_usage(acknowledged)
        if not page.has_more:
            return len(acknowledged) == len(page.batches)
    return False


def _read_batch_id(payload: JsonValue) -> str | None:
    if isinstance(payload, dict):
        try:
            return _batch_identifier.validate_python(payload.get("batch_id"))
        except ValidationError:
            pass
    return None


async def _ingest_received_batch(
    session: AsyncSession, replica_id: uuid.UUID, gateway_id: uuid.UUID | None, payload: JsonValue
) -> str | None:
    try:
        batch = UsageBatch.model_validate(payload)
        # A bad later row must undo this batch's earlier rows.
        async with session.begin_nested():
            await ingest_usage_batch(session, replica_id, gateway_id, batch)
    except (ValidationError, InvalidUsageBatch) as exc:
        return str(exc)
    except DBAPIError as exc:
        # asyncpg exposes some data exceptions as generic DBAPIError; use the driver code.
        sqlstate = getattr(exc.orig, "sqlstate", "") or ""
        sqlite_code = getattr(exc.orig, "sqlite_errorcode", None)
        if exc.connection_invalidated:
            raise
        if sqlstate == "23503" or sqlite_code == sqlite3.SQLITE_CONSTRAINT_FOREIGNKEY:
            # usage_batches has one foreign key: user_id references users.id.
            return "Referenced user does not exist"
        sqlite_data_errors = (sqlite3.SQLITE_CONSTRAINT_CHECK, sqlite3.SQLITE_TOOBIG)
        if sqlstate.startswith("22") or sqlstate == "23514" or sqlite_code in sqlite_data_errors:
            return str(exc.orig)
        raise
    return None


async def _get_source(
    session: AsyncSession,
    address: str,
    replica_id: uuid.UUID,
    lock_token: uuid.UUID,
    *,
    for_update: bool = False,
) -> tuple[uuid.UUID, uuid.UUID | None] | None:
    # Verify the SSH endpoint still belongs to the claimed replica before ingesting its usage.
    stmt = (
        select(
            GatewayReplicaModel.id,
            GatewayReplicaModel.lock_token,
            GatewayReplicaModel.gateway_id,
            GatewayModel.id,
        )
        .outerjoin(GatewayModel, GatewayModel.gateway_replica_id == GatewayReplicaModel.id)
        .where(GatewayReplicaModel.active == True, GatewayReplicaModel.ip_address == address)
        .limit(2)
    )
    if for_update:
        stmt = stmt.with_for_update(key_share=True, of=GatewayReplicaModel)
    rows = (await session.execute(stmt)).all()
    if len(rows) != 1:
        return None
    source_id, token, parent, legacy_parent = rows[0]
    if source_id != replica_id or token != lock_token:
        return None
    if parent is not None and legacy_parent is not None and parent != legacy_parent:
        return None
    # An unbound replica can only replay an already committed batch.
    return source_id, parent or legacy_parent


@asynccontextmanager
async def _ingestion_session():
    sqlite = get_db().dialect_name == "sqlite"
    async with _sqlite_ingestion_lock if sqlite else nullcontext():
        async with get_session_ctx() as session:
            if sqlite:
                # SQLite ignores FOR UPDATE. Reserve the writer before reading the
                # token so a new claim cannot slip between the check and the insert.
                await session.execute(text("BEGIN IMMEDIATE"))
            yield session
