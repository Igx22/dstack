import asyncio

import httpx
from sqlalchemy import select

from dstack._internal.core.errors import SSHError
from dstack._internal.server.db import get_db, get_session_ctx
from dstack._internal.server.models import (
    GatewayReplicaModel,
)
from dstack._internal.server.services.gateways import (
    GatewayConnection,
    gateway_connections_pool,
    init_gateways,
)
from dstack._internal.server.services.locking import advisory_lock_ctx
from dstack._internal.utils.logging import get_logger

logger = get_logger(__name__)


async def init_gateways_in_background():
    async with get_session_ctx() as session:
        await init_gateways(session=session)


async def process_gateways_connections():
    await _remove_inactive_connections()
    await _process_active_connections()


async def _remove_inactive_connections():
    connections = await gateway_connections_pool.all()
    if not connections:
        return
    async with get_session_ctx() as session:
        res = await session.scalars(
            select(GatewayReplicaModel.id).where(GatewayReplicaModel.active == True)
        )
        active_replica_ids = set(res)
    for conn in connections:
        if conn.replica_id not in active_replica_ids:
            await gateway_connections_pool.remove(conn.replica_id)


async def _process_active_connections():
    connections = await gateway_connections_pool.all()
    # Two server processes on a single host cannot process
    # gateway connections and init gateway connections concurrently:
    # Race conditions cause conflicting tunnels being opened.
    async with get_session_ctx() as session:
        async with advisory_lock_ctx(
            bind=session,
            dialect_name=get_db().dialect_name,
            resource="gateway_tunnels",
        ):
            await asyncio.gather(*(_process_connection(conn) for conn in connections))


async def _process_connection(conn: GatewayConnection):
    try:
        await conn.check_or_restart()
    except SSHError as e:
        logger.warning("Connection to gateway %s failed: %s", conn.ip_address, e)
        return

    try:
        await conn.try_collect_stats()
    except httpx.HTTPError as e:
        logger.warning("Failed to collect stats from gateway %s: %r", conn.ip_address, e)
    except Exception:
        logger.exception("Failed to collect stats from gateway %s", conn.ip_address)
