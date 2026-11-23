import asyncio
from uuid import UUID
from weakref import WeakValueDictionary

from dstack._internal.server.services.gateways.connection import GatewayConnection
from dstack._internal.server.settings import SERVER_PORT


class GatewayConnectionsPool:
    def __init__(self) -> None:
        self._connections: dict[UUID, GatewayConnection] = {}
        self._locks: WeakValueDictionary[UUID, asyncio.Lock] = WeakValueDictionary()

    async def get_or_add(
        self,
        replica_id: UUID,
        hostname: str,
        id_rsa: str,
        close_existing_tunnel: bool = False,
    ) -> GatewayConnection:
        async with self._locks.setdefault(replica_id, asyncio.Lock()):
            cached = self._connections.get(replica_id)
            if cached is not None:
                if cached.ip_address == hostname:
                    return cached
                del self._connections[replica_id]
                await cached.close()
            connection = GatewayConnection(replica_id, hostname, id_rsa, SERVER_PORT)
            try:
                await connection.open(close_existing_tunnel=close_existing_tunnel)
            except (Exception, asyncio.CancelledError):
                await connection.close()
                raise
            self._connections[replica_id] = connection
            return connection

    async def remove(self, replica_id: UUID) -> bool:
        async with self._locks.setdefault(replica_id, asyncio.Lock()):
            connection = self._connections.pop(replica_id, None)
            if connection is None:
                return False
            await connection.close()
            return True

    async def remove_all(self) -> None:
        # Include replicas still opening, before their connections have been published.
        replica_ids = set(self._connections).union(self._locks)
        await asyncio.gather(
            *(self.remove(replica_id) for replica_id in replica_ids), return_exceptions=True
        )

    async def all(self) -> list[GatewayConnection]:
        return list(self._connections.values())


gateway_connections_pool: GatewayConnectionsPool = GatewayConnectionsPool()
