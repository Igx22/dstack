import asyncio
import hashlib
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from dstack._internal.core.errors import SSHError
from dstack._internal.server.services.gateways.connection import GatewayConnection
from dstack._internal.server.services.gateways.pool import GatewayConnectionsPool


@pytest.mark.asyncio
async def test_recycled_address_does_not_reuse_previous_replicas_tunnel():
    pool = GatewayConnectionsPool()
    old_id, new_id = uuid.uuid4(), uuid.uuid4()
    with patch("dstack._internal.server.services.gateways.pool.GatewayConnection") as factory:
        old = AsyncMock(ip_address="host")
        replacement = AsyncMock(ip_address="host")
        factory.side_effect = [old, replacement]
        assert await pool.get_or_add(old_id, "host", "key") is old
        assert await pool.get_or_add(new_id, "host", "key") is replacement
        assert await pool.remove(old_id)
        old.close.assert_awaited_once()
        replacement.close.assert_not_awaited()
        assert await pool.get_or_add(new_id, "host", "key") is replacement
        assert await pool.all() == [replacement]


@pytest.mark.asyncio
async def test_changed_address_reopens_tunnel_for_same_replica():
    pool = GatewayConnectionsPool()
    replica_id = uuid.uuid4()
    old = AsyncMock(ip_address="old-host")
    updated = AsyncMock(ip_address="new-host")
    with patch(
        "dstack._internal.server.services.gateways.pool.GatewayConnection",
        side_effect=[old, updated],
    ):
        assert await pool.get_or_add(replica_id, "old-host", "key") is old
        assert await pool.get_or_add(replica_id, "new-host", "key") is updated
        old.close.assert_awaited_once()
        updated.open.assert_awaited_once()
        assert await pool.all() == [updated]


@pytest.mark.asyncio
async def test_opening_one_replica_does_not_block_another_at_the_same_address():
    pool = GatewayConnectionsPool()
    slow_id, other_id = uuid.uuid4(), uuid.uuid4()
    entered, release = asyncio.Event(), asyncio.Event()

    async def delayed_open(**kwargs):
        entered.set()
        await release.wait()

    slow, other = AsyncMock(ip_address="host"), AsyncMock(ip_address="host")
    slow.open.side_effect = delayed_open
    with patch(
        "dstack._internal.server.services.gateways.pool.GatewayConnection",
        side_effect=[slow, other],
    ):
        first = asyncio.create_task(pool.get_or_add(slow_id, "host", "key"))
        await entered.wait()
        second = asyncio.create_task(pool.get_or_add(slow_id, "host", "key"))
        try:
            async with asyncio.timeout(2):
                assert await pool.get_or_add(other_id, "host", "key") is other
            assert await pool.all() == [other]
        finally:
            release.set()
            assert await first is slow
            assert await second is slow
        slow.open.assert_awaited_once()


@pytest.mark.asyncio
async def test_socket_paths_are_scoped_to_replica_and_invalidated_after_close(tmp_path):
    # Two server processes can have independent pools that use the same server directory.
    with (
        patch("dstack._internal.server.settings.SERVER_DIR_PATH", tmp_path),
        patch("dstack._internal.server.services.gateways.connection.SSHTunnel", autospec=True),
    ):
        first = GatewayConnection(uuid.uuid4(), "host", "key", 8000)
        second = GatewayConnection(uuid.uuid4(), "host", "key", 8000)
        restarted = GatewayConnection(second.replica_id, "host", "key", 8000)
        assert second.connection_dir == tmp_path / "gateway-connections" / str(second.replica_id)
        first.connection_dir.mkdir(parents=True)
        first.gateway_socket_path.touch()
        second.connection_dir.mkdir(parents=True)
        second.gateway_socket_path.touch()
        assert first.gateway_socket_path.resolve() != second.gateway_socket_path.resolve()
        assert second.gateway_socket_path.resolve() == restarted.gateway_socket_path.resolve()
        await first.close()
        assert not first.connection_dir.exists()
        assert second.gateway_socket_path.exists()
        assert not await first.check_or_restart()
        with pytest.raises(SSHError):
            async with first.client():
                pytest.fail("Retired connection returned a usable client")
        assert not first.gateway_socket_path.parent.exists()
        assert not first.gateway_socket_path.parent.is_symlink()


@pytest.mark.asyncio
async def test_startup_closes_pre_upgrade_control_socket_without_using_it_for_forwarding(tmp_path):
    address_dir = tmp_path / "gateway-connections" / "host"
    address_dir.mkdir(parents=True)
    old_control = address_dir / "control.sock"
    old_control.touch()
    key_file = tmp_path / "identity"
    key_file.write_bytes(b"new key")
    hashed_dir = address_dir / hashlib.sha256(b"new key").hexdigest()
    hashed_dir.mkdir()
    hashed_control = hashed_dir / "control.sock"
    hashed_control.touch()
    other_dir = address_dir / hashlib.sha256(b"other key").hexdigest()
    other_dir.mkdir()
    (other_dir / "control.sock").touch()
    tunnels = {}

    def tunnel(**kwargs):
        control = Path(kwargs["control_sock_path"]).resolve()
        mock = AsyncMock(destination=kwargs["destination"], identity_path=key_file)
        mock.acheck.return_value = False
        tunnels[control] = mock
        return mock

    with (
        patch("dstack._internal.server.settings.SERVER_DIR_PATH", tmp_path),
        patch(
            "dstack._internal.server.services.gateways.connection.SSHTunnel", side_effect=tunnel
        ),
    ):
        conn = GatewayConnection(uuid.uuid4(), "host", "new key", 8000)
        await conn.open(close_existing_tunnel=True)
        tunnels[old_control].aclose.assert_awaited_once()
        tunnels[old_control].aopen.assert_not_awaited()
        tunnels[hashed_control].aclose.assert_awaited_once()
        tunnels[hashed_control].aopen.assert_not_awaited()
        assert other_dir / "control.sock" not in tunnels
        assert conn.tunnel is tunnels[conn.connection_dir / "control.sock"]
        assert conn.gateway_socket_path.resolve().parent != address_dir
        await conn.close()
