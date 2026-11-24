from typing import AsyncGenerator

from dstack._internal.proxy.lib.auth import BaseProxyAuthProvider
from dstack._internal.proxy.lib.deps import ProxyDependencyInjector
from dstack._internal.proxy.lib.models import Project
from dstack._internal.proxy.lib.repo import BaseProxyRepo


class ProxyTestDependencyInjector(ProxyDependencyInjector):
    def __init__(self, repo: BaseProxyRepo, auth: BaseProxyAuthProvider) -> None:
        super().__init__()
        self._repo = repo
        self._auth = auth

    async def get_repo(self) -> AsyncGenerator[BaseProxyRepo, None]:
        yield self._repo

    async def get_auth_provider(self) -> AsyncGenerator[BaseProxyAuthProvider, None]:
        yield self._auth


def make_project(name: str) -> Project:
    return Project(name=name, ssh_private_key="secret")
