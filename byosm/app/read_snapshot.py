from __future__ import annotations

from starlette.types import ASGIApp, Receive, Scope, Send

from ..reads import snapshot


class ReadSnapshot:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "GET":
            await self.app(scope, receive, send)
            return
        with snapshot():
            await self.app(scope, receive, send)
