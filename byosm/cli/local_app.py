from __future__ import annotations

import secrets
from typing import Annotated

import typer

from ..config import MAX_PARALLEL_JOBS, find_workspace
from .base import app, log

DEFAULT_PORT = 8765


@app.command("app")
def app_(
    host: Annotated[
        str, typer.Option(help="bind address; anything but 127.0.0.1 exposes the API to your network")
    ] = "127.0.0.1",
    port: int = DEFAULT_PORT,
    open_browser: Annotated[bool, typer.Option("--open/--no-open", help="open the UI in the browser")] = True,
    parallel: Annotated[int, typer.Option(help="how many jobs run at the same time")] = MAX_PARALLEL_JOBS,
) -> None:
    """Local web app and API: manage specialists, run jobs and watch them live (REST + WebSocket, token protected)."""
    import webbrowser

    import uvicorn

    from ..app import create_app

    token = secrets.token_urlsafe(24)
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}/#token={token}"
    log(f"byosm app on {url}\nAPI docs: http://{host}:{port}/docs (send Authorization: Bearer <token>)")
    if open_browser:
        webbrowser.open(url)
    uvicorn.run(create_app(find_workspace(), token, [host], parallel), host=host, port=port, log_level="warning")
