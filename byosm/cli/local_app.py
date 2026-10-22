from __future__ import annotations

import os
import secrets
from typing import Annotated

import typer

from ..config import MAX_PARALLEL_JOBS, find_workspace
from .base import app, log

DEFAULT_PORT = 8765
TOKEN_ENV = "BYOSM_APP_TOKEN"
HOSTS_ENV = "BYOSM_APP_HOSTS"
ANY_ADDRESS = ("0.0.0.0", "::")
SHUTDOWN_GRACE_S = 3


def allowed_hosts(given: list[str] | None) -> list[str]:
    from_env = (h.strip() for h in os.environ.get(HOSTS_ENV, "").split(","))
    return [*(given or []), *filter(None, from_env)]


@app.command("app")
def app_(
    host: Annotated[
        str, typer.Option(help="bind address; anything but 127.0.0.1 exposes the API to your network")
    ] = "127.0.0.1",
    port: int = DEFAULT_PORT,
    open_browser: Annotated[bool, typer.Option("--open/--no-open", help="open the UI in the browser")] = True,
    parallel: Annotated[int, typer.Option(help="how many jobs run at the same time")] = MAX_PARALLEL_JOBS,
    token: Annotated[
        str, typer.Option(envvar=TOKEN_ENV, help="fixed API token (default: a new random one each start)")
    ] = "",
    allow_host: Annotated[
        list[str] | None,
        typer.Option(help=f"extra Host header to accept besides localhost (repeatable; or {HOSTS_ENV}=a,b)"),
    ] = None,
) -> None:
    """Local web app and API: manage specialists, run jobs and watch them live (REST + WebSocket, token protected)."""
    import webbrowser

    import uvicorn

    from ..app import create_app

    secret = token or secrets.token_urlsafe(24)
    shown = "<your token>" if token else secret
    url = f"http://{'127.0.0.1' if host in ANY_ADDRESS else host}:{port}/#token={shown}"
    log(f"byosm app on {url}\nAPI docs: http://{host}:{port}/docs (send Authorization: Bearer <token>)")
    if open_browser:
        webbrowser.open(url)
    from ..app.lock import WorkspaceBusyError

    hosts = [host, *allowed_hosts(allow_host)]
    try:
        api = create_app(find_workspace(), secret, hosts, parallel, port)
    except WorkspaceBusyError as e:
        log(str(e))
        raise typer.Exit(1) from e
    uvicorn.run(api, host=host, port=port, log_level="warning", timeout_graceful_shutdown=SHUTDOWN_GRACE_S)
