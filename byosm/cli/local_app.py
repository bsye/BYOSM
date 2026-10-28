from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Annotated

import typer

from ..config import find_workspace
from .base import app, log

DEFAULT_PORT = 8765
TOKEN_ENV = "BYOSM_APP_TOKEN"
HOSTS_ENV = "BYOSM_APP_HOSTS"
ANY_ADDRESS = ("0.0.0.0", "::")
LOOPBACK = ("127.0.0.1", "localhost", "::1")
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
    parallel: Annotated[
        int | None, typer.Option(help="most jobs at the same time (default: as many as CPUs and free memory allow)")
    ] = None,
    token: Annotated[
        str,
        typer.Option(
            envvar=TOKEN_ENV,
            help="fixed API token (default: none on 127.0.0.1, a new random one when bound to another address)",
        ),
    ] = "",
    allow_host: Annotated[
        list[str] | None,
        typer.Option(help=f"extra Host header to accept besides localhost (repeatable; or {HOSTS_ENV}=a,b)"),
    ] = None,
    reload: Annotated[
        bool, typer.Option("--reload/--no-reload", help="restart the app when its code changes; running jobs carry on")
    ] = False,
) -> None:
    """Local web app and API: manage specialists, run jobs and watch them live (REST + WebSocket, token protected)."""
    import webbrowser

    import uvicorn

    from ..app import create_app
    from ..app.ui_build import ensure_built

    ensure_built(log)

    secret = token or ("" if host in LOOPBACK else secrets.token_urlsafe(24))
    shown = "<your token>" if token else secret
    url = f"http://{'127.0.0.1' if host in ANY_ADDRESS else host}:{port}/" + (f"#token={shown}" if secret else "")
    auth = " (send Authorization: Bearer <token>)" if secret else ", no token: it only listens on this machine"
    log(f"byosm app on {url}\nAPI docs: http://{host}:{port}/docs{auth}")
    if open_browser:
        webbrowser.open(url)
    from ..app.lock import WorkspaceBusyError

    hosts = [host, *allowed_hosts(allow_host)]
    serve = {"host": host, "port": port, "log_level": "warning", "timeout_graceful_shutdown": SHUTDOWN_GRACE_S}
    if reload:
        _reloading(find_workspace(), secret, hosts, parallel, serve)
        return
    try:
        api = create_app(find_workspace(), secret, hosts, parallel, port)
    except WorkspaceBusyError as e:
        log(str(e))
        raise typer.Exit(1) from e
    uvicorn.run(api, **serve)


def _reloading(ws: Path, secret: str, hosts: list[str], parallel: int | None, serve: dict) -> None:
    import uvicorn

    import byosm

    from ..app.factory import settings_env
    from ..app.ui_build import watch

    os.environ.update(settings_env(ws, secret, hosts, parallel, serve["port"]))
    code = str(Path(byosm.__file__).parent)
    log(f"reloading on changes in {code}")
    builder = watch(log)
    try:
        uvicorn.run("byosm.app.factory:from_env", factory=True, reload=True, reload_dirs=[code], **serve)
    finally:
        if builder:
            builder.terminate()
