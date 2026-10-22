from __future__ import annotations

from typing import Annotated

import typer

from .base import _spec, app, log, store_app

ExpOpt = Annotated[str | None, typer.Option("--exp", help="one experiment (default: all weights and the export)")]


@store_app.command("push")
def push(name: str, exp: ExpOpt = None) -> None:
    """Upload a specialist's weights and export to the private artifact repo."""
    from ..artifacts import push as upload

    p, _ = _spec(name)
    upload(p, exp, log)


@store_app.command("pull")
def pull(name: str, exp: ExpOpt = None) -> None:
    """Download a specialist's weights and export from the private artifact repo."""
    from ..artifacts import pull as download

    p, _ = _spec(name)
    download(p, exp, log)


__all__ = ["app"]
