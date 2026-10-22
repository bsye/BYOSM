from __future__ import annotations

from typing import Annotated

import typer

from .base import app, db_app, log


@db_app.command("url")
def url() -> None:
    """Print where the database lives (DATABASE_URL, or a SQLite file in the workspace)."""
    from ..config import find_workspace
    from ..db import database_url

    log(database_url(find_workspace()))


@db_app.command("import")
def import_(
    delete: Annotated[bool, typer.Option(help="delete the imported files afterwards")] = False,
) -> None:
    """Move specs, experiments, results and reports from specialists/*/ files into the database."""
    from ..config import find_workspace
    from ..db.importer import import_workspace

    done = import_workspace(find_workspace(), log)
    log(f"imported {len(done)} files")
    if delete:
        for f in done:
            f.unlink()
        log(f"deleted {len(done)} files")


__all__ = ["app"]
