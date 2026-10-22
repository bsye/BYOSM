from __future__ import annotations

from .base import app, db_app, log


@db_app.command("url")
def url() -> None:
    """Print where the database lives (DATABASE_URL, or a SQLite file in the workspace)."""
    from ..config import find_workspace
    from ..db import database_url

    log(database_url(find_workspace()))


__all__ = ["app"]
