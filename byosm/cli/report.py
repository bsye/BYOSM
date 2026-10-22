from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from .base import app, log


@app.command()
def report(
    page: Annotated[
        Path | None, typer.Option(help="file holding the <!-- report:<name> --> blocks (default docs/results.md)")
    ] = None,
    check: Annotated[bool, typer.Option("--check", help="write nothing; exit 1 if any block is out of date")] = False,
) -> None:
    """Regenerate the result tables on the results page from the results database, between <!-- report:<name> --> markers."""
    from ..report import RESULTS_PAGE, sync

    page = page or RESULTS_PAGE
    current, missing, extra = sync(page, check)
    for key in missing:
        log(f"no <!-- report:{key} --> block in {page}" + ("" if check else ", appended at the end"))
    for key in extra:
        log(f"block <!-- report:{key} --> in {page} is unknown or repeated: remove it by hand")
    if check and not current:
        log(f"{page} is out of date: run byosm report")
    if extra or (check and not current):
        raise typer.Exit(1)
    log(f"{page} is up to date" if current or check else f"updated {page}")
