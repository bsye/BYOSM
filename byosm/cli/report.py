from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from .base import app, log


@app.command()
def report(
    readme: Annotated[Path, typer.Option(help="file holding the <!-- report:<name> --> blocks")] = Path("README.md"),
    check: Annotated[bool, typer.Option("--check", help="write nothing; exit 1 if any block is out of date")] = False,
) -> None:
    """Regenerate the README result tables from specialists/*/board.json, between <!-- report:<name> --> markers."""
    from ..report import sync

    current, missing, extra = sync(readme, check)
    for key in missing:
        log(f"no <!-- report:{key} --> block in {readme}" + ("" if check else ", appended at the end"))
    for key in extra:
        log(f"block <!-- report:{key} --> in {readme} is unknown or repeated: remove it by hand")
    if check and not current:
        log(f"{readme} is out of date: run byosm report")
    if extra or (check and not current):
        raise typer.Exit(1)
    log(f"{readme} is up to date" if current or check else f"updated {readme}")
