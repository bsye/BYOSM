from __future__ import annotations

import typer

from .base import log
from .games import poker_app


def _blueprint_argument() -> str:
    from ..games.poker.sharingan import DEFAULT

    return typer.Argument(DEFAULT, help="which published blueprint")


@poker_app.command("blueprint-fetch")
def blueprint_fetch(name: str = _blueprint_argument()) -> None:
    """Download a published poker blueprint and its card buckets, checked against the published checksums."""
    from ..config import find_workspace
    from ..games.poker.sharingan import blueprint
    from ..games.poker.sharingan.weights import fetch

    try:
        blueprint(name)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    fetch(find_workspace(), name, log)


@poker_app.command("blueprint-gauntlet")
def blueprint_gauntlet(
    name: str = _blueprint_argument(),
    deals: int = typer.Option(None, help="deals, each played from every seat (default 20,000)"),
) -> None:
    """Seat a published blueprint against its own project's simple bots and compare with its published scores."""
    from ..config import find_workspace
    from ..games.poker.sharingan.gauntlet import DEALS, gauntlet

    try:
        gauntlet(find_workspace(), name, deals or DEALS, log)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
