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


def _deals_option() -> int:
    from ..games.poker.sharingan.arena import DEALS

    return typer.Option(DEALS, help="deals, each played from every seat")


@poker_app.command("arena-gauntlet")
def arena_gauntlet(name: str, deals: int = _deals_option()) -> None:
    """Seat a student or a published blueprint against the arena's six simple bots, five of each at a time."""
    from ..config import find_workspace
    from ..games.poker.sharingan.gauntlet import gauntlet

    try:
        gauntlet(find_workspace(), name, deals, log)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e


@poker_app.command("arena-versus")
def arena_versus(hero: str, field: str, deals: int = _deals_option()) -> None:
    """Seat one HERO against five FIELD in the arena: each a student, a published blueprint or a simple bot."""
    from ..config import find_workspace
    from ..games.poker.sharingan.arena import versus

    try:
        made = versus(find_workspace(), hero, field, deals, log)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    log(
        f"{hero} against five {field}: {made['mbb_per_hand']:+.1f} ±{made['ci95']:.1f} mbb/hand "
        f"over {made['hands']:,} hands ({made['fallbacks']} of {made['requests']} decisions fell back)"
    )
