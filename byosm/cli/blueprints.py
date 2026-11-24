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


@poker_app.command("blueprint-decisions")
def blueprint_decisions(
    name: str = _blueprint_argument(),
    hands: int = typer.Option(None, help="how many of Pluribus's 10,000 hands to replay (default all)"),
) -> None:
    """Compare a published blueprint's move with the logged move at every decision of Pluribus's hands."""
    from ..config import find_workspace
    from ..games.poker.sharingan.compare import ALL, against_pluribus, compared, pluribus_hands

    try:
        made = compared(find_workspace(), name, "pluribus", pluribus_hands(hands), against_pluribus, log)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    for label, found in sorted(made["groups"].items()):
        if found["decisions"]:
            log(
                f"{label}: {found['decisions']:,} decisions, same kind of move {found['same_kind']:.1%}, "
                f"probability on the logged move {found['mean_weight']:.3f}, fell back {found['fell_back']:.1%}"
            )
    log(f"{ALL}: {made['errors'] or 'no'} refused decisions, {made['skipped'] or 'no'} skipped hands")
