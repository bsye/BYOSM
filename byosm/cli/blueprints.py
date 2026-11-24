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


def _told(made: dict) -> None:
    for label, found in sorted(made["groups"].items()):
        if found["decisions"]:
            log(
                f"{label}: {found['decisions']:,} decisions, same kind of move {found['same_kind']:.1%}, "
                f"probability on the logged move {found['mean_weight']:.3f}, fell back {found['fell_back']:.1%}"
            )
    shown = f"{made['hands_showing_cards']:,} of {made['hands']:,} hands show cards"
    log(f"{shown}; {made['errors'] or 'no'} refused decisions; {made['skipped'] or 'no'} skipped hands")


@poker_app.command("blueprint-decisions")
def blueprint_decisions(
    name: str = _blueprint_argument(),
    hands: int = typer.Option(None, help="how many of Pluribus's 10,000 hands to replay (default all)"),
    showdown: bool = typer.Option(False, help="only the moves of players who reached a showdown, as real logs show"),
) -> None:
    """Compare a published blueprint's move with the logged move at every decision of Pluribus's hands."""
    from ..config import find_workspace
    from ..games.poker.sharingan.compare import against_pluribus, compared, everyone, pluribus_hands, shown_down

    source, kept = ("pluribus-showdown", shown_down) if showdown else ("pluribus", everyone)
    try:
        _told(compared(find_workspace(), name, source, pluribus_hands(hands), against_pluribus, log, kept))
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e


@poker_app.command("real-hands-fetch")
def real_hands_fetch() -> None:
    """Download 21.6 million real online hands from Zenodo (1.88 GB, CC BY 4.0) and check the published checksum."""
    from ..config import find_workspace
    from ..games.poker.real_hands import fetched

    log(f"real hands are at {fetched(find_workspace(), log)}")


@poker_app.command("blueprint-real-hands")
def blueprint_real_hands(
    name: str = _blueprint_argument(),
    share: float = typer.Option(0.02, help="share of the archive's files to replay, picked by a fixed hash"),
) -> None:
    """Compare a published blueprint's move with what real online players did, by stake and by site."""
    from ..config import find_workspace
    from ..games.poker.real_hands import hands
    from ..games.poker.sharingan.compare import by_stake, compared, full_stack

    ws = find_workspace()
    try:
        _told(compared(ws, name, f"real-{share:g}", hands(ws, share), by_stake, log, full_stack))
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
