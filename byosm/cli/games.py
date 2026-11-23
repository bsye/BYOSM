from __future__ import annotations

import typer

from .base import app, log

poker_app = typer.Typer(no_args_is_help=True, help="Poker: pick the move a solver would make in a hold'em spot.")
app.add_typer(poker_app, name="poker")

POKER_TASK = "Given a 6-max no-limit hold'em spot, pick the optimal move among the ones offered"


@poker_app.command("new")
def poker_new(name: str, description: str = typer.Argument(POKER_TASK)) -> None:
    """Create a poker specialist trained on PokerBench's solver spots and scored on its 11k test spots."""
    from ..config import find_workspace
    from ..games.poker.create import create

    try:
        create(find_workspace(), name, description, log)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    log(f"next: byosm build {name}")
