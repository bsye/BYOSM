from __future__ import annotations

import typer

from .base import app, log

poker_app = typer.Typer(no_args_is_help=True, help="Poker: pick the move a solver would make in a hold'em spot.")
app.add_typer(poker_app, name="poker")

POKER_TASK = "Given a 6-max no-limit hold'em spot, pick the optimal move among the ones offered"


@poker_app.command("new")
def poker_new(
    name: str,
    description: str = typer.Argument(POKER_TASK),
    spots: int = typer.Option(None, help="PokerBench training spots, drawn at random (default 60,000)"),
) -> None:
    """Create a poker specialist trained on PokerBench's solver spots and scored on its 11k test spots."""
    from ..config import find_workspace
    from ..games.poker.create import TRAIN_SPOTS, create

    try:
        create(find_workspace(), name, description, log, spots or TRAIN_SPOTS)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    log(f"next: byosm build {name}")


@poker_app.command("teacher-train")
def teacher_train(iters: int = typer.Option(200_000_000, help="MCCFR traversals")) -> None:
    """Train the open-source Pluribus rebuild's blueprint in a sealed container; it resumes from its checkpoint."""
    from ..benchmarks.pluribus import root
    from ..config import find_workspace
    from ..games.poker.teacher import train

    log(f"saved {train(find_workspace(), root(), iters, log)}")


@poker_app.command("teacher-benchmark")
def teacher_benchmark(
    iters: int = typer.Option(200_000_000, help="which blueprint, by its training traversals"),
) -> None:
    """Score that blueprint on how often it agrees with Pluribus's 15,169 decisions."""
    from ..benchmarks.pluribus import root
    from ..config import find_workspace
    from ..games.poker.teacher import benchmark

    benchmark(find_workspace(), root(), iters, log)


@poker_app.command("teacher-play")
def teacher_play(
    field: str = typer.Option("caller", help="who fills the other five seats: caller or random"),
    hands: int = typer.Option(200_000, help="hands to play"),
    search: bool = typer.Option(False, help="the teacher searches during each hand, like Pluribus"),
    iters: int = typer.Option(200_000_000, help="which blueprint, by its training traversals"),
) -> None:
    """Seat the teacher at a 6-player table and measure what it wins, in mbb/hand with a 95% CI."""
    from ..benchmarks.pluribus import root
    from ..config import find_workspace
    from ..games.poker.table import play

    try:
        won = play(find_workspace(), root(), iters, field, hands, search, log)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    log(f"{won['mbb_per_hand']:+.1f} mbb/hand (95% CI ±{won['ci95']:.1f}) over {won['hands']} hands against {field}")
