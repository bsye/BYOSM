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
def teacher_train(
    iters: int = typer.Option(200_000_000, help="MCCFR traversals in total; a resumed run trains only what is left"),
    buckets: int = typer.Option(12, help="card groups per betting round after the flop (Pluribus used 200)"),
    snapshot_avg: bool = typer.Option(
        False, help="store later rounds as the mean of strategy snapshots, as Pluribus did"
    ),
    linear_cap: int = typer.Option(None, help="stop growing the weight of later traversals after this many"),
) -> None:
    """Train the open-source Pluribus rebuild's strategy in a sealed container; it resumes from its checkpoint."""
    from ..benchmarks.pluribus import root
    from ..config import find_workspace
    from ..games.poker.teacher import Recipe, train

    recipe = Recipe(iters, buckets, snapshot_avg, linear_cap)
    log(f"saved {train(find_workspace(), root(), recipe, log)}")


def _strategy_option() -> str:
    from ..games.poker.teacher import DEFAULT

    return typer.Option(DEFAULT, "--strategy", help="which trained strategy, by the name teacher-train saved")


@poker_app.command("teacher-benchmark")
def teacher_benchmark(name: str = _strategy_option()) -> None:
    """Score a trained strategy on how often it agrees with Pluribus's 15,169 decisions, and write its labels."""
    from ..benchmarks.pluribus import root
    from ..config import find_workspace
    from ..games.poker.teacher import benchmark

    try:
        benchmark(find_workspace(), root(), name, log)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e


def _won(won: dict, against: str) -> str:
    return (
        f"{won['mbb_per_hand']:+.1f} mbb/hand (95% CI ±{won['ci95']:.1f}) over {won['hands']} hands against {against}"
    )


@poker_app.command("teacher-play")
def teacher_play(
    name: str = _strategy_option(),
    field: str = typer.Option(
        "caller", help="who fills the other five seats: caller, random, or itself (with --search: what search adds)"
    ),
    hands: int = typer.Option(200_000, help="hands to play"),
    search: bool = typer.Option(False, help="the teacher searches during each hand, like Pluribus"),
) -> None:
    """Seat a trained strategy at a 6-player table of simple bots and measure what it wins, with a 95% CI."""
    from ..benchmarks.pluribus import root
    from ..config import find_workspace
    from ..games.poker.table import play

    try:
        log(_won(play(find_workspace(), root(), name, field, hands, search, log), field))
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e


@poker_app.command("teacher-versus")
def teacher_versus(focal: str, field: str, hands: int = typer.Option(200_000, help="hands to play")) -> None:
    """Seat one trained strategy against five copies of another and measure what it wins, with a 95% CI."""
    from ..benchmarks.pluribus import root
    from ..config import find_workspace
    from ..games.poker.table import versus

    try:
        log(_won(versus(find_workspace(), root(), focal, field, hands, log), f"five copies of {field}"))
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e


@poker_app.command("teacher-infosets")
def teacher_infosets(
    name: str = _strategy_option(),
    hands: int = typer.Option(2_000_000, help="hands of self-play that count how often each situation comes up"),
) -> None:
    """Write every situation a trained strategy knows, with its legal moves, its probabilities and its visits."""
    from ..benchmarks.pluribus import root
    from ..config import find_workspace
    from ..games.poker.infosets import export

    try:
        found = export(find_workspace(), root(), name, hands, log)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    log(f"{found['records']} situations, {found['visited']} of them reached in {found['hands']} hands of self-play")
