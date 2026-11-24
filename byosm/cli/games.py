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
    start_from: str = typer.Option(None, help="continue this strategy's training instead of starting over"),
) -> None:
    """Train the open-source Pluribus rebuild's strategy in a sealed container; it resumes from its checkpoint."""
    from ..benchmarks.pluribus import root
    from ..config import find_workspace
    from ..games.poker.teacher import Recipe, train

    recipe = Recipe(iters, buckets, snapshot_avg, linear_cap)
    try:
        log(f"saved {train(find_workspace(), root(), recipe, log, start_from)}")
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e


def _strategy_option() -> str:
    from ..games.poker.teacher import DEFAULT

    return typer.Option(DEFAULT, "--strategy", help="which trained strategy, by the name teacher-train saved")


@poker_app.command("teacher-benchmark")
def teacher_benchmark(name: str = _strategy_option()) -> None:
    """Score a trained strategy on how often it agrees with Pluribus's 15,169 decisions, and write its labels."""
    from ..benchmarks.pluribus import root
    from ..config import find_workspace
    from ..games.poker.table import TOTAL, agreement

    try:
        found = agreement(find_workspace(), root(), name, log)[TOTAL]
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    log(
        f"{found['top_move']:.1%} of Pluribus's moves on the {found['covered']:.1%} of {found['decisions']} decisions covered"
    )


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


@poker_app.command("teacher-remove")
def teacher_remove(name: str) -> None:
    """Delete every file stored for a trained strategy: its strategy, checkpoint, exports and labels."""
    from ..config import find_workspace
    from ..games.poker.teacher import remove

    try:
        gone = remove(find_workspace(), name)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    log(f"removed {len(gone)} files of {name}: {', '.join(gone)}")


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


@poker_app.command("student")
def poker_student(
    name: str = _strategy_option(),
    rounds: int = typer.Option(0, help="extra rounds that add the situations the student itself reaches at the table"),
) -> None:
    """Train a small student on a strategy's exported situations and save it as a strategy the engine can seat."""
    from ..benchmarks.pluribus import root
    from ..config import find_workspace
    from ..games.poker.distill import build

    try:
        made = build(find_workspace(), root(), name, log, rounds)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    log(
        f"saved {made['student']}: held-out KL {made['kl']:.4f}, same top move {made['top_move']:.1%}, "
        f"model {made['model_mb']} MB"
    )
    log(f"next: byosm poker teacher-versus {made['student']} {name} --hands 1000000")


def _student_option() -> str:
    from ..games.poker.distill import STUDENT
    from ..games.poker.teacher import DEFAULT

    return typer.Option(f"{DEFAULT}{STUDENT}", "--student", help="which student, by the name poker student saved")


@poker_app.command("advise")
def poker_advise(hand: str, name: str = _student_option()) -> None:
    """Read a hand history (a .phh file) that stops at a decision and print how often to play each legal move."""
    import time
    from pathlib import Path

    from ..benchmarks.pluribus import root
    from ..config import find_workspace
    from ..games.poker.advisor import Advisor

    try:
        advisor = Advisor.start(find_workspace(), root(), name, log)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    try:
        started = time.perf_counter()
        found = advisor.advise(Path(hand).read_text())
        took = (time.perf_counter() - started) * 1000
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    finally:
        advisor.close()
    log(f"seat {found['seat']} to act on the {found['street']}:")
    for move in sorted(found["moves"], key=lambda m: -m["probability"]):
        log(f"  {move['probability']:6.1%}  {move['move']}")
    log(f"answered in {took:.0f} ms")
