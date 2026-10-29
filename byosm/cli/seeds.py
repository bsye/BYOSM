from __future__ import annotations

from typing import Annotated

import typer

from .base import _spec, app, log


@app.command()
def seeds(
    name: str,
    exp: Annotated[str | None, typer.Option(help="recipe to replicate (default: the promoted model)")] = None,
    n: Annotated[int | None, typer.Option(help="clean runs wanted (default: the spec's finalist_seeds)")] = None,
) -> None:
    """Train a recipe again on other seeds until it has n runs whose calib was held out, then print the seed noise."""
    from ..evals.selection import selected
    from ..replicas import replicate, score_rest, seed_sd

    p, base = _spec(name)
    target = exp or p.promoted
    if not target:
        raise typer.BadParameter("nothing promoted yet; pass --exp")
    made = replicate(name, target, n or base.finalist_seeds, log)
    sd = seed_sd(p, selected(base, p.board().get(target, {})))
    log(f"trained {len(made)} seeds of {target}; seed sd on calib: {'unknown' if sd is None else f'{sd:.4f}'}")
    score_rest(name, made, log)


@app.command()
def vote(name: str, exp: str) -> None:
    """Decide again whether an experiment beats the promoted model on the selection suites, training the seeds the vote needs."""
    from ..promotion import maybe_promote

    p, _ = _spec(name)
    if exp not in p.experiments():
        raise typer.BadParameter(f"unknown experiment; have {p.experiments()}")
    maybe_promote(p, exp, log)


@app.command()
def curve(name: str) -> None:
    """Train the promoted recipe on 25% and 50% of its training rows and estimate what twice the data would add on calib."""
    from ..curve import learning_curve

    _spec(name)
    learning_curve(name, log)


@app.command("time")
def time_(
    name: str,
    exp: Annotated[list[str] | None, typer.Option(help="experiments to time (default: those timed while busy)")] = None,
) -> None:
    """Time experiments again on the primary suite once the machine is quiet, so their CPU latency is fair."""
    from ..evals.timing import retime

    _spec(name)
    retime(name, exp or None, log)
