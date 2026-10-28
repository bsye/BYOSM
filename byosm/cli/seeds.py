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
    from ..replicas import replicate, seed_sd

    p, base = _spec(name)
    target = exp or p.promoted
    if not target:
        raise typer.BadParameter("nothing promoted yet; pass --exp")
    made = replicate(name, target, n or base.finalist_seeds, log)
    sd = seed_sd(p, base.primary_suite())
    log(f"trained {len(made)} seeds of {target}; seed sd on calib: {'unknown' if sd is None else f'{sd:.4f}'}")
