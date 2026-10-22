from __future__ import annotations

from typing import Annotated

import typer

from .base import _spec, app, log


@app.command("ablate")
def ablate(
    name: str,
    set_: Annotated[list[str], typer.Option("--set", help="path=value to switch on, e.g. train.pretrain_steps=200")],
    base: Annotated[str | None, typer.Option(help="experiment to start from (default: the promoted one)")] = None,
    seeds: Annotated[str, typer.Option(help="comma-separated seeds; each builds a matched off/on pair")] = "0,1",
) -> None:
    """Measure one change: build matched pairs with and without it per seed, compare them with a paired bootstrap on
    calib and test, and call it better only if every seed agrees."""
    import yaml

    from ..ablation import ablate as run

    p, _ = _spec(name)
    change = {k: yaml.safe_load(v) for k, _, v in (kv.partition("=") for kv in set_)}
    run(name, base or p.promoted, change, [int(s) for s in seeds.split(",")], log)


__all__ = ["app"]
