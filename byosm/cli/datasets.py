from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from .base import _spec, app, log


@app.command("request-data")
def request_data(name: str) -> None:
    """Write a data request: a template (csv and xlsx), the rows needed per label and why."""
    from ..config import load_config
    from ..data.request import write_request

    p, spec = _spec(name)
    out = write_request(p, spec, "Requested by hand.", load_config(p.ws))
    log(f"wrote {out}: fill template.xlsx (or template.csv), then `byosm check-data {name} <file>`")


@app.command("check-data")
def check_data(
    name: str,
    file: Path,
    judge: Annotated[bool, typer.Option(help="ask the LLM to review rows the probe flags")] = True,
) -> None:
    """Check a filled table (format, coverage, leakage, likely mislabels) and train on it if it is usable."""
    from ..config import load_config
    from ..data.check import check

    p, spec = _spec(name)
    report = check(spec, p, file.resolve(), judge, load_config(p.ws), log)
    log(report["markdown"])
    log(f"verdict: {report['verdict']}")
    if report["verdict"] == "fix":
        raise typer.Exit(1)
    spec.data.local_dir = str(file.resolve())
    p.save_spec(spec)
    log(f"{name} now trains on {file}; next: `byosm auto {name}`")


@app.command("continue")
def continue_training(
    name: str,
    start: Annotated[
        str | None, typer.Option("--from", help="the experiment to continue (default: the promoted one)")
    ] = None,
    data: Annotated[
        Path | None, typer.Option(help="a table of new labelled rows to add, checked before training")
    ] = None,
    epochs: Annotated[int | None, typer.Option(help="epochs to train for (default: up to 3)")] = None,
    run: Annotated[bool, typer.Option("--run", help="train it right away")] = False,
) -> None:
    """Keep training a model from its weights on more data, gently enough that it keeps what it learned."""
    from ..config import find_workspace, load_config
    from ..data.check import check
    from ..pipeline import build, next_exp
    from ..train.warm import continued

    p, base = _spec(name)
    parent = start or p.promoted
    if not parent:
        raise typer.BadParameter(f"{name} has no trained experiment to continue from yet")
    spec = continued(p.exp_spec(parent), parent, epochs)
    if data:
        table = data if data.is_absolute() else find_workspace() / data
        report = check(spec, p, table.resolve(), True, load_config(p.ws), log)
        if report["verdict"] == "fix":
            log(report["markdown"])
            raise typer.Exit(1)
        spec.data.local_dir = str(table.resolve())
    exp = next_exp(p)
    why = f"continue {parent} at learning rate {spec.train.lr_backbone:g} for {spec.train.epochs} epochs"
    p.save_experiment(
        exp, spec=spec, proposal={"parent": parent, "applied": [{"id": "continue", "why": why, "ops": []}]}
    )
    log(f"created {name}/{exp}: {why}" + (f", adding {data}" if data else ""))
    if run:
        build(name, exp, log)
