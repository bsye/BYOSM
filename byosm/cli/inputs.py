from __future__ import annotations

from typing import Annotated

import typer

from .base import _spec, inputs_app, log

SetOpt = Annotated[str, typer.Option("--set", help="name of the input set")]


@inputs_app.command("write")
def write(
    name: str, per_label: Annotated[int, typer.Option(help="new inputs per label")] = 10, set_: SetOpt = "workbench"
) -> None:
    """Have the LLM write new labelled inputs and add them to a set; nothing is scored."""
    from ..evals.workbench import write_more

    p, spec = _spec(name)
    log(f"added {write_more(p, spec, set_, per_label, log)} inputs to {set_}")


@inputs_app.command("sample")
def sample(name: str, n: Annotated[int, typer.Option(help="rows to add")] = 20, set_: SetOpt = "benchmark") -> None:
    """Add rows from the specialist's held-out test split to a set; nothing is scored."""
    from ..evals.workbench import sample_benchmark

    p, spec = _spec(name)
    log(f"added {sample_benchmark(p, spec, set_, n)} test rows to {set_}")


@inputs_app.command("run")
def run(
    name: str,
    set_: SetOpt = "workbench",
    model: Annotated[
        list[str] | None, typer.Option(help="ours, a rival id, or several (default: the saved choice)")
    ] = None,
) -> None:
    """Score every input of a set with each chosen model; rows already scored by a model are kept."""
    from ..evals.workbench import chosen_models
    from ..evals.workbench import run as score

    p, spec = _spec(name)
    score(p, spec, set_, model or chosen_models(p, spec), log)
