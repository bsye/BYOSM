from __future__ import annotations

from typing import Annotated

import typer

from .base import _spec, log, suite_app


@suite_app.command("public")
def suite_public(
    name: str,
    benchmark: Annotated[
        list[str] | None, typer.Argument(help="public benchmarks to add (default: every one that fits and is missing)")
    ] = None,
) -> None:
    """Add public benchmarks as frozen suites; each is kept only if always-positive scores what the benchmark implies.
    `byosm eval` then scores every model and rival on them with the same code."""
    from ..benchmarks.add import add, chosen

    p, _ = _spec(name)
    try:
        found = chosen(p, benchmark or [])
        for b in found:
            log(f"{b.name}: {b.about}")
            add(p, b, log)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    if found:
        log(f"next: byosm eval {name} " + " ".join(f"--suite {b.name}" for b in found))
    else:
        log(f"{name} already has every public benchmark that fits it")
