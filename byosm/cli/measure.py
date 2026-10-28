from __future__ import annotations

from typing import Annotated, Any

import typer

from .base import app, log


def _line(n: int, c: dict[str, Any]) -> str:
    return f"{n}. {c['lane']}  {c['specialist']}, {c['title']}: {c['why']}  (byosm {' '.join(c['args'])})"


@app.command()
def measure(
    names: Annotated[list[str] | None, typer.Argument(help="specialists or forecaster domains to measure")] = None,
    every: Annotated[bool, typer.Option("--all", help="every specialist and forecaster")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="print the plan and queue nothing")] = False,
) -> None:
    """Queue every measurement still missing for the promoted models: the overfitting check, the hard public
    benchmarks, seed noise, the learning curve while data is thin, and pretraining v2 for forecasters. CPU checks go
    first and GPU trainings run one specialist at a time; the app runs them as jobs and never cancels running ones."""
    from ..app.lock import serving
    from ..config import find_workspace
    from ..measure.plan import plan
    from ..measure.run import queue

    if not names and not every:
        raise typer.BadParameter("name the specialists or forecasters to measure, or pass --all")
    ws = find_workspace()
    try:
        found = (plan if dry_run else queue)(ws, names or None)
    except KeyError as e:
        raise typer.BadParameter(str(e).strip("'\"")) from e
    for n, c in enumerate(found, 1):
        log(_line(n, c))
    if not found:
        log("nothing to measure: every promoted model has its measurements")
    elif not dry_run:
        held = serving(ws)
        where = f"the app on port {held.get('port')} starts them now" if held else "`byosm app` starts them"
        log(f"queued {len(found)} measurements; {where}, and the Jobs page shows their progress")
