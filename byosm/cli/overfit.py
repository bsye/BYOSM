from __future__ import annotations

from typing import Annotated, Any

import typer
from rich.table import Table

from ..format import fmt
from .base import _spec, app, console, log

SEVERITY_STYLE = {"ok": "green", "warn": "yellow", "fail": "red"}
DIGITS = 3


def _num(v: Any) -> str:
    return fmt(v, DIGITS) if isinstance(v, float) else str(v)


def _numbers(nums: dict[str, Any]) -> str:
    return ", ".join(f"{k.replace('_', ' ')} {_num(v)}" for k, v in nums.items() if v is not None)


def _table(r: dict[str, Any]) -> Table:
    scores = "  ".join(f"{k} {_num(v)}" for k, v in r["scores"].items())
    where = f"{r['specialist']}/{r['experiment']}"
    t = Table(title=f"{where} - is it overfitting? ({r['metric']} on {r['suite']}: {scores})")
    t.add_column("check")
    t.add_column("")
    t.add_column("finding")
    t.add_column("numbers", style="dim")
    for f in r["findings"]:
        sev = f"[{SEVERITY_STYLE[f['severity']]}]{f['severity']}[/]"
        t.add_row(" ".join(filter(None, (f["title"], f.get("suite")))), sev, f["explanation"], _numbers(f["numbers"]))
    return t


def print_overfit(r: dict[str, Any]) -> None:
    console.print(_table(r))
    for f in r["findings"]:
        for e in f.get("examples", []):
            log(f"  {f.get('suite')} test: {e['test'][:90]!r}  train: {e['train'][:90]!r}")
    for s in r["skipped"]:
        log(f"  skipped {s}")
    log(f"{r['verdict']}: {r['summary']}")


@app.command()
def overfit(
    name: str,
    exp: Annotated[str | None, typer.Option(help="experiment to check (default: promoted)")] = None,
    force: Annotated[bool, typer.Option(help="ignore the cached report")] = False,
) -> None:
    """Check whether a specialist memorised its training data, over-used calib, or its benchmark leaks into training."""
    from ..evals.overfit import check

    _spec(name)
    print_overfit(check(name, exp, force=force, log=log))
