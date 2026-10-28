from __future__ import annotations

from typing import Annotated

import typer

from .base import _spec, lab_app, log


@lab_app.command("write")
def lab_write(
    name: str,
    ask: Annotated[str, typer.Option(help="what the method should try")] = "",
    paper: Annotated[str, typer.Option(help="arXiv link of a paper whose method sections the LLM reads")] = "",
) -> None:
    """Have the LLM write a new training method for the specialist; it waits for approval before it can run."""
    from ..lab.author import write

    _spec(name)
    write(name, ask, log, paper)


@lab_app.command("ls")
def lab_ls(name: str) -> None:
    """List the written methods with their status and result."""
    from ..lab.drafts import drafts

    p, _ = _spec(name)
    for d in drafts(p):
        result = d.get("result") or {}
        outcome = "sure lead" if result.get("won") else result.get("experiment") or d.get("error") or ""
        log(f"{d['id']}  {d['status']:9s}  {d['name']}: {d['hypothesis'][:80]}  {outcome}")


@lab_app.command("show")
def lab_show(name: str, method: str) -> None:
    """Print a method's code, checks and result."""
    from ..lab.drafts import draft

    p, _ = _spec(name)
    d = draft(p, method)
    if d is None:
        raise typer.BadParameter(f"no lab method {method}")
    log(f"{d['name']} ({d['status']}): {d['hypothesis']}\n\n{d['code']}\n")
    log("checks: " + ("; ".join(d["problems"]) or "pass"))


@lab_app.command("approve")
def lab_approve(name: str, method: str) -> None:
    """Approve a method's exact code so it can run."""
    from ..lab.drafts import APPROVED, decide

    p, _ = _spec(name)
    try:
        decide(p, method, APPROVED)
    except (KeyError, ValueError) as e:
        raise typer.BadParameter(str(e)) from e
    log(f"approved {method}: run it with `byosm lab run {name} {method}`")


@lab_app.command("run")
def lab_run(name: str, method: str) -> None:
    """Train the promoted recipe with an approved method and vote on it like any experiment."""
    from ..lab.run import run

    _spec(name)
    try:
        run(name, method, log)
    except PermissionError as e:
        raise typer.BadParameter(str(e)) from e
