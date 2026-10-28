from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import typer

from .base import lab_app, log

if TYPE_CHECKING:
    from ..lab.owners import Owner


def _owner(name: str) -> Owner:
    from ..lab.owners import owner

    try:
        return owner(name)
    except KeyError as e:
        raise typer.BadParameter(str(e)) from e


@lab_app.command("write")
def lab_write(
    name: str,
    ask: Annotated[str, typer.Option(help="what the method should try")] = "",
    paper: Annotated[str, typer.Option(help="arXiv link of a paper whose method sections the LLM reads")] = "",
) -> None:
    """Have the LLM write a new training method for the specialist or forecaster; it waits for approval before it can run."""
    from ..lab.author import write

    _owner(name)
    write(name, ask, log, paper)


@lab_app.command("ls")
def lab_ls(name: str) -> None:
    """List the written methods with their status and result."""
    from ..lab.contract import CONTRACTS
    from ..lab.drafts import drafts
    from ..lab.owners import kind_of

    found = _owner(name)
    won = CONTRACTS[kind_of(found)].won
    for d in drafts(found):
        result = d.get("result") or {}
        outcome = won if result.get("won") else result.get("experiment") or d.get("error") or ""
        log(f"{d['id']}  {d['status']:9s}  {d['name']}: {d['hypothesis'][:80]}  {outcome}")


@lab_app.command("show")
def lab_show(name: str, method: str) -> None:
    """Print a method's code, checks and result."""
    from ..lab.drafts import draft

    d = draft(_owner(name), method)
    if d is None:
        raise typer.BadParameter(f"no lab method {method}")
    log(f"{d['name']} ({d['status']}): {d['hypothesis']}\n\n{d['code']}\n")
    log("checks: " + ("; ".join(d["problems"]) or "pass"))


@lab_app.command("approve")
def lab_approve(name: str, method: str) -> None:
    """Approve a method's exact code so it can run."""
    from ..lab.drafts import APPROVED, decide

    try:
        decide(_owner(name), method, APPROVED)
    except (KeyError, ValueError) as e:
        raise typer.BadParameter(str(e)) from e
    log(f"approved {method}: run it with `byosm lab run {name} {method}`")


@lab_app.command("run")
def lab_run(name: str, method: str) -> None:
    """Train with an approved method: a specialist's promoted recipe, voted on like any experiment, or a forecaster's patch transformers, promoted only on a better calib score."""
    from ..lab.run import run

    _owner(name)
    try:
        run(name, method, log)
    except PermissionError as e:
        raise typer.BadParameter(str(e)) from e
