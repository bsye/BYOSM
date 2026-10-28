from __future__ import annotations

from typing import Annotated

import typer

from ..auto.ready import BUDGET_H, MAX_ROUNDS
from ..config import PUBLISH_LICENSE
from ..project import Project
from .base import _spec, app, log
from .improve import build_or_hint
from .train import board


@app.command()
def auto(
    task: Annotated[str, typer.Argument(help="the task in plain language, or the name of a specialist to resume")],
    prompt: Annotated[
        str | None, typer.Argument(help="the task, when the first argument names a new specialist")
    ] = None,
    modality: Annotated[str | None, typer.Option(help="text | audio | image | document (inferred if omitted)")] = None,
    budget: Annotated[float, typer.Option(help="wall-clock budget in hours")] = 6.0,
    max_experiments: int = 8,
    margin: Annotated[float, typer.Option(help="must beat the best baseline by this much")] = 0.0,
    no_distill: Annotated[bool, typer.Option("--no-distill")] = False,
    max_params: Annotated[float, typer.Option(help="distill if the winner has more params (M) than this")] = 25.0,
    publish_repo: Annotated[
        str | None, typer.Option("--publish", help="also publish to this Hugging Face repo (private)")
    ] = None,
    license: str = PUBLISH_LICENSE,
    no_llm: Annotated[bool, typer.Option(help="rule-based proposals only")] = False,
) -> None:
    """One command: prompt -> spec -> benchmark + baselines -> experiments until it wins (or the budget ends) -> distill -> export."""
    from ..auto import run
    from ..tabular.compose import forecast_if_asked

    name, prompt = (task, prompt) if prompt or Project(task).exists() else (None, task)
    if name is None and forecast_if_asked(prompt, log):
        return
    final = run(
        name,
        prompt,
        modality,
        budget,
        max_experiments,
        margin,
        not no_distill,
        max_params,
        publish_repo,
        license,
        not no_llm,
        log,
    )
    if final.get("data_request"):
        log(f"\nstopped: {final['stop_reason']}")
        return
    board(final["name"], detail=True)
    from ..auto.report import SUMMARY

    p = Project(final["name"])
    verdict = "WON" if final["won"] else "did not win"
    log(f"\n{verdict}: {final['stop_reason']} ({final['hours']} h).\n\n{p.report(SUMMARY) or ''}")


@app.command()
def distill(
    name: str,
    teacher: Annotated[str | None, typer.Option(help="teacher experiment (default: promoted)")] = None,
    run: Annotated[bool, typer.Option("--run", help="build it now")] = False,
) -> None:
    """Create a tiny student experiment that learns from a teacher experiment (and its labels)."""
    from ..distill import STUDENTS, create

    exp = create(name, teacher)
    p, spec = _spec(name)
    log(f"created {name}/{exp}: {STUDENTS[spec.modality][1]} student of {teacher or p.promoted}")
    build_or_hint(name, exp, run)


@app.command("next")
def next_(
    name: str,
    run_all: Annotated[bool, typer.Option("--all", help="run the steps, re-reading the health after each one")] = False,
) -> None:
    """Show what to do next for a specialist (a held-out set too small, rivals not scored, no overfitting check,
    an untried proposal, a stale export), and with --all do them, picking each next step from the health again."""
    from ..next_steps import suggestions

    steps = suggestions(name)
    for s in steps:
        command = s["args"] or ["lab", "approve", name, s["approve"]]
        log(f"- {s['title']}: {s['why']}  (byosm {' '.join(command)})")
    if not steps:
        log(f"{name}: nothing to do")
    if run_all and steps:
        ready(name, MAX_ROUNDS, BUDGET_H)


@app.command("ready")
def ready(
    name: str,
    rounds: Annotated[int, typer.Option(help="most steps to run")] = MAX_ROUNDS,
    budget: Annotated[float, typer.Option(help="wall-clock budget in hours")] = BUDGET_H,
) -> None:
    """Run the suggested next steps, one after the other, until the specialist's health says it is ready for
    production, no untried step is left, or the budget runs out."""
    from ..auto.ready import until_ready

    out = until_ready(name, lambda args: app(args, standalone_mode=False), log, rounds, budget)
    log(f"{name}: {'ready for production' if out['ready'] else 'not ready, ' + out['stop']}")


@app.command("ideas")
def ideas() -> None:
    """Search the web for small specialists worth building next, each with a public benchmark and a rival to beat."""
    from ..config import find_workspace
    from ..ideas import find_ideas

    for i in find_ideas(find_workspace(), log):
        log(f"- {i['title']}: {i['prompt']}  (benchmark {i['dataset']}, rival {i.get('rival') or '-'})")
