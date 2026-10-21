from __future__ import annotations

import json
from typing import Annotated

import typer

from ..config import PUBLISH_LICENSE, Project
from .base import _spec, app, log
from .train import board, build


def _build_or_hint(name: str, exp: str, run: bool) -> None:
    if run:
        build(name, exp)
    else:
        log(f"next: byosm build {name} --exp {exp}")


def _log_suite_diagnosis(s: str, r: dict, binary: bool) -> None:
    bb = r["best_baseline"]
    if binary and r["precision"] is not None:
        stats = f"P={r['precision']:.3f} R={r['recall']:.3f}"
    else:
        stats = f"acc={r['accuracy']:.3f} macro_f1={r['macro_f1']:.3f}"
    thr = f" thr={r['threshold']:.2f}" if r["threshold"] is not None else ""
    gap = f"  | best baseline {bb[0]}={bb[1]:.4f} (gap {bb[1] - r['objective']:+.4f})" if bb else ""
    log(f"  {s:10s} objective={r['objective']:.4f} {stats}{thr}" + gap)
    if not binary and r["weakest"]:
        log(f"             weakest: {', '.join(f'{n} {f:.2f}' for n, f in r['weakest'])}")
    if not binary and r["confusions"]:
        log(f"             confused: {', '.join(f'{g}->{p} ({c})' for g, p, c in r['confusions'][:5])}")


def _log_diagnosis(dg: dict) -> None:
    curve = ", ".join(f"{c:.2f}" for c in dg["curve"])
    log(
        f"diagnosis of {dg['experiment']} ({dg['n_train']} train examples; dev curve {curve}; "
        f"best epoch {dg['best_epoch'] + 1}):"
    )
    for s, r in dg["suites"].items():
        _log_suite_diagnosis(s, r, dg["binary"])


def _log_proposal(pr: dict) -> None:
    log(f"\n[{pr['id']}] {pr['why']}")
    for o in pr["ops"]:
        v = json.dumps(o["value"])
        log(f"    {o['op']} {o['path']} = {v[:160]}")
    c = pr.get("cost") or {}
    minutes = c.get("minutes")
    if minutes == minutes and minutes is not None:
        log(
            f"    est. cost ~{c['minutes']:.0f} min"
            + (f" ({c['new_items']} new items to ingest)" if c.get("new_items") else "")
        )


@app.command()
def improve(
    name: str,
    apply: Annotated[str | None, typer.Option(help="comma-separated proposal ids")] = None,
    run: Annotated[bool, typer.Option("--run", help="build the new experiment")] = False,
    no_llm: Annotated[bool, typer.Option(help="rule-based proposals only")] = False,
) -> None:
    """Diagnose the board and propose the next experiment (spec edits with evidence)."""
    from .. import improve as imp

    if apply:
        exp = imp.apply(name, [s.strip() for s in apply.split(",")])
        log(f"created experiment {exp} -> {Project(name).exp_dir(exp) / 'spec.yaml'}")
        _build_or_hint(name, exp, run)
        return
    dg, props = imp.propose(name, not no_llm, log)
    _log_diagnosis(dg)
    if not props:
        log("no proposals - the board has no obvious gap to act on")
        return
    for pr in props:
        _log_proposal(pr)
    log(f"\napply: byosm improve {name} --apply {','.join(p['id'] for p in props)} --run")


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

    name, prompt = (task, prompt) if prompt or Project(task).exists() else (None, task)
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
    board(final["name"], detail=True)
    p = Project(final["name"])
    verdict = "WON" if final["won"] else "did not win"
    log(f"\n{verdict}: {final['stop_reason']} ({final['hours']} h). Report: {p.root / 'auto.md'}")


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
    _build_or_hint(name, exp, run)
