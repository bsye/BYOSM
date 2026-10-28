from __future__ import annotations

import json
from typing import Annotated

import typer

from ..project import Project
from .base import app, log
from .train import build


def build_or_hint(name: str, exp: str, run: bool) -> None:
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


def _each(name: str, ids: list[str], run: bool, force: bool, screen: bool) -> None:
    from .. import improve as imp
    from ..events import stage
    from ..rungs import screen as screened

    exps = []
    for pid in ids:
        try:
            exps.append(imp.apply(name, [pid], force=force))
        except ValueError as e:
            log(f"skipped {pid}: {str(e).splitlines()[0]}")
    with stage("Screen with short runs"):
        kept = screened(name, exps, log) if run and screen else exps
    for exp in kept:
        with stage(f"Train {exp}"):
            build_or_hint(name, exp, run)


@app.command()
def improve(
    name: str,
    apply: Annotated[str | None, typer.Option(help="comma-separated proposal ids")] = None,
    run: Annotated[bool, typer.Option("--run", help="build the new experiment")] = False,
    each: Annotated[bool, typer.Option("--each", help="one experiment per proposal, one after the other")] = False,
    no_llm: Annotated[bool, typer.Option(help="rule-based proposals only")] = False,
    force: Annotated[bool, typer.Option("--force", help="run a proposal again even if it already ran")] = False,
    screen: Annotated[
        bool, typer.Option(help="with --each --run: short runs first, full training for the best")
    ] = True,
) -> None:
    """Diagnose the board and propose the next experiment (spec edits with evidence)."""
    from .. import improve as imp

    ids = [s.strip() for s in (apply or "").split(",") if s.strip()]
    if each:
        _each(name, ids, run, force, screen)
        return
    if apply:
        try:
            exp = imp.apply(name, ids, force=force)
        except imp.AlreadyTriedError as e:
            raise typer.BadParameter(f"{e}; pass --force to run it again") from e
        log(f"created experiment {exp} -> {Project(name).exp_dir(exp) / 'spec.yaml'}")
        build_or_hint(name, exp, run)
        return
    dg, props = imp.propose(name, not no_llm, log)
    _log_diagnosis(dg)
    if not props:
        log("no proposals - the board has no obvious gap to act on")
        return
    for pr in props:
        _log_proposal(pr)
    log(f"\napply: byosm improve {name} --apply {','.join(p['id'] for p in props)} --run")
