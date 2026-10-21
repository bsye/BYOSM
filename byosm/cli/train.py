from __future__ import annotations

import json
from typing import Annotated

import typer
from rich.table import Table

from ..format import cpu, fmt, interval, params
from .base import _spec, app, console, exp_app, log


@app.command()
def build(
    name: str,
    exp: Annotated[str | None, typer.Option(help="experiment name (default: next vN)")] = None,
    force_data: bool = False,
    check_overfit: Annotated[bool, typer.Option("--overfit", help="then run `byosm overfit` on the promoted")] = False,
) -> None:
    """Data -> LeJEPA -> train -> score on every suite (+ baselines) -> promote if the primary suite improves."""
    from ..pipeline import build as run

    run(name, exp, log, force_data)
    board(name, detail=True)
    if check_overfit:
        from .overfit import overfit

        overfit(name)


@app.command()
def data(name: str, exp: str | None = None, force: bool = False) -> None:
    """Only build (or show) the dataset an experiment would train on."""
    from ..config import load_config
    from ..data.store import Dataset
    from ..pipeline import ensure_data, exp_spec

    p, base = _spec(name)
    spec = exp_spec(p, exp) if exp else base
    dd = ensure_data(p, spec, load_config(p.ws), log, force)
    log(json.dumps({s: v["n"] for s, v in Dataset(dd).summary().items()}))


@app.command("eval")
def eval_(
    name: str,
    exp: Annotated[list[str], typer.Option(help="only these experiments")] = None,
    suite: Annotated[list[str], typer.Option()] = None,
    baseline: Annotated[list[str], typer.Option(help="only these baselines")] = None,
    force: bool = False,
) -> None:
    """Score experiments and baselines on every suite (results cached by suite fingerprint)."""
    from .. import evals

    evals.run(name, exp or None, baseline or None, suite or None, force, log)
    board(name, detail=True)


def _board_table(name: str, metric: str, suites: list[str], prim: str, detail: bool) -> Table:
    t = Table(title=f"{name} - test {metric} per suite (primary: {prim}; selection uses its calib split)")
    t.add_column("")
    t.add_column("params", justify="right")
    t.add_column("cpu", justify="right")
    for s in suites:
        t.add_column(s, justify="right")
        if s == prim:
            t.add_column("calib", justify="right", style="dim")
        if detail:
            t.add_column("95% CI", justify="right", style="dim")
            t.add_column("P/R", justify="right", style="dim")
        t.add_column("auc", justify="right", style="dim")
    return t


def _suite_cells(r: dict, s: str, best: float | None, prim: str, detail: bool) -> list[str]:
    v = r.get(s)
    pr = r.get(f"{s}:p/r")
    auc = r.get(f"{s}:auc")
    cells = ["-" if v is None else (f"[bold]{v:.4f}[/]" if v == best else f"{v:.4f}")]
    if s == prim:
        cells.append(fmt(r.get(f"{s}:calib"), 4, "-"))
    if detail:
        cells.append(interval(r.get(f"{s}:ci")))
        cells.append(f"{pr[0]:.3f}/{pr[1]:.3f}" if pr and pr[0] is not None else "")
    cells.append(f"{auc:.3f}" if auc else "")
    return cells


@app.command()
def board(
    name: str, detail: Annotated[bool, typer.Option(help="also show precision/recall per suite")] = False
) -> None:
    """Leaderboard: experiments vs baselines, objective per suite."""
    from .. import evals

    p, spec = _spec(name)
    suites, rows = evals.board_rows(name)
    prim = spec.primary_suite()
    t = _board_table(name, spec.objective.metric, suites, prim, detail)
    best = {s: max((r[s] for r in rows if r.get(s) is not None), default=None) for s in suites}
    for r in rows:
        star = " *" if r["promoted"] else ""
        colour = "green" if r["kind"] == "experiment" else "yellow"
        cells = [f"[{colour}]{r['id']}{star}[/]", params(r.get("params")), cpu(r)]
        for s in suites:
            cells += _suite_cells(r, s, best[s], prim, detail)
        t.add_row(*cells)
    console.print(t)
    log("* = promoted (used by predict / export / serve)")


@exp_app.command("create")
def exp_create(
    name: str,
    exp: str,
    set_: Annotated[list[str], typer.Option("--set", help="path=value (YAML value), e.g. document.route=visual")],
    base: Annotated[
        str | None, typer.Option(help="start from this experiment's spec (default: base spec.yaml)")
    ] = None,
) -> None:
    """e.g. byosm exp create doctype visual --set document.route=visual"""
    import yaml

    from ..improve import apply_ops
    from ..spec import TaskSpec

    p, spec = _spec(name)
    if base:
        spec = TaskSpec.load(p.exp_dir(base) / "spec.yaml")
    ops = []
    for kv in set_:
        k, v = kv.split("=", 1)
        ops.append({"op": "set", "path": k, "value": yaml.safe_load(v)})
    created = apply_ops(spec, ops)
    d = p.exp_dir(exp)
    d.mkdir(parents=True, exist_ok=True)
    created.save(d / "spec.yaml")
    proposal = {"parent": base or "spec.yaml", "applied": [{"id": "manual", "why": " ".join(set_), "ops": ops}]}
    (d / "proposal.json").write_text(json.dumps(proposal, indent=2))
    log(f"created {name}/{exp} ({', '.join(set_)}); data key {created.data_key()} -> byosm build {name} --exp {exp}")


@app.command()
def promote(name: str, exp: str) -> None:
    """Choose which experiment ships."""
    p, _ = _spec(name)
    if exp not in p.experiments():
        raise typer.BadParameter(f"unknown experiment; have {p.experiments()}")
    p.promote(exp)
    log(f"promoted {name}/{exp}")
