from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from ..report.systems import stream_line
from ..system.board import BOARD
from .base import console, log, system_app


def _project(name: str):
    from ..system import SystemProject

    sp = SystemProject(name)
    if not sp.exists():
        raise typer.BadParameter(f"no system {name!r}: byosm system define {name} FILE.yaml")
    return sp


@system_app.command("define")
def define(name: str, file: Path) -> None:
    """Define (or redefine) a team of specialists from a YAML file of nodes and suites."""
    from ..system import SystemProject, SystemSpec

    SystemProject(name).save_spec(SystemSpec.load(file))
    log(f"defined system {name}; next: byosm system build {name}")


def _fmt(summary: dict) -> str:
    return f"{summary['exact_frame']:.3f}"


def _show(board: dict) -> None:
    for suite, b in board.items():
        t = Table(title=f"{suite}: exact frame (action and every argument right)")
        for col in ("", "test exact [95% CI]", "calib exact", "action acc", "arg F1", "ms / request"):
            t.add_column(col)
        for r, v in b["rivals"].items():
            ci = f"{_fmt(v['test'])} [{v['ci']['lo']:.3f}, {v['ci']['hi']:.3f}]"
            ms = f"{v['latency_ms']:.1f}" if v["latency_ms"] else "-"
            t.add_row(
                r,
                ci,
                _fmt(v["calib"]),
                f"{v['test']['action_accuracy']:.3f}",
                "-" if v["test"]["arg_f1"] is None else f"{v['test']['arg_f1']:.3f}",
                ms,
            )
        console.print(t)
        if b.get("stream"):
            log(stream_line(b["stream"]))
        for r, pv in b["paired"].items():
            test, calib = pv["test"], pv["calib"]
            log(
                f"  system - {r}: test {test['delta']:+.3f} (P(better) {test['p_better']:.2f}), calib {calib['delta']:+.3f}"
            )


@system_app.command("build")
def build(name: str) -> None:
    """Build every node as a specialist, export one bundle for the team, and score it end to end."""
    from ..system import build_node, evaluate, export_system

    sp = _project(name)
    spec = sp.spec()
    for node in spec.nodes.values():
        build_node(node, spec, log)
    export_system(sp, log)
    _show(evaluate(sp, log))


@system_app.command("eval")
def eval_(name: str, force: Annotated[bool, typer.Option(help="rescore cached rivals")] = False) -> None:
    """Score the exported team and its rivals (an LLM doing the whole task) on every system suite."""
    from ..system import evaluate

    _show(evaluate(_project(name), log, force))


@system_app.command("board")
def board(name: str) -> None:
    """Show the last end-to-end scores."""
    _show(_project(name).report(BOARD))


def _typed(system, text: str) -> dict:
    st = system.stream()
    for i in range(1, len(text) + 1):
        if st.step(text[:i])["fired"]:
            log(f"fired after {i}/{len(text)} characters: {text[:i]!r}")
    return st.finish(text)


@system_app.command("run")
def run(
    name: str, text: str, stream: Annotated[bool, typer.Option(help="feed the text one keystroke at a time")] = False
) -> None:
    """Run the exported team on one request: action, arguments and the tool call it maps to."""
    from ..rt import System

    system = System(_project(name).export_dir)
    log(json.dumps(_typed(system, text) if stream else system.run(text), indent=2))
