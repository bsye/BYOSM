from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.table import Table

from ..config import Project, write_default_config
from .base import _spec, app, baseline_app, console, log, suite_app

if TYPE_CHECKING:
    from ..spec import EvalSuite, TaskSpec


@app.command()
def init() -> None:
    """Create byosm.yaml (LLM providers per role) in the current directory."""
    log(
        f"wrote {write_default_config(Path.cwd())} - set providers/models per role: "
        "spec, generate, judge, eval, analyze, baseline"
    )


@app.command()
def new(
    name: str,
    description: str,
    modality: Annotated[str | None, typer.Option(help="text | audio | image (inferred if omitted)")] = None,
) -> None:
    """Compile a plain-language task into specialists/<name>/spec.yaml."""
    from ..spec import compile_spec

    p = Project(name)
    if p.exists():
        raise typer.BadParameter(f"{name} already exists at {p.root}")
    with console.status("compiling task spec..."):
        spec = compile_spec(name, description, modality)
    if spec.modality == "audio" and spec.task_type == "frame":
        spec.baselines = ["energy", "silero"]
    elif spec.modality == "text":
        spec.baselines = ["llm"]
    p.ensure()
    spec.save(p.spec_path)
    _show_spec(spec)
    log(f"\nspec: {p.spec_path}\nnext: review it, optionally `byosm suite add {name} ...`, then `byosm build {name}`")


def _show_spec(spec: TaskSpec) -> None:
    t = Table(show_header=False, box=None, pad_edge=False)
    t.add_row("[bold]task", f"{spec.name}: {spec.modality}/{spec.task_type} - {spec.summary}")
    t.add_row("[bold]labels", "\n".join(f"[cyan]{label.name}[/]: {label.definition}" for label in spec.labels))
    o = spec.objective
    constraint = f" (constraint {o.constraint})" if "_at_" in o.metric else ""
    t.add_row("[bold]objective", f"{o.metric} >= {o.target}" + constraint)
    t.add_row("[bold]backbone", spec.backbone)
    d = spec.data
    sources = [s.dataset for s in d.hf_sources]
    t.add_row("[bold]data", f"synthetic={d.synthetic} train={d.train_size} sources={sources} queries={d.hf_queries}")
    t.add_row("[bold]evals", ", ".join(e.name + (" (primary)" if e.primary else "") for e in spec.evals))
    t.add_row("[bold]baselines", ", ".join(spec.baselines) or "-")
    console.print(t)


@app.command()
def show(name: str) -> None:
    """Print a specialist's spec summary."""
    _show_spec(_spec(name)[1])


@suite_app.command("add")
def suite_add(
    name: str,
    suite: str,
    hf: Annotated[str | None, typer.Option(help="Hugging Face dataset id")] = None,
    format: Annotated[
        str, typer.Option(help="rows | markup (labels as inline [type : words]) | segment_files")
    ] = "rows",
    config: str | None = None,
    split: str = "train",
    input_column: str | None = None,
    label_column: str | None = None,
    local: Annotated[
        Path | None, typer.Option(help="local jsonl with input, label and optional split (calib | test)")
    ] = None,
    max_items: int = 400,
    max_files: int = 20,
    primary: Annotated[bool, typer.Option(help="promotion is decided on this suite")] = False,
) -> None:
    """Add a frozen eval suite. HF suites use the source's calib/test hash partitions (never its train partition)."""
    from ..spec import EvalSuite, HFSource

    p, spec = _spec(name)
    if any(e.name == suite for e in spec.evals):
        raise typer.BadParameter(f"suite {suite} exists")
    if hf:
        default_column = "audio" if format == "rows" and spec.modality == "audio" else None
        src = HFSource(
            dataset=hf,
            format=format,
            config=config,
            split=split,
            input_column=input_column or default_column,
            label_column=label_column,
            max_files=max_files,
            max_rows=max_items * 2,
        )
        es = EvalSuite(name=suite, kind="hf", source=src, max_items=max_items, primary=primary)
    elif local:
        es = EvalSuite(name=suite, kind="local", path=str(local.resolve()), primary=primary)
    else:
        raise typer.BadParameter("pass --hf or --local")
    _append_suite(p, spec, es)
    log(f"added suite {suite} to {name}; it is materialized and frozen on the next `byosm eval {name}`")


def _append_suite(p: Project, spec: TaskSpec, es: EvalSuite) -> None:
    if es.primary:
        for e in spec.evals:
            e.primary = False
    spec.evals.append(es)
    spec.save(p.spec_path)


@suite_app.command("ls")
def suite_ls(name: str) -> None:
    for e in _spec(name)[1].evals:
        where = e.source.dataset if e.source else e.path or "own dev/test"
        log(f"{e.name:12s} {e.kind:8s} {where}{'  (primary)' if e.primary else ''}")


@baseline_app.command("add")
def baseline_add(
    name: str,
    baseline: str,
    score: Annotated[bool, typer.Option("--score", help="score it on every suite right away")] = False,
) -> None:
    """Add a baseline: a registered name (see `byosm baseline available`), hf:<repo> or llm:<role>."""
    from ..predictors import get_baseline

    p, spec = _spec(name)
    try:
        get_baseline(baseline, spec)
    except (KeyError, ValueError, OSError) as e:
        raise typer.BadParameter(str(e)) from e
    if baseline not in spec.baselines:
        spec.baselines.append(baseline)
        spec.save(p.spec_path)
    log(f"{name} baselines: {spec.baselines}")
    if score:
        from .. import evals

        evals.run(name, None, [baseline], None, False, log)


@baseline_app.command("available")
def baseline_available() -> None:
    from ..predictors import BASELINES

    for k, c in sorted(BASELINES.items()):
        log(f"{k:10s} {'/'.join(c.modalities) or 'any'} {'/'.join(c.task_types)}  {c.summary}")
