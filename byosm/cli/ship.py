from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from ..config import PUBLISH_LICENSE, Project, find_workspace, list_projects
from .base import _spec, app, console, log


@app.command()
def export(
    name: str,
    exp: str | None = None,
    out: Path | None = None,
    precision: Annotated[str, typer.Option(help="auto (int8 if it costs <= 0.5 pt on calib) | int8 | fp32")] = "auto",
) -> None:
    """Export to a torch-free bundle: ONNX + manifest + byosm_rt.py (encode / predict / stream().step / serve)."""
    from ..export import export as run

    d = run(name, exp, out, log, precision)
    log(f"exported to {d}\n  python {d / 'byosm_rt.py'} predict {d} <input>\n  python {d / 'byosm_rt.py'} serve {d}")


@app.command()
def publish(
    name: str,
    repo: Annotated[str, typer.Option(help="Hugging Face repo id, e.g. you/byosm-vad")],
    license: Annotated[
        str, typer.Option(help="model licence for the card (check your training data's licences)")
    ] = PUBLISH_LICENSE,
    public: Annotated[bool, typer.Option("--public", help="create the repo public (default: private)")] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="stage files + model card locally, upload nothing")
    ] = False,
) -> None:
    """Push the exported bundle to the Hugging Face Hub with a generated model card:
    what it does, how it was built, test eval per suite, comparison with every baseline, lineage, usage."""
    from ..hub import publish as run

    stage = run(name, repo, license, not public, dry_run, log)
    log(f"model card: {stage / 'README.md'}")


@app.command()
def predict(name: str, input: str) -> None:
    """Run the exported (or promoted) specialist on one input: text, or a path to an audio/image file."""
    from .. import rt

    p, _ = _spec(name)
    if not (p.root / "export" / "manifest.json").exists():
        raise typer.BadParameter(f"export first: byosm export {name}")
    rt._main(["predict", str(p.root / "export"), input])


@app.command()
def serve(names: list[str], port: int = 8800, host: str = "127.0.0.1") -> None:
    """Local HTTP API over exported bundles: GET /v1/models, POST /v1/<name>/predict|encode, POST /v1/predict."""
    from .. import rt

    rt.serve([rt.load(Project(n).root / "export") for n in names], host, port)


@app.command("ls")
def ls() -> None:
    """Specialists in this workspace."""
    t = Table()
    for c in ("name", "modality", "backbone", "experiments", "promoted", "exported"):
        t.add_column(c)
    from ..spec import TaskSpec

    for n in list_projects():
        p = Project(n)
        s = TaskSpec.load(p.spec_path)
        exported = "yes" if (p.root / "export" / "manifest.json").exists() else "-"
        t.add_row(n, f"{s.modality}/{s.task_type}", s.backbone, ", ".join(p.experiments()), p.promoted or "-", exported)
    console.print(t)
    log(f"workspace: {find_workspace()}")


@app.command()
def bench(names: list[str], generalist_backbone: str | None = None) -> None:
    """Thesis experiment: N specialists vs one shared-backbone generalist on the same held-out tests."""
    from ..bench import run

    out = run(names, generalist_backbone, log)
    t = Table(title="specialists vs generalist (held-out test objective)")
    for c in ("task", "metric", "specialist", "generalist", "delta"):
        t.add_column(c)
    for r in out["tasks"]:
        delta = f"{r['specialist'] - r['generalist']:+.4f}"
        t.add_row(r["task"], r["metric"], f"{r['specialist']:.4f}", f"{r['generalist']:.4f}", delta)
    console.print(t)
    lat = out["latency_all_tasks_ms"]
    log(
        f"params: specialists {out['specialists_total_params'] / 1e6:.2f}M vs generalist "
        f"{out['generalist_params'] / 1e6:.2f}M ({out['generalist_backbone']})"
    )
    log(
        f"all tasks for one input (cpu): specialists in parallel {lat['specialists_parallel_p50']:.2f}ms "
        f"vs generalist {lat['generalist_p50']:.2f}ms"
    )
