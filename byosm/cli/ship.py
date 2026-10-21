from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from ..config import PUBLISH_LICENSE, Project, find_workspace, list_projects
from .base import _spec, app, console, log

MODEL_DOCKERFILE = Path(__file__).resolve().parents[2] / "docker" / "model.Dockerfile"
MODEL_PORT = 8800


def _bundle(name: str) -> Path:
    from ..export.bundle import is_exported

    p, _ = _spec(name)
    if not is_exported(p.root / "export"):
        raise typer.BadParameter(f"export first: byosm export {name}")
    return p.root / "export"


@app.command()
def export(
    name: str,
    exp: str | None = None,
    out: Path | None = None,
    precision: Annotated[str, typer.Option(help="auto (int8 if it costs <= 0.5 pt on calib) | int8 | fp32")] = "auto",
) -> None:
    """Export to a torch-free bundle: ONNX + manifest + byosm_rt.py (encode / predict / stream().step / serve)."""
    from ..export import export as run

    try:
        d = run(name, exp, out, log, precision)
    except FileExistsError as e:
        raise typer.BadParameter(str(e), param_hint="--out") from e
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

    rt._main(["predict", str(_bundle(name)), input])


@app.command()
def containerize(
    name: str, tag: Annotated[str | None, typer.Option(help="image tag (default: byosm-<name>)")] = None
) -> None:
    """Build a Docker image that serves one exported bundle in isolation: python-slim, the bundle's requirements, non-root."""
    import shutil
    import subprocess

    bundle = _bundle(name)
    docker = shutil.which("docker")
    if not docker or not MODEL_DOCKERFILE.exists():
        log(f"needs the docker CLI on PATH and {MODEL_DOCKERFILE} (a source checkout of byosm)")
        raise typer.Exit(1)
    image = tag or f"byosm-{name.lower()}"
    built = subprocess.run([docker, "build", "-f", str(MODEL_DOCKERFILE), "-t", image, str(bundle)])
    if built.returncode:
        raise typer.Exit(built.returncode)
    log(
        f"built {image}\n  docker run --rm --read-only -p {MODEL_PORT}:{MODEL_PORT} {image}\n"
        f'  curl -d \'{{"input": "..."}}\' http://127.0.0.1:{MODEL_PORT}/v1/{name}/predict'
    )


@app.command()
def serve(names: list[str], port: int = MODEL_PORT, host: str = "127.0.0.1") -> None:
    """Local HTTP API over exported bundles: GET /v1/models, POST /v1/<name>/predict|encode, POST /v1/predict."""
    from .. import rt

    rt.serve([rt.load(Project(n).root / "export") for n in names], host, port)


@app.command("ls")
def ls() -> None:
    """Specialists in this workspace."""
    t = Table()
    for c in ("name", "modality", "backbone", "experiments", "promoted", "exported"):
        t.add_column(c)
    from ..export.bundle import is_exported
    from ..spec import TaskSpec

    for n in list_projects():
        p = Project(n)
        s = TaskSpec.load(p.spec_path)
        exported = "yes" if is_exported(p.root / "export") else "-"
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
