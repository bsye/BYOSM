"""byojl - tiny System-1 specialists, from a sentence to a local model.

  define     byojl new vad "voice activity detection for a voice agent..."      -> spec.yaml (labels, objective, data plan)
             byojl suite add vad ava --hf nccratliri/vad-human-ava-speech --format segment_files
             byojl baseline add vad silero
  train      byojl build vad                     -> data (cached) -> LeJEPA -> train -> eval every suite -> promote
  compare    byojl board vad                     -> experiments vs baselines on every suite
  improve    byojl improve vad                   -> diagnoses + proposals;  --apply a,b --run  -> next experiment
  ship       byojl export vad                    -> export/vad: ONNX + manifest + byojl_rt.py (numpy+onnxruntime only)
             byojl publish vad --repo you/byojl-vad   -> Hugging Face: bundle + model card (eval, comparison, lineage)
             byojl serve vad eot --port 8800     -> local HTTP API over exported bundles
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import Project, find_workspace, list_projects, write_default_config

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
suite_app = typer.Typer(no_args_is_help=True, help="Manage eval suites (frozen test sets) of a specialist.")
baseline_app = typer.Typer(no_args_is_help=True, help="Manage baselines (competitors scored on every suite).")
app.add_typer(suite_app, name="suite")
app.add_typer(baseline_app, name="baseline")
console = Console()


def log(msg: str) -> None:
    console.print(msg, highlight=False, markup=False)


def _spec(name: str):  # noqa: ANN202
    from .spec import TaskSpec

    p = Project(name)
    if not p.exists():
        raise typer.BadParameter(f"no specialist {name!r} in {p.ws / 'specialists'}")
    return p, TaskSpec.load(p.spec_path)


# --------------------------------------------------------------------------------------------- define


@app.command()
def init() -> None:
    """Create byojl.yaml (LLM providers per role) in the current directory."""
    log(f"wrote {write_default_config(Path.cwd())} - set providers/models per role: spec, generate, judge, eval, analyze, baseline")


@app.command()
def new(name: str, description: str, modality: Optional[str] = typer.Option(None, help="text | audio | image (inferred if omitted)")) -> None:
    """Compile a plain-language task into specialists/<name>/spec.yaml."""
    from .spec import compile_spec

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
    log(f"\nspec: {p.spec_path}\nnext: review it, optionally `byojl suite add {name} ...`, then `byojl build {name}`")


def _show_spec(spec) -> None:  # noqa: ANN001
    t = Table(show_header=False, box=None, pad_edge=False)
    t.add_row("[bold]task", f"{spec.name}: {spec.modality}/{spec.task_type} - {spec.summary}")
    t.add_row("[bold]labels", "\n".join(f"[cyan]{l.name}[/]: {l.definition}" for l in spec.labels))
    o = spec.objective
    t.add_row("[bold]objective", f"{o.metric} >= {o.target}" + (f" (constraint {o.constraint})" if "_at_" in o.metric else ""))
    t.add_row("[bold]backbone", spec.backbone)
    t.add_row("[bold]data", f"synthetic={spec.data.synthetic} train={spec.data.train_size} sources={[s.dataset for s in spec.data.hf_sources]} queries={spec.data.hf_queries}")
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
    hf: Optional[str] = typer.Option(None, help="Hugging Face dataset id"),
    format: str = typer.Option("rows", help="rows | segment_files"),
    config: Optional[str] = None,
    split: str = "train",
    input_column: Optional[str] = None,
    label_column: Optional[str] = None,
    local: Optional[Path] = typer.Option(None, help="local jsonl with input/label[/split=calib|test]"),
    max_items: int = 400,
    max_files: int = 20,
    primary: bool = typer.Option(False, help="promotion is decided on this suite"),
) -> None:
    """Add a frozen eval suite. HF suites use the source's calib/test hash partitions (never its train partition)."""
    from .spec import EvalSuite, HFSource

    p, spec = _spec(name)
    if any(e.name == suite for e in spec.evals):
        raise typer.BadParameter(f"suite {suite} exists")
    if hf:
        src = HFSource(dataset=hf, format=format, config=config, split=split, input_column=input_column or ("audio" if format == "rows" and spec.modality == "audio" else None),
                       label_column=label_column, max_files=max_files, max_rows=max_items * 2)
        es = EvalSuite(name=suite, kind="hf", source=src, max_items=max_items, primary=primary)
    elif local:
        es = EvalSuite(name=suite, kind="local", path=str(local.resolve()), primary=primary)
    else:
        raise typer.BadParameter("pass --hf or --local")
    if primary:
        for e in spec.evals:
            e.primary = False
    spec.evals.append(es)
    spec.save(p.spec_path)
    log(f"added suite {suite} to {name}; it is materialized and frozen on the next `byojl eval {name}`")


@suite_app.command("ls")
def suite_ls(name: str) -> None:
    for e in _spec(name)[1].evals:
        log(f"{e.name:12s} {e.kind:8s} {(e.source.dataset if e.source else e.path or 'own dev/test')}{'  (primary)' if e.primary else ''}")


@baseline_app.command("add")
def baseline_add(name: str, baseline: str) -> None:
    """Add a baseline (see `byojl baseline available`)."""
    from .predictors import BASELINES

    p, spec = _spec(name)
    if baseline not in BASELINES:
        raise typer.BadParameter(f"unknown baseline; available: {sorted(BASELINES)}")
    if baseline not in spec.baselines:
        spec.baselines.append(baseline)
        spec.save(p.spec_path)
    log(f"{name} baselines: {spec.baselines}")


@baseline_app.command("available")
def baseline_available() -> None:
    from .predictors import BASELINES

    for k, c in sorted(BASELINES.items()):
        log(f"{k:10s} {'/'.join(c.modalities) or 'any'} {'/'.join(c.task_types)}  {(c.__doc__ or '').strip().splitlines()[0]}")


# --------------------------------------------------------------------------------------------- train / eval


@app.command()
def build(name: str, exp: Optional[str] = typer.Option(None, help="experiment name (default: next vN)"), force_data: bool = False) -> None:
    """Data -> LeJEPA -> train -> score on every suite (+ baselines) -> promote if the primary suite improves."""
    from .pipeline import build as run

    run(name, exp, log, force_data)
    board(name)


@app.command()
def data(name: str, exp: Optional[str] = None, force: bool = False) -> None:
    """Only build (or show) the dataset an experiment would train on."""
    from .config import load_config
    from .data.store import Dataset
    from .pipeline import ensure_data, exp_spec

    p, _ = _spec(name)
    spec = exp_spec(p, exp) if exp else _spec(name)[1]
    dd = ensure_data(p, spec, load_config(p.ws), log, force)
    log(json.dumps({s: v["n"] for s, v in Dataset(dd).summary().items()}))


@app.command("eval")
def eval_(name: str, exp: list[str] = typer.Option(None, help="only these experiments"), suite: list[str] = typer.Option(None),
          baseline: list[str] = typer.Option(None, help="only these baselines"), force: bool = False) -> None:
    """Score experiments and baselines on every suite (results cached by suite fingerprint)."""
    from . import evals

    evals.run(name, exp or None, baseline or None, suite or None, force, log)
    board(name)


@app.command()
def board(name: str, detail: bool = typer.Option(False, help="also show precision/recall per suite")) -> None:
    """Leaderboard: experiments vs baselines, objective per suite."""
    from . import evals

    p, spec = _spec(name)
    suites, rows = evals.board_rows(name)
    t = Table(title=f"{name} - {spec.objective.metric} per suite (primary: {spec.primary_suite()})")
    t.add_column("")
    t.add_column("params", justify="right")
    t.add_column("cpu", justify="right")
    for s in suites:
        t.add_column(s, justify="right")
        if detail:
            t.add_column("P/R", justify="right", style="dim")
        t.add_column("auc", justify="right", style="dim")
    best = {s: max((r[s] for r in rows if r.get(s) is not None), default=None) for s in suites}
    for r in rows:
        star = " *" if r["promoted"] else ""
        cpu = f"{r['stream_ms']:.3f}/{r['chunk_ms']:.0f}ms" if r.get("stream_ms") else (f"{r['latency_ms']:.2f}ms" if r.get("latency_ms") else "-")
        cells = [f"[{'green' if r['kind'] == 'experiment' else 'yellow'}]{r['id']}{star}[/]", f"{r['params'] / 1e6:.2f}M" if r.get("params") else "-", cpu]
        for s in suites:
            v = r.get(s)
            pr = r.get(f"{s}:p/r")
            auc = r.get(f"{s}:auc")
            cells.append("-" if v is None else (f"[bold]{v:.4f}[/]" if v == best[s] else f"{v:.4f}"))
            if detail:
                cells.append(f"{pr[0]:.3f}/{pr[1]:.3f}" if pr and pr[0] is not None else "")
            cells.append(f"{auc:.3f}" if auc else "")
        t.add_row(*cells)
    console.print(t)
    log("* = promoted (used by predict / export / serve)")


exp_app = typer.Typer(no_args_is_help=True, help="Create experiment variants (ablations) by editing spec paths.")
app.add_typer(exp_app, name="exp")


@exp_app.command("create")
def exp_create(name: str, exp: str, set_: list[str] = typer.Option(..., "--set", help="path=value (YAML value), e.g. document.route=visual"),
               base: Optional[str] = typer.Option(None, help="start from this experiment's spec (default: base spec.yaml)")) -> None:
    """e.g. byojl exp create doctype visual --set document.route=visual"""
    import yaml

    from .improve import apply_ops
    from .spec import TaskSpec

    p, spec = _spec(name)
    if base:
        spec = TaskSpec.load(p.exp_dir(base) / "spec.yaml")
    ops = []
    for kv in set_:
        k, v = kv.split("=", 1)
        ops.append({"op": "set", "path": k, "value": yaml.safe_load(v)})
    new = apply_ops(spec, ops)
    d = p.exp_dir(exp)
    d.mkdir(parents=True, exist_ok=True)
    new.save(d / "spec.yaml")
    (d / "proposal.json").write_text(json.dumps({"parent": base or "spec.yaml", "applied": [{"id": "manual", "why": " ".join(set_), "ops": ops}]}, indent=2))
    log(f"created {name}/{exp} ({', '.join(set_)}); data key {new.data_key()} -> byojl build {name} --exp {exp}")


@app.command()
def promote(name: str, exp: str) -> None:
    """Choose which experiment ships."""
    p, _ = _spec(name)
    if exp not in p.experiments():
        raise typer.BadParameter(f"unknown experiment; have {p.experiments()}")
    p.promote(exp)
    log(f"promoted {name}/{exp}")


@app.command()
def improve(name: str, apply: Optional[str] = typer.Option(None, help="comma-separated proposal ids"), run: bool = typer.Option(False, "--run", help="build the new experiment"),
            no_llm: bool = typer.Option(False, help="rule-based proposals only")) -> None:
    """Diagnose the board and propose the next experiment (spec edits with evidence)."""
    from . import improve as imp

    if apply:
        exp = imp.apply(name, [s.strip() for s in apply.split(",")])
        log(f"created experiment {exp} -> {Project(name).exp_dir(exp) / 'spec.yaml'}")
        if run:
            build(name, exp)
        else:
            log(f"next: byojl build {name} --exp {exp}")
        return
    dg, props = imp.propose(name, not no_llm, log)
    log(f"diagnosis of {dg['experiment']} ({dg['n_train']} train examples; dev curve {', '.join(f'{c:.2f}' for c in dg['curve'])}; best epoch {dg['best_epoch'] + 1}):")
    for s, r in dg["suites"].items():
        bb = r["best_baseline"]
        stats = f"P={r['precision']:.3f} R={r['recall']:.3f}" if dg["binary"] and r["precision"] is not None else f"acc={r['accuracy']:.3f} macro_f1={r['macro_f1']:.3f}"
        thr = f" thr={r['threshold']:.2f}" if r["threshold"] is not None else ""
        log(f"  {s:10s} objective={r['objective']:.4f} {stats}{thr}" + (f"  | best baseline {bb[0]}={bb[1]:.4f} (gap {bb[1] - r['objective']:+.4f})" if bb else ""))
        if not dg["binary"] and r["weakest"]:
            log(f"             weakest: {', '.join(f'{n} {f:.2f}' for n, f in r['weakest'])}")
        if not dg["binary"] and r["confusions"]:
            log(f"             confused: {', '.join(f'{g}->{p} ({c})' for g, p, c in r['confusions'][:5])}")
    if not props:
        log("no proposals - the board has no obvious gap to act on")
        return
    for pr in props:
        log(f"\n[{pr['id']}] {pr['why']}")
        for o in pr["ops"]:
            v = json.dumps(o["value"])
            log(f"    {o['op']} {o['path']} = {v[:160]}")
    log(f"\napply: byojl improve {name} --apply {','.join(p['id'] for p in props)} --run")


# --------------------------------------------------------------------------------------------- ship


@app.command()
def export(name: str, exp: Optional[str] = None, out: Optional[Path] = None) -> None:
    """Export to a torch-free bundle: ONNX + manifest + byojl_rt.py (encode / predict / stream().step / serve)."""
    from .export import export as run

    d = run(name, exp, out, log)
    log(f"exported to {d}\n  python {d / 'byojl_rt.py'} predict {d} <input>\n  python {d / 'byojl_rt.py'} serve {d}")


@app.command()
def publish(
    name: str,
    repo: str = typer.Option(..., help="Hugging Face repo id, e.g. you/byojl-vad"),
    license: str = typer.Option("other", help="model licence for the card (check your training data's licences)"),
    public: bool = typer.Option(False, "--public", help="create the repo public (default: private)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="stage files + model card locally, upload nothing"),
) -> None:
    """Push the exported bundle to the Hugging Face Hub with a generated model card:
    what it does, how it was built, test eval per suite, comparison with every baseline, lineage, usage."""
    from .hub import publish as run

    stage = run(name, repo, license, not public, dry_run, log)
    log(f"model card: {stage / 'README.md'}")


@app.command()
def predict(name: str, input: str) -> None:
    """Run the exported (or promoted) specialist on one input: text, or a path to an audio/image file."""
    from . import rt

    p, _ = _spec(name)
    if not (p.root / "export" / "manifest.json").exists():
        raise typer.BadParameter(f"export first: byojl export {name}")
    rt._main(["predict", str(p.root / "export"), input])


@app.command()
def serve(names: list[str], port: int = 8800, host: str = "127.0.0.1") -> None:
    """Local HTTP API over exported bundles: GET /v1/models, POST /v1/<name>/predict|encode, POST /v1/predict."""
    from . import rt

    rt.serve([rt.load(Project(n).root / "export") for n in names], host, port)


@app.command("ls")
def ls() -> None:
    """Specialists in this workspace."""
    t = Table()
    for c in ("name", "modality", "backbone", "experiments", "promoted", "exported"):
        t.add_column(c)
    from .spec import TaskSpec

    for n in list_projects():
        p = Project(n)
        s = TaskSpec.load(p.spec_path)
        t.add_row(n, f"{s.modality}/{s.task_type}", s.backbone, ", ".join(p.experiments()), p.promoted or "-", "yes" if (p.root / "export" / "manifest.json").exists() else "-")
    console.print(t)
    log(f"workspace: {find_workspace()}")


@app.command()
def bench(names: list[str], generalist_backbone: Optional[str] = None) -> None:
    """Thesis experiment: N specialists vs one shared-backbone generalist on the same held-out tests."""
    from .bench import run

    out = run(names, generalist_backbone, log)
    t = Table(title="specialists vs generalist (held-out test objective)")
    for c in ("task", "metric", "specialist", "generalist", "delta"):
        t.add_column(c)
    for r in out["tasks"]:
        t.add_row(r["task"], r["metric"], f"{r['specialist']:.4f}", f"{r['generalist']:.4f}", f"{r['specialist'] - r['generalist']:+.4f}")
    console.print(t)
    lat = out["latency_all_tasks_ms"]
    log(f"params: specialists {out['specialists_total_params'] / 1e6:.2f}M vs generalist {out['generalist_params'] / 1e6:.2f}M ({out['generalist_backbone']})")
    log(f"all tasks for one input (cpu): specialists in parallel {lat['specialists_parallel_p50']:.2f}ms vs generalist {lat['generalist_p50']:.2f}ms")


if __name__ == "__main__":
    app()
