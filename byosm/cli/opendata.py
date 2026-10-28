from __future__ import annotations

import typer

from ..series.corpus import CORPUS_GB, PRETRAIN_STEPS
from .base import app, log

opendata_app = typer.Typer(no_args_is_help=True, help="Open data: search portals, download and cache files.")
app.add_typer(opendata_app, name="opendata")


@opendata_app.command("fetch")
def fetch(url: str) -> None:
    """Download an open-data file once into the cache (zip archives are unpacked) and list its tables."""
    from ..opendata.cache import fetch as get

    for name in get(url, log)["files"]:
        log(f"- {name}")


@opendata_app.command("search")
def search(query: str) -> None:
    """Search the open-data portals (data.gov.ie, data.gov.uk, data.europa.eu) for tables."""
    from ..opendata.catalog import search as find

    for d in find(query):
        log(f"- [{d['source']}] {d['title']} ({d['licence']}): {', '.join(f['format'] for f in d['files'])}")


@opendata_app.command("find")
def find(request: str) -> None:
    """Find open data for a request: the LLM searches the web and names portal searches, tables and new portals."""
    from ..config import find_workspace
    from ..opendata.finder import find as look

    for d in look(find_workspace(), request, log)["datasets"]:
        log(f"- [{d['source']}] {d['title']}: {', '.join(f['format'] for f in d['files'])}")


forecast_app = typer.Typer(no_args_is_help=True, help="Forecasts: will an entity in a register end within N years.")
app.add_typer(forecast_app, name="forecast")


@forecast_app.command("new")
def forecast_new(name: str, spec: str) -> None:
    """Save a forecast from a YAML spec (source table, start and end columns, horizon, cutoffs, features)."""
    from pathlib import Path

    import yaml

    from ..config import find_workspace
    from ..tabular.task import create

    log(f"saved {create(find_workspace(), name, yaml.safe_load(Path(spec).read_text())).title}")


@forecast_app.command("run")
def forecast_run(name: str) -> None:
    """Build yearly snapshots, check every feature for leakage, and score gradient boosting against its rivals."""
    from ..config import find_workspace
    from ..tabular.task import run

    run(find_workspace(), name, log)


@forecast_app.command("publish")
def forecast_publish(
    name: str,
    repo: str = typer.Option(..., help="Hugging Face repo id, e.g. you/irish-company-closure"),
    public: bool = typer.Option(False, "--public", help="create the repo public (default: private)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="stage the files and card locally, upload nothing"),
) -> None:
    """Push a forecast's model, spec and results card to the Hugging Face Hub; the scored entities are never uploaded."""
    from ..config import find_workspace
    from ..tabular.publish import publish, stage

    if dry_run:
        log(f"staged {stage(find_workspace(), name)}")
        return
    publish(find_workspace(), name, repo, not public, log)


options_app = typer.Typer(no_args_is_help=True, help="Choose among options offered with each input, such as tools.")
app.add_typer(options_app, name="options")


@options_app.command("new")
def options_new(
    name: str,
    source: str = typer.Option(
        "lockon/xlam-function-calling-60k", help="a supported dataset of requests with offered tools"
    ),
    situations: int = typer.Option(6000, help="how many requests to turn into pairs"),
    held_out: bool = typer.Option(False, help="calib and test on tools and a layout that training never sees"),
) -> None:
    """Create a specialist that scores (request, option) pairs, so options never seen in training can be chosen."""
    from ..config import find_workspace
    from ..options.create import build_table, pair_plan, spec_for
    from ..project import Project

    ws = find_workspace()
    p = Project(name).ensure()
    plan = pair_plan(held_out)
    p.save_spec(spec_for(name, build_table(ws, name, source, situations, plan, log), source, plan))
    log(f"saved {name}: run `byosm build {name}` to train it")


@options_app.command("score")
def options_score(
    name: str, threshold: float = typer.Option(0.5, help="an option is chosen at or above this chance")
) -> None:
    """Score the exported specialist per request: exactly the right tools chosen, and no tool when none fits."""
    from ..config import find_workspace
    from ..options.select import score_export

    out = score_export(find_workspace(), name, threshold)
    for part in ("exact", "tool_needed", "no_tool"):
        s = out[part]
        log(f"  {part}: {s['accuracy']} ({s['ci'][0]} to {s['ci'][1]}) on {s['n']} requests")


@options_app.command("rival")
def options_rival(
    name: str,
    role: str = typer.Option("baseline", help="the LLM role in byosm.yaml to compare against"),
    sample: int = typer.Option(200, help="how many held-out requests to compare on"),
) -> None:
    """Compare the exported specialist with an LLM choosing tools for the same held-out requests."""
    from ..config import find_workspace, load_config
    from ..options.create import table_path
    from ..options.rivals import rival_scores
    from ..options.select import exported_chance, save

    ws = find_workspace()
    out = rival_scores(table_path(ws, name), load_config(ws), exported_chance(name), sample, role)
    save(ws, name, compare=out)
    for who, s in [("this model", out["model"]), *out["rivals"].items()]:
        log(
            f"  {who}: exact {s['exact']['accuracy']} ({s['exact']['ci'][0]} to {s['exact']['ci'][1]}) on {out['sample']} requests"
        )


MAX_ROUNDS = 4
series_app = typer.Typer(no_args_is_help=True, help="Time series: specialists against Timestral on GIFT-Eval.")
app.add_typer(series_app, name="series")


@series_app.command("run")
def series_run(
    domain: str = typer.Argument(..., help="energy, sales, operations, or one GIFT-Eval dataset such as solar/H"),
    until_rival: bool = typer.Option(
        False,
        "--until-rival",
        help="Keep adding versions until one beats the best rival, trying untried prediction-time techniques on the promoted models before training more",
    ),
    rounds: int = typer.Option(MAX_ROUNDS, help="Most versions to add with --until-rival"),
) -> None:
    """Train a new version of a forecasting specialist with twice the training of the promoted one and its prediction options, and score it against seasonal naive and Timestral."""
    from ..config import find_workspace
    from ..series.run import run_domain
    from ..series.run import until_rival as loop

    if until_rival:
        loop(find_workspace(), domain, rounds, log)
    else:
        run_domain(find_workspace(), domain, log)


@series_app.command("corpus")
def series_corpus(
    gb: float = typer.Option(CORPUS_GB, help="Gigabytes to sample from the GIFT-Eval pretraining corpus"),
) -> None:
    """Download a balanced sample of GIFT-Eval's pretraining corpus, which shares no series with its test sets."""
    from ..series.corpus import download

    download(gb, log)


@series_app.command("pretrain")
def series_pretrain(
    steps: int = typer.Option(PRETRAIN_STEPS, help="Training steps on the GPU"),
    size: str = typer.Option("base", help="base (6M parameters) or large (21M)"),
) -> None:
    """Pretrain the base forecaster on GIFT-Eval's pretraining corpus and the training portions of its datasets, never their test windows."""
    from ..series.pretrain import pretrain

    pretrain(log, steps, size)
