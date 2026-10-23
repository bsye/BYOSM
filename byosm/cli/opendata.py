from __future__ import annotations

import typer

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
