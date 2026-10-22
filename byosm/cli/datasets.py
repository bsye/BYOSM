from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from .base import _spec, app, log


@app.command("request-data")
def request_data(name: str) -> None:
    """Write a data request: a template (csv and xlsx), the rows needed per label and why."""
    from ..config import load_config
    from ..data.request import write_request

    p, spec = _spec(name)
    out = write_request(p.root, spec, "Requested by hand.", load_config(p.ws))
    log(f"wrote {out}: fill template.xlsx (or template.csv), then `byosm check-data {name} <file>`")


@app.command("check-data")
def check_data(
    name: str,
    file: Path,
    judge: Annotated[bool, typer.Option(help="ask the LLM to review rows the probe flags")] = True,
) -> None:
    """Check a filled table (format, coverage, leakage, likely mislabels) and train on it if it is usable."""
    from ..config import load_config
    from ..data.check import check
    from ..data.request import REQUEST_DIR

    p, spec = _spec(name)
    report = check(spec, p.root, file.resolve(), judge, load_config(p.ws), log)
    log(f"verdict: {report['verdict']} (report: {p.root / REQUEST_DIR / 'report.md'})")
    if report["verdict"] == "fix":
        raise typer.Exit(1)
    spec.data.local_dir = str(file.resolve())
    p.save_spec(spec)
    log(f"{name} now trains on {file}; next: `byosm auto {name}`")
