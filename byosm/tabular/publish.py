from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from ..config import PUBLISH_LICENSE, STATE_DIR

HUB_FOLDER = "hub"
KEPT_RESULTS = ("splits", "scores", "over_time", "leakage", "dropped")
CARD = """---
{front}---

# {title}

A forecast built with [BYOSM](https://github.com/bsye): will an entity in a register end within {horizon}?
It is a gradient-boosted model on yearly snapshots of the register, tested on later years than it was trained on.

## Test results

Tested on the entities alive on 1 January {test}, with 95% bootstrap confidence intervals.

| Model | AUC | 95% CI | Average precision |
|---|---|---|---|
{scores}

## Accuracy by test year

| Year | Gradient boosting | Logistic regression |
|---|---|---|
{years}

## Left out

These columns were left out because they leak the answer (written after the outcome):

{left_out}

## Data

Source: {source} ({licence}). Each snapshot keeps the entities alive on 1 January of a year and labels whether they end
within the horizon; train, calibration and test use different years.

## Files

- `model.joblib`: the scikit-learn model used to score today, with its categories and typical values.
- `spec.yaml`: the forecast spec (columns, horizon, years).
- `results.json`: every number on this card.
"""


def _licence(ws: Path, url: str) -> str:
    from ..opendata.library import entries

    return next((e["licence"] for e in entries(ws) if e["file"]["url"] == url and e.get("licence")), "see the source")


def card(ws: Path, row: dict[str, Any], license: str) -> str:
    spec, r = row["spec"], row["result"]
    scores = "\n".join(
        f"| {m} | {s['auc']:.3f} | {s['auc_ci'][0]:.3f} to {s['auc_ci'][1]:.3f} | {s['ap']:.3f} |"
        for m, s in r["scores"].items()
    )
    years = "\n".join(
        f"| {y['year']} | {y['scores']['gradient boosting']['auc']:.3f} | {y['scores']['logistic regression']['auc']:.3f} |"
        for y in r.get("over_time", [])
    )
    left = {**spec.get("excluded", {}), **r.get("dropped", {})}
    front = yaml.safe_dump(
        {"license": license, "tags": ["byosm", "forecast", "tabular-classification"], "library_name": "sklearn"}
    )
    horizon = "a year" if spec["horizon_years"] == 1 else f"{spec['horizon_years']} years"
    return CARD.format(
        front=front,
        title=spec["title"],
        horizon=horizon,
        test=spec["cutoffs"]["test"][0],
        scores=scores,
        years=years or "| - | - | - |",
        left_out="\n".join(f"- `{c}`: {why}" for c, why in left.items()) or "- none",
        source=spec["source"],
        licence=_licence(ws, spec["source"]),
    )


def stage(ws: Path, name: str, license: str = PUBLISH_LICENSE) -> Path:
    from .task import _row, model_path

    row = _row(ws, name)
    if not row or not row.get("result") or not model_path(ws, name).exists():
        raise RuntimeError(f"run the forecast first: byosm forecast run {name}")
    out = ws / STATE_DIR / "forecasts" / HUB_FOLDER / name
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    shutil.copy(model_path(ws, name), out / "model.joblib")
    (out / "spec.yaml").write_text(yaml.safe_dump(row["spec"], sort_keys=False))
    (out / "results.json").write_text(json.dumps({k: row["result"].get(k) for k in KEPT_RESULTS}, indent=1))
    (out / "README.md").write_text(card(ws, row, license))
    return out


def publish(
    ws: Path, name: str, repo: str, private: bool, log: Callable[[str], None], license: str = PUBLISH_LICENSE
) -> str:
    from huggingface_hub import HfApi

    out = stage(ws, name, license)
    api = HfApi()
    api.create_repo(repo, private=private, exist_ok=True)
    if private and not api.model_info(repo).private:
        raise RuntimeError(f"{repo} already exists and is public; pass --public to publish to it")
    api.upload_folder(folder_path=str(out), repo_id=repo, commit_message=f"BYOSM forecast {name}")
    log(f"  published https://huggingface.co/{repo} ({'private' if private else 'public'})")
    return repo
