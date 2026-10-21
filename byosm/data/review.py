from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..spec import TaskSpec
from .store import read, write

REVIEW_FILE = "panel_review.jsonl"
REVIEW_ROWS = 200
REVIEWED_SOURCE = "reviewed"


def reviewed_labels(path: Path) -> dict[str, str]:
    return {str(r["input"]): r["label"] for r in read(path)}


def save_review(path: Path, input: str, label: str) -> dict:
    row = {"input": input, "label": label, "time": time.time()}
    write(path, [*(r for r in read(path) if str(r["input"]) != input), row])
    return row


def review_rows(spec: TaskSpec, pool: list[dict], post: np.ndarray, votes: dict[str, np.ndarray]) -> list[dict]:
    names = spec.label_names
    order = np.argsort(post.max(1))[:REVIEW_ROWS]
    return [
        {
            "input": pool[i]["input"],
            "panel": names[int(post[i].argmax())],
            "sure": round(float(post[i].max()), 3),
            **{t: names[int(v[i].argmax())] for t, v in votes.items()},
        }
        for i in order
    ]


def write_review(data_dir: Path, rows: list[dict]) -> None:
    write(data_dir / REVIEW_FILE, rows)


def load_review(data_dir: Path) -> list[dict]:
    return read(data_dir / REVIEW_FILE)
