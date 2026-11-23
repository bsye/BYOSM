from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .infosets import RECORDS, described, reach, records
from .teacher import STRATEGY, run, strategy, work_dir

STUDENT = "-student"
ROUND = "-r"
REACH_HANDS = 2_000_000
MODEL = ".pt"
TABLE = ".probs"
REPORT = "poker-student:"
MB = 1024 * 1024
Log = Callable[[str], None]


def build(ws: Path, hands_dir: Path, name: str, log: Log, rounds: int = 0) -> dict:
    import numpy as np
    import torch

    from ...config import device
    from ...db.workspace import save_workspace_report
    from ...train.gpu import gpu_slot
    from .student import fit, table

    meta, found = described(ws, name), records(ws, name)
    span = int(found["bucket"].max()) + 1
    folder, student = work_dir(ws), f"{name}{STUDENT}{f'{ROUND}{rounds}' if rounds else ''}"
    files = ["--like", strategy(ws, name), "--records", f"{name}{RECORDS}", "--probs", f"{student}{TABLE}"]
    shape = ["--width", str(meta["width"]), "--hist", str(meta["hist"])]
    visits = np.asarray(found["visits"]).astype(np.int64)
    for round_ in range(rounds + 1):
        if round_:
            log(f"  round {round_}/{rounds}: seating the student with five teacher copies to see what it reaches")
            visits = visits + reach(ws, hands_dir, name, student, round_, REACH_HANDS, log)
        with gpu_slot(device(), log) as dev:
            net, score = fit(found, span, log, dev, visits=visits)
            log(f"  writing the student's move probabilities for all {len(found)} situations")
            table(net, found, folder / f"{student}{TABLE}", dev)
        torch.save(net.state_dict(), folder / f"{student}{MODEL}")
        run(ws, hands_dir, ["student", *files, *shape, "--out", f"{student}{STRATEGY}"], log)
    size = (folder / f"{student}{MODEL}").stat().st_size / MB
    result = {**score, "student": student, "teacher": name, "model_mb": round(size, 2), "rounds": rounds}
    save_workspace_report(ws, f"{REPORT}{student}", result)
    return result
