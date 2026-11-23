from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .infosets import RECORDS, described, records
from .teacher import STRATEGY, run, strategy, work_dir

STUDENT = "-student"
MODEL = ".student.pt"
TABLE = ".student.probs"
REPORT = "poker-student:"
MB = 1024 * 1024
Log = Callable[[str], None]


def build(ws: Path, hands_dir: Path, name: str, log: Log) -> dict:
    import torch

    from ...config import device
    from ...db.workspace import save_workspace_report
    from ...train.gpu import gpu_slot
    from .student import fit, table

    meta, found = described(ws, name), records(ws, name)
    span = int(found["bucket"].max()) + 1
    folder, student = work_dir(ws), f"{name}{STUDENT}"
    with gpu_slot(device(), log) as dev:
        net, score = fit(found, span, log, dev)
        log(f"  writing the student's move probabilities for all {len(found)} situations")
        table(net, found, folder / f"{name}{TABLE}", dev)
    torch.save(net.state_dict(), folder / f"{name}{MODEL}")
    files = ["--like", strategy(ws, name), "--records", f"{name}{RECORDS}", "--probs", f"{name}{TABLE}"]
    shape = ["--width", str(meta["width"]), "--hist", str(meta["hist"])]
    run(ws, hands_dir, ["student", *files, *shape, "--out", f"{student}{STRATEGY}"], log)
    size = (folder / f"{name}{MODEL}").stat().st_size / MB
    result = {**score, "student": student, "teacher": name, "model_mb": round(size, 2)}
    save_workspace_report(ws, f"{REPORT}{student}", result)
    return result
