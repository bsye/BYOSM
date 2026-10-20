from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..config import Project
from ..evals.selection import calibrated_threshold
from ..runtime import Specialist
from ..spec import TaskSpec
from .bundle import check_out_dir, clear_out_dir, manifest_for, seal, write_bundle
from .frontends import build_frontend, frontend_kind
from .graph import export_model, export_stream
from .quantize import quantize
from .verify import verify

__all__ = ["export"]

QUANTIZE_MIN_BYTES = 4e6
REPORTED_METRICS = ("precision", "recall", "roc_auc")


def _suite_metrics(v: dict) -> dict:
    extra = {k: round(v["metrics"][k], 4) for k in REPORTED_METRICS if v["metrics"].get(k) is not None}
    return {"objective": round(v["objective"], 4), **extra}


def _calibrated_decision(p: Project, exp: str, base: TaskSpec) -> tuple[dict, dict]:
    board = p.board()
    d = base.decision.model_dump(mode="json")
    if d.get("threshold") is None:
        d["threshold"] = calibrated_threshold(p, exp)
    return d, {s: _suite_metrics(v) for s, v in board.get(exp, {}).items()}


def export(
    name: str,
    exp: str | None = None,
    out: Path | None = None,
    log: Callable[[str], None] = print,
    precision: str = "auto",
) -> Path:
    p = Project(name)
    exp = exp or p.promoted
    if not exp:
        raise RuntimeError("nothing promoted yet; pass --exp")
    out = out or (p.root / "export")
    check_out_dir(out)
    sp = Specialist(p.exp_dir(exp), "cpu")
    base = TaskSpec.load(p.spec_path)
    decision, metrics = _calibrated_decision(p, exp, base)
    kind = frontend_kind(sp)
    clear_out_dir(out)
    frontend, names, example, dyn = build_frontend(sp, kind, out)
    export_model(sp, kind, names, example, dyn, out, log)
    if kind == "logmel":
        export_stream(sp, out, log)
    write_bundle(out, sp, kind, manifest_for(sp, exp, decision, frontend, metrics))
    seal(out, lambda: _check(out, sp, p, base, kind, log, precision))
    return out


def _check(out: Path, sp: Specialist, p: Project, base: TaskSpec, kind: str, log: Callable, precision: str) -> None:
    verify(out, sp, p, kind, log)
    if precision != "fp32" and kind != "logmel" and (out / "model.onnx").stat().st_size > QUANTIZE_MIN_BYTES:
        quantize(out, sp, p, base, log, force=precision == "int8")
