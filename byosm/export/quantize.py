from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Project
from ..modalities import get_modality
from ..runtime import Specialist
from .bundle import load_runtime

VARIANTS = (["MatMul", "Gemm", "Gather"], ["MatMul", "Gemm"])
MAX_OBJECTIVE_DROP = 0.005
CALIB_ROWS = 400

Evaluator = Callable[[str], dict]


def _calib_evaluator(out: Path, sp: Specialist, p: Project, base: Any, log: Callable[[str], None]) -> Evaluator | None:
    from .. import evals, metrics

    if sp.spec.task_type != "classify":
        return None
    suite = next(s for s in base.evals if s.name == base.primary_suite())
    calib, _, root = evals.materialize(p, base, suite, log)
    rows = calib[:CALIB_ROWS]
    mod = get_modality(sp.spec, str(root))
    rt = load_runtime(out)
    idx = sp.spec.label_index
    y = np.array([idx[r["label"]] for r in rows])
    thr = json.loads((out / "manifest.json").read_text())["decision"].get("threshold")
    xs = [str(mod._path(r)) if sp.spec.modality == "document" else mod.load(r) for r in rows]

    def evaluate(f: str) -> dict:
        m = rt.Specialist(out, threads=4, model_file=f)
        m._run(xs[0])
        t0 = time.time()
        probs = np.stack([m._run(x)[1] for x in xs])
        return {
            "objective": metrics.compute(sp.spec, y, probs, thr)["objective"],
            "ms": (time.time() - t0) / len(xs) * 1000,
        }

    return evaluate


def _search(
    out: Path, evaluate: Evaluator | None, fp: dict | None, force: bool
) -> tuple[dict | None, list[str], dict[str, dict]]:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    tried: dict[str, dict] = {}
    for ops in VARIANTS:
        quantize_dynamic(
            str(out / "model.onnx"),
            str(out / "model.int8.onnx"),
            weight_type=QuantType.QInt8,
            per_channel=True,
            op_types_to_quantize=ops,
        )
        q = evaluate("model.int8.onnx") if evaluate else None
        tried["+".join(ops)] = {**(q or {}), "mb": (out / "model.int8.onnx").stat().st_size / 1e6}
        if force or (fp is not None and q["objective"] >= fp["objective"] - MAX_OBJECTIVE_DROP):
            return q, ops, tried
    return None, ops, tried


def quantize(out: Path, sp: Specialist, p: Project, base: Any, log: Callable[[str], None], force: bool = False) -> None:
    evaluate = _calib_evaluator(out, sp, p, base, log)
    fp = evaluate("model.onnx") if evaluate else None
    fp_mb = (out / "model.onnx").stat().st_size / 1e6
    q, ops, tried = _search(out, evaluate, fp, force)
    man = json.loads((out / "manifest.json").read_text())
    info = {"fp32": {**(fp or {}), "mb": fp_mb}, "tried": tried}
    if force or q is not None:
        (out / "model.onnx").unlink()
        man.update(model_file="model.int8.onnx", precision="int8", quantization={**info, "ops": ops})
        t = tried["+".join(ops)]
        detail = (
            f"calib {base.objective.metric} {fp['objective']:.4f} -> {q['objective']:.4f}, "
            f"{fp['ms']:.0f} -> {q['ms']:.0f} ms/item, "
            if fp and q
            else ""
        )
        log(f"  int8 ({'+'.join(ops)}): {detail}{fp_mb:.0f} -> {t['mb']:.0f} MB -> shipping int8")
    else:
        (out / "model.int8.onnx").unlink()
        man.update(precision="fp32", quantization={**info, "rejected": True})
        log("  int8 rejected (every variant loses more than 0.5 pt on calib) -> shipping fp32")
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
