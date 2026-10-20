from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from ..config import Project, load_config
from .loop import improve_until_stop
from .report import _report
from .standing import standing
from .state import Log, Session


def _define(p: Project, log: Log, prompt: str | None, modality: str | None, cfg: Any, echo: Callable) -> None:
    from ..discover import discover
    from ..spec import compile_spec

    if not prompt:
        raise ValueError(f"{p.name} doesn't exist yet: pass the task prompt")
    spec = compile_spec(p.name, prompt, modality, cfg)
    log(
        "define",
        f"{spec.modality}/{spec.task_type}, {len(spec.labels)} labels, backbone {spec.backbone}",
        f"compiled from the prompt; objective {spec.objective.metric} >= {spec.objective.target}",
    )
    spec, decisions = discover(spec, spec.search, cfg, echo)
    for d in decisions:
        log("discover:" + d["step"], d["choice"], d["why"])
    p.ensure()
    spec.save(p.spec_path)


def _build_v1(s: Session) -> None:
    from ..pipeline import build

    ts = time.time()
    build(s.name, "v1", s.echo)
    s.durations.append(time.time() - ts)
    st = standing(s.name, s.margin)
    s.log("build", "v1", f"{st['suite']} {st['metric']} {st['ours']:.4f}; best baseline {st['best_baseline']}")


def _distill_student(s: Session, st: dict, meta: dict) -> None:
    from .. import distill as dst
    from ..pipeline import build

    teacher = st["exp"]
    exp = dst.create(s.name, teacher)
    build(s.name, exp, s.echo)
    stu = s.p.board().get(exp, {}).get(st["suite"], {}).get("objective")
    smeta = json.loads((s.p.exp_dir(exp) / "meta.json").read_text())
    still = stu is not None and (st["best_baseline"] is None or stu > st["best_baseline"][1] + s.margin)
    if still:
        Project(s.name).promote(exp)
    outcome = (
        "student still beats every baseline -> promoted" if still else "student no longer wins -> keep the teacher"
    )
    s.log(
        "distill",
        exp,
        f"{meta['params'] / 1e6:.1f}M teacher {teacher} {st['ours']:.4f} -> "
        f"{smeta['params'] / 1e6:.1f}M student {stu:.4f}; " + outcome,
    )


def _distill(s: Session, max_params_m: float) -> None:
    st = standing(s.name, s.margin)
    if not st["exp"]:
        return
    meta = json.loads((s.p.exp_dir(st["exp"]) / "meta.json").read_text())
    large = meta["params"] / 1e6 > max_params_m
    if meta.get("distilled_from"):
        s.log("distill", None, "already a distilled student")
    elif not large:
        s.log("distill", None, f"not needed: {meta['params'] / 1e6:.1f}M <= {max_params_m}M")
    elif time.time() + s.mean_duration() <= s.deadline:
        _distill_student(s, st, meta)
    else:
        s.log("distill", None, "skipped: not enough time budget left")


def _ship(s: Session, publish: str | None, license: str) -> None:
    from ..export import export

    out = export(s.name, log=s.echo)
    s.log("export", str(out.relative_to(s.p.ws)), "torch-free ONNX bundle (encode / predict / serve)")
    if publish:
        from ..hub import publish as pub

        pub(s.name, publish, license, private=True, log=s.echo)
        s.log("publish", publish, "private Hugging Face repo with generated model card (flip public on the Hub)")


def run(
    name: str,
    prompt: str | None = None,
    modality: str | None = None,
    budget_h: float = 6.0,
    max_experiments: int = 8,
    margin: float = 0.0,
    distill: bool = True,
    max_params_m: float = 25.0,
    publish: str | None = None,
    license: str = "other",
    use_llm: bool = True,
    echo: Callable[[str], None] = print,
) -> dict[str, Any]:
    from ..spec import TaskSpec

    p = Project(name)
    cfg = load_config(p.ws)
    log = Log(p, echo)
    t0 = time.time()
    if not p.exists():
        _define(p, log, prompt, modality, cfg, echo)
    s = Session(name, p, log, TaskSpec.load(p.spec_path), margin, t0 + budget_h * 3600, echo)
    if not p.experiments():
        _build_v1(s)
    reason = improve_until_stop(s, max_experiments, use_llm)
    log("stop", reason, "")
    if distill:
        _distill(s, max_params_m)
    _ship(s, publish, license)
    final = standing(name, margin)
    final["hours"] = round((time.time() - t0) / 3600, 2)
    final["stop_reason"] = reason
    log.save(final)
    _report(p, log.decisions, final)
    return final
