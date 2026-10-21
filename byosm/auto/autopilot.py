from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from ..config import PUBLISH_LICENSE, Project, load_config
from ..data.request import missing_labels
from .loop import improve_until_stop
from .report import _report
from .standing import standing
from .state import Log, Session


def _compile(name: str | None, prompt: str | None, modality: str | None, cfg: Any) -> Any:
    from ..spec import compile_spec

    if not prompt:
        raise ValueError(f"{name} doesn't exist yet: pass the task prompt")
    return compile_spec(name, prompt, modality, cfg)


def _define(p: Project, log: Log, spec: Any, cfg: Any, echo: Callable) -> None:
    from ..discover import discover

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


def _request_data(s: Session, cfg: Any, t0: float) -> dict[str, Any]:
    from ..data.request import write_request

    missing = missing_labels(s.base)
    reason = f"No public dataset covers {', '.join(missing)}, and {s.base.modality} examples cannot be generated."
    out = write_request(s.p.root, s.base, reason, cfg)
    s.log("data", str(out.relative_to(s.p.ws)), reason + " Wrote a template and the rows needed per label.")
    stop = f"data needed: fill {out / 'template.xlsx'}, then `byosm check-data {s.name} <file>`"
    final = {"name": s.name, "won": False, "stop_reason": stop, "data_request": str(out)}
    final["hours"] = round((time.time() - t0) / 3600, 2)
    s.log.save(final)
    return final


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
    try:
        build(s.name, exp, s.echo)
    except Exception as ex:
        s.p.promote(teacher)
        s.log("distill", exp, f"student build failed ({type(ex).__name__}: {ex}) -> keep the teacher {teacher}")
        return
    smeta = s.p.meta(exp)
    s.p.promote(exp)
    stu = standing(s.name, s.margin)
    if not stu["won"]:
        s.p.promote(teacher)
    outcome = "student still wins -> promoted" if stu["won"] else "student no longer wins -> keep the teacher"
    ours = stu["ours"] if stu["ours"] is not None else float("nan")
    s.log(
        "distill",
        exp,
        f"{meta['params'] / 1e6:.1f}M teacher {teacher} {st['ours']:.4f} -> "
        f"{smeta['params'] / 1e6:.1f}M student {ours:.4f}; " + outcome,
    )


def _distill(s: Session, max_params_m: float) -> None:
    st = standing(s.name, s.margin)
    if not st["exp"]:
        return
    meta = s.p.meta(st["exp"])
    large = meta["params"] / 1e6 > max_params_m
    if meta.get("distilled_from"):
        s.log("distill", None, "already a distilled student")
    elif not large:
        s.log("distill", None, f"not needed: {meta['params'] / 1e6:.1f}M <= {max_params_m}M")
    elif time.time() + s.mean_duration() <= s.deadline:
        _distill_student(s, st, meta)
    else:
        s.log("distill", None, "skipped: not enough time budget left")


def _ship(s: Session, won: bool, publish: str | None, license: str) -> None:
    from ..export import export

    out = export(s.name, log=s.echo)
    scope = "" if won else "; local only, it did not win"
    s.log("export", str(out.relative_to(s.p.ws)), "torch-free ONNX bundle (encode / predict / serve)" + scope)
    if publish and not won:
        s.log("publish", None, f"skipped {publish}: it did not win, so nothing goes to the Hub")
    elif publish:
        from ..hub import publish as pub

        pub(s.name, publish, license, private=True, log=s.echo)
        s.log("publish", publish, "private Hugging Face repo with generated model card (flip public on the Hub)")


def _finish(s: Session, distill: bool, max_params_m: float, publish: str | None, license: str) -> None:
    won = standing(s.name, s.margin)["won"]
    if distill and won:
        _distill(s, max_params_m)
    elif distill:
        s.log("distill", None, "skipped: it did not win, so there is no winner to shrink")
    _ship(s, standing(s.name, s.margin)["won"], publish, license)


def run(
    name: str | None,
    prompt: str | None = None,
    modality: str | None = None,
    budget_h: float = 6.0,
    max_experiments: int = 8,
    margin: float = 0.0,
    distill: bool = True,
    max_params_m: float = 25.0,
    publish: str | None = None,
    license: str = PUBLISH_LICENSE,
    use_llm: bool = True,
    echo: Callable[[str], None] = print,
) -> dict[str, Any]:
    from ..spec import TaskSpec

    cfg = load_config()
    fresh = None if name and Project(name).exists() else _compile(name, prompt, modality, cfg)
    name = fresh.name if fresh else name
    p = Project(name)
    log = Log(p, echo)
    t0 = time.time()
    if fresh:
        _define(p, log, fresh, cfg, echo)
    s = Session(name, p, log, TaskSpec.load(p.spec_path), margin, t0 + budget_h * 3600, echo)
    if not p.experiments() and missing_labels(s.base):
        return _request_data(s, cfg, t0)
    if not p.experiments():
        _build_v1(s)
    reason = improve_until_stop(s, max_experiments, use_llm)
    log("stop", reason, "")
    _finish(s, distill, max_params_m, publish, license)
    final = standing(name, margin)
    final["hours"] = round((time.time() - t0) / 3600, 2)
    final["stop_reason"] = reason
    final["name"] = name
    log.save(final)
    _report(p, log.decisions, final)
    return final
