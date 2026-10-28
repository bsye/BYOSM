from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .evals.selection import compare, sure_lead
from .improve import apply_ops
from .project import Project

PREFIX = "ablation:"
ARM_PREFIX = "abl-"
SPLITS = ("calib", "test")
SEED_PATH = "train.seed"


def _slug(change: dict[str, Any]) -> str:
    text = "-".join(f"{k.rpartition('.')[2]}-{v}" for k, v in change.items())
    return "".join(c if c.isalnum() or c == "-" else "-" for c in text.lower())


def _ops(change: dict[str, Any]) -> list[dict]:
    return [{"op": "set", "path": k, "value": v} for k, v in change.items()]


def _arm(p: Project, base: str, exp: str, change: dict[str, Any], seed: int, why: str) -> str:
    spec = apply_ops(p.exp_spec(base), _ops({**change, SEED_PATH: seed}))
    proposal = {"parent": base, "applied": [{"id": "ablation", "why": why, "ops": _ops(change)}]}
    p.save_experiment(exp, spec=spec, proposal=proposal)
    return exp


def _pairs(p: Project, base: str, change: dict[str, Any], seeds: list[int]) -> list[tuple[int, str, str]]:
    slug, before = _slug(change), {k: _get(p.exp_spec(base), k) for k in change}
    return [
        (
            seed,
            _arm(p, base, f"{ARM_PREFIX}{slug}-off-s{seed}", before, seed, f"ablation baseline: {before}"),
            _arm(p, base, f"{ARM_PREFIX}{slug}-on-s{seed}", change, seed, f"ablation change: {change}"),
        )
        for seed in seeds
    ]


def _get(spec: Any, path: str) -> Any:
    out: Any = spec.model_dump(mode="json")
    for part in path.split("."):
        out = out[part]
    return out


def _verdict(rows: list[dict]) -> str:
    if all(r["calib"] and sure_lead(r["calib"]) and r["calib"]["delta"] > 0 and r["test_delta"] > 0 for r in rows):
        return "better"
    if all(r["calib"] and sure_lead({"p_better": 1 - r["calib"]["p_better"]}) for r in rows):
        return "worse"
    return "not sure"


def ablate(name: str, base: str, change: dict[str, Any], seeds: list[int], log: Callable[[str], None] = print) -> dict:
    from .pipeline import build

    p = Project(name)
    spec, prim = p.spec(), p.spec().primary_suite()
    rows = []
    for seed, off, on in _pairs(p, base, change, seeds):
        for exp in (off, on):
            if exp not in p.experiments():
                build(name, exp, log, promote=False)
        board = p.board()
        a, b = board[on][prim], board[off][prim]
        cmp = {s: compare(spec, a, b, s) for s in SPLITS}
        row = {"seed": seed, "off": off, "on": on, "calib": cmp["calib"], "test": cmp["test"]}
        rows.append({**row, "test_delta": a["objective"] - b["objective"]})
        log(f"  seed {seed}: calib {_line(cmp['calib'])}, test {_line(cmp['test'])}")
    report = {"base": base, "change": change, "suite": prim, "seeds": rows, "verdict": _verdict(rows)}
    p.save_report(PREFIX + _slug(change), report)
    log(f"verdict for {change}: {report['verdict']}")
    return report


def _line(cmp: dict | None) -> str:
    return f"{cmp['delta']:+.4f} (P(better) {cmp['p_better']:.2f})" if cmp else "no paired units"


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def summaries(p: Project) -> list[dict]:
    return [
        {
            "change": r["change"],
            "base": r["base"],
            "verdict": r["verdict"],
            "seeds": [
                {
                    "seed": s["seed"],
                    "calib": (s["calib"] or {}).get("delta"),
                    "p_better": (s["calib"] or {}).get("p_better"),
                    "test": s["test_delta"],
                }
                for s in r["seeds"]
            ],
            "calib": _mean([s["calib"]["delta"] for s in r["seeds"] if s["calib"]]),
            "test": _mean([s["test_delta"] for s in r["seeds"]]),
        }
        for r in p.reports(PREFIX).values()
    ]


def is_arm(exp: str) -> bool:
    return exp.startswith(ARM_PREFIX)
