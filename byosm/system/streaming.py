from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from ..rt import Specialist, System, TextGate
from .frames import SPLITS
from .spec import SystemProject

CONFIDENCE_GRID = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99)
STABLE_GRID = (1, 2, 3, 5, 8, 13)


def _trace(router: Specialist, text: str) -> list[list]:
    out = []
    for i in range(1, len(text) + 1):
        if text[:i].strip():
            p = router.predict(text[:i])
            out.append([i, p["label"], p["probs"][p["label"]]])
    return out


def traces(sp: SystemProject, suite: str, fp: str, data: dict, log: Callable[[str], None]) -> dict:
    kind = f"trace:{suite}:{fp}"
    if cached := sp.report(kind):
        return cached
    system = System(sp.export_dir)
    router = system.nodes[system._role("router")]
    n = sum(len(r["input"]) for s in SPLITS for r in data[s])
    log(f"  replaying {n} keystrokes through the router...")
    t0 = time.perf_counter()
    out = {s: [_trace(router, r["input"]) for r in data[s]] for s in SPLITS}
    res = {"traces": out, "ms_per_keystroke": (time.perf_counter() - t0) * 1000 / max(1, n)}
    sp.save_report(kind, res)
    return res


def replay(trace: list[list], length: int, gate: dict | None) -> tuple[str, float, bool]:
    g = TextGate(**gate) if gate else None
    for i, label, conf in trace:
        if g is not None and g.push(label, conf):
            return label, i / length, i < length
    return trace[-1][1], 1.0, False


def stream_units(trs: list[list], gold: list[dict], gate: dict | None) -> dict[str, list]:
    out: dict[str, list] = {"typed": [], "early": [], "wrong": []}
    for tr, g in zip(trs, gold, strict=True):
        action, typed, early = replay(tr, len(g["input"]), gate)
        out["typed"].append(typed)
        out["early"].append(int(early))
        out["wrong"].append(int(action != g["action"]))
    return out


def stream_summary(u: dict[str, list]) -> dict[str, float]:
    return {
        "typed_share": float(np.mean(u["typed"])),
        "early_share": float(np.mean(u["early"])),
        "wrong_per_100": float(np.mean(u["wrong"]) * 100),
    }


def tune(trs: list[list], gold: list[dict], max_extra_wrong: float) -> dict | None:
    allowed = stream_summary(stream_units(trs, gold, None))["wrong_per_100"] + max_extra_wrong
    best: tuple[float, dict] | None = None
    for c in CONFIDENCE_GRID:
        for k in STABLE_GRID:
            gate = {"min_confidence": c, "stable_for": k}
            s = stream_summary(stream_units(trs, gold, gate))
            if s["wrong_per_100"] <= allowed and (best is None or s["typed_share"] < best[0]):
                best = (s["typed_share"], gate)
    return best[1] if best else None


def stream_board(sp: SystemProject, suite: str, fp: str, data: dict, log: Callable[[str], None]) -> dict:
    tr = traces(sp, suite, fp, data, log)
    gate = tune(tr["traces"]["calib"], data["calib"], sp.spec().stream.max_extra_wrong)
    out = {"gate": gate, "ms_per_keystroke": tr["ms_per_keystroke"]}
    for s in SPLITS:
        out[s] = stream_summary(stream_units(tr["traces"][s], data[s], gate))
        out[f"{s}_at_end"] = stream_summary(stream_units(tr["traces"][s], data[s], None))
    return out
