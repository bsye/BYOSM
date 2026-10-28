from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..spec import TaskSpec

SPAN, SPAN_S, AT_S = "span", "span_s", "at_s"
THRESHOLDS = np.round(np.arange(0.0, 1.01, 0.01), 2)
ACTION_DELAYS_S = np.round(np.arange(0.2, 1.01, 0.1), 2)
TIMEOUTS_S = np.round(np.arange(1.0, 3.51, 0.5), 2)
LATENCY_BUDGETS_MS = (300, 600)
EPS = 1e-9
POLICY = ("threshold", "action_delay", "timeout", "cutoff", "latency")


@dataclass
class Spans:
    eot: np.ndarray
    duration: np.ndarray
    fire: np.ndarray


def spans(spec: TaskSpec, rows: list[dict], scores: np.ndarray) -> Spans:
    found: dict[str, list[tuple[dict, float]]] = {}
    for r, p in zip(rows, scores, strict=True):
        found.setdefault(r["meta"][SPAN], []).append((r, float(p)))
    positive = spec.label_names[spec.pos_index]
    fire = np.full((len(THRESHOLDS), len(found)), np.inf)
    for j, part in enumerate(found.values()):
        for r, p in part:
            fire[:, j] = np.where(p > THRESHOLDS, np.minimum(fire[:, j], r["meta"][AT_S]), fire[:, j])
    eot = np.array([part[0][0]["label"] == positive for part in found.values()], bool)
    duration = np.array([part[0][0]["meta"][SPAN_S] for part in found.values()], float)
    return Spans(eot, duration, fire)


def sweep(s: Spans) -> np.ndarray:
    hold, eot = s.duration[~s.eot], s.fire[:, s.eot]
    hold_fire = s.fire[:, ~s.eot]
    out = []
    for delay in ACTION_DELAYS_S:
        cut_by_model = np.isfinite(hold_fire) & (hold > np.maximum(delay, hold_fire) + EPS)
        eot_fire = np.maximum(delay, eot)
        for timeout in TIMEOUTS_S[delay <= TIMEOUTS_S + EPS]:
            cutoff = (cut_by_model | (hold > timeout + EPS)).mean(1)
            detected = np.isfinite(eot) & (eot_fire <= timeout + EPS)
            latency = np.where(detected, eot_fire, timeout).mean(1)
            fixed = np.full(len(THRESHOLDS), delay), np.full(len(THRESHOLDS), timeout)
            out.append(np.column_stack([THRESHOLDS, *fixed, cutoff, latency]))
    return np.concatenate(out)


def best(table: np.ndarray, budget_ms: int) -> int | None:
    feasible = np.flatnonzero(table[:, 4] <= budget_ms / 1000 + EPS)
    if not len(feasible):
        return None
    t = table[feasible]
    return int(feasible[np.lexsort((t[:, 2], t[:, 1], -t[:, 0], t[:, 4], t[:, 3]))[0]])


def point(table: np.ndarray, i: int | None) -> dict[str, float] | None:
    return None if i is None else dict(zip(POLICY, (float(v) for v in table[i]), strict=True))


def joined(a: Spans, b: Spans) -> Spans:
    return Spans(np.concatenate([a.eot, b.eot]), np.concatenate([a.duration, b.duration]), np.hstack([a.fire, b.fire]))


def usable(s: Spans) -> bool:
    return bool(s.eot.any() and (~s.eot).any())


def scores(spec: TaskSpec, calib: list[dict], test: list[dict], pc: Any, pt: np.ndarray) -> dict[str, Any]:
    pos = spec.pos_index
    tested = spans(spec, test, pt[:, pos])
    chosen = spans(spec, calib, pc[:, pos]) if calib else None
    everything = joined(chosen, tested) if chosen else tested
    if not usable(everything):
        return {}
    table = sweep(everything)
    out: dict[str, Any] = {"all": {str(b): point(table, best(table, b)) for b in LATENCY_BUDGETS_MS}}
    out |= {"hold_spans": int((~everything.eot).sum()), "eot_spans": int(everything.eot.sum())}
    if chosen and usable(chosen) and usable(tested):
        on_test, on_calib = sweep(tested), sweep(chosen)
        out["held_out"] = {str(b): point(on_test, best(on_calib, b)) for b in LATENCY_BUDGETS_MS}
    return out
