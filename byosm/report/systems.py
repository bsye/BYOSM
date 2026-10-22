from __future__ import annotations

from ..config import find_workspace
from ..system.board import BOARD, OURS
from ..system.spec import SystemProject, system_names
from .cells import num, score

SYSTEM = "system"
HEAD = (
    "| team | exact frame [95% CI] | calib exact frame | action accuracy | argument F1 | CPU per request |\n"
    "|---|---|---|---|---|---|"
)


def reported_systems() -> list[str]:
    ws = find_workspace()
    return [n for n in system_names(ws) if SystemProject(n, ws).report(BOARD)]


def _row(name: str, v: dict) -> str:
    ms = f"{v['latency_ms']:.1f}ms" if v.get("latency_ms") else "-"
    t = v["test"]
    return (
        f"| {name} | {score(t['exact_frame'], v['ci'])} | {num(v['calib']['exact_frame'])} | "
        f"{num(t['action_accuracy'])} | {num(t['arg_f1'])} | {ms} |"
    )


def _suite(name: str, suite: str, b: dict) -> list[str]:
    rows = [_row(name if r == OURS else r, v) for r, v in b["rivals"].items()]
    paired = [
        f"{name} minus {r}: test {pv['test']['delta']:+.3f} (P(better) = {pv['test']['p_better']:.2f}), "
        f"calib {pv['calib']['delta']:+.3f}"
        for r, pv in b["paired"].items()
    ]
    foot = (
        f"Suite `{suite}`: {b['rows']['test']} test and {b['rows']['calib']} calib requests. Exact frame = the action and "
        "every argument right. CPU is the whole team per request (every node), from the exported bundle. "
        f"Paired bootstrap on exact frame: {'; '.join(paired)}."
    )
    return [HEAD, *rows, "", foot, *(["", stream_line(b["stream"])] if b.get("stream") else [])]


def stream_line(st: dict) -> str:
    t, end = st["test"], st["test_at_end"]
    if not st["gate"]:
        return (
            "While typing: no gate kept wrong actions within the allowance on calib, so the action waits for the end."
        )
    g = st["gate"]
    return (
        f"While typing (gate tuned on calib: confidence {g['min_confidence']}, {g['stable_for']} agreeing keystroke{'s' if g['stable_for'] > 1 else ''}): "
        f"the action fires after {t['typed_share']:.0%} of the request on average, {t['early_share']:.0%} of requests "
        f"before the last keystroke, with {t['wrong_per_100']:.1f} wrong actions per 100 against "
        f"{end['wrong_per_100']:.1f} when waiting for the end; {st['ms_per_keystroke']:.1f}ms per keystroke. "
        "The LLM needs the whole request."
    )


def system_block(name: str) -> str:
    board = SystemProject(name).report(BOARD)
    return "\n\n".join("\n".join(_suite(name, s, b)) for s, b in board.items())
