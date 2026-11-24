from __future__ import annotations

from pathlib import Path

from .distill import REPORT as STUDENTS
from .table import AGREEMENT, ITSELF, REPORT, STREETS, TOTAL, VERSUS

BLOCK = "poker"
FIELDS = {"caller": "always-call bots", "random": "random players"}
MISSING = "-"
TABLE_HEAD = "| strategy | the other five seats | mbb/hand [95% CI] | hands |\n|---|---|---|---|"
AGREEMENT_HEAD = (
    "| strategy | Pluribus's move, all streets | preflop | flop | turn | river | mean probability on it | covered |\n"
    "|---|---|---|---|---|---|---|---|"
)
STUDENT_HEAD = (
    "| student | teacher | model | held-out KL to the teacher | same top move | rounds on its own reach |\n"
    "|---|---|---|---|---|---|"
)
FOOT = (
    "Six seats, 100 big blinds, mbb/hand = thousandths of a big blind a hand. Against copies of a strategy the result "
    "is the plain mean; against callers and random players the engine's AIVAT estimate. The Pluribus rows count its "
    "exact move in the strategy's bet menu, over the logged decisions the strategy has an answer for (covered)."
)


def against(key: str, r: dict) -> str:
    if VERSUS in key:
        return f"copies of {r['field']}"
    if r["field"] == ITSELF:
        return "copies of itself; it searches, they do not"
    return FIELDS[r["field"]] + (", it searches" if r.get("search") else "")


def played(key: str, r: dict) -> str:
    low, high = r["mbb_per_hand"] - r["ci95"], r["mbb_per_hand"] + r["ci95"]
    return (
        f"| {r['strategy']} | {against(key, r)} | {r['mbb_per_hand']:+.1f} [{low:+.1f}, {high:+.1f}] | {r['hands']:,} |"
    )


def agreeing(r: dict) -> str:
    streets = " | ".join(f"{r[s]['top_move']:.1%}" if s in r else MISSING for s in STREETS)
    total = r[TOTAL]
    return (
        f"| {r['strategy']} | {total['top_move']:.1%} | {streets} | {total['mean_prob']:.3f} | {total['covered']:.1%} |"
    )


def taught(r: dict) -> str:
    return (
        f"| {r['student']} | {r['teacher']} | {r['model_mb']} MB | {r['kl']:.4f} | {r['top_move']:.1%} | "
        f"{r.get('rounds', 0)} |"
    )


def block(ws: Path) -> str | None:
    from ...db.workspace import workspace_reports

    tables = [
        [TABLE_HEAD, *(played(k, r) for k, r in sorted(workspace_reports(ws, REPORT).items()))],
        [AGREEMENT_HEAD, *(agreeing(r) for _, r in sorted(workspace_reports(ws, AGREEMENT).items()))],
        [STUDENT_HEAD, *(taught(r) for _, r in sorted(workspace_reports(ws, STUDENTS).items()))],
    ]
    shown = ["\n".join(t) for t in tables if len(t) > 1]
    return "\n\n".join([*shown, FOOT]) if shown else None
