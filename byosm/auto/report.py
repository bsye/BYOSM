from __future__ import annotations

from ..config import Project
from ..format import fmt, interval, params


def _selection_line(final: dict) -> str:
    bb = final["best_baseline"]
    vs = f" vs best baseline {bb[0]} {bb[1]:.4f}" if bb else ""
    return f"- selection (`{final['suite']}` calib split): {final['ours']:.4f} {final['metric']}{vs}"


def _test_line(final: dict) -> str:
    t = final.get("test") or {}
    if t.get("ours") is None:
        return "- held-out test: n/a"
    ci, vs, bb = t.get("ci95"), t.get("vs_best_baseline"), final["best_baseline"]
    line = f"- held-out test: {t['ours']:.4f}" + (f" (95% CI {interval(ci, 4)})" if ci else "")
    if vs and bb:
        line += f" vs {bb[0]} {t['baselines'][bb[0]]:.4f}: delta {vs['delta']:+.4f}, P(better) {vs['p_better']:.2f}"
    return line


def _board_lines(name: str) -> list[str]:
    from .. import evals

    suites, rows = evals.board_rows(name)
    lines = ["| | params | " + " | ".join(suites) + " |", "|---|---|" + "---|" * len(suites)]
    for r in rows:
        cells = " | ".join(fmt(r.get(s)) for s in suites)
        lines.append(f"| {r['id']}{' *' if r['promoted'] else ''} | {params(r['params'])} | {cells} |")
    return lines


def _report(p: Project, decisions: list[dict], final: dict) -> None:
    verdict = "WON" if final["won"] else "did not win"
    lines = [
        f"# {p.name}: autopilot report",
        "",
        f"**Result:** {final['exp']}: **{verdict}** ({final['stop_reason']}, {final['hours']} h).",
        "",
        _selection_line(final),
        _test_line(final),
        "",
        "## Board",
        "",
        *_board_lines(p.name),
        "",
        "## Decisions",
        "",
    ]
    lines += [f"- `{d['t']}` **{d['step']}** → {d['choice']}: {d['why']}" for d in decisions]
    (p.root / "auto.md").write_text("\n".join(lines) + "\n")
