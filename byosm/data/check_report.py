from __future__ import annotations

from collections import Counter

MAX_LISTED = 30
VERDICT_LINE = {
    "ready": "**Ready to train.**",
    "needs more": "**Usable, but more rows are needed** for a model and a score you can trust.",
    "fix": "**Fix these rows first.** Training would learn from or be scored on broken rows.",
}


def _coverage(cov: dict) -> list[str]:
    lines = [
        "| label | rows | of which test | train rows (min) | train rows (recommended) | test rows |",
        "|---|---|---|---|---|---|",
    ]
    for n, c in cov.items():
        lines.append(
            f"| {n} | {c['rows']} | {c['test']} | {c.get('train_min', '-')} | {c.get('train_recommended', '-')} | {c.get('test_needed', '-')} |"
        )
    return lines


def _issues(issues: list) -> list[str]:
    by_kind = Counter(k for _, k, _ in issues)
    lines = [f"- **{k}**: {n} row(s)" for k, n in by_kind.most_common()]
    shown = [f"  - row {i}: {k}" + (f" ({d})" if d else "") for i, k, d in issues[:MAX_LISTED]]
    more = [f"  - ...and {len(issues) - MAX_LISTED} more in report.json"] if len(issues) > MAX_LISTED else []
    return lines + shown + more


def _noise(noise: dict | None) -> list[str]:
    if not noise:
        return ["Label check skipped: it needs text (or document text) and at least two rows per label."]
    lines = [f"A quick probe trained on the other rows agrees with {noise['agreement']:.0%} of the labels."]
    if noise["confusions"]:
        lines += ["", "Most mixed up (given -> probe's guess):"]
        lines += [f"- {c['given']} -> {c['probe']}: {c['rows']} rows" for c in noise["confusions"]]
    sus = noise["suspects"]
    if sus:
        lines += ["", "Rows that are probably mislabeled (the probe is confident and disagrees):"]
        for s in sus[:MAX_LISTED]:
            llm = f"; the LLM reviewer says {s['llm']}: {s['why']}" if s.get("llm") else ""
            lines.append(f"- row {s['row']}: labeled {s['given']}, looks like {s['probe']}{llm}")
    return lines


def render(report: dict) -> str:
    lines = [f"# Data check: {report['file']}", "", VERDICT_LINE[report["verdict"]], ""]
    if report["coverage"]:
        lines += ["## Coverage", "", *_coverage(report["coverage"]), ""]
    if report["shortfalls"]:
        lines += ["Short of the suggestion:", *(f"- {s}" for s in report["shortfalls"]), ""]
    if report["issues"]:
        lines += ["## Problems", "", *_issues(report["issues"]), ""]
    lines += ["## Labels", "", *_noise(report["noise"])]
    return "\n".join(lines) + "\n"
