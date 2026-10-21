from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..rt import tag_parts
from ..spec import TaskSpec
from ..text import markup_tags, match_label, normalize
from .request import REQUEST_DIR, counts, score_rows_needed
from .store import TEST_SHARE
from .table import LABEL_COL, SPLIT_CHOICES, SPLIT_COL, columns, input_column, read_table, resolve_input

BLOCKING = {"missing column", "empty input", "unknown label", "unknown argument type", "bad split", "missing file"}
BLOCKING |= {"test row also in training", "same input, different labels"}
FIRST_DATA_ROW = 2


def _row_issues(spec: TaskSpec, r: dict, base: Path) -> list[tuple[str, str]]:
    raw, out = r.get(input_column(spec), ""), []
    if not raw:
        return [("empty input", "")]
    if r.get(SPLIT_COL, "") not in ("", *SPLIT_CHOICES):
        out.append(("bad split", f"{r[SPLIT_COL]!r}: use {' or '.join(SPLIT_CHOICES)}, or leave it blank"))
    if spec.task_type == "frame":
        kinds = {tag_parts(t)[1] for t in spec.label_names} - {""}
        out += [("unknown argument type", k) for k in {tag_parts(t)[1] for t in markup_tags(raw)[1]} - kinds - {""}]
    elif r.get(LABEL_COL, "") not in spec.label_names:
        guess = match_label(r.get(LABEL_COL, ""), spec.label_names)
        out.append(("unknown label", f"{r.get(LABEL_COL, '')!r}" + (f" (did you mean {guess}?)" if guess else "")))
    if spec.modality != "text":
        f = Path(resolve_input(spec, raw, base))
        if not f.exists() or f.stat().st_size == 0:
            out.append(("missing file", str(f)))
    return out


def _key(spec: TaskSpec, raw: str, base: Path) -> str:
    if spec.modality == "text":
        return normalize(raw)
    return hashlib.sha1(Path(resolve_input(spec, raw, base)).read_bytes()).hexdigest()


def _duplicates(spec: TaskSpec, rows: list[tuple[int, dict]], base: Path) -> list[tuple[int, str, str]]:
    groups: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for i, r in rows:
        groups[_key(spec, r[input_column(spec)], base)].append((i, r))
    out = []
    for g in (g for g in groups.values() if len(g) > 1):
        nums = ", ".join(str(i) for i, _ in g)
        if len({r.get(LABEL_COL) for _, r in g}) > 1:
            out.append((g[0][0], "same input, different labels", f"rows {nums}"))
        elif len({r.get(SPLIT_COL) == "test" for _, r in g}) > 1:
            out.append((g[0][0], "test row also in training", f"rows {nums}"))
        else:
            out += [(i, "duplicate", f"same as row {g[0][0]}") for i, _ in g[1:]]
    return out


def _coverage(spec: TaskSpec, rows: list[dict], need: dict) -> dict[str, dict[str, Any]]:
    if spec.task_type == "frame":
        tags = Counter(tag_parts(t)[1] for r in rows for t in markup_tags(r[input_column(spec)])[1])
        labels = {n: tags.get(tag_parts(n)[1], 0) for n in spec.label_names if n.startswith("B-")}
    else:
        labels = Counter(r[LABEL_COL] for r in rows)
    marked = Counter(r.get(LABEL_COL, "") for r in rows if r.get(SPLIT_COL) == "test")
    out = {}
    for n in labels if spec.task_type == "frame" else spec.label_names:
        have = labels.get(n, 0)
        test = marked.get(n, 0) if marked else round(have * TEST_SHARE)
        want = need.get(n, need.get(next(iter(need), ""), {}))
        out[n] = {"rows": have, "test": test, **want}
    return out


def _needs(spec: TaskSpec, root: Path) -> dict:
    req = root / REQUEST_DIR / "request.json"
    return json.loads(req.read_text())["counts"] if req.exists() else counts(spec, [])


def _shortfalls(cov: dict, spec: TaskSpec) -> list[str]:
    out = [
        f"{n}: {c['rows']} rows, needs {c['train_min']} or more"
        for n, c in cov.items()
        if c["rows"] < c.get("train_min", 0)
    ]
    total, _ = score_rows_needed(spec.objective.target, len(spec.labels))
    test = sum(c["test"] for c in cov.values())
    return out + ([f"test: {test} rows, a trustworthy score needs about {total}"] if test < total else [])


def _noise(spec: TaskSpec, rows: list[tuple[int, dict]], base: Path, judge: bool, cfg: dict | None) -> dict | None:
    from .probe import input_text, probe
    from .probe import judge as llm_judge

    if spec.task_type == "frame":
        return None
    texts = [input_text(spec, resolve_input(spec, r[input_column(spec)], base)) for _, r in rows]
    if any(t is None for t in texts):
        return None
    found = probe(texts, [r[LABEL_COL] for _, r in rows])
    if found and judge:
        found["suspects"] = llm_judge(spec, texts, found["suspects"], cfg)
    for s in (found or {}).get("suspects", []):
        s["row"] = rows[s.pop("index")][0]
    return found


def check(
    spec: TaskSpec, root: Path, path: Path, judge: bool = True, cfg: dict | None = None, log: Callable = print
) -> dict:
    rows = list(enumerate(read_table(path), start=FIRST_DATA_ROW))
    if rows and any(c not in rows[0][1] for c in columns(spec)):
        missing = [c for c in columns(spec) if c not in rows[0][1]]
        issues = [(1, "missing column", ", ".join(missing))]
        return _finish(root, {"file": str(path), "issues": issues, "coverage": {}, "noise": None, "shortfalls": []})
    issues = [(i, k, d) for i, r in rows for k, d in _row_issues(spec, r, path.parent)]
    bad = {i for i, _, _ in issues}
    good = [(i, r) for i, r in rows if i not in bad]
    issues += _duplicates(spec, good, path.parent)
    cov = _coverage(spec, [r for _, r in good], _needs(spec, root))
    log(f"  checked {len(rows)} rows: {len(issues)} issues; probing labels...")
    report = {
        "file": str(path),
        "issues": issues,
        "coverage": cov,
        "shortfalls": _shortfalls(cov, spec),
        "noise": _noise(spec, good, path.parent, judge, cfg),
    }
    return _finish(root, report)


def _finish(root: Path, report: dict) -> dict:
    from .check_report import render

    blocking = any(k in BLOCKING for _, k, _ in report["issues"])
    report["verdict"] = "fix" if blocking else "needs more" if report["shortfalls"] else "ready"
    out = root / REQUEST_DIR
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "report.md").write_text(render(report))
    return report
