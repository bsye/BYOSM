from __future__ import annotations

import io

from ..project import Project
from ..spec import TaskSpec
from .test_sets import test_sets
from .workbench import OURS, chosen_models

RESULTS_SHEET, SUMMARY_SHEET = "Results", "Summary"
ROW_FIELDS = ("input", "label", "source")
XLSX_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _shown(p: Project, model: str) -> str:
    return f"{p.promoted} (this model)" if model == OURS else model


def results_workbook(p: Project, spec: TaskSpec, suite: str) -> bytes:
    from openpyxl import Workbook

    found = next((t for t in test_sets(p, spec) if t["suite"] == suite), None)
    if found is None:
        raise KeyError(f"no test set {suite!r}")
    rows, got, models = found["rows"], found["results"], chosen_models(p, spec)
    wb = Workbook()
    sheet = wb.active
    sheet.title = RESULTS_SHEET
    sheet.append([*ROW_FIELDS, *(f"{_shown(p, m)} {c}" for m in models for c in ("answer", "confidence", "right"))])
    for r in rows:
        answers = [got.get(m, {}).get(r["id"]) for m in models]
        cells = [[a["label"], a["confidence"], a["label"] == r["label"]] if a else [None] * 3 for a in answers]
        sheet.append([str(r.get(f, "")) for f in ROW_FIELDS] + [c for trio in cells for c in trio])
    summary = wb.create_sheet(SUMMARY_SHEET)
    summary.append(["model", "scored", "right", "accuracy"])
    for m in models:
        scored = [r for r in rows if r["id"] in got.get(m, {})]
        right = sum(got[m][r["id"]]["label"] == r["label"] for r in scored)
        summary.append([_shown(p, m), len(scored), right, right / len(scored) if scored else None])
    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()
