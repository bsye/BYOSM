from __future__ import annotations

import difflib
import time
from pathlib import Path
from typing import Any

from ..db.workspace import save_workspace_report, workspace_report

PROVEN = "proven-methods"
PROVEN_AFTER = 2
CANDIDATE, PROVEN_STATUS = "candidate", "proven"
BUILTIN_PREFIX = "builtin:"
METHODS_DIR = Path(__file__).with_name("methods")
PATCH_DIR = ".byosm/lab/patches"
REPO = Path(__file__).parents[2]


def registry(ws: Path | None) -> dict[str, dict[str, Any]]:
    return workspace_report(ws, PROVEN) or {}


def slug(name: str) -> str:
    kept = "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")
    return "_".join(part for part in kept.split("_") if part) or "method"


def record_win(ws: Path | None, specialist: str, modality: str, draft: dict[str, Any]) -> dict[str, Any]:
    found = registry(ws)
    entry = found.get(draft["id"]) or {
        **{k: draft.get(k) for k in ("id", "name", "hypothesis", "code", "paper")},
        "modality": modality,
        "won_on": [],
        "status": CANDIDATE,
    }
    entry["won_on"] = sorted({*entry["won_on"], specialist})
    if len(entry["won_on"]) >= PROVEN_AFTER and entry["status"] != PROVEN_STATUS:
        entry |= {"status": PROVEN_STATUS, "proven": time.time(), "patch": str(write_patch(ws, entry))}
    found[draft["id"]] = entry
    save_workspace_report(ws, PROVEN, found)
    return entry


def candidates(ws: Path | None, specialist: str, modality: str) -> list[dict[str, Any]]:
    return [e for e in registry(ws).values() if e["modality"] == modality and specialist not in e["won_on"]]


def _diff(rel: str, old: str | None, new: str) -> str:
    before = "/dev/null" if old is None else f"a/{rel}"
    return "".join(difflib.unified_diff((old or "").splitlines(True), new.splitlines(True), before, f"b/{rel}"))


def _family(entry: dict[str, Any], name: str) -> str:
    evidence = f"[{entry['paper']}]" if entry.get("paper") else "[]"
    won = ", ".join(entry["won_on"])
    return f"""
- id: lab-{name.replace("_", "-")}
  title: "{entry["name"]}"
  effect: "{entry["hypothesis"]} Won a sure lead in the lab on {won}."
  paths: [train.lab_method={BUILTIN_PREFIX}{name}]
  modalities: [{entry["modality"]}]
  cost: 1.0
  prior: 0.5
  evidence: {evidence}
  status: available
  targets: []
  added: {time.strftime("%Y-%m-%d")}
"""


def _test(name: str) -> str:
    return f'''from byosm.lab.plugin import builtin, builtin_code
from byosm.lab.static import problems


def test_the_{name}_method_passes_its_checks_and_loads():
    assert not problems(builtin_code("{name}")) and builtin("{name}")
'''


def write_patch(ws: Path | None, entry: dict[str, Any]) -> Path:
    name = slug(entry["name"])
    catalogue = REPO / "byosm/improve/techniques.yaml"
    old = catalogue.read_text()
    parts = [
        _diff(f"byosm/lab/methods/{name}.py", None, entry["code"].rstrip() + "\n"),
        _diff("byosm/improve/techniques.yaml", old, old.rstrip() + "\n" + _family(entry, name)),
        _diff(f"tests/test_lab_method_{name}.py", None, _test(name)),
    ]
    path = Path(ws or ".") / PATCH_DIR / f"{name}.patch"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts))
    return path


LAB_SOURCE = "lab"
LAB_PATH = "train.lab_method"


def note_win(p: Any, exp: str) -> dict[str, Any] | None:
    from .drafts import draft

    spec = p.exp_spec(exp)
    did = spec.train.lab_method
    if not did or did.startswith(BUILTIN_PREFIX):
        return None
    found = draft(p, did) or registry(p.ws).get(did)
    return record_win(p.ws, p.name, spec.modality, found) if found else None


def proposals(p: Any, spec: Any) -> list[dict[str, Any]]:
    out = []
    for e in candidates(p.ws, p.name, spec.modality):
        why = f"Lab method {e['name']} won a sure lead on {', '.join(e['won_on'])}: {e['hypothesis']}"
        op = {"op": "set", "path": LAB_PATH, "value": e["id"]}
        out.append({"id": f"lab-{slug(e['name'])}", "why": why, "ops": [op], "source": LAB_SOURCE})
    return out
