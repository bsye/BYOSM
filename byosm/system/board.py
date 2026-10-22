from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path

from ..config import Project
from ..evals.identity import baseline_identity
from ..events import stage
from ..rt import System, tag_parts
from ..spec import TaskSpec
from .bundle import set_gate, system_manifest
from .frames import SPLITS, exact_bootstrap, materialize, summary, units
from .rival import PROMPT_VERSION, llm_frames
from .spec import SystemProject, SystemSpec, SystemSuite
from .streaming import stream_board

OURS = "system"
BOARD_FILE = "board.json"


def vocabulary(spec: SystemSpec) -> tuple[list[str], list[str]]:
    router, tagger = spec.role("router"), spec.role("tagger")
    actions = TaskSpec.load(Project(router[1].specialist).spec_path).label_names if router else []
    tags = TaskSpec.load(Project(tagger[1].specialist).spec_path).label_names if tagger else []
    return actions, sorted({tag_parts(t)[1] for t in tags} - {""})


def _fingerprint(sp: SystemProject, rival: str, texts: list[str]) -> str:
    who = json.dumps(system_manifest(sp)["nodes"]) if rival == OURS else f"{baseline_identity(rival)}{PROMPT_VERSION}"
    blob = who + json.dumps(vocabulary(sp.spec())) + json.dumps(texts)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def _ours(sp: SystemProject, texts: list[str]) -> tuple[list[dict], float]:
    system = System(sp.export_dir)
    t0 = time.perf_counter()
    frames = [system.run(t) for t in texts]
    ms = (time.perf_counter() - t0) * 1000 / max(1, len(texts))
    return [{"action": f["action"], "args": [[a["type"], a["text"]] for a in f["args"]]} for f in frames], ms


def _frames(sp: SystemProject, rival: str, texts: list[str], log: Callable[[str], None]) -> tuple[list[dict], float]:
    if rival == OURS:
        return _ours(sp, texts)
    if rival == "llm" or rival.startswith("llm:"):
        return llm_frames(texts, *vocabulary(sp.spec()), log, rival[4:] or "baseline"), None
    raise ValueError(f"unknown system rival {rival!r}: use llm or llm:<role>")


def _result(sp: SystemProject, suite: str, rival: str, data: dict, log: Callable, force: bool) -> dict:
    path: Path = sp.root / "results" / rival.replace(":", "_") / f"{suite}.json"
    texts = [r["input"] for s in SPLITS for r in data[s]]
    fp = _fingerprint(sp, rival, texts)
    if not force and path.exists() and json.loads(path.read_text())["fingerprint"] == fp:
        return json.loads(path.read_text())
    log(f"  scoring {rival} on {suite} ({len(texts)} requests)...")
    frames, ms = _frames(sp, rival, texts, log)
    cut = len(data[SPLITS[0]])
    res = {"fingerprint": fp, "latency_ms": ms, "frames": {SPLITS[0]: frames[:cut], SPLITS[1]: frames[cut:]}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res))
    return res


def _suite_board(sp: SystemProject, suite: SystemSuite, log: Callable, force: bool) -> dict:
    data = materialize(sp, suite)
    rivals = [OURS, *sp.spec().baselines]
    res = {r: _result(sp, suite.name, r, data, log, force) for r in rivals}
    u = {r: {s: units(res[r]["frames"][s], data[s]) for s in SPLITS} for r in rivals}
    out = {"rows": {s: len(data[s]) for s in SPLITS}, "rivals": {}, "paired": {}}
    if sp.spec().role("router"):
        texts = [r["input"] for s in SPLITS for r in data[s]]
        out["stream"] = stream_board(sp, suite.name, _fingerprint(sp, OURS, texts), data, log)
    for r in rivals:
        out["rivals"][r] = {
            **{s: summary(u[r][s]) for s in SPLITS},
            "ci": exact_bootstrap(u[r]["test"]),
            "latency_ms": res[r]["latency_ms"],
        }
        if r != OURS:
            out["paired"][r] = {s: exact_bootstrap(u[OURS][s], u[r][s]) for s in SPLITS}
    return out


@stage("Score the team")
def evaluate(sp: SystemProject, log: Callable[[str], None] = print, force: bool = False) -> dict:
    board = {s.name: _suite_board(sp, s, log, force) for s in sp.spec().evals}
    first = next(iter(board.values()), {})
    if "stream" in first:
        set_gate(sp, first["stream"]["gate"])
    (sp.root / BOARD_FILE).write_text(json.dumps(board, indent=2))
    return board
