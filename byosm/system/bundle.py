from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

from ..config import Project
from ..export.bundle import RT_SOURCE
from ..rt import NODES_DIR, SYSTEM_MANIFEST
from .spec import SystemProject

RT_NAME = "byosm_rt.py"


def system_manifest(sp: SystemProject) -> dict:
    spec = sp.spec()
    nodes = {
        k: {"role": n.role, "specialist": n.specialist, "experiment": Project(n.specialist).promoted}
        for k, n in spec.nodes.items()
    }
    tools = {k: t.model_dump(mode="json") for k, t in spec.tools.items()}
    return {"name": spec.name, "description": spec.description, "nodes": nodes, "tools": tools}


def _gate(sp: SystemProject) -> dict | None:
    return json.loads(sp.gate_path.read_text()) if sp.gate_path.exists() else None


def set_gate(sp: SystemProject, gate: dict | None) -> None:
    sp.gate_path.write_text(json.dumps(gate))
    path = sp.export_dir / SYSTEM_MANIFEST
    if path.exists():
        path.write_text(json.dumps({**json.loads(path.read_text()), "gate": gate}, indent=2))


def export_system(sp: SystemProject, log: Callable[[str], None] = print) -> Path:
    out = sp.export_dir
    if out.exists():
        shutil.rmtree(out)
    manifest = system_manifest(sp)
    for key, node in manifest["nodes"].items():
        shutil.copytree(Project(node["specialist"]).root / "export", out / NODES_DIR / key)
        (out / NODES_DIR / key / RT_NAME).unlink(missing_ok=True)
    shutil.copy(RT_SOURCE, out / RT_NAME)
    (out / SYSTEM_MANIFEST).write_text(json.dumps({**manifest, "gate": _gate(sp)}, indent=2))
    log(f"[system] exported {sp.name} -> {out} ({', '.join(manifest['nodes'])})")
    return out
