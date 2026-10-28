from __future__ import annotations

import json
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..deploy import docker_cli
from .containers import close_network, oom_killed, open_network, refusals, stop
from .image import ensure_image

OOM_LINE = "stopped: the job used all of its {} GB of container memory"
REFUSED_LINE = "  network: {}; add the host to jobs.network in byosm.yaml if the job needs it"


def _terminated(*_: object) -> None:
    raise SystemExit(-signal.SIGTERM)


def _code(plan: dict[str, Any], status: int) -> int:
    inner = Path(plan["inner"])
    if inner.exists():
        return int(inner.read_text())
    if oom_killed(plan["container"]):
        print(OOM_LINE.format(plan["memory_gb"]), flush=True)
    return status


def _clean(plan: dict[str, Any]) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    stop(plan["container"])
    for line in refusals(plan["job"]):
        print(REFUSED_LINE.format(line), flush=True)
    close_network(plan["job"])


def main() -> None:
    plan = json.loads(sys.argv[1])
    signal.signal(signal.SIGTERM, _terminated)
    try:
        ensure_image(plan["image"], plan["context"], plan["torch"])
        open_network(plan["job"], plan["image"], plan["network"])
        code = _code(plan, subprocess.Popen([docker_cli(), "run", *plan["run"]]).wait())
    finally:
        _clean(plan)
    sys.exit(code)
