from __future__ import annotations

import json
import shutil
import socket
import subprocess
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

LABEL = "byosm.specialist"
MODEL_PORT = 8800
DOCKERFILE = Path(__file__).resolve().parents[1] / "docker" / "model.Dockerfile"
CPUS = "1"
MEMORY = "1g"
PREFIX = "byosm-"
TIMEOUT_S = 30
READY_S = 60


def _docker() -> str:
    found = shutil.which("docker")
    if not found:
        raise RuntimeError("needs the docker CLI on PATH (Docker Desktop, OrbStack or Colima)")
    return found


def _run(*args: str, timeout: int = TIMEOUT_S) -> subprocess.CompletedProcess:
    return subprocess.run([_docker(), *args], capture_output=True, text=True, timeout=timeout)


def image_of(name: str) -> str:
    return f"{PREFIX}{name.lower()}"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def build(name: str, bundle: Path, log: Callable[[str], None] = print, tag: str | None = None) -> str:
    if not DOCKERFILE.exists():
        raise RuntimeError(f"needs {DOCKERFILE} (a source checkout of byosm)")
    image = tag or image_of(name)
    log(f"  building {image} from {bundle}")
    built = subprocess.run([_docker(), "build", "-f", str(DOCKERFILE), "-t", image, str(bundle)])
    if built.returncode:
        raise RuntimeError(f"docker build failed with exit {built.returncode}")
    return image


def has_image(name: str) -> bool:
    return _run("image", "inspect", image_of(name)).returncode == 0


def deploy(name: str, log: Callable[[str], None] = print) -> dict[str, Any]:
    stop(name)
    port = _free_port()
    flags = ["--rm", "--read-only", "--cpus", CPUS, "--memory", MEMORY, "--label", f"{LABEL}={name}"]
    started = _run(
        "run", "-d", "--name", image_of(name), "-p", f"127.0.0.1:{port}:{MODEL_PORT}", *flags, image_of(name)
    )
    if started.returncode:
        raise RuntimeError(f"docker run failed: {started.stderr.strip()[-300:]}")
    deadline = time.time() + READY_S
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2):
                log(
                    f"  {name} serves on http://127.0.0.1:{port}/v1/{name}/predict ({CPUS} CPU, {MEMORY} memory, read-only)"
                )
                return {"name": name, "port": port}
        except OSError:
            time.sleep(1)
    raise RuntimeError(f"the container started but did not answer on port {port} within {READY_S} s")


def stop(name: str) -> None:
    _run("rm", "-f", image_of(name))


def _stats() -> dict[str, dict[str, str]]:
    out = _run("stats", "--no-stream", "--format", "{{json .}}")
    rows = [json.loads(line) for line in out.stdout.splitlines() if line.strip()]
    return {r["Name"]: {"cpu": r.get("CPUPerc", ""), "memory": r.get("MemUsage", "")} for r in rows}


def _host_port(ports: str) -> int | None:
    host, _, _ = ports.partition("->")
    port = host.rpartition(":")[2]
    return int(port) if port.isdecimal() else None


def deployments(with_stats: bool = True) -> list[dict[str, Any]]:
    if not shutil.which("docker"):
        return []
    out = _run("ps", "--filter", f"label={LABEL}", "--format", "{{json .}}")
    rows = [json.loads(line) for line in out.stdout.splitlines() if line.strip()]
    stats = _stats() if rows and with_stats else {}
    return [
        {
            "name": r["Labels"].partition(f"{LABEL}=")[2].partition(",")[0],
            "container": r["Names"],
            "image": r["Image"],
            "port": _host_port(r.get("Ports", "")),
            "status": r.get("Status", ""),
            **stats.get(r["Names"], {}),
        }
        for r in rows
    ]


def predict(name: str, text: str) -> dict[str, Any]:
    found = next((d for d in deployments(with_stats=False) if d["name"] == name), None)
    if not found or not found["port"]:
        raise KeyError(f"{name} is not running in a container")
    body = json.dumps({"input": text}).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{found['port']}/v1/{name}/predict", data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    t0 = time.time()
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as r:
        answer = json.loads(r.read())
    return {"answer": answer, "ms": round((time.time() - t0) * 1000, 1)}
